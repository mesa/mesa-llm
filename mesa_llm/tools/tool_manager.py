import asyncio
import concurrent.futures
import contextlib
import copy
import inspect
import json
import logging
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, get_type_hints

from terminal_style import style

from mesa_llm.tools.tool_decorator import _GLOBAL_TOOL_REGISTRY

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

logger = logging.getLogger(__name__)

_UNSET = object()
ToolRef = Callable | str
ToolSelection = ToolRef | list[ToolRef] | tuple[ToolRef, ...] | None


class ToolManager:
    """
    Manager for registering, organizing, and executing explicit LLM-callable
    tools. Bare managers expose no tools; pass ``tools=`` to configure the
    exact capabilities a manager should expose.

    Attributes:
        - tools: A dictionary of tools of the form {tool_name: tool_function}. E.g. {"get_current_weather": get_current_weather}.
        - **instances** (class-level list) - ToolManager instances.

    Methods:
        - **register(fn)** - Register tool function to this manager
        - **add_tool_to_all(fn)** - Add tool to all ToolManager instances
        - **get_tools_schema(tools=<inherit>)** → *list[dict]* - Get OpenAI-compatible schemas
        - **call_tools(agent, llm_response)** → *list[dict]* - Execute LLM-recommended tools
        - **has_tool(name)** → *bool* - Check if tool is registered

    Tool Execution Flow:
        1. **Tool Registration**: Functions decorated with `@tool` are registered for explicit lookup, or added directly with `@tool(tool_manager=...)`
        2. **Schema Generation**: Tool decorators analyze function signatures and docstrings to create function calling schemas
        3. **LLM Integration**: Reasoning strategies receive tool schemas and can request specific tool calls
        4. **Argument Validation**: ToolManager validates LLM-provided arguments against function signatures with automatic type coercion
        5. **Execution**: Tools are called with validated arguments, including automatic agent parameter injection
        6. **Result Handling**: Tool outputs are captured and added to agent memory for future reasoning
    """

    instances: ClassVar[list["ToolManager"]] = []

    def __init__(
        self,
        tools: list[ToolRef] | tuple[ToolRef, ...] | None = None,
        extra_tools: dict[str, Callable] | None = None,
    ):
        ToolManager.instances.append(self)
        self.tools: dict[str, Callable] = {}

        if tools is not None:
            self.register_many(tools)

        if extra_tools:
            warnings.warn(
                "`extra_tools` is deprecated; pass explicit `tools=` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.tools.update(extra_tools)

    def register(self, fn: Callable):
        """Register a tool function by name"""
        name = fn.__name__
        self.tools[name] = fn  # storing the name & function pair as a dictionary

    def register_many(self, tools: list[ToolRef] | tuple[ToolRef, ...]):
        """Register explicit tool callables or registered tool names."""
        for tool_ref in tools:
            self.register(self._resolve_registration_tool_ref(tool_ref))

    @classmethod
    def add_tool_to_all(cls, fn: Callable):
        """Add a tool to all instances"""
        for instance in cls.instances:
            instance.register(fn)

    def _get_tool_schema(self, tool: ToolRef, schema_name: str | None = None) -> dict:
        fn = self._resolve_configured_tool_ref(tool) if isinstance(tool, str) else tool
        schema_name = schema_name or getattr(fn, "__name__", repr(fn))
        schema = getattr(fn, "__tool_schema__", None)
        if schema is None:
            return {"error": f"Tool {schema_name} missing __tool_schema__"}

        if schema.get("function", {}).get("name") == schema_name:
            return schema

        aliased_schema = copy.deepcopy(schema)
        aliased_schema.setdefault("function", {})["name"] = schema_name
        return aliased_schema

    def get_tool_schema(self, fn: Callable, schema_name: str | None = None) -> dict:
        """Deprecated compatibility alias for the private single-tool helper."""
        warnings.warn(
            "`get_tool_schema` is deprecated; use `get_tools_schema` for "
            "configured or narrowed tool selections.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._get_tool_schema(fn, schema_name)

    def _unknown_tool_error(self, tool_name: str) -> ValueError:
        return ValueError(
            style(
                "Unknown tool name(s): "
                f"{[tool_name]}. Available tools: {sorted(self.tools)}",
                color="red",
            )
        )

    def _invalid_tool_ref_error(self, tool_ref: Any) -> TypeError:
        return TypeError(
            style(
                "Tools must be callables or registered tool names, "
                f"got {type(tool_ref).__name__}.",
                color="red",
            )
        )

    def _resolve_registration_tool_ref(self, tool_ref: ToolRef) -> Callable:
        """Resolve constructor tool references.

        Constructors may opt into globally registered bare ``@tool`` names.
        Per-call selectors intentionally use the stricter configured-only
        resolver below.
        """
        if callable(tool_ref):
            return tool_ref

        if isinstance(tool_ref, str):
            if tool_ref in self.tools:
                return self.tools[tool_ref]
            if tool_ref in _GLOBAL_TOOL_REGISTRY:
                return _GLOBAL_TOOL_REGISTRY[tool_ref]
            raise ValueError(
                style(
                    "Unknown tool name(s): "
                    f"{[tool_ref]}. Available tools: "
                    f"{sorted(set(self.tools) | set(_GLOBAL_TOOL_REGISTRY))}",
                    color="red",
                )
            )

        raise self._invalid_tool_ref_error(tool_ref)

    def _resolve_configured_tool_ref(self, tool_ref: ToolRef) -> Callable:
        """Resolve per-call tool selectors against configured tools only."""
        return self._resolve_configured_tool_item(tool_ref)[1]

    def _resolve_configured_tool_item(self, tool_ref: ToolRef) -> tuple[str, Callable]:
        """Resolve a per-call selector while preserving configured tool names."""
        if isinstance(tool_ref, str):
            if tool_ref in self.tools:
                return tool_ref, self.tools[tool_ref]
            raise self._unknown_tool_error(tool_ref)

        if callable(tool_ref):
            tool_name = getattr(tool_ref, "__name__", repr(tool_ref))
            if tool_name in self.tools and self.tools[tool_name] is tool_ref:
                return tool_name, self.tools[tool_name]
            for configured_name, configured_fn in self.tools.items():
                if configured_fn is tool_ref:
                    return configured_name, configured_fn
            raise self._unknown_tool_error(tool_name)

        raise self._invalid_tool_ref_error(tool_ref)

    def _normalize_tool_selection(self, tools: ToolSelection) -> list[ToolRef]:
        """Normalize one or many explicit tool selectors to a list."""
        if tools is None:
            return []
        if isinstance(tools, list | tuple):
            return list(tools)
        if isinstance(tools, str) or callable(tools):
            return [tools]
        raise self._invalid_tool_ref_error(tools)

    def _resolve_tool_refs(self, tools: ToolSelection) -> list[Callable]:
        return [
            self._resolve_configured_tool_ref(tool_ref)
            for tool_ref in self._normalize_tool_selection(tools)
        ]

    def _resolve_tool_items(self, tools: ToolSelection) -> list[tuple[str, Callable]]:
        return [
            self._resolve_configured_tool_item(tool_ref)
            for tool_ref in self._normalize_tool_selection(tools)
        ]

    def _get_tool_execution_map(
        self,
        tools: ToolSelection | object = _UNSET,
    ) -> dict[str, Callable]:
        if tools is _UNSET:
            return self.tools
        if tools is None:
            return {}
        return dict(self._resolve_tool_items(tools))

    def get_tools_schema(
        self,
        tools: ToolSelection | object = _UNSET,
        selected_tools: ToolSelection | object = _UNSET,
    ) -> list[dict]:
        """Return schemas for configured tools or an explicit per-call override.

        Omitting ``tools`` returns the manager's configured tools.
        Passing ``tools=None`` or ``tools=[]`` returns no tools.
        Passing an explicit callable, name, or sequence returns exactly those
        configured tools.
        """
        if selected_tools is not _UNSET:
            warnings.warn(
                "`selected_tools` is deprecated; use `tools=` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if tools is not _UNSET:
                raise ValueError("Use either `tools` or `selected_tools`, not both.")
            tools = _UNSET if selected_tools is None else selected_tools

        if tools is _UNSET:
            selected_items = list(self.tools.items())
        elif tools is None:
            selected_items = []
        else:
            selected_items = self._resolve_tool_items(tools)

        return [
            self._get_tool_schema(fn, schema_name=name) for name, fn in selected_items
        ]

    def get_all_tools_schema(
        self,
        tools: ToolSelection | object = _UNSET,
        selected_tools: ToolSelection | object = _UNSET,
    ) -> list[dict]:
        """Deprecated compatibility alias for ``get_tools_schema``."""
        warnings.warn(
            "`get_all_tools_schema` is deprecated; use `get_tools_schema` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if selected_tools is not _UNSET:
            if tools is not _UNSET:
                raise ValueError("Use either `tools` or `selected_tools`, not both.")
            tools = _UNSET if selected_tools is None else selected_tools
        elif tools is None:
            tools = _UNSET
        return self.get_tools_schema(tools=tools)

    def call(self, name: str, arguments: dict) -> str:
        """Call a registered tool with validated args"""
        if name not in self.tools:
            raise ValueError(style(f"Tool '{name}' not found", color="red"))
        return self.tools[name](**arguments)

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    async def _process_tool_call(
        self,
        agent: "LLMAgent",
        tool_call: Any,
        index: int,
        available_tools: dict[str, Callable],
    ) -> dict:
        """
        Internal helper to process a single tool call consistently.
        Supports both synchronous and asynchronous tool functions.
        """

        # Safe extraction
        function_obj = getattr(tool_call, "function", None)
        function_name = getattr(function_obj, "name", "unknown")
        tool_call_id = getattr(tool_call, "id", "unknown")
        raw_args = getattr(function_obj, "arguments", "{}")

        try:
            # Validate tool existence
            if function_name not in available_tools:
                raise ValueError(
                    style(
                        f"Function '{function_name}' not found in ToolManager",
                        color="red",
                    )
                )

            # Parse JSON arguments safely
            try:
                function_args = json.loads(raw_args or "{}")
            except json.JSONDecodeError as e:
                raise ValueError(
                    style(f"Invalid JSON in function arguments: {e}", color="red")
                ) from e

            function_to_call = available_tools[function_name]

            # Inspect signature BEFORE calling
            sig = inspect.signature(function_to_call)
            expects_agent = "agent" in sig.parameters

            # Filter arguments to only those accepted by the function, with type coercion based on annotations
            try:
                hints = get_type_hints(function_to_call)
            except (NameError, AttributeError, TypeError):
                hints = getattr(function_to_call, "__annotations__", {})

            coerce: dict[type, type] = {float: float, int: int}
            filtered_args = {}
            for k, v in function_args.items():
                if k not in sig.parameters:
                    continue
                expected = hints.get(k)
                coerce_fn = coerce.get(expected)
                new_value = v
                if coerce_fn is not None and not isinstance(v, expected):
                    with contextlib.suppress(ValueError, TypeError):
                        new_value = coerce_fn(v)
                filtered_args[k] = new_value

            if expects_agent:
                filtered_args["agent"] = agent

            # Execute (sync or async)
            if inspect.iscoroutinefunction(function_to_call):
                function_response = await function_to_call(**filtered_args)
            else:
                function_response = await asyncio.to_thread(
                    function_to_call, **filtered_args
                )

            # Only treat None as empty
            if function_response is None:
                function_response = f"{function_name} executed successfully"

            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "response": str(function_response),
            }

        except Exception as e:
            logger.exception(
                "Error executing tool call %s (%s): %s",
                index + 1,
                function_name,
                e,
            )
            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "response": f"Error: {e!s}",
            }

    def call_tools(
        self,
        agent: "LLMAgent",
        llm_response: Any,
        tools: ToolSelection | object = _UNSET,
    ) -> list[dict]:
        """
        Synchronous tool execution with safe async bridge.
        """

        tool_calls = getattr(llm_response, "tool_calls", [])
        if not tool_calls:
            return []

        available_tools = self._get_tool_execution_map(tools)

        async def _run_all():
            tasks = [
                self._process_tool_call(agent, tc, i, available_tools)
                for i, tc in enumerate(tool_calls)
            ]
            return await asyncio.gather(*tasks)

        try:
            return asyncio.run(_run_all())
        except RuntimeError:
            # Fallback if event loop already running
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.submit(lambda: asyncio.run(_run_all())).result()

    async def acall_tools(
        self,
        agent: "LLMAgent",
        llm_response: Any,
        tools: ToolSelection | object = _UNSET,
    ) -> list[dict]:
        """
        Asynchronous tool execution (parallel via asyncio.gather).
        """

        tool_calls = getattr(llm_response, "tool_calls", [])
        if not tool_calls:
            return []

        available_tools = self._get_tool_execution_map(tools)

        tasks = [
            self._process_tool_call(agent, tc, i, available_tools)
            for i, tc in enumerate(tool_calls)
        ]

        return await asyncio.gather(*tasks)
