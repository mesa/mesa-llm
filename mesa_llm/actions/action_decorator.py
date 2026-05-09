from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

from mesa_llm.tools.tool_decorator import _parse_docstring, _python_to_json_type

if TYPE_CHECKING:
    from mesa_llm.actions.action_manager import ActionManager


_GLOBAL_ACTION_REGISTRY: dict[str, Callable] = {}


@dataclass(frozen=True)
class ActionMetadata:
    """Metadata generated for an action function."""

    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str]
    return_description: str | None = None


def action(
    fn: Callable | None = None,
    *,
    action_manager: ActionManager | None = None,
    ignore_agent: bool = True,
):
    """
    Convert a Python function into a Mesa-LLM action.

    The decorator generates action metadata and a JSON-schema-compatible
    action spec from type hints and Google-style docstrings. Any ``agent``
    parameter is always omitted from the schema because Mesa-LLM injects it
    during local execution. Bare ``@action`` registration stores the function
    in the global registry so callers can opt in explicitly by name or by
    configuring an action manager; it does not make the action implicitly
    available.

    Args:
        fn: The function to decorate.
        action_manager: Optional action manager to register the function with.
        ignore_agent: Deprecated compatibility parameter. Actions always omit
            ``agent``; passing ``False`` raises ``ValueError``.

    Returns:
        The decorated function.
    """

    if ignore_agent is not True:
        raise ValueError(
            "`@action(ignore_agent=False)` is not supported. Action `agent` "
            "parameters are always injected by Mesa-LLM and are never exposed "
            "in action schemas."
        )

    def decorator(func: Callable):
        name = func.__name__
        description, arg_docs, return_docs = _parse_docstring(func, ignore_agent=True)

        sig = inspect.signature(func)
        try:
            type_hints = get_type_hints(func)
        except (NameError, AttributeError, TypeError):
            type_hints = getattr(func, "__annotations__", {})

        action_params = {
            param_name: param
            for param_name, param in sig.parameters.items()
            if param_name.lower() != "agent"
        }

        properties = {}
        for param_name in action_params:
            raw_type = type_hints.get(param_name, Any)
            properties[param_name] = {
                **_python_to_json_type(raw_type),
                "description": arg_docs.get(param_name, ""),
            }

        metadata = ActionMetadata(
            name=name,
            description=description,
            parameters=properties,
            required=list(action_params),
            return_description=return_docs,
        )
        schema = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(action_params),
            },
        }
        if return_docs:
            schema["returns"] = return_docs

        func.__action_metadata__ = metadata
        func.__action_schema__ = schema

        if action_manager:
            action_manager.register(func)
        else:
            _GLOBAL_ACTION_REGISTRY[name] = func

        return func

    if fn is not None:
        return decorator(fn)

    return decorator
