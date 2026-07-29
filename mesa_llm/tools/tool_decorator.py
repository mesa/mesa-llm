from __future__ import annotations

import inspect
import math
import re
import textwrap
import warnings
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

try:  # Python 3.10+ provides UnionType for PEP 604 unions (e.g., int | str)
    from types import UnionType  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback for very old Python
    UnionType = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from mesa_llm.tools.tool_manager import ToolManager


_GLOBAL_TOOL_REGISTRY: dict[str, Callable] = {}
_TOOL_CALLBACKS: list[Callable[[Callable], None]] = []


def add_tool_callback(callback: Callable[[Callable], None]):
    """Deprecated no-op for the old implicit global tool propagation hook."""
    warnings.warn(
        "`add_tool_callback` is deprecated and no longer called for bare "
        "`@tool` registrations; configure tools explicitly instead.",
        DeprecationWarning,
        stacklevel=2,
    )


# ---------- helper functions ----------------------------------------------------
class DocstringParsingError(Exception):
    """Raised when a Google-style docstring cannot be parsed."""


_ARG_HEADER_RE = re.compile(r"^\s*Args?:\s*$", re.IGNORECASE)
_RET_HEADER_RE = re.compile(r"^\s*Returns?:\s*$", re.IGNORECASE)
_PARAM_LINE_RE = re.compile(r"^\s*(\w+)\s*:\s*(.+)$")


def _validate_numeric_literal_schema_ambiguity(annotation: Any) -> None:
    """Reject Python numeric Literal alternatives that JSON Schema conflates."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        _reject_json_equivalent_numeric_literal_values(args)
        return

    if origin is Annotated:
        if args:
            _validate_numeric_literal_schema_ambiguity(args[0])
        return

    if origin is Union or (UnionType is not None and origin is UnionType):
        numeric_values_by_branch = [
            _nested_numeric_literal_values(member) for member in args
        ]
        for index, branch_values in enumerate(numeric_values_by_branch):
            for other_branch_values in numeric_values_by_branch[index + 1 :]:
                _reject_cross_branch_numeric_literal_ambiguity(
                    branch_values,
                    other_branch_values,
                )
        for member in args:
            _validate_numeric_literal_schema_ambiguity(member)
        return

    if origin in {list, set}:
        if args:
            _validate_numeric_literal_schema_ambiguity(args[0])
        return

    if origin is tuple:
        for item_annotation in args:
            if item_annotation is not Ellipsis:
                _validate_numeric_literal_schema_ambiguity(item_annotation)
        return

    if origin is dict and len(args) >= 2:
        _validate_numeric_literal_schema_ambiguity(args[1])


def _nested_numeric_literal_values(
    annotation: Any,
    annotation_path: tuple[tuple[Any, int], ...] = (),
) -> list[tuple[tuple[tuple[Any, int], ...], int | float]]:
    """Return numeric Literal values nested within one union member."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        return [
            (annotation_path, value) for value in args if type(value) in {int, float}
        ]
    if origin is Annotated and args:
        return _nested_numeric_literal_values(args[0], annotation_path)
    if origin is Union or (UnionType is not None and origin is UnionType):
        return [
            value
            for member in args
            for value in _nested_numeric_literal_values(member, annotation_path)
        ]

    if origin in {list, set} and args:
        return _nested_numeric_literal_values(
            args[0],
            (*annotation_path, ("array", 0)),
        )

    if origin is tuple:
        return [
            value
            for index, item_annotation in enumerate(args)
            if item_annotation is not Ellipsis
            for value in _nested_numeric_literal_values(
                item_annotation,
                (*annotation_path, (tuple, index)),
            )
        ]

    if origin is dict and len(args) >= 2:
        return _nested_numeric_literal_values(
            args[1],
            (*annotation_path, (dict, 1)),
        )

    return []


def _reject_cross_branch_numeric_literal_ambiguity(
    values: list[tuple[tuple[tuple[Any, int], ...], int | float]],
    other_values: list[tuple[tuple[tuple[Any, int], ...], int | float]],
) -> None:
    for annotation_path, value in values:
        for other_annotation_path, other_value in other_values:
            if (
                annotation_path == other_annotation_path
                and type(value) is not type(other_value)
                and value == other_value
            ):
                _raise_json_equivalent_numeric_literal_error(value, other_value)


def _reject_json_equivalent_numeric_literal_values(
    values: tuple[Any, ...] | list[int | float],
) -> None:
    numeric_values = [value for value in values if type(value) in {int, float}]
    for index, value in enumerate(numeric_values):
        for other_value in numeric_values[index + 1 :]:
            if type(value) is not type(other_value) and value == other_value:
                _raise_json_equivalent_numeric_literal_error(value, other_value)


def _raise_json_equivalent_numeric_literal_error(
    value: int | float,
    other_value: int | float,
) -> None:
    raise TypeError(
        "Literal schemas cannot contain JSON-equivalent numeric "
        "values with different Python types; "
        f"got {value!r} ({type(value).__name__}) and "
        f"{other_value!r} ({type(other_value).__name__})."
    )


def _python_to_json_type(py_type: Any) -> dict[str, Any]:
    """
    Convert Python type hints to JSON Schema type definitions.

    Handles:
    - Basic types: int, str, float, bool
    - Collections: list, tuple, set
    - Generics: list[int], tuple[int, int], etc.
    - Literal types: Literal["a", "b"]
    - Union types: Union[int, str], int | str
    - Optional types: Optional[int], int | None
    - Nested types: list[tuple[int, str]]
    """

    _validate_numeric_literal_schema_ambiguity(py_type)

    # Handle None type
    if py_type is type(None):
        return {"type": "null"}

    # Handle string annotations by trying to evaluate them
    if isinstance(py_type, str):
        # Try to handle common string representations
        try:
            # Handle basic generic patterns like "list[int]", "tuple[int, int]"
            if "[" in py_type and "]" in py_type:
                base_type = py_type.split("[")[0].strip()
                # Extract the content inside brackets
                inner_content = py_type[py_type.find("[") + 1 : py_type.rfind("]")]

                # Map string type names to actual types
                type_mapping = {
                    "int": int,
                    "str": str,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "tuple": tuple,
                    "dict": dict,
                    "set": set,
                }

                if base_type in type_mapping:
                    base = type_mapping[base_type]
                    if base in (list, tuple, set):
                        # Handle array-like types
                        if "," in inner_content:
                            # Multiple types like tuple[int, str]
                            return {
                                "type": "array",
                                "items": {"type": "string"},
                            }  # Fallback for mixed types
                        else:
                            # Single type like list[int]
                            item_type = type_mapping.get(inner_content.strip(), str)
                            return {
                                "type": "array",
                                "items": _python_to_json_type(item_type),
                            }

            # Try to get the base type for simple cases
            base_type = py_type.split("[")[0].strip()
            type_mapping = {
                "int": int,
                "str": str,
                "float": float,
                "bool": bool,
                "list": list,
                "tuple": tuple,
                "dict": dict,
            }
            if base_type in type_mapping:
                py_type = type_mapping[base_type]

        except Exception:
            # If parsing fails, default to string
            return {"type": "string"}

    # Get the origin and args for generic types
    origin = get_origin(py_type)
    args = get_args(py_type)

    # Handle Literal before the generic fallback treats it as an object.
    if origin is Literal:
        literal_type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            type(None): "null",
        }
        literal_types = []
        for value in args:
            value_type = type(value)
            if value_type not in literal_type_mapping or (
                value_type is float and not math.isfinite(value)
            ):
                raise TypeError(
                    "Literal schemas support only finite JSON scalar values; "
                    f"got {value!r}."
                )
            literal_types.append(literal_type_mapping[value_type])

        schema: dict[str, Any] = {"enum": list(args)}
        if literal_types and len(set(literal_types)) == 1:
            schema = {"type": literal_types[0], **schema}
        return schema

    # Handle Union types (including Optional which is Union[T, None])
    if origin is Union or (UnionType is not None and origin is UnionType):
        # Check if it's Optional (Union with None)
        non_none_args = [arg for arg in args if arg is not type(None)]

        if len(non_none_args) == 1 and type(None) in args:
            # This is Optional[T] - handle the non-None type but allow null
            base_schema = _python_to_json_type(non_none_args[0])
            if "enum" in base_schema:
                return {"anyOf": [base_schema, {"type": "null"}]}
            # Add null as an allowed type
            if "type" in base_schema:
                if isinstance(base_schema["type"], list):
                    base_schema["type"].append("null")
                else:
                    base_schema["type"] = [base_schema["type"], "null"]
            else:
                base_schema = {"anyOf": [base_schema, {"type": "null"}]}
            return base_schema

        elif len(non_none_args) > 1:
            # Multiple non-None types - create anyOf schema
            return {"anyOf": [_python_to_json_type(arg) for arg in args]}
        else:
            # Only None type
            return {"type": "null"}

    # Handle generic types
    if origin is not None:
        # Handle list, tuple, set as arrays
        if origin in (list, tuple, set):
            if args:
                # Handle tuple with specific types like tuple[int, str]
                if origin is tuple and len(args) > 1:
                    # For tuples with multiple specific types, we'll use array with mixed items
                    # JSON Schema doesn't handle tuples with different types perfectly
                    item_schemas = [_python_to_json_type(arg) for arg in args]
                    # If all items have the same type, use that type
                    if (
                        len(
                            {
                                item.get("type")
                                for item in item_schemas
                                if "type" in item
                            }
                        )
                        == 1
                    ):
                        return {"type": "array", "items": item_schemas[0]}
                    else:
                        # Mixed types - use anyOf for items
                        return {"type": "array", "items": {"anyOf": item_schemas}}
                else:
                    # Single type parameter like list[int] or tuple[int, ...]
                    item_type = args[0]
                    return {"type": "array", "items": _python_to_json_type(item_type)}
            else:
                # No type parameters - generic array
                return {"type": "array", "items": {"type": "string"}}

        # Handle dict
        elif origin is dict:
            if len(args) >= 2:
                # dict[str, int] -> object with string values of int type
                value_type = _python_to_json_type(args[1])
                return {"type": "object", "additionalProperties": value_type}
            else:
                return {"type": "object"}

        # Use the origin type for other generics
        py_type = origin

    # Handle basic Python types
    type_mapping = {
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
        bytes: {"type": "string", "format": "byte"},
        list: {"type": "array", "items": {"type": "string"}},
        tuple: {"type": "array", "items": {"type": "string"}},
        set: {"type": "array", "items": {"type": "string"}},
        dict: {"type": "object"},
    }

    return type_mapping.get(py_type, {"type": "object"})


def _parse_docstring(
    func: callable,
    ignore_agent: bool = True,
) -> tuple[str, dict[str, str], str | None]:
    """
    Parse a function's Google-style docstring.

    Args:
        func: The function to parse the docstring of.
        ignore_agent: If True, skip validating docstring entries for any
            parameter named `agent`. Default is True.

    Returns:
        summary: One-line/high-level description that appears before the *Args* section.
        param_desc: Mapping *param name → description* (text only, no types).
        return_desc: Description of the value in the *Returns* section, or *None* if that section is absent.
    """
    # ---------- fetch & pre-process -------------------------------------------------
    raw = inspect.getdoc(func) or ""
    if not raw:
        raise DocstringParsingError(f"{func.__name__} has no docstring.")

    # Normalise indentation & line endings
    lines = textwrap.dedent(raw).strip().splitlines()

    # ---------- locate block boundaries -------------------------------------------
    try:
        args_idx = next(i for i, ln in enumerate(lines) if _ARG_HEADER_RE.match(ln))
    except StopIteration:
        args_idx = None

    try:
        ret_idx = next(i for i, ln in enumerate(lines) if _RET_HEADER_RE.match(ln))
    except StopIteration:
        ret_idx = None

    # Short description = from top up to first blank line or Args:
    cut = (
        args_idx
        if args_idx is not None
        else ret_idx
        if ret_idx is not None
        else len(lines)
    )
    for i, ln in enumerate(lines[:cut]):
        if ln.strip() == "":
            cut = i
            break
    summary = " ".join(ln.strip() for ln in lines[:cut]).strip()

    # ---------- parse *Args* -------------------------------------------------------
    param_desc: dict[str, str] = {}
    if args_idx is not None:
        i = args_idx + 1
        while i < len(lines) and lines[i].strip() == "":
            i += 1  # skip blank lines

        while i < len(lines) and (ret_idx is None or i < ret_idx):
            m = _PARAM_LINE_RE.match(lines[i])
            if not m:
                raise DocstringParsingError(
                    f"Malformed parameter line in {func.__name__}: '{lines[i]}'"
                )
            name, desc = m.groups()
            desc_lines = [desc.rstrip()]
            i += 1
            # grab any following indented continuation lines
            while (
                i < len(lines)
                and (ret_idx is None or i < ret_idx)
                and (lines[i].startswith(" ") or lines[i].startswith("\t"))
                and not _PARAM_LINE_RE.match(
                    lines[i]
                )  # Don't treat other parameters as continuation
            ):
                desc_lines.append(lines[i].strip())
                i += 1
            param_desc[name] = " ".join(desc_lines).strip()
            # skip possible extra blank lines
            while i < len(lines) and lines[i].strip() == "":
                i += 1

    # ---------- parse *Returns* ----------------------------------------------------
    return_desc: str | None = None
    if ret_idx is not None:
        ret_body = [ln.strip() for ln in lines[ret_idx + 1 :] if ln.strip()]
        return_desc = " ".join(ret_body) if ret_body else None

    # ---------- validation ---------------------------------------------------------
    sig_params: list[str] = [
        p.name
        for p in inspect.signature(func).parameters.values()
        if not (ignore_agent and p.name.lower() == "agent")
    ]
    missing = [p for p in sig_params if p not in param_desc]
    if missing:
        raise DocstringParsingError(
            f"Docstring for {func.__name__} is missing descriptions for: {missing}"
        )

    return summary, param_desc, return_desc


# ---------- decorator ----------------------------------------------------


def tool(
    fn: Callable | None = None,
    *,
    tool_manager: ToolManager | None = None,
    ignore_agent: bool = True,
):
    """
    Convert a Python function into an LLM-compatible tool.

    The decorator generates a JSON schema from type hints and Google-style
    docstrings, optionally omitting an ``agent`` parameter that Mesa-LLM injects
    during execution. Passing ``tool_manager=`` registers the decorated function
    directly with that manager. Bare ``@tool`` registration stores the function
    in the global registry so callers can opt in explicitly by name or by
    configuring a manager or agent with that tool; it does not make the tool
    automatically available to every manager.

    Args:
        fn: The function to decorate.
        tool_manager: Optional tool manager to register the function with.
        ignore_agent: Whether to omit the ``agent`` parameter from the schema.

    Returns:
        The decorated function.
    """

    def decorator(func: Callable):
        name = func.__name__
        description, arg_docs, return_docs = _parse_docstring(
            func, ignore_agent=ignore_agent
        )

        sig = inspect.signature(func)
        try:
            type_hints = get_type_hints(func)
        except (NameError, AttributeError, TypeError):
            # Fallback to using annotations directly if type_hints evaluation fails
            type_hints = getattr(func, "__annotations__", {})

        properties = {}

        # filter out  agent argument if ignore_agent is True
        if ignore_agent:
            required_params = {
                param_name: _param
                for param_name, _param in sig.parameters.items()
                if param_name.lower() != "agent"
            }
        else:
            required_params = sig.parameters

        for param_name, _param in required_params.items():
            raw_type = type_hints.get(param_name, Any)
            type_schema = _python_to_json_type(raw_type)
            properties[param_name] = {
                **type_schema,
                "description": arg_docs.get(param_name, ""),
            }
            if not arg_docs.get(param_name):
                warnings.warn(
                    f'Missing docstring for argument "{param_name}" in tool "{name}"',
                    stacklevel=2,
                )

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description + " returns: " + (return_docs or ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(required_params),
                },
            },
        }

        func.__tool_schema__ = schema

        if tool_manager:
            tool_manager.register(func)
        else:
            _GLOBAL_TOOL_REGISTRY[name] = func

        return func

    # If fn is provided, it means @tool was used without parentheses
    if fn is not None:
        return decorator(fn)

    # Otherwise, return the decorator for later use
    return decorator
