from __future__ import annotations

import builtins
import inspect
import math
import typing
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import UnionType
from typing import TYPE_CHECKING, Any, get_type_hints

from mesa_llm.tools.tool_decorator import _parse_docstring, _python_to_json_type

if TYPE_CHECKING:
    from mesa_llm.actions.action_manager import ActionManager


_GLOBAL_ACTION_REGISTRY: dict[str, Callable] = {}
_ACTION_ANNOTATION_GLOBALS: dict[str, Any] = {
    name: getattr(typing, name)
    for name in getattr(typing, "__all__", ())
    if hasattr(typing, name)
}
_ACTION_ANNOTATION_GLOBALS.update(
    {
        "Any": Any,
        "NoneType": type(None),
        "any": Any,
        "typing": typing,
    }
)


class ActionAnnotationResolutionError(ValueError):
    """Raised when a non-agent action parameter annotation cannot be resolved."""


class ActionAnnotationContractError(TypeError):
    """Raised when an action annotation cannot be validated and exposed safely."""


class ActionSignatureError(ValueError):
    """Raised when an action signature cannot be exposed as JSON arguments."""


_ACTION_OBJECT_VALUE_PATH_SEGMENT = ("object", "value")
_ACTION_ARRAY_WILDCARD_PATH_SEGMENT = ("array", None, None)


def _validate_action_numeric_literal_ambiguity(
    func: Callable,
    param_name: str,
    annotation: Any,
) -> None:
    """Reject JSON-equivalent numeric Literals in overlapping action shapes."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Literal:
        numeric_values = [value for value in args if type(value) in {int, float}]
        for index, value in enumerate(numeric_values):
            for other_value in numeric_values[index + 1 :]:
                _reject_action_numeric_literal_pair(
                    func,
                    param_name,
                    value,
                    other_value,
                )
        return

    if origin is typing.Annotated:
        if args:
            _validate_action_numeric_literal_ambiguity(func, param_name, args[0])
        return

    if origin in {typing.Union, UnionType}:
        branch_occurrences = [
            _action_numeric_literal_occurrences(member) for member in args
        ]
        for index, occurrences in enumerate(branch_occurrences):
            for other_occurrences in branch_occurrences[index + 1 :]:
                for annotation_path, value in occurrences:
                    for other_path, other_value in other_occurrences:
                        if _action_literal_paths_are_compatible(
                            annotation_path,
                            other_path,
                        ):
                            _reject_action_numeric_literal_pair(
                                func,
                                param_name,
                                value,
                                other_value,
                            )
        for member in args:
            _validate_action_numeric_literal_ambiguity(func, param_name, member)
        return

    if origin in {list, set} and args:
        _validate_action_numeric_literal_ambiguity(func, param_name, args[0])
        return

    if origin is tuple:
        for item_annotation in args:
            if item_annotation is not Ellipsis:
                _validate_action_numeric_literal_ambiguity(
                    func,
                    param_name,
                    item_annotation,
                )
        return

    if origin is dict and len(args) == 2:
        _validate_action_numeric_literal_ambiguity(func, param_name, args[1])


def _action_numeric_literal_occurrences(
    annotation: Any,
    annotation_path: tuple[tuple[Any, ...], ...] = (),
) -> list[tuple[tuple[tuple[Any, ...], ...], int | float]]:
    """Collect exact numeric Literal values at canonical JSON paths."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Literal:
        return [
            (annotation_path, value) for value in args if type(value) in {int, float}
        ]
    if origin is typing.Annotated and args:
        return _action_numeric_literal_occurrences(args[0], annotation_path)
    if origin in {typing.Union, UnionType}:
        return [
            occurrence
            for member in args
            for occurrence in _action_numeric_literal_occurrences(
                member,
                annotation_path,
            )
        ]
    if origin in {list, set} and args:
        return _action_numeric_literal_occurrences(
            args[0],
            (*annotation_path, _ACTION_ARRAY_WILDCARD_PATH_SEGMENT),
        )
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return _action_numeric_literal_occurrences(
                args[0],
                (*annotation_path, _ACTION_ARRAY_WILDCARD_PATH_SEGMENT),
            )
        tuple_length = len(args)
        return [
            occurrence
            for position, item_annotation in enumerate(args)
            for occurrence in _action_numeric_literal_occurrences(
                item_annotation,
                (*annotation_path, ("array", tuple_length, position)),
            )
        ]
    if origin is dict and len(args) == 2:
        return _action_numeric_literal_occurrences(
            args[1],
            (*annotation_path, _ACTION_OBJECT_VALUE_PATH_SEGMENT),
        )
    return []


def _action_literal_paths_are_compatible(
    annotation_path: tuple[tuple[Any, ...], ...],
    other_path: tuple[tuple[Any, ...], ...],
) -> bool:
    """Return whether two canonical paths can address the same JSON value."""
    if len(annotation_path) != len(other_path):
        return False

    for segment, other_segment in zip(annotation_path, other_path, strict=True):
        if segment[0] != other_segment[0]:
            return False
        if segment[0] == "object":
            if segment != other_segment:
                return False
        elif (
            segment != _ACTION_ARRAY_WILDCARD_PATH_SEGMENT
            and other_segment != _ACTION_ARRAY_WILDCARD_PATH_SEGMENT
            and segment != other_segment
        ):
            return False
    return True


def _reject_action_numeric_literal_pair(
    func: Callable,
    param_name: str,
    value: int | float,
    other_value: int | float,
) -> None:
    if type(value) is type(other_value) or value != other_value:
        return
    raise ActionAnnotationContractError(
        "Ambiguous numeric Literal contract for action "
        f"{getattr(func, '__name__', repr(func))!r} parameter {param_name!r}: "
        "JSON-equivalent values with different Python numeric types occur at "
        "compatible JSON paths; "
        f"got {value!r} ({type(value).__name__}) and "
        f"{other_value!r} ({type(other_value).__name__})."
    )


def _action_annotation_to_json_type(annotation: Any) -> dict[str, Any]:
    """Convert a normalized action annotation without applying tool policy."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Annotated and args:
        return _action_annotation_to_json_type(args[0])

    if origin in {typing.Union, UnionType}:
        non_none_args = [member for member in args if member is not type(None)]
        if len(non_none_args) == 1 and type(None) in args:
            base_schema = _action_annotation_to_json_type(non_none_args[0])
            if "enum" in base_schema:
                return {"anyOf": [base_schema, {"type": "null"}]}
            if "type" in base_schema:
                base_type = base_schema["type"]
                base_schema["type"] = (
                    [*base_type, "null"]
                    if isinstance(base_type, list)
                    else [base_type, "null"]
                )
                return base_schema
            return {"anyOf": [base_schema, {"type": "null"}]}
        if len(non_none_args) > 1:
            return {
                "anyOf": [_action_annotation_to_json_type(member) for member in args]
            }
        return {"type": "null"}

    if origin is list:
        return {
            "type": "array",
            "items": _action_annotation_to_json_type(args[0]),
        }

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return {
                "type": "array",
                "items": _action_annotation_to_json_type(args[0]),
            }
        tuple_length = len(args)
        return {
            "type": "array",
            "items": _action_annotation_to_json_type(args[0]),
            "minItems": tuple_length,
            "maxItems": tuple_length,
        }

    if origin is dict:
        return {
            "type": "object",
            "additionalProperties": _action_annotation_to_json_type(args[1]),
        }

    return _python_to_json_type(annotation)


def _validate_action_callable(func: Callable) -> None:
    """Reject callable kinds that cannot return one completed action result."""
    if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
        raise ActionSignatureError(
            "Action "
            f"{getattr(func, '__name__', repr(func))!r} must return one completed "
            "result; generator and async-generator functions are not supported "
            "as actions."
        )


@dataclass(frozen=True)
class ActionMetadata:
    """Metadata generated for an action function."""

    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str]
    return_description: str | None = None


def _get_action_type_hints(
    func: Callable,
    parameter_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Resolve type hints for non-agent action parameters only."""
    annotations = getattr(func, "__annotations__", {})
    if parameter_names is None:
        selected_names = [
            param_name
            for param_name in annotations
            if param_name != "return" and param_name.lower() != "agent"
        ]
    else:
        selected_names = [
            param_name
            for param_name in parameter_names
            if param_name.lower() != "agent"
        ]

    type_hints = {}
    for param_name in selected_names:
        if param_name not in annotations:
            raise ActionAnnotationContractError(
                "Action "
                f"{getattr(func, '__name__', repr(func))!r} parameter "
                f"{param_name!r} must have a supported runtime annotation. "
                "Only the framework-injected 'agent' parameter may be "
                "unannotated."
            )

        resolved_annotation = _resolve_action_annotation(
            func,
            param_name,
            annotations[param_name],
        )
        _validate_action_numeric_literal_ambiguity(
            func,
            param_name,
            resolved_annotation,
        )
        normalized_annotation = _normalize_action_parameter_annotation(
            func,
            param_name,
            resolved_annotation,
        )
        _validate_action_numeric_literal_ambiguity(
            func,
            param_name,
            normalized_annotation,
        )
        _action_annotation_to_json_type(normalized_annotation)
        type_hints[param_name] = normalized_annotation
    return type_hints


def _resolve_action_annotation(
    func: Callable,
    param_name: str,
    annotation: Any,
) -> Any:
    def annotation_holder():
        pass

    annotation_holder.__annotations__ = {"value": annotation}
    annotation_holder.__module__ = getattr(func, "__module__", None)

    try:
        return get_type_hints(
            annotation_holder,
            globalns=_get_action_annotation_globalns(func),
        )["value"]
    except (AttributeError, NameError, SyntaxError, TypeError, ValueError) as exc:
        raise ActionAnnotationResolutionError(
            "Could not resolve annotation for action "
            f"{getattr(func, '__name__', repr(func))!r} parameter "
            f"{param_name!r}: {annotation!r}. Action parameter annotations "
            "must be importable at runtime or use built-in/typing types. "
            "The injected 'agent' parameter is ignored and does not need to "
            "be importable."
        ) from exc


def _get_action_annotation_globalns(func: Callable) -> dict[str, Any]:
    globalns = dict(_ACTION_ANNOTATION_GLOBALS)
    func_globals = getattr(func, "__globals__", None)
    if func_globals is not None:
        globalns.update(func_globals)
    globalns.setdefault("__builtins__", builtins.__dict__)
    return globalns


def _normalize_action_parameter_annotation(
    func: Callable,
    param_name: str,
    annotation: Any,
    *,
    annotation_path: str | None = None,
) -> Any:
    """Validate and normalize one model-exposed action parameter annotation."""
    annotation_path = annotation_path or param_name

    if annotation is Any or annotation is object:
        raise _unsupported_action_annotation_error(
            func,
            param_name,
            annotation_path,
            annotation,
            "unconstrained Any/object annotations are not supported",
        )

    if annotation in {int, float, str, bool, type(None)}:
        return annotation

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Annotated:
        if not args:
            raise _unsupported_action_annotation_error(
                func,
                param_name,
                annotation_path,
                annotation,
                "Annotated must include a supported base annotation",
            )
        return _normalize_action_parameter_annotation(
            func,
            param_name,
            args[0],
            annotation_path=annotation_path,
        )

    if origin is typing.Literal:
        if not args:
            raise _unsupported_action_annotation_error(
                func,
                param_name,
                annotation_path,
                annotation,
                "Literal must contain at least one value",
            )
        return annotation

    if origin in {typing.Union, UnionType}:
        if not args:
            raise _unsupported_action_annotation_error(
                func,
                param_name,
                annotation_path,
                annotation,
                "unions must contain supported member annotations",
            )
        normalized_args = tuple(
            _normalize_action_parameter_annotation(
                func,
                param_name,
                member,
                annotation_path=f"{annotation_path}.union[{index}]",
            )
            for index, member in enumerate(args)
        )
        normalized_union = normalized_args[0]
        for member in normalized_args[1:]:
            normalized_union |= member
        return normalized_union

    if origin is set:
        raise _unsupported_action_annotation_error(
            func,
            param_name,
            annotation_path,
            annotation,
            "set annotations are not supported for action parameters or nested "
            "set elements",
        )

    if origin is list:
        if len(args) != 1:
            raise _unsupported_action_annotation_error(
                func,
                param_name,
                annotation_path,
                annotation,
                "collections must declare one supported item annotation",
            )
        item_annotation = _normalize_action_parameter_annotation(
            func,
            param_name,
            args[0],
            annotation_path=f"{annotation_path}.item",
        )
        return list[item_annotation]

    if origin is tuple:
        if not args:
            raise _unsupported_action_annotation_error(
                func,
                param_name,
                annotation_path,
                annotation,
                "tuples must declare supported item annotations",
            )
        if len(args) == 2 and args[1] is Ellipsis:
            item_annotation = _normalize_action_parameter_annotation(
                func,
                param_name,
                args[0],
                annotation_path=f"{annotation_path}.item",
            )
            return tuple[item_annotation, ...]
        if Ellipsis in args:
            raise _unsupported_action_annotation_error(
                func,
                param_name,
                annotation_path,
                annotation,
                "ellipsis is supported only in tuple[T, ...]",
            )
        item_annotations = tuple(
            _normalize_action_parameter_annotation(
                func,
                param_name,
                item_annotation,
                annotation_path=f"{annotation_path}[{index}]",
            )
            for index, item_annotation in enumerate(args)
        )
        if any(
            item_annotation != item_annotations[0]
            for item_annotation in item_annotations[1:]
        ):
            raise _unsupported_action_annotation_error(
                func,
                param_name,
                annotation_path,
                annotation,
                "fixed tuple action annotations must be homogeneous; every "
                "position must use the same supported annotation",
            )
        return tuple[item_annotations]

    if origin is dict:
        if len(args) != 2:
            raise _unsupported_action_annotation_error(
                func,
                param_name,
                annotation_path,
                annotation,
                "dictionaries must declare string keys and a supported value type",
            )
        key_annotation = _normalize_action_parameter_annotation(
            func,
            param_name,
            args[0],
            annotation_path=f"{annotation_path}.key",
        )
        if key_annotation is not str:
            raise _unsupported_action_annotation_error(
                func,
                param_name,
                f"{annotation_path}.key",
                args[0],
                "JSON object schemas support only unconstrained string keys",
            )
        value_annotation = _normalize_action_parameter_annotation(
            func,
            param_name,
            args[1],
            annotation_path=f"{annotation_path}.value",
        )
        return dict[str, value_annotation]

    if annotation in {list, tuple, set, dict}:
        raise _unsupported_action_annotation_error(
            func,
            param_name,
            annotation_path,
            annotation,
            "bare collections are unconstrained; provide item/key/value annotations",
        )

    raise _unsupported_action_annotation_error(
        func,
        param_name,
        annotation_path,
        annotation,
        "the annotation is not supported by action schema and runtime validation",
    )


def _unsupported_action_annotation_error(
    func: Callable,
    param_name: str,
    annotation_path: str,
    annotation: Any,
    reason: str,
) -> ActionAnnotationContractError:
    return ActionAnnotationContractError(
        "Unsupported annotation for action "
        f"{getattr(func, '__name__', repr(func))!r} parameter "
        f"{param_name!r} at {annotation_path!r}: {annotation!r}; {reason}."
    )


def _validate_injected_agent_parameters(
    func: Callable,
    signature: inspect.Signature | None = None,
) -> tuple[str, ...]:
    """Return injected-agent names, rejecting ambiguous action signatures."""
    if signature is None:
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return ()

    injected_agent_parameters = tuple(
        param_name
        for param_name, param in signature.parameters.items()
        if param_name.casefold() == "agent" and _is_keyword_injectable_parameter(param)
    )
    if len(injected_agent_parameters) > 1:
        raise ActionSignatureError(
            "Action "
            f"{getattr(func, '__name__', repr(func))!r} has multiple injected "
            f"'agent' parameters: {list(injected_agent_parameters)}. At most one "
            "keyword-compatible parameter named 'agent' case-insensitively is "
            "supported."
        )

    return injected_agent_parameters


def _get_action_parameters(
    func: Callable,
    signature: inspect.Signature | None = None,
) -> dict[str, inspect.Parameter]:
    """Return non-agent action parameters, rejecting unsupported signatures."""
    signature = signature or inspect.signature(func)
    injected_agent_parameters = _validate_injected_agent_parameters(func, signature)

    action_params: dict[str, inspect.Parameter] = {}
    unsupported_params = []

    for param_name, param in signature.parameters.items():
        if param_name in injected_agent_parameters:
            continue
        if param.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            unsupported_params.append(_format_unsupported_action_parameter(param))
            continue
        action_params[param_name] = param

    if unsupported_params:
        raise ActionSignatureError(
            "Unsupported parameter(s) for action "
            f"{getattr(func, '__name__', repr(func))!r}: "
            f"{unsupported_params}. Action parameters exposed to the model and "
            "the injected 'agent' parameter must be keyword-compatible named "
            "parameters because action arguments are passed as keyword "
            "arguments. Positional-only parameters, *args, and **kwargs are "
            "not supported."
        )

    return action_params


def _get_action_parameter_contract(
    func: Callable,
    signature: inspect.Signature | None = None,
) -> tuple[dict[str, inspect.Parameter], dict[str, Any]]:
    """Return validated non-agent parameters and normalized type hints."""
    action_params = _get_action_parameters(func, signature)
    type_hints = _get_action_type_hints(func, parameter_names=action_params)
    _validate_action_parameter_defaults(func, action_params, type_hints)
    return action_params, type_hints


def _validate_action_parameter_defaults(
    func: Callable,
    action_params: Mapping[str, inspect.Parameter],
    type_hints: Mapping[str, Any],
) -> None:
    """Require declared action defaults to satisfy annotations without coercion."""
    for param_name, param in action_params.items():
        if param.default is inspect.Parameter.empty:
            continue

        annotation = type_hints[param_name]
        if _action_default_matches_annotation(param.default, annotation):
            continue

        raise ActionSignatureError(
            "Invalid default for action "
            f"{getattr(func, '__name__', repr(func))!r} parameter "
            f"{param_name!r}: {param.default!r} "
            f"({type(param.default).__name__}) does not match normalized "
            f"annotation {annotation!r} without coercion."
        )


def _action_default_matches_annotation(value: Any, annotation: Any) -> bool:
    """Return whether a default exactly satisfies a normalized annotation."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Annotated:
        return bool(args) and _action_default_matches_annotation(value, args[0])
    if origin in {typing.Union, UnionType}:
        return any(_action_default_matches_annotation(value, member) for member in args)
    if origin is typing.Literal:
        return any(
            type(value) is type(candidate) and value == candidate for candidate in args
        )

    if annotation is type(None):
        return value is None
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is float:
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and math.isfinite(value)
        )
    if annotation is str:
        return isinstance(value, str)
    if annotation is bool:
        return isinstance(value, bool)

    if origin is list:
        return isinstance(value, list) and all(
            _action_default_matches_annotation(item, args[0]) for item in value
        )

    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(args) == 2 and args[1] is Ellipsis:
            return all(
                _action_default_matches_annotation(item, args[0]) for item in value
            )
        return len(value) == len(args) and all(
            _action_default_matches_annotation(item, item_annotation)
            for item, item_annotation in zip(value, args, strict=True)
        )

    if origin is dict:
        return isinstance(value, dict) and all(
            _action_default_matches_annotation(key, args[0])
            and _action_default_matches_annotation(item, args[1])
            for key, item in value.items()
        )

    return False


def _is_keyword_injectable_parameter(param: inspect.Parameter) -> bool:
    return param.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


def _get_required_action_parameter_names(
    action_params: Mapping[str, inspect.Parameter],
) -> list[str]:
    """Return schema-required non-agent parameters in signature order."""
    return [
        param_name
        for param_name, param in action_params.items()
        if param.default is inspect.Parameter.empty
    ]


def _format_unsupported_action_parameter(param: inspect.Parameter) -> str:
    if param.kind is inspect.Parameter.POSITIONAL_ONLY:
        kind = "positional-only"
    elif param.kind is inspect.Parameter.VAR_POSITIONAL:
        kind = "*args"
    elif param.kind is inspect.Parameter.VAR_KEYWORD:
        kind = "**kwargs"
    else:
        kind = str(param.kind)
    return f"{param.name!r} ({kind})"


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
        _validate_action_callable(func)
        name = func.__name__
        sig = inspect.signature(func)
        action_params, type_hints = _get_action_parameter_contract(func, sig)
        description, arg_docs, return_docs = _parse_docstring(func, ignore_agent=True)

        required_params = _get_required_action_parameter_names(action_params)

        properties = {}
        for param_name in action_params:
            raw_type = type_hints[param_name]
            properties[param_name] = {
                **_action_annotation_to_json_type(raw_type),
                "description": arg_docs.get(param_name, ""),
            }

        metadata = ActionMetadata(
            name=name,
            description=description,
            parameters=properties,
            required=list(required_params),
            return_description=return_docs,
        )
        schema = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required_params),
                "additionalProperties": False,
            },
        }
        if return_docs:
            schema["returns"] = return_docs

        func.__action_metadata__ = metadata
        func.__action_schema__ = schema

        if action_manager:
            action_manager.register(func)
        else:
            if (
                name in _GLOBAL_ACTION_REGISTRY
                and _GLOBAL_ACTION_REGISTRY[name] is not func
            ):
                raise ValueError(
                    f"Action name {name!r} is already registered to a different "
                    "callable in the global action registry."
                )
            if name not in _GLOBAL_ACTION_REGISTRY:
                _GLOBAL_ACTION_REGISTRY[name] = func

        return func

    if fn is not None:
        return decorator(fn)

    return decorator
