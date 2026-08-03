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
        _python_to_json_type(resolved_annotation)
        normalized_annotation = _normalize_action_parameter_annotation(
            func,
            param_name,
            resolved_annotation,
        )
        _python_to_json_type(normalized_annotation)
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
        _python_to_json_type(annotation)
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


def _get_action_parameters(
    func: Callable,
    signature: inspect.Signature | None = None,
) -> dict[str, inspect.Parameter]:
    """Return non-agent action parameters, rejecting unsupported signatures."""
    signature = signature or inspect.signature(func)
    action_params: dict[str, inspect.Parameter] = {}
    unsupported_params = []

    for param_name, param in signature.parameters.items():
        if param_name.lower() == "agent" and _is_keyword_injectable_parameter(param):
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
        name = func.__name__
        sig = inspect.signature(func)
        action_params, type_hints = _get_action_parameter_contract(func, sig)
        description, arg_docs, return_docs = _parse_docstring(func, ignore_agent=True)

        required_params = _get_required_action_parameter_names(action_params)

        properties = {}
        for param_name in action_params:
            raw_type = type_hints[param_name]
            properties[param_name] = {
                **_python_to_json_type(raw_type),
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
