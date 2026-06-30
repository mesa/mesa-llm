from __future__ import annotations

import builtins
import inspect
import typing
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
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
    if not annotations:
        return {}

    allowed_names = set(parameter_names) if parameter_names is not None else None
    type_hints = {}
    for param_name, annotation in annotations.items():
        if param_name == "return" or param_name.lower() == "agent":
            continue
        if allowed_names is not None and param_name not in allowed_names:
            continue
        type_hints[param_name] = _resolve_action_annotation(
            func,
            param_name,
            annotation,
        )
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
        action_params = _get_action_parameters(func, sig)
        description, arg_docs, return_docs = _parse_docstring(func, ignore_agent=True)

        type_hints = _get_action_type_hints(func, parameter_names=action_params)
        required_params = _get_required_action_parameter_names(action_params)

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
