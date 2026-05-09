from __future__ import annotations

import contextlib
import copy
import inspect
from collections.abc import Callable
from types import UnionType
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, Field
from terminal_style import style

from mesa_llm.actions.action_decorator import _GLOBAL_ACTION_REGISTRY

_UNSET = object()
ActionRef = Callable | str
ActionSelection = ActionRef | list[ActionRef] | tuple[ActionRef, ...] | None


class ActionChoice(BaseModel):
    """Structured model output for one locally validated action choice."""

    name: str = Field(description="Name of the configured action to execute.")
    arguments: dict[str, Any] = Field(
        description="JSON-compatible arguments for the selected action."
    )
    rationale: str | None = Field(
        default=None,
        description="Optional explanation for why this action was selected.",
    )


class ActionManager:
    """
    Manager for registering and querying explicit Mesa-LLM actions.

    Bare managers expose no actions; pass ``actions=`` to configure the exact
    action capabilities a manager should expose. Constructor string references
    may resolve globally registered bare ``@action`` names, while per-call
    selectors are constrained to the manager's configured actions.
    """

    instances: ClassVar[list[ActionManager]] = []

    def __init__(
        self,
        actions: list[ActionRef] | tuple[ActionRef, ...] | None = None,
    ):
        ActionManager.instances.append(self)
        self.actions: dict[str, Callable] = {}

        if actions is not None:
            self.register_many(actions)

    def register(self, fn: Callable):
        """Register an action function by name."""
        self.actions[fn.__name__] = fn

    def register_many(self, actions: list[ActionRef] | tuple[ActionRef, ...]):
        """Register explicit action callables or registered action names."""
        for action_ref in actions:
            self.register(self._resolve_registration_action_ref(action_ref))

    def has_action(self, name: str) -> bool:
        """Return whether this manager has a configured action by name."""
        return name in self.actions

    def available_actions(
        self,
        agent: Any = None,
        actions: ActionSelection | object = _UNSET,
    ) -> dict[str, Callable]:
        """Return configured actions, optionally narrowed by explicit selectors."""
        del agent
        if actions is _UNSET:
            return dict(self.actions)
        if actions is None:
            return {}
        return dict(self._resolve_action_items(actions))

    def get_actions_schema(
        self,
        agent: Any = None,
        actions: ActionSelection | object = _UNSET,
    ) -> list[dict[str, Any]]:
        """Return action schemas for configured actions or an explicit subset."""
        return [
            self._get_action_schema(fn, schema_name=name)
            for name, fn in self.available_actions(agent, actions).items()
        ]

    def _get_action_schema(
        self,
        action: ActionRef,
        schema_name: str | None = None,
    ) -> dict[str, Any]:
        fn = (
            self._resolve_configured_action_ref(action)
            if isinstance(action, str)
            else action
        )
        schema_name = schema_name or getattr(fn, "__name__", repr(fn))
        schema = getattr(fn, "__action_schema__", None)
        if schema is None:
            return {"error": f"Action {schema_name} missing __action_schema__"}

        if schema.get("name") == schema_name:
            return schema

        aliased_schema = copy.deepcopy(schema)
        aliased_schema["name"] = schema_name
        return aliased_schema

    def _unknown_action_error(self, action_name: str) -> ValueError:
        return self._unknown_action_choice_error(action_name, self.actions)

    def _unknown_action_choice_error(
        self,
        action_name: str,
        available_actions: dict[str, Callable],
    ) -> ValueError:
        return ValueError(
            style(
                "Unknown action name(s): "
                f"{[action_name]}. Available actions: {sorted(available_actions)}",
                color="red",
            )
        )

    def _invalid_action_ref_error(self, action_ref: Any) -> TypeError:
        return TypeError(
            style(
                "Actions must be callables or registered action names, "
                f"got {type(action_ref).__name__}.",
                color="red",
            )
        )

    def _resolve_registration_action_ref(self, action_ref: ActionRef) -> Callable:
        """Resolve constructor action references."""
        if callable(action_ref):
            return action_ref

        if isinstance(action_ref, str):
            if action_ref in self.actions:
                return self.actions[action_ref]
            if action_ref in _GLOBAL_ACTION_REGISTRY:
                return _GLOBAL_ACTION_REGISTRY[action_ref]
            raise ValueError(
                style(
                    "Unknown action name(s): "
                    f"{[action_ref]}. Available actions: "
                    f"{sorted(set(self.actions) | set(_GLOBAL_ACTION_REGISTRY))}",
                    color="red",
                )
            )

        raise self._invalid_action_ref_error(action_ref)

    def _resolve_configured_action_ref(self, action_ref: ActionRef) -> Callable:
        """Resolve per-call action selectors against configured actions only."""
        return self._resolve_configured_action_item(action_ref)[1]

    def _resolve_configured_action_item(
        self,
        action_ref: ActionRef,
    ) -> tuple[str, Callable]:
        """Resolve a selector while preserving configured action names."""
        if isinstance(action_ref, str):
            if action_ref in self.actions:
                return action_ref, self.actions[action_ref]
            raise self._unknown_action_error(action_ref)

        if callable(action_ref):
            action_name = getattr(action_ref, "__name__", repr(action_ref))
            if action_name in self.actions and self.actions[action_name] is action_ref:
                return action_name, self.actions[action_name]
            for configured_name, configured_fn in self.actions.items():
                if configured_fn is action_ref:
                    return configured_name, configured_fn
            raise self._unknown_action_error(action_name)

        raise self._invalid_action_ref_error(action_ref)

    def _normalize_action_selection(
        self,
        actions: ActionSelection,
    ) -> list[ActionRef]:
        """Normalize one or many explicit action selectors to a list."""
        if actions is None:
            return []
        if isinstance(actions, list | tuple):
            return list(actions)
        if isinstance(actions, str) or callable(actions):
            return [actions]
        raise self._invalid_action_ref_error(actions)

    def _resolve_action_items(
        self,
        actions: ActionSelection,
    ) -> list[tuple[str, Callable]]:
        return [
            self._resolve_configured_action_item(action_ref)
            for action_ref in self._normalize_action_selection(actions)
        ]

    def validate(
        self,
        agent: Any,
        action_choice: ActionChoice | dict[str, Any],
        actions: ActionSelection | object = _UNSET,
    ) -> ActionChoice:
        """Validate one structured action choice without executing it.

        Omitted ``actions`` validates against the manager's configured actions.
        Explicit ``actions=None`` or ``[]`` validates against no actions, and an
        explicit selector narrows validation to that configured subset.
        """
        choice = self._coerce_action_choice(action_choice)
        available_actions = self.available_actions(agent=agent, actions=actions)
        if choice.name not in available_actions:
            raise self._unknown_action_choice_error(choice.name, available_actions)

        validated_arguments = self._validate_action_arguments(
            available_actions[choice.name],
            choice,
        )
        return ActionChoice(
            name=choice.name,
            arguments=validated_arguments,
            rationale=choice.rationale,
        )

    def execute(
        self,
        agent: Any,
        action_choice: ActionChoice | dict[str, Any],
        actions: ActionSelection | object = _UNSET,
    ) -> Any:
        """Validate and execute one configured action locally."""
        choice = self.validate(agent, action_choice, actions=actions)
        action_fn = self.available_actions(agent=agent, actions=actions)[choice.name]

        call_arguments = dict(choice.arguments)
        agent_parameter = self._get_agent_parameter_name(action_fn)
        if agent_parameter is not None:
            call_arguments[agent_parameter] = agent

        return action_fn(**call_arguments)

    def _coerce_action_choice(
        self,
        action_choice: ActionChoice | dict[str, Any],
    ) -> ActionChoice:
        if isinstance(action_choice, ActionChoice):
            return action_choice
        if isinstance(action_choice, dict):
            return ActionChoice(**action_choice)
        raise TypeError(
            style(
                "Action choice must be an ActionChoice or dict, "
                f"got {type(action_choice).__name__}.",
                color="red",
            )
        )

    def _validate_action_arguments(
        self,
        action_fn: Callable,
        action_choice: ActionChoice,
    ) -> dict[str, Any]:
        contract = self._get_action_argument_contract(action_fn)
        arguments = dict(action_choice.arguments)
        argument_names = set(arguments)

        supplied_agent_arguments = sorted(
            name for name in argument_names if name.lower() == "agent"
        )
        if supplied_agent_arguments:
            raise ValueError(
                style(
                    "Action arguments must not include framework-injected "
                    f"argument(s) for action {action_choice.name!r}: "
                    f"{supplied_agent_arguments}",
                    color="red",
                )
            )

        missing = sorted(contract["required"] - argument_names)
        if missing:
            raise ValueError(
                style(
                    "Missing required argument(s) for action "
                    f"{action_choice.name!r}: {missing}",
                    color="red",
                )
            )

        if not contract["accepts_extra_arguments"]:
            extra = sorted(argument_names - contract["allowed"])
            if extra:
                raise ValueError(
                    style(
                        "Unexpected argument(s) for action "
                        f"{action_choice.name!r}: {extra}",
                        color="red",
                    )
                )

        return self._validate_and_coerce_action_argument_types(
            action_fn,
            action_choice.name,
            arguments,
            contract,
        )

    def _validate_and_coerce_action_argument_types(
        self,
        action_fn: Callable,
        action_name: str,
        arguments: dict[str, Any],
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            type_hints = get_type_hints(action_fn)
        except (NameError, AttributeError, TypeError):
            type_hints = getattr(action_fn, "__annotations__", {})

        if not type_hints:
            return arguments

        coerced_arguments = dict(arguments)
        for argument_name in contract["allowed"]:
            if argument_name not in coerced_arguments:
                continue
            if argument_name not in type_hints:
                continue

            coerced_arguments[argument_name] = self._validate_and_coerce_value(
                action_name=action_name,
                argument_path=argument_name,
                value=coerced_arguments[argument_name],
                expected_type=type_hints[argument_name],
            )

        return coerced_arguments

    def _validate_and_coerce_value(
        self,
        *,
        action_name: str,
        argument_path: str,
        value: Any,
        expected_type: Any,
    ) -> Any:
        expected_type = self._normalize_action_annotation(expected_type)
        if expected_type is Any:
            return value

        origin = get_origin(expected_type)
        args = get_args(expected_type)

        if origin is Annotated:
            if not args:
                return value
            return self._validate_and_coerce_value(
                action_name=action_name,
                argument_path=argument_path,
                value=value,
                expected_type=args[0],
            )

        if origin in {Union, UnionType}:
            return self._validate_and_coerce_union_value(
                action_name=action_name,
                argument_path=argument_path,
                value=value,
                expected_type=expected_type,
                union_args=args,
            )

        if origin is Literal:
            if value in args:
                return value
            raise self._invalid_action_argument_type_error(
                action_name,
                argument_path,
                expected_type,
                value,
            )

        if expected_type is type(None):
            if value is None:
                return value
            raise self._invalid_action_argument_type_error(
                action_name,
                argument_path,
                expected_type,
                value,
            )

        if expected_type in {int, float}:
            coerced_value = self._coerce_numeric_action_value(value, expected_type)
            if self._is_valid_numeric_action_value(coerced_value, expected_type):
                return coerced_value
            raise self._invalid_action_argument_type_error(
                action_name,
                argument_path,
                expected_type,
                value,
            )

        if expected_type is str:
            if isinstance(value, str):
                return value
            raise self._invalid_action_argument_type_error(
                action_name,
                argument_path,
                expected_type,
                value,
            )

        if expected_type is bool:
            if isinstance(value, bool):
                return value
            raise self._invalid_action_argument_type_error(
                action_name,
                argument_path,
                expected_type,
                value,
            )

        if origin in {list, tuple, set} or expected_type in {list, tuple, set}:
            return self._validate_and_coerce_sequence_value(
                action_name=action_name,
                argument_path=argument_path,
                value=value,
                expected_type=expected_type,
                origin=origin,
                args=args,
            )

        if origin is dict or expected_type is dict:
            return self._validate_and_coerce_mapping_value(
                action_name=action_name,
                argument_path=argument_path,
                value=value,
                args=args,
            )

        return value

    def _validate_and_coerce_union_value(
        self,
        *,
        action_name: str,
        argument_path: str,
        value: Any,
        expected_type: Any,
        union_args: tuple[Any, ...],
    ) -> Any:
        for union_type in union_args:
            with contextlib.suppress(ValueError):
                return self._validate_and_coerce_value(
                    action_name=action_name,
                    argument_path=argument_path,
                    value=value,
                    expected_type=union_type,
                )

        raise self._invalid_action_argument_type_error(
            action_name,
            argument_path,
            expected_type,
            value,
        )

    def _validate_and_coerce_sequence_value(
        self,
        *,
        action_name: str,
        argument_path: str,
        value: Any,
        expected_type: Any,
        origin: Any,
        args: tuple[Any, ...],
    ) -> Any:
        if not isinstance(value, (list, tuple, set)):
            raise self._invalid_action_argument_type_error(
                action_name,
                argument_path,
                expected_type,
                value,
            )

        container_type = origin or expected_type
        item_types = args
        if container_type is tuple and item_types and item_types[-1] is not Ellipsis:
            if len(value) != len(item_types):
                raise self._invalid_action_argument_type_error(
                    action_name,
                    argument_path,
                    expected_type,
                    value,
                )
            coerced_items = [
                self._validate_and_coerce_value(
                    action_name=action_name,
                    argument_path=f"{argument_path}[{index}]",
                    value=item,
                    expected_type=item_types[index],
                )
                for index, item in enumerate(value)
            ]
        else:
            item_type = item_types[0] if item_types else Any
            coerced_items = [
                self._validate_and_coerce_value(
                    action_name=action_name,
                    argument_path=f"{argument_path}[{index}]",
                    value=item,
                    expected_type=item_type,
                )
                for index, item in enumerate(value)
            ]

        if container_type is tuple:
            return tuple(coerced_items)
        if container_type is set:
            try:
                return set(coerced_items)
            except TypeError as exc:
                raise self._invalid_action_argument_type_error(
                    action_name,
                    argument_path,
                    expected_type,
                    value,
                ) from exc
        return list(coerced_items)

    def _validate_and_coerce_mapping_value(
        self,
        *,
        action_name: str,
        argument_path: str,
        value: Any,
        args: tuple[Any, ...],
    ) -> dict[Any, Any]:
        if not isinstance(value, dict):
            raise self._invalid_action_argument_type_error(
                action_name,
                argument_path,
                dict,
                value,
            )

        key_type = args[0] if len(args) >= 1 else Any
        value_type = args[1] if len(args) >= 2 else Any
        return {
            self._validate_and_coerce_value(
                action_name=action_name,
                argument_path=f"{argument_path}.<key>",
                value=key,
                expected_type=key_type,
            ): self._validate_and_coerce_value(
                action_name=action_name,
                argument_path=f"{argument_path}[{key!r}]",
                value=item_value,
                expected_type=value_type,
            )
            for key, item_value in value.items()
        }

    def _coerce_numeric_action_value(self, value: Any, expected_type: type) -> Any:
        if isinstance(value, bool):
            return value
        if self._is_valid_numeric_action_value(value, expected_type):
            return value

        with contextlib.suppress(ValueError, TypeError):
            coerced_value = expected_type(value)
            if expected_type is int and not self._is_lossless_int_coercion(
                value, coerced_value
            ):
                return value
            return coerced_value

        return value

    def _is_lossless_int_coercion(self, value: Any, coerced_value: int) -> bool:
        if isinstance(value, str):
            return True
        return coerced_value == value

    def _is_valid_numeric_action_value(self, value: Any, expected_type: type) -> bool:
        if isinstance(value, bool):
            return False
        if expected_type is int:
            return isinstance(value, int)
        if expected_type is float:
            return isinstance(value, int | float)
        return False

    def _normalize_action_annotation(self, annotation: Any) -> Any:
        if not isinstance(annotation, str):
            return annotation

        return {
            "Any": Any,
            "any": Any,
            "bool": bool,
            "dict": dict,
            "float": float,
            "int": int,
            "list": list,
            "None": type(None),
            "NoneType": type(None),
            "object": Any,
            "set": set,
            "str": str,
            "tuple": tuple,
        }.get(annotation, Any)

    def _invalid_action_argument_type_error(
        self,
        action_name: str,
        argument_path: str,
        expected_type: Any,
        value: Any,
    ) -> ValueError:
        return ValueError(
            style(
                "Invalid argument type for action "
                f"{action_name!r}: {argument_path!r} expected "
                f"{self._format_action_expected_type(expected_type)}, "
                f"got {type(value).__name__}.",
                color="red",
            )
        )

    def _format_action_expected_type(self, expected_type: Any) -> str:
        expected_type = self._normalize_action_annotation(expected_type)
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        if expected_type is Any:
            return "Any"
        if origin is Annotated and args:
            return self._format_action_expected_type(args[0])
        if origin in {Union, UnionType}:
            return " | ".join(self._format_action_expected_type(arg) for arg in args)
        if origin is Literal:
            return "one of " + repr(args)
        if expected_type is type(None):
            return "None"
        if origin is not None:
            return str(expected_type).replace("typing.", "")
        if hasattr(expected_type, "__name__"):
            return expected_type.__name__
        return repr(expected_type)

    def _get_action_argument_contract(self, action_fn: Callable) -> dict[str, Any]:
        signature_allowed: set[str] = set()
        signature_required: set[str] = set()
        accepts_extra_arguments = False
        signature_available = False

        try:
            signature = inspect.signature(action_fn)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            signature_available = True
            for param_name, param in signature.parameters.items():
                if param.kind is inspect.Parameter.VAR_KEYWORD:
                    accepts_extra_arguments = True
                    continue
                if param.kind is inspect.Parameter.VAR_POSITIONAL:
                    continue
                if param_name.lower() == "agent":
                    continue

                signature_allowed.add(param_name)
                if param.default is inspect.Parameter.empty:
                    signature_required.add(param_name)

        schema = getattr(action_fn, "__action_schema__", None)
        schema_allowed: set[str] = set()
        schema_required: set[str] = set()
        if isinstance(schema, dict):
            parameters = schema.get("parameters", {})
            if isinstance(parameters, dict):
                properties = parameters.get("properties", {})
                if isinstance(properties, dict):
                    schema_allowed = set(properties)
                required = parameters.get("required", [])
                if isinstance(required, list | tuple):
                    schema_required = set(required)

        return {
            "allowed": signature_allowed if signature_available else schema_allowed,
            "required": (
                signature_required if signature_available else schema_required
            ),
            "accepts_extra_arguments": accepts_extra_arguments,
        }

    def _get_agent_parameter_name(self, action_fn: Callable) -> str | None:
        try:
            signature = inspect.signature(action_fn)
        except (TypeError, ValueError):
            return None

        for param_name, param in signature.parameters.items():
            if param_name.lower() != "agent":
                continue
            if param.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.VAR_POSITIONAL,
            }:
                continue
            return param_name

        return None
