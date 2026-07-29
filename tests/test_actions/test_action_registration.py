from __future__ import annotations

import gc
import weakref
from typing import TYPE_CHECKING, Any, Literal

import pytest

from mesa_llm.actions import (
    ActionManager,
    ActionMetadata,
    action,
    default_actions,
    wait,
)
from mesa_llm.actions.action_decorator import _GLOBAL_ACTION_REGISTRY

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class _UnsupportedActionPayload:
    pass


@pytest.fixture(autouse=True)
def restore_global_action_registry():
    """Keep bare @action registrations local to each test."""
    original_registry = dict(_GLOBAL_ACTION_REGISTRY)
    yield
    _GLOBAL_ACTION_REGISTRY.clear()
    _GLOBAL_ACTION_REGISTRY.update(original_registry)


def test_action_generates_metadata_and_schema_from_type_hints_and_docstring():
    @action
    def visit(agent, location: str, duration: int, tags: list[str]) -> str:
        """Visit a location.

        Args:
            location: Destination name.
            duration: Number of turns to stay.
            tags: Labels for the visit.

        Returns:
            Visit confirmation.
        """
        del agent
        return f"{location}:{duration}:{tags}"

    metadata = visit.__action_metadata__
    schema = visit.__action_schema__

    assert metadata == ActionMetadata(
        name="visit",
        description="Visit a location.",
        parameters={
            "location": {"type": "string", "description": "Destination name."},
            "duration": {
                "type": "integer",
                "description": "Number of turns to stay.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Labels for the visit.",
            },
        },
        required=["location", "duration", "tags"],
        return_description="Visit confirmation.",
    )
    assert schema == {
        "name": "visit",
        "description": "Visit a location.",
        "parameters": {
            "type": "object",
            "properties": metadata.parameters,
            "required": ["location", "duration", "tags"],
        },
        "returns": "Visit confirmation.",
    }


def test_action_schema_exposes_literal_and_nullable_union():
    @action
    def select_value(
        agent,
        mode: Literal["a", "b"],
        optional_mode: Literal["a", "b"] | None,
        value: int | str | None,
    ) -> str:
        """Select a typed value.

        Args:
            mode: Selection mode.
            optional_mode: Optional selection mode.
            value: Optional selected value.

        Returns:
            Selection confirmation.
        """
        del agent, optional_mode, value
        return mode

    properties = select_value.__action_schema__["parameters"]["properties"]

    assert properties["mode"] == {
        "type": "string",
        "enum": ["a", "b"],
        "description": "Selection mode.",
    }
    assert properties["optional_mode"] == {
        "anyOf": [
            {
                "type": "string",
                "enum": ["a", "b"],
            },
            {"type": "null"},
        ],
        "description": "Optional selection mode.",
    }
    assert properties["value"] == {
        "anyOf": [
            {"type": "integer"},
            {"type": "string"},
            {"type": "null"},
        ],
        "description": "Optional selected value.",
    }


def test_action_schema_exposes_unambiguous_integral_numeric_literals():
    @action
    def select_numeric_value(
        agent,
        float_value: Literal[1.0],
        integer_value: Literal[1],
    ) -> tuple[float, int]:
        """Select numeric literal values.

        Args:
            float_value: Float literal to select.
            integer_value: Integer literal to select.

        Returns:
            The selected values.
        """
        del agent
        return float_value, integer_value

    properties = select_numeric_value.__action_schema__["parameters"]["properties"]

    assert properties["float_value"] == {
        "type": "number",
        "enum": [1.0],
        "description": "Float literal to select.",
    }
    assert properties["integer_value"] == {
        "type": "integer",
        "enum": [1],
        "description": "Integer literal to select.",
    }


def test_action_schema_rejects_ambiguous_numeric_literal_without_registration():
    with pytest.raises(
        TypeError,
        match=r"JSON-equivalent|ambiguous|indistinguishable",
    ):

        @action
        def select_numeric_value(agent, value: Literal[1, 1.0]) -> int | float:
            """Select a numeric value.

            Args:
                value: Numeric value to select.

            Returns:
                The selected value.
            """
            del agent
            return value

    assert "select_numeric_value" not in _GLOBAL_ACTION_REGISTRY


@pytest.mark.parametrize(
    "annotation",
    [
        Literal[1] | Literal[1.0],
        list[Literal[1] | Literal[1.0]],
    ],
    ids=["split-union", "nested-split-union"],
)
def test_action_schema_rejects_split_ambiguous_numeric_literal_without_registration(
    annotation,
):
    def select_numeric_value(agent, value) -> int | float:
        """Select a numeric value.

        Args:
            value: Numeric value to select.

        Returns:
            The selected value.
        """
        del agent
        return value

    select_numeric_value.__annotations__["value"] = annotation

    with pytest.raises(
        TypeError,
        match=r"JSON-equivalent|ambiguous|indistinguishable",
    ):
        action(select_numeric_value)

    assert "select_numeric_value" not in _GLOBAL_ACTION_REGISTRY
    assert not hasattr(select_numeric_value, "__action_metadata__")
    assert not hasattr(select_numeric_value, "__action_schema__")


@pytest.mark.parametrize(
    "annotation",
    [
        list[Literal[1]] | set[Literal[1.0]],
        dict[str, list[Literal[1]]] | dict[str, set[Literal[1.0]]],
    ],
    ids=["collection-branches", "nested-dictionary-branches"],
)
def test_action_schema_rejects_structurally_ambiguous_literal_without_side_effects(
    annotation,
):
    mutations = []

    def select_numeric_value(agent, value) -> int | float:
        """Select a numeric value.

        Args:
            value: Numeric value to select.

        Returns:
            The selected value.
        """
        mutations.append((agent, value))
        return value

    select_numeric_value.__annotations__["value"] = annotation

    with pytest.raises(
        TypeError,
        match=r"JSON-equivalent|ambiguous|indistinguishable",
    ):
        action(select_numeric_value)

    assert "select_numeric_value" not in _GLOBAL_ACTION_REGISTRY
    assert not hasattr(select_numeric_value, "__action_metadata__")
    assert not hasattr(select_numeric_value, "__action_schema__")
    assert mutations == []


def test_action_rejects_empty_literal_without_registration_or_mutation():
    mutations = []

    def select_value(agent, value) -> str:
        """Select a literal value.

        Args:
            value: Literal value to select.

        Returns:
            Selection confirmation.
        """
        mutations.append((agent, value))
        return "selected"

    select_value.__annotations__["value"] = Literal[()]

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(select_value)

    message = str(exc_info.value)
    assert "Literal" in message
    assert "empty" in message.lower() or "at least one" in message.lower()
    assert "select_value" not in _GLOBAL_ACTION_REGISTRY
    assert not hasattr(select_value, "__action_metadata__")
    assert not hasattr(select_value, "__action_schema__")
    assert mutations == []


def test_action_omits_agent_parameter_from_schema_by_default():
    @action
    def move(agent, direction: str) -> str:
        """Move in a direction.

        Args:
            direction: Direction to move.

        Returns:
            Move confirmation.
        """
        del agent
        return direction

    params = move.__action_schema__["parameters"]

    assert set(params["properties"]) == {"direction"}
    assert params["required"] == ["direction"]
    assert "agent" not in move.__action_metadata__.parameters

    manager = ActionManager(actions=[move])
    manager_params = manager.get_actions_schema()[0]["parameters"]

    assert set(manager_params["properties"]) == {"direction"}
    assert manager_params["required"] == ["direction"]


def test_action_schema_marks_only_non_default_non_agent_parameters_required():
    @action
    def send_message(
        agent,
        recipient_id: int,
        message: str,
        priority: int = 1,
        channel: str = "local",
    ) -> str:
        """Send a message.

        Args:
            recipient_id: Recipient agent id.
            message: Message body.
            priority: Delivery priority.
            channel: Delivery channel.

        Returns:
            Message confirmation.
        """
        del agent, priority, channel
        return message

    params = send_message.__action_schema__["parameters"]

    assert set(params["properties"]) == {
        "recipient_id",
        "message",
        "priority",
        "channel",
    }
    assert params["required"] == ["recipient_id", "message"]
    assert send_message.__action_metadata__.required == ["recipient_id", "message"]


def test_action_ignore_agent_false_is_disabled_with_clear_error():
    with pytest.raises(ValueError, match=r"ignore_agent=False.*not supported"):

        @action(ignore_agent=False)
        def expose_agent(agent) -> str:
            """Attempt to expose the framework-injected agent.

            Returns:
                Exposure confirmation.
            """
            del agent
            return "exposed"


def test_bare_action_global_registration_can_be_opted_into_by_name():
    @action
    def registered_action(agent, message: str) -> str:
        """Registered action.

        Args:
            message: Message to store.

        Returns:
            The message.
        """
        del agent
        return message

    assert _GLOBAL_ACTION_REGISTRY["registered_action"] is registered_action

    manager = ActionManager(actions=["registered_action"])

    assert manager.actions == {"registered_action": registered_action}
    assert manager.available_actions() == {"registered_action": registered_action}


def test_action_manager_argument_registers_directly_with_that_manager():
    manager = ActionManager()

    @action(action_manager=manager)
    def direct_action(agent, score: float) -> str:
        """Direct action.

        Args:
            score: Score to record.

        Returns:
            Direct action confirmation.
        """
        del agent
        return f"score={score}"

    expected_schema = {
        "name": "direct_action",
        "description": "Direct action.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "description": "Score to record.",
                },
            },
            "required": ["score"],
        },
        "returns": "Direct action confirmation.",
    }

    assert manager.available_actions() == {"direct_action": direct_action}
    assert manager.get_actions_schema() == [expected_schema]
    assert manager.get_actions_schema(actions="direct_action") == [expected_schema]
    assert direct_action.__action_schema__ == expected_schema
    assert "direct_action" not in _GLOBAL_ACTION_REGISTRY


def test_action_manager_and_exclusively_owned_callable_can_be_garbage_collected():
    class ActionState:
        value = "performed"

    def register_owned_action(manager):
        state = ActionState()

        @action(action_manager=manager)
        def owned_action(agent) -> str:
            """Use state owned exclusively by this action.

            Returns:
                The owned state value.
            """
            del agent
            return state.value

        return weakref.ref(owned_action), weakref.ref(state)

    manager = ActionManager()
    action_ref, state_ref = register_owned_action(manager)
    manager_ref = weakref.ref(manager)

    assert manager.available_actions()["owned_action"](None) == "performed"
    assert action_ref() is manager.actions["owned_action"]
    assert state_ref() is not None

    del manager
    gc.collect()

    assert manager_ref() is None
    assert action_ref() is None
    assert state_ref() is None


def test_action_schema_resolves_postponed_annotations_with_type_checking_agent_import():
    @action
    def notify_neighbor(
        agent: LLMAgent,
        listener_agent_ids: list[int],
        message: str,
    ) -> str:
        """Notify a neighboring agent.

        Args:
            listener_agent_ids: Listener ids.
            message: Message body.

        Returns:
            Notification confirmation.
        """
        del agent
        return message

    params = notify_neighbor.__action_schema__["parameters"]

    assert set(params["properties"]) == {"listener_agent_ids", "message"}
    assert params["required"] == ["listener_agent_ids", "message"]
    assert params["properties"]["listener_agent_ids"] == {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Listener ids.",
    }
    assert params["properties"]["message"] == {
        "type": "string",
        "description": "Message body.",
    }
    assert "agent" not in notify_neighbor.__action_metadata__.parameters


def test_action_schema_ignores_string_llm_agent_parameter_annotation():
    @action
    def inspect_agent(agent: LLMAgent, note: str) -> str:
        """Inspect the injected agent safely.

        Args:
            note: Inspection note.

        Returns:
            Inspection confirmation.
        """
        del agent
        return note

    params = inspect_agent.__action_schema__["parameters"]

    assert set(params["properties"]) == {"note"}
    assert params["required"] == ["note"]
    assert "agent" not in inspect_agent.__action_metadata__.parameters


def test_action_rejects_missing_exposed_parameter_annotation_without_registration():
    def apply_payload(agent, payload) -> str:
        """Apply a payload.

        Args:
            payload: Payload to apply.

        Returns:
            Payload confirmation.
        """
        del agent, payload
        return "applied"

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(apply_payload)

    message = str(exc_info.value)
    assert "apply_payload" in message
    assert "payload" in message
    assert "annotation" in message.lower()
    assert "apply_payload" not in _GLOBAL_ACTION_REGISTRY
    assert not hasattr(apply_payload, "__action_metadata__")
    assert not hasattr(apply_payload, "__action_schema__")


@pytest.mark.parametrize(
    "annotation",
    [
        Any,
        object,
        _UnsupportedActionPayload,
        list[Any],
        list[_UnsupportedActionPayload],
        dict[str, object],
        list,
        dict,
        tuple,
        set,
    ],
    ids=[
        "any",
        "object",
        "custom",
        "nested-any",
        "nested-custom",
        "nested-object",
        "bare-list",
        "bare-dict",
        "bare-tuple",
        "bare-set",
    ],
)
def test_action_rejects_unsupported_or_unconstrained_parameter_annotation(
    annotation,
):
    def apply_payload(agent, payload) -> str:
        """Apply a payload.

        Args:
            payload: Payload to apply.

        Returns:
            Payload confirmation.
        """
        del agent, payload
        return "applied"

    apply_payload.__annotations__["payload"] = annotation

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(apply_payload)

    message = str(exc_info.value)
    assert "apply_payload" in message
    assert "payload" in message
    assert "annotation" in message.lower()
    assert "apply_payload" not in _GLOBAL_ACTION_REGISTRY
    assert not hasattr(apply_payload, "__action_metadata__")
    assert not hasattr(apply_payload, "__action_schema__")


@pytest.mark.parametrize(
    "annotation",
    [
        set[list[int]],
        set[tuple[int, list[int]]],
        set[set[int]],
        set[dict[str, int]],
    ],
    ids=[
        "list-element",
        "nested-list-element",
        "set-element",
        "dict-element",
    ],
)
def test_action_rejects_recursively_unhashable_set_element_annotation(
    annotation,
):
    def apply_values(agent, values) -> str:
        """Apply a set of values.

        Args:
            values: Values to apply.

        Returns:
            Application confirmation.
        """
        del agent, values
        return "applied"

    apply_values.__annotations__["values"] = annotation

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(apply_values)

    message = str(exc_info.value)
    assert "apply_values" in message
    assert "values" in message
    assert "hashable" in message.lower() or "set element" in message.lower()
    assert "apply_values" not in _GLOBAL_ACTION_REGISTRY
    assert not hasattr(apply_values, "__action_metadata__")
    assert not hasattr(apply_values, "__action_schema__")


def test_action_exempts_injected_agent_and_return_annotations_from_contract():
    @action
    def inspect_payload(
        agent: Any,
        payload: list[int],
    ) -> _UnsupportedActionPayload:
        """Inspect a payload.

        Args:
            payload: Integer payload to inspect.

        Returns:
            Inspection result.
        """
        del agent, payload
        return _UnsupportedActionPayload()

    params = inspect_payload.__action_schema__["parameters"]

    assert params["properties"] == {
        "payload": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Integer payload to inspect.",
        }
    }
    assert params["required"] == ["payload"]


def test_action_preserves_supported_fixed_tuple_annotation():
    @action
    def move_to(agent, coordinates: tuple[int, int]) -> tuple[int, int]:
        """Move to fixed coordinates.

        Args:
            coordinates: Coordinates to move to.

        Returns:
            The selected coordinates.
        """
        del agent
        return coordinates

    assert move_to.__action_schema__["parameters"]["properties"]["coordinates"] == {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Coordinates to move to.",
    }


def test_action_rejects_non_agent_positional_only_parameter():
    with pytest.raises(ValueError) as exc_info:

        @action
        def target_position(position: str, /, speed: int) -> str:
            """Target a position.

            Args:
                position: Position to target.
                speed: Movement speed.

            Returns:
                Target confirmation.
            """
            return f"{position}:{speed}"

    message = str(exc_info.value)

    assert "target_position" in message
    assert "position" in message
    assert "positional-only" in message


def test_action_rejects_non_agent_varargs_parameter():
    with pytest.raises(ValueError) as exc_info:

        @action
        def collect_items(agent, *items: str) -> str:
            """Collect items.

            Args:
                items: Items to collect.

            Returns:
                Collection confirmation.
            """
            del agent
            return ",".join(items)

    message = str(exc_info.value)

    assert "collect_items" in message
    assert "items" in message
    assert "*args" in message or "variadic positional" in message


def test_action_rejects_non_agent_varkwargs_parameter():
    with pytest.raises(ValueError) as exc_info:

        @action
        def configure_agent(agent, **options: str) -> str:
            """Configure an agent.

            Args:
                options: Configuration options.

            Returns:
                Configuration confirmation.
            """
            del agent
            return ",".join(sorted(options))

    message = str(exc_info.value)

    assert "configure_agent" in message
    assert "options" in message
    assert "**kwargs" in message or "variadic keyword" in message


def test_action_rejects_positional_only_agent_parameter():
    with pytest.raises(ValueError) as exc_info:

        @action
        def inspect_agent(agent, /, note: str) -> str:
            """Inspect an agent.

            Args:
                note: Inspection note.

            Returns:
                Inspection confirmation.
            """
            del agent
            return note

    message = str(exc_info.value)

    assert "inspect_agent" in message
    assert "agent" in message
    assert "positional-only" in message


def test_action_rejects_varargs_agent_parameter():
    with pytest.raises(ValueError) as exc_info:

        @action
        def capture_agent_args(*agent) -> str:
            """Capture variadic agent arguments.

            Returns:
                Capture confirmation.
            """
            return repr(agent)

    message = str(exc_info.value)

    assert "capture_agent_args" in message
    assert "agent" in message
    assert "*args" in message or "variadic positional" in message


def test_action_rejects_varkwargs_agent_parameter():
    with pytest.raises(ValueError) as exc_info:

        @action
        def capture_agent_kwargs(**agent) -> str:
            """Capture variadic agent keyword arguments.

            Returns:
                Capture confirmation.
            """
            return repr(agent)

    message = str(exc_info.value)

    assert "capture_agent_kwargs" in message
    assert "agent" in message
    assert "**kwargs" in message or "variadic keyword" in message


def test_empty_action_managers_expose_no_actions():
    assert ActionManager().available_actions() == {}
    assert ActionManager(actions=None).available_actions() == {}
    assert ActionManager(actions=[]).available_actions() == {}

    manager = ActionManager()

    assert manager.get_actions_schema() == []
    assert manager.get_actions_schema(actions=None) == []
    assert manager.get_actions_schema(actions=[]) == []


def test_callable_configuration_exposes_exactly_that_action():
    @action
    def first_action(agent, amount: int) -> int:
        """First action.

        Args:
            amount: Amount to return.

        Returns:
            The amount.
        """
        del agent
        return amount

    @action
    def second_action(agent, amount: int) -> int:
        """Second action.

        Args:
            amount: Amount to return.

        Returns:
            The amount.
        """
        del agent
        return amount

    manager = ActionManager(actions=[first_action])

    assert manager.actions == {"first_action": first_action}
    assert manager.available_actions() == {"first_action": first_action}
    assert "second_action" not in manager.actions


def test_available_actions_and_schema_selector_behavior():
    @action
    def action_a(agent, value: int) -> int:
        """Action A.

        Args:
            value: Value to return.

        Returns:
            The value.
        """
        del agent
        return value

    @action
    def action_b(agent, name: str) -> str:
        """Action B.

        Args:
            name: Name to return.

        Returns:
            The name.
        """
        del agent
        return name

    manager = ActionManager(actions=[action_a, action_b])

    assert manager.available_actions() == {
        "action_a": action_a,
        "action_b": action_b,
    }
    assert manager.available_actions(actions=None) == {}
    assert manager.available_actions(actions=[]) == {}
    assert manager.available_actions(actions="action_a") == {"action_a": action_a}
    assert manager.available_actions(actions=action_b) == {"action_b": action_b}

    assert [schema["name"] for schema in manager.get_actions_schema()] == [
        "action_a",
        "action_b",
    ]
    assert manager.get_actions_schema(actions=None) == []
    assert manager.get_actions_schema(actions=[]) == []
    action_a_schema_names = [
        schema["name"] for schema in manager.get_actions_schema(actions="action_a")
    ]
    action_b_schema_names = [
        schema["name"] for schema in manager.get_actions_schema(actions=action_b)
    ]

    assert action_a_schema_names == ["action_a"]
    assert action_b_schema_names == ["action_b"]


def test_unknown_or_unconfigured_names_and_callables_fail_fast():
    @action
    def configured_action(agent, value: int) -> int:
        """Configured action.

        Args:
            value: Value to return.

        Returns:
            The value.
        """
        del agent
        return value

    @action
    def unconfigured_action(agent, value: int) -> int:
        """Unconfigured action.

        Args:
            value: Value to return.

        Returns:
            The value.
        """
        del agent
        return value

    manager = ActionManager(actions=[configured_action])

    with pytest.raises(ValueError, match="Unknown action name"):
        ActionManager(actions=["missing_action"])

    with pytest.raises(ValueError, match="Unknown action name"):
        manager.available_actions(actions="unconfigured_action")

    with pytest.raises(ValueError, match="Unknown action name"):
        manager.get_actions_schema(actions=["unconfigured_action"])

    with pytest.raises(ValueError, match="Unknown action name"):
        manager.available_actions(actions=unconfigured_action)

    with pytest.raises(ValueError, match="Unknown action name"):
        manager.get_actions_schema(actions=[unconfigured_action])


def test_default_actions_returns_exactly_wait():
    defaults = default_actions()

    assert defaults == (wait,)
    assert isinstance(defaults, tuple)


def test_wait_has_action_metadata_schema_and_is_explicitly_configurable():
    assert wait.__action_metadata__ == ActionMetadata(
        name="wait",
        description="Take no action for this turn.",
        parameters={},
        required=[],
        return_description="A confirmation that the agent waited.",
    )
    assert wait.__action_schema__ == {
        "name": "wait",
        "description": "Take no action for this turn.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "returns": "A confirmation that the agent waited.",
    }

    callable_manager = ActionManager(actions=[wait])
    named_manager = ActionManager(actions=["wait"])

    assert callable_manager.available_actions() == {"wait": wait}
    assert named_manager.available_actions() == {"wait": wait}
    assert callable_manager.get_actions_schema() == [wait.__action_schema__]
    assert named_manager.get_actions_schema(actions="wait") == [
        wait.__action_schema__,
    ]
