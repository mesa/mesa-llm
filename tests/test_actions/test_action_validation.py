from __future__ import annotations

import asyncio
import gc
import math
import threading
from contextlib import suppress
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import pytest

from mesa_llm.actions import (
    ActionChoice,
    ActionManager,
    ActionPostCommitError,
    action,
)
from mesa_llm.actions.action_decorator import _GLOBAL_ACTION_REGISTRY

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

    class MissingPayloadType:
        pass


@pytest.fixture(autouse=True)
def restore_global_action_registry():
    """Keep bare @action registrations local to each test."""
    original_registry = dict(_GLOBAL_ACTION_REGISTRY)
    yield
    _GLOBAL_ACTION_REGISTRY.clear()
    _GLOBAL_ACTION_REGISTRY.update(original_registry)


_FINAL_A10_DEFERRED_KINDS = ("generator", "async-generator")


class _FinalA10CleanupAbort(BaseException):
    pass


class _FinalA10UnformattableCleanupArgument:
    def __str__(self):
        raise RuntimeError("cleanup argument __str__ failed")

    def __repr__(self):
        raise RuntimeError("cleanup argument __repr__ failed")


class _FinalA10Iterator:
    def __init__(self):
        self.next_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.next_calls += 1
        raise StopIteration


class _FinalA10Result:
    pass


class _FinalA11CleanupAbort(BaseException):
    pass


class _FinalA11CustomAwaitable:
    def __init__(self, result):
        self.result = result
        self.await_calls = 0

    def __await__(self):
        self.await_calls += 1

        async def resolve():
            return self.result

        return resolve().__await__()


class _FinalA11CancelFailureFuture(asyncio.Future):
    def __init__(self, cleanup_error):
        super().__init__()
        self.cleanup_error = cleanup_error
        self.cancel_calls = 0

    def cancel(self, msg=None):
        del msg
        self.cancel_calls += 1
        raise self.cleanup_error


_FINAL_A10_COMPLETED_RESULT_FACTORIES = (
    pytest.param("execute", lambda: ["completed"], id="execute-list"),
    pytest.param("aexecute", lambda: ("completed",), id="aexecute-tuple"),
    pytest.param(
        "execute",
        lambda: {"status": "completed"},
        id="execute-dict",
    ),
    pytest.param(
        "aexecute",
        lambda: iter(["completed"]),
        id="aexecute-list-iterator",
    ),
    pytest.param("execute", _FinalA10Iterator, id="execute-custom-iterator"),
    pytest.param("aexecute", _FinalA10Result, id="aexecute-custom-result"),
)


def _make_final_a10_deferred_result(kind, state, cleanup_error=None):
    if kind == "generator":

        def deferred_result():
            state.body_calls += 1
            try:
                yield "deferred"
            finally:
                state.cleanup_calls += 1
                if cleanup_error is not None:
                    raise cleanup_error

    else:

        async def deferred_result():
            state.body_calls += 1
            try:
                yield "deferred"
            finally:
                state.cleanup_calls += 1
                if cleanup_error is not None:
                    raise cleanup_error

    return deferred_result()


def _assert_final_a10_deferred_result_closed(kind, result):
    frame = result.gi_frame if kind == "generator" else result.ag_frame
    assert frame is None


def _assert_final_a10_deferred_result_error(error, action_name):
    assert type(error) is TypeError
    message = str(error)
    assert action_name in message
    assert "generator" in message.casefold()


def _assert_final_a10_safe_cleanup_note(error, cleanup_error_type):
    notes = getattr(error, "__notes__", ())
    assert len(notes) == 1
    assert type(notes[0]) is str
    assert cleanup_error_type.__name__ in notes[0]
    assert any(word in notes[0].casefold() for word in ("cleanup", "close"))


def _assert_final_a11_nested_awaitable_error(error, action_name):
    assert type(error) is TypeError
    assert not isinstance(error, ActionPostCommitError)
    message = str(error).casefold()
    assert action_name.casefold() in message
    assert "one completed result" in message
    assert "nested awaitable" in message
    assert any(word in message for word in ("unsupported", "not supported"))


def test_action_choice_constructs_with_default_rationale():
    choice = ActionChoice(
        name="choose_destination",
        arguments={"destination": "library"},
    )

    assert choice.name == "choose_destination"
    assert choice.arguments == {"destination": "library"}
    assert choice.rationale is None

    reasoned_choice = ActionChoice(
        name="choose_destination",
        arguments={"destination": "library"},
        rationale="The library is nearby.",
    )

    assert reasoned_choice.rationale == "The library is nearby."


def test_validate_returns_choice_identifying_configured_action():
    @action
    def record_message(agent, message: str) -> str:
        """Record a message.

        Args:
            message: Message to record.

        Returns:
            Recorded message.
        """
        del agent
        return message

    agent = SimpleNamespace()
    manager = ActionManager(actions=[record_message])
    choice = ActionChoice(
        name="record_message",
        arguments={"message": "hello"},
        rationale="Share a greeting.",
    )

    validated = manager.validate(agent, choice)

    assert validated.name == "record_message"
    assert validated.arguments == {"message": "hello"}
    assert manager.available_actions()[validated.name] is record_message


def test_validate_unknown_action_name_fails_fast():
    @action
    def configured_action(agent) -> str:
        """Configured action.

        Returns:
            Confirmation.
        """
        del agent
        return "configured"

    manager = ActionManager(actions=[configured_action])

    with pytest.raises(ValueError, match="Unknown action name"):
        manager.validate(
            SimpleNamespace(),
            ActionChoice(name="missing_action", arguments={}),
        )


def test_validate_rejects_action_outside_explicit_narrowing():
    @action
    def selected_action(agent) -> str:
        """Selected action.

        Returns:
            Selection confirmation.
        """
        del agent
        return "selected"

    @action
    def configured_but_narrowed_out_action(agent) -> str:
        """Configured but narrowed out action.

        Returns:
            Narrowing confirmation.
        """
        del agent
        return "narrowed out"

    manager = ActionManager(
        actions=[selected_action, configured_but_narrowed_out_action],
    )
    agent = SimpleNamespace()

    validated = manager.validate(
        agent,
        ActionChoice(name="selected_action", arguments={}),
        actions=[selected_action],
    )

    assert validated.name == "selected_action"

    with pytest.raises(ValueError, match="Unknown action name"):
        manager.validate(
            agent,
            ActionChoice(name="configured_but_narrowed_out_action", arguments={}),
            actions=[selected_action],
        )


def test_validate_rejects_missing_required_arguments():
    @action
    def move_to(agent, destination: str) -> str:
        """Move to a destination.

        Args:
            destination: Destination name.

        Returns:
            Move confirmation.
        """
        del agent
        return destination

    manager = ActionManager(actions=[move_to])

    with pytest.raises(ValueError, match="Missing required argument"):
        manager.validate(
            SimpleNamespace(),
            ActionChoice(name="move_to", arguments={}),
        )


def test_validate_and_execute_allow_omitted_default_arguments():
    @action
    def repeat_message(agent, message: str, times: int = 2, suffix: str = "!") -> str:
        """Repeat a message.

        Args:
            message: Message to repeat.
            times: Number of repetitions.
            suffix: Suffix appended to each repetition.

        Returns:
            Repeated message.
        """
        agent.calls.append((message, times, suffix))
        return (message + suffix) * times

    agent = SimpleNamespace(calls=[])
    manager = ActionManager(actions=[repeat_message])
    choice = ActionChoice(name="repeat_message", arguments={"message": "go"})

    validated = manager.validate(agent, choice)
    result = manager.execute(agent, choice)

    assert validated.arguments == {"message": "go"}
    assert result == "go!go!"
    assert agent.calls == [("go", 2, "!")]


@pytest.mark.parametrize(
    ("annotation", "default_value"),
    [
        pytest.param(int, 4, id="integer"),
        pytest.param(float, 2.5, id="finite-float"),
        pytest.param(float, 1, id="integral-value-for-float"),
        pytest.param(str, "ready", id="string"),
        pytest.param(bool, True, id="boolean"),
        pytest.param(int | None, None, id="optional-none"),
        pytest.param(list[int], [1, 2], id="list"),
        pytest.param(tuple[int, int], (1, 2), id="fixed-tuple"),
        pytest.param(dict[str, int], {"one": 1}, id="dictionary"),
        pytest.param(Literal["safe", "fast"], "safe", id="literal"),
    ],
)
def test_execute_uses_annotation_compatible_python_default(
    annotation,
    default_value,
):
    def use_default(agent, value) -> object:
        """Use a configured default value.

        Args:
            value: Value to use.

        Returns:
            The configured value.
        """
        agent.values.append(value)
        return value

    use_default.__annotations__["value"] = annotation
    use_default.__defaults__ = (default_value,)
    manager = ActionManager()
    use_default = action(action_manager=manager)(use_default)
    agent = SimpleNamespace(values=[])
    choice = ActionChoice(name="use_default", arguments={})

    validated = manager.validate(agent, choice)
    result = manager.execute(agent, choice)

    assert manager.actions == {"use_default": use_default}
    assert validated.arguments == {}
    assert result is default_value
    assert len(agent.values) == 1
    assert agent.values[0] is default_value


def test_validate_rejects_unexpected_extra_arguments():
    @action
    def speak(agent, message: str) -> str:
        """Speak a message.

        Args:
            message: Message to speak.

        Returns:
            Spoken message.
        """
        del agent
        return message

    manager = ActionManager(actions=[speak])

    with pytest.raises(ValueError, match="Unexpected argument"):
        manager.validate(
            SimpleNamespace(),
            ActionChoice(
                name="speak",
                arguments={"message": "hello", "volume": "loud"},
            ),
        )


def test_validate_rejects_llm_supplied_agent_argument_before_mutation():
    @action
    def mark_called(agent) -> str:
        """Mark the live agent as called.

        Returns:
            Mutation confirmation.
        """
        agent.called = True
        return "called"

    agent = SimpleNamespace(called=False)
    manager = ActionManager(actions=[mark_called])

    with pytest.raises(ValueError, match="framework-injected"):
        manager.validate(
            agent,
            ActionChoice(
                name="mark_called",
                arguments={"agent": SimpleNamespace(called=True)},
            ),
        )

    assert agent.called is False


def test_execute_rejects_postponed_list_annotation_non_list_string_before_mutation():
    @action
    def notify_agents(agent: LLMAgent, listener_agent_ids: list[int]) -> str:
        """Notify agents by id.

        Args:
            listener_agent_ids: Agent ids to notify.

        Returns:
            Notification confirmation.
        """
        agent.notified_ids.extend(listener_agent_ids)
        return "notified"

    agent = SimpleNamespace(notified_ids=[])
    manager = ActionManager(actions=[notify_agents])

    with pytest.raises(
        ValueError,
        match=r"Invalid argument type.*listener_agent_ids.*list\[int\]",
    ):
        manager.execute(
            agent,
            ActionChoice(
                name="notify_agents",
                arguments={"listener_agent_ids": "not a list"},
            ),
        )

    assert agent.notified_ids == []


def test_validate_resolves_postponed_optional_int_annotation():
    @action
    def set_cooldown(agent: LLMAgent, cooldown: int | None) -> int | None:
        """Set an optional cooldown.

        Args:
            cooldown: Optional cooldown duration.

        Returns:
            The normalized cooldown.
        """
        del agent
        return cooldown

    agent = SimpleNamespace()
    manager = ActionManager(actions=[set_cooldown])

    none_choice = manager.validate(
        agent,
        ActionChoice(name="set_cooldown", arguments={"cooldown": None}),
    )
    int_choice = manager.validate(
        agent,
        ActionChoice(name="set_cooldown", arguments={"cooldown": "4"}),
    )

    assert none_choice.arguments == {"cooldown": None}
    assert int_choice.arguments == {"cooldown": 4}
    assert isinstance(int_choice.arguments["cooldown"], int)

    with pytest.raises(
        ValueError,
        match=r"Invalid argument type.*cooldown.*int.*None",
    ):
        manager.validate(
            agent,
            ActionChoice(name="set_cooldown", arguments={"cooldown": "later"}),
        )


def test_execute_accepts_declared_literal_and_rejects_undeclared_before_mutation():
    @action
    def set_mode(agent, mode: Literal["a", "b"]) -> str:
        """Set the agent mode.

        Args:
            mode: Mode to apply.

        Returns:
            Applied mode.
        """
        agent.mode = mode
        return mode

    agent = SimpleNamespace(mode=None)
    manager = ActionManager(actions=[set_mode])

    result = manager.execute(
        agent,
        ActionChoice(name="set_mode", arguments={"mode": "b"}),
    )

    assert result == "b"
    assert agent.mode == "b"

    with pytest.raises(ValueError, match=r"Invalid argument type.*mode.*one of"):
        manager.execute(
            agent,
            ActionChoice(name="set_mode", arguments={"mode": "c"}),
        )

    assert agent.mode == "b"


def test_execute_numeric_literal_validation_normalizes_schema_equivalent_numbers():
    @action
    def set_integer(agent, value: Literal[1]) -> int:
        """Set an integer literal.

        Args:
            value: Integer literal to apply.

        Returns:
            Applied integer.
        """
        agent.values.append(value)
        return value

    @action
    def set_float(agent, value: Literal[1.0]) -> float:
        """Set a float literal.

        Args:
            value: Float literal to apply.

        Returns:
            Applied float.
        """
        agent.values.append(value)
        return value

    @action
    def set_boolean(agent, value: Literal[True]) -> bool:
        """Set a Boolean literal.

        Args:
            value: Boolean literal to apply.

        Returns:
            Applied Boolean.
        """
        agent.values.append(value)
        return value

    agent = SimpleNamespace(values=[])
    manager = ActionManager(actions=[set_integer, set_float, set_boolean])

    integer_result = manager.execute(
        agent,
        ActionChoice(name="set_integer", arguments={"value": 1.0}),
    )
    assert integer_result == 1
    assert type(integer_result) is int
    assert agent.values == [1]

    float_result = manager.execute(
        agent,
        ActionChoice(name="set_float", arguments={"value": 1}),
    )
    assert float_result == 1.0
    assert type(float_result) is float
    assert agent.values == [1, 1.0]

    with pytest.raises(ValueError, match=r"Invalid argument type.*value.*one of"):
        manager.execute(
            agent,
            ActionChoice(name="set_integer", arguments={"value": True}),
        )
    assert agent.values == [1, 1.0]

    with pytest.raises(ValueError, match=r"Invalid argument type.*value.*one of"):
        manager.execute(
            agent,
            ActionChoice(name="set_float", arguments={"value": True}),
        )
    assert agent.values == [1, 1.0]

    assert (
        manager.execute(
            agent,
            ActionChoice(name="set_boolean", arguments={"value": True}),
        )
        is True
    )
    assert agent.values == [1, 1.0, True]

    with pytest.raises(ValueError, match=r"Invalid argument type.*value.*one of"):
        manager.execute(
            agent,
            ActionChoice(name="set_boolean", arguments={"value": 1}),
        )
    assert agent.values == [1, 1.0, True]


def test_execute_nonintegral_and_nonnumeric_literals_remain_exact_before_mutation():
    @action
    def set_value(agent, value: Literal[1.5, "ready"]) -> float | str:
        """Set an exact literal value.

        Args:
            value: Exact value to apply.

        Returns:
            Applied value.
        """
        agent.values.append(value)
        return value

    agent = SimpleNamespace(values=[])
    manager = ActionManager(actions=[set_value])

    assert (
        manager.execute(
            agent,
            ActionChoice(name="set_value", arguments={"value": 1.5}),
        )
        == 1.5
    )
    assert (
        manager.execute(
            agent,
            ActionChoice(name="set_value", arguments={"value": "ready"}),
        )
        == "ready"
    )
    assert agent.values == [1.5, "ready"]

    with pytest.raises(ValueError, match=r"Invalid argument type.*value.*one of"):
        manager.execute(
            agent,
            ActionChoice(name="set_value", arguments={"value": 1}),
        )

    assert agent.values == [1.5, "ready"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (4, 4),
        ("ready", "ready"),
        (None, None),
    ],
)
def test_validate_accepts_every_nullable_union_member(value, expected):
    @action
    def set_value(agent, selected_value: int | str | None) -> str:
        """Set an optional typed value.

        Args:
            selected_value: Value to apply.

        Returns:
            Value confirmation.
        """
        agent.value = selected_value
        return "set"

    agent = SimpleNamespace(value="unchanged")
    manager = ActionManager(actions=[set_value])

    validated = manager.validate(
        agent,
        ActionChoice(
            name="set_value",
            arguments={"selected_value": value},
        ),
    )

    assert validated.arguments == {"selected_value": expected}
    assert agent.value == "unchanged"


def test_execute_annotated_action_parameter_uses_consistent_schema_and_runtime_type():
    @action
    def set_priority(agent, priority: Annotated[int, "priority"]) -> int:
        """Set a typed priority.

        Args:
            priority: Priority to apply.

        Returns:
            Applied priority.
        """
        agent.priorities.append(priority)
        return priority

    schema = set_priority.__action_schema__["parameters"]["properties"]["priority"]
    agent = SimpleNamespace(priorities=[])
    manager = ActionManager(actions=[set_priority])

    result = manager.execute(
        agent,
        ActionChoice(name="set_priority", arguments={"priority": "2"}),
    )

    assert schema == {"type": "integer", "description": "Priority to apply."}
    assert result == 2
    assert type(result) is int
    assert agent.priorities == [2]


def test_action_decorator_unresolved_annotation_fails_before_manager_registration():
    mutations = []

    def apply_payload(agent: LLMAgent, payload: MissingPayloadType) -> str:
        """Apply a typed payload.

        Args:
            payload: Payload to apply.

        Returns:
            Payload confirmation.
        """
        mutations.append((agent, payload))
        return "applied"

    manager = ActionManager()

    with pytest.raises(
        ValueError,
        match=r"Could not resolve annotation.*payload.*MissingPayloadType",
    ):
        action(action_manager=manager)(apply_payload)

    assert manager.actions == {}
    assert not hasattr(apply_payload, "__action_schema__")
    assert mutations == []


def test_action_decorator_with_missing_annotation_fails_before_manager_registration():
    mutations = []

    def apply_payload(agent, payload) -> str:
        """Apply an untyped payload.

        Args:
            payload: Payload to apply.

        Returns:
            Payload confirmation.
        """
        mutations.append((agent, payload))
        return "applied"

    manager = ActionManager()

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(action_manager=manager)(apply_payload)

    message = str(exc_info.value)
    assert "apply_payload" in message
    assert "payload" in message
    assert "annotation" in message.lower()
    assert manager.actions == {}
    assert not hasattr(apply_payload, "__action_schema__")
    assert mutations == []


def test_action_decorator_with_unsupported_annotation_fails_before_manager_registration():
    mutations = []

    def apply_payload(agent, payload: object) -> str:
        """Apply an unsupported payload.

        Args:
            payload: Payload to apply.

        Returns:
            Payload confirmation.
        """
        mutations.append((agent, payload))
        return "applied"

    manager = ActionManager()

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(action_manager=manager)(apply_payload)

    message = str(exc_info.value)
    assert "apply_payload" in message
    assert "payload" in message
    assert "annotation" in message.lower()
    assert manager.actions == {}
    assert not hasattr(apply_payload, "__action_schema__")
    assert mutations == []


def test_action_decorator_with_set_annotation_fails_before_manager_registration():
    mutations = []

    def apply_values(agent, values: set[int]) -> str:
        """Apply a set of values.

        Args:
            values: Values to apply.

        Returns:
            Application confirmation.
        """
        mutations.append((agent, values))
        return "applied"

    manager = ActionManager()

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(action_manager=manager)(apply_values)

    message = str(exc_info.value)
    assert "apply_values" in message
    assert "values" in message
    assert "set" in message.lower()
    assert manager.actions == {}
    assert not hasattr(apply_values, "__action_metadata__")
    assert not hasattr(apply_values, "__action_schema__")
    assert mutations == []


def test_action_decorator_with_tuple_set_fails_before_execution_or_registration():
    mutations = []

    def apply_values(agent, values: set[tuple[int, int]]) -> set[tuple[int, int]]:
        """Apply a set of coordinate values.

        Args:
            values: Coordinate values to apply.

        Returns:
            Applied coordinate values.
        """
        mutations.append((agent, values))
        return values

    manager = ActionManager()

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(action_manager=manager)(apply_values)

    message = str(exc_info.value)
    assert "apply_values" in message
    assert "values" in message
    assert "set" in message.lower()
    assert manager.actions == {}
    assert not hasattr(apply_values, "__action_metadata__")
    assert not hasattr(apply_values, "__action_schema__")
    assert mutations == []


def test_validate_does_not_execute_action_or_mutate_state():
    @action
    def mutating_action(agent, amount: int) -> str:
        """Mutate state if executed.

        Args:
            amount: Amount to add.

        Returns:
            Mutation confirmation.
        """
        agent.called = True
        agent.model.counter += amount
        return "mutated"

    model = SimpleNamespace(counter=0)
    agent = SimpleNamespace(called=False, model=model)
    manager = ActionManager(actions=[mutating_action])

    validated = manager.validate(
        agent,
        ActionChoice(name="mutating_action", arguments={"amount": 5}),
    )

    assert validated.name == "mutating_action"
    assert agent.called is False
    assert model.counter == 0


def test_final_a10_execute_rejects_and_closes_async_generator_in_sync_context():
    kind = "async-generator"
    state = SimpleNamespace(
        producer_calls=0,
        body_calls=0,
        cleanup_calls=0,
        result=None,
    )
    manager = ActionManager()

    @action(action_manager=manager)
    def final_a10_sync_deferred(agent) -> object:
        """Return a deferred result from an ordinary synchronous callable."""
        del agent
        state.producer_calls += 1
        state.result = _make_final_a10_deferred_result(kind, state)
        return state.result

    try:
        with pytest.raises(TypeError) as exc_info:
            manager.execute(
                SimpleNamespace(),
                ActionChoice(name="final_a10_sync_deferred", arguments={}),
            )

        _assert_final_a10_deferred_result_error(
            exc_info.value,
            "final_a10_sync_deferred",
        )
        assert state.producer_calls == 1
        assert state.body_calls == 0
        assert state.cleanup_calls == 0
        _assert_final_a10_deferred_result_closed(kind, state.result)
    finally:
        if state.result is not None:
            if kind == "generator":
                with suppress(BaseException):
                    state.result.close()
            else:
                with suppress(BaseException):
                    asyncio.run(state.result.aclose())


@pytest.mark.asyncio
async def test_final_a10_execute_rejects_async_generator_in_active_loop_without_offloading(
    monkeypatch,
):
    state = SimpleNamespace(
        producer_calls=0,
        body_calls=0,
        cleanup_calls=0,
        result=None,
    )
    manager = ActionManager()

    @action(action_manager=manager)
    def final_a10_active_loop_async_generator(agent) -> object:
        """Return an async generator while an event loop is already running."""
        del agent
        state.producer_calls += 1
        state.result = _make_final_a10_deferred_result("async-generator", state)
        return state.result

    forbidden_calls = []

    def forbidden(mechanism):
        def fail(*args, **kwargs):
            del args, kwargs
            forbidden_calls.append(mechanism)
            raise AssertionError(f"unexpected deferred-result cleanup via {mechanism}")

        return fail

    loop = asyncio.get_running_loop()
    tasks_before = asyncio.all_tasks(loop)
    original_task_factory = loop.get_task_factory()
    loop.set_task_factory(forbidden("event-loop task"))
    monkeypatch.setattr(asyncio, "run", forbidden("nested asyncio.run"))
    monkeypatch.setattr(asyncio, "Runner", forbidden("nested asyncio.Runner"))
    monkeypatch.setattr(asyncio, "new_event_loop", forbidden("nested event loop"))
    monkeypatch.setattr(threading.Thread, "start", forbidden("thread"))

    try:
        with pytest.raises(TypeError) as exc_info:
            manager.execute(
                SimpleNamespace(),
                ActionChoice(
                    name="final_a10_active_loop_async_generator",
                    arguments={},
                ),
            )

        _assert_final_a10_deferred_result_error(
            exc_info.value,
            "final_a10_active_loop_async_generator",
        )
        assert forbidden_calls == []
        assert asyncio.all_tasks(loop) == tasks_before
        assert state.producer_calls == 1
        assert state.body_calls == 0
        assert state.cleanup_calls == 0
        assert state.result.ag_frame is not None
        notes = getattr(exc_info.value, "__notes__", ())
        assert len(notes) == 1
        assert "not closed" in notes[0]
        assert "event loop" in notes[0]
    finally:
        loop.set_task_factory(original_task_factory)
        if state.result is not None:
            with suppress(BaseException):
                await state.result.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _FINAL_A10_DEFERRED_KINDS)
@pytest.mark.parametrize(
    "producer",
    [
        pytest.param("sync", id="sync-callable"),
        pytest.param("async", id="async-function-awaited"),
        pytest.param("sync-coroutine", id="sync-callable-coroutine-awaited"),
    ],
)
async def test_final_a10_aexecute_rejects_deferred_results_at_each_boundary(
    kind,
    producer,
):
    state = SimpleNamespace(
        producer_calls=0,
        await_completions=0,
        body_calls=0,
        cleanup_calls=0,
        result=None,
    )
    manager = ActionManager()

    def completed_deferred_result():
        state.result = _make_final_a10_deferred_result(kind, state)
        return state.result

    if producer == "sync":

        def producer_action(agent) -> object:
            """Return a deferred result from a synchronous action."""
            del agent
            state.producer_calls += 1
            return completed_deferred_result()

    elif producer == "async":

        async def producer_action(agent) -> object:
            """Return a deferred result after awaiting an async action."""
            del agent
            state.producer_calls += 1
            await asyncio.sleep(0)
            state.await_completions += 1
            return completed_deferred_result()

    else:

        def producer_action(agent) -> object:
            """Return a coroutine which resolves to a deferred result."""
            del agent
            state.producer_calls += 1

            async def complete_production():
                await asyncio.sleep(0)
                state.await_completions += 1
                return completed_deferred_result()

            return complete_production()

    action_name = f"final_a10_{producer.replace('-', '_')}_{kind.replace('-', '_')}"
    producer_action.__name__ = action_name
    producer_action = action(action_manager=manager)(producer_action)

    try:
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(
                SimpleNamespace(),
                ActionChoice(name=action_name, arguments={}),
            )

        _assert_final_a10_deferred_result_error(exc_info.value, action_name)
        assert manager.actions[action_name] is producer_action
        assert state.producer_calls == 1
        assert state.await_completions == (producer != "sync")
        assert state.body_calls == 0
        assert state.cleanup_calls == 0
        _assert_final_a10_deferred_result_closed(kind, state.result)
    finally:
        if state.result is not None:
            if kind == "generator":
                with suppress(BaseException):
                    state.result.close()
            else:
                with suppress(BaseException):
                    await state.result.aclose()


@pytest.mark.parametrize("kind", _FINAL_A10_DEFERRED_KINDS)
@pytest.mark.parametrize("cleanup_failure", ["exception", "base-exception"])
def test_final_a10_execute_started_deferred_cleanup_failure_contract(
    kind,
    cleanup_failure,
):
    cleanup_error = (
        RuntimeError(f"{kind} cleanup failed")
        if cleanup_failure == "exception"
        else _FinalA10CleanupAbort(f"{kind} cleanup aborted")
    )
    state = SimpleNamespace(
        producer_calls=0,
        body_calls=0,
        cleanup_calls=0,
        result=None,
    )
    state.result = _make_final_a10_deferred_result(kind, state, cleanup_error)
    if kind == "generator":
        first_value = next(state.result)
    else:
        first_step = anext(state.result)
        with pytest.raises(StopIteration) as exc_info:
            first_step.send(None)
        first_value = exc_info.value.value
    assert first_value == "deferred"

    manager = ActionManager()

    @action(action_manager=manager)
    def final_a10_sync_started_deferred(agent) -> object:
        """Return a started deferred result to synchronous execution."""
        del agent
        state.producer_calls += 1
        return state.result

    choice = ActionChoice(name="final_a10_sync_started_deferred", arguments={})
    expected_error = (
        TypeError if cleanup_failure == "exception" else _FinalA10CleanupAbort
    )

    try:
        with pytest.raises(expected_error) as exc_info:
            manager.execute(SimpleNamespace(), choice)

        if cleanup_failure == "exception":
            _assert_final_a10_deferred_result_error(
                exc_info.value,
                "final_a10_sync_started_deferred",
            )
            notes = getattr(exc_info.value, "__notes__", ())
            assert len(notes) == 1
            assert str(cleanup_error) in notes[0]
            assert any(word in notes[0].casefold() for word in ("cleanup", "close"))
        else:
            assert exc_info.value is cleanup_error

        assert state.producer_calls == 1
        assert state.body_calls == 1
        assert state.cleanup_calls == 1
        _assert_final_a10_deferred_result_closed(kind, state.result)
    finally:
        if kind == "generator":
            with suppress(BaseException):
                state.result.close()
        else:
            with suppress(BaseException):
                asyncio.run(state.result.aclose())


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _FINAL_A10_DEFERRED_KINDS)
@pytest.mark.parametrize("cleanup_failure", ["exception", "base-exception"])
async def test_final_a10_deferred_result_cleanup_failure_contract(
    kind,
    cleanup_failure,
):
    cleanup_error = (
        RuntimeError(f"{kind} cleanup failed")
        if cleanup_failure == "exception"
        else _FinalA10CleanupAbort(f"{kind} cleanup aborted")
    )
    state = SimpleNamespace(
        producer_calls=0,
        body_calls=0,
        cleanup_calls=0,
        result=None,
    )
    state.result = _make_final_a10_deferred_result(kind, state, cleanup_error)
    if kind == "generator":
        assert next(state.result) == "deferred"
    else:
        assert await anext(state.result) == "deferred"

    manager = ActionManager()

    @action(action_manager=manager)
    def final_a10_started_deferred(agent) -> object:
        """Return an already-started deferred result for cleanup observation."""
        del agent
        state.producer_calls += 1
        return state.result

    choice = ActionChoice(name="final_a10_started_deferred", arguments={})

    try:
        if cleanup_failure == "exception":
            with pytest.raises(TypeError) as exc_info:
                await manager.aexecute(SimpleNamespace(), choice)

            _assert_final_a10_deferred_result_error(
                exc_info.value,
                "final_a10_started_deferred",
            )
            notes = getattr(exc_info.value, "__notes__", ())
            assert len(notes) == 1
            assert str(cleanup_error) in notes[0]
            assert any(word in notes[0].casefold() for word in ("cleanup", "close"))
        else:
            with pytest.raises(_FinalA10CleanupAbort) as exc_info:
                await manager.aexecute(SimpleNamespace(), choice)

            assert exc_info.value is cleanup_error

        assert state.producer_calls == 1
        assert state.body_calls == 1
        assert state.cleanup_calls == 1
        _assert_final_a10_deferred_result_closed(kind, state.result)
    finally:
        if kind == "generator":
            with suppress(BaseException):
                state.result.close()
        else:
            with suppress(BaseException):
                await state.result.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "entrypoint"),
    [
        pytest.param("generator", "execute", id="execute-generator"),
        pytest.param(
            "async-generator",
            "aexecute",
            id="aexecute-async-generator",
        ),
    ],
)
async def test_final_a10_unformattable_cleanup_exception_preserves_primary_error(
    kind,
    entrypoint,
):
    cleanup_argument = _FinalA10UnformattableCleanupArgument()
    cleanup_error = RuntimeError(cleanup_argument)
    with pytest.raises(RuntimeError, match="__str__ failed"):
        str(cleanup_error)
    with pytest.raises(RuntimeError, match="__repr__ failed"):
        repr(cleanup_error)

    state = SimpleNamespace(
        producer_calls=0,
        body_calls=0,
        cleanup_calls=0,
        result=None,
    )
    state.result = _make_final_a10_deferred_result(kind, state, cleanup_error)
    if kind == "generator":
        assert next(state.result) == "deferred"
    else:
        assert await anext(state.result) == "deferred"

    manager = ActionManager()

    def producer_action(agent) -> object:
        """Return a started deferred result with hostile cleanup diagnostics."""
        del agent
        state.producer_calls += 1
        return state.result

    action_name = f"final_a10_{entrypoint}_unformattable_cleanup"
    producer_action.__name__ = action_name
    producer_action = action(action_manager=manager)(producer_action)
    choice = ActionChoice(name=action_name, arguments={})

    try:
        with pytest.raises(TypeError) as exc_info:
            if entrypoint == "execute":
                manager.execute(SimpleNamespace(), choice)
            else:
                await manager.aexecute(SimpleNamespace(), choice)

        _assert_final_a10_deferred_result_error(exc_info.value, action_name)
        assert not isinstance(exc_info.value, ActionPostCommitError)
        _assert_final_a10_safe_cleanup_note(exc_info.value, type(cleanup_error))
        assert manager.actions[action_name] is producer_action
        assert state.producer_calls == 1
        assert state.body_calls == 1
        assert state.cleanup_calls == 1
        _assert_final_a10_deferred_result_closed(kind, state.result)
    finally:
        if kind == "generator":
            with suppress(BaseException):
                state.result.close()
        else:
            with suppress(BaseException):
                await state.result.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("execution", "result_factory"),
    _FINAL_A10_COMPLETED_RESULT_FACTORIES,
)
async def test_final_a10_manager_accepts_non_generator_completed_results(
    execution,
    result_factory,
):
    expected = result_factory()
    manager = ActionManager()

    @action(action_manager=manager)
    def final_a10_completed_control(agent) -> object:
        """Return a completed non-generator control result."""
        del agent
        return expected

    choice = ActionChoice(name="final_a10_completed_control", arguments={})
    if execution == "execute":
        returned = manager.execute(SimpleNamespace(), choice)
    else:
        returned = await manager.aexecute(SimpleNamespace(), choice)

    assert returned is expected
    if isinstance(expected, _FinalA10Iterator):
        assert expected.next_calls == 0
    elif type(expected).__name__ == "list_iterator":
        assert expected.__length_hint__() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "producer",
    [
        pytest.param("async", id="async-action"),
        pytest.param("sync-coroutine", id="sync-callable-coroutine"),
    ],
)
async def test_final_a11_aexecute_rejects_and_closes_nested_native_coroutine(
    producer,
    recwarn,
):
    state = SimpleNamespace(
        producer_calls=0,
        await_completions=0,
        outer_mutations=0,
        inner_mutations=0,
        nested=None,
    )
    manager = ActionManager()

    def make_nested_coroutine(agent):
        async def mutate_inner():
            agent.inner_mutations += 1
            return "nested completed"

        agent.nested = mutate_inner()
        return agent.nested

    if producer == "async":

        async def producer_action(agent) -> object:
            """Return a nested coroutine after the supported async boundary."""
            agent.producer_calls += 1
            await asyncio.sleep(0)
            agent.await_completions += 1
            agent.outer_mutations += 1
            return make_nested_coroutine(agent)

    else:

        def producer_action(agent) -> object:
            """Return one coroutine which resolves to a nested coroutine."""
            agent.producer_calls += 1

            async def complete_outer():
                await asyncio.sleep(0)
                agent.await_completions += 1
                agent.outer_mutations += 1
                return make_nested_coroutine(agent)

            return complete_outer()

    action_name = f"final_a11_{producer.replace('-', '_')}_native"
    producer_action.__name__ = action_name
    producer_action = action(action_manager=manager)(producer_action)

    try:
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(
                state,
                ActionChoice(name=action_name, arguments={}),
            )

        _assert_final_a11_nested_awaitable_error(exc_info.value, action_name)
        assert manager.actions[action_name] is producer_action
        assert state.producer_calls == 1
        assert state.await_completions == 1
        assert state.outer_mutations == 1
        assert state.inner_mutations == 0
        assert state.nested.cr_frame is None
        await asyncio.sleep(0)
        assert state.inner_mutations == 0
    finally:
        if state.nested is not None and state.nested.cr_frame is not None:
            state.nested.close()

    gc.collect()
    assert not [
        warning for warning in recwarn if "was never awaited" in str(warning.message)
    ]


@pytest.mark.asyncio
async def test_final_a11_aexecute_cancels_nested_future():
    state = SimpleNamespace(producer_calls=0, nested=None)
    manager = ActionManager()

    @action(action_manager=manager)
    def final_a11_nested_future(agent) -> object:
        """Return a Future which resolves to another Future."""
        agent.producer_calls += 1
        loop = asyncio.get_running_loop()
        outer = loop.create_future()
        agent.nested = loop.create_future()
        outer.set_result(agent.nested)
        return outer

    with pytest.raises(TypeError) as exc_info:
        await manager.aexecute(
            state,
            ActionChoice(name="final_a11_nested_future", arguments={}),
        )

    _assert_final_a11_nested_awaitable_error(
        exc_info.value,
        "final_a11_nested_future",
    )
    assert state.producer_calls == 1
    assert state.nested.cancelled()


@pytest.mark.asyncio
async def test_final_a11_aexecute_cancels_nested_task_without_claiming_rollback():
    release_mutation = asyncio.Event()
    nested_started = asyncio.Event()
    state = SimpleNamespace(
        producer_calls=0,
        mutations=0,
        outer=None,
        nested=None,
    )
    manager = ActionManager()

    @action(action_manager=manager)
    def final_a11_nested_task(agent) -> object:
        """Return a Task which resolves to an independently scheduled Task."""
        agent.producer_calls += 1

        async def mutate_after_release():
            nested_started.set()
            await release_mutation.wait()
            agent.mutations += 1

        async def resolve_to_nested():
            await asyncio.sleep(0)
            return agent.nested

        agent.nested = asyncio.create_task(mutate_after_release())
        agent.outer = asyncio.create_task(resolve_to_nested())
        return agent.outer

    try:
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(
                state,
                ActionChoice(name="final_a11_nested_task", arguments={}),
            )

        _assert_final_a11_nested_awaitable_error(
            exc_info.value,
            "final_a11_nested_task",
        )
        assert state.producer_calls == 1
        assert nested_started.is_set()
        assert state.nested.cancelling() > 0
        with pytest.raises(asyncio.CancelledError):
            await state.nested
        assert state.nested.cancelled()
        release_mutation.set()
        await asyncio.sleep(0)
        assert state.mutations == 0
    finally:
        release_mutation.set()
        for task in (state.outer, state.nested):
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                with suppress(asyncio.CancelledError):
                    await task


@pytest.mark.asyncio
async def test_final_a11_aexecute_rejects_current_task_without_self_cancellation():
    state = SimpleNamespace(producer_calls=0, returned_task=None)
    manager = ActionManager()
    execution_task = asyncio.current_task()
    assert execution_task is not None
    cancelling_before = execution_task.cancelling()

    @action(action_manager=manager)
    async def final_a11_current_task(agent) -> object:
        """Return the Task currently executing ActionManager.aexecute."""
        agent.producer_calls += 1
        agent.returned_task = asyncio.current_task()
        return agent.returned_task

    with pytest.raises(TypeError) as exc_info:
        await manager.aexecute(
            state,
            ActionChoice(name="final_a11_current_task", arguments={}),
        )

    error = exc_info.value
    _assert_final_a11_nested_awaitable_error(error, "final_a11_current_task")
    assert state.producer_calls == 1
    assert state.returned_task is execution_task
    assert execution_task.cancelling() == cancelling_before
    notes = getattr(error, "__notes__", ())
    assert len(notes) == 1
    note = notes[0].casefold()
    assert "cancel" in note
    assert any(word in note for word in ("current", "self"))
    assert any(word in note for word in ("unsafe", "skipped", "not"))

    await asyncio.sleep(0)
    assert execution_task.cancelling() == cancelling_before


@pytest.mark.asyncio
async def test_final_a11_aexecute_rejects_custom_nested_awaitable_without_driving_it():
    nested = _FinalA11CustomAwaitable("must not execute")
    outer = _FinalA11CustomAwaitable(nested)
    state = SimpleNamespace(producer_calls=0)
    manager = ActionManager()

    @action(action_manager=manager)
    def final_a11_custom_awaitable(agent) -> object:
        """Return a custom awaitable which resolves to another awaitable."""
        agent.producer_calls += 1
        return outer

    with pytest.raises(TypeError) as exc_info:
        await manager.aexecute(
            state,
            ActionChoice(name="final_a11_custom_awaitable", arguments={}),
        )

    error = exc_info.value
    _assert_final_a11_nested_awaitable_error(
        error,
        "final_a11_custom_awaitable",
    )
    assert state.producer_calls == 1
    assert outer.await_calls == 1
    assert nested.await_calls == 0
    notes = getattr(error, "__notes__", ())
    assert len(notes) == 1
    assert "cleanup" in notes[0].casefold()
    assert any(word in notes[0].casefold() for word in ("cannot", "no safe", "not"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cleanup_failure",
    [
        pytest.param("exception", id="cleanup-exception"),
        pytest.param("base-exception", id="cleanup-base-exception"),
    ],
)
async def test_final_a11_nested_future_cleanup_failure_contract(cleanup_failure):
    cleanup_error = (
        RuntimeError("nested Future cancellation failed")
        if cleanup_failure == "exception"
        else _FinalA11CleanupAbort("nested Future cancellation aborted")
    )
    state = SimpleNamespace(producer_calls=0, nested=None)
    manager = ActionManager()

    @action(action_manager=manager)
    async def final_a11_cancel_failure(agent) -> object:
        """Return a nested Future whose cancellation raises."""
        agent.producer_calls += 1
        agent.nested = _FinalA11CancelFailureFuture(cleanup_error)
        return agent.nested

    try:
        if cleanup_failure == "exception":
            with pytest.raises(TypeError) as exc_info:
                await manager.aexecute(
                    state,
                    ActionChoice(name="final_a11_cancel_failure", arguments={}),
                )

            error = exc_info.value
            _assert_final_a11_nested_awaitable_error(
                error,
                "final_a11_cancel_failure",
            )
            notes = getattr(error, "__notes__", ())
            assert len(notes) == 1
            assert type(cleanup_error).__name__ in notes[0]
            assert str(cleanup_error) in notes[0]
            assert any(word in notes[0].casefold() for word in ("cleanup", "cancel"))
        else:
            with pytest.raises(_FinalA11CleanupAbort) as exc_info:
                await manager.aexecute(
                    state,
                    ActionChoice(name="final_a11_cancel_failure", arguments={}),
                )

            assert exc_info.value is cleanup_error

        assert state.producer_calls == 1
        assert state.nested.cancel_calls == 1
    finally:
        if state.nested is not None and not state.nested.done():
            asyncio.Future.cancel(state.nested)


@pytest.mark.asyncio
async def test_final_a11_nested_coroutine_cleanup_does_not_schedule_or_offload(
    monkeypatch,
):
    state = SimpleNamespace(producer_calls=0, inner_calls=0, nested=None)
    manager = ActionManager()

    @action(action_manager=manager)
    async def final_a11_no_background_cleanup(agent) -> object:
        """Return a nested coroutine for cleanup-boundary observation."""
        agent.producer_calls += 1

        async def inner():
            agent.inner_calls += 1

        agent.nested = inner()
        return agent.nested

    forbidden_calls = []

    def forbidden(mechanism):
        def fail(*args, **kwargs):
            del args, kwargs
            forbidden_calls.append(mechanism)
            raise AssertionError(f"unexpected nested-awaitable cleanup via {mechanism}")

        return fail

    loop = asyncio.get_running_loop()
    tasks_before = asyncio.all_tasks(loop)
    original_task_factory = loop.get_task_factory()
    loop.set_task_factory(forbidden("event-loop task"))
    monkeypatch.setattr(asyncio, "run", forbidden("nested asyncio.run"))
    monkeypatch.setattr(asyncio, "Runner", forbidden("nested asyncio.Runner"))
    monkeypatch.setattr(asyncio, "new_event_loop", forbidden("nested event loop"))
    monkeypatch.setattr(threading.Thread, "start", forbidden("thread"))

    try:
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(
                state,
                ActionChoice(name="final_a11_no_background_cleanup", arguments={}),
            )

        _assert_final_a11_nested_awaitable_error(
            exc_info.value,
            "final_a11_no_background_cleanup",
        )
        assert forbidden_calls == []
        assert asyncio.all_tasks(loop) == tasks_before
        assert state.producer_calls == 1
        assert state.inner_calls == 0
        assert state.nested.cr_frame is None
    finally:
        loop.set_task_factory(original_task_factory)
        if state.nested is not None and state.nested.cr_frame is not None:
            state.nested.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("producer", "result_factory"),
    [
        pytest.param("async", lambda: "completed", id="async-string"),
        pytest.param("async", lambda: ["completed"], id="async-list"),
        pytest.param(
            "async",
            lambda: {"status": "completed"},
            id="async-dict",
        ),
        pytest.param(
            "sync-coroutine",
            lambda: {"status": "completed"},
            id="sync-callable-one-coroutine",
        ),
    ],
)
async def test_final_a11_aexecute_preserves_completed_result_controls(
    producer,
    result_factory,
):
    expected = result_factory()
    state = SimpleNamespace(producer_calls=0, await_completions=0)
    manager = ActionManager()

    if producer == "async":

        async def completed_action(agent) -> object:
            """Resolve an asynchronous action to one completed result."""
            agent.producer_calls += 1
            await asyncio.sleep(0)
            agent.await_completions += 1
            return expected

    else:

        def completed_action(agent) -> object:
            """Return one coroutine which resolves to a completed result."""
            agent.producer_calls += 1

            async def complete():
                await asyncio.sleep(0)
                agent.await_completions += 1
                return expected

            return complete()

    action_name = f"final_a11_completed_{producer.replace('-', '_')}"
    completed_action.__name__ = action_name
    completed_action = action(action_manager=manager)(completed_action)

    returned = await manager.aexecute(
        state,
        ActionChoice(name=action_name, arguments={}),
    )

    assert manager.actions[action_name] is completed_action
    assert returned is expected
    assert state.producer_calls == 1
    assert state.await_completions == 1


@pytest.mark.asyncio
async def test_execute_rejects_async_action_without_body_or_warning_and_aexecute_runs(
    recwarn,
):
    @action
    async def async_increment_counter(agent, amount: int) -> str:
        """Increment asynchronously.

        Args:
            amount: Amount to add.

        Returns:
            Mutation confirmation.
        """
        agent.body_started = True
        await asyncio.sleep(0)
        agent.counter += amount
        return "async incremented"

    agent = SimpleNamespace(counter=0, body_started=False)
    manager = ActionManager(actions=[async_increment_counter])
    choice = ActionChoice(
        name="async_increment_counter",
        arguments={"amount": "4"},
    )

    with pytest.raises(TypeError, match="Async actions require"):
        manager.execute(agent, choice)

    assert agent.body_started is False
    assert agent.counter == 0
    gc.collect()
    assert not [
        warning for warning in recwarn if "was never awaited" in str(warning.message)
    ]

    result = await manager.aexecute(agent, choice)

    assert result == "async incremented"
    assert agent.body_started is True
    assert agent.counter == 4


def test_execute_still_runs_sync_action_after_async_rejection_guard():
    @action
    def increment_counter(agent, amount: int) -> str:
        """Increment synchronously.

        Args:
            amount: Amount to add.

        Returns:
            Mutation confirmation.
        """
        agent.counter += amount
        return "sync incremented"

    agent = SimpleNamespace(counter=1)
    manager = ActionManager(actions=[increment_counter])

    result = manager.execute(
        agent,
        ActionChoice(name="increment_counter", arguments={"amount": "2"}),
    )

    assert result == "sync incremented"
    assert agent.counter == 3


@pytest.mark.asyncio
async def test_aexecute_awaits_async_action_body_and_applies_side_effect():
    @action
    async def async_mutating_action(agent, amount: int) -> str:
        """Mutate state asynchronously.

        Args:
            amount: Amount to add.

        Returns:
            Mutation confirmation.
        """
        await asyncio.sleep(0)
        agent.counter += amount
        agent.awaited = True
        return "async mutated"

    agent = SimpleNamespace(counter=0, awaited=False)
    manager = ActionManager(actions=[async_mutating_action])

    result = await manager.aexecute(
        agent,
        ActionChoice(
            name="async_mutating_action",
            arguments={"amount": "4"},
        ),
    )

    assert result == "async mutated"
    assert agent.counter == 4
    assert agent.awaited is True


@pytest.mark.asyncio
async def test_aexecute_validation_failure_happens_before_async_execution():
    @action
    async def async_mutating_action(agent, amount: int) -> str:
        """Mutate state asynchronously.

        Args:
            amount: Amount to add.

        Returns:
            Mutation confirmation.
        """
        await asyncio.sleep(0)
        agent.counter += amount
        return "async mutated"

    agent = SimpleNamespace(counter=0)
    manager = ActionManager(actions=[async_mutating_action])

    with pytest.raises(ValueError, match="Missing required argument"):
        await manager.aexecute(
            agent,
            ActionChoice(name="async_mutating_action", arguments={}),
        )

    assert agent.counter == 0


@pytest.mark.asyncio
async def test_execute_closes_native_coroutine_returned_by_sync_action_without_warning(
    recwarn,
):
    async def deferred_mutation(agent) -> str:
        agent.body_started = True
        await asyncio.sleep(0)
        agent.counter += 1
        return "mutated"

    @action
    def return_coroutine(agent):
        """Return a native coroutine from a synchronous action."""
        coroutine = deferred_mutation(agent)
        agent.coroutine = coroutine
        return coroutine

    agent = SimpleNamespace(counter=0, body_started=False, coroutine=None)
    manager = ActionManager(actions=[return_coroutine])

    try:
        with pytest.raises(TypeError, match="Async actions require"):
            manager.execute(
                agent,
                ActionChoice(name="return_coroutine", arguments={}),
            )

        assert agent.coroutine.cr_frame is None
        await asyncio.sleep(0)
        assert agent.body_started is False
        assert agent.counter == 0
    finally:
        if agent.coroutine is not None and agent.coroutine.cr_frame is not None:
            agent.coroutine.close()

    gc.collect()
    assert not [
        warning for warning in recwarn if "was never awaited" in str(warning.message)
    ]


@pytest.mark.asyncio
async def test_execute_cancels_returned_task_before_deferred_mutation():
    @action
    def schedule_mutation(agent, amount: int):
        """Schedule a deferred mutation.

        Args:
            amount: Amount to add.
        """

        async def mutate_later() -> str:
            await asyncio.sleep(0)
            agent.counter += amount
            return "mutated"

        task = asyncio.create_task(mutate_later())
        agent.task = task
        return task

    agent = SimpleNamespace(counter=0, task=None)
    manager = ActionManager(actions=[schedule_mutation])

    try:
        with pytest.raises(TypeError, match="Async actions require"):
            manager.execute(
                agent,
                ActionChoice(
                    name="schedule_mutation",
                    arguments={"amount": 3},
                ),
            )

        assert agent.task.cancelling() > 0
        with pytest.raises(asyncio.CancelledError):
            await agent.task
        await asyncio.sleep(0)
        assert agent.task.cancelled()
        assert agent.counter == 0
    finally:
        if agent.task is not None:
            if not agent.task.done():
                agent.task.cancel()
            with suppress(asyncio.CancelledError):
                await agent.task


@pytest.mark.asyncio
async def test_execute_cancels_returned_future():
    @action
    def return_future(agent):
        """Return a pending Future from a synchronous action."""
        future = asyncio.get_running_loop().create_future()
        agent.future = future
        return future

    agent = SimpleNamespace(future=None)
    manager = ActionManager(actions=[return_future])

    try:
        with pytest.raises(TypeError, match="Async actions require"):
            manager.execute(
                agent,
                ActionChoice(name="return_future", arguments={}),
            )

        assert agent.future.cancelled()
    finally:
        if agent.future is not None:
            if not agent.future.done():
                agent.future.cancel()
            with suppress(asyncio.CancelledError):
                await agent.future


def test_execute_validation_failure_prevents_task_creation():
    @action
    def schedule_mutation(agent, amount: int):
        """Schedule a deferred mutation.

        Args:
            amount: Amount to add.
        """
        agent.invocations += 1
        agent.task = asyncio.create_task(asyncio.sleep(0))
        return agent.task

    agent = SimpleNamespace(invocations=0, task=None)
    manager = ActionManager(actions=[schedule_mutation])

    with pytest.raises(ValueError, match="Missing required argument"):
        manager.execute(
            agent,
            ActionChoice(name="schedule_mutation", arguments={}),
        )

    assert agent.invocations == 0
    assert agent.task is None


@pytest.mark.asyncio
async def test_aexecute_awaits_task_returned_by_sync_action():
    @action
    def schedule_mutation(agent, amount: int):
        """Schedule a deferred mutation.

        Args:
            amount: Amount to add.
        """

        async def mutate_later() -> str:
            await asyncio.sleep(0)
            agent.counter += amount
            return "task completed"

        task = asyncio.create_task(mutate_later())
        agent.task = task
        return task

    agent = SimpleNamespace(counter=0, task=None)
    manager = ActionManager(actions=[schedule_mutation])

    try:
        result = await manager.aexecute(
            agent,
            ActionChoice(
                name="schedule_mutation",
                arguments={"amount": 4},
            ),
        )

        assert result == "task completed"
        assert agent.task.done()
        assert agent.counter == 4
    finally:
        if agent.task is not None and not agent.task.done():
            agent.task.cancel()
            with suppress(asyncio.CancelledError):
                await agent.task


@pytest.mark.asyncio
async def test_aexecute_awaits_future_returned_by_sync_action():
    @action
    def schedule_future_completion(agent, amount: int):
        """Schedule a Future completion.

        Args:
            amount: Amount to add.
        """
        future = asyncio.get_running_loop().create_future()
        agent.future = future

        def complete_future():
            if future.cancelled():
                return
            agent.counter += amount
            future.set_result("future completed")

        asyncio.get_running_loop().call_soon(complete_future)
        return future

    agent = SimpleNamespace(counter=0, future=None)
    manager = ActionManager(actions=[schedule_future_completion])

    try:
        result = await manager.aexecute(
            agent,
            ActionChoice(
                name="schedule_future_completion",
                arguments={"amount": 5},
            ),
        )

        assert result == "future completed"
        assert agent.future.done()
        assert agent.counter == 5
    finally:
        if agent.future is not None and not agent.future.done():
            agent.future.cancel()
            with suppress(asyncio.CancelledError):
                await agent.future


@pytest.mark.parametrize(
    ("annotation", "raw_value", "expected", "expected_type"),
    [
        (int | Literal["007"], "007", "007", str),
        (list[int] | str, "[1, 2]", "[1, 2]", str),
        (Literal["4"] | int, "4", "4", str),
        (str | list[int], "[3, 4]", "[3, 4]", str),
        (int | bool, True, True, bool),
        (str | None, None, None, type(None)),
        (int | float, 4.0, 4.0, float),
        (tuple[int, ...] | list[int], [1, 2], [1, 2], list),
    ],
    ids=[
        "literal-string-after-int",
        "string-after-list",
        "literal-string-before-int",
        "string-before-list",
        "bool-after-int",
        "none-after-string",
        "float-after-int",
        "list-after-tuple",
    ],
)
def test_validate_union_prefers_exact_member_before_coercion(
    annotation,
    raw_value,
    expected,
    expected_type,
):
    def capture_value(agent, value) -> object:
        """Capture a typed value.

        Args:
            value: Value to capture.

        Returns:
            The captured value.
        """
        del agent
        return value

    capture_value.__annotations__["value"] = annotation
    capture_value = action(capture_value)
    manager = ActionManager(actions=[capture_value])

    validated = manager.validate(
        SimpleNamespace(),
        ActionChoice(name="capture_value", arguments={"value": raw_value}),
    )

    assert validated.arguments == {"value": expected}
    assert type(validated.arguments["value"]) is expected_type


@pytest.mark.parametrize(
    ("annotation", "expected", "expected_type"),
    [
        (int | float, 4, int),
        (float | int, 4.0, float),
    ],
    ids=["int-first", "float-first"],
)
def test_validate_union_coercion_without_exact_match_uses_declaration_order(
    annotation,
    expected,
    expected_type,
):
    def capture_value(agent, value) -> object:
        """Capture a typed value.

        Args:
            value: Value to capture.

        Returns:
            The captured value.
        """
        del agent
        return value

    capture_value.__annotations__["value"] = annotation
    capture_value = action(capture_value)
    manager = ActionManager(actions=[capture_value])

    validated = manager.validate(
        SimpleNamespace(),
        ActionChoice(name="capture_value", arguments={"value": "4"}),
    )

    assert validated.arguments == {"value": expected}
    assert type(validated.arguments["value"]) is expected_type


def test_validate_coerces_exact_numeric_string_arguments():
    @action
    def record_measurement(agent, amount: int, ratio: float) -> tuple[int, float]:
        """Record typed numeric values.

        Args:
            amount: Integer amount to record.
            ratio: Floating point ratio to record.

        Returns:
            The typed numeric values.
        """
        del agent
        return amount, ratio

    agent = SimpleNamespace()
    manager = ActionManager(actions=[record_measurement])

    validated = manager.validate(
        agent,
        ActionChoice(
            name="record_measurement",
            arguments={"amount": "4", "ratio": "2.5"},
        ),
    )

    assert validated.arguments == {"amount": 4, "ratio": 2.5}
    assert isinstance(validated.arguments["amount"], int)
    assert isinstance(validated.arguments["ratio"], float)


@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (12, 12),
        (12.0, 12),
        ("12", 12),
        ("-12", -12),
        ("+3", 3),
    ],
)
def test_validate_accepts_exact_lossless_integer_values(amount, expected):
    @action
    def record_amount(agent, amount: int) -> int:
        """Record an integer amount.

        Args:
            amount: Integer amount to record.

        Returns:
            The normalized amount.
        """
        del agent
        return amount

    agent = SimpleNamespace()
    manager = ActionManager(actions=[record_amount])

    validated = manager.validate(
        agent,
        ActionChoice(name="record_amount", arguments={"amount": amount}),
    )

    assert validated.arguments == {"amount": expected}
    assert type(validated.arguments["amount"]) is int


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (2.5, 2.5),
        (2, 2),
        ("2.5", 2.5),
        ("-1", -1.0),
        ("1e3", 1000.0),
    ],
)
def test_validate_accepts_finite_float_values_and_exact_strings(ratio, expected):
    @action
    def record_ratio(agent, ratio: float) -> float:
        """Record a floating point ratio.

        Args:
            ratio: Ratio to record.

        Returns:
            The normalized ratio.
        """
        del agent
        return ratio

    agent = SimpleNamespace()
    manager = ActionManager(actions=[record_ratio])

    validated = manager.validate(
        agent,
        ActionChoice(name="record_ratio", arguments={"ratio": ratio}),
    )

    assert validated.arguments == {"ratio": expected}
    assert math.isfinite(validated.arguments["ratio"])


def test_execute_accepts_numpy_numeric_scalars_across_nested_contracts():
    @action
    def record_values(
        agent,
        count: int,
        ratio: float,
        measurement: int | float,
        batches: list[list[int | float]],
    ) -> tuple[int, float, int | float, list[list[int | float]]]:
        """Record NumPy numeric values after validation.

        Args:
            count: Integer count.
            ratio: Floating point ratio.
            measurement: Numeric measurement.
            batches: Nested numeric batches.

        Returns:
            The normalized values.
        """
        normalized = (count, ratio, measurement, batches)
        agent.recorded.append(normalized)
        return normalized

    agent = SimpleNamespace(recorded=[])
    manager = ActionManager(actions=[record_values])

    result = manager.execute(
        agent,
        ActionChoice(
            name="record_values",
            arguments={
                "count": np.int64(12),
                "ratio": np.float64(2.5),
                "measurement": np.float32(1.25),
                "batches": [[np.int32(2), np.float32(3.5)]],
            },
        ),
    )

    assert result == (12, 2.5, 1.25, [[2, 3.5]])
    assert type(result[0]) is int
    assert isinstance(result[1], float | np.floating)
    assert math.isfinite(result[1])
    assert isinstance(result[2], float | np.floating)
    assert math.isfinite(result[2])
    assert type(result[3][0][0]) is int
    assert isinstance(result[3][0][1], float | np.floating)
    assert math.isfinite(result[3][0][1])
    assert agent.recorded == [result]


@pytest.mark.parametrize("numpy_boolean", [np.bool_(True), np.bool_(False)])
def test_execute_rejects_numpy_boolean_for_int_before_mutation(numpy_boolean):
    @action
    def increment_counter(agent, amount: int) -> str:
        """Increment a counter.

        Args:
            amount: Amount to add.

        Returns:
            Mutation confirmation.
        """
        agent.counter += amount
        return "incremented"

    agent = SimpleNamespace(counter=0)
    manager = ActionManager(actions=[increment_counter])

    with pytest.raises(ValueError, match=r"Invalid argument type.*amount.*int"):
        manager.execute(
            agent,
            ActionChoice(
                name="increment_counter",
                arguments={"amount": numpy_boolean},
            ),
        )

    assert agent.counter == 0


@pytest.mark.parametrize("numpy_boolean", [np.bool_(True), np.bool_(False)])
def test_execute_rejects_numpy_boolean_for_float_before_mutation(numpy_boolean):
    @action
    def scale_total(agent, ratio: float) -> str:
        """Scale a total.

        Args:
            ratio: Floating point multiplier.

        Returns:
            Mutation confirmation.
        """
        agent.total *= ratio
        return "scaled"

    agent = SimpleNamespace(total=10.0)
    manager = ActionManager(actions=[scale_total])

    with pytest.raises(ValueError, match=r"Invalid argument type.*ratio.*float"):
        manager.execute(
            agent,
            ActionChoice(
                name="scale_total",
                arguments={"ratio": numpy_boolean},
            ),
        )

    assert agent.total == 10.0


@pytest.mark.parametrize("numpy_boolean", [np.bool_(True), np.bool_(False)])
def test_execute_rejects_numpy_boolean_for_numeric_union_before_mutation(
    numpy_boolean,
):
    @action
    def record_measurement(agent, measurement: int | float) -> str:
        """Record a numeric measurement.

        Args:
            measurement: Numeric measurement.

        Returns:
            Mutation confirmation.
        """
        agent.measurements.append(measurement)
        return "recorded"

    agent = SimpleNamespace(measurements=[])
    manager = ActionManager(actions=[record_measurement])

    with pytest.raises(
        ValueError,
        match=r"Invalid argument type.*measurement.*int \| float",
    ):
        manager.execute(
            agent,
            ActionChoice(
                name="record_measurement",
                arguments={"measurement": numpy_boolean},
            ),
        )

    assert agent.measurements == []


@pytest.mark.parametrize("numpy_boolean", [np.bool_(True), np.bool_(False)])
def test_execute_rejects_numpy_boolean_in_nested_sequence_before_mutation(
    numpy_boolean,
):
    @action
    def record_batches(agent, batches: list[list[int | float]]) -> str:
        """Record nested numeric batches.

        Args:
            batches: Nested numeric batches.

        Returns:
            Mutation confirmation.
        """
        agent.batches.extend(batches)
        return "recorded"

    agent = SimpleNamespace(batches=[])
    manager = ActionManager(actions=[record_batches])

    with pytest.raises(ValueError, match=r"Invalid argument type.*batches"):
        manager.execute(
            agent,
            ActionChoice(
                name="record_batches",
                arguments={"batches": [[1, numpy_boolean]]},
            ),
        )

    assert agent.batches == []


@pytest.mark.parametrize(
    "non_finite_value",
    [math.nan, math.inf, -math.inf],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_execute_int_float_union_rejects_non_finite_before_mutation(
    non_finite_value,
):
    @action
    def set_coordinate(agent, coordinate: int | float) -> str:
        """Set a finite coordinate.

        Args:
            coordinate: Coordinate to set.

        Returns:
            Coordinate update confirmation.
        """
        if not math.isfinite(coordinate):
            raise ValueError("coordinate must be finite")
        agent.coordinate = coordinate
        return "coordinate set"

    agent = SimpleNamespace(coordinate=1.0)
    manager = ActionManager(actions=[set_coordinate])
    choice = ActionChoice(
        name="set_coordinate",
        arguments={"coordinate": non_finite_value},
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid argument type.*coordinate.*int \| float",
    ):
        manager.execute(agent, choice)

    assert agent.coordinate == 1.0


@pytest.mark.parametrize(
    "non_finite_value",
    [math.inf, -math.inf],
    ids=["positive-infinity", "negative-infinity"],
)
def test_execute_int_only_rejects_infinity_with_standard_error_before_mutation(
    non_finite_value,
):
    @action
    def increment_counter(agent, amount: int) -> str:
        """Increment a counter.

        Args:
            amount: Amount to add.

        Returns:
            Counter update confirmation.
        """
        agent.counter += amount
        return "incremented"

    agent = SimpleNamespace(counter=0)
    manager = ActionManager(actions=[increment_counter])

    with pytest.raises(ValueError, match=r"Invalid argument type.*amount.*int"):
        manager.execute(
            agent,
            ActionChoice(
                name="increment_counter",
                arguments={"amount": non_finite_value},
            ),
        )

    assert agent.counter == 0


@pytest.mark.parametrize(
    "invalid_ratio",
    [
        True,
        math.nan,
        math.inf,
        -math.inf,
        "nan",
        "NaN",
        "inf",
        "-inf",
        "Infinity",
        "-Infinity",
        "ratio 2.5",
    ],
)
def test_execute_rejects_non_finite_or_prose_float_before_mutation(invalid_ratio):
    @action
    def scale_total(agent, ratio: float) -> str:
        """Scale the live agent total.

        Args:
            ratio: Floating point multiplier.

        Returns:
            Scale confirmation.
        """
        agent.total *= ratio
        return "scaled"

    agent = SimpleNamespace(total=10.0)
    manager = ActionManager(actions=[scale_total])

    with pytest.raises(ValueError, match=r"Invalid argument type.*ratio.*float"):
        manager.execute(
            agent,
            ActionChoice(
                name="scale_total",
                arguments={"ratio": invalid_ratio},
            ),
        )

    assert agent.total == 10.0


@pytest.mark.parametrize(
    "invalid_citizen_id",
    [
        "Agent 2",
        "Citizen 3",
        "ID is 4",
        "Do not arrest Agent 12",
        "Citizens 12 and 13",
    ],
)
def test_execute_rejects_integer_labels_and_prose_before_mutation(
    invalid_citizen_id,
):
    manager = ActionManager()

    @action(action_manager=manager)
    def arrest_citizen(agent, citizen_id: int) -> int:
        """Arrest a citizen by id.

        Args:
            citizen_id: Citizen id to arrest.

        Returns:
            The normalized citizen id.
        """
        agent.arrested_ids.append(citizen_id)
        return citizen_id

    agent = SimpleNamespace(arrested_ids=[])

    with pytest.raises(ValueError, match=r"Invalid argument type.*citizen_id.*int"):
        manager.execute(
            agent,
            ActionChoice(
                name="arrest_citizen",
                arguments={"citizen_id": invalid_citizen_id},
            ),
        )

    assert agent.arrested_ids == []


@pytest.mark.parametrize(
    ("listener_agent_ids", "expected"),
    [
        ([12, 13], [12, 13]),
        ("[12, 13]", [12, 13]),
        ('["12", "13"]', [12, 13]),
    ],
)
def test_validate_accepts_real_and_strict_json_int_lists(
    listener_agent_ids,
    expected,
):
    @action
    def notify_agents(agent, listener_agent_ids: list[int]) -> list[int]:
        """Notify agents by id.

        Args:
            listener_agent_ids: Agent ids to notify.

        Returns:
            The normalized agent ids.
        """
        del agent
        return listener_agent_ids

    agent = SimpleNamespace()
    manager = ActionManager(actions=[notify_agents])

    validated = manager.validate(
        agent,
        ActionChoice(
            name="notify_agents",
            arguments={"listener_agent_ids": listener_agent_ids},
        ),
    )

    assert validated.arguments == {"listener_agent_ids": expected}


@pytest.mark.parametrize(
    "invalid_listener_agent_ids",
    [
        "Agent 2",
        "Notify Agent 2",
        "agents 1 and 2",
        "[1, 2",
        "[1,]",
        '["Agent 2"]',
    ],
)
def test_execute_rejects_non_json_or_labeled_int_lists_before_mutation(
    invalid_listener_agent_ids,
):
    @action
    def notify_agents(agent, listener_agent_ids: list[int]) -> str:
        """Notify agents by id.

        Args:
            listener_agent_ids: Agent ids to notify.

        Returns:
            Notification confirmation.
        """
        agent.notified_ids.extend(listener_agent_ids)
        return "notified"

    agent = SimpleNamespace(notified_ids=[])
    manager = ActionManager(actions=[notify_agents])

    with pytest.raises(
        ValueError,
        match=r"Invalid argument type.*listener_agent_ids",
    ):
        manager.execute(
            agent,
            ActionChoice(
                name="notify_agents",
                arguments={"listener_agent_ids": invalid_listener_agent_ids},
            ),
        )

    assert agent.notified_ids == []


def test_validate_coerces_json_string_tuple_arguments():
    @action
    def move_to(agent, target_coordinates: tuple[int, int]) -> tuple[int, int]:
        """Move to coordinates.

        Args:
            target_coordinates: Two-dimensional target coordinates.

        Returns:
            The normalized coordinates.
        """
        del agent
        return target_coordinates

    agent = SimpleNamespace()
    manager = ActionManager(actions=[move_to])

    validated = manager.validate(
        agent,
        ActionChoice(
            name="move_to",
            arguments={"target_coordinates": "[2, 3]"},
        ),
    )

    assert validated.arguments == {"target_coordinates": (2, 3)}


def test_execute_rejects_prose_coordinate_tuple_before_mutation():
    @action
    def move_to(agent, target_coordinates: tuple[int, int]) -> str:
        """Move to coordinates.

        Args:
            target_coordinates: Two-dimensional target coordinates.

        Returns:
            Move confirmation.
        """
        agent.pos = target_coordinates
        return "moved"

    agent = SimpleNamespace(pos=(0, 0))
    manager = ActionManager(actions=[move_to])

    with pytest.raises(
        ValueError,
        match=r"Invalid argument type.*target_coordinates.*tuple\[int, int\]",
    ):
        manager.execute(
            agent,
            ActionChoice(
                name="move_to",
                arguments={"target_coordinates": "coords 1 and 2"},
            ),
        )

    assert agent.pos == (0, 0)


@pytest.mark.parametrize(
    "invalid_amount",
    [
        True,
        2.9,
        "2.9",
        "bad",
        "Agent 2",
        "Citizen 3",
        "ID is 4",
        "Do not arrest Agent 12",
        "Citizen 12.5",
        "agents 1 and 2",
    ],
)
def test_execute_invalid_int_args_fail_before_execution_and_mutation(invalid_amount):
    @action
    def increment_counter(agent, amount: int) -> str:
        """Increment the counter.

        Args:
            amount: Amount to add.

        Returns:
            Mutation confirmation.
        """
        agent.counter += amount
        return "incremented"

    agent = SimpleNamespace(counter=0)
    manager = ActionManager(actions=[increment_counter])

    with pytest.raises(ValueError, match=r"Invalid argument type.*amount.*int"):
        manager.execute(
            agent,
            ActionChoice(
                name="increment_counter",
                arguments={"amount": invalid_amount},
            ),
        )

    assert agent.counter == 0


def test_execute_calls_configured_action_with_validated_arguments_and_returns_result():
    @action
    def record_score(agent, amount: int, label: str) -> dict:
        """Record a score.

        Args:
            amount: Score amount.
            label: Score label.

        Returns:
            Recorded score details.
        """
        agent.model.counter += amount
        agent.labels.append(label)
        return {"counter": agent.model.counter, "label": label}

    model = SimpleNamespace(counter=0)
    agent = SimpleNamespace(model=model, labels=[])
    manager = ActionManager(actions=[record_score])

    result = manager.execute(
        agent,
        ActionChoice(
            name="record_score",
            arguments={"amount": 3, "label": "bonus"},
        ),
    )

    assert result == {"counter": 3, "label": "bonus"}
    assert model.counter == 3
    assert agent.labels == ["bonus"]


def test_execute_uses_coerced_arguments_and_injects_live_agent():
    @action
    def scale_total(agent, amount: int, multiplier: float) -> str:
        """Scale the live agent total.

        Args:
            amount: Integer amount to scale.
            multiplier: Floating point multiplier.

        Returns:
            Live agent name.
        """
        agent.observed_arguments.append(
            (amount, type(amount), multiplier, type(multiplier)),
        )
        agent.total += amount * multiplier
        return agent.name

    agent = SimpleNamespace(name="live-agent", observed_arguments=[], total=0)
    manager = ActionManager(actions=[scale_total])

    result = manager.execute(
        agent,
        ActionChoice(
            name="scale_total",
            arguments={"amount": "4", "multiplier": "2.5"},
        ),
    )

    assert result == "live-agent"
    assert agent.observed_arguments == [(4, int, 2.5, float)]
    assert agent.total == 10.0


def test_execute_injects_agent_when_action_accepts_agent_parameter():
    @action
    def capture_agent(agent, message: str) -> str:
        """Capture the provided agent.

        Args:
            message: Message to record.

        Returns:
            Agent-specific message.
        """
        agent.messages.append(message)
        return f"{agent.name}:{message}"

    agent = SimpleNamespace(name="agent-1", messages=[])
    manager = ActionManager(actions=[capture_agent])

    result = manager.execute(
        agent,
        ActionChoice(name="capture_agent", arguments={"message": "hello"}),
    )

    assert result == "agent-1:hello"
    assert agent.messages == ["hello"]


def test_action_schema_omits_injected_agent_and_execute_still_injects_it():
    @action
    def capture_agent_identity(agent: LLMAgent, message: str) -> str:
        """Capture the injected agent.

        Args:
            message: Message to capture.

        Returns:
            Agent identity confirmation.
        """
        agent.messages.append(message)
        return f"{agent.name}:{message}"

    params = capture_agent_identity.__action_schema__["parameters"]
    agent = SimpleNamespace(name="agent-1", messages=[])
    manager = ActionManager(actions=[capture_agent_identity])

    result = manager.execute(
        agent,
        ActionChoice(
            name="capture_agent_identity",
            arguments={"message": "hello"},
        ),
    )

    assert set(params["properties"]) == {"message"}
    assert params["required"] == ["message"]
    assert "agent" not in capture_agent_identity.__action_metadata__.parameters
    assert result == "agent-1:hello"
    assert agent.messages == ["hello"]


def test_execute_supports_action_without_agent_parameter():
    @action
    def add_without_agent(amount: int, increment: int) -> int:
        """Add values without needing an agent.

        Args:
            amount: Base value.
            increment: Value to add.

        Returns:
            Sum of the values.
        """
        return amount + increment

    agent = SimpleNamespace(name="unused")
    manager = ActionManager(actions=[add_without_agent])

    result = manager.execute(
        agent,
        ActionChoice(
            name="add_without_agent",
            arguments={"amount": 4, "increment": 5},
        ),
    )

    assert result == 9


def test_execute_respects_explicit_narrowed_actions():
    @action
    def selected_action(agent, amount: int) -> str:
        """Selected action.

        Args:
            amount: Amount to add.

        Returns:
            Selection confirmation.
        """
        agent.selected += amount
        return "selected"

    @action
    def other_action(agent, amount: int) -> str:
        """Other action.

        Args:
            amount: Amount to add.

        Returns:
            Other action confirmation.
        """
        agent.other += amount
        return "other"

    agent = SimpleNamespace(selected=0, other=0)
    manager = ActionManager(actions=[selected_action, other_action])

    result = manager.execute(
        agent,
        ActionChoice(name="selected_action", arguments={"amount": 2}),
        actions=[selected_action],
    )

    assert result == "selected"
    assert agent.selected == 2
    assert agent.other == 0


def test_execute_invalid_action_name_fails_before_execution():
    @action
    def configured_action(agent) -> str:
        """Configured action.

        Returns:
            Mutation confirmation.
        """
        agent.mutations.append("configured")
        return "configured"

    agent = SimpleNamespace(mutations=[])
    manager = ActionManager(actions=[configured_action])

    with pytest.raises(ValueError, match="Unknown action name"):
        manager.execute(
            agent,
            ActionChoice(name="missing_action", arguments={}),
        )

    assert agent.mutations == []


def test_execute_missing_required_args_fail_before_execution_and_mutation():
    @action
    def increment_counter(agent, amount: int) -> str:
        """Increment the counter.

        Args:
            amount: Amount to add.

        Returns:
            Mutation confirmation.
        """
        agent.counter += amount
        return "incremented"

    agent = SimpleNamespace(counter=0)
    manager = ActionManager(actions=[increment_counter])

    with pytest.raises(ValueError, match="Missing required argument"):
        manager.execute(
            agent,
            ActionChoice(name="increment_counter", arguments={}),
        )

    assert agent.counter == 0


def test_execute_unexpected_extra_args_fail_before_execution_and_mutation():
    @action
    def store_message(agent, message: str) -> str:
        """Store a message.

        Args:
            message: Message to store.

        Returns:
            Stored message.
        """
        agent.messages.append(message)
        return message

    agent = SimpleNamespace(messages=[])
    manager = ActionManager(actions=[store_message])

    with pytest.raises(ValueError, match="Unexpected argument"):
        manager.execute(
            agent,
            ActionChoice(
                name="store_message",
                arguments={"message": "hello", "volume": "loud"},
            ),
        )

    assert agent.messages == []


def test_execute_narrowed_out_action_fails_before_execution_and_mutation():
    @action
    def allowed_action(agent) -> str:
        """Allowed action.

        Returns:
            Allowed action confirmation.
        """
        agent.allowed += 1
        return "allowed"

    @action
    def narrowed_out_action(agent) -> str:
        """Narrowed out action.

        Returns:
            Narrowed out action confirmation.
        """
        agent.narrowed_out += 1
        return "narrowed out"

    agent = SimpleNamespace(allowed=0, narrowed_out=0)
    manager = ActionManager(actions=[allowed_action, narrowed_out_action])

    with pytest.raises(ValueError, match="Unknown action name"):
        manager.execute(
            agent,
            ActionChoice(name="narrowed_out_action", arguments={}),
            actions=[allowed_action],
        )

    assert agent.allowed == 0
    assert agent.narrowed_out == 0
