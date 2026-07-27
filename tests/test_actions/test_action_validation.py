from __future__ import annotations

import asyncio
import gc
import math
from contextlib import suppress
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from mesa_llm.actions import ActionChoice, ActionManager, action
from mesa_llm.actions.action_decorator import _GLOBAL_ACTION_REGISTRY

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

    class MissingPayloadType:
        pass


@pytest.fixture(autouse=True)
def restore_global_action_registry():
    """Keep bare @action registrations local to each test."""
    original_registry = dict(_GLOBAL_ACTION_REGISTRY)
    ActionManager.instances.clear()
    yield
    _GLOBAL_ACTION_REGISTRY.clear()
    _GLOBAL_ACTION_REGISTRY.update(original_registry)
    ActionManager.instances.clear()


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


def test_execute_unresolved_postponed_non_agent_annotation_fails_closed_before_mutation():
    def apply_payload(agent: LLMAgent, payload: MissingPayloadType) -> str:
        """Apply a typed payload.

        Args:
            payload: Payload to apply.

        Returns:
            Payload confirmation.
        """
        agent.applied_payloads.append(payload)
        return "applied"

    agent = SimpleNamespace(applied_payloads=[])
    manager = ActionManager(actions=[apply_payload])

    with pytest.raises(
        ValueError,
        match=r"Could not resolve annotation.*payload.*MissingPayloadType",
    ):
        manager.execute(
            agent,
            ActionChoice(
                name="apply_payload",
                arguments={"payload": {"value": 1}},
            ),
        )

    assert agent.applied_payloads == []


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
    "non_finite_value",
    [math.inf, -math.inf],
    ids=["positive-infinity", "negative-infinity"],
)
def test_validate_int_float_union_preserves_infinity_for_domain_validation(
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

    validated = manager.validate(agent, choice)

    assert validated.arguments == {"coordinate": non_finite_value}
    with pytest.raises(ValueError, match="coordinate must be finite"):
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


def test_execute_rejects_embedded_float_text_before_mutation():
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
                arguments={"ratio": "ratio 2.5"},
            ),
        )

    assert agent.total == 10.0


def test_validate_coerces_single_embedded_integer_from_llm_text():
    @action
    def arrest_citizen(agent, citizen_id: int) -> int:
        """Arrest a citizen by id.

        Args:
            citizen_id: Citizen id to arrest.

        Returns:
            The normalized citizen id.
        """
        del agent
        return citizen_id

    agent = SimpleNamespace()
    manager = ActionManager(actions=[arrest_citizen])

    validated = manager.validate(
        agent,
        ActionChoice(
            name="arrest_citizen",
            arguments={"citizen_id": "Citizen 12"},
        ),
    )

    assert validated.arguments == {"citizen_id": 12}
    assert isinstance(validated.arguments["citizen_id"], int)


def test_validate_rejects_ambiguous_embedded_integers_for_scalar_int():
    @action
    def arrest_citizen(agent, citizen_id: int) -> int:
        """Arrest a citizen by id.

        Args:
            citizen_id: Citizen id to arrest.

        Returns:
            The normalized citizen id.
        """
        del agent
        return citizen_id

    agent = SimpleNamespace()
    manager = ActionManager(actions=[arrest_citizen])

    with pytest.raises(ValueError, match=r"Invalid argument type.*citizen_id.*int"):
        manager.validate(
            agent,
            ActionChoice(
                name="arrest_citizen",
                arguments={"citizen_id": "Citizens 12 and 13"},
            ),
        )


def test_validate_coerces_string_json_list_to_int_list():
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
            arguments={"listener_agent_ids": "[12, 13]"},
        ),
    )

    assert validated.arguments == {"listener_agent_ids": [12, 13]}


def test_validate_coerces_single_free_text_id_to_int_list():
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
            arguments={"listener_agent_ids": "Agent 12"},
        ),
    )

    assert validated.arguments == {"listener_agent_ids": [12]}


def test_execute_rejects_multi_id_prose_int_list_before_mutation():
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
        match=r"Invalid argument type.*listener_agent_ids.*list\[int\]",
    ):
        manager.execute(
            agent,
            ActionChoice(
                name="notify_agents",
                arguments={"listener_agent_ids": "agents 1 and 2"},
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
    ["bad", 2.9, "2.9", "Citizen 12.5", "agents 1 and 2"],
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
