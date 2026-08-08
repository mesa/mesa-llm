# tests/test_llm_agent.py

import asyncio
import gc
import json
import logging
import warnings
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from litellm.exceptions import APIConnectionError, RateLimitError, Timeout
from mesa.agent import Agent
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.model import Model
from mesa.space import ContinuousSpace, MultiGrid, SingleGrid

import mesa_llm.actions as action_exports
from mesa_llm import (
    ActionPostCommitError as RootActionPostCommitError,
)
from mesa_llm import (
    Plan,
)
from mesa_llm.actions import (
    ActionChoice,
    ActionManager,
    ActionPostCommitError,
    action,
    wait,
)
from mesa_llm.actions.action_decorator import _GLOBAL_ACTION_REGISTRY
from mesa_llm.actions.action_manager import _UNSET as _ACTIONS_UNSET
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.episodic_memory import EpisodicMemory
from mesa_llm.memory.st_memory import ShortTermMemory
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.reasoning.reasoning import _UNSET as _TOOLS_UNSET
from mesa_llm.tools.tool_decorator import tool
from mesa_llm.tools.tool_manager import ToolManager


def test_apply_plan_adds_to_memory(monkeypatch):
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos):
            system_prompt = "You are an agent in a simulation."
            agents = LLMAgent.create_agents(
                self,
                n=1,
                reasoning=ReActReasoning,
                system_prompt=system_prompt,
                vision=-1,
                internal_state=["test_state"],
            )

            x, y = pos

            agent = agents.to_list()[0]
            self.grid.place_agent(agent, (x, y))
            return agent

    model = DummyModel()
    agent = model.add_agent((1, 1))
    agent.memory = ShortTermMemory(
        agent=agent,
        n=5,
        display=True,
    )

    # fake response returned by the tool manager
    fake_response = [{"tool": "foo", "argument": "bar"}]

    # monkeypatch the tool manager so no real tool calls are made
    monkeypatch.setattr(
        agent.tool_manager, "call_tools", lambda agent, llm_response: fake_response
    )

    plan = Plan(step=0, llm_plan="do something")

    resp = agent.apply_plan(plan)

    assert resp == fake_response

    # "action" is an additive event type, so it is stored as a list
    assert "action" in agent.memory.step_content
    actions = agent.memory.step_content["action"]
    assert isinstance(actions, list)
    assert len(actions) == 1
    assert "tool_calls" in actions[0]
    assert actions[0]["tool_calls"][0] == {"tool": "foo", "argument": "bar"}


def test_llm_agent_tools_constructor_tri_state():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @tool
    def agent_constructor_tool(agent, value: int) -> int:
        """Agent constructor tool.
        Args:
            agent: The agent making the request (provided automatically)
            value: Input.
        Returns:
            Output.
        """
        return value

    model = DummyModel()

    no_tools_agent = LLMAgent(model, reasoning=ReActReasoning, tools=None)
    empty_tools_agent = LLMAgent(model, reasoning=ReActReasoning, tools=[])
    explicit_tools_agent = LLMAgent(
        model,
        reasoning=ReActReasoning,
        tools=[agent_constructor_tool],
    )

    assert no_tools_agent._tool_manager.tools == {}
    assert empty_tools_agent._tool_manager.tools == {}
    assert explicit_tools_agent._tool_manager.tools == {
        "agent_constructor_tool": agent_constructor_tool
    }


def test_llm_agent_actions_constructor_tri_state():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    model = DummyModel()

    no_actions_agent = LLMAgent(model, reasoning=ReActReasoning, actions=None)
    empty_actions_agent = LLMAgent(model, reasoning=ReActReasoning, actions=[])
    explicit_actions_agent = LLMAgent(
        model,
        reasoning=ReActReasoning,
        actions=[wait],
    )

    assert no_actions_agent._action_manager.actions == {}
    assert empty_actions_agent._action_manager.actions == {}
    assert explicit_actions_agent._action_manager.actions == {"wait": wait}
    assert explicit_actions_agent._action_manager.available_actions() == {"wait": wait}


def test_llm_agent_actions_constructor_accepts_registered_action_name():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    original_registry = dict(_GLOBAL_ACTION_REGISTRY)
    try:

        @action
        def llm_agent_registered_action(agent, value: int) -> int:
            """Registered action.

            Args:
                value: Value to return.

            Returns:
                The value.
            """
            del agent
            return value

        agent = LLMAgent(
            DummyModel(),
            reasoning=ReActReasoning,
            actions=["llm_agent_registered_action"],
        )

        assert agent._action_manager.actions == {
            "llm_agent_registered_action": llm_agent_registered_action
        }
        assert agent._action_manager.available_actions() == {
            "llm_agent_registered_action": llm_agent_registered_action
        }
    finally:
        _GLOBAL_ACTION_REGISTRY.clear()
        _GLOBAL_ACTION_REGISTRY.update(original_registry)


def _exception_signature(call):
    try:
        call()
    except Exception as exc:
        return type(exc), str(exc)
    pytest.fail("Expected construction to raise an exception.")


def _assert_failed_llm_agent_construction_is_cleaned_up(
    monkeypatch,
    constructor,
    expected_exception,
    expected_exception_object=None,
):
    model = Model(rng=42)
    baseline_agent = Agent(model)
    baseline_count = len(model.agents)
    registered_agents = []
    original_register_agent = model.register_agent

    def capture_registered_agent(agent):
        result = original_register_agent(agent)
        registered_agents.append(agent)
        return result

    monkeypatch.setattr(model, "register_agent", capture_registered_agent)

    try:
        constructor(model)
    except Exception as exc:
        initialization_error = exc
    else:
        pytest.fail("Expected LLMAgent construction to raise an exception.")

    assert (type(initialization_error), str(initialization_error)) == expected_exception
    if expected_exception_object is not None:
        assert initialization_error is expected_exception_object
    assert len(registered_agents) == 1

    failed_agent = registered_agents.pop()
    failed_agent_ref = weakref.ref(failed_agent)

    assert len(model.agents) == baseline_count
    assert baseline_agent in model.agents
    assert failed_agent not in model.agents
    assert all(
        failed_agent not in typed_agents
        for typed_agents in model.agents_by_type.values()
    )

    initialization_error.__traceback__ = None
    del initialization_error, failed_agent
    gc.collect()

    assert failed_agent_ref() is None


def test_llm_agent_registration_failure_after_mesa_mutation_is_cleaned_up():
    registration_error = RuntimeError("register_agent failed after registration")

    class RaiseAfterRegistrationModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.registered_agent_ref = None

        def register_agent(self, agent):
            super().register_agent(agent)
            self.registered_agent_ref = weakref.ref(agent)
            raise registration_error

    model = RaiseAfterRegistrationModel()

    def construct_and_assert_cleanup():
        with pytest.raises(RuntimeError) as exc_info:
            LLMAgent(model, reasoning=ReActReasoning)

        assert exc_info.value is registration_error
        assert str(exc_info.value) == "register_agent failed after registration"

        failed_agent_ref = model.registered_agent_ref
        assert failed_agent_ref is not None
        failed_agent = failed_agent_ref()
        assert failed_agent is not None
        assert failed_agent not in model.agents
        assert all(
            failed_agent not in typed_agents
            for typed_agents in model.agents_by_type.values()
        )

        exc_info.value.__traceback__ = None
        return failed_agent_ref

    failed_agent_ref = construct_and_assert_cleanup()
    gc.collect()

    assert failed_agent_ref() is None


def test_llm_agent_unknown_action_failure_cleans_up_mesa_registration(monkeypatch):
    missing_action_name = "missing_llm_agent_lifecycle_action"
    expected_exception = _exception_signature(
        lambda: ActionManager(actions=[missing_action_name])
    )

    _assert_failed_llm_agent_construction_is_cleaned_up(
        monkeypatch,
        lambda model: LLMAgent(
            model,
            reasoning=ReActReasoning,
            actions=[missing_action_name],
        ),
        expected_exception,
    )


def test_llm_agent_undecorated_action_failure_cleans_up_mesa_registration(
    monkeypatch,
):
    def undecorated_action(agent, value: int) -> int:
        """Return a value supplied by an agent."""
        del agent
        return value

    expected_exception = _exception_signature(
        lambda: ActionManager(actions=[undecorated_action])
    )

    _assert_failed_llm_agent_construction_is_cleaned_up(
        monkeypatch,
        lambda model: LLMAgent(
            model,
            reasoning=ReActReasoning,
            actions=[undecorated_action],
        ),
        expected_exception,
    )


def test_llm_agent_invalid_tool_failure_cleans_up_mesa_registration(monkeypatch):
    invalid_tool = object()
    expected_exception = _exception_signature(lambda: ToolManager(tools=[invalid_tool]))

    _assert_failed_llm_agent_construction_is_cleaned_up(
        monkeypatch,
        lambda model: LLMAgent(
            model,
            reasoning=ReActReasoning,
            tools=[invalid_tool],
        ),
        expected_exception,
    )


def test_llm_agent_reasoning_failure_cleans_up_mesa_registration(monkeypatch):
    reasoning_error = RuntimeError("reasoning constructor failed")

    class FailingReasoning(ReActReasoning):
        def __init__(self, agent):
            del agent
            raise reasoning_error

    _assert_failed_llm_agent_construction_is_cleaned_up(
        monkeypatch,
        lambda model: LLMAgent(model, reasoning=FailingReasoning),
        (RuntimeError, "reasoning constructor failed"),
        expected_exception_object=reasoning_error,
    )


def test_llm_agent_late_failure_bypasses_overridden_remove_during_cleanup(
    monkeypatch,
):
    late_initialization_error = RuntimeError("late LLMAgent initialization failed")

    class LateFailingLLMAgent(LLMAgent):
        remove_calls = 0

        @LLMAgent.system_prompt.setter
        def system_prompt(self, value):
            del value
            raise late_initialization_error

        def remove(self):
            type(self).remove_calls += 1
            raise AssertionError("failed-construction cleanup called remove override")

    _assert_failed_llm_agent_construction_is_cleaned_up(
        monkeypatch,
        lambda model: LateFailingLLMAgent(model, reasoning=ReActReasoning),
        (RuntimeError, "late LLMAgent initialization failed"),
        expected_exception_object=late_initialization_error,
    )

    assert LateFailingLLMAgent.remove_calls == 0


def test_successful_llm_agent_construction_remains_registered():
    model = Model(rng=42)
    baseline_agent = Agent(model)
    baseline_count = len(model.agents)

    agent = LLMAgent(model, reasoning=ReActReasoning)

    assert len(model.agents) == baseline_count + 1
    assert baseline_agent in model.agents
    assert agent in model.agents
    assert agent in model.agents_by_type[type(agent)]


def test_llm_agent_does_not_expose_public_action_manager_property():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    agent = LLMAgent(DummyModel(), reasoning=ReActReasoning, actions=[wait])

    assert agent._action_manager.available_actions() == {"wait": wait}
    assert not hasattr(agent, "action_manager")


def test_execute_action_validates_before_mutation_and_executes_configured_action():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    def increment_counter(agent, amount: int) -> str:
        """Increment the counter.

        Args:
            amount: Amount to add.

        Returns:
            Mutation confirmation.
        """
        agent.counter += amount
        return "incremented"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[increment_counter],
    )
    agent.counter = 0

    with pytest.raises(ValueError, match="Missing required argument"):
        agent.execute_action(
            ActionChoice(name="increment_counter", arguments={}),
        )

    assert agent.counter == 0

    result = agent.execute_action(
        ActionChoice(
            name="increment_counter",
            arguments={"amount": "2"},
        ),
    )

    assert result == "incremented"
    assert agent.counter == 2


def test_execute_action_respects_omitted_explicit_and_narrowed_actions():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    def selected_action(agent, amount: int) -> str:
        """Selected action.

        Args:
            amount: Amount to add.

        Returns:
            Selection confirmation.
        """
        agent.selected += amount
        return "selected"

    @action(action_manager=ActionManager())
    def other_action(agent, amount: int) -> str:
        """Other action.

        Args:
            amount: Amount to add.

        Returns:
            Other confirmation.
        """
        agent.other += amount
        return "other"

    @action(action_manager=ActionManager())
    def unconfigured_action(agent) -> str:
        """Unconfigured action.

        Returns:
            Unconfigured confirmation.
        """
        agent.unconfigured += 1
        return "unconfigured"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[selected_action, other_action],
    )
    agent.selected = 0
    agent.other = 0
    agent.unconfigured = 0

    assert (
        agent.execute_action(
            ActionChoice(name="other_action", arguments={"amount": 3}),
        )
        == "other"
    )
    assert agent.other == 3

    for no_actions in [None, []]:
        with pytest.raises(ValueError, match="Unknown action name"):
            agent.execute_action(
                ActionChoice(name="selected_action", arguments={"amount": 1}),
                actions=no_actions,
            )

    assert agent.selected == 0

    assert (
        agent.execute_action(
            ActionChoice(name="selected_action", arguments={"amount": 2}),
            actions=["selected_action"],
        )
        == "selected"
    )
    assert agent.selected == 2

    with pytest.raises(ValueError, match="Unknown action name"):
        agent.execute_action(
            ActionChoice(name="other_action", arguments={"amount": 1}),
            actions=[selected_action],
        )

    with pytest.raises(ValueError, match="Unknown action name"):
        agent.execute_action(
            ActionChoice(name="selected_action", arguments={"amount": 1}),
            actions=[unconfigured_action],
        )

    assert agent.selected == 2
    assert agent.other == 3
    assert agent.unconfigured == 0


def _action_choice_response(content, reasoning_content=None):
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _close_possible_coroutine(result):
    possible_coroutine = getattr(result, "result", result)
    if asyncio.iscoroutine(possible_coroutine):
        possible_coroutine.close()


def _assert_rejects_async_action_without_unawaited_warning(call):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        result = None
        try:
            result = call()
        except TypeError as exc:
            message = str(exc).lower()
            assert any(term in message for term in ("async", "await", "coroutine"))
        else:
            _close_possible_coroutine(result)
            pytest.fail("Expected sync execution to reject an async action.")
        finally:
            result = None
            gc.collect()

    unawaited_warnings = [
        warning
        for warning in caught_warnings
        if issubclass(warning.category, RuntimeWarning)
        and "was never awaited" in str(warning.message)
    ]
    assert unawaited_warnings == []


def _make_local_action_choice_agent(
    llm_model: str = "gemini/gemini-2.0-flash",
):
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    def local_increment_counter(agent, amount: int) -> str:
        """Increment the counter.

        Args:
            amount: Amount to add.

        Returns:
            Mutation confirmation.
        """
        agent.counter += amount
        return "incremented"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        llm_model=llm_model,
        actions=[local_increment_counter],
    )
    agent.counter = 0
    return agent, local_increment_counter


_ACTION_SELECTION_TRANSPORT_ERRORS = (
    APIConnectionError,
    Timeout,
    RateLimitError,
)


def _make_action_selection_transport_error(error_type):
    return error_type(
        message="selection transport failure",
        llm_provider="openai",
        model="openai/gpt-4o",
        max_retries=0,
        num_retries=0,
    )


def _action_selection_state(agent):
    return (
        agent.counter,
        tuple(agent.internal_state),
        agent.model.steps,
        tuple(agent.memory.short_term_memory),
        dict(agent.memory.step_content),
    )


def _assert_normalized_action_selection_transport_error(
    error,
    error_type,
):
    assert type(error) is error_type
    assert error.llm_provider == "openai"
    assert error.model == "openai/gpt-4o"
    assert error.num_retries == 0
    assert error.max_retries == 0
    assert "selection transport failure" in str(error)

    if error_type is RateLimitError:
        assert "Rate limit exceeded for model 'openai/gpt-4o'." in str(error)
        assert "https://developers.openai.com/api/docs/guides/rate-limits" in str(error)
    else:
        assert str(error) == (
            f"litellm.{error_type.__name__}: selection transport failure"
        )


@pytest.mark.parametrize(
    ("content", "expected_rationale"),
    [
        (
            json.dumps(
                {
                    "name": "local_increment_counter",
                    "arguments": {"amount": "3"},
                    "rationale": "Plain local JSON.",
                },
            ),
            "Plain local JSON.",
        ),
        (
            "```json\n"
            + json.dumps(
                {
                    "name": "local_increment_counter",
                    "arguments": {"amount": "4"},
                    "rationale": "Fenced local JSON.",
                },
            )
            + "\n```",
            "Fenced local JSON.",
        ),
        (
            "I will commit to this action:\n"
            + json.dumps(
                {
                    "name": "local_increment_counter",
                    "arguments": {"amount": "5"},
                    "rationale": "Embedded local JSON.",
                },
            )
            + "\nNo tools are needed.",
            "Embedded local JSON.",
        ),
    ],
)
def test_choose_action_parses_local_json_fallbacks_and_validates_choice(
    content,
    expected_rationale,
):
    agent, _ = _make_local_action_choice_agent()
    agent.llm.generate = Mock(return_value=_action_choice_response(content))

    choice = agent.choose_action("Choose one local action.")

    assert choice.name == "local_increment_counter"
    assert isinstance(choice.arguments["amount"], int)
    assert choice.rationale == expected_rationale
    assert agent.counter == 0

    agent.llm.generate.assert_called_once()
    call_kwargs = agent.llm.generate.call_args.kwargs
    assert call_kwargs["tool_schema"] is None
    assert call_kwargs["tool_choice"] == "none"
    assert call_kwargs["response_format"] is ActionChoice
    assert call_kwargs["suppress_thinking"] is True


@pytest.mark.parametrize(
    "content",
    [
        '{"name": "local_increment_counter", "arguments": {"amount":',
        json.dumps({"name": "local_increment_counter"}),
        json.dumps(
            {
                "name": "local_increment_counter",
                "arguments": {"amount": 1},
            },
        )
        + "\n"
        + json.dumps(
            {
                "name": "local_increment_counter",
                "arguments": {"amount": 2},
            },
        ),
    ],
)
def test_choose_action_rejects_invalid_or_ambiguous_local_output_before_mutation(
    content,
):
    agent, _ = _make_local_action_choice_agent()
    agent.llm.generate = Mock(return_value=_action_choice_response(content))
    agent.execute_action = Mock(side_effect=AssertionError("must not execute"))

    with pytest.raises(ValueError):
        agent.choose_action("Choose one local action.")

    assert agent.counter == 0
    agent.llm.generate.assert_called_once()
    agent.execute_action.assert_not_called()


def test_choose_action_invalid_output_makes_one_provider_request_without_tools(
    monkeypatch,
):
    calls = []

    def _invalid_completion(**kwargs):
        calls.append(kwargs)
        return _action_choice_response("not valid action JSON")

    monkeypatch.setattr("mesa_llm.module_llm.completion", _invalid_completion)
    agent, _ = _make_local_action_choice_agent(llm_model="ollama/llama3.2:3b")

    with pytest.raises(ValueError):
        agent.choose_action("Choose one local action.")

    assert len(calls) == 1
    provider_kwargs = calls[0]
    assert provider_kwargs["tools"] is None
    assert provider_kwargs["tool_choice"] is None
    assert provider_kwargs["response_format"] is ActionChoice
    assert provider_kwargs["think"] is False
    assert agent.counter == 0


@pytest.mark.parametrize(
    "error_type",
    _ACTION_SELECTION_TRANSPORT_ERRORS,
    ids=lambda error_type: error_type.__name__,
)
def test_act_transport_error_is_one_shot_and_mutation_free(
    monkeypatch,
    error_type,
):
    provider_calls = []
    original_error = _make_action_selection_transport_error(error_type)

    def _raise_transport_error(**kwargs):
        provider_calls.append(kwargs)
        if len(provider_calls) > 1:
            raise AssertionError("action selection retried after a transport error")
        raise original_error

    monkeypatch.setattr(
        "mesa_llm.module_llm.completion",
        _raise_transport_error,
    )
    agent, _ = _make_local_action_choice_agent(llm_model="openai/gpt-4o")
    agent.recorder = Mock()
    agent.execute_action = Mock(
        side_effect=AssertionError("transport failure must prevent action execution")
    )
    state_before = _action_selection_state(agent)

    with pytest.raises(error_type) as exc_info:
        agent.act("Choose and execute one local action.")

    assert len(provider_calls) == 1
    provider_kwargs = provider_calls[0]
    assert provider_kwargs["num_retries"] == 0
    assert provider_kwargs["max_retries"] == 0
    assert provider_kwargs["fallbacks"] == []
    assert provider_kwargs["tools"] is None
    assert provider_kwargs["tool_choice"] is None
    assert provider_kwargs["response_format"] is ActionChoice
    _assert_normalized_action_selection_transport_error(
        exc_info.value,
        error_type,
    )
    assert _action_selection_state(agent) == state_before
    agent.execute_action.assert_not_called()
    agent.recorder.record_event.assert_not_called()


def test_act_extra_argument_fails_after_one_provider_request_before_mutation(
    monkeypatch,
):
    provider_calls = []

    def _extra_argument_completion(**kwargs):
        provider_calls.append(kwargs)
        return _action_choice_response(
            json.dumps(
                {
                    "name": "local_increment_counter",
                    "arguments": {"amount": 1, "undeclared": "reject me"},
                    "rationale": "Attempt an invalid local action.",
                }
            )
        )

    monkeypatch.setattr(
        "mesa_llm.module_llm.completion",
        _extra_argument_completion,
    )
    agent, _ = _make_local_action_choice_agent()
    agent.recorder = Mock()
    agent.execute_action = Mock(
        side_effect=AssertionError("act must not execute an invalid selection")
    )

    with pytest.raises(ValueError, match="Unexpected argument"):
        agent.act("Choose and execute one local action.")

    assert len(provider_calls) == 1
    assert agent.counter == 0
    assert "action" not in agent.memory.step_content
    agent.execute_action.assert_not_called()
    agent.recorder.record_event.assert_not_called()


def test_choose_action_prefers_final_content_over_reasoning_json():
    agent, _ = _make_local_action_choice_agent()
    agent.llm.generate = Mock(
        return_value=_action_choice_response(
            json.dumps(
                {
                    "name": "local_increment_counter",
                    "arguments": {"amount": "8"},
                    "rationale": "Use final content.",
                },
            ),
            reasoning_content=(
                "Reasoning considered another object: "
                '{"name": "local_increment_counter", "arguments": {"amount": 1}}'
            ),
        ),
    )

    choice = agent.choose_action("Choose one local action.")

    assert choice.name == "local_increment_counter"
    assert choice.arguments == {"amount": 8}
    assert choice.rationale == "Use final content."
    assert agent.counter == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "```json\n"
        + json.dumps(
            {
                "name": "local_increment_counter",
                "arguments": {"amount": "6"},
                "rationale": "Async fenced local JSON.",
            },
        )
        + "\n```",
        "The selected action is "
        + json.dumps(
            {
                "name": "local_increment_counter",
                "arguments": {"amount": "7"},
                "rationale": "Async embedded local JSON.",
            },
        ),
    ],
)
async def test_achoose_action_parses_local_json_fallbacks_without_tools(content):
    agent, _ = _make_local_action_choice_agent()
    agent.llm.agenerate = AsyncMock(return_value=_action_choice_response(content))

    choice = await agent.achoose_action("Choose one async local action.")

    assert choice.name == "local_increment_counter"
    assert isinstance(choice.arguments["amount"], int)
    assert agent.counter == 0

    agent.llm.agenerate.assert_awaited_once()
    call_kwargs = agent.llm.agenerate.call_args.kwargs
    assert call_kwargs["tool_schema"] is None
    assert call_kwargs["tool_choice"] == "none"
    assert call_kwargs["response_format"] is ActionChoice
    assert call_kwargs["suppress_thinking"] is True


@pytest.mark.asyncio
async def test_achoose_action_invalid_output_makes_one_provider_request_without_tools(
    monkeypatch,
):
    calls = []

    async def _invalid_acompletion(**kwargs):
        calls.append(kwargs)
        return _action_choice_response("not valid action JSON")

    monkeypatch.setattr("mesa_llm.module_llm.acompletion", _invalid_acompletion)
    agent, _ = _make_local_action_choice_agent(llm_model="ollama_chat/llama3.2:3b")

    with pytest.raises(ValueError):
        await agent.achoose_action("Choose one async local action.")

    assert len(calls) == 1
    provider_kwargs = calls[0]
    assert provider_kwargs["tools"] is None
    assert provider_kwargs["tool_choice"] is None
    assert provider_kwargs["response_format"] is ActionChoice
    assert provider_kwargs["think"] is False
    assert agent.counter == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    _ACTION_SELECTION_TRANSPORT_ERRORS,
    ids=lambda error_type: error_type.__name__,
)
async def test_aact_transport_error_is_one_shot_and_mutation_free(
    monkeypatch,
    error_type,
):
    provider_calls = []
    original_error = _make_action_selection_transport_error(error_type)

    async def _raise_transport_error(**kwargs):
        await asyncio.sleep(0)
        provider_calls.append(kwargs)
        if len(provider_calls) > 1:
            raise AssertionError(
                "async action selection retried after a transport error"
            )
        raise original_error

    sync_completion = Mock(
        side_effect=AssertionError("async action selection called completion()")
    )
    monkeypatch.setattr("mesa_llm.module_llm.completion", sync_completion)
    monkeypatch.setattr(
        "mesa_llm.module_llm.acompletion",
        _raise_transport_error,
    )
    agent, _ = _make_local_action_choice_agent(llm_model="openai/gpt-4o")
    agent.recorder = Mock()
    agent.execute_action = Mock(
        side_effect=AssertionError("async action selection called execute_action()")
    )
    agent.aexecute_action = AsyncMock(
        side_effect=AssertionError("transport failure must prevent action execution")
    )
    state_before = _action_selection_state(agent)

    with pytest.raises(error_type) as exc_info:
        await agent.aact("Choose and execute one async local action.")

    assert len(provider_calls) == 1
    provider_kwargs = provider_calls[0]
    assert provider_kwargs["num_retries"] == 0
    assert provider_kwargs["max_retries"] == 0
    assert provider_kwargs["fallbacks"] == []
    assert provider_kwargs["tools"] is None
    assert provider_kwargs["tool_choice"] is None
    assert provider_kwargs["response_format"] is ActionChoice
    _assert_normalized_action_selection_transport_error(
        exc_info.value,
        error_type,
    )
    assert _action_selection_state(agent) == state_before
    sync_completion.assert_not_called()
    agent.execute_action.assert_not_called()
    agent.aexecute_action.assert_not_awaited()
    agent.recorder.record_event.assert_not_called()


@pytest.mark.asyncio
async def test_aact_extra_argument_fails_after_one_provider_request_before_mutation(
    monkeypatch,
):
    provider_calls = []

    async def _extra_argument_acompletion(**kwargs):
        provider_calls.append(kwargs)
        return _action_choice_response(
            json.dumps(
                {
                    "name": "local_increment_counter",
                    "arguments": {"amount": 1, "undeclared": "reject me"},
                    "rationale": "Attempt an invalid async local action.",
                }
            )
        )

    monkeypatch.setattr(
        "mesa_llm.module_llm.acompletion",
        _extra_argument_acompletion,
    )
    agent, _ = _make_local_action_choice_agent()
    agent.recorder = Mock()
    agent.aexecute_action = AsyncMock(
        side_effect=AssertionError("aact must not execute an invalid selection")
    )

    with pytest.raises(ValueError, match="Unexpected argument"):
        await agent.aact("Choose and execute one async local action.")

    assert len(provider_calls) == 1
    assert agent.counter == 0
    assert "action" not in agent.memory.step_content
    agent.aexecute_action.assert_not_awaited()
    agent.recorder.record_event.assert_not_called()


def test_choose_action_uses_structured_output_context_and_does_not_execute():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    def increment_counter(agent, amount: int) -> str:
        """Increment the counter.

        Args:
            amount: Amount to add.

        Returns:
            Mutation confirmation.
        """
        agent.counter += amount
        return "incremented"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[increment_counter],
    )
    agent.counter = 0
    agent.llm.generate = Mock(
        return_value=_action_choice_response(
            json.dumps(
                {
                    "name": "increment_counter",
                    "arguments": {"amount": "4"},
                    "rationale": "Need to update the counter.",
                },
            ),
        ),
    )

    choice = agent.choose_action(
        "Choose the next committed action.",
        system_prompt="system action prompt",
    )

    assert choice.name == "increment_counter"
    assert choice.arguments == {"amount": 4}
    assert choice.rationale == "Need to update the counter."
    assert agent.counter == 0

    agent.llm.generate.assert_called_once()
    call_kwargs = agent.llm.generate.call_args.kwargs
    assert call_kwargs["response_format"] is ActionChoice
    assert call_kwargs["tool_schema"] is None
    assert call_kwargs["tool_choice"] == "none"
    assert call_kwargs["system_prompt"] == "system action prompt"
    assert call_kwargs["suppress_thinking"] is True

    action_context = call_kwargs["prompt"][0]
    assert "Available actions:" in action_context
    assert (
        "The `arguments` object may contain only properties declared for "
        "the selected action."
    ) in action_context
    assert '"name": "increment_counter"' in action_context
    assert '"amount"' in action_context
    assert "Choose the next committed action." in call_kwargs["prompt"][1]


def test_choose_action_fails_fast_when_no_actions_are_available():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    no_action_agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=None,
    )
    no_action_agent.llm.generate = Mock()

    with pytest.raises(ValueError, match="No actions are available"):
        no_action_agent.choose_action("Choose an action.")

    no_action_agent.llm.generate.assert_not_called()

    configured_agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[wait],
    )
    configured_agent.llm.generate = Mock()

    for no_actions in [None, []]:
        with pytest.raises(ValueError, match="No actions are available"):
            configured_agent.choose_action("Choose an action.", actions=no_actions)

    configured_agent.llm.generate.assert_not_called()


def test_choose_action_respects_narrowed_actions_and_validates_returned_choice():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    def selected_action(agent, amount: int) -> str:
        """Selected action.

        Args:
            amount: Amount to add.

        Returns:
            Selection confirmation.
        """
        agent.selected += amount
        return "selected"

    @action(action_manager=ActionManager())
    def other_action(agent, amount: int) -> str:
        """Other action.

        Args:
            amount: Amount to add.

        Returns:
            Other confirmation.
        """
        agent.other += amount
        return "other"

    @action(action_manager=ActionManager())
    def unconfigured_action(agent) -> str:
        """Unconfigured action.

        Returns:
            Unconfigured confirmation.
        """
        agent.unconfigured += 1
        return "unconfigured"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[selected_action, other_action],
    )
    agent.selected = 0
    agent.other = 0
    agent.unconfigured = 0
    agent.llm.generate = Mock(
        return_value=_action_choice_response(
            json.dumps(
                {
                    "name": "selected_action",
                    "arguments": {"amount": 5},
                },
            ),
        ),
    )

    choice = agent.choose_action(
        "Choose from the narrowed action set.",
        actions=[selected_action],
    )

    assert choice.name == "selected_action"
    assert choice.arguments == {"amount": 5}
    action_context = agent.llm.generate.call_args.kwargs["prompt"][0]
    assert '"name": "selected_action"' in action_context
    assert '"name": "other_action"' not in action_context
    assert agent.selected == 0
    assert agent.other == 0

    agent.llm.generate = Mock(
        return_value=_action_choice_response(
            json.dumps(
                {
                    "name": "other_action",
                    "arguments": {"amount": 1},
                },
            ),
        ),
    )
    with pytest.raises(ValueError, match="Unknown action name"):
        agent.choose_action(
            "Choose from the narrowed action set.",
            actions=[selected_action],
        )

    assert agent.selected == 0
    assert agent.other == 0

    agent.llm.generate = Mock()
    with pytest.raises(ValueError, match="Unknown action name"):
        agent.choose_action(
            "Choose from an unconfigured action set.",
            actions=[unconfigured_action],
        )

    agent.llm.generate.assert_not_called()
    assert agent.unconfigured == 0


@pytest.mark.asyncio
async def test_achoose_action_uses_structured_output_context_and_does_not_execute():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    def increment_counter(agent, amount: int) -> str:
        """Increment the counter.

        Args:
            amount: Amount to add.

        Returns:
            Mutation confirmation.
        """
        agent.counter += amount
        return "incremented"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[increment_counter],
    )
    agent.counter = 0
    agent.llm.agenerate = AsyncMock(
        return_value=_action_choice_response(
            json.dumps(
                {
                    "name": "increment_counter",
                    "arguments": {"amount": "4"},
                    "rationale": "Need to update the counter.",
                },
            ),
        ),
    )

    choice = await agent.achoose_action(
        "Choose the next committed action.",
        system_prompt="async system action prompt",
    )

    assert choice.name == "increment_counter"
    assert choice.arguments == {"amount": 4}
    assert choice.rationale == "Need to update the counter."
    assert agent.counter == 0

    agent.llm.agenerate.assert_awaited_once()
    call_kwargs = agent.llm.agenerate.call_args.kwargs
    assert call_kwargs["response_format"] is ActionChoice
    assert call_kwargs["tool_schema"] is None
    assert call_kwargs["tool_choice"] == "none"
    assert call_kwargs["system_prompt"] == "async system action prompt"
    assert call_kwargs["suppress_thinking"] is True

    action_context = call_kwargs["prompt"][0]
    assert "Available actions:" in action_context
    assert (
        "The `arguments` object may contain only properties declared for "
        "the selected action."
    ) in action_context
    assert '"name": "increment_counter"' in action_context
    assert '"amount"' in action_context
    assert "Choose the next committed action." in call_kwargs["prompt"][1]


@pytest.mark.asyncio
async def test_achoose_action_fails_fast_when_no_actions_are_available():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    no_action_agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=None,
    )
    no_action_agent.llm.agenerate = AsyncMock()

    with pytest.raises(ValueError, match="No actions are available"):
        await no_action_agent.achoose_action("Choose an action.")

    no_action_agent.llm.agenerate.assert_not_called()

    configured_agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[wait],
    )
    configured_agent.llm.agenerate = AsyncMock()

    for no_actions in [None, []]:
        with pytest.raises(ValueError, match="No actions are available"):
            await configured_agent.achoose_action(
                "Choose an action.",
                actions=no_actions,
            )

    configured_agent.llm.agenerate.assert_not_called()


@pytest.mark.asyncio
async def test_achoose_action_respects_omitted_explicit_and_narrowed_actions():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    def selected_action(agent, amount: int) -> str:
        """Selected action.

        Args:
            amount: Amount to add.

        Returns:
            Selection confirmation.
        """
        agent.selected += amount
        return "selected"

    @action(action_manager=ActionManager())
    def other_action(agent, amount: int) -> str:
        """Other action.

        Args:
            amount: Amount to add.

        Returns:
            Other confirmation.
        """
        agent.other += amount
        return "other"

    @action(action_manager=ActionManager())
    def unconfigured_action(agent) -> str:
        """Unconfigured action.

        Returns:
            Unconfigured confirmation.
        """
        agent.unconfigured += 1
        return "unconfigured"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[selected_action, other_action],
    )
    agent.selected = 0
    agent.other = 0
    agent.unconfigured = 0
    agent.llm.agenerate = AsyncMock(
        return_value=_action_choice_response(
            json.dumps(
                {
                    "name": "other_action",
                    "arguments": {"amount": 2},
                },
            ),
        ),
    )

    omitted_choice = await agent.achoose_action("Use the configured action set.")

    assert omitted_choice.name == "other_action"
    assert omitted_choice.arguments == {"amount": 2}
    omitted_context = agent.llm.agenerate.call_args.kwargs["prompt"][0]
    assert '"name": "selected_action"' in omitted_context
    assert '"name": "other_action"' in omitted_context
    assert agent.selected == 0
    assert agent.other == 0

    for no_actions in [None, []]:
        agent.llm.agenerate = AsyncMock()
        with pytest.raises(ValueError, match="No actions are available"):
            await agent.achoose_action(
                "Explicitly disable actions.",
                actions=no_actions,
            )
        agent.llm.agenerate.assert_not_called()

    agent.llm.agenerate = AsyncMock(
        return_value=_action_choice_response(
            json.dumps(
                {
                    "name": "selected_action",
                    "arguments": {"amount": 5},
                },
            ),
        ),
    )

    narrowed_choice = await agent.achoose_action(
        "Choose from the narrowed action set.",
        actions=[selected_action],
    )

    assert narrowed_choice.name == "selected_action"
    assert narrowed_choice.arguments == {"amount": 5}
    narrowed_context = agent.llm.agenerate.call_args.kwargs["prompt"][0]
    assert '"name": "selected_action"' in narrowed_context
    assert '"name": "other_action"' not in narrowed_context
    assert agent.selected == 0
    assert agent.other == 0

    agent.llm.agenerate = AsyncMock(
        return_value=_action_choice_response(
            json.dumps(
                {
                    "name": "other_action",
                    "arguments": {"amount": 1},
                },
            ),
        ),
    )
    with pytest.raises(ValueError, match="Unknown action name"):
        await agent.achoose_action(
            "Choose from the narrowed action set.",
            actions=[selected_action],
        )

    agent.llm.agenerate = AsyncMock()
    with pytest.raises(ValueError, match="Unknown action name"):
        await agent.achoose_action(
            "Choose from an unconfigured action set.",
            actions=[unconfigured_action],
        )

    agent.llm.agenerate.assert_not_called()
    assert agent.selected == 0
    assert agent.other == 0
    assert agent.unconfigured == 0


def test_plan_delegates_to_reasoning_without_exposing_actions():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @tool
    def workflow_plan_tool(agent, value: int) -> int:
        """Workflow plan tool.

        Args:
            value: Value to return.

        Returns:
            The value.
        """
        del agent
        return value

    @action(action_manager=ActionManager())
    def workflow_action(agent) -> str:
        """Workflow action.

        Returns:
            Action result.
        """
        agent.executed = True
        return "executed"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        tools=[workflow_plan_tool],
        actions=[workflow_action],
    )
    expected_plan = Plan(step=3, llm_plan="planned", ttl=2, tools=[workflow_plan_tool])
    obs = object()
    agent.reasoning.plan = Mock(return_value=expected_plan)
    agent.choose_action = Mock()
    agent.execute_action = Mock()
    agent._action_manager.get_actions_schema = Mock(
        side_effect=AssertionError("plan() must not expose action specs")
    )

    result = agent.plan(
        prompt="Create a read-only plan.",
        obs=obs,
        ttl=2,
        tools=[workflow_plan_tool],
        tool_calls="required",
    )

    assert result is expected_plan
    agent.reasoning.plan.assert_called_once()
    plan_kwargs = agent.reasoning.plan.call_args.kwargs
    assert plan_kwargs["prompt"] == "Create a read-only plan."
    assert plan_kwargs["obs"] is obs
    assert plan_kwargs["ttl"] == 2
    assert plan_kwargs["tools"] == [workflow_plan_tool]
    assert plan_kwargs["tool_calls"] == "required"
    if "selected_tools" in plan_kwargs:
        assert plan_kwargs["selected_tools"] is _TOOLS_UNSET
    assert "actions" not in agent.reasoning.plan.call_args.kwargs
    agent.choose_action.assert_not_called()
    agent.execute_action.assert_not_called()
    agent._action_manager.get_actions_schema.assert_not_called()


def test_act_calls_public_wrappers_in_order_and_returns_act_result():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    agent = LLMAgent(DummyModel(), reasoning=ReActReasoning, actions=[wait])
    choice = ActionChoice(
        name="wait", arguments={}, rationale="No state change needed."
    )
    calls = []

    def fake_choose_action(
        prompt,
        actions=_ACTIONS_UNSET,
        system_prompt=None,
    ):
        calls.append(("choose_action", prompt, actions, system_prompt))
        return choice

    def fake_execute_action(action_choice, actions=_ACTIONS_UNSET):
        calls.append(("execute_action", action_choice, actions))
        return "waited"

    agent.plan = Mock(side_effect=AssertionError("act() must not call plan()"))
    agent.choose_action = fake_choose_action
    agent.execute_action = fake_execute_action

    result = agent.act("Take one turn.")

    assert [call[0] for call in calls] == ["choose_action", "execute_action"]
    assert calls[0] == (
        "choose_action",
        "Take one turn.",
        _ACTIONS_UNSET,
        None,
    )
    assert calls[1] == ("execute_action", choice, _ACTIONS_UNSET)
    agent.plan.assert_not_called()
    assert result.__class__.__name__ == "ActResult"
    assert result.action is choice
    assert result.result == "waited"
    assert not hasattr(result, "plan")
    assert not hasattr(result, "success")


def test_act_preserves_explicit_action_selector():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    agent = LLMAgent(DummyModel(), reasoning=ReActReasoning, actions=[wait])
    choice = ActionChoice(name="wait", arguments={})
    calls = []

    def fake_choose_action(
        prompt,
        actions=_ACTIONS_UNSET,
        system_prompt=None,
    ):
        calls.append(("choose_action", actions))
        return choice

    def fake_execute_action(action_choice, actions=_ACTIONS_UNSET):
        calls.append(("execute_action", actions))
        return "waited"

    agent.plan = Mock(side_effect=AssertionError("act() must not call plan()"))
    agent.choose_action = fake_choose_action
    agent.execute_action = fake_execute_action

    result = agent.act("Take one turn.", actions=[wait])

    assert result.result == "waited"
    assert calls == [
        ("choose_action", [wait]),
        ("execute_action", [wait]),
    ]
    agent.plan.assert_not_called()


def test_plan_then_act_composition_passes_plan_through_prompt():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    agent = LLMAgent(DummyModel(), reasoning=ReActReasoning, actions=[wait])
    plan = Plan(step=1, llm_plan="planned")
    choice = ActionChoice(name="wait", arguments={})
    agent.plan = Mock(side_effect=AssertionError("act() must not call plan()"))
    agent.choose_action = Mock(return_value=choice)
    agent.execute_action = Mock(return_value="waited")

    prompt = f"Use this plan: {plan}"
    result = agent.act(prompt=prompt)

    assert result.action is choice
    assert result.result == "waited"
    agent.plan.assert_not_called()
    agent.choose_action.assert_called_once_with(
        prompt,
        actions=_ACTIONS_UNSET,
        system_prompt=None,
    )
    agent.execute_action.assert_called_once_with(choice, actions=_ACTIONS_UNSET)


def test_act_fails_fast_when_explicit_actions_expose_no_actions():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    agent = LLMAgent(DummyModel(), reasoning=ReActReasoning, actions=[wait])
    agent.plan = Mock(side_effect=AssertionError("act() must not call plan()"))
    agent.execute_action = Mock()
    agent.llm.generate = Mock()

    for no_actions in [None, []]:
        with pytest.raises(ValueError, match="No actions are available"):
            agent.act("Take one turn.", actions=no_actions)

    agent.plan.assert_not_called()
    agent.llm.generate.assert_not_called()
    agent.execute_action.assert_not_called()


@pytest.mark.asyncio
async def test_aact_awaits_action_choice_executes_once_and_returns_act_result():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    agent = LLMAgent(DummyModel(), reasoning=ReActReasoning, actions=[wait])
    choice = ActionChoice(
        name="wait", arguments={}, rationale="No state change needed."
    )
    agent.plan = Mock(side_effect=AssertionError("aact() must not call plan()"))
    agent.reasoning.aplan = AsyncMock(
        side_effect=AssertionError("aact() must not call aplan()")
    )
    agent.achoose_action = AsyncMock(return_value=choice)
    agent.execute_action = Mock(
        side_effect=AssertionError("aact() must not call sync execute_action()")
    )
    agent.aexecute_action = AsyncMock(return_value="waited")

    result = await agent.aact(
        "Take one async turn.",
        actions=[wait],
        system_prompt="async action system prompt",
    )

    agent.achoose_action.assert_awaited_once_with(
        "Take one async turn.",
        actions=[wait],
        system_prompt="async action system prompt",
    )
    agent.aexecute_action.assert_awaited_once_with(choice, actions=[wait])
    agent.execute_action.assert_not_called()
    agent.plan.assert_not_called()
    agent.reasoning.aplan.assert_not_called()
    assert result.__class__.__name__ == "ActResult"
    assert result.action is choice
    assert result.result == "waited"
    assert not hasattr(result, "plan")
    assert not hasattr(result, "success")


@pytest.mark.asyncio
async def test_aact_records_successful_execution_and_does_not_record_failures():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    async def recorded_async_action(agent, amount: int) -> str:
        """Recorded async action.

        Args:
            amount: Amount to add.

        Returns:
            Execution result.
        """
        await asyncio.sleep(0)
        assert "action" not in agent.memory.step_content
        agent.counter += amount
        agent.awaited = True
        return "recorded"

    successful_agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[recorded_async_action],
    )
    successful_agent.memory = ShortTermMemory(
        agent=successful_agent,
        n=5,
        display=False,
    )
    successful_agent.recorder = Mock()
    successful_agent.counter = 0
    successful_agent.awaited = False
    successful_agent.llm.agenerate = AsyncMock(
        return_value=_action_choice_response(
            json.dumps(
                {
                    "name": "recorded_async_action",
                    "arguments": {"amount": 3},
                },
            ),
        ),
    )

    result = await successful_agent.aact("Take one recorded async action.")

    assert result.action.name == "recorded_async_action"
    assert result.action.arguments == {"amount": 3}
    assert result.result == "recorded"
    assert successful_agent.counter == 3
    assert successful_agent.awaited is True
    expected_content = {
        "action": {
            "name": "recorded_async_action",
            "arguments": {"amount": 3},
            "rationale": None,
        },
        "result": "recorded",
    }
    actions = successful_agent.memory.step_content["action"]
    assert actions == [expected_content]
    successful_agent.recorder.record_event.assert_called_once_with(
        "action",
        content=expected_content,
        agent_id=successful_agent.unique_id,
        metadata={"source": "LLMAgent.execute_action"},
    )

    @action(action_manager=ActionManager())
    async def failing_async_action(agent, amount: int) -> str:
        """Failing async action.

        Args:
            amount: Amount to add.

        Returns:
            Never returns.
        """
        await asyncio.sleep(0)
        agent.started = True
        del amount
        raise RuntimeError("async action failed")

    failing_agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[failing_async_action],
    )
    failing_agent.memory = ShortTermMemory(agent=failing_agent, n=5, display=False)
    failing_agent.recorder = Mock()
    failing_agent.started = False
    failing_agent.llm.agenerate = AsyncMock(
        return_value=_action_choice_response(
            json.dumps(
                {
                    "name": "failing_async_action",
                    "arguments": {"amount": 1},
                },
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="async action failed"):
        await failing_agent.aact("Take one failing async action.")

    assert failing_agent.started is True
    assert "action" not in failing_agent.memory.step_content
    failing_agent.recorder.record_event.assert_not_called()


class _ObserverAbort(BaseException):
    """Sentinel proving observer boundaries catch Exception, not BaseException."""


def _make_sync_post_commit_agent(result):
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    def pd03_sync_action(agent, amount: int) -> object:
        """Execute one observable synchronous mutation.

        Args:
            amount: Validated amount.

        Returns:
            The committed result sentinel.
        """
        agent.action_calls += 1
        agent.timeline.append("action")
        return result

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[pd03_sync_action],
    )
    agent.action_calls = 0
    agent.timeline = []
    choice = ActionChoice(
        name="pd03_sync_action",
        arguments={"amount": "2"},
        rationale="Exercise post-commit observers.",
    )
    return agent, choice


def _make_async_post_commit_agent(result):
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    async def pd03_async_action(agent, amount: int) -> object:
        """Execute one observable asynchronous mutation.

        Args:
            amount: Validated amount.

        Returns:
            The committed result sentinel.
        """
        agent.action_calls += 1
        await asyncio.sleep(0)
        agent.action_await_completions += 1
        agent.timeline.append("action")
        return result

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[pd03_async_action],
    )
    agent.action_calls = 0
    agent.action_await_completions = 0
    agent.timeline = []
    choice = ActionChoice(
        name="pd03_async_action",
        arguments={"amount": "2"},
        rationale="Exercise asynchronous post-commit observers.",
    )
    return agent, choice


def _install_sync_post_commit_observers(
    agent,
    *,
    memory_error=None,
    recorder_error=None,
    recorder_present=True,
):
    def observe_memory(*args, **kwargs):
        agent.timeline.append("memory")
        if memory_error is not None:
            raise memory_error

    def observe_recorder(*args, **kwargs):
        agent.timeline.append("recorder")
        if recorder_error is not None:
            raise recorder_error

    agent.memory = SimpleNamespace(
        add_to_memory=Mock(side_effect=observe_memory),
    )
    agent.recorder = (
        SimpleNamespace(record_event=Mock(side_effect=observe_recorder))
        if recorder_present
        else None
    )


def _install_async_post_commit_observers(
    agent,
    *,
    memory_error=None,
    recorder_error=None,
    recorder_present=True,
):
    async def observe_memory(*args, **kwargs):
        agent.timeline.append("memory")
        await asyncio.sleep(0)
        if memory_error is not None:
            raise memory_error

    def observe_recorder(*args, **kwargs):
        agent.timeline.append("recorder")
        if recorder_error is not None:
            raise recorder_error

    agent.memory = SimpleNamespace(
        add_to_memory=Mock(
            side_effect=AssertionError("async execution must not use sync memory")
        ),
        aadd_to_memory=AsyncMock(side_effect=observe_memory),
    )
    agent.recorder = (
        SimpleNamespace(record_event=Mock(side_effect=observe_recorder))
        if recorder_present
        else None
    )


def test_action_post_commit_error_has_established_public_exports():
    assert action_exports.ActionPostCommitError is ActionPostCommitError
    assert RootActionPostCommitError is ActionPostCommitError


def test_action_post_commit_error_rejects_empty_observer_errors():
    action_choice = ActionChoice(name="constructor_action", arguments={})

    with pytest.raises(ValueError, match="at least one error"):
        ActionPostCommitError(action_choice, object(), {})


def test_action_post_commit_error_rejects_unsupported_observer_key():
    action_choice = ActionChoice(name="constructor_action", arguments={})

    with pytest.raises(ValueError, match="unsupported key"):
        ActionPostCommitError(
            action_choice,
            object(),
            {"cache": RuntimeError("cache failed")},
        )


@pytest.mark.parametrize(
    "invalid_error",
    [
        pytest.param("not an exception", id="non-exception"),
        pytest.param(_ObserverAbort("observer aborted"), id="base-exception-only"),
    ],
)
def test_action_post_commit_error_rejects_non_exception_values(invalid_error):
    action_choice = ActionChoice(name="constructor_action", arguments={})

    with pytest.raises(TypeError, match="Exception instance"):
        ActionPostCommitError(
            action_choice,
            object(),
            {"memory": invalid_error},
        )


def test_action_post_commit_error_copies_observer_mapping_shallowly():
    action_choice = ActionChoice(name="constructor_action", arguments={})
    memory_error = RuntimeError("memory failed")
    observer_errors = {"memory": memory_error}

    error = ActionPostCommitError(action_choice, object(), observer_errors)

    assert error.observer_errors is not observer_errors
    assert error.observer_errors["memory"] is memory_error

    observer_errors["memory"] = RuntimeError("replacement memory failure")
    observer_errors["recorder"] = ValueError("late recorder failure")

    assert set(error.observer_errors) == {"memory"}
    assert error.observer_errors["memory"] is memory_error


def test_action_post_commit_error_preserves_valid_dual_mapping_and_message():
    action_choice = ActionChoice(
        name="constructor_action",
        arguments={"amount": 2},
        rationale="Exercise constructor validation.",
    )
    result = object()
    memory_error = RuntimeError("memory failed")
    recorder_error = ValueError("recorder failed")
    observer_errors = {
        "memory": memory_error,
        "recorder": recorder_error,
    }

    error = ActionPostCommitError(action_choice, result, observer_errors)

    assert error.committed is True
    assert error.action is action_choice
    assert error.result is result
    assert error.observer_errors is not observer_errors
    assert error.observer_errors["memory"] is memory_error
    assert error.observer_errors["recorder"] is recorder_error
    assert str(error) == (
        "Action 'constructor_action' committed, but post-commit observer(s) "
        "failed: memory, recorder."
    )


@pytest.mark.parametrize(
    ("memory_fails", "recorder_fails"),
    [
        pytest.param(True, False, id="memory"),
        pytest.param(False, True, id="recorder"),
        pytest.param(True, True, id="both"),
    ],
)
def test_execute_action_reports_post_commit_observer_exceptions(
    memory_fails,
    recorder_fails,
):
    result = object()
    memory_error = RuntimeError("memory observer failed")
    recorder_error = ValueError("recorder observer failed")
    agent, choice = _make_sync_post_commit_agent(result)
    _install_sync_post_commit_observers(
        agent,
        memory_error=memory_error if memory_fails else None,
        recorder_error=recorder_error if recorder_fails else None,
    )

    with pytest.raises(ActionPostCommitError) as exc_info:
        agent.execute_action(choice)

    error = exc_info.value
    assert error.committed is True
    assert isinstance(error.action, ActionChoice)
    assert error.action.name == choice.name
    assert error.action.arguments == {"amount": 2}
    assert error.action.rationale == choice.rationale
    assert error.result is result
    expected_keys = {
        name
        for name, failed in (
            ("memory", memory_fails),
            ("recorder", recorder_fails),
        )
        if failed
    }
    assert isinstance(error.observer_errors, dict)
    assert set(error.observer_errors) == expected_keys
    if memory_fails:
        assert error.observer_errors["memory"] is memory_error
    if recorder_fails:
        assert error.observer_errors["recorder"] is recorder_error

    assert agent.action_calls == 1
    assert agent.timeline == ["action", "memory", "recorder"]
    agent.memory.add_to_memory.assert_called_once()
    agent.recorder.record_event.assert_called_once()
    assert agent.memory.add_to_memory.call_args.kwargs["content"]["result"] is result
    assert agent.recorder.record_event.call_args.kwargs["content"]["result"] is result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("memory_fails", "recorder_fails"),
    [
        pytest.param(True, False, id="memory"),
        pytest.param(False, True, id="recorder"),
        pytest.param(True, True, id="both"),
    ],
)
async def test_aexecute_action_reports_post_commit_observer_exceptions(
    memory_fails,
    recorder_fails,
):
    result = object()
    memory_error = RuntimeError("async memory observer failed")
    recorder_error = ValueError("async recorder observer failed")
    agent, choice = _make_async_post_commit_agent(result)
    _install_async_post_commit_observers(
        agent,
        memory_error=memory_error if memory_fails else None,
        recorder_error=recorder_error if recorder_fails else None,
    )

    with pytest.raises(ActionPostCommitError) as exc_info:
        await agent.aexecute_action(choice)

    error = exc_info.value
    assert error.committed is True
    assert isinstance(error.action, ActionChoice)
    assert error.action.name == choice.name
    assert error.action.arguments == {"amount": 2}
    assert error.action.rationale == choice.rationale
    assert error.result is result
    expected_keys = {
        name
        for name, failed in (
            ("memory", memory_fails),
            ("recorder", recorder_fails),
        )
        if failed
    }
    assert isinstance(error.observer_errors, dict)
    assert set(error.observer_errors) == expected_keys
    if memory_fails:
        assert error.observer_errors["memory"] is memory_error
    if recorder_fails:
        assert error.observer_errors["recorder"] is recorder_error

    assert agent.action_calls == 1
    assert agent.action_await_completions == 1
    assert agent.timeline == ["action", "memory", "recorder"]
    agent.memory.aadd_to_memory.assert_awaited_once()
    agent.memory.add_to_memory.assert_not_called()
    agent.recorder.record_event.assert_called_once()
    assert agent.memory.aadd_to_memory.call_args.kwargs["content"]["result"] is result
    assert agent.recorder.record_event.call_args.kwargs["content"]["result"] is result


def test_execute_action_succeeds_without_optional_recorder():
    result = object()
    agent, choice = _make_sync_post_commit_agent(result)
    _install_sync_post_commit_observers(agent, recorder_present=False)

    returned_result = agent.execute_action(choice)

    assert returned_result is result
    assert agent.action_calls == 1
    assert agent.timeline == ["action", "memory"]
    agent.memory.add_to_memory.assert_called_once()
    assert agent.recorder is None


@pytest.mark.asyncio
async def test_aexecute_action_succeeds_without_optional_recorder():
    result = object()
    agent, choice = _make_async_post_commit_agent(result)
    _install_async_post_commit_observers(agent, recorder_present=False)

    returned_result = await agent.aexecute_action(choice)

    assert returned_result is result
    assert agent.action_calls == 1
    assert agent.action_await_completions == 1
    assert agent.timeline == ["action", "memory"]
    agent.memory.aadd_to_memory.assert_awaited_once()
    agent.memory.add_to_memory.assert_not_called()
    assert agent.recorder is None


def test_execute_action_preserves_pre_commit_failures_without_observers():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    action_error = RuntimeError("sync action body failed")

    @action(action_manager=ActionManager())
    def pd03_failing_sync_action(agent, amount: int) -> None:
        """Fail during synchronous action execution.

        Args:
            amount: Required amount.
        """
        agent.action_calls += 1
        raise action_error

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[pd03_failing_sync_action],
    )
    agent.action_calls = 0
    agent.memory = SimpleNamespace(add_to_memory=Mock())
    agent.recorder = SimpleNamespace(record_event=Mock())

    with pytest.raises(ValueError, match="Missing required argument"):
        agent.execute_action(
            ActionChoice(name="pd03_failing_sync_action", arguments={})
        )

    assert agent.action_calls == 0
    agent.memory.add_to_memory.assert_not_called()
    agent.recorder.record_event.assert_not_called()

    with pytest.raises(RuntimeError) as exc_info:
        agent.execute_action(
            ActionChoice(
                name="pd03_failing_sync_action",
                arguments={"amount": 1},
            )
        )

    assert exc_info.value is action_error
    assert agent.action_calls == 1
    agent.memory.add_to_memory.assert_not_called()
    agent.recorder.record_event.assert_not_called()


@pytest.mark.asyncio
async def test_aexecute_action_preserves_pre_commit_failures_without_observers():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    action_error = RuntimeError("async action body failed")

    @action(action_manager=ActionManager())
    async def pd03_failing_async_action(agent, amount: int) -> None:
        """Fail during asynchronous action execution.

        Args:
            amount: Required amount.
        """
        agent.action_calls += 1
        await asyncio.sleep(0)
        agent.action_await_completions += 1
        raise action_error

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[pd03_failing_async_action],
    )
    agent.action_calls = 0
    agent.action_await_completions = 0
    agent.memory = SimpleNamespace(
        add_to_memory=Mock(),
        aadd_to_memory=AsyncMock(),
    )
    agent.recorder = SimpleNamespace(record_event=Mock())

    with pytest.raises(ValueError, match="Missing required argument"):
        await agent.aexecute_action(
            ActionChoice(name="pd03_failing_async_action", arguments={})
        )

    assert agent.action_calls == 0
    assert agent.action_await_completions == 0
    agent.memory.aadd_to_memory.assert_not_awaited()
    agent.memory.add_to_memory.assert_not_called()
    agent.recorder.record_event.assert_not_called()

    with pytest.raises(RuntimeError) as exc_info:
        await agent.aexecute_action(
            ActionChoice(
                name="pd03_failing_async_action",
                arguments={"amount": 1},
            )
        )

    assert exc_info.value is action_error
    assert agent.action_calls == 1
    assert agent.action_await_completions == 1
    agent.memory.aadd_to_memory.assert_not_awaited()
    agent.memory.add_to_memory.assert_not_called()
    agent.recorder.record_event.assert_not_called()


@pytest.mark.parametrize("failing_observer", ["memory", "recorder"])
def test_execute_action_does_not_convert_observer_base_exceptions(failing_observer):
    result = object()
    observer_abort = _ObserverAbort(f"sync {failing_observer} aborted")
    agent, choice = _make_sync_post_commit_agent(result)
    _install_sync_post_commit_observers(
        agent,
        memory_error=observer_abort if failing_observer == "memory" else None,
        recorder_error=observer_abort if failing_observer == "recorder" else None,
    )

    with pytest.raises(_ObserverAbort) as exc_info:
        agent.execute_action(choice)

    assert exc_info.value is observer_abort
    assert agent.action_calls == 1
    agent.memory.add_to_memory.assert_called_once()
    if failing_observer == "recorder":
        agent.recorder.record_event.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_observer", ["memory", "recorder"])
async def test_aexecute_action_does_not_convert_observer_base_exceptions(
    failing_observer,
):
    result = object()
    observer_abort = _ObserverAbort(f"async {failing_observer} aborted")
    agent, choice = _make_async_post_commit_agent(result)
    _install_async_post_commit_observers(
        agent,
        memory_error=observer_abort if failing_observer == "memory" else None,
        recorder_error=observer_abort if failing_observer == "recorder" else None,
    )

    with pytest.raises(_ObserverAbort) as exc_info:
        await agent.aexecute_action(choice)

    assert exc_info.value is observer_abort
    assert agent.action_calls == 1
    assert agent.action_await_completions == 1
    agent.memory.aadd_to_memory.assert_awaited_once()
    agent.memory.add_to_memory.assert_not_called()
    if failing_observer == "recorder":
        agent.recorder.record_event.assert_called_once()


@pytest.mark.asyncio
async def test_aexecute_action_validates_before_async_execution_and_recording():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    async def async_recorded_action(agent, amount: int) -> str:
        """Recorded async action.

        Args:
            amount: Amount to add.

        Returns:
            Execution result.
        """
        await asyncio.sleep(0)
        agent.counter += amount
        agent.started = True
        return "recorded"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[async_recorded_action],
    )
    agent.memory = ShortTermMemory(agent=agent, n=5, display=False)
    agent.recorder = Mock()
    agent.counter = 0
    agent.started = False

    with pytest.raises(ValueError, match="Missing required argument"):
        await agent.aexecute_action(
            ActionChoice(name="async_recorded_action", arguments={}),
        )

    assert agent.counter == 0
    assert agent.started is False
    assert "action" not in agent.memory.step_content
    agent.recorder.record_event.assert_not_called()


@pytest.mark.asyncio
async def test_aexecute_action_awaits_memory_before_recording_success():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    timeline = []

    @action(action_manager=ActionManager())
    async def ordered_async_action(agent, amount: int) -> dict[str, int]:
        """Apply an amount after yielding to the event loop.

        Args:
            amount: Amount to apply.

        Returns:
            Applied amount.
        """
        await asyncio.sleep(0)
        agent.counter += amount
        timeline.append("action-complete")
        return {"applied": amount}

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[ordered_async_action],
    )
    agent.counter = 0
    choice = ActionChoice(
        name="ordered_async_action",
        arguments={"amount": 3},
        rationale="Apply the requested amount.",
    )
    expected_result = {"applied": 3}
    expected_content = {
        "action": choice.model_dump(),
        "result": expected_result,
    }

    async def record_memory(*, type, content):
        assert type == "action"
        assert content == expected_content
        assert agent.counter == 3
        timeline.append("memory-start")
        await asyncio.sleep(0)
        timeline.append("memory-complete")

    agent.memory = SimpleNamespace(
        add_to_memory=Mock(
            side_effect=AssertionError("async execution must not record synchronously")
        ),
        aadd_to_memory=AsyncMock(side_effect=record_memory),
    )
    agent.recorder = Mock()
    agent.recorder.record_event.side_effect = lambda *args, **kwargs: timeline.append(
        "recorder"
    )

    result = await agent.aexecute_action(choice)

    assert result == expected_result
    assert timeline == [
        "action-complete",
        "memory-start",
        "memory-complete",
        "recorder",
    ]
    agent.memory.aadd_to_memory.assert_awaited_once_with(
        type="action",
        content=expected_content,
    )
    agent.memory.add_to_memory.assert_not_called()
    agent.recorder.record_event.assert_called_once_with(
        "action",
        content=expected_content,
        agent_id=agent.unique_id,
        metadata={"source": "LLMAgent.execute_action"},
    )


@pytest.mark.asyncio
async def test_aexecute_action_does_not_record_validation_or_execution_failures():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    async def failing_async_action(agent, amount: int) -> None:
        """Raise after asynchronous execution begins.

        Args:
            amount: Required amount.
        """
        await asyncio.sleep(0)
        agent.started = True
        del amount
        raise RuntimeError("action failed")

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[failing_async_action],
    )
    agent.started = False
    agent.memory = SimpleNamespace(
        add_to_memory=Mock(),
        aadd_to_memory=AsyncMock(),
    )
    agent.recorder = Mock()

    with pytest.raises(ValueError, match="Missing required argument"):
        await agent.aexecute_action(
            ActionChoice(name="failing_async_action", arguments={})
        )

    assert agent.started is False
    agent.memory.aadd_to_memory.assert_not_awaited()
    agent.memory.add_to_memory.assert_not_called()
    agent.recorder.record_event.assert_not_called()

    with pytest.raises(RuntimeError, match="action failed"):
        await agent.aexecute_action(
            ActionChoice(name="failing_async_action", arguments={"amount": 1})
        )

    assert agent.started is True
    agent.memory.aadd_to_memory.assert_not_awaited()
    agent.memory.add_to_memory.assert_not_called()
    agent.recorder.record_event.assert_not_called()


def test_execute_action_keeps_synchronous_memory_recording():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    def synchronous_recording_action(agent, amount: int) -> str:
        """Apply an amount synchronously.

        Args:
            amount: Amount to apply.

        Returns:
            Completion status.
        """
        agent.counter += amount
        return "recorded"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[synchronous_recording_action],
    )
    agent.counter = 0
    choice = ActionChoice(
        name="synchronous_recording_action",
        arguments={"amount": 2},
    )
    expected_content = {
        "action": choice.model_dump(),
        "result": "recorded",
    }
    agent.memory = SimpleNamespace(
        add_to_memory=Mock(),
        aadd_to_memory=AsyncMock(
            side_effect=AssertionError("sync execution must not use async memory")
        ),
    )
    agent.recorder = None

    result = agent.execute_action(choice)

    assert result == "recorded"
    assert agent.counter == 2
    agent.memory.add_to_memory.assert_called_once_with(
        type="action",
        content=expected_content,
    )
    agent.memory.aadd_to_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_aexecute_action_uses_episodic_memory_async_importance_grading():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    async def episodic_async_action(agent) -> str:
        """Return after yielding to the event loop."""
        del agent
        await asyncio.sleep(0)
        return "remembered"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[episodic_async_action],
    )
    memory = EpisodicMemory(agent=agent, llm_model="provider/test_model")
    memory.agrade_event_importance = AsyncMock(return_value=4)
    memory.grade_event_importance = Mock(
        side_effect=AssertionError("async recording must not grade synchronously")
    )
    agent.memory = memory
    agent.recorder = None
    choice = ActionChoice(name="episodic_async_action", arguments={})
    expected_content = {
        "action": choice.model_dump(),
        "result": "remembered",
    }

    result = await agent.aexecute_action(choice)

    assert result == "remembered"
    memory.agrade_event_importance.assert_awaited_once_with(
        "action",
        expected_content,
    )
    memory.grade_event_importance.assert_not_called()
    assert len(memory.memory_entries) == 1
    assert memory.memory_entries[0].content == {
        "action": {
            **expected_content,
            "importance": 4,
        }
    }


def test_execute_action_records_successful_action_event_after_execution():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    def recorded_action(agent, amount: int) -> str:
        """Recorded action.

        Args:
            amount: Amount to add.

        Returns:
            Execution result.
        """
        agent.counter += amount
        return "recorded"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[recorded_action],
    )
    agent.memory = ShortTermMemory(agent=agent, n=5, display=False)
    agent.recorder = Mock()
    agent.counter = 0

    result = agent.execute_action(
        ActionChoice(name="recorded_action", arguments={"amount": 3}),
    )

    assert result == "recorded"
    assert agent.counter == 3
    expected_content = {
        "action": {
            "name": "recorded_action",
            "arguments": {"amount": 3},
            "rationale": None,
        },
        "result": "recorded",
    }
    actions = agent.memory.step_content["action"]
    assert actions == [expected_content]
    agent.recorder.record_event.assert_called_once_with(
        "action",
        content=expected_content,
        agent_id=agent.unique_id,
        metadata={"source": "LLMAgent.execute_action"},
    )


def test_execute_action_does_not_record_successful_event_for_failures():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    def failing_action(agent, amount: int) -> str:
        """Failing action.

        Args:
            amount: Amount to add.

        Returns:
            Never returns.
        """
        del agent, amount
        raise RuntimeError("action failed")

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[failing_action],
    )
    agent.memory = ShortTermMemory(agent=agent, n=5, display=False)
    agent.recorder = Mock()

    with pytest.raises(ValueError, match="Missing required argument"):
        agent.execute_action(ActionChoice(name="failing_action", arguments={}))

    assert "action" not in agent.memory.step_content
    agent.recorder.record_event.assert_not_called()

    with pytest.raises(RuntimeError, match="action failed"):
        agent.execute_action(
            ActionChoice(name="failing_action", arguments={"amount": 1}),
        )

    assert "action" not in agent.memory.step_content
    agent.recorder.record_event.assert_not_called()


@pytest.mark.asyncio
async def test_execute_action_rejects_async_action_without_recording_and_aexecute_records():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    async def async_recorded_action(agent, amount: int) -> str:
        """Recorded async action.

        Args:
            amount: Amount to add.

        Returns:
            Execution result.
        """
        agent.started = True
        await asyncio.sleep(0)
        agent.counter += amount
        return "recorded"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[async_recorded_action],
    )
    agent.memory = ShortTermMemory(agent=agent, n=5, display=False)
    agent.recorder = Mock()
    agent.counter = 0
    agent.started = False

    choice = ActionChoice(
        name="async_recorded_action",
        arguments={"amount": 3},
    )

    _assert_rejects_async_action_without_unawaited_warning(
        lambda: agent.execute_action(choice),
    )

    assert agent.counter == 0
    assert agent.started is False
    assert "action" not in agent.memory.step_content
    agent.recorder.record_event.assert_not_called()

    result = await agent.aexecute_action(choice)

    assert result == "recorded"
    assert agent.counter == 3
    assert agent.started is True
    expected_content = {
        "action": {
            "name": "async_recorded_action",
            "arguments": {"amount": 3},
            "rationale": None,
        },
        "result": "recorded",
    }
    assert agent.memory.step_content["action"] == [expected_content]
    agent.recorder.record_event.assert_called_once_with(
        "action",
        content=expected_content,
        agent_id=agent.unique_id,
        metadata={"source": "LLMAgent.execute_action"},
    )


def test_act_rejects_async_action_without_recording_success_or_warning():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @action(action_manager=ActionManager())
    async def async_act_action(agent, amount: int) -> str:
        """Async act action.

        Args:
            amount: Amount to add.

        Returns:
            Execution result.
        """
        agent.started = True
        await asyncio.sleep(0)
        agent.counter += amount
        return "acted"

    agent = LLMAgent(
        DummyModel(),
        reasoning=ReActReasoning,
        actions=[async_act_action],
    )
    agent.memory = ShortTermMemory(agent=agent, n=5, display=False)
    agent.recorder = Mock()
    agent.counter = 0
    agent.started = False
    agent.llm.generate = Mock(
        return_value=_action_choice_response(
            json.dumps(
                {
                    "name": "async_act_action",
                    "arguments": {"amount": 2},
                },
            ),
        ),
    )

    _assert_rejects_async_action_without_unawaited_warning(
        lambda: agent.act("Take one async action."),
    )

    assert agent.counter == 0
    assert agent.started is False
    assert "action" not in agent.memory.step_content
    agent.recorder.record_event.assert_not_called()
    agent.llm.generate.assert_called_once()


def test_llm_agent_tool_manager_property_is_deprecated():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    agent = LLMAgent(DummyModel(), reasoning=ReActReasoning)
    replacement = ToolManager()

    with pytest.warns(DeprecationWarning, match="agent.tool_manager"):
        assert agent.tool_manager is agent._tool_manager

    with pytest.warns(DeprecationWarning, match="agent.tool_manager"):
        agent.tool_manager = replacement

    assert agent._tool_manager is replacement


def test_apply_plan_executes_per_call_tool_selector():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    @tool
    def override_apply_tool(agent, value: int) -> str:
        """Override apply tool.
        Args:
            agent: The agent making the request (provided automatically)
            value: Input.
        Returns:
            Output.
        """
        return f"{agent.unique_id}:{value}"

    model = DummyModel()
    agent = LLMAgent(model, reasoning=ReActReasoning, tools=[override_apply_tool])
    agent.memory = Mock()

    mock_tool_call = Mock()
    mock_tool_call.id = "call_override"
    mock_tool_call.function.name = "override_apply_tool"
    mock_tool_call.function.arguments = '{"value": "7"}'

    mock_message = Mock()
    mock_message.tool_calls = [mock_tool_call]

    plan = Plan(step=0, llm_plan=mock_message, tools=[override_apply_tool])

    result = agent.apply_plan(plan)

    assert result == [
        {
            "tool_call_id": "call_override",
            "role": "tool",
            "name": "override_apply_tool",
            "response": f"{agent.unique_id}:7",
        }
    ]


def test_apply_plan_preserves_multiple_tool_calls(monkeypatch):
    """All tool call results must be preserved when the LLM returns >1 tool call."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.grid = MultiGrid(5, 5, torus=False)

    model = DummyModel()
    agent = LLMAgent.create_agents(
        model,
        n=1,
        reasoning=ReActReasoning,
        system_prompt="test",
        vision=-1,
        internal_state=["test_state"],
    ).to_list()[0]
    model.grid.place_agent(agent, (1, 1))
    agent.memory = ShortTermMemory(agent=agent, n=5, display=False)

    fake_response = [
        {
            "tool_call_id": "1",
            "role": "tool",
            "name": "move_one_step",
            "response": "agent moved to (3, 4)",
        },
        {
            "tool_call_id": "2",
            "role": "tool",
            "name": "arrest_citizen",
            "response": "Citizen 12 arrested",
        },
    ]
    monkeypatch.setattr(
        agent.tool_manager, "call_tools", lambda agent, llm_response: fake_response
    )

    plan = Plan(step=0, llm_plan="do something")
    agent.apply_plan(plan)

    # "action" is an additive event type, so it is stored as a list
    actions = agent.memory.step_content.get("action")
    assert actions is not None
    assert isinstance(actions, list) and len(actions) == 1
    assert "tool_calls" in actions[0]
    assert len(actions[0]["tool_calls"]) == 2
    assert actions[0]["tool_calls"][0] == {
        "name": "move_one_step",
        "response": "agent moved to (3, 4)",
    }
    assert actions[0]["tool_calls"][1] == {
        "name": "arrest_citizen",
        "response": "Citizen 12 arrested",
    }


@pytest.mark.asyncio
async def test_aapply_plan_preserves_multiple_tool_calls(monkeypatch):
    """Async variant: all tool call results must be preserved."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.grid = MultiGrid(5, 5, torus=False)

    model = DummyModel()
    agent = LLMAgent.create_agents(
        model,
        n=1,
        reasoning=ReActReasoning,
        system_prompt="test",
        vision=-1,
        internal_state=["test_state"],
    ).to_list()[0]
    model.grid.place_agent(agent, (1, 1))
    agent.memory = ShortTermMemory(agent=agent, n=5, display=False)

    fake_response = [
        {
            "tool_call_id": "1",
            "role": "tool",
            "name": "move_one_step",
            "response": "agent moved to (3, 4)",
        },
        {
            "tool_call_id": "2",
            "role": "tool",
            "name": "arrest_citizen",
            "response": "Citizen 12 arrested",
        },
    ]

    async def fake_acall_tools(agent, llm_response):
        return fake_response

    monkeypatch.setattr(agent.tool_manager, "acall_tools", fake_acall_tools)

    plan = Plan(step=0, llm_plan="do something")
    await agent.aapply_plan(plan)

    # "action" is an additive event type, so it is stored as a list
    actions = agent.memory.step_content.get("action")
    assert actions is not None
    assert isinstance(actions, list) and len(actions) == 1
    assert "tool_calls" in actions[0]
    assert len(actions[0]["tool_calls"]) == 2
    assert actions[0]["tool_calls"][0] == {
        "name": "move_one_step",
        "response": "agent moved to (3, 4)",
    }
    assert actions[0]["tool_calls"][1] == {
        "name": "arrest_citizen",
        "response": "Citizen 12 arrested",
    }


def test_generate_obs_with_one_neighbor(monkeypatch):
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=45)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos, agent_class=LLMAgent):
            system_prompt = "You are an agent in a simulation."
            agents = agent_class.create_agents(
                self,
                n=1,
                reasoning=ReActReasoning,
                system_prompt=system_prompt,
                vision=-1,
                internal_state=["test_state"],
            )
            x, y = pos
            agent = agents.to_list()[0]
            self.grid.place_agent(agent, (x, y))
            return agent

    model = DummyModel()

    agent = model.add_agent((1, 1))
    agent.memory = ShortTermMemory(
        agent=agent,
        n=5,
        display=True,
    )
    agent.unique_id = 1

    neighbor = model.add_agent((1, 2))
    neighbor.memory = ShortTermMemory(
        agent=agent,
        n=5,
        display=True,
    )
    neighbor.unique_id = 2
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    obs = agent.generate_obs()

    assert obs.self_state["agent_unique_id"] == 1
    assert "system_prompt" not in obs.self_state

    # we should have exactly one neighboring agent in local_state
    assert len(obs.local_state) == 1

    # extract the neighbor
    key = next(iter(obs.local_state.keys()))
    assert key == "LLMAgent 2"

    entry = obs.local_state[key]
    assert entry["position"] == (1, 2)
    assert entry["internal_state"] == ["test_state"]


def test_send_message_updates_both_agents_memory(monkeypatch):
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=45)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos, agent_class=LLMAgent):
            system_prompt = "You are an agent in a simulation."
            agents = agent_class.create_agents(
                self,
                n=1,
                reasoning=lambda agent: None,
                system_prompt=system_prompt,
                vision=-1,
                internal_state=["test_state"],
            )
            x, y = pos
            agent = agents.to_list()[0]
            self.grid.place_agent(agent, (x, y))
            return agent

    model = DummyModel()
    sender = model.add_agent((0, 0))
    sender.memory = ShortTermMemory(
        agent=sender,
        n=5,
        display=True,
    )
    sender.unique_id = 1

    recipient = model.add_agent((1, 1))
    recipient.memory = ShortTermMemory(
        agent=recipient,
        n=5,
        display=True,
    )
    recipient.unique_id = 2

    recorded_calls = []

    def fake_add_to_memory(*args, **kwargs):
        recorded_calls.append(("sender", kwargs))

    def fake_recipient_add_to_memory(*args, **kwargs):
        recorded_calls.append(("recipient", kwargs))

    # monkeypatch both agents' memory modules
    monkeypatch.setattr(sender.memory, "add_to_memory", fake_add_to_memory)
    monkeypatch.setattr(recipient.memory, "add_to_memory", fake_recipient_add_to_memory)

    result = sender.send_message("hello", recipients=[recipient])
    assert result == "sent message 'hello' to [2]"

    # sender + recipient memory => should be called twice
    assert len(recorded_calls) == 2
    sender_call = next(call for label, call in recorded_calls if label == "sender")
    recipient_call = next(
        call for label, call in recorded_calls if label == "recipient"
    )
    assert sender_call["type"] == "message"
    assert sender_call["content"]["message"] == "hello"
    assert sender_call["content"]["sender"] == sender.unique_id
    assert sender_call["content"]["recipients"] == [recipient.unique_id]
    assert recipient_call["type"] == "message"
    assert recipient_call["content"]["message"] == "hello"
    assert recipient_call["content"]["sender"] == sender.unique_id
    assert "recipients" not in recipient_call["content"]


@pytest.mark.asyncio
async def test_asend_message_updates_both_agents_memory(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=45)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos, agent_class=LLMAgent):
            system_prompt = "You are an agent in a simulation."
            agents = agent_class.create_agents(
                self,
                n=1,
                reasoning=lambda agent: None,
                system_prompt=system_prompt,
                vision=-1,
                internal_state=["test_state"],
            )
            x, y = pos
            agent = agents.to_list()[0]
            self.grid.place_agent(agent, (x, y))
            return agent

    model = DummyModel()
    sender = model.add_agent((0, 0))
    sender.memory = ShortTermMemory(
        agent=sender,
        n=5,
        display=True,
    )
    sender.unique_id = 1

    recipient = model.add_agent((1, 1))
    recipient.memory = ShortTermMemory(
        agent=recipient,
        n=5,
        display=True,
    )
    recipient.unique_id = 2

    recorded_calls = []

    async def fake_aadd_to_memory(*args, **kwargs):
        recorded_calls.append(("sender", kwargs))

    async def fake_recipient_aadd_to_memory(*args, **kwargs):
        recorded_calls.append(("recipient", kwargs))

    monkeypatch.setattr(sender.memory, "aadd_to_memory", fake_aadd_to_memory)
    monkeypatch.setattr(
        recipient.memory, "aadd_to_memory", fake_recipient_aadd_to_memory
    )

    result = await sender.asend_message("hello", recipients=[recipient])
    assert result == "sent message 'hello' to [2]"

    assert len(recorded_calls) == 2
    sender_call = next(call for label, call in recorded_calls if label == "sender")
    recipient_call = next(
        call for label, call in recorded_calls if label == "recipient"
    )
    assert sender_call["type"] == "message"
    assert sender_call["content"]["message"] == "hello"
    assert sender_call["content"]["sender"] == sender.unique_id
    assert sender_call["content"]["recipients"] == [recipient.unique_id]
    assert recipient_call["type"] == "message"
    assert recipient_call["content"]["message"] == "hello"
    assert recipient_call["content"]["sender"] == sender.unique_id
    assert "recipients" not in recipient_call["content"]


@pytest.mark.asyncio
async def test_aapply_plan_adds_to_memory(monkeypatch):
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos):
            system_prompt = "You are an agent in a simulation."
            agents = LLMAgent.create_agents(
                self,
                n=1,
                reasoning=ReActReasoning,
                system_prompt=system_prompt,
                vision=-1,
                internal_state=["test_state"],
            )

            x, y = pos
            agent = agents.to_list()[0]
            self.grid.place_agent(agent, (x, y))
            return agent

    model = DummyModel()
    agent = model.add_agent((1, 1))

    # optional: you can replace with async memory stub
    async def fake_aadd_to_memory(*args, **kwargs):
        pass

    monkeypatch.setattr(agent.memory, "aadd_to_memory", fake_aadd_to_memory)

    # fake async tool response
    fake_response = [{"tool": "foo", "argument": "bar"}]

    async def fake_acall_tools(agent, llm_response):
        return fake_response

    monkeypatch.setattr(agent.tool_manager, "acall_tools", fake_acall_tools)

    plan = Plan(step=0, llm_plan="do something")

    resp = await agent.aapply_plan(plan)

    assert resp == fake_response


@pytest.mark.asyncio
async def test_agenerate_obs_with_one_neighbor(monkeypatch):
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=45)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos):
            agents = LLMAgent.create_agents(
                self,
                n=1,
                reasoning=ReActReasoning,
                system_prompt="You are an agent.",
                vision=-1,
                internal_state=["test_state"],
            )
            x, y = pos
            agent = agents.to_list()[0]
            self.grid.place_agent(agent, (x, y))
            return agent

    model = DummyModel()

    agent = model.add_agent((1, 1))
    neighbor = model.add_agent((1, 2))

    agent.unique_id = 1
    neighbor.unique_id = 2

    async def fake_aadd_to_memory(*args, **kwargs):
        pass

    monkeypatch.setattr(agent.memory, "aadd_to_memory", fake_aadd_to_memory)

    obs = await agent.agenerate_obs()

    assert obs.self_state["agent_unique_id"] == 1
    assert len(obs.local_state) == 1

    key = next(iter(obs.local_state.keys()))
    assert key == "LLMAgent 2"

    entry = obs.local_state[key]
    assert entry["position"] == (1, 2)
    assert entry["internal_state"] == ["test_state"]


@pytest.mark.asyncio
async def test_async_wrapper_calls_pre_and_post(monkeypatch):
    class CustomAgent(LLMAgent):
        async def astep(self):
            self.user_called = True
            return "done"

    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=1)
            self.grid = MultiGrid(3, 3, torus=False)

    model = DummyModel()

    agent = CustomAgent.create_agents(
        model,
        n=1,
        reasoning=lambda agent: None,
        system_prompt="test",
        vision=-1,
        internal_state=[],
    ).to_list()[0]

    calls = {"pre": 0, "post": 0}

    async def fake_aprocess_step(pre_step=False):
        if pre_step:
            calls["pre"] += 1
        else:
            calls["post"] += 1

    monkeypatch.setattr(agent.memory, "aprocess_step", fake_aprocess_step)

    result = await agent.astep()

    assert result == "done"
    assert calls["pre"] == 1
    assert calls["post"] == 1
    assert agent.user_called is True


@pytest.mark.asyncio
async def test_astep_fallback_warns_once_for_step_only_subclass(monkeypatch):
    class StepOnlyAgent(LLMAgent):
        def step(self):
            self.step_calls = getattr(self, "step_calls", 0) + 1

    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=1)
            self.grid = MultiGrid(3, 3, torus=False)

    model = DummyModel()

    agent = StepOnlyAgent.create_agents(
        model,
        n=1,
        reasoning=lambda agent: None,
        system_prompt="test",
        vision=-1,
        internal_state=[],
    ).to_list()[0]

    monkeypatch.setattr(agent.memory, "process_step", lambda pre_step=False: None)

    async def fake_aprocess_step(pre_step=False):
        return None

    monkeypatch.setattr(agent.memory, "aprocess_step", fake_aprocess_step)

    with pytest.warns(RuntimeWarning, match="Override astep\\(\\)"):
        await agent.astep()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await agent.astep()

    assert agent.step_calls == 2


class MockCell:
    """Minimal mock of a CellAgent cell with just a coordinate attribute."""

    def __init__(self, coordinate):
        self.coordinate = coordinate


def _make_agent(model, vision=0, internal_state=None):
    """Helper: create one LLMAgent and attach fresh ShortTermMemory."""
    agents = LLMAgent.create_agents(
        model,
        n=1,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=vision,
        internal_state=internal_state or ["test"],
    )
    agent = agents.to_list()[0]
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    return agent


def test_safer_cell_access_agent_with_cell_no_pos(monkeypatch):
    """Agent location falls back to cell.coordinate when pos=None."""
    model = Model(rng=42)
    agent = _make_agent(model)
    agent.pos = None
    agent.cell = MockCell(coordinate=(3, 4))
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)

    obs = agent.generate_obs()

    assert obs.self_state["location"] == (3, 4)


def test_safer_cell_access_agent_without_cell_or_pos(monkeypatch):
    """Agent location returns None gracefully when neither pos nor cell exists."""
    model = Model(rng=42)
    agent = _make_agent(model)
    agent.pos = None
    if hasattr(agent, "cell"):
        delattr(agent, "cell")
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)

    obs = agent.generate_obs()

    assert obs.self_state["location"] is None


def test_safer_cell_access_neighbor_with_cell_no_pos(monkeypatch):
    """Neighbor position uses cell.coordinate when neighbor.pos=None."""

    class GridModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.grid = MultiGrid(3, 3, torus=False)

    model = GridModel()
    agents = LLMAgent.create_agents(
        model,
        n=2,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=-1,
        internal_state=["test"],
    )
    agent, neighbor = agents
    agent.unique_id = 1
    neighbor.unique_id = 2
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    neighbor.memory = ShortTermMemory(agent=neighbor, n=5, display=True)

    model.grid.place_agent(agent, (1, 1))
    neighbor.pos = None
    neighbor.cell = MockCell(coordinate=(2, 2))

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)
    obs = agent.generate_obs()

    assert obs.local_state["LLMAgent 2"]["position"] == (2, 2)


def test_safer_cell_access_neighbor_without_cell_or_pos(monkeypatch):
    """Neighbor position returns None when neighbor has neither pos nor cell."""

    class GridModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.grid = MultiGrid(3, 3, torus=False)

    model = GridModel()
    agents = LLMAgent.create_agents(
        model,
        n=2,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=-1,
        internal_state=["test"],
    )
    agent, neighbor = agents
    agent.unique_id = 1
    neighbor.unique_id = 2
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    neighbor.memory = ShortTermMemory(agent=neighbor, n=5, display=True)

    model.grid.place_agent(agent, (1, 1))
    neighbor.pos = None
    if hasattr(neighbor, "cell"):
        delattr(neighbor, "cell")

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)
    obs = agent.generate_obs()

    assert obs.local_state["LLMAgent 2"]["position"] is None


def test_generate_obs_with_continuous_space(monkeypatch):
    """Agents within vision radius are included; those outside are not."""

    class ContModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=False)

    model = ContModel()
    agents = LLMAgent.create_agents(
        model,
        n=3,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=2.0,
        internal_state=["test"],
    )
    agent, nearby, far = agents
    agent.unique_id = 1
    nearby.unique_id = 2
    far.unique_id = 3
    for a in agents:
        a.memory = ShortTermMemory(agent=a, n=5, display=True)

    model.space.place_agent(agent, (5.0, 5.0))
    model.space.place_agent(nearby, (6.0, 5.0))  # distance ≈ 1.0
    model.space.place_agent(far, (9.0, 9.0))  # distance ≈ 5.66

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)
    obs = agent.generate_obs()

    assert len(obs.local_state) == 1
    assert "LLMAgent 2" in obs.local_state
    assert "LLMAgent 3" not in obs.local_state


def test_generate_obs_vision_all_agents(monkeypatch):
    """vision=-1 returns all other agents regardless of position."""

    class GridModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.grid = MultiGrid(10, 10, torus=False)

    model = GridModel()
    agents = LLMAgent.create_agents(
        model,
        n=4,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=-1,
        internal_state=["test"],
    )
    for idx, a in enumerate(agents):
        a.unique_id = idx + 1
        a.memory = ShortTermMemory(agent=a, n=5, display=True)
        model.grid.place_agent(a, (idx, idx))

    agent = agents.to_list()[0]
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)
    obs = agent.generate_obs()

    # Should see all 3 other agents
    assert len(obs.local_state) == 3
    assert "LLMAgent 2" in obs.local_state
    assert "LLMAgent 3" in obs.local_state
    assert "LLMAgent 4" in obs.local_state


def test_generate_obs_no_grid_with_vision(monkeypatch):
    """When the model has no grid/space, generate_obs falls back to empty neighbors."""
    model = Model(rng=42)  # no grid, no space
    agents = LLMAgent.create_agents(
        model,
        n=2,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=5,
        internal_state=["test"],
    )
    agent = agents.to_list()[0]
    agent.unique_id = 1
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)

    obs = agent.generate_obs()

    assert len(obs.local_state) == 0


def test_generate_obs_standard_grid_with_vision_radius(monkeypatch):
    """
    Tests spatial neighborhood lookup for an LLMAgent on a SingleGrid
    when a positive vision radius is set.

    Verifies that:
    - Agents within the specified vision distance are detected.
    - The observation includes nearby agents in local_state.
    - The SingleGrid neighbor lookup branch is executed.
    """

    class GridModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            # Reverted to width/height for SingleGrid
            self.grid = SingleGrid(width=5, height=5, torus=False)

    model = GridModel()
    agent = LLMAgent(model=model, reasoning=ReActReasoning, vision=1)
    neighbor = LLMAgent(model=model, reasoning=ReActReasoning)

    # Place agents within vision distance
    model.grid.place_agent(agent, (2, 2))
    model.grid.place_agent(neighbor, (2, 3))

    # Mock memory to bypass API logic
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    obs = agent.generate_obs()

    assert len(obs.local_state) == 1
    assert "LLMAgent" in str(obs.local_state)


def test_generate_obs_orthogonal_grid_branches(monkeypatch):
    """
    Tests the OrthogonalMooreGrid-specific observation logic in generate_obs().

    Checks the following:
    - When the agent is properly added to a cell, its location is correctly detected and included in self_state.
    - When the agent is not present in any grid cell, generate_obs() handles the situation gracefully and returns an empty local_state without errors.

    Covers Orthogonal grid-specific branches including
    cell-based lookup and fallback behavior.
    """

    class OrthoModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            # Pass self.random to ensure reproducibility
            self.grid = OrthogonalMooreGrid(dimensions=(5, 5), random=self.random)

    model = OrthoModel()
    agent = LLMAgent(model=model, reasoning=ReActReasoning, vision=1)

    # Mock memory to bypass API logic
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    agent_cell = next(
        cell for cell in model.grid.all_cells if cell.coordinate == (2, 2)
    )
    agent_cell.add_agent(agent)
    agent.pos = (2, 2)

    obs = agent.generate_obs()
    assert obs.self_state["location"] == (2, 2)

    agent_cell.remove_agent(agent)
    obs = agent.generate_obs()

    assert len(obs.local_state) == 0


def test_generate_obs_with_non_llm_neighbor(monkeypatch):
    """
    _build_observation should work when a neighbor is a plain Mesa Agent
    that has no internal_state attribute (e.g. a rule-based agent in a mixed sim).
    """
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class PlainAgent(Agent):
        """A regular Mesa agent with NO internal_state, simulates non-LLM agents."""

        def step(self):
            pass

    class MixedModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.grid = MultiGrid(5, 5, torus=False)

    model = MixedModel()
    llm_agent = LLMAgent(model=model, reasoning=ReActReasoning, vision=-1)
    plain = PlainAgent(model=model)

    model.grid.place_agent(llm_agent, (2, 2))
    model.grid.place_agent(plain, (3, 3))

    monkeypatch.setattr(llm_agent.memory, "add_to_memory", lambda *a, **kw: None)

    obs = llm_agent.generate_obs()

    plain_key = f"PlainAgent {plain.unique_id}"
    assert plain_key in obs.local_state
    # Non-LLM agent should have an empty internal_state
    assert obs.local_state[plain_key]["internal_state"] == []


@pytest.mark.asyncio
async def test_agenerate_obs_with_non_llm_neighbor(monkeypatch):
    """
    Async path shares _build_observation, must work for agenerate_obs().
    """
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class PlainAgent(Agent):
        def step(self):
            pass

    class MixedModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.grid = MultiGrid(5, 5, torus=False)

    model = MixedModel()
    llm_agent = LLMAgent(model=model, reasoning=ReActReasoning, vision=-1)
    plain = PlainAgent(model=model)

    model.grid.place_agent(llm_agent, (2, 2))
    model.grid.place_agent(plain, (3, 3))

    async def fake_aadd_to_memory(*args, **kwargs):
        pass

    monkeypatch.setattr(llm_agent.memory, "aadd_to_memory", fake_aadd_to_memory)

    obs = await llm_agent.agenerate_obs()

    plain_key = f"PlainAgent {plain.unique_id}"
    assert plain_key in obs.local_state
    assert obs.local_state[plain_key]["internal_state"] == []


# ---------------------------------------------------------------------------
# send_message / asend_message - store unique_ids, not Agent objects (#156)
# ---------------------------------------------------------------------------


def _make_send_message_model(monkeypatch):
    """Shared setup: two-agent MultiGrid model with ShortTermMemory."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=45)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos):
            agents = LLMAgent.create_agents(
                self,
                n=1,
                reasoning=lambda agent: None,
                system_prompt="Test",
                vision=-1,
                internal_state=[],
            )
            agent = agents.to_list()[0]
            self.grid.place_agent(agent, pos)
            return agent

    model = DummyModel()

    sender = model.add_agent((0, 0))
    sender.memory = ShortTermMemory(agent=sender, n=5, display=True)
    sender.unique_id = 10

    recipient = model.add_agent((1, 1))
    recipient.memory = ShortTermMemory(agent=recipient, n=5, display=True)
    recipient.unique_id = 20

    return sender, recipient


def test_send_message_stores_serializable_ids(monkeypatch):
    """send_message stores sender/recipients as unique_ids, not Agent objects."""
    sender, recipient = _make_send_message_model(monkeypatch)

    captured = {}

    def capture_content(type, content):
        captured.update(content)

    monkeypatch.setattr(recipient.memory, "add_to_memory", capture_content)
    monkeypatch.setattr(sender.memory, "add_to_memory", lambda *a, **kw: None)

    sender.send_message("hello", recipients=[recipient])

    assert captured["sender"] == 10
    assert captured["message"] == "hello"

    # Must not raise TypeError when serializing
    data = json.loads(json.dumps(captured))
    assert data["sender"] == 10
    assert "recipients" not in data  # recipients only stored in sender, not recipient
    assert data["message"] == "hello"


# ---------------------------------------------------------------------------
# recorder attribute initialised to None (#218)
# ---------------------------------------------------------------------------


def test_llm_agent_has_recorder_attribute():
    """LLMAgent instances must expose a `recorder` attribute so that
    @record_model can attach a SimulationRecorder via hasattr()."""
    model = Model(rng=42)
    agent = LLMAgent(
        model=model,
        reasoning=ReActReasoning,
        system_prompt="test",
    )

    assert hasattr(agent, "recorder")
    assert agent.recorder is None


@pytest.mark.asyncio
async def test_asend_message_stores_serializable_ids(monkeypatch):
    """asend_message stores sender/recipients as unique_ids, not Agent objects."""
    sender, recipient = _make_send_message_model(monkeypatch)

    captured = {}

    async def capture_content(type, content):
        captured.update(content)

    async def noop(*a, **kw):
        pass

    monkeypatch.setattr(recipient.memory, "aadd_to_memory", capture_content)
    monkeypatch.setattr(sender.memory, "aadd_to_memory", noop)

    await sender.asend_message("hello", recipients=[recipient])

    assert captured["sender"] == 10
    assert (
        "recipients" not in captured
    )  # recipients only stored in sender, not recipient
    assert captured["message"] == "hello"

    data = json.loads(json.dumps(captured))
    assert data["sender"] == 10
    assert data["message"] == "hello"
    assert "recipients" not in data


def test_send_message_skips_non_llm_recipient(monkeypatch, caplog):
    """send_message should mirror speak_to when a recipient has no memory."""
    sender, recipient = _make_send_message_model(monkeypatch)

    class RuleAgent(Agent):
        def step(self):
            pass

    skipped = RuleAgent(model=sender.model)
    skipped.unique_id = 30
    sender.model.grid.place_agent(skipped, (2, 2))

    recorded_calls = []

    def fake_sender_add_to_memory(*args, **kwargs):
        recorded_calls.append(("sender", kwargs))

    def fake_recipient_add_to_memory(*args, **kwargs):
        recorded_calls.append(("recipient", kwargs))

    monkeypatch.setattr(sender.memory, "add_to_memory", fake_sender_add_to_memory)
    monkeypatch.setattr(recipient.memory, "add_to_memory", fake_recipient_add_to_memory)

    with caplog.at_level(logging.WARNING, logger="mesa_llm.llm_agent"):
        result = sender.send_message("hello", recipients=[recipient, skipped])

    assert result == (
        "sent message 'hello' to [20]; skipped [30] because they have no `memory` attribute"
    )
    assert len(recorded_calls) == 2
    sender_call = next(call for label, call in recorded_calls if label == "sender")
    recipient_call = next(
        call for label, call in recorded_calls if label == "recipient"
    )
    assert sender_call["content"]["recipients"] == [20]
    assert recipient_call["content"]["sender"] == 10
    assert any(
        "30" in record.message and "send_message" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_asend_message_skips_non_llm_recipient(monkeypatch, caplog):
    """asend_message should mirror speak_to when a recipient has no memory."""
    sender, recipient = _make_send_message_model(monkeypatch)

    class RuleAgent(Agent):
        def step(self):
            pass

    skipped = RuleAgent(model=sender.model)
    skipped.unique_id = 30
    sender.model.grid.place_agent(skipped, (2, 2))

    recorded_calls = []

    async def fake_sender_add_to_memory(*args, **kwargs):
        recorded_calls.append(("sender", kwargs))

    async def fake_recipient_add_to_memory(*args, **kwargs):
        recorded_calls.append(("recipient", kwargs))

    monkeypatch.setattr(sender.memory, "aadd_to_memory", fake_sender_add_to_memory)
    monkeypatch.setattr(
        recipient.memory, "aadd_to_memory", fake_recipient_add_to_memory
    )

    with caplog.at_level(logging.WARNING, logger="mesa_llm.llm_agent"):
        result = await sender.asend_message("hello", recipients=[recipient, skipped])

    assert result == (
        "sent message 'hello' to [20]; skipped [30] because they have no `memory` attribute"
    )
    assert len(recorded_calls) == 2
    sender_call = next(call for label, call in recorded_calls if label == "sender")
    recipient_call = next(
        call for label, call in recorded_calls if label == "recipient"
    )
    assert sender_call["content"]["recipients"] == [20]
    assert recipient_call["content"]["sender"] == 10
    assert any(
        "30" in record.message and "send_message" in record.message
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# _build_observation — None pos handling (#244)
# ---------------------------------------------------------------------------


def test_generate_obs_with_none_pos(monkeypatch):
    """generate_obs must not crash when agent.pos is None and has no cell."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.grid = MultiGrid(3, 3, torus=False)

    model = DummyModel()

    agent = LLMAgent.create_agents(
        model,
        n=1,
        reasoning=ReActReasoning,
        system_prompt="Test prompt",
        vision=1,
        internal_state=[],
    ).to_list()[0]

    # Agent is explicitly NOT placed on the grid
    agent.pos = None
    if hasattr(agent, "cell"):
        delattr(agent, "cell")

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    obs = agent.generate_obs()

    assert obs is not None
    assert obs.self_state["location"] is None
    assert len(obs.local_state) == 0


def test_system_prompt_proxies_llm_prompt(basic_agent):
    """Agent system_prompt should proxy the underlying LLM prompt state."""
    basic_agent.system_prompt = "Updated prompt"

    assert basic_agent.system_prompt == "Updated prompt"
    assert basic_agent.llm.system_prompt == "Updated prompt"

    basic_agent.llm.system_prompt = "LLM prompt"

    assert basic_agent.system_prompt == "LLM prompt"
