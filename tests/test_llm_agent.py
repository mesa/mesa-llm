# tests/test_llm_agent.py

import json
import logging
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mesa.agent import Agent
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.model import Model
from mesa.space import ContinuousSpace, MultiGrid, SingleGrid

from mesa_llm import Plan
from mesa_llm.actions import ActionChoice, ActionManager, action, wait
from mesa_llm.actions.action_decorator import _GLOBAL_ACTION_REGISTRY
from mesa_llm.actions.action_manager import _UNSET as _ACTIONS_UNSET
from mesa_llm.llm_agent import LLMAgent
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


def test_llm_agent_actions_constructor_unknown_name_fails_fast():
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)

    with pytest.raises(ValueError, match="Unknown action name"):
        LLMAgent(
            DummyModel(),
            reasoning=ReActReasoning,
            actions=["missing_llm_agent_action"],
        )


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


def _action_choice_response(content):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


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

    action_context = call_kwargs["prompt"][0]
    assert "Available actions:" in action_context
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
