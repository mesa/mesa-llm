# tests/test_reasoning/test_rewoo.py

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from mesa_llm.reasoning.reasoning import Observation, Plan
from mesa_llm.reasoning.rewoo import ReWOOReasoning


def _tool_call(tool_id: str):
    return {
        "id": tool_id,
        "type": "function",
        "function": {"name": "mock_tool", "arguments": "{}"},
    }


class TestReWOOReasoning:
    """Test the ReWOOReasoning class."""

    def test_rewoo_reasoning_initialization(self, mock_agent):
        """Test ReWOOReasoning initialization."""
        reasoning = ReWOOReasoning(mock_agent)

        assert reasoning.agent == mock_agent
        assert reasoning.remaining_tool_calls == 0
        assert reasoning.current_plan is None
        assert reasoning.current_obs is None

    def test_get_rewoo_system_prompt(self, mock_agent):
        """Test get_rewoo_system_prompt."""
        mock_agent.system_prompt = "Agent persona"
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory content"
        mock_agent.memory.format_short_term.return_value = "Short term memory content"

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.current_obs = Observation(
            step=1, self_state={"test": "data"}, local_state={}
        )

        prompt = reasoning.get_rewoo_system_prompt(reasoning.current_obs)

        assert "Long term memory content" in prompt
        assert "Short term memory content" in prompt
        assert "Agent Persona" in prompt
        assert "Agent persona" in prompt
        assert "Current Observation" in prompt
        assert "plan" in prompt
        assert "step_1" in prompt
        assert "contingency" in prompt

    def test_get_rewoo_system_prompt_omits_empty_persona(self, mock_agent):
        """Empty agent persona should not add a persona section."""
        mock_agent.system_prompt = None
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory content"
        mock_agent.memory.format_short_term.return_value = "Short term memory content"

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.current_obs = Observation(
            step=1, self_state={"test": "data"}, local_state={}
        )

        prompt = reasoning.get_rewoo_system_prompt(reasoning.current_obs)

        assert "Agent Persona" not in prompt

    def test_plan_with_remaining_tool_calls(self, mock_agent):
        """Test plan method when there are remaining tool calls."""
        mock_agent.generate_obs = Mock()
        mock_agent.memory = Mock()
        mock_agent.step_prompt = None

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.remaining_tool_calls = 2
        reasoning.current_plan = Mock()

        # Create actual mock objects for tool calls
        mock_tool_1 = Mock()
        mock_tool_2 = Mock()
        mock_tool_3 = Mock()
        reasoning.current_plan.tool_calls = [
            mock_tool_1,
            mock_tool_2,
            mock_tool_3,
        ]  # 3 tool calls
        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})

        result = reasoning.plan(ttl=5)

        assert isinstance(result, Plan)
        assert result.ttl == 5
        assert result.llm_plan.tool_calls == [mock_tool_2]  # Should get index 1 (3-2)
        assert reasoning.remaining_tool_calls == 1
        mock_agent.generate_obs.assert_not_called()
        mock_agent.memory.add_to_memory.assert_not_called()

    def test_plan_new_plan_generation(self, llm_response_factory, mock_agent):
        """Test plan method when generating a new plan."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = llm_response_factory(content="Test plan content")
        mock_exec_response = llm_response_factory(
            content="Execution plan",
            tool_calls=[_tool_call("call_1"), _tool_call("call_2")],
        )

        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        def execute_with_ttl(*args, **kwargs):
            return Plan(
                step=1,
                llm_plan=mock_exec_response.choices[0].message,
                ttl=kwargs.get("ttl", 1),
            )

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(side_effect=execute_with_ttl)

        result = reasoning.plan(ttl=4)

        assert isinstance(result, Plan)
        assert result.ttl == 4
        assert reasoning.remaining_tool_calls == 1
        assert len(result.llm_plan.tool_calls) == 1
        assert reasoning.current_plan == mock_exec_response.choices[0].message
        assert reasoning.current_obs is not None
        reasoning.execute_tool_call.assert_called_once_with(
            "Test plan content",
            ttl=4,
            tool_calls="auto",
        )
        mock_agent.generate_obs.assert_called_once()

    def test_plan_with_custom_prompt(self, llm_response_factory, mock_agent):
        """Test plan method with custom prompt."""
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = llm_response_factory(content="Custom plan content")
        mock_exec_response = llm_response_factory(
            content="Execution plan",
            tool_calls=[_tool_call("call_1")],
        )

        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(
            return_value=Plan(step=1, llm_plan=mock_exec_response.choices[0].message)
        )

        result = reasoning.plan(prompt="Custom prompt")

        assert isinstance(result, Plan)
        assert reasoning.remaining_tool_calls == 0

    def test_plan_with_custom_tool_calls(self, llm_response_factory, mock_agent):
        """Test plan method forwards a custom execution tool choice."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = llm_response_factory(content="Test plan content")
        mock_exec_response = llm_response_factory(
            content="Execution plan",
            tool_calls=[_tool_call("call_1")],
        )
        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(
            return_value=Plan(step=1, llm_plan=mock_exec_response.choices[0].message)
        )

        reasoning.plan(tool_calls="required")

        reasoning.execute_tool_call.assert_called_once_with(
            "Test plan content",
            ttl=1,
            tool_calls="required",
        )

    def test_plan_with_tools(self, llm_response_factory, mock_agent):
        """Test plan method with explicit tools."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = llm_response_factory(content="Test plan content")
        mock_exec_response = llm_response_factory(
            content="Execution plan",
            tool_calls=[_tool_call("call_1")],
        )

        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(
            return_value=Plan(step=1, llm_plan=mock_exec_response.choices[0].message)
        )

        tools = ["tool1", "tool2"]
        result = reasoning.plan(tools=tools)

        assert isinstance(result, Plan)
        mock_agent.tool_manager.get_tools_schema.assert_called_with(tools=tools)

    def test_plan_preserves_tools_for_remaining_tool_calls(
        self, llm_response_factory, mock_agent
    ):
        """ReWOO continuation plans keep the original narrowed tool selection."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = llm_response_factory(content="Test plan content")
        mock_exec_response = llm_response_factory(
            content="Execution plan",
            tool_calls=[_tool_call("call_1"), _tool_call("call_2")],
        )
        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(
            return_value=Plan(step=1, llm_plan=mock_exec_response.choices[0].message)
        )

        tools = ["tool1"]
        reasoning.plan(tools=tools)
        continuation = reasoning.plan()

        assert continuation.tools == tools
        assert continuation.llm_plan.tool_calls == [
            mock_exec_response.choices[0].message.tool_calls[1]
        ]

    def test_plan_uses_scoped_system_prompt(self, llm_response_factory, mock_agent):
        """ReWOO plan should pass system prompt per call and not mutate llm state."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.llm.system_prompt = "base-system-prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_agent.llm.generate.return_value = llm_response_factory(
            content="Test plan content"
        )

        mock_plan_without_tool_calls = Mock(spec=[])
        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(
            return_value=Plan(step=1, llm_plan=mock_plan_without_tool_calls)
        )

        obs = Observation(step=1, self_state={}, local_state={})
        expected_prompt = reasoning.get_rewoo_system_prompt(obs)
        reasoning.plan(obs=obs)

        assert mock_agent.llm.system_prompt == "base-system-prompt"
        assert (
            mock_agent.llm.generate.call_args.kwargs["system_prompt"] == expected_prompt
        )

    def test_plan_no_prompt_error(self, mock_agent):
        """Test plan method raises error when no prompt is provided."""
        mock_agent.step_prompt = None
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()

        reasoning = ReWOOReasoning(mock_agent)

        with pytest.raises(
            ValueError, match=r"No prompt provided and agent.step_prompt is None"
        ):
            reasoning.plan()

    def test_plan_with_no_tool_calls(self, llm_response_factory, mock_agent):
        """Test plan method when execution returns no tool calls."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = llm_response_factory(content="Test plan content")
        mock_exec_response = llm_response_factory(content="Execution plan")

        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        # Create a mock plan that doesn't have tool_calls attribute
        mock_plan_without_tool_calls = Mock(spec=[])  # spec=[] means no attributes
        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(
            return_value=Plan(step=1, llm_plan=mock_plan_without_tool_calls)
        )

        result = reasoning.plan()

        assert isinstance(result, Plan)
        assert reasoning.remaining_tool_calls == 0

    def test_plan_with_tool_calls_set_to_none(self, llm_response_factory, mock_agent):
        """Test plan method when execution returns ``tool_calls=None``."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = llm_response_factory(content="Test plan content")
        mock_agent.llm.generate.return_value = mock_plan_response

        mock_plan_with_none_tool_calls = Mock()
        mock_plan_with_none_tool_calls.tool_calls = None
        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(
            return_value=Plan(step=1, llm_plan=mock_plan_with_none_tool_calls)
        )

        result = reasoning.plan()

        assert isinstance(result, Plan)
        assert reasoning.remaining_tool_calls == 0

    def test_aplan_with_remaining_tool_calls(self, mock_agent):
        """Test aplan method when there are remaining tool calls."""
        mock_agent.generate_obs = Mock()
        mock_agent.memory = Mock()
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.step_prompt = None

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.remaining_tool_calls = 1
        reasoning.current_plan = Mock()

        # Create actual mock objects for tool calls
        mock_tool_1 = Mock()
        mock_tool_2 = Mock()
        reasoning.current_plan.tool_calls = [mock_tool_1, mock_tool_2]  # 2 tool calls
        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})

        result = asyncio.run(reasoning.aplan(ttl=6))

        assert isinstance(result, Plan)
        assert result.ttl == 6
        assert result.llm_plan.tool_calls == [mock_tool_2]  # Should get index 1 (2-1)
        assert reasoning.remaining_tool_calls == 0
        mock_agent.generate_obs.assert_not_called()
        mock_agent.memory.aadd_to_memory.assert_not_awaited()

    def test_aplan_new_plan_generation(self, llm_response_factory, mock_agent):
        """Test aplan uses agenerate_obs (async) not generate_obs (sync)."""
        mock_agent = Mock()
        mock_agent.agenerate_obs = AsyncMock(
            return_value=Observation(step=1, self_state={}, local_state={})
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = llm_response_factory(content="Async plan content")
        mock_exec_response = llm_response_factory(
            content="Async execution plan",
            tool_calls=[
                _tool_call("call_1"),
                _tool_call("call_2"),
                _tool_call("call_3"),
            ],
        )

        mock_agent.llm.agenerate = AsyncMock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        async def aexecute_with_ttl(*args, **kwargs):
            return Plan(
                step=1,
                llm_plan=mock_exec_response.choices[0].message,
                ttl=kwargs.get("ttl", 1),
            )

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(side_effect=aexecute_with_ttl)

        result = asyncio.run(reasoning.aplan("test prompt", ttl=7))

        assert isinstance(result, Plan)
        assert result.ttl == 7
        assert reasoning.remaining_tool_calls == 2
        assert len(result.llm_plan.tool_calls) == 1
        assert reasoning.current_plan == mock_exec_response.choices[0].message
        assert reasoning.current_obs is not None
        reasoning.aexecute_tool_call.assert_called_once_with(
            "Async plan content",
            ttl=7,
            tool_calls="auto",
        )
        mock_agent.agenerate_obs.assert_awaited_once()

    def test_plan_uses_provided_obs_without_regeneration(self):
        """Fresh planning should use provided obs and skip generate_obs()."""
        mock_agent = Mock()
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.generate_obs = Mock()
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = Mock()
        mock_plan_response.choices = [Mock()]
        mock_plan_response.choices[0].message.content = "Test plan content"
        mock_exec_response = Mock()
        mock_exec_response.choices = [Mock()]
        mock_exec_response.choices[0].message = Mock()
        mock_exec_response.choices[0].message.tool_calls = [Mock()]
        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        def execute_with_ttl(*args, **kwargs):
            return Plan(
                step=1,
                llm_plan=mock_exec_response.choices[0].message,
                ttl=kwargs.get("ttl", 1),
            )

        provided_obs = Observation(step=5, self_state={}, local_state={})
        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(side_effect=execute_with_ttl)

        result = reasoning.plan(obs=provided_obs, ttl=2)

        assert isinstance(result, Plan)
        assert result.ttl == 2
        assert reasoning.current_obs is provided_obs
        mock_agent.generate_obs.assert_not_called()

    def test_aplan_uses_provided_obs_without_regeneration(self):
        """Async fresh planning should use provided obs and skip obs generation."""
        mock_agent = Mock()
        mock_agent.generate_obs = Mock()
        mock_agent.agenerate_obs = AsyncMock()
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = Mock()
        mock_plan_response.choices = [Mock()]
        mock_plan_response.choices[0].message.content = "Async plan content"
        mock_exec_response = Mock()
        mock_exec_response.choices = [Mock()]
        mock_exec_response.choices[0].message = Mock()
        mock_exec_response.choices[0].message.tool_calls = [Mock()]
        mock_agent.llm.agenerate = AsyncMock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        async def aexecute_with_ttl(*args, **kwargs):
            return Plan(
                step=1,
                llm_plan=mock_exec_response.choices[0].message,
                ttl=kwargs.get("ttl", 1),
            )

        provided_obs = Observation(step=6, self_state={}, local_state={})
        reasoning = ReWOOReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(side_effect=aexecute_with_ttl)

        result = asyncio.run(reasoning.aplan("test prompt", obs=provided_obs, ttl=3))

        assert isinstance(result, Plan)
        assert result.ttl == 3
        assert reasoning.current_obs is provided_obs
        mock_agent.generate_obs.assert_not_called()
        mock_agent.agenerate_obs.assert_not_called()

    def test_aplan_with_tools(self, llm_response_factory, mock_agent):
        """Test aplan method with explicit tools."""
        mock_agent.agenerate_obs = AsyncMock(
            return_value=Observation(step=1, self_state={}, local_state={})
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = llm_response_factory(content="Async plan content")
        mock_exec_response = llm_response_factory(
            content="Async execution plan",
            tool_calls=[_tool_call("call_1")],
        )

        mock_agent.llm.agenerate = AsyncMock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(
            return_value=Plan(step=1, llm_plan=mock_exec_response.choices[0].message)
        )

        tools = ["tool1", "tool2"]
        result = asyncio.run(reasoning.aplan("test prompt", tools=tools))

        assert isinstance(result, Plan)
        mock_agent.tool_manager.get_tools_schema.assert_called_with(tools=tools)

    def test_aplan_preserves_tools_for_remaining_tool_calls(
        self, llm_response_factory, mock_agent
    ):
        """Async ReWOO continuation plans keep the original narrowed tools."""
        mock_agent.agenerate_obs = AsyncMock(
            return_value=Observation(step=1, self_state={}, local_state={})
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = llm_response_factory(content="Async plan content")
        mock_exec_response = llm_response_factory(
            content="Async execution plan",
            tool_calls=[_tool_call("call_1"), _tool_call("call_2")],
        )
        mock_agent.llm.agenerate = AsyncMock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(
            return_value=Plan(step=1, llm_plan=mock_exec_response.choices[0].message)
        )

        tools = ["tool1"]
        asyncio.run(reasoning.aplan("test prompt", tools=tools))
        continuation = asyncio.run(reasoning.aplan("test prompt"))

        assert continuation.tools == tools
        assert continuation.llm_plan.tool_calls == [
            mock_exec_response.choices[0].message.tool_calls[1]
        ]

    def test_aplan_uses_scoped_system_prompt(self, llm_response_factory, mock_agent):
        """Async ReWOO plan should pass system prompt per call and not mutate llm state."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.llm.system_prompt = "base-system-prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_agent.llm.agenerate = AsyncMock(
            return_value=llm_response_factory(content="Async plan content")
        )

        mock_plan_without_tool_calls = Mock(spec=[])
        reasoning = ReWOOReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(
            return_value=Plan(step=1, llm_plan=mock_plan_without_tool_calls)
        )

        obs = Observation(step=1, self_state={}, local_state={})
        expected_prompt = reasoning.get_rewoo_system_prompt(obs)
        asyncio.run(reasoning.aplan(obs=obs))

        assert mock_agent.llm.system_prompt == "base-system-prompt"
        assert (
            mock_agent.llm.agenerate.call_args.kwargs["system_prompt"]
            == expected_prompt
        )

    def test_aplan_with_no_tool_calls(self, llm_response_factory, mock_agent):
        """Test aplan method when execution returns no tool calls."""
        mock_agent.agenerate_obs = AsyncMock(
            return_value=Observation(step=1, self_state={}, local_state={})
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = llm_response_factory(content="Async plan content")
        mock_exec_response = llm_response_factory(content="Async execution plan")

        mock_agent.llm.agenerate = AsyncMock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        mock_plan_without_tool_calls = Mock(spec=[])
        reasoning = ReWOOReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(
            return_value=Plan(step=1, llm_plan=mock_plan_without_tool_calls)
        )

        result = asyncio.run(reasoning.aplan("test prompt"))

        assert isinstance(result, Plan)
        assert reasoning.remaining_tool_calls == 0

    def test_aplan_with_tool_calls_set_to_none(self, llm_response_factory, mock_agent):
        """Test aplan method when execution returns ``tool_calls=None``."""
        mock_agent.agenerate_obs = AsyncMock(
            return_value=Observation(step=1, self_state={}, local_state={})
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = llm_response_factory(content="Async plan content")
        mock_agent.llm.agenerate = AsyncMock(return_value=mock_plan_response)

        mock_plan_with_none_tool_calls = Mock()
        mock_plan_with_none_tool_calls.tool_calls = None
        reasoning = ReWOOReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(
            return_value=Plan(step=1, llm_plan=mock_plan_with_none_tool_calls)
        )

        result = asyncio.run(reasoning.aplan("test prompt"))

        assert isinstance(result, Plan)
        assert reasoning.remaining_tool_calls == 0

    def test_aplan_uses_step_prompt_when_no_prompt_given(self, mock_agent):
        """Test aplan falls back to agent.step_prompt like sync plan does."""
        mock_agent.step_prompt = "Default step prompt"
        default_obs = Observation(step=1, self_state={}, local_state={})
        mock_agent.generate_obs.return_value = default_obs
        mock_agent.agenerate_obs = AsyncMock(return_value=default_obs)
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = Mock()
        mock_plan_response.choices = [Mock()]
        mock_plan_response.choices[0].message.content = "Async plan content"

        mock_exec_response = Mock()
        mock_exec_response.choices = [Mock()]
        mock_exec_response.choices[0].message = Mock()
        mock_exec_response.choices[0].message.tool_calls = [Mock()]

        mock_agent.llm.agenerate = AsyncMock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(
            return_value=Plan(step=1, llm_plan=mock_exec_response.choices[0].message)
        )

        # Call without prompt — should use agent.step_prompt
        result = asyncio.run(reasoning.aplan())
        assert isinstance(result, Plan)

    def test_remaining_tool_calls_decrement(self, mock_agent):
        """Test that remaining_tool_calls is properly decremented."""
        mock_agent.step_prompt = None

        reasoning = ReWOOReasoning(mock_agent)

        # Create actual mock objects for tool calls
        mock_tool_1 = Mock(name="tool_1")
        mock_tool_2 = Mock(name="tool_2")
        mock_tool_3 = Mock(name="tool_3")

        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})

        # Test the index calculation logic separately for each scenario

        # Scenario 1: 3 remaining out of 3 total -> index should be 0
        reasoning.remaining_tool_calls = 3
        reasoning.current_plan = Mock()
        reasoning.current_plan.tool_calls = [mock_tool_1, mock_tool_2, mock_tool_3]

        result1 = reasoning.plan()
        assert reasoning.remaining_tool_calls == 2
        assert result1.llm_plan.tool_calls == [mock_tool_1]  # index 0 (3-3=0)

        # Scenario 2: 2 remaining out of 3 total -> index should be 1
        reasoning.remaining_tool_calls = 2
        reasoning.current_plan = Mock()
        reasoning.current_plan.tool_calls = [mock_tool_1, mock_tool_2, mock_tool_3]

        result2 = reasoning.plan()
        assert reasoning.remaining_tool_calls == 1
        assert result2.llm_plan.tool_calls == [mock_tool_2]  # index 1 (3-2=1)

        # Scenario 3: 1 remaining out of 3 total -> index should be 2
        reasoning.remaining_tool_calls = 1
        reasoning.current_plan = Mock()
        reasoning.current_plan.tool_calls = [mock_tool_1, mock_tool_2, mock_tool_3]

        result3 = reasoning.plan()
        assert reasoning.remaining_tool_calls == 0
        assert result3.llm_plan.tool_calls == [mock_tool_3]  # index 2 (3-1=2)

    def test_aplan_uses_async_memory_method(self, llm_response_factory, mock_agent):
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.agenerate_obs = AsyncMock(
            return_value=Observation(step=1, self_state={}, local_state={})
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = ""
        mock_agent.memory.format_short_term.return_value = ""
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_plan_response = llm_response_factory(content="Async plan content")
        mock_exec_response = llm_response_factory(
            content="Exec",
            tool_calls=[_tool_call("call_1")],
        )
        mock_agent.llm.agenerate = AsyncMock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(
            return_value=Plan(step=1, llm_plan=mock_exec_response.choices[0].message)
        )

        asyncio.run(reasoning.aplan())

        mock_agent.memory.aadd_to_memory.assert_awaited_once()
        mock_agent.memory.add_to_memory.assert_not_called()

    def test_sequential_replay_dispatches_distinct_tools(self):
        """Regression: plan() replay must dispatch A→B→C, not A→A→A.

        Before the fix, `current_plan = self.current_plan` was an alias, so
        `current_plan.tool_calls = [tool_a]` mutated self.current_plan.tool_calls
        in-place. On step 2, len became 1 and index -1 (Python wrap) returned
        tool_a again. This test sets current_plan ONCE and never resets it.
        """
        mock_agent = Mock()
        mock_agent.step_prompt = None

        tool_a = Mock(name="tool_a")
        tool_b = Mock(name="tool_b")
        tool_c = Mock(name="tool_c")

        mock_plan = Mock()
        mock_plan.tool_calls = [tool_a, tool_b, tool_c]

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.current_plan = mock_plan
        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})
        reasoning.remaining_tool_calls = 3

        result1 = reasoning.plan()
        result2 = reasoning.plan()
        result3 = reasoning.plan()

        assert result1.llm_plan.tool_calls == [tool_a], "Step 1 should dispatch tool A"
        assert result2.llm_plan.tool_calls == [tool_b], (
            "Step 2 should dispatch tool B, not A"
        )
        assert result3.llm_plan.tool_calls == [tool_c], (
            "Step 3 should dispatch tool C, not A"
        )
        assert reasoning.remaining_tool_calls == 0

    def test_sequential_replay_dispatches_distinct_tools_async(self):
        """Async regression: aplan() replay must dispatch A→B→C, not A→A→A."""
        mock_agent = Mock()
        mock_agent.step_prompt = None

        tool_a = Mock(name="tool_a")
        tool_b = Mock(name="tool_b")
        tool_c = Mock(name="tool_c")

        mock_plan = Mock()
        mock_plan.tool_calls = [tool_a, tool_b, tool_c]

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.current_plan = mock_plan
        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})
        reasoning.remaining_tool_calls = 3

        result1 = asyncio.run(reasoning.aplan())
        result2 = asyncio.run(reasoning.aplan())
        result3 = asyncio.run(reasoning.aplan())

        assert result1.llm_plan.tool_calls == [tool_a], "Step 1 should dispatch tool A"
        assert result2.llm_plan.tool_calls == [tool_b], (
            "Step 2 should dispatch tool B, not A"
        )
        assert result3.llm_plan.tool_calls == [tool_c], (
            "Step 3 should dispatch tool C, not A"
        )
        assert reasoning.remaining_tool_calls == 0


class TestReWOOSignatureConsistency:
    def test_plan_accepts_obs_kwarg(self):
        """plan() must accept obs= keyword without raising TypeError."""
        mock_agent = Mock()
        mock_agent.step_prompt = None
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        reasoning = ReWOOReasoning(mock_agent)
        reasoning.remaining_tool_calls = 1
        reasoning.current_plan = Mock()
        reasoning.current_plan.tool_calls = [Mock()]
        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})

        result = reasoning.plan(obs=Observation(step=5, self_state={}, local_state={}))
        assert isinstance(result, Plan)

    def test_plan_accepts_ttl_kwarg(self):
        """plan() must accept ttl= keyword without raising TypeError."""
        mock_agent = Mock()
        mock_agent.step_prompt = None
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        reasoning = ReWOOReasoning(mock_agent)
        reasoning.remaining_tool_calls = 1
        reasoning.current_plan = Mock()
        reasoning.current_plan.tool_calls = [Mock()]
        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})

        result = reasoning.plan(ttl=3)
        assert isinstance(result, Plan)
        assert result.ttl == 3

    def test_aplan_accepts_obs_kwarg(self):
        """aplan() must accept obs= keyword without raising TypeError."""
        mock_agent = Mock()
        mock_agent.generate_obs = Mock()
        mock_agent.step_prompt = None

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.remaining_tool_calls = 1
        reasoning.current_plan = Mock()
        reasoning.current_plan.tool_calls = [Mock()]
        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})

        result = asyncio.run(
            reasoning.aplan(obs=Observation(step=5, self_state={}, local_state={}))
        )
        assert isinstance(result, Plan)

    def test_aplan_accepts_ttl_kwarg(self):
        """aplan() must accept ttl= keyword without raising TypeError."""
        mock_agent = Mock()
        mock_agent.generate_obs = Mock()
        mock_agent.step_prompt = None

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.remaining_tool_calls = 1
        reasoning.current_plan = Mock()
        reasoning.current_plan.tool_calls = [Mock()]
        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})

        result = asyncio.run(reasoning.aplan(ttl=5))
        assert isinstance(result, Plan)
        assert result.ttl == 5

    def test_each_planned_tool_call_is_returned_once(
        self, llm_response_factory, mock_agent
    ):
        """A fresh plan and its replays return every tool call exactly once."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}
        exec_message = (
            llm_response_factory(
                content="Execution plan",
                tool_calls=[_tool_call("a"), _tool_call("b"), _tool_call("c")],
            )
            .choices[0]
            .message
        )

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(
            return_value=Plan(step=1, llm_plan=exec_message)
        )
        mock_agent.llm.generate.return_value = llm_response_factory(content="plan")

        returned = []
        for _ in range(3):
            plan = reasoning.plan(
                obs=Observation(step=1, self_state={}, local_state={})
            )
            returned.extend(call.id for call in plan.llm_plan.tool_calls)

        assert returned == ["a", "b", "c"]
        assert reasoning.execute_tool_call.call_count == 1
