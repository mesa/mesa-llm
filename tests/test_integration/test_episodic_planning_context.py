"""Stored episodic context must reach both planning completion paths."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mesa_llm.memory.episodic_memory import EpisodicMemory
from mesa_llm.reasoning.cot import CoTReasoning
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.reasoning.reasoning import Observation, Plan
from mesa_llm.reasoning.rewoo import ReWOOReasoning

FACT = "the bridge is closed"


@pytest.fixture
def episodic_agent(monkeypatch):
    """Use real episodic storage without a provider call or API key."""
    monkeypatch.setattr(
        "mesa_llm.module_llm.ModuleLLM.__init__", lambda self, *args, **kwargs: None
    )
    agent = SimpleNamespace(
        model=SimpleNamespace(steps=1), system_prompt="Keep the village safe."
    )
    memory = EpisodicMemory(agent, llm_model="ollama/llama3.2", display=False)
    memory.grade_event_importance = Mock(return_value=5)
    memory.agrade_event_importance = AsyncMock(return_value=5)
    agent.memory = memory
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="wait"))]
    )
    agent.llm = SimpleNamespace(
        generate=Mock(return_value=response), agenerate=AsyncMock(return_value=response)
    )
    agent._tool_manager = SimpleNamespace(get_tools_schema=list)
    return agent


@pytest.mark.parametrize("reasoning_type", [CoTReasoning, ReWOOReasoning])
@pytest.mark.parametrize("has_memory", [False, True])
def test_sync_planning_uses_episodic_context(
    episodic_agent, reasoning_type, has_memory
):
    """Planning includes selected memories and handles empty episodic storage."""
    agent = episodic_agent
    if has_memory:
        agent.memory.add_to_memory("observation", {"detail": FACT})
    reasoning = reasoning_type(agent)
    reasoning.execute_tool_call = Mock(
        return_value=Plan(step=1, llm_plan=SimpleNamespace(tool_calls=[]))
    )
    reasoning.plan(prompt="choose an action", obs=Observation(1, {}, {}))
    prompt = agent.llm.generate.call_args.kwargs["system_prompt"]
    assert (FACT in prompt) is has_memory
    assert "Top 30 memory entries:" in prompt
    assert "Keep the village safe." in prompt
    assert "Current Observation" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_type", [CoTReasoning, ReWOOReasoning])
@pytest.mark.parametrize("has_memory", [False, True])
async def test_async_planning_uses_episodic_context(
    episodic_agent, reasoning_type, has_memory
):
    """Async planning includes the same context without a provider completion."""
    agent = episodic_agent
    if has_memory:
        agent.memory.add_to_memory("observation", {"detail": FACT})
    reasoning = reasoning_type(agent)
    reasoning.aexecute_tool_call = AsyncMock(
        return_value=Plan(step=1, llm_plan=SimpleNamespace(tool_calls=[]))
    )
    await reasoning.aplan(prompt="choose an action", obs=Observation(1, {}, {}))
    prompt = agent.llm.agenerate.call_args.kwargs["system_prompt"]
    assert (FACT in prompt) is has_memory
    assert "Top 30 memory entries:" in prompt
    assert "Keep the village safe." in prompt
    assert "Current Observation" in prompt


def test_react_still_includes_episodic_context(episodic_agent):
    """The existing ReAct path remains a positive control."""
    agent = episodic_agent
    agent.memory.add_to_memory("observation", {"detail": FACT})
    assert FACT in str(ReActReasoning(agent).get_react_prompt(Observation(1, {}, {})))


@pytest.fixture
def limited_episodic_agent(episodic_agent):
    """Keep the highest-ranked entry from several stored memories."""
    agent = episodic_agent
    agent.memory.considered_entries = 1
    agent.memory.grade_event_importance.side_effect = [1, 2, 5]
    for step, detail in enumerate(
        ["the bakery is open", "the market is busy", FACT], start=1
    ):
        agent.model.steps = step
        agent.memory.add_to_memory("observation", {"detail": detail})
    agent.memory.grade_event_importance.side_effect = None
    assert len(agent.memory.memory_entries) == 3
    assert agent.memory.retrieve_top_k_entries(1) == [agent.memory.memory_entries[-1]]
    return agent


@pytest.mark.parametrize("reasoning_type", [CoTReasoning, ReWOOReasoning])
def test_sync_planning_respects_episodic_selection(
    limited_episodic_agent, reasoning_type
):
    """Only the selected memory reaches the synchronous planning prompt."""
    agent = limited_episodic_agent
    reasoning = reasoning_type(agent)
    reasoning.execute_tool_call = Mock(
        return_value=Plan(step=3, llm_plan=SimpleNamespace(tool_calls=[]))
    )
    reasoning.plan(prompt="choose an action", obs=Observation(3, {}, {}))
    prompt = agent.llm.generate.call_args.kwargs["system_prompt"]
    assert "Top 1 memory entries:" in prompt
    assert FACT in prompt
    assert "the bakery is open" not in prompt
    assert "the market is busy" not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_type", [CoTReasoning, ReWOOReasoning])
async def test_async_planning_respects_episodic_selection(
    limited_episodic_agent, reasoning_type
):
    """Only the selected memory reaches the asynchronous planning prompt."""
    agent = limited_episodic_agent
    reasoning = reasoning_type(agent)
    reasoning.aexecute_tool_call = AsyncMock(
        return_value=Plan(step=3, llm_plan=SimpleNamespace(tool_calls=[]))
    )
    await reasoning.aplan(prompt="choose an action", obs=Observation(3, {}, {}))
    prompt = agent.llm.agenerate.call_args.kwargs["system_prompt"]
    assert "Top 1 memory entries:" in prompt
    assert FACT in prompt
    assert "the bakery is open" not in prompt
    assert "the market is busy" not in prompt
