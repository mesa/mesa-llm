"""Memory events reach the SimulationRecorder (observation, plan, message)."""

import asyncio
from unittest.mock import Mock

import pytest
from mesa.model import Model
from mesa.space import MultiGrid

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.episodic_memory import EpisodicMemory
from mesa_llm.memory.st_memory import ShortTermMemory
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.recording.record_model import record_model
from mesa_llm.recording.simulation_recorder import SimulationRecorder


@pytest.fixture
def agent(recorder):
    agent = Mock()
    agent.unique_id = 7
    agent.recorder = recorder
    return agent


def _types(recorder):
    return [event.event_type for event in recorder.events]


def test_observation_and_plan_are_recorded_for_the_agent(agent, recorder):
    memory = ShortTermMemory(agent=agent, n=5, display=False)

    memory.add_to_memory("observation", {"self_state": {"x": 1}})
    memory.add_to_memory("plan", {"content": "go north"})

    assert _types(recorder) == ["observation", "plan"]
    assert [event.agent_id for event in recorder.events] == [7, 7]
    assert recorder.events[0].content == {"self_state": {"x": 1}}


def test_a_message_is_recorded_once_by_the_sender_with_its_recipients(agent, recorder):
    memory = ShortTermMemory(agent=agent, n=5, display=False)

    # What the recipient's memory receives: no `recipients` key.
    memory.add_to_memory("message", {"message": "hello", "sender": 3})
    assert recorder.events == []

    # What the sender's memory receives once per send.
    memory.add_to_memory(
        "message", {"message": "hello", "sender": 7, "recipients": [8, 9]}
    )
    assert _types(recorder) == ["message"]
    event = recorder.events[0]
    assert event.agent_id == 7
    assert event.content == {"message": "hello", "recipient_ids": [8, 9]}


def test_actions_are_left_to_the_agent_so_they_are_not_recorded_twice(agent, recorder):
    memory = ShortTermMemory(agent=agent, n=5, display=False)

    memory.add_to_memory("action", {"tool_calls": []})

    assert recorder.events == []


def test_async_writes_are_recorded(agent, recorder):
    memory = ShortTermMemory(agent=agent, n=5, display=False)

    asyncio.run(memory.aadd_to_memory("observation", {"self_state": {}}))

    assert _types(recorder) == ["observation"]


def test_episodic_memory_records_too(agent, recorder):
    memory = EpisodicMemory(agent=agent, llm_model="gemini/gemini-2.0-flash")
    memory.grade_event_importance = Mock(return_value=3)

    memory.add_to_memory("observation", {"self_state": {}})

    assert _types(recorder) == ["observation"]
    # The recorded content is what the agent produced, not the graded copy.
    assert recorder.events[0].content == {"self_state": {}}


def test_a_failing_recorder_does_not_break_the_agent(agent):
    agent.recorder = Mock(spec=SimulationRecorder)
    agent.recorder.record_event.side_effect = RuntimeError("disk full")
    memory = ShortTermMemory(agent=agent, n=5, display=False)

    memory.add_to_memory("observation", {"self_state": {}})

    assert memory.step_content["observation"] == {"self_state": {}}


def test_memory_without_a_recorder_is_unchanged():
    agent = Mock(spec=["unique_id"])
    agent.unique_id = 1
    memory = ShortTermMemory(agent=agent, n=5, display=False)

    memory.add_to_memory("observation", {"self_state": {}})

    assert memory.step_content["observation"] == {"self_state": {}}


def test_record_model_captures_observations_and_messages(monkeypatch, temp_dir):
    """The scenario from the bug report, without any LLM call."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    @record_model(output_dir=str(temp_dir))
    class M(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.grid = MultiGrid(3, 3, torus=False)
            agents = LLMAgent.create_agents(
                self,
                n=2,
                reasoning=ReActReasoning,
                system_prompt="x",
                vision=-1,
                internal_state=["s"],
            )
            for agent, pos in zip(agents.to_list(), [(0, 0), (1, 1)], strict=True):
                self.grid.place_agent(agent, pos)
                agent.memory = ShortTermMemory(agent=agent, n=5, display=False)

        def step(self):
            first, second = self.agents.to_list()
            first.generate_obs()
            first.send_message("hello", [second])

    model = M()
    model.step()

    recorded = [event.event_type for event in model.recorder.events]
    assert recorded.count("observation") == 1
    assert recorded.count("message") == 1
    message = model.recorder.get_events_by_type("message")[0]
    sender, recipient = model.agents.to_list()
    assert message.agent_id == sender.unique_id
    assert message.content["recipient_ids"] == [recipient.unique_id]
