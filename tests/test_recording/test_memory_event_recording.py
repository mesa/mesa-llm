"""Regression tests: agent memory events reach the SimulationRecorder.

Observations, plans and messages used to be written to memory only, so a
``@record_model`` recording held nothing but step markers and actions from
``execute_action``.
"""

import asyncio
import logging

import pytest
from mesa.model import Model
from mesa.space import MultiGrid

from mesa_llm import Plan
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.episodic_memory import EpisodicMemory
from mesa_llm.memory.st_memory import ShortTermMemory
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.recording.agent_analysis import AgentViewer
from mesa_llm.recording.record_model import record_model


@pytest.fixture
def recorded_model(temp_dir, monkeypatch):
    """A @record_model model with two LLM agents using ShortTermMemory."""
    # These models hold events, so the atexit auto-save would try to write
    # into temp_dir after it has been removed.
    monkeypatch.setattr(
        "mesa_llm.recording.record_model.atexit.register", lambda func: func
    )

    @record_model(output_dir=str(temp_dir))
    class RecordedModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.grid = MultiGrid(3, 3, torus=False)
            agents = LLMAgent.create_agents(
                self,
                n=2,
                reasoning=ReActReasoning,
                system_prompt="Test",
                vision=-1,
                internal_state=["test"],
            )
            for agent, pos in zip(agents.to_list(), [(0, 0), (1, 1)]):
                self.grid.place_agent(agent, pos)
                agent.memory = ShortTermMemory(agent=agent, n=5, display=False)

        def step(self):
            pass

    return RecordedModel()


def _events(model, event_type):
    return model.recorder.get_events_by_type(event_type)


def test_observation_is_recorded(recorded_model):
    sender, _ = recorded_model.agents.to_list()

    sender.generate_obs()

    observations = _events(recorded_model, "observation")
    assert len(observations) == 1
    assert observations[0].agent_id == sender.unique_id
    assert "self_state" in observations[0].content
    assert observations[0].metadata == {"source": "ShortTermMemory.add_to_memory"}


def test_plan_is_recorded(recorded_model):
    agent, _ = recorded_model.agents.to_list()

    agent.memory.add_to_memory(type="plan", content={"reasoning": "go north"})

    plans = _events(recorded_model, "plan")
    assert [(e.agent_id, e.content) for e in plans] == [
        (agent.unique_id, {"reasoning": "go north"})
    ]


def test_recorded_content_is_a_snapshot(recorded_model):
    agent, _ = recorded_model.agents.to_list()
    content = {"reasoning": ["first"]}

    agent.memory.add_to_memory(type="plan", content=content)
    content["reasoning"].append("mutated later")

    assert _events(recorded_model, "plan")[0].content == {"reasoning": ["first"]}


def test_message_is_recorded_once_from_sender(recorded_model):
    sender, recipient = recorded_model.agents.to_list()

    sender.send_message("hello", [recipient])

    messages = _events(recorded_model, "message")
    # One event, even though the message lands in two memories.
    assert len(messages) == 1
    assert messages[0].agent_id == sender.unique_id
    assert messages[0].content == {
        "message": "hello",
        "recipient_ids": [recipient.unique_id],
    }


def test_async_message_and_observation_are_recorded(recorded_model):
    sender, recipient = recorded_model.agents.to_list()

    async def run():
        await sender.agenerate_obs()
        await sender.asend_message("hi", [recipient])

    asyncio.run(run())

    assert len(_events(recorded_model, "observation")) == 1
    messages = _events(recorded_model, "message")
    assert len(messages) == 1
    assert messages[0].content["recipient_ids"] == [recipient.unique_id]


def test_apply_plan_action_is_recorded_once(recorded_model, monkeypatch):
    agent, _ = recorded_model.agents.to_list()
    tool_calls = [
        {"tool_call_id": "1", "role": "tool", "name": "move", "response": "ok"}
    ]
    monkeypatch.setattr(
        agent.tool_manager, "call_tools", lambda agent, llm_response: tool_calls
    )

    agent.apply_plan(Plan(step=0, llm_plan="move"))

    actions = _events(recorded_model, "action")
    assert len(actions) == 1
    assert actions[0].content == {"tool_calls": [{"name": "move", "response": "ok"}]}
    assert actions[0].metadata == {"source": "LLMAgent.apply_plan"}


def test_memory_hook_skips_self_recorded_types(recorded_model):
    agent, _ = recorded_model.agents.to_list()

    agent.memory.add_to_memory(type="action", content={"tool_calls": []})
    agent.memory.add_to_memory(type="message", content={"message": "x"})

    assert _events(recorded_model, "action") == []
    assert _events(recorded_model, "message") == []


def test_recorder_failure_does_not_break_memory(recorded_model, monkeypatch, caplog):
    agent, _ = recorded_model.agents.to_list()

    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(recorded_model.recorder, "record_event", broken)

    with caplog.at_level(logging.WARNING):
        agent.memory.add_to_memory(type="plan", content={"reasoning": "x"})
        status = agent.send_message("hello", [recorded_model.agents.to_list()[1]])

    assert agent.memory.step_content["plan"] == {"reasoning": "x"}
    assert "sent message" in status
    assert "Recording 'plan' event" in caplog.text
    assert "Recording 'message' event" in caplog.text


def test_no_recorder_records_nothing(recorded_model):
    agent, _ = recorded_model.agents.to_list()
    agent.recorder = None
    before = len(recorded_model.recorder.events)

    agent.generate_obs()

    assert len(recorded_model.recorder.events) == before


def test_episodic_memory_events_are_recorded(recorded_model, monkeypatch):
    agent, _ = recorded_model.agents.to_list()
    agent.memory = EpisodicMemory(agent=agent, llm_model="gemini/x", display=False)
    monkeypatch.setattr(agent.memory, "grade_event_importance", lambda t, c: 3)

    agent.memory.add_to_memory(type="observation", content={"seen": 1})

    observations = _events(recorded_model, "observation")
    assert len(observations) == 1
    # The recorder keeps the original content, not the graded copy.
    assert observations[0].content == {"seen": 1}
    assert observations[0].metadata == {"source": "EpisodicMemory.add_to_memory"}


def test_agent_viewer_shows_recorded_conversation(recorded_model, capsys):
    sender, recipient = recorded_model.agents.to_list()
    sender.generate_obs()
    sender.send_message("hello", [recipient])

    path = recorded_model.save_recording("recording.json")
    viewer = AgentViewer(str(path))

    assert {e["event_type"] for e in viewer.agent_events[sender.unique_id]} == {
        "observation",
        "message",
    }
    viewer.view_agent_conversations(sender.unique_id)
    output = capsys.readouterr().out
    assert "not found" not in output
    assert "hello" in output
    assert f"To agents [{recipient.unique_id}]" in output


def test_agent_viewer_formats_apply_plan_actions():
    formatted = AgentViewer._format_action_event(
        {"tool_calls": [{"name": "move", "response": "moved to (1, 1)"}]}
    )

    assert formatted == "ACTION: tool calls\nmove -> moved to (1, 1)"
