"""Record identity, not payload identity, defines a historical occurrence."""

from types import SimpleNamespace

import pytest

from examples.negotiation.agents import (
    _dialogue_event_payloads,
    _recent_memory_content_sources,
    get_dialogue_history,
)
from mesa_llm.memory.memory import MemoryEntry


def _agent():
    return SimpleNamespace(
        unique_id=1,
        model=SimpleNamespace(steps=3, agents=[]),
        memory=SimpleNamespace(step_content={}),
    )


@pytest.mark.parametrize("container", ["short_term_memory", "memory_entries"])
@pytest.mark.parametrize("second_step", [1, 2])
def test_distinct_records_with_shared_content_are_preserved(container, second_step):
    agent = _agent()
    shared = {"message": [{"sender": 7, "message": "Repeated offer."}]}
    first = MemoryEntry(content=shared, step=1, agent=agent)
    second = MemoryEntry(content=shared, step=second_step, agent=agent)
    first._event_order = ["message"]
    second._event_order = ["message"]
    setattr(agent.memory, container, [first, second])
    assert first is not second
    assert first.content is second.content
    # Includes the empty live buffer.
    assert len(_recent_memory_content_sources(agent)) == 3
    line = "- Agent 7: Repeated offer."
    assert get_dialogue_history(agent) == f"{line}\n{line}"
    assert get_dialogue_history(agent, max_messages=1) == line


@pytest.mark.parametrize("alias", ["memory_entries", "buffer", "_current_step_entry"])
def test_same_entry_alias_is_counted_once_but_distinct_record_is_not(alias):
    agent = _agent()
    shared = {"message": [{"sender": 7, "message": "Offer."}]}
    first = MemoryEntry(content=shared, step=1, agent=agent)
    second = MemoryEntry(content=shared, step=2, agent=agent)
    agent.memory.short_term_memory = [first, second]
    setattr(agent.memory, alias, [first] if alias == "memory_entries" else first)
    # A live buffer alias is not a third historical record.
    agent.memory.step_content = shared
    assert len(_recent_memory_content_sources(agent)) == 2
    assert len(list(_dialogue_event_payloads(agent))) == 2
    assert get_dialogue_history(agent).splitlines() == ["- Agent 7: Offer."] * 2


def test_shared_content_records_keep_their_own_sidecars():
    agent = _agent()
    message = {"sender": 7, "message": "Incoming."}
    outgoing = {
        "action": {
            "name": "speak_to",
            "arguments": {"listener_agents_unique_ids": [7], "message": "Outgoing."},
        },
        "result": {"delivered": [7]},
    }
    shared = {"message": [message], "action": [outgoing]}
    first = MemoryEntry(content=shared, step=1, agent=agent)
    second = MemoryEntry(content=shared, step=2, agent=agent)
    first._event_order = ["message", "action"]
    second._event_order = ["action", "message"]
    agent.memory.short_term_memory = [first]
    agent.memory.memory_entries = [second]
    assert list(_dialogue_event_payloads(agent)) == [
        ("message", message),
        ("action", outgoing),
        ("action", outgoing),
        ("message", message),
    ]
    assert shared == {"message": [message], "action": [outgoing]}
    assert first._event_order == ["message", "action"]
    assert second._event_order == ["action", "message"]


def test_distinct_live_content_follows_shared_historical_records():
    agent = _agent()
    shared = {"message": [{"sender": 7, "message": "Old."}]}
    agent.memory.short_term_memory = [
        MemoryEntry(content=shared, step=1, agent=agent),
        MemoryEntry(content=shared, step=2, agent=agent),
    ]
    agent.memory.step_content = {"message": [{"sender": 8, "message": "New."}]}
    agent.memory._step_event_order = ["message"]
    assert get_dialogue_history(agent).splitlines() == [
        "- Agent 7: Old.",
        "- Agent 7: Old.",
        "- Agent 8: New.",
    ]
    assert get_dialogue_history(agent, max_messages=2).splitlines() == [
        "- Agent 7: Old.",
        "- Agent 8: New.",
    ]
