from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples.negotiation.agents import get_dialogue_history
from mesa_llm.memory.lt_memory import LongTermMemory
from mesa_llm.memory.memory import _ordered_event_payloads
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.memory.st_memory import ShortTermMemory


class SellerAgent:
    def __init__(self):
        self.unique_id = 1
        self.step_prompt = None
        self.model = SimpleNamespace(steps=1, agents=[])
        self.memory = None


class BuyerAgent:
    def __init__(self):
        self.unique_id = 2


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _make_memory(memory_kind: str, agent):
    if memory_kind == "short_term":
        return ShortTermMemory(agent=agent, n=5, display=False)
    if memory_kind == "stlt":
        return STLTMemory(
            agent=agent,
            llm_model="gemini/gemini-2.0-flash",
            display=False,
        )
    if memory_kind == "long_term":
        memory = LongTermMemory(
            agent=agent,
            llm_model="gemini/gemini-2.0-flash",
            display=False,
        )
        memory.llm.generate = Mock(return_value=_response("summary"))
        return memory
    raise AssertionError(f"unknown memory kind: {memory_kind}")


def _finalized_entry(memory_kind: str, memory):
    if memory_kind == "long_term":
        return memory.buffer
    return list(memory.short_term_memory)[-1]


def _speak_to_event(message: str, recipient_id: int = 2):
    return {
        "action": {
            "name": "speak_to",
            "arguments": {
                "listener_agents_unique_ids": [recipient_id],
                "message": message,
            },
        },
        "result": {
            "requested": [recipient_id],
            "delivered": [recipient_id],
            "skipped": [],
            "failed": [],
        },
    }


@pytest.mark.parametrize("memory_kind", ["short_term", "stlt", "long_term"])
def test_legacy_staged_message_precedes_fresh_reply(memory_kind):
    agent = SellerAgent()
    buyer = BuyerAgent()
    agent.model.agents = [agent, buyer]
    memory = _make_memory(memory_kind, agent)
    agent.memory = memory

    incoming = {"message": "Offer first", "sender": buyer.unique_id}
    reply = _speak_to_event("Reply second", buyer.unique_id)

    # Simulate a legacy or partially initialized staged segment whose grouped
    # content exists but whose private chronology sidecar does not.
    memory.step_content = {"message": [incoming]}
    memory._step_event_order = []
    memory.process_step(pre_step=True)

    memory.add_to_memory("action", reply)
    agent.model.steps = 2
    memory.process_step(pre_step=False)

    entry = _finalized_entry(memory_kind, memory)
    assert entry._event_order == ["message", "action"]
    assert list(
        _ordered_event_payloads(
            entry.content,
            entry._event_order,
            ("message", "action"),
        )
    ) == [("message", incoming), ("action", reply)]

    history = get_dialogue_history(agent, max_messages=10)
    assert history.index("Offer first") < history.index("Reply second")


@pytest.mark.parametrize("memory_kind", ["short_term", "stlt", "long_term"])
def test_each_step_segment_is_normalized_before_order_concatenation(memory_kind):
    agent = SellerAgent()
    memory = _make_memory(memory_kind, agent)
    agent.memory = memory

    staged_message = {"event": "staged message"}
    staged_action = {"event": "staged action"}
    current_message = {"event": "current message"}
    current_action = {"event": "current action"}

    memory.step_content = {
        "message": [staged_message],
        "action": [staged_action],
    }
    memory._step_event_order = []
    memory.process_step(pre_step=True)

    memory.add_to_memory("message", current_message)
    memory.add_to_memory("action", current_action)
    agent.model.steps = 2
    memory.process_step(pre_step=False)

    entry = _finalized_entry(memory_kind, memory)
    assert entry._event_order == [
        "message",
        "action",
        "message",
        "action",
    ]
    assert list(
        _ordered_event_payloads(
            entry.content,
            entry._event_order,
            ("message", "action"),
        )
    ) == [
        ("message", staged_message),
        ("action", staged_action),
        ("message", current_message),
        ("action", current_action),
    ]


def test_non_additive_list_does_not_invalidate_dialogue_sidecar():
    message = {"event": "message"}
    action = {"event": "action"}
    content = {
        "action": [action],
        "observation": ["list-valued", "but not additive chronology"],
        "message": [message],
    }

    assert list(
        _ordered_event_payloads(
            content,
            ["message", "action"],
            ("message", "action"),
        )
    ) == [("message", message), ("action", action)]


@pytest.mark.parametrize("memory_kind", ["short_term", "stlt", "long_term"])
def test_fresh_append_repairs_legacy_current_order_before_and_after_finalization(
    memory_kind,
):
    agent = SellerAgent()
    buyer = BuyerAgent()
    agent.model.agents = [agent, buyer]
    memory = _make_memory(memory_kind, agent)
    agent.memory = memory

    old_message = {"message": "Old incoming", "sender": buyer.unique_id}
    old_action = _speak_to_event("Old reply", buyer.unique_id)
    fresh_message = {"message": "Fresh incoming", "sender": buyer.unique_id}

    memory.step_content = {
        "message": [old_message],
        "action": [old_action],
    }
    memory._step_event_order = []
    event_order_alias = memory._step_event_order

    memory.add_to_memory("message", fresh_message)

    assert memory._step_event_order is event_order_alias
    assert memory._step_event_order == ["message", "action", "message"]
    assert list(
        _ordered_event_payloads(
            memory.step_content,
            memory._step_event_order,
            ("message", "action"),
        )
    ) == [
        ("message", old_message),
        ("action", old_action),
        ("message", fresh_message),
    ]

    current_history = get_dialogue_history(agent, max_messages=10)
    assert current_history.index("Old incoming") < current_history.index("Old reply")
    assert current_history.index("Old reply") < current_history.index("Fresh incoming")

    memory.process_step(pre_step=True)
    agent.model.steps = 2
    memory.process_step(pre_step=False)

    entry = _finalized_entry(memory_kind, memory)
    assert entry._event_order == ["message", "action", "message"]
    assert list(
        _ordered_event_payloads(
            entry.content,
            entry._event_order,
            ("message", "action"),
        )
    ) == [
        ("message", old_message),
        ("action", old_action),
        ("message", fresh_message),
    ]

    finalized_history = get_dialogue_history(agent, max_messages=10)
    assert finalized_history.index("Old incoming") < finalized_history.index(
        "Old reply"
    )
    assert finalized_history.index("Old reply") < finalized_history.index(
        "Fresh incoming"
    )


def test_long_term_rollback_preserves_legacy_current_before_reentrant_fresh():
    agent = SellerAgent()
    memory = _make_memory("long_term", agent)
    agent.memory = memory

    # Install an empty staged half so the next process_step enters consolidation.
    memory.process_step(pre_step=True)

    old_message = {"event": "old message"}
    old_action = {"event": "old action"}
    fresh_message = {"event": "fresh message"}
    memory.step_content = {
        "message": [old_message],
        "action": [old_action],
    }
    memory._step_event_order = []
    current_content = memory.step_content
    current_order = memory._step_event_order
    failure = RuntimeError("consolidation failed")

    def fail_with_reentrant_arrival(_prompt):
        memory.add_to_memory("message", fresh_message)
        raise failure

    memory.llm.generate = Mock(side_effect=fail_with_reentrant_arrival)

    with pytest.raises(RuntimeError, match="consolidation failed") as exc_info:
        memory.process_step(pre_step=False)

    assert exc_info.value is failure
    assert memory.step_content is current_content
    assert memory._step_event_order is current_order
    assert memory.step_content == {
        "message": [old_message, fresh_message],
        "action": [old_action],
    }
    assert memory._step_event_order == ["message", "action", "message"]
    assert list(
        _ordered_event_payloads(
            memory.step_content,
            memory._step_event_order,
            ("message", "action"),
        )
    ) == [
        ("message", old_message),
        ("action", old_action),
        ("message", fresh_message),
    ]

    memory.llm.generate = Mock(return_value=_response("retry summary"))
    memory.process_step(pre_step=False)

    assert memory.buffer._event_order == ["message", "action", "message"]
    assert list(
        _ordered_event_payloads(
            memory.buffer.content,
            memory.buffer._event_order,
            ("message", "action"),
        )
    ) == [
        ("message", old_message),
        ("action", old_action),
        ("message", fresh_message),
    ]
    assert memory.long_term_memory == "retry summary"
