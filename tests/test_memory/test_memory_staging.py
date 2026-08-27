"""Tests for memory staging area additive event handling (issue #137).

Verifies that concurrent events of the same type within a single step
are accumulated rather than overwritten.
"""

import asyncio
from collections import deque
from dataclasses import asdict, astuple, fields
from unittest.mock import AsyncMock, Mock, call

import pytest

from mesa_llm.memory.episodic_memory import EpisodicMemory
from mesa_llm.memory.lt_memory import LongTermMemory
from mesa_llm.memory.memory import Memory, MemoryEntry
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.memory.st_memory import ShortTermMemory


# ---------------------------------------------------------------------------
# Concrete Memory subclass for unit-testing the base class behaviour
# ---------------------------------------------------------------------------
class ConcreteMemory(Memory):
    def get_prompt_ready(self) -> str:
        return ""

    def get_communication_history(self) -> str:
        return ""

    def process_step(self, pre_step: bool = False):
        pass


@pytest.fixture
def agent():
    a = Mock()
    a.__class__.__name__ = "TestAgent"
    a.unique_id = 1
    a.model = Mock()
    a.model.steps = 1
    a.step_prompt = None
    return a


def _make_buffered_memory(kind, agent, llm_response_factory):
    if kind == "short_term":
        return ShortTermMemory(agent=agent, n=5, display=False)

    if kind == "stlt":
        return STLTMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )

    if kind == "long_term":
        mem = LongTermMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        response = llm_response_factory("summary")
        mem.llm.generate = Mock(return_value=response)
        mem.llm.agenerate = AsyncMock(return_value=response)
        return mem

    raise ValueError(f"Unknown memory kind: {kind}")


def _get_finalized_entry(memory):
    if hasattr(memory, "short_term_memory"):
        return list(memory.short_term_memory)[-1]
    return memory.buffer


def _get_staged_entry(memory):
    if isinstance(memory, ShortTermMemory):
        return memory._current_step_entry
    if isinstance(memory, STLTMemory):
        return memory.short_term_memory[-1]
    return memory.buffer


def _final_a12_ordered_events():
    return [
        ("message", {"event": "message 1"}),
        ("action", {"event": "action 1"}),
        ("message", {"event": "message 2"}),
        ("action", {"event": "action 2"}),
    ]


def _final_int_01_partial_short_term(agent, step_content=None):
    """Build the historical minimal ShortTermMemory test double."""
    memory = ShortTermMemory.__new__(ShortTermMemory)
    memory.n = 5
    memory.short_term_memory = deque()
    memory.step_content = {} if step_content is None else step_content
    memory.additive_event_types = {"message", "action"}
    memory.display = False
    memory.agent = agent
    return memory


ADDITIVE_EVENT_CASES = [
    (
        "message",
        {"message": "before", "sender": "A1", "recipients": ["A3"]},
        {"message": "after", "sender": "A2", "recipients": ["A3"]},
    ),
    (
        "action",
        {"tool_calls": [{"name": "wait", "response": "ok"}]},
        {"tool_calls": [{"name": "move", "response": "done"}]},
    ),
]


# ===================================================================
# Tests for Memory.add_to_memory additive behaviour
# ===================================================================
class TestAdditiveMemory:
    """Verify that additive event types accumulate instead of overwriting."""

    def test_single_message_stored_as_list(self, agent):
        mem = ConcreteMemory(agent=agent)
        mem.add_to_memory("message", {"sender": "A1", "msg": "Hello"})
        assert isinstance(mem.step_content["message"], list)
        assert len(mem.step_content["message"]) == 1
        assert mem.step_content["message"][0]["sender"] == "A1"

    def test_multiple_messages_all_preserved(self, agent):
        """Core regression test for issue #137."""
        mem = ConcreteMemory(agent=agent)
        mem.add_to_memory("message", {"sender": "A1", "msg": "Attack!"})
        mem.add_to_memory("message", {"sender": "A2", "msg": "Defend!"})
        mem.add_to_memory("message", {"sender": "A3", "msg": "Retreat!"})

        msgs = mem.step_content["message"]
        assert isinstance(msgs, list)
        assert len(msgs) == 3
        senders = {m["sender"] for m in msgs}
        assert senders == {"A1", "A2", "A3"}

    def test_multiple_actions_all_preserved(self, agent):
        mem = ConcreteMemory(agent=agent)
        mem.add_to_memory("action", {"name": "move", "response": "ok"})
        mem.add_to_memory("action", {"name": "speak", "response": "done"})

        actions = mem.step_content["action"]
        assert isinstance(actions, list)
        assert len(actions) == 2

    def test_observation_still_overwrites(self, agent):
        """Types outside additive_event_types should keep overwrite semantics."""
        mem = ConcreteMemory(agent=agent)
        mem.add_to_memory("observation", {"pos": (0, 0)})
        mem.add_to_memory("observation", {"pos": (1, 1)})

        obs = mem.step_content["observation"]
        assert isinstance(obs, dict)
        assert obs == {"pos": (1, 1)}

    def test_non_additive_types_overwrite(self, agent):
        """Types not in additive_event_types should still overwrite."""
        mem = ConcreteMemory(agent=agent)
        mem.add_to_memory("Plan", {"content": "plan A"})
        mem.add_to_memory("Plan", {"content": "plan B"})

        assert mem.step_content["Plan"] == {"content": "plan B"}

    def test_custom_additive_event_types_can_include_observation(self, agent):
        mem = ConcreteMemory(agent=agent, additive_event_types=["observation"])
        first = {"pos": (0, 0)}
        second = {"pos": (1, 1)}

        mem.add_to_memory("observation", first)
        mem.add_to_memory("observation", second)

        assert mem.step_content["observation"] == [first, second]

    def test_mixed_types_in_same_step(self, agent):
        """Different types coexist correctly in step_content."""
        mem = ConcreteMemory(agent=agent)
        mem.add_to_memory("observation", {"pos": (0, 0)})
        mem.add_to_memory("message", {"sender": "A1", "msg": "hi"})
        mem.add_to_memory("message", {"sender": "A2", "msg": "hey"})
        mem.add_to_memory("Plan", {"content": "do something"})

        assert isinstance(mem.step_content["observation"], dict)
        assert isinstance(mem.step_content["message"], list)
        assert len(mem.step_content["message"]) == 2
        assert isinstance(mem.step_content["Plan"], dict)

    @pytest.mark.parametrize("memory_kind", ["short_term", "stlt", "long_term"])
    @pytest.mark.parametrize(("event_type", "before", "after"), ADDITIVE_EVENT_CASES)
    def test_buffered_memories_preserve_additive_events_across_step_boundary(
        self, agent, llm_response_factory, memory_kind, event_type, before, after
    ):
        """Additive events from both halves of a step must survive finalization."""
        mem = _make_buffered_memory(memory_kind, agent, llm_response_factory)

        mem.add_to_memory(event_type, before)
        mem.process_step(pre_step=True)

        agent.model.steps = 2
        mem.add_to_memory(event_type, after)
        mem.process_step(pre_step=False)

        finalized_entry = _get_finalized_entry(mem)
        assert finalized_entry.content[event_type] == [before, after]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("memory_kind", ["short_term", "stlt", "long_term"])
    async def test_async_buffered_memories_preserve_messages_across_step_boundary(
        self, agent, llm_response_factory, memory_kind
    ):
        """Async finalization should preserve pre-step and post-step messages."""
        before = {"message": "before", "sender": "A1", "recipients": ["A3"]}
        after = {"message": "after", "sender": "A2", "recipients": ["A3"]}
        mem = _make_buffered_memory(memory_kind, agent, llm_response_factory)

        await mem.aadd_to_memory("message", before)
        await mem.aprocess_step(pre_step=True)

        agent.model.steps = 2
        await mem.aadd_to_memory("message", after)
        await mem.aprocess_step(pre_step=False)

        finalized_entry = _get_finalized_entry(mem)
        assert finalized_entry.content["message"] == [before, after]


# ===================================================================
# Tests for MemoryEntry.__str__ with list values
# ===================================================================
class TestMemoryEntryDisplay:
    """Ensure MemoryEntry formats list-valued content correctly."""

    def test_str_with_list_content(self, agent):
        content = {
            "message": [
                {"sender": "A1", "msg": "Attack!"},
                {"sender": "A2", "msg": "Defend!"},
            ]
        }
        entry = MemoryEntry(content=content, step=1, agent=agent)
        result = str(entry)
        assert "A1" in result
        assert "A2" in result
        assert "Attack!" in result
        assert "Defend!" in result


# ===================================================================
# Tests for ShortTermMemory.get_communication_history with lists
# ===================================================================
class TestShortTermCommunicationHistory:
    """Ensure communication history handles list-valued messages."""

    def test_communication_history_with_multiple_messages(self, agent):
        mem = ShortTermMemory(agent=agent, n=5, display=False)

        # Simulate a step with multiple messages
        mem.add_to_memory(
            "message", {"message": "Attack!", "sender": "A1", "recipients": ["A3"]}
        )
        mem.add_to_memory(
            "message", {"message": "Defend!", "sender": "A2", "recipients": ["A3"]}
        )

        # Process pre-step then post-step to finalize
        mem.process_step(pre_step=True)
        agent.model.steps = 2
        mem.process_step(pre_step=False)

        history = mem.get_communication_history()
        assert "Attack!" in history
        assert "Defend!" in history

    def test_communication_history_with_no_messages(self, agent):
        mem = ShortTermMemory(agent=agent, n=5, display=False)
        assert mem.get_communication_history() == ""

    def test_communication_history_skips_non_message_entries(self, agent):
        """Entries without a 'message' key must be skipped."""
        mem = ShortTermMemory(agent=agent, n=5, display=False)
        mem.short_term_memory.append(
            MemoryEntry(agent=agent, content={"observation": {"pos": (1, 1)}}, step=1)
        )
        mem.short_term_memory.append(
            MemoryEntry(
                agent=agent,
                content={"message": {"sender": "A1", "msg": "hi"}},
                step=2,
            )
        )
        history = mem.get_communication_history()
        assert "hi" in history
        assert "observation" not in history

    def test_communication_history_with_legacy_single_message(self, agent):
        """Cover the non-list branch for backward compat with legacy data."""
        mem = ShortTermMemory(agent=agent, n=5, display=False)
        # Directly inject a legacy single-dict message entry
        entry = MemoryEntry(
            agent=agent,
            content={"message": {"sender": "A1", "msg": "legacy"}},
            step=1,
        )
        mem.short_term_memory.append(entry)
        history = mem.get_communication_history()
        assert "legacy" in history


# ===================================================================
# Tests for STLTMemory.get_communication_history with lists
# ===================================================================
class TestSTLTCommunicationHistory:
    """Ensure STLTMemory communication history handles list-valued messages."""

    def test_stlt_communication_history_with_multiple_messages(self, agent):
        mem = STLTMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        # Inject entries with list-valued messages
        entry = MemoryEntry(
            agent=agent,
            content={
                "message": [
                    {"message": "Hello!", "sender": "A1", "recipients": ["A3"]},
                    {"message": "World!", "sender": "A2", "recipients": ["A3"]},
                ]
            },
            step=1,
        )
        mem.short_term_memory.append(entry)
        history = mem.get_communication_history()
        assert "Hello!" in history
        assert "World!" in history

    def test_stlt_communication_history_with_legacy_single_message(self, agent):
        """Cover the non-list branch."""
        mem = STLTMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        entry = MemoryEntry(
            agent=agent,
            content={"message": {"sender": "A1", "msg": "legacy"}},
            step=1,
        )
        mem.short_term_memory.append(entry)
        history = mem.get_communication_history()
        assert "legacy" in history

    def test_stlt_communication_history_skips_non_message_entries(self, agent):
        """Entries without a 'message' key must be skipped."""
        mem = STLTMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        mem.short_term_memory.append(
            MemoryEntry(agent=agent, content={"observation": {"pos": (1, 1)}}, step=1)
        )
        mem.short_term_memory.append(
            MemoryEntry(
                agent=agent,
                content={"message": {"sender": "A1", "msg": "hi"}},
                step=2,
            )
        )
        history = mem.get_communication_history()
        assert "hi" in history
        assert "observation" not in history

    def test_stlt_communication_history_no_messages(self, agent):
        mem = STLTMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        assert mem.get_communication_history() == ""


# ===================================================================
# Tests for EpisodicMemory.get_communication_history with lists
# ===================================================================
class TestEpisodicCommunicationHistory:
    """Ensure EpisodicMemory communication history handles list-valued messages."""

    def test_episodic_communication_history_with_list_messages(self, agent):
        """Cover the list branch in EpisodicMemory.get_communication_history."""
        mem = EpisodicMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        entry = MemoryEntry(
            agent=agent,
            content={
                "message": [
                    {"sender": "A1", "msg": "Attack!"},
                    {"sender": "A2", "msg": "Defend!"},
                ]
            },
            step=1,
        )
        mem.memory_entries.append(entry)
        history = mem.get_communication_history()
        assert "Attack!" in history
        assert "Defend!" in history

    def test_episodic_communication_history_with_scalar_message(self, agent):
        """Cover the non-list (scalar) branch in EpisodicMemory.get_communication_history."""
        mem = EpisodicMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        entry = MemoryEntry(
            agent=agent,
            content={"message": {"sender": "A1", "msg": "solo"}},
            step=1,
        )
        mem.memory_entries.append(entry)
        history = mem.get_communication_history()
        assert "solo" in history

    def test_episodic_communication_history_skips_non_message_entries(self, agent):
        """Entries without a 'message' key must be skipped."""
        mem = EpisodicMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        mem.memory_entries.append(
            MemoryEntry(agent=agent, content={"observation": {"pos": (1, 1)}}, step=1)
        )
        mem.memory_entries.append(
            MemoryEntry(
                agent=agent,
                content={"message": {"sender": "A1", "msg": "hi"}},
                step=2,
            )
        )
        history = mem.get_communication_history()
        assert "hi" in history
        assert "observation" not in history

    def test_episodic_communication_history_empty(self, agent):
        """Empty memory should return empty string."""
        mem = EpisodicMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        assert mem.get_communication_history() == ""


# ===================================================================
# Tests for MemoryEntry.__str__ edge cases
# ===================================================================
class TestMemoryEntryEdgeCases:
    """Cover edge cases in MemoryEntry formatting."""

    def test_str_with_list_of_non_dict_items(self, agent):
        """Cover the branch where list items are not dicts."""
        content = {"action": ["moved north", "picked up item"]}
        entry = MemoryEntry(content=content, step=1, agent=agent)
        result = str(entry)
        assert "moved north" in result
        assert "picked up item" in result

    def test_str_with_scalar_value(self, agent):
        """Cover the else branch where value is neither list nor dict."""
        content = {"status": "idle"}
        entry = MemoryEntry(content=content, step=1, agent=agent)
        result = str(entry)
        assert "idle" in result


# ===================================================================
# Tests for legacy migration path in add_to_memory
# ===================================================================
class TestLegacyMigration:
    """Cover the migration path from single-dict to list in add_to_memory."""

    def test_legacy_single_dict_migrated_to_list(self, agent):
        """If step_content already has a plain dict for an additive type,
        adding another entry should migrate it to a list."""
        mem = ConcreteMemory(agent=agent)
        # Directly inject a legacy single-dict value
        mem.step_content["message"] = {"sender": "A1", "msg": "old"}
        mem.add_to_memory("message", {"sender": "A2", "msg": "new"})

        msgs = mem.step_content["message"]
        assert isinstance(msgs, list)
        assert len(msgs) == 2
        assert msgs[0]["sender"] == "A1"
        assert msgs[1]["sender"] == "A2"


class TestFinalA12EventOrdering:
    """Regression coverage for private cross-event chronology metadata."""

    def test_final_a12_base_memory_records_order_without_changing_content_shape(
        self, agent
    ):
        memory = ConcreteMemory(agent=agent)
        events = _final_a12_ordered_events()

        memory.add_to_memory(*events[0])
        memory.add_to_memory("observation", {"position": [1, 2]})
        for event in events[1:]:
            memory.add_to_memory(*event)

        assert memory.step_content == {
            "message": [events[0][1], events[2][1]],
            "observation": {"position": [1, 2]},
            "action": [events[1][1], events[3][1]],
        }
        assert memory._step_event_order == [
            "message",
            "action",
            "message",
            "action",
        ]
        assert "_event_order" not in memory.step_content

    def test_final_a12_sidecar_is_not_visible_or_part_of_equality(self, agent):
        content = {"message": [{"message": "private sidecar"}]}
        first = MemoryEntry(content=content, step=1, agent=agent)
        second = MemoryEntry(content=content, step=1, agent=agent)
        first._event_order = ["message"]
        second._event_order = ["action"]

        assert first == second
        assert "_event_order" not in repr(first)
        assert "_event_order" not in str(first)
        assert "_event_order" not in first.content

    def test_final_a12_order_sidecar_preserves_public_dataclass_shape(self, agent):
        memory = ShortTermMemory(agent=agent, n=5, display=False)
        for event in _final_a12_ordered_events():
            memory.add_to_memory(*event)
        memory.process_step(pre_step=True)
        memory.process_step(pre_step=False)

        entry = memory.short_term_memory[-1]
        field_names = [field.name for field in fields(entry)]
        dictionary_shape = asdict(entry)
        tuple_shape = astuple(entry)

        assert entry._event_order == ["message", "action", "message", "action"]
        assert field_names == ["content", "step", "agent"]
        assert list(dictionary_shape) == ["content", "step", "agent"]
        assert len(tuple_shape) == 3
        assert dictionary_shape["content"] == entry.content
        assert dictionary_shape["step"] == entry.step
        assert tuple_shape[:2] == (entry.content, entry.step)

    def test_final_a12_sidecar_is_absent_from_prompt_ready_memory(self, agent):
        memory = ShortTermMemory(agent=agent, n=5, display=False)
        for event in _final_a12_ordered_events():
            memory.add_to_memory(*event)

        memory.process_step(pre_step=True)
        memory.process_step(pre_step=False)

        entry = memory.short_term_memory[-1]
        assert entry._event_order == ["message", "action", "message", "action"]
        assert "_event_order" not in entry.content
        assert "_event_order" not in memory.get_prompt_ready()

    @pytest.mark.parametrize("memory_kind", ["short_term", "stlt", "long_term"])
    def test_final_a12_buffered_memory_merges_staged_before_current_order(
        self,
        agent,
        llm_response_factory,
        memory_kind,
    ):
        memory = _make_buffered_memory(memory_kind, agent, llm_response_factory)
        events = _final_a12_ordered_events()

        for event in events[:2]:
            memory.add_to_memory(*event)
        memory.process_step(pre_step=True)

        staged_entry = _get_staged_entry(memory)
        assert staged_entry._event_order == ["message", "action"]
        assert memory._step_event_order == []

        for event in events[2:]:
            memory.add_to_memory(*event)
        assert memory._step_event_order == ["message", "action"]

        agent.model.steps = 2
        memory.process_step(pre_step=False)

        finalized_entry = _get_finalized_entry(memory)
        assert finalized_entry._event_order == [
            "message",
            "action",
            "message",
            "action",
        ]
        assert finalized_entry.content == {
            "message": [events[0][1], events[2][1]],
            "action": [events[1][1], events[3][1]],
        }
        assert memory.step_content == {}
        assert memory._step_event_order == []
        if memory_kind == "long_term":
            memory.llm.generate.assert_called_once()
            memory.llm.agenerate.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("memory_kind", ["short_term", "stlt", "long_term"])
    async def test_final_a12_async_buffered_memory_transfers_and_resets_order(
        self,
        agent,
        llm_response_factory,
        memory_kind,
    ):
        memory = _make_buffered_memory(memory_kind, agent, llm_response_factory)
        events = _final_a12_ordered_events()

        for event in events[:2]:
            await memory.aadd_to_memory(*event)
        await memory.aprocess_step(pre_step=True)
        assert _get_staged_entry(memory)._event_order == ["message", "action"]
        assert memory._step_event_order == []

        for event in events[2:]:
            await memory.aadd_to_memory(*event)
        agent.model.steps = 2
        await memory.aprocess_step(pre_step=False)

        finalized_entry = _get_finalized_entry(memory)
        assert finalized_entry._event_order == [
            "message",
            "action",
            "message",
            "action",
        ]
        assert finalized_entry.content == {
            "message": [events[0][1], events[2][1]],
            "action": [events[1][1], events[3][1]],
        }
        assert memory.step_content == {}
        assert memory._step_event_order == []
        if memory_kind == "long_term":
            memory.llm.generate.assert_not_called()
            memory.llm.agenerate.assert_awaited_once()

    def test_final_a12_episodic_sync_writes_one_marker_and_one_grade_per_event(
        self, agent
    ):
        memory = EpisodicMemory(
            agent=agent,
            llm_model="openai/test",
            display=False,
        )
        events = _final_a12_ordered_events()
        grading = Mock(side_effect=(1, 2, 3, 4))
        memory.grade_event_importance = grading
        memory.llm.generate = Mock(
            side_effect=AssertionError("provider grading must remain mocked")
        )
        memory.llm.agenerate = AsyncMock(
            side_effect=AssertionError("async provider grading must not run")
        )

        for event in events:
            memory.add_to_memory(*event)
        memory.process_step(pre_step=True)
        memory.process_step(pre_step=False)

        assert grading.call_args_list == [call(*event) for event in events]
        assert [entry._event_order for entry in memory.memory_entries] == [
            [event_type] for event_type, _content in events
        ]
        assert [next(iter(entry.content)) for entry in memory.memory_entries] == [
            event_type for event_type, _content in events
        ]
        assert memory.step_content == {}
        assert memory._step_event_order == []
        memory.llm.generate.assert_not_called()
        memory.llm.agenerate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_final_a12_episodic_async_writes_one_marker_and_one_grade_per_event(
        self, agent
    ):
        memory = EpisodicMemory(
            agent=agent,
            llm_model="openai/test",
            display=False,
        )
        events = _final_a12_ordered_events()
        grading = AsyncMock(side_effect=(1, 2, 3, 4))
        memory.agrade_event_importance = grading
        memory.grade_event_importance = Mock(
            side_effect=AssertionError("sync grading must not run")
        )
        memory.llm.generate = Mock(
            side_effect=AssertionError("sync provider must not run")
        )
        memory.llm.agenerate = AsyncMock(
            side_effect=AssertionError("provider grading must remain mocked")
        )

        for event in events:
            await memory.aadd_to_memory(*event)
        await memory.aprocess_step(pre_step=True)
        await memory.aprocess_step(pre_step=False)

        assert grading.await_args_list == [call(*event) for event in events]
        assert [entry._event_order for entry in memory.memory_entries] == [
            [event_type] for event_type, _content in events
        ]
        assert [next(iter(entry.content)) for entry in memory.memory_entries] == [
            event_type for event_type, _content in events
        ]
        assert memory.step_content == {}
        assert memory._step_event_order == []
        memory.grade_event_importance.assert_not_called()
        memory.llm.generate.assert_not_called()
        memory.llm.agenerate.assert_not_awaited()


class TestFinalInt01PartialMemoryCompatibility:
    """Legacy memory instances can lazily adopt event-order tracking."""

    def test_final_int_01_partial_short_term_preserves_legacy_then_orders_fresh(
        self, agent
    ):
        legacy_message = {"message": "legacy message", "sender": 1}
        legacy_action = {"event": "legacy action"}
        legacy_content = {
            "observation": {"position": [1, 2]},
            "message": [legacy_message],
            "action": [legacy_action],
        }
        memory = _final_int_01_partial_short_term(agent, legacy_content)

        memory.process_step(pre_step=True)
        assert memory._current_step_entry.content is legacy_content
        assert memory._current_step_entry._event_order == []

        agent.model.steps = 2
        memory.process_step(pre_step=False)

        legacy_entry = memory.short_term_memory[-1]
        assert legacy_entry.content == legacy_content
        assert legacy_entry.content["message"] == [legacy_message]
        assert legacy_entry.content["action"] == [legacy_action]
        assert legacy_entry._event_order == ["message", "action"]

        fresh_events = _final_a12_ordered_events()
        for event in fresh_events:
            memory.add_to_memory(*event)

        memory.process_step(pre_step=True)
        agent.model.steps = 3
        memory.process_step(pre_step=False)

        fresh_entry = memory.short_term_memory[-1]
        assert fresh_entry.content == {
            "message": [fresh_events[0][1], fresh_events[2][1]],
            "action": [fresh_events[1][1], fresh_events[3][1]],
        }
        assert fresh_entry._event_order == [
            "message",
            "action",
            "message",
            "action",
        ]

    def test_final_int_01_partial_instances_have_isolated_event_order(self, agent):
        first = _final_int_01_partial_short_term(agent)
        second = _final_int_01_partial_short_term(agent)

        first.add_to_memory("message", {"message": "first"})
        second.add_to_memory("action", {"event": "second"})

        assert first._step_event_order == ["message"]
        assert second._step_event_order == ["action"]
        assert first._step_event_order is not second._step_event_order

    def test_final_int_01_partial_instance_recovers_from_malformed_order(self, agent):
        memory = _final_int_01_partial_short_term(agent)
        memory._step_event_order = ("unusable historical marker",)

        memory.add_to_memory("message", {"message": "fresh"})

        assert memory._step_event_order == ["message"]
        assert isinstance(memory._step_event_order, list)

    def test_final_int_01_valid_order_list_preserves_identity(self, agent):
        memory = _final_int_01_partial_short_term(agent)
        existing_order = []
        memory._step_event_order = existing_order

        memory.add_to_memory("action", {"event": "fresh"})

        assert memory._step_event_order is existing_order
        assert existing_order == ["action"]


class _FinalA12ConsolidationAbort(BaseException):
    """Non-Exception abort used to verify transparent rollback."""


def _final_a12_long_term_transition(agent):
    memory = LongTermMemory(
        agent=agent,
        llm_model="openai/test",
        display=False,
    )
    agent.memory = memory
    staged_message = {"message": "staged-message", "sender": 2}
    staged_action = {"event": "staged-action"}
    current_message = {"message": "current-message", "sender": 3}
    current_action = {"event": "current-action"}

    memory.add_to_memory("message", staged_message)
    memory.add_to_memory("action", staged_action)
    memory.process_step(pre_step=True)
    memory.add_to_memory("message", current_message)
    memory.add_to_memory("action", current_action)
    memory.add_to_memory("plan", {"phase": "current"})

    return memory, staged_message, staged_action, current_message, current_action


class TestFinalA12LongTermAtomicConsolidation:
    """Exercise observable atomicity around long-term consolidation."""

    @pytest.mark.asyncio
    async def test_final_a12_atomic_async_success_keeps_reentrant_arrivals_next(
        self,
        agent,
        llm_response_factory,
    ):
        memory, staged_message, staged_action, current_message, current_action = (
            _final_a12_long_term_transition(agent)
        )
        memory.long_term_memory = "old summary"
        entered = asyncio.Event()
        release = asyncio.Event()
        prompts = []
        response = llm_response_factory("new summary")

        async def consolidate(prompt):
            prompts.append(prompt)
            entered.set()
            await release.wait()
            return response

        memory.llm.generate = Mock(
            side_effect=AssertionError("sync provider must not run")
        )
        memory.llm.agenerate = AsyncMock(side_effect=consolidate)

        task = asyncio.create_task(memory.aprocess_step())
        await entered.wait()

        candidate = memory.buffer
        assert candidate.content == {
            "message": [staged_message, current_message],
            "action": [staged_action, current_action],
            "plan": {"phase": "current"},
        }
        assert candidate._event_order == ["message", "action"] * 2
        assert memory.step_content == {}
        assert memory._step_event_order == []

        fresh_message = {"message": "fresh-during-consolidation", "sender": 4}
        fresh_action = {"event": "fresh-action-during-consolidation"}
        fresh_plan = {"phase": "fresh"}
        memory.add_to_memory("message", fresh_message)
        memory.add_to_memory("action", fresh_action)
        memory.add_to_memory("plan", fresh_plan)

        assert "staged-message" in prompts[0]
        assert "current-message" in prompts[0]
        assert "fresh-during-consolidation" not in prompts[0]
        assert "fresh-action-during-consolidation" not in prompts[0]
        assert memory.step_content == {
            "message": [fresh_message],
            "action": [fresh_action],
            "plan": fresh_plan,
        }
        assert memory._step_event_order == ["message", "action"]

        release.set()
        await task

        assert memory.buffer is candidate
        assert memory.long_term_memory == "new summary"
        assert memory.step_content == {
            "message": [fresh_message],
            "action": [fresh_action],
            "plan": fresh_plan,
        }
        assert memory._step_event_order == ["message", "action"]
        memory.llm.agenerate.assert_awaited_once()
        memory.llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_final_a12_atomic_async_failure_restores_aliases_then_retries(
        self,
        agent,
        llm_response_factory,
    ):
        memory, staged_message, staged_action, current_message, current_action = (
            _final_a12_long_term_transition(agent)
        )
        memory.long_term_memory = "old summary"
        staged_entry = memory.buffer
        staged_content = staged_entry.content
        staged_messages = staged_content["message"]
        staged_actions = staged_content["action"]
        current_content = memory.step_content
        current_messages = current_content["message"]
        current_actions = current_content["action"]
        current_order = memory._step_event_order
        failure = RuntimeError("transient consolidation failure")
        fresh_message = {"message": "fresh-after-prompt", "sender": 4}
        fresh_action = {"event": "fresh-action-after-prompt"}
        fresh_plan = {"phase": "fresh"}
        prompts = []
        response = llm_response_factory("retry summary")

        async def consolidate(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                memory.long_term_memory = "partial external mutation"
                memory.add_to_memory("message", fresh_message)
                memory.add_to_memory("action", fresh_action)
                memory.add_to_memory("plan", fresh_plan)
                raise failure
            return response

        memory.llm.generate = Mock(
            side_effect=AssertionError("sync provider must not run")
        )
        memory.llm.agenerate = AsyncMock(side_effect=consolidate)

        with pytest.raises(
            RuntimeError, match="transient consolidation failure"
        ) as exc:
            await memory.aprocess_step()

        assert exc.value is failure
        assert memory.buffer is staged_entry
        assert memory.buffer.step is None
        assert memory.buffer.content is staged_content
        assert memory.buffer.content["message"] is staged_messages
        assert memory.buffer.content["action"] is staged_actions
        assert memory.step_content is current_content
        assert memory.step_content["message"] is current_messages
        assert memory.step_content["action"] is current_actions
        assert memory._step_event_order is current_order
        assert memory.long_term_memory == "old summary"
        assert current_messages == [current_message, fresh_message]
        assert current_actions == [current_action, fresh_action]
        assert current_messages[0] is current_message
        assert current_messages[1] is fresh_message
        assert current_actions[0] is current_action
        assert current_actions[1] is fresh_action
        assert memory.step_content["plan"] is fresh_plan
        assert memory._step_event_order == ["message", "action"] * 2
        assert "fresh-after-prompt" not in prompts[0]

        await memory.aprocess_step()

        assert memory.buffer.content["message"] == [
            staged_message,
            current_message,
            fresh_message,
        ]
        assert memory.buffer.content["action"] == [
            staged_action,
            current_action,
            fresh_action,
        ]
        assert memory.buffer.content["message"][0] is staged_message
        assert memory.buffer.content["message"][1] is current_message
        assert memory.buffer.content["message"][2] is fresh_message
        assert memory.buffer.content["plan"] is fresh_plan
        assert memory.buffer._event_order == [
            "message",
            "action",
            "message",
            "action",
            "message",
            "action",
        ]
        assert memory.step_content == {}
        assert memory._step_event_order == []
        assert memory.long_term_memory == "retry summary"
        assert "fresh-after-prompt" in prompts[1]
        assert memory.llm.agenerate.await_count == 2
        memory.llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_final_a12_atomic_task_cancellation_rolls_back_then_retries(
        self,
        agent,
        llm_response_factory,
    ):
        memory, staged_message, staged_action, current_message, current_action = (
            _final_a12_long_term_transition(agent)
        )
        memory.long_term_memory = "old summary"
        staged_entry = memory.buffer
        current_content = memory.step_content
        current_order = memory._step_event_order
        entered = asyncio.Event()
        attempts = 0
        response = llm_response_factory("retry summary")
        fresh_message = {"message": "fresh-before-cancel", "sender": 4}

        async def consolidate(_prompt):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                entered.set()
                await asyncio.Event().wait()
            return response

        memory.llm.generate = Mock(
            side_effect=AssertionError("sync provider must not run")
        )
        memory.llm.agenerate = AsyncMock(side_effect=consolidate)

        task = asyncio.create_task(memory.aprocess_step())
        await entered.wait()
        memory.add_to_memory("message", fresh_message)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert task.cancelled()
        assert memory.buffer is staged_entry
        assert memory.buffer.step is None
        assert memory.step_content is current_content
        assert memory._step_event_order is current_order
        assert memory.step_content["message"] == [current_message, fresh_message]
        assert memory._step_event_order == ["message", "action", "message"]
        assert memory.long_term_memory == "old summary"

        await memory.aprocess_step()

        assert memory.buffer.content["message"] == [
            staged_message,
            current_message,
            fresh_message,
        ]
        assert memory.buffer.content["action"] == [
            staged_action,
            current_action,
        ]
        assert memory.buffer._event_order == [
            "message",
            "action",
            "message",
            "action",
            "message",
        ]
        assert memory.step_content == {}
        assert memory._step_event_order == []
        assert memory.long_term_memory == "retry summary"
        assert memory.llm.agenerate.await_count == 2
        memory.llm.generate.assert_not_called()

    def test_final_a12_atomic_sync_success_displays_only_committed_candidate(
        self,
        agent,
        llm_response_factory,
        monkeypatch,
    ):
        memory, staged_message, staged_action, current_message, current_action = (
            _final_a12_long_term_transition(agent)
        )
        memory.display = True
        fresh_message = {"message": "fresh-sync-success", "sender": 4}
        prompts = []
        displayed = []

        def consolidate(prompt):
            prompts.append(prompt)
            memory.add_to_memory("message", fresh_message)
            return llm_response_factory("sync summary")

        def capture_display(entry):
            displayed.append(entry)

        monkeypatch.setattr(MemoryEntry, "display", capture_display)
        memory.llm.generate = Mock(side_effect=consolidate)
        memory.llm.agenerate = AsyncMock(
            side_effect=AssertionError("async provider must not run")
        )

        memory.process_step()

        assert displayed == [memory.buffer]
        assert memory.buffer.content["message"] == [staged_message, current_message]
        assert memory.buffer.content["action"] == [staged_action, current_action]
        assert fresh_message not in memory.buffer.content["message"]
        assert memory.step_content == {"message": [fresh_message]}
        assert memory._step_event_order == ["message"]
        assert "fresh-sync-success" not in prompts[0]
        assert memory.long_term_memory == "sync summary"
        memory.llm.generate.assert_called_once()
        memory.llm.agenerate.assert_not_awaited()

    def test_final_a12_atomic_sync_base_exception_is_identical_and_retryable(
        self,
        agent,
        llm_response_factory,
    ):
        memory, staged_message, staged_action, current_message, current_action = (
            _final_a12_long_term_transition(agent)
        )
        memory.long_term_memory = "old summary"
        staged_entry = memory.buffer
        current_content = memory.step_content
        current_order = memory._step_event_order
        abort = _FinalA12ConsolidationAbort("stop now")
        fresh_action = {"event": "fresh-before-sync-abort"}
        calls = 0

        def consolidate(_prompt):
            nonlocal calls
            calls += 1
            if calls == 1:
                memory.long_term_memory = "partial external mutation"
                memory.add_to_memory("action", fresh_action)
                raise abort
            return llm_response_factory("sync retry summary")

        memory.llm.generate = Mock(side_effect=consolidate)
        memory.llm.agenerate = AsyncMock(
            side_effect=AssertionError("async provider must not run")
        )

        with pytest.raises(_FinalA12ConsolidationAbort) as exc:
            memory.process_step()

        assert exc.value is abort
        assert memory.buffer is staged_entry
        assert memory.buffer.step is None
        assert memory.step_content is current_content
        assert memory._step_event_order is current_order
        assert memory.long_term_memory == "old summary"
        assert memory.step_content["action"] == [current_action, fresh_action]
        assert memory._step_event_order == ["message", "action", "action"]

        memory.process_step()

        assert memory.buffer.content["message"] == [staged_message, current_message]
        assert memory.buffer.content["action"] == [
            staged_action,
            current_action,
            fresh_action,
        ]
        assert memory.buffer._event_order == [
            "message",
            "action",
            "message",
            "action",
            "action",
        ]
        assert memory.step_content == {}
        assert memory._step_event_order == []
        assert memory.long_term_memory == "sync retry summary"
        assert memory.llm.generate.call_count == 2
        memory.llm.agenerate.assert_not_awaited()
