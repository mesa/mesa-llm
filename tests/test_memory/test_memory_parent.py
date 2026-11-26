import asyncio
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from mesa_llm.memory.episodic_memory import EpisodicMemory
from mesa_llm.memory.lt_memory import LongTermMemory
from mesa_llm.memory.memory import Memory, MemoryEntry, _format_message_entry
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.module_llm import ModuleLLM

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class TestMemoryEntry:
    """Test the MemoryEntry dataclass"""

    def test_memory_entry_creation(self, mock_agent):
        """Test MemoryEntry creation and basic functionality"""

        content = {"observation": "Test content", "metadata": "value"}
        entry = MemoryEntry(content=content, step=1, agent=mock_agent)

        assert entry.content == content
        assert entry.step == 1
        assert entry.agent == mock_agent

    def test_memory_entry_str(self, mock_agent):
        """Test MemoryEntry string representation"""

        content = {"observation": "Test content", "type": "observation"}
        entry = MemoryEntry(content=content, step=1, agent=mock_agent)

        str_repr = str(entry)
        assert "Test content" in str_repr
        assert "observation" in str_repr

    def test_memory_entry_str_with_list_of_dicts(self):
        """Test MemoryEntry string representation with list values (e.g. tool_calls)."""
        mock_agent = Mock()
        content = {
            "action": [
                {"name": "move_one_step", "response": "moved"},
                {"name": "arrest_citizen", "response": "arrested"},
            ]
        }
        entry = MemoryEntry(content=content, step=1, agent=mock_agent)
        str_repr = str(entry)
        assert "move_one_step" in str_repr
        assert "arrest_citizen" in str_repr

    def test_memory_entry_str_with_list_of_strings(self):
        """Test MemoryEntry string representation with a list of plain strings."""
        mock_agent = Mock()
        content = {"tags": ["alpha", "beta"]}
        entry = MemoryEntry(content=content, step=1, agent=mock_agent)
        str_repr = str(entry)
        assert "alpha" in str_repr
        assert "beta" in str_repr

    def test_memory_entry_str_with_nested_tool_calls_list(self):
        """Test MemoryEntry string representation with nested tool_calls list under action."""
        mock_agent = Mock()
        content = {
            "action": {
                "tool_calls": [
                    {"name": "move_one_step", "response": "moved"},
                    {"name": "arrest_citizen", "response": "arrested"},
                ]
            }
        }
        entry = MemoryEntry(content=content, step=1, agent=mock_agent)
        str_repr = str(entry)
        assert "move_one_step" in str_repr
        assert "arrest_citizen" in str_repr


class MemoryMock(Memory):
    def __init__(
        self,
        agent: "LLMAgent",
        llm_model: str | None = None,
        display: bool = True,
        additive_event_types: list[str] | set[str] | tuple[str, ...] | None = None,
    ):
        super().__init__(
            agent,
            llm_model,
            display,
            additive_event_types=additive_event_types,
        )

    def get_prompt_ready(self) -> str:
        return ""

    def get_communication_history(self) -> str:
        return ""

    def process_step(self, pre_step: bool = False):
        """
        Mock implementation of process_step for testing purposes.
        Since this is a test mock, we can use a simple pass implementation.
        """


class TestMemoryParent:
    """Test the Memory class"""

    def test_memory_init(self):
        """Test the init of Memory class"""
        mock_agent = Mock()
        memory = MemoryMock(agent=mock_agent, llm_model="provider/test_model")

        # Parameters init
        assert memory.display
        assert memory.step_content == {}
        assert memory.additive_event_types == {"message", "action"}

        # llm init with ModuleLLM
        assert isinstance(memory.llm, ModuleLLM)
        assert memory.llm.llm_model == "provider/test_model"

        memory = MemoryMock(agent=mock_agent)
        assert not hasattr(memory, "llm")

    def test_memory_init_custom_additive_event_types(self):
        """Custom additive event types should be configurable per memory."""
        mock_agent = Mock()
        memory = MemoryMock(
            agent=mock_agent, additive_event_types=["message", "observation"]
        )

        assert memory.additive_event_types == {"message", "observation"}

    def test_memory_init_empty_additive_event_types(self):
        """An explicit empty additive config should stay empty."""
        mock_agent = Mock()
        memory = MemoryMock(agent=mock_agent, additive_event_types=[])

        assert memory.additive_event_types == set()

    def test_add_to_memory(self, mock_agent):
        memory = MemoryMock(agent=mock_agent)
        # Test basic addition with observation
        memory.add_to_memory("observation", {"step": 1, "content": "Test content"})

        # Test with planning
        memory.add_to_memory("planning", {"plan": "Test plan", "importance": "high"})

        # Test with action
        memory.add_to_memory("action", {"action": "Test action"})

        # Should be non-empty step_content after adding to memory
        assert memory.step_content != {}
        assert "observation" in memory.step_content

    def test_add_to_memory_rejects_non_dict_content(self, mock_agent):
        memory = MemoryMock(agent=mock_agent)

        with pytest.raises(TypeError) as exc_info:
            memory.add_to_memory("plan", "raw string plan")

        assert (
            str(exc_info.value)
            == "Expected 'content' to be dict, got str: 'raw string plan'"
        )

    def test_aadd_to_memory_rejects_non_dict_content(self, mock_agent):
        memory = MemoryMock(agent=mock_agent)

        with pytest.raises(TypeError) as exc_info:
            asyncio.run(memory.aadd_to_memory("plan", "raw async string plan"))

        assert (
            str(exc_info.value)
            == "Expected 'content' to be dict, got str: 'raw async string plan'"
        )


class TestFormatMessageEntry:
    """Unit tests for the _format_message_entry helper."""

    def test_plain_string_passthrough(self):
        """Legacy/test entries that store message as a plain string are returned as-is."""
        assert _format_message_entry("Hello") == "Hello"

    def test_nested_dict_with_sender(self):
        """Real speak_to payload: dict with 'message' text and 'sender' id."""
        msg = {"message": "hello world", "sender": 42, "recipients": [7]}
        assert _format_message_entry(msg) == "Agent 42 says: hello world"

    def test_nested_dict_without_sender(self):
        """Dict with message text but no sender — render text only."""
        msg = {"message": "standalone note"}
        assert _format_message_entry(msg) == "standalone note"

    def test_nested_dict_without_message_key_falls_back_to_str(self):
        """Dict lacking 'message' key falls back to str() of the whole dict."""
        msg = {"foo": "bar"}
        assert _format_message_entry(msg) == str(msg)

    def test_episodic_payload_with_importance(self):
        """EpisodicMemory adds 'importance' to the content dict — should still format cleanly."""
        msg = {"message": "critical update", "sender": 5, "importance": 4}
        assert _format_message_entry(msg) == "Agent 5 says: critical update"


def test_memory_subclasses_forward_completion_kwargs(mock_agent):
    """LLM-backed memory subclasses should forward completion_kwargs to their ModuleLLM."""
    sampling = {"temperature": 0.1, "seed": 123}

    stlt = STLTMemory(
        agent=mock_agent, llm_model="openai/gpt-4o-mini", completion_kwargs=sampling
    )
    lt = LongTermMemory(
        agent=mock_agent, llm_model="openai/gpt-4o-mini", completion_kwargs=sampling
    )
    ep = EpisodicMemory(
        agent=mock_agent, llm_model="openai/gpt-4o-mini", completion_kwargs=sampling
    )

    assert stlt.llm.completion_kwargs == sampling
    assert lt.llm.completion_kwargs == sampling
    assert ep.llm.completion_kwargs == sampling
