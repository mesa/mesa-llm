from unittest.mock import AsyncMock, patch

import pytest

from mesa_llm.memory.lt_memory import LongTermMemory
from mesa_llm.memory.memory import MemoryEntry
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.reasoning.reasoning import Observation


class TestLTMemory:
    """Test the Memory class core functionality"""

    def test_memory_initialization(self, mock_agent):
        """Test Memory class initialization with defaults and custom values"""
        memory = LongTermMemory(
            agent=mock_agent,
            llm_model="provider/test_model",
        )

        assert memory.agent == mock_agent
        assert memory.long_term_memory == ""
        assert memory.llm.system_prompt is not None

    def test_update_long_term_memory(self, mock_agent, mock_llm, llm_response_factory):
        """Check that long_term_memory gets the actual text, not the whole response object."""
        mock_llm.generate.return_value = llm_response_factory(
            "Updated long-term memory"
        )

        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")
        memory.llm = mock_llm
        memory.long_term_memory = "Previous memory"

        memory.buffer = MemoryEntry(
            agent=mock_agent,
            content={"message": "Test message"},
            step=1,
        )

        memory._update_long_term_memory()

        call_args = mock_llm.generate.call_args[0][0]
        assert "new memory entry" in call_args
        assert "Long term memory" in call_args

        assert isinstance(memory.long_term_memory, str)
        assert memory.long_term_memory == "Updated long-term memory"

    def test_long_term_memory_stores_string_not_response_object(
        self, mock_agent, mock_llm, llm_response_factory
    ):
        """Make sure long_term_memory is always a plain string.
        Before this fix, it was storing the whole LLM response object instead
        of just the text — which broke any prompt that used the memory.
        """
        mock_llm.generate.return_value = llm_response_factory(
            "This is the summary text"
        )

        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")
        memory.llm = mock_llm
        memory.buffer = MemoryEntry(
            agent=mock_agent, content={"observation": "test"}, step=1
        )

        memory._update_long_term_memory()

        assert isinstance(memory.long_term_memory, str), (
            "long_term_memory must be a string, not a ModelResponse object"
        )
        assert memory.long_term_memory == "This is the summary text"

    # process step test
    def test_process_step(self, mock_agent, llm_response_factory):
        """Test process_step functionality"""
        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")

        # Add some content
        memory.add_to_memory("observation", {"content": "Test observation"})
        memory.add_to_memory("plan", {"content": "Test plan"})

        # Process the step
        with (
            patch("rich.console.Console"),
            patch.object(
                memory.llm,
                "generate",
                return_value=llm_response_factory("mocked summary"),
            ),
        ):
            memory.process_step(pre_step=True)
            assert isinstance(memory.buffer, MemoryEntry)

            memory.process_step(pre_step=False)
            assert memory.long_term_memory == "mocked summary"

    # format memories test
    def test_format_long_term(self, mock_agent):
        """Test formatting long-term memory"""
        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")
        memory.long_term_memory = "Long-term summary"

        assert memory.format_long_term() == "Long-term summary"

    @pytest.mark.asyncio
    async def test_aupdate_long_term_memory(
        self, mock_agent, mock_llm, llm_response_factory
    ):
        """Same as above but for the async version — makes sure it also
        saves just the text, not the whole response object."""
        mock_llm.agenerate = AsyncMock(
            return_value=llm_response_factory("async summary")
        )

        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")
        memory.llm = mock_llm
        memory.buffer = "buffer"

        await memory._aupdate_long_term_memory()

        mock_llm.agenerate.assert_called_once()
        assert isinstance(memory.long_term_memory, str)
        assert memory.long_term_memory == "async summary"

    @pytest.mark.asyncio
    async def test_aprocess_step(self, mock_agent, llm_response_factory):
        """
        Test asynchronous aprocess_step functionality

        This test is performed in 2 parts ,
            - If pre_step = True then a new memory entry is created and this must be verified.
            - If pre_step = False then a according to the aprocess_step function the previous content is restored and this is set to as the new memory entry
              the check verifies this behavior.
        """
        memory = LongTermMemory(agent=mock_agent, llm_model="provider/test_model")

        # populate with content
        memory.add_to_memory("observation", {"content": "Test observation"})
        memory.add_to_memory("plan", {"content": "Test plan"})

        # Mock async LLM call
        memory.llm.agenerate = AsyncMock(
            return_value=llm_response_factory("mocked async summary")
        )

        with patch("rich.console.Console"):
            await memory.aprocess_step(pre_step=True)
            assert isinstance(memory.buffer, MemoryEntry)
            assert memory.buffer.step is None

            await memory.aprocess_step(pre_step=False)
            assert memory.long_term_memory == "mocked async summary"
            assert memory.step_content == {}
            assert memory.buffer.step is not None


class TestLTMemoryCommunicationHistory:
    """Regression tests for LongTermMemory.get_communication_history().

    LongTermMemory previously returned the hardcoded placeholder
    "communication history is in memory of the agent", unlike ShortTermMemory,
    STLTMemory and EpisodicMemory which all render readable message text. The
    placeholder was always truthy, so ReActReasoning injected it into every
    prompt. See issue #327.
    """

    def _memory(self, mock_agent, **kwargs):
        return LongTermMemory(
            agent=mock_agent,
            llm_model="provider/test_model",
            display=False,
            **kwargs,
        )

    def test_get_communication_history_nested_dict(self, mock_agent):
        """The nested dict produced by speak_to renders as readable text."""
        memory = self._memory(mock_agent)

        memory.add_to_memory(
            type="message",
            content={
                "message": "meet me at the north",
                "sender": 7,
                "recipients": [1],
            },
        )

        history = memory.get_communication_history()

        assert "Agent 7 says: meet me at the north" in history
        assert "Step 1:" in history
        assert "communication history is in memory of the agent" not in history

    def test_get_communication_history_multiple_messages(self, mock_agent):
        """Several messages in one step each produce their own line."""
        memory = self._memory(mock_agent)

        memory.add_to_memory(
            type="message",
            content={"message": "first", "sender": 1, "recipients": [2]},
        )
        memory.add_to_memory(
            type="message",
            content={"message": "second", "sender": 2, "recipients": [1]},
        )

        history = memory.get_communication_history()

        assert "Agent 1 says: first" in history
        assert "Agent 2 says: second" in history

    def test_get_communication_history_list_entry(self, mock_agent):
        """A directly appended list-valued entry renders one line per message."""
        memory = self._memory(mock_agent)

        memory.communication_history.append(
            MemoryEntry(
                agent=mock_agent,
                content={
                    "message": [
                        {"message": "first", "sender": 1, "recipients": [2]},
                        {"message": "second", "sender": 2, "recipients": [1]},
                    ]
                },
                step=4,
            )
        )

        history = memory.get_communication_history()

        assert "Step 4: Agent 1 says: first" in history
        assert "Step 4: Agent 2 says: second" in history

    def test_get_communication_history_skips_non_message_entries(self, mock_agent):
        """Plans and observations must not leak into the communication history."""
        memory = self._memory(mock_agent)

        memory.add_to_memory(type="plan", content={"content": "Test plan"})
        memory.add_to_memory(type="observation", content={"content": "Test obs"})
        memory.add_to_memory(
            type="message",
            content={"message": "only this", "sender": 3, "recipients": [1]},
        )

        history = memory.get_communication_history()

        assert "Agent 3 says: only this" in history
        assert "Test plan" not in history
        assert "Test obs" not in history

    def test_get_communication_history_returns_empty_string_when_no_messages(
        self, mock_agent
    ):
        """No messages must yield a falsy string, not the old placeholder.

        This is what stops ReActReasoning from appending an empty
        "last communication:" section on every step.
        """
        memory = self._memory(mock_agent)

        memory.add_to_memory(type="plan", content={"content": "Test plan"})

        assert memory.get_communication_history() == ""
        assert not memory.get_communication_history()

    def test_communication_history_is_bounded(self, mock_agent):
        """The log is bounded, so long simulations cannot grow it without limit."""
        memory = self._memory(mock_agent, communication_history_capacity=3)

        for i in range(5):
            memory.add_to_memory(
                type="message",
                content={"message": f"msg-{i}", "sender": i, "recipients": [9]},
            )

        history = memory.get_communication_history()

        assert len(memory.communication_history) == 3
        assert "msg-0" not in history
        assert "msg-1" not in history
        assert "msg-2" in history
        assert "msg-4" in history

    def test_invalid_communication_history_capacity_raises(self, mock_agent):
        """Capacity below 1 would silently discard every message."""
        with pytest.raises(ValueError, match="communication_history_capacity"):
            self._memory(mock_agent, communication_history_capacity=0)

    @pytest.mark.asyncio
    async def test_async_add_to_memory_records_communication(self, mock_agent):
        """The async path delegates to add_to_memory, so it records too."""
        memory = self._memory(mock_agent)

        await memory.aadd_to_memory(
            type="message",
            content={"message": "async hello", "sender": 5, "recipients": [1]},
        )

        history = memory.get_communication_history()

        assert "Agent 5 says: async hello" in history

    def test_add_to_memory_still_rejects_non_dict_content(self, mock_agent):
        """Base-class validation runs before anything is logged."""
        memory = self._memory(mock_agent)

        with pytest.raises(TypeError):
            memory.add_to_memory(type="message", content="not a dict")

        assert len(memory.communication_history) == 0

    def test_react_prompt_omits_section_when_history_empty(self, mock_agent):
        """ReActReasoning must not append a 'last communication' section.

        Previously the placeholder string was always truthy, so the section was
        appended to every prompt with meaningless content.
        """
        memory = self._memory(mock_agent)
        mock_agent.memory = memory

        reasoning = ReActReasoning(agent=mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})

        prompt_list = reasoning.get_react_prompt(obs)

        assert not any("last communication" in part for part in prompt_list)

        memory.add_to_memory(
            type="message",
            content={"message": "ping", "sender": 2, "recipients": [1]},
        )
        prompt_list = reasoning.get_react_prompt(obs)

        assert any("last communication" in part for part in prompt_list)
        assert any("Agent 2 says: ping" in part for part in prompt_list)
