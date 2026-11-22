import json
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest

from mesa_llm.memory.episodic_memory import (
    EpisodicMemory,
    EventGrade,
    normalize_dict_values,
)
from mesa_llm.memory.memory import MemoryEntry


def test_normalize_dict_floats_logic():
    """
    Function to check whether the values are normalised properly.
        - Hardcoded dict values are used currently to ensure that the normalization logic works.
        - Checks both cases, ie when the range = 0 and when its not 0.
    """
    d = {0: 10, 1: 20, 2: 30}
    norm = normalize_dict_values(d, 0, 1)
    assert norm[0] == 0.0
    assert norm[1] == 0.5
    assert norm[2] == 1.0

    # Checks normalized value when range is 0
    d_tie = {0: 5, 1: 5}
    norm_tie = normalize_dict_values(d_tie, 0, 1)
    assert norm_tie[0] == 0.5
    assert norm_tie[1] == 0.5


def test_normalize_dict_floats_logic_when_empty():
    """
    Function to check whether normalize_dict_values correctly returns an empty dict
    """
    norm = normalize_dict_values({}, 0, 1)
    assert norm == {}


class TestEpisodicMemory:
    """Core functionality test"""

    def test_event_grade_schema_includes_field_description(self):
        """Structured output schema should keep the grading instructions."""
        schema = EventGrade.model_json_schema()

        assert schema["properties"]["grade"]["description"] == (
            "Integer score representing the importance of the event"
        )

    def test_memory_init(self, episodic_mock_agent):
        """Test EpisodicMemory class initialization with defaults and custom values"""
        memory = EpisodicMemory(
            agent=episodic_mock_agent,
            max_capacity=10,
            considered_entries=5,
            llm_model="provider/test_model",
        )

        assert memory.agent == episodic_mock_agent
        assert memory.max_capacity == 10
        assert memory.considered_entries == 5
        assert isinstance(memory.memory_entries, deque)
        assert memory.memory_entries.maxlen == 10
        assert memory.system_prompt is not None
        """FYI: The above line may not always work; use the one below if needed."""
        # assert isinstance(memory.system_prompt,str), memory.system_prompt.strip() != ""

    def test_add_memory_entry(self, episodic_mock_agent):
        """Test adding memories to Episodic memory"""
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({"grade": 3})
        memory.llm.generate = MagicMock(return_value=mock_response)

        # Test basic addition with observation
        memory.add_to_memory("observation", {"step": 1, "content": "Test content"})

        # Test with planning
        memory.add_to_memory("planning", {"plan": "Test plan", "importance": "high"})

        # Test with action
        memory.add_to_memory("action", {"action": "Test action"})

        # EpisodicMemory should not rely on transient step buffers.
        assert memory.step_content == {}
        assert len(memory.memory_entries) == 3, (
            "add_to_memory graded the event but never created a MemoryEntry"
        )

    def test_finalize_entry_consistency(self, mock_agent):
        """Minimal tests for the helper function _finalize_entry().
        - This test ensures that:
        - A `MemoryEntry` object is created and stored in episodic memory.
        - The stored entry contains the correct importance score.
        - The entry is stamped with the current agent step.
        """
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")
        graded_content = {"data": "test", "importance": 4}

        memory._finalize_entry("observation", graded_content)

        assert memory.memory_entries[0].content["observation"]["importance"] == 4
        assert memory.step_content == {}
        assert isinstance(memory.memory_entries[0], MemoryEntry)
        assert memory.memory_entries[0].step == mock_agent.model.steps

    def test_grade_event_importance(self, episodic_mock_agent, llm_response_factory):
        """Test grading event importance"""
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        # 1. Set up a specific grade for this test
        memory.llm.generate = MagicMock(
            return_value=llm_response_factory(content=json.dumps({"grade": 5}))
        )

        # 2. Call the method
        grade = memory.grade_event_importance("observation", {"data": "critical info"})

        # 3. Assert the grade is correct
        assert grade == 5

        # 4. Assert the LLM was called correctly
        memory.llm.generate.assert_called_once()

        # Check that the system prompt was set on the llm object
        assert memory.llm.system_prompt == memory.system_prompt

    def test_retrieve_top_k_entries(self, episodic_mock_agent):
        """Test the sorting logic for retrieving entries (importance - recency_penalty)."""
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )
        # Set current step
        episodic_mock_agent.model.steps = 100

        # Very important but old
        entry_a = MemoryEntry(
            content={"importance": 5, "id": "A"}, step=98, agent=episodic_mock_agent
        )
        # score = 1 - (100 - 99) = 0
        entry_b = MemoryEntry(
            content={"importance": 1, "id": "B"}, step=99, agent=episodic_mock_agent
        )
        # score = 4 - (100 - 90) = -6
        entry_c = MemoryEntry(
            content={"importance": 4, "id": "C"}, step=90, agent=episodic_mock_agent
        )
        # score = 4 - (100 - 95) = -1
        entry_d = MemoryEntry(
            content={"importance": 4, "id": "D"}, step=95, agent=episodic_mock_agent
        )

        memory.memory_entries.extend([entry_a, entry_b, entry_c, entry_d])

        top_entries = memory.retrieve_top_k_entries(1)

        # The highly important memory should win
        assert len(top_entries) == 1
        assert top_entries[0] == entry_a

    def test_process_step_pre_step(self, episodic_mock_agent):
        """
        EpisodicMemory process_step is a no-op: it should not call add_to_memory
        and should not mutate transient buffers.
        """
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        # Pre-populate step_content
        memory.step_content = {"observation": {"data": "test"}}

        # Spy on add_to_memory and call process_step with pre_step=True
        memory.add_to_memory = MagicMock()
        memory.process_step(pre_step=True)

        memory.add_to_memory.assert_not_called()

        # no-op should keep buffer unchanged
        assert memory.step_content == {"observation": {"data": "test"}}

    def test_process_step_pre_step_does_not_append_entries(self, mock_agent):
        """
        Regression test: process_step/pre_step must not append synthetic entries.
        """
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({"grade": 3})
        memory.llm.generate = MagicMock(return_value=mock_response)

        memory.add_to_memory("action", {"action": "real event"})
        existing_entries = list(memory.memory_entries)
        assert len(existing_entries) == 1

        memory.step_content = {"observation": {"data": "buffer-only"}}
        memory.process_step(pre_step=True)

        assert list(memory.memory_entries) == existing_entries
        assert len(memory.memory_entries) == 1
        assert memory.step_content == {"observation": {"data": "buffer-only"}}

    @pytest.mark.asyncio
    async def test_aprocess_step_pre_step(self, episodic_mock_agent):
        """
        Async process_step is also a no-op for episodic memory.
        """
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        memory.step_content = {"observation": {"data": "test"}}
        memory.aadd_to_memory = AsyncMock()

        await memory.aprocess_step(pre_step=True)

        memory.aadd_to_memory.assert_not_awaited()

        assert memory.step_content == {"observation": {"data": "test"}}

    @pytest.mark.asyncio
    async def test_aprocess_step_pre_step_does_not_append_entries(self, mock_agent):
        """
        Async regression test: process_step/pre_step must not append synthetic entries.
        """
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({"grade": 3})))
        ]
        memory.llm.agenerate = AsyncMock(return_value=mock_response)

        await memory.aadd_to_memory("action", {"action": "real event"})
        existing_entries = list(memory.memory_entries)
        assert len(existing_entries) == 1

        memory.step_content = {"observation": {"data": "buffer-only"}}
        await memory.aprocess_step(pre_step=True)

        assert list(memory.memory_entries) == existing_entries
        assert len(memory.memory_entries) == 1
        assert memory.step_content == {"observation": {"data": "buffer-only"}}

    @pytest.mark.asyncio
    async def test_async_add_memory_entry(
        self, episodic_mock_agent, llm_response_factory
    ):
        """
        The aadd_to_memory function assigns an 'importance' value to the content and then calls the add_to_memory function

        The test function does the following
            - mocks the llm to produece a pre-determined grading.
            - then calls the aad_to_memory function
            - checks that entries are persisted directly into memory_entries.
        """
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        memory.llm.agenerate = AsyncMock(
            return_value=llm_response_factory(content=json.dumps({"grade": 3}))
        )

        # adds content into the memory using the async counter part of add_to_memory function
        await memory.aadd_to_memory("observation", {"content": "Test content"})
        await memory.aadd_to_memory("planning", {"plan": "Test plan"})
        await memory.aadd_to_memory("action", {"action": "Test action"})

        new_entry = memory.memory_entries[0]

        for new_entry in memory.memory_entries:
            event_type = next(iter(new_entry.content.keys()))
            assert new_entry.content[event_type]["importance"] == 3
        assert memory.step_content == {}
        assert len(memory.memory_entries) == 3, (
            "aadd_to_memory graded the event but never created a MemoryEntry"
        )

    def test_build_grade_prompt_no_previous_entries(self, episodic_mock_agent):
        """
        The _build_grade_prompt function inserts 'No previous memory entries this message if there are no entries passed to it.

        This test function checks to see if this fall-back indeed works correctly
            - No memory entries are added before the _build_grade_prompt function call
            - So when the memory is empty we expect to see 'No previous memory entries' in the returned prompt.
        """
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        prompt = memory._build_grade_prompt("observation", {"data": "test"})

        # checks if the fallback condition actaually works
        assert "No previous memory entries" in prompt
        assert "observation" in prompt

    def test_build_grade_prompt_with_previous_entries(self, episodic_mock_agent):
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )
        for i in range(3):
            memory.memory_entries.append(
                MemoryEntry(
                    agent=episodic_mock_agent,
                    content={"observation": {"data": f"entry_{i}"}},
                    step=i,
                )
            )

        prompt = memory._build_grade_prompt("observation", {"data": "new"})

        assert prompt.count("previous memory entries:") == 1
        header_pos = prompt.index("previous memory entries:")
        assert header_pos < prompt.index("entry_0")
        assert (
            prompt.index("entry_0") < prompt.index("entry_1") < prompt.index("entry_2")
        )

    def test_get_communication_history(self, episodic_mock_agent):
        """
        Return a formatted string of all messages stored in memory.

        This function:
        - Looks through all memory entries
        - Selects only entries that contain a "message" field
        - Formats each message as: "step <step_number>: <message>"
        - Combines them into one single string

        Returns:
            str: A string containing all communication messages
                from memory, separated by new lines.
        """
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        entry_with_message = MemoryEntry(
            content={"importance": 3, "message": "Hello"},
            step=1,
            agent=episodic_mock_agent,
        )

        entry_without_message = MemoryEntry(
            content={"importance": 2, "data": "No message here"},
            step=2,
            agent=episodic_mock_agent,
        )

        memory.memory_entries.append(entry_with_message)
        memory.memory_entries.append(entry_without_message)

        history = memory.get_communication_history()
        assert history == "Step 1: Hello\n\n"

    def test_get_communication_history_nested_dict(self, episodic_mock_agent):
        """
        Regression test: get_communication_history must render readable text when
        the message entry is a nested dict (the real structure produced by speak_to).

        speak_to stores:
            add_to_memory(type="message", content={"message": <text>, "sender": <id>, ...})

        EpisodicMemory._finalize_entry wraps this under the type key, so:
            entry.content = {"message": {"message": <text>, "sender": <id>, "importance": N}}

        The old code rendered the raw dict; the fixed code must produce
        "Agent <id> says: <text>".
        """
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        # Simulate what _finalize_entry produces after speak_to + importance grading
        entry = MemoryEntry(
            content={
                "message": {
                    "message": "meet me at the north",
                    "sender": 7,
                    "importance": 3,
                }
            },
            step=5,
            agent=episodic_mock_agent,
        )
        memory.memory_entries.append(entry)

        history = memory.get_communication_history()

        assert history == "Step 5: Agent 7 says: meet me at the north\n\n"

    def test_get_communication_history_skips_non_message_entries(
        self, episodic_mock_agent
    ):
        """Entries without a 'message' key must not appear in communication history."""
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        entry_obs = MemoryEntry(
            content={"observation": {"position": (1, 1), "importance": 2}},
            step=3,
            agent=episodic_mock_agent,
        )
        entry_msg = MemoryEntry(
            content={"message": {"message": "hello", "sender": 1, "importance": 2}},
            step=4,
            agent=episodic_mock_agent,
        )
        memory.memory_entries.extend([entry_obs, entry_msg])

        history = memory.get_communication_history()
        assert history == "Step 4: Agent 1 says: hello\n\n"

    def test_retrieve_empty_memory(self, mock_agent):
        """
        Function to verify empty list is returned when retrieval of memory is empty
        """
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        result = memory.retrieve_top_k_entries(3)

        assert result == []

    def test_extract_importance_flat(self, mock_agent):
        """Function to return importance when stored at top level"""
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")
        entry = MemoryEntry(
            content={"importance": 5, "message": "hello"},
            step=1,
            agent=mock_agent,
        )
        result = memory._extract_importance(entry)
        assert result == 5

    def test_extract_importance_nested(self, mock_agent):
        """Should return importance when nested inside another dict"""
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        entry = MemoryEntry(
            content={"message": {"importance": 4, "text": "nested"}},
            step=1,
            agent=mock_agent,
        )
        result = memory._extract_importance(entry)

        assert result == 4

    def test_extract_importance_missing(self, mock_agent):
        """Should fallback to 1 when importance is absent"""
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        entry = MemoryEntry(
            content={"message": {"text": "no importance"}},
            step=1,
            agent=mock_agent,
        )
        result = memory._extract_importance(entry)
        assert result == 1
