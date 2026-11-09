"""Regression tests for STLTMemory when LLM consolidation fails (issue #359).

Rule under test: a short-term entry is only removed after its consolidation into
long-term memory succeeded. On failure the batch stays at the left of the deque,
in order, long-term memory is unchanged, and the exception propagates.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mesa_llm.memory.st_lt_memory import STLTMemory

CAPACITY = 2
CONSOLIDATION = 2


def make_memory(mock_agent, mock_llm):
    memory = STLTMemory(
        agent=mock_agent,
        short_term_capacity=CAPACITY,
        consolidation_capacity=CONSOLIDATION,
        display=False,
        llm_model="provider/test_model",
    )
    memory.llm = mock_llm
    return memory


def run_step(memory, agent, i):
    memory.add_to_memory("observation", {"content": f"step_{i}"})
    memory.process_step(pre_step=True)
    agent.model.steps = i + 1
    memory.process_step(pre_step=False)


async def arun_step(memory, agent, i):
    memory.add_to_memory("observation", {"content": f"step_{i}"})
    await memory.aprocess_step(pre_step=True)
    agent.model.steps = i + 1
    await memory.aprocess_step(pre_step=False)


def fill_to_threshold(memory, agent):
    """Fill STM to capacity + consolidation_capacity: the next step consolidates."""
    for i in range(CAPACITY + CONSOLIDATION):
        run_step(memory, agent, i)
    assert len(memory.short_term_memory) == CAPACITY + CONSOLIDATION


async def afill_to_threshold(memory, agent):
    for i in range(CAPACITY + CONSOLIDATION):
        await arun_step(memory, agent, i)
    assert len(memory.short_term_memory) == CAPACITY + CONSOLIDATION


def batch_of(memory):
    """The entries that will be consolidated next (oldest first)."""
    return list(memory.short_term_memory)[:CONSOLIDATION]


def assert_same_objects(actual, expected):
    """Compare by identity: the very same MemoryEntry objects, same order."""
    assert len(actual) == len(expected)
    assert all(a is b for a, b in zip(actual, expected, strict=True))


def observation_texts(memory):
    return [e.content["observation"]["content"] for e in memory.short_term_memory]


def response_with_content(content):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


INVALID_RESPONSES = {
    "none": response_with_content(None),
    "empty": response_with_content(""),
    "whitespace": response_with_content("   \n"),
    "no_choices": SimpleNamespace(choices=[]),
}


def test_sync_failure_keeps_batch_and_ltm(mock_agent, mock_llm):
    memory = make_memory(mock_agent, mock_llm)
    fill_to_threshold(memory, mock_agent)
    batch = batch_of(memory)
    mock_llm.generate.side_effect = RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):
        run_step(memory, mock_agent, 4)

    assert_same_objects(batch_of(memory), batch)
    assert memory.long_term_memory == ""


@pytest.mark.asyncio
async def test_async_failure_keeps_batch_and_ltm(mock_agent, mock_llm):
    memory = make_memory(mock_agent, mock_llm)
    await afill_to_threshold(memory, mock_agent)
    batch = batch_of(memory)
    mock_llm.agenerate = AsyncMock(side_effect=RuntimeError("provider down"))

    with pytest.raises(RuntimeError, match="provider down"):
        await arun_step(memory, mock_agent, 4)

    assert_same_objects(batch_of(memory), batch)
    assert memory.long_term_memory == ""


@pytest.mark.asyncio
async def test_async_cancellation_keeps_batch(mock_agent, mock_llm):
    memory = make_memory(mock_agent, mock_llm)
    await afill_to_threshold(memory, mock_agent)
    batch = batch_of(memory)
    mock_llm.agenerate = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await arun_step(memory, mock_agent, 4)

    assert_same_objects(batch_of(memory), batch)


@pytest.mark.parametrize("kind", [*INVALID_RESPONSES, "bare_mock"])
def test_sync_invalid_response_keeps_batch_and_ltm(mock_agent, mock_llm, kind):
    memory = make_memory(mock_agent, mock_llm)
    fill_to_threshold(memory, mock_agent)
    memory.long_term_memory = "good summary"
    batch = batch_of(memory)
    mock_llm.generate.return_value = (
        Mock() if kind == "bare_mock" else INVALID_RESPONSES[kind]
    )

    with pytest.raises(ValueError):
        run_step(memory, mock_agent, 4)

    assert_same_objects(batch_of(memory), batch)
    assert memory.long_term_memory == "good summary"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(INVALID_RESPONSES))
async def test_async_invalid_response_keeps_batch_and_ltm(mock_agent, mock_llm, kind):
    memory = make_memory(mock_agent, mock_llm)
    await afill_to_threshold(memory, mock_agent)
    memory.long_term_memory = "good summary"
    batch = batch_of(memory)
    mock_llm.agenerate = AsyncMock(return_value=INVALID_RESPONSES[kind])

    with pytest.raises(ValueError):
        await arun_step(memory, mock_agent, 4)

    assert_same_objects(batch_of(memory), batch)
    assert memory.long_term_memory == "good summary"


def test_stm_keeps_accepting_observations_during_failure(mock_agent, mock_llm):
    memory = make_memory(mock_agent, mock_llm)
    fill_to_threshold(memory, mock_agent)
    mock_llm.generate.side_effect = RuntimeError("provider down")

    for i in (4, 5, 6):
        with pytest.raises(RuntimeError):
            run_step(memory, mock_agent, i)

    assert observation_texts(memory)[-3:] == ["step_4", "step_5", "step_6"]
    assert len(memory.short_term_memory) == CAPACITY + CONSOLIDATION + 3


def test_retry_consolidates_same_batch_in_order(
    mock_agent, mock_llm, llm_response_factory
):
    memory = make_memory(mock_agent, mock_llm)
    fill_to_threshold(memory, mock_agent)
    mock_llm.generate.side_effect = RuntimeError("provider down")
    with pytest.raises(RuntimeError):
        run_step(memory, mock_agent, 4)

    mock_llm.generate.side_effect = None
    mock_llm.generate.return_value = llm_response_factory("recovered")
    run_step(memory, mock_agent, 5)

    prompt = mock_llm.generate.call_args[0][0]
    to_consolidate = prompt.split("Existing long term memory")[0]
    assert to_consolidate.index("step_0") < to_consolidate.index("step_1")
    assert "step_2" not in to_consolidate
    assert memory.long_term_memory == "recovered"


def test_success_removes_exactly_the_consolidated_batch(
    mock_agent, mock_llm, llm_response_factory
):
    memory = make_memory(mock_agent, mock_llm)
    mock_llm.generate.return_value = llm_response_factory("summary")
    fill_to_threshold(memory, mock_agent)
    before = list(memory.short_term_memory)

    run_step(memory, mock_agent, 4)

    after = list(memory.short_term_memory)
    assert_same_objects(after[:-1], before[CONSOLIDATION:])
    assert observation_texts(memory)[-1] == "step_4"
    assert memory.long_term_memory == "summary"
