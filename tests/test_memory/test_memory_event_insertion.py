"""Insertion scaling and compatibility of mutable event-order metadata."""

import asyncio
import copy
import operator
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mesa_llm.memory.lt_memory import LongTermMemory
from mesa_llm.memory.memory import _ordered_event_payloads
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.memory.st_memory import ShortTermMemory


def _memory(kind="short_term"):
    agent = SimpleNamespace(model=SimpleNamespace(steps=1), step_prompt="")
    if kind == "short_term":
        memory = ShortTermMemory(agent=agent, display=False)
    else:
        cls = STLTMemory if kind == "stlt" else LongTermMemory
        memory = cls(agent=agent, display=False, llm_model="openai/test")
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))]
        )
        memory.llm.generate = Mock(return_value=response)
        memory.llm.agenerate = AsyncMock(return_value=response)
    agent.memory = memory
    return memory


def _append(memory, count):
    for index in range(count):
        memory.add_to_memory(
            "message" if index % 2 == 0 else "action", {"index": index}
        )


def _events(content, order):
    return _ordered_event_payloads(content, order, ("message", "action"))


@pytest.mark.parametrize("count", [512, 2048])
def test_ordinary_appends_do_not_revisit_buffered_payloads(count):
    memory = _memory()
    _append(memory, 2)
    visits = []

    class CountedList(list):
        def __iter__(self):
            for value in super().__iter__():
                visits.append(1)
                yield value

    for kind in ("message", "action"):
        memory.step_content[kind] = CountedList(memory.step_content[kind])
    normalize = Mock(wraps=memory._normalized_step_event_order)
    memory._normalized_step_event_order = normalize
    _append(memory, count)
    assert normalize.call_count <= 1
    assert len(visits) <= 4
    assert memory._step_event_order == ["message", "action"] * ((count + 2) // 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["short_term", "stlt", "long_term"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_fast_appends_survive_staging_and_repeated_finalization(
    kind, asynchronous
):
    memory = _memory(kind)
    normalize = Mock(wraps=memory._normalized_step_event_order)
    memory._normalized_step_event_order = normalize

    async def process(pre_step):
        if asynchronous:
            await memory.aprocess_step(pre_step=pre_step)
        else:
            memory.process_step(pre_step=pre_step)

    for step in range(1, 4):
        memory.agent.model.steps = step
        for pre_step in (True, False):
            normalize.reset_mock()
            for index in range(100):
                event_type = "message" if index % 2 == 0 else "action"
                payload = {"step": step, "pre_step": pre_step, "index": index}
                if asynchronous:
                    await memory.aadd_to_memory(event_type, payload)
                else:
                    memory.add_to_memory(event_type, payload)
            assert normalize.call_count <= 1
            await process(pre_step)
        entry = memory.buffer if kind == "long_term" else memory.short_term_memory[-1]
        assert entry._event_order == ["message", "action"] * 100
        events = _events(entry.content, entry._event_order)
        assert len(events) == 200
        assert [payload["pre_step"] for _, payload in events] == (
            [True] * 100 + [False] * 100
        )


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "mutate",
    [
        lambda order: operator.setitem(order, 0, "invalid"),
        lambda order: operator.setitem(order, slice(None), ["action"] * 4),
        lambda order: operator.delitem(order, slice(1, 3)),
        lambda order: order.append("message"),
        lambda order: order.extend(["action", "invalid"]),
        lambda order: order.insert(1, "invalid"),
        lambda order: order.pop(),
        lambda order: order.remove("message"),
        lambda order: order.clear(),
        lambda order: order.reverse(),
        lambda order: order.sort(),
        lambda order: operator.iadd(order, ["invalid"]),
        lambda order: operator.imul(order, 2),
    ],
    ids=[
        "item",
        "slice",
        "delete",
        "append",
        "extend",
        "insert",
        "pop",
        "remove",
        "clear",
        "reverse",
        "sort",
        "iadd",
        "imul",
    ],
)
def test_in_place_sidecar_edits_are_normalized_without_losing_aliases(legacy, mutate):
    memory = _memory()
    _append(memory, 4)
    if legacy:
        memory._step_event_order = list(memory._step_event_order)
    alias = memory._step_event_order
    # Warm the path before editing through a retained alias.
    memory.add_to_memory("observation", {"position": [1, 2]})
    mutate(alias)
    expected = memory._normalized_step_event_order(memory.step_content, alias)
    memory.add_to_memory("message", {"new": True})
    assert memory._step_event_order is alias
    assert alias == [*expected, "message"]
    assert len(_events(memory.step_content, alias)) == 5


@pytest.mark.parametrize(
    "change",
    [
        "replace-content",
        "clear-content",
        "append-payload",
        "replace-sidecar",
        "additive-types",
    ],
)
def test_external_buffer_changes_invalidate_cached_order(change):
    memory = _memory()
    _append(memory, 4)
    if change == "replace-content":
        memory.step_content = {"message": [{"restored": 1}]}
    elif change == "clear-content":
        memory.step_content.clear()
    elif change == "append-payload":
        memory.step_content["action"].append({"restored": 2})
    elif change == "replace-sidecar":
        memory._step_event_order = ["invalid"]
    else:
        memory.additive_event_types.add("observation")
        memory.step_content["observation"] = {"restored": 3}
    expected = memory._normalized_step_event_order(
        memory.step_content, memory._step_event_order
    )
    alias = memory._step_event_order
    memory.add_to_memory("message", {"new": True})
    assert memory._step_event_order is alias
    assert alias == [*expected, "message"]


def test_partial_memory_repairs_legacy_content_and_retains_plain_list_identity():
    memory = ShortTermMemory.__new__(ShortTermMemory)
    memory.step_content = {"message": {"legacy": 1}, "action": [{"legacy": 2}]}
    memory.additive_event_types = {"message", "action"}
    legacy_order = []
    memory.__dict__["_step_event_order"] = legacy_order
    memory.add_to_memory("message", {"new": 3})
    assert memory._step_event_order is legacy_order
    assert legacy_order == ["message", "action", "message"]
    legacy_order[0] = "invalid"
    memory.add_to_memory("action", {"new": 4})
    assert legacy_order == ["message", "message", "action", "action"]


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_copied_memory_revalidates_shared_or_copied_sidecars(copier):
    memory = _memory()
    _append(memory, 4)
    copied = copier(memory)
    copied._step_event_order[0] = "invalid"
    expected = copied._normalized_step_event_order(
        copied.step_content, copied._step_event_order
    )
    copied.add_to_memory("action", {"copied": True})
    assert copied._step_event_order == [*expected, "action"]
    expected_original = memory._normalized_step_event_order(
        memory.step_content, memory._step_event_order
    )
    memory.add_to_memory("message", {"original": True})
    assert memory._step_event_order == [*expected_original, "message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_long_term_rollback_restores_aliases_and_invalidates_append_cache(
    asynchronous, cancel
):
    memory = _memory("long_term")
    memory.add_to_memory("message", {"event": "staged"})
    memory.process_step(pre_step=True)
    _append(memory, 4)
    original_content = memory.step_content
    original_order = memory._step_event_order
    staged = memory.buffer
    failure = asyncio.CancelledError() if cancel else RuntimeError("retry")

    def fail(_prompt):
        memory.add_to_memory("message", {"event": "reentrant"})
        raise failure

    memory.llm.generate = Mock(side_effect=fail)
    memory.llm.agenerate = AsyncMock(side_effect=fail)
    with pytest.raises(type(failure)) as exc_info:
        if asynchronous:
            await memory.aprocess_step()
        else:
            memory.process_step()
    assert exc_info.value is failure
    assert memory.buffer is staged
    assert memory.step_content is original_content
    assert memory._step_event_order is original_order
    assert original_order == ["message", "action", "message", "action", "message"]
    memory.add_to_memory("action", {"event": "after-rollback"})
    assert original_order == ["message", "action"] * 3
    normalize = Mock(wraps=memory._normalized_step_event_order)
    memory._normalized_step_event_order = normalize
    _append(memory, 100)
    assert normalize.call_count <= 1
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="retried"))]
    )
    memory.llm.generate = Mock(return_value=response)
    memory.llm.agenerate = AsyncMock(return_value=response)
    if asynchronous:
        await memory.aprocess_step()
    else:
        memory.process_step()
    assert memory.long_term_memory == "retried"
    assert len(_events(memory.buffer.content, memory.buffer._event_order)) == 107
    assert memory.step_content == {}
    assert memory._step_event_order == []
