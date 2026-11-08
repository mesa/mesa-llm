"""Regression tests for LLMAgent step finalization on every wrapper path."""

from __future__ import annotations

import logging
import warnings

import pytest
from mesa.model import Model

from mesa_llm.actions import (
    ActionChoice,
    ActionManager,
    ActionPostCommitError,
    action,
)
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.memory.st_memory import ShortTermMemory


class _ReasoningStub:
    def __init__(self, agent):
        self.agent = agent


class _FinalizationModel(Model):
    def __init__(self):
        super().__init__(rng=42)


class _GeneratedSyncAgent(LLMAgent):
    def step(self):
        self.body_calls += 1
        return self.test_body()


class _ExplicitAsyncAgent(LLMAgent):
    async def astep(self):
        self.body_calls += 1
        return await self.test_body()


class _DefaultAstepFallbackAgent(LLMAgent):
    def step(self):
        self.body_calls += 1
        return self.test_body()


_SYNC_ACTION_MANAGER = ActionManager()


@action(action_manager=_SYNC_ACTION_MANAGER)
def finalization_sync_commit(agent) -> object:
    """Commit and return the configured test result.

    Returns:
        The committed result.
    """
    agent.action_commit_calls += 1
    return agent.committed_result


_ASYNC_ACTION_MANAGER = ActionManager()


@action(action_manager=_ASYNC_ACTION_MANAGER)
async def finalization_async_commit(agent) -> object:
    """Commit and return the configured asynchronous test result.

    Returns:
        The committed result.
    """
    agent.action_commit_calls += 1
    return agent.committed_result


class _ControlledAbort(BaseException):
    """Safe sentinel for cancellation-like failures outside Exception."""


class _RaisingRepr:
    def __repr__(self):
        raise RuntimeError("controlled repr failure")


class _RaisingStr:
    def __str__(self):
        raise RuntimeError("controlled str failure")


class _ControlledRecorder:
    def __init__(self, agent, error=None):
        self.agent = agent
        self.error = error
        self.calls = []
        self.action_staged_at_entry = []

    def record_event(self, event_type, *, content, agent_id, metadata):
        self.calls.append(
            {
                "event_type": event_type,
                "content": content,
                "agent_id": agent_id,
                "metadata": metadata,
            }
        )
        self.action_staged_at_entry.append("action" in self.agent.memory.step_content)
        if self.error is not None:
            raise self.error


_PATHS = (
    pytest.param("generated-sync", id="generated-sync-step"),
    pytest.param("explicit-async", id="explicit-async-astep"),
    pytest.param("default-fallback", id="default-astep-fallback"),
)
_MEMORY_KINDS = (
    pytest.param("short-term", id="short-term-memory"),
    pytest.param("st-lt", id="st-lt-memory"),
)


def _make_agent(path: str, memory_kind: str, *, with_action: bool = False):
    agent_class = {
        "generated-sync": _GeneratedSyncAgent,
        "explicit-async": _ExplicitAsyncAgent,
        "default-fallback": _DefaultAstepFallbackAgent,
    }[path]
    configured_actions = []
    if with_action:
        configured_actions = [
            finalization_async_commit
            if path == "explicit-async"
            else finalization_sync_commit
        ]

    model = _FinalizationModel()
    agent = agent_class(
        model=model,
        reasoning=_ReasoningStub,
        llm_model="openai/finalization-test",
        actions=configured_actions,
    )
    agent.body_calls = 0
    agent.action_commit_calls = 0
    agent.committed_result = {"status": "committed"}

    if memory_kind == "short-term":
        agent.memory = ShortTermMemory(agent=agent, n=32, display=False)
    else:
        # The suite creates at most two entries. This capacity keeps the real
        # STLT implementation below its provider-backed consolidation boundary.
        agent.memory = STLTMemory(
            agent=agent,
            short_term_capacity=32,
            consolidation_capacity=4,
            display=False,
            llm_model="openai/finalization-test",
        )
    return agent


async def _run_step(path: str, agent):
    if path == "generated-sync":
        return agent.step()
    if path == "default-fallback":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return await agent.astep()
    return await agent.astep()


def _set_body(agent, path: str, sync_body, async_body):
    agent.test_body = async_body if path == "explicit-async" else sync_body


def _instrument_step_processing(
    monkeypatch,
    agent,
    path: str,
    *,
    pre_error=None,
    post_error=None,
):
    calls = []
    method_name = "process_step" if path == "generated-sync" else "aprocess_step"
    original = getattr(agent.memory, method_name)

    if path == "generated-sync":

        def observed(*, pre_step=False):
            calls.append(pre_step)
            if pre_step and pre_error is not None:
                raise pre_error
            result = original(pre_step=pre_step)
            if not pre_step and post_error is not None:
                raise post_error
            return result

    else:

        async def observed(*, pre_step=False):
            calls.append(pre_step)
            if pre_step and pre_error is not None:
                raise pre_error
            result = await original(pre_step=pre_step)
            if not pre_step and post_error is not None:
                raise post_error
            return result

    monkeypatch.setattr(agent.memory, method_name, observed)
    return calls


def _instrument_action_memory_observer(monkeypatch, agent, path: str, error=None):
    calls = []
    method_name = "aadd_to_memory" if path == "explicit-async" else "add_to_memory"
    original = getattr(agent.memory, method_name)

    if path == "explicit-async":

        async def observed(*, type, content):
            if type == "action":
                calls.append(content)
                if error is not None:
                    raise error
            return await original(type=type, content=content)

    else:

        def observed(*, type, content):
            if type == "action":
                calls.append(content)
                if error is not None:
                    raise error
            return original(type=type, content=content)

    monkeypatch.setattr(agent.memory, method_name, observed)
    return calls


def _entries(memory):
    return list(memory.short_term_memory)


def _assert_memory_closed(memory):
    assert memory.step_content == {}
    if isinstance(memory, ShortTermMemory):
        assert memory._current_step_entry is None
    else:
        assert isinstance(memory, STLTMemory)
        assert all(entry.step is not None for entry in memory.short_term_memory)


def _assert_entry(memory, index: int, *, step: int, content: dict):
    entry = _entries(memory)[index]
    assert entry.step == step
    assert entry.content == content


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("memory_kind", _MEMORY_KINDS)
async def test_success_finalizes_once_and_preserves_return_value(
    monkeypatch,
    path,
    memory_kind,
):
    agent = _make_agent(path, memory_kind)
    agent.model.steps = 7
    agent.memory.add_to_memory("before", {"turn": "successful"})
    returned_sentinel = object()

    def sync_body():
        agent.memory.add_to_memory("during", {"turn": "successful"})
        return returned_sentinel

    async def async_body():
        await agent.memory.aadd_to_memory("during", {"turn": "successful"})
        return returned_sentinel

    _set_body(agent, path, sync_body, async_body)
    processing_calls = _instrument_step_processing(monkeypatch, agent, path)

    returned = await _run_step(path, agent)

    assert returned is returned_sentinel
    assert agent.body_calls == 1
    assert processing_calls == [True, False]
    _assert_memory_closed(agent.memory)
    assert len(_entries(agent.memory)) == 1
    _assert_entry(
        agent.memory,
        0,
        step=7,
        content={
            "before": {"turn": "successful"},
            "during": {"turn": "successful"},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("memory_kind", _MEMORY_KINDS)
@pytest.mark.parametrize(
    ("memory_fails", "recorder_fails"),
    (
        pytest.param(True, False, id="memory-observer"),
        pytest.param(False, True, id="recorder-after-memory"),
        pytest.param(True, True, id="dual-observers"),
    ),
)
async def test_action_post_commit_error_still_finalizes_real_memory_once(
    monkeypatch,
    path,
    memory_kind,
    memory_fails,
    recorder_fails,
):
    agent = _make_agent(path, memory_kind, with_action=True)
    agent.model.steps = 11
    agent.memory.add_to_memory("before", {"turn": "committed-action"})
    memory_error = RuntimeError("controlled memory observer failure")
    recorder_error = ValueError("controlled recorder observer failure")
    memory_calls = _instrument_action_memory_observer(
        monkeypatch,
        agent,
        path,
        memory_error if memory_fails else None,
    )
    recorder = _ControlledRecorder(
        agent,
        recorder_error if recorder_fails else None,
    )
    agent.recorder = recorder
    action_name = (
        "finalization_async_commit"
        if path == "explicit-async"
        else "finalization_sync_commit"
    )
    choice = ActionChoice(
        name=action_name,
        arguments={},
        rationale="exercise step finalization",
    )

    def sync_body():
        return agent.execute_action(choice)

    async def async_body():
        return await agent.aexecute_action(choice)

    _set_body(agent, path, sync_body, async_body)
    processing_calls = _instrument_step_processing(monkeypatch, agent, path)

    with pytest.raises(ActionPostCommitError) as exc_info:
        await _run_step(path, agent)

    error = exc_info.value
    expected_observers = {
        observer
        for observer, failed in (
            ("memory", memory_fails),
            ("recorder", recorder_fails),
        )
        if failed
    }
    assert error.committed is True
    assert error.result is agent.committed_result
    assert set(error.observer_errors) == expected_observers
    if memory_fails:
        assert error.observer_errors["memory"] is memory_error
    if recorder_fails:
        assert error.observer_errors["recorder"] is recorder_error
    assert agent.body_calls == 1
    assert agent.action_commit_calls == 1
    assert len(memory_calls) == 1
    assert len(recorder.calls) == 1
    assert recorder.action_staged_at_entry == [not memory_fails]
    assert processing_calls == [True, False]

    _assert_memory_closed(agent.memory)
    assert len(_entries(agent.memory)) == 1
    entry = _entries(agent.memory)[0]
    assert entry.step == 11
    expected_keys = {"before"} | ({"action"} if not memory_fails else set())
    assert set(entry.content) == expected_keys
    assert entry.content["before"] == {"turn": "committed-action"}
    if not memory_fails:
        assert len(entry.content["action"]) == 1
        action_event = entry.content["action"][0]
        assert action_event["action"] == choice.model_dump()
        assert action_event["result"] == agent.committed_result
        assert action_event["result"] is not agent.committed_result


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("memory_kind", _MEMORY_KINDS)
@pytest.mark.parametrize(
    "error_class",
    (
        pytest.param(RuntimeError, id="exception"),
        pytest.param(_ControlledAbort, id="custom-base-exception"),
    ),
)
async def test_body_failure_finalizes_and_cannot_leak_into_next_step(
    monkeypatch,
    path,
    memory_kind,
    error_class,
):
    agent = _make_agent(path, memory_kind)
    agent.model.steps = 20
    agent.memory.add_to_memory("before", {"turn": "failed"})
    body_error = error_class("controlled body failure")

    def failing_sync_body():
        agent.memory.add_to_memory("during", {"turn": "failed"})
        raise body_error

    async def failing_async_body():
        await agent.memory.aadd_to_memory("during", {"turn": "failed"})
        raise body_error

    _set_body(agent, path, failing_sync_body, failing_async_body)
    processing_calls = _instrument_step_processing(monkeypatch, agent, path)

    with pytest.raises(error_class) as exc_info:
        await _run_step(path, agent)

    assert exc_info.value is body_error
    assert processing_calls == [True, False]
    _assert_memory_closed(agent.memory)
    assert len(_entries(agent.memory)) == 1
    _assert_entry(
        agent.memory,
        0,
        step=20,
        content={
            "before": {"turn": "failed"},
            "during": {"turn": "failed"},
        },
    )

    agent.model.steps = 21
    agent.memory.add_to_memory("before", {"turn": "next"})
    next_result = object()

    def next_sync_body():
        agent.memory.add_to_memory("during", {"turn": "next"})
        return next_result

    async def next_async_body():
        await agent.memory.aadd_to_memory("during", {"turn": "next"})
        return next_result

    _set_body(agent, path, next_sync_body, next_async_body)

    returned = await _run_step(path, agent)

    assert returned is next_result
    assert agent.body_calls == 2
    assert processing_calls == [True, False, True, False]
    _assert_memory_closed(agent.memory)
    assert len(_entries(agent.memory)) == 2
    _assert_entry(
        agent.memory,
        0,
        step=20,
        content={
            "before": {"turn": "failed"},
            "during": {"turn": "failed"},
        },
    )
    _assert_entry(
        agent.memory,
        1,
        step=21,
        content={
            "before": {"turn": "next"},
            "during": {"turn": "next"},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("memory_kind", _MEMORY_KINDS)
async def test_successful_body_propagates_post_failure_after_real_finalization(
    monkeypatch,
    path,
    memory_kind,
):
    agent = _make_agent(path, memory_kind)
    agent.model.steps = 30
    agent.memory.add_to_memory("before", {"turn": "post-failed"})
    post_error = RuntimeError("controlled post-step failure")

    def sync_body():
        agent.memory.add_to_memory("during", {"turn": "post-failed"})
        return object()

    async def async_body():
        await agent.memory.aadd_to_memory("during", {"turn": "post-failed"})
        return object()

    _set_body(agent, path, sync_body, async_body)
    processing_calls = _instrument_step_processing(
        monkeypatch,
        agent,
        path,
        post_error=post_error,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await _run_step(path, agent)

    assert exc_info.value is post_error
    assert agent.body_calls == 1
    assert processing_calls == [True, False]
    _assert_memory_closed(agent.memory)
    assert len(_entries(agent.memory)) == 1
    _assert_entry(
        agent.memory,
        0,
        step=30,
        content={
            "before": {"turn": "post-failed"},
            "during": {"turn": "post-failed"},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("memory_kind", _MEMORY_KINDS)
async def test_body_failure_remains_primary_when_post_base_exception_also_fails(
    monkeypatch,
    caplog,
    path,
    memory_kind,
):
    agent = _make_agent(path, memory_kind)
    agent.model.steps = 40
    agent.memory.add_to_memory("before", {"turn": "both-failed"})
    body_error = LookupError("controlled body failure")
    post_error = _ControlledAbort("controlled post-step abort")

    def sync_body():
        agent.memory.add_to_memory("during", {"turn": "both-failed"})
        raise body_error

    async def async_body():
        await agent.memory.aadd_to_memory("during", {"turn": "both-failed"})
        raise body_error

    _set_body(agent, path, sync_body, async_body)
    processing_calls = _instrument_step_processing(
        monkeypatch,
        agent,
        path,
        post_error=post_error,
    )

    with (
        caplog.at_level(logging.ERROR, logger="mesa_llm.llm_agent"),
        pytest.raises(LookupError) as exc_info,
    ):
        await _run_step(path, agent)

    assert exc_info.value is body_error
    assert agent.body_calls == 1
    assert processing_calls == [True, False]
    notes = getattr(body_error, "__notes__", [])
    assert len(notes) == 1
    assert "post" in notes[0].casefold() or "finaliz" in notes[0].casefold()
    assert str(post_error) in notes[0]
    diagnostic_records = [
        record
        for record in caplog.records
        if record.exc_info is not None and record.exc_info[1] is post_error
    ]
    assert len(diagnostic_records) == 1
    assert (
        "post" in diagnostic_records[0].getMessage().casefold()
        or "finaliz" in diagnostic_records[0].getMessage().casefold()
    )
    _assert_memory_closed(agent.memory)
    assert len(_entries(agent.memory)) == 1
    _assert_entry(
        agent.memory,
        0,
        step=40,
        content={
            "before": {"turn": "both-failed"},
            "during": {"turn": "both-failed"},
        },
    )


def test_sync_body_failure_survives_unrepresentable_finalizer(monkeypatch):
    agent = _make_agent("generated-sync", "short-term")
    body_error = LookupError("controlled sync body failure")
    post_error = _ControlledAbort(_RaisingRepr())

    def sync_body():
        raise body_error

    _set_body(agent, "generated-sync", sync_body, None)
    processing_calls = _instrument_step_processing(
        monkeypatch,
        agent,
        "generated-sync",
        post_error=post_error,
    )

    with pytest.raises(LookupError) as exc_info:
        agent.step()

    assert exc_info.value is body_error
    assert processing_calls == [True, False]


@pytest.mark.asyncio
async def test_async_body_failure_survives_unprintable_finalizer_and_logger(
    monkeypatch,
):
    agent = _make_agent("explicit-async", "short-term")
    body_error = LookupError("controlled async body failure")
    post_error = RuntimeError(_RaisingStr())
    logger_calls = []

    async def async_body():
        raise body_error

    def failing_logger(*args, **kwargs):
        logger_calls.append((args, kwargs))
        raise _ControlledAbort("controlled logger failure")

    _set_body(agent, "explicit-async", None, async_body)
    processing_calls = _instrument_step_processing(
        monkeypatch,
        agent,
        "explicit-async",
        post_error=post_error,
    )
    monkeypatch.setattr(
        logging.getLogger("mesa_llm.llm_agent"),
        "error",
        failing_logger,
    )

    with pytest.raises(LookupError) as exc_info:
        await _run_step("explicit-async", agent)

    assert exc_info.value is body_error
    assert processing_calls == [True, False]
    assert len(logger_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("memory_kind", _MEMORY_KINDS)
async def test_pre_failure_skips_body_and_post(monkeypatch, path, memory_kind):
    agent = _make_agent(path, memory_kind)
    agent.model.steps = 50
    staged_before_pre = {"turn": "pre-failed"}
    agent.memory.add_to_memory("before", staged_before_pre)
    pre_error = RuntimeError("controlled pre-step failure")

    def sync_body():
        raise AssertionError("body must not run after pre-step failure")

    async def async_body():
        raise AssertionError("body must not run after pre-step failure")

    _set_body(agent, path, sync_body, async_body)
    processing_calls = _instrument_step_processing(
        monkeypatch,
        agent,
        path,
        pre_error=pre_error,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await _run_step(path, agent)

    assert exc_info.value is pre_error
    assert agent.body_calls == 0
    assert processing_calls == [True]
    assert agent.memory.step_content == {"before": staged_before_pre}
    assert _entries(agent.memory) == []
    if isinstance(agent.memory, ShortTermMemory):
        assert agent.memory._current_step_entry is None
    else:
        assert all(entry.step is not None for entry in agent.memory.short_term_memory)
