"""Exercise cleanup at rejection, without test-driven cleanup hiding the result."""

import asyncio
import concurrent.futures
import gc
import weakref
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import mesa_llm.actions.action_manager as manager_module
import mesa_llm.llm_agent as agent_module
from mesa_llm.actions import ActionChoice, ActionManager, action
from mesa_llm.llm_agent import LLMAgent


def _choice_for_payload(manager, payload, producer):
    if producer == "async-action":

        @action(action_manager=manager)
        async def return_payload(agent) -> object:
            """Return a deferred payload after the supported await."""
            del agent
            return payload

    else:
        wrapper = (
            asyncio.get_running_loop().create_future()
            if producer == "asyncio-wrapper"
            else concurrent.futures.Future()
        )
        wrapper.set_result(payload)

        @action(action_manager=manager)
        def return_payload(agent) -> object:
            """Return a completed wrapper around deferred work."""
            del agent
            return wrapper

    return ActionChoice(name=return_payload.__name__, arguments={})


def _assert_rejection(error):
    assert type(error) is TypeError
    assert "one completed result" in str(error)


def _assert_unresolved(error):
    assert any(
        "cleanup unresolved" in note.casefold()
        for note in getattr(error, "__notes__", ())
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "producer", ["async-action", "asyncio-wrapper", "concurrent-wrapper"]
)
async def test_gather_async_finally_finishes_before_rejection(producer):
    manager = ActionManager()
    started = asyncio.Event()
    events = []
    diagnostics = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))

    async def child_body():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            events.append("start")
            await asyncio.sleep(0)
            events.append("end")

    child = asyncio.create_task(child_body())
    aggregate = asyncio.gather(child)
    try:
        await started.wait()
        choice = _choice_for_payload(manager, aggregate, producer)
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(SimpleNamespace(), choice)

        _assert_rejection(exc_info.value)
        assert aggregate.done()
        assert child.done()
        assert child.cancelled()
        assert events == ["start", "end"]
        # A public Future interface cannot certify arbitrary hidden child work.
        _assert_unresolved(exc_info.value)
        await asyncio.sleep(0)
        assert events == ["start", "end"]
        # Release all aggregate references without retrieving its exception here:
        # the manager, not test teardown, must already have consumed it.
        aggregate_ref = weakref.ref(aggregate)
        manager.actions.clear()
        del exc_info, choice, aggregate
        gc.collect()
        await asyncio.sleep(0)
        assert aggregate_ref() is None
        assert diagnostics == []
    finally:
        if not child.done():
            child.cancel()
        await asyncio.gather(child, return_exceptions=True)
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_terminal_gather_does_not_certify_all_children_finished():
    manager = ActionManager()
    slow_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release = asyncio.Event()

    async def slow_child():
        try:
            slow_started.set()
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release.wait()

    async def fast_child():
        await asyncio.Event().wait()

    slow = asyncio.create_task(slow_child())
    fast = asyncio.create_task(fast_child())
    aggregate = asyncio.gather(fast, slow)
    try:
        await slow_started.wait()
        choice = _choice_for_payload(manager, aggregate, "async-action")
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(SimpleNamespace(), choice)
        _assert_rejection(exc_info.value)
        assert aggregate.done()
        assert cleanup_started.is_set()
        assert not slow.done()
        _assert_unresolved(exc_info.value)
    finally:
        release.set()
        for child in (fast, slow):
            if not child.done() and child.cancelling() == 0:
                child.cancel()
        await asyncio.gather(fast, slow, return_exceptions=True)
        with suppress(asyncio.CancelledError):
            await aggregate


@pytest.mark.asyncio
async def test_gather_timeout_is_explicit_and_does_not_recancel_children(monkeypatch):
    monkeypatch.setattr(
        manager_module, "_TASK_CANCELLATION_DRAIN_TIMEOUT_SECONDS", 0.01
    )
    manager = ActionManager()
    started = asyncio.Event()
    release = asyncio.Event()
    cancellation_seen = asyncio.Event()

    async def child_body():
        try:
            started.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()

    child = asyncio.create_task(child_body())
    aggregate = asyncio.gather(child)
    try:
        await started.wait()
        choice = _choice_for_payload(manager, aggregate, "async-action")
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(SimpleNamespace(), choice)
        _assert_rejection(exc_info.value)
        _assert_unresolved(exc_info.value)
        assert cancellation_seen.is_set()
        assert child.cancelling() == 1
        assert not child.done()
        assert not aggregate.done()
    finally:
        release.set()
        await child
        with suppress(asyncio.CancelledError):
            await aggregate


@pytest.mark.asyncio
async def test_gather_rejection_precedes_agent_observers(monkeypatch):
    manager = ActionManager()
    agent = object.__new__(LLMAgent)
    agent.unique_id = 1
    agent._action_manager = manager
    agent.memory = SimpleNamespace(add_to_memory=Mock(), aadd_to_memory=AsyncMock())
    agent.recorder = Mock()
    deepcopy = Mock(side_effect=AssertionError("rejected result reached deepcopy"))
    monkeypatch.setattr(agent_module, "copy", SimpleNamespace(deepcopy=deepcopy))
    started = asyncio.Event()
    cleaned = []

    async def child_body():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.append(True)

    child = asyncio.create_task(child_body())
    aggregate = asyncio.gather(child)
    try:
        await started.wait()
        choice = _choice_for_payload(manager, aggregate, "async-action")
        with pytest.raises(TypeError) as exc_info:
            await agent.aexecute_action(choice)
        _assert_rejection(exc_info.value)
        assert child.done()
        assert cleaned == [True]
        deepcopy.assert_not_called()
        agent.memory.add_to_memory.assert_not_called()
        agent.memory.aadd_to_memory.assert_not_awaited()
        agent.recorder.record_event.assert_not_called()
    finally:
        if not child.done():
            child.cancel()
        await asyncio.gather(child, return_exceptions=True)
        with suppress(asyncio.CancelledError):
            await aggregate


@pytest.mark.asyncio
async def test_foreign_pending_future_is_not_cancelled_or_drained():
    foreign_loop = asyncio.new_event_loop()
    future = foreign_loop.create_future()
    manager = ActionManager()
    try:
        choice = _choice_for_payload(manager, future, "async-action")
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(SimpleNamespace(), choice)
        _assert_rejection(exc_info.value)
        _assert_unresolved(exc_info.value)
        assert not future.done()
    finally:
        future.cancel()
        foreign_loop.close()


@pytest.mark.asyncio
async def test_future_drain_failure_preserves_primary_rejection(monkeypatch):
    class PendingAfterCancel(asyncio.Future):
        def cancel(self, msg=None):
            return True

    future = PendingAfterCancel()
    manager = ActionManager()
    failure = RuntimeError("drain failed")
    wait = AsyncMock(side_effect=failure)
    monkeypatch.setattr(manager_module.asyncio, "wait", wait)
    try:
        choice = _choice_for_payload(manager, future, "async-action")
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(SimpleNamespace(), choice)
        _assert_rejection(exc_info.value)
        assert any("drain failed" in note for note in exc_info.value.__notes__)
        wait.assert_awaited_once()
    finally:
        asyncio.Future.cancel(future)


@pytest.mark.asyncio
async def test_sync_completed_wrapper_diagnoses_terminal_composite():
    manager = ActionManager()
    release = asyncio.Event()
    started = asyncio.Event()

    async def slow_child():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            await release.wait()

    async def fail_child():
        raise RuntimeError("first child failed")

    child = asyncio.create_task(slow_child())
    await started.wait()
    aggregate = asyncio.gather(child, fail_child())
    try:
        # Observe completion without consuming the aggregate's exception.
        await asyncio.wait({aggregate})
        assert not child.done()
        choice = _choice_for_payload(manager, aggregate, "concurrent-wrapper")
        with pytest.raises(TypeError) as exc_info:
            manager.execute(SimpleNamespace(), choice)
        _assert_rejection(exc_info.value)
        _assert_unresolved(exc_info.value)
        assert any("first child failed" in note for note in exc_info.value.__notes__)
    finally:
        release.set()
        child.cancel()
        await asyncio.gather(child, return_exceptions=True)
        with suppress(RuntimeError):
            await aggregate
