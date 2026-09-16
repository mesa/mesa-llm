"""Async-generator rejection must not depend on a finalizer's release signal."""

import asyncio
import concurrent.futures
import gc
import threading
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import mesa_llm.actions.action_manager as manager_module
import mesa_llm.llm_agent as agent_module
from mesa_llm.actions import ActionChoice, ActionManager, action
from mesa_llm.llm_agent import LLMAgent


@pytest.fixture(autouse=True)
def short_cleanup_budget(monkeypatch):
    monkeypatch.setattr(
        manager_module, "_TASK_CANCELLATION_DRAIN_TIMEOUT_SECONDS", 0.03
    )


def _manager(payload, asynchronous=True):
    manager = ActionManager()
    if asynchronous:

        @action(action_manager=manager)
        async def invalid_result(agent) -> object:
            """Return an invalid result after the supported await."""
            del agent
            return payload

    else:

        @action(action_manager=manager)
        def invalid_result(agent) -> object:
            """Return an invalid result from a synchronous action."""
            del agent
            return payload

    return manager, ActionChoice(name=invalid_result.__name__, arguments={})


async def _wrap(generator, kind):
    if kind == "direct":
        return generator
    if kind == "task":

        async def complete():
            return generator

        wrapper = asyncio.create_task(complete())
        await wrapper
        return wrapper
    if kind == "mixed":
        inner = await _wrap(generator, "task")
        wrapper = concurrent.futures.Future()
        wrapper.set_result(inner)
        return await _wrap(wrapper, "asyncio")
    wrapper = (
        asyncio.get_running_loop().create_future()
        if kind == "asyncio"
        else concurrent.futures.Future()
    )
    wrapper.set_result(generator)
    return wrapper


def _assert_rejection(error, unresolved=False):
    assert type(error) is TypeError
    assert "one completed result" in str(error)
    notes = "\n".join(getattr(error, "__notes__", ())).casefold()
    if unresolved:
        assert "cleanup unresolved" in notes
        assert "async-generator" in notes
        assert "timeout" in notes
    return notes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper_kind", ["asyncio", "concurrent", "task", "mixed", "direct"]
)
@pytest.mark.parametrize("finalizer", ["waiting", "suppress-cancellation", "yielding"])
async def test_waiting_finalizer_is_rejected_before_release(wrapper_kind, finalizer):
    release = asyncio.Event()
    entered = asyncio.Event()
    events = []

    async def stream():
        try:
            yield 1
        finally:
            entered.set()
            events.append("entered")
            try:
                if finalizer == "yielding":
                    while not release.is_set():
                        await asyncio.sleep(0)
                else:
                    await release.wait()
            except asyncio.CancelledError:
                if finalizer != "suppress-cancellation":
                    raise
                events.append("suppressed")
                await release.wait()
            events.append("released")

    generator = stream()
    assert await anext(generator) == 1
    payload = await _wrap(generator, wrapper_kind)
    manager, choice = _manager(payload)
    tasks_before = asyncio.all_tasks()
    execution = asyncio.create_task(manager.aexecute(SimpleNamespace(), choice))
    try:
        # asyncio.wait is only a test watchdog; it does not cancel execution.
        done, _ = await asyncio.wait({execution}, timeout=1)
        assert execution in done, "rejection waited for the finalizer's release"
        assert entered.is_set()
        assert not release.is_set()
        with pytest.raises(TypeError) as exc_info:
            execution.result()
        _assert_rejection(exc_info.value, unresolved=True)
        assert generator.ag_frame is None
        assert not generator.ag_running
        assert events == ["entered"]
        assert asyncio.all_tasks() == tasks_before
        assert asyncio.current_task().cancelling() == 0
        release.set()
        await asyncio.sleep(0)
        assert events == ["entered"]
    finally:
        release.set()
        if not execution.done():
            with suppress(TypeError):
                await execution
        await generator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper_kind", ["asyncio", "concurrent", "task", "mixed", "direct"]
)
@pytest.mark.parametrize(
    "finalizer", ["immediate", "yield-once", "ready-future", "released-event"]
)
async def test_prompt_finalizers_still_finish_before_rejection(wrapper_kind, finalizer):
    events = []
    loop = asyncio.get_running_loop()

    async def stream():
        try:
            yield 1
        finally:
            if finalizer == "yield-once":
                await asyncio.sleep(0)
            elif finalizer == "ready-future":
                ready = loop.create_future()
                ready.set_result(None)
                await ready
            elif finalizer == "released-event":
                release = asyncio.Event()
                loop.call_soon(release.set)
                await release.wait()
            events.append("closed")

    generator = stream()
    assert await anext(generator) == 1
    payload = await _wrap(generator, wrapper_kind)
    manager, choice = _manager(payload)
    tasks_before = asyncio.all_tasks()
    with pytest.raises(TypeError) as exc_info:
        await manager.aexecute(SimpleNamespace(), choice)
    assert _assert_rejection(exc_info.value) == ""
    assert generator.ag_frame is None
    assert events == ["closed"]
    assert asyncio.all_tasks() == tasks_before


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_kind", ["concurrent", "asyncio", "task"])
@pytest.mark.parametrize("failure_type", [ValueError, TimeoutError])
async def test_finalizer_errors_remain_cleanup_notes(wrapper_kind, failure_type):
    failure = failure_type("finalizer failure")

    async def stream():
        try:
            yield 1
        finally:
            waiter = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_soon(waiter.set_exception, failure)
            await waiter

    generator = stream()
    assert await anext(generator) == 1
    manager, choice = _manager(await _wrap(generator, wrapper_kind))
    with pytest.raises(TypeError) as exc_info:
        await manager.aexecute(SimpleNamespace(), choice)
    notes = _assert_rejection(exc_info.value)
    assert "finalizer failure" in notes
    assert "cleanup unresolved" not in notes
    assert generator.ag_frame is None


class CleanupAbort(BaseException):
    """Control-flow exception that cleanup must not turn into a TypeError."""


@pytest.mark.asyncio
async def test_finalizer_base_exception_retains_identity():
    failure = CleanupAbort("stop")

    async def stream():
        try:
            yield 1
        finally:
            await asyncio.sleep(0)
            raise failure

    generator = stream()
    await anext(generator)
    manager, choice = _manager(await _wrap(generator, "concurrent"))
    with pytest.raises(CleanupAbort) as exc_info:
        await manager.aexecute(SimpleNamespace(), choice)
    assert exc_info.value is failure
    assert generator.ag_frame is None


@pytest.mark.asyncio
async def test_external_cancellation_is_not_converted_to_rejection():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def stream():
        try:
            yield 1
        finally:
            entered.set()
            await release.wait()

    generator = stream()
    await anext(generator)
    manager, choice = _manager(await _wrap(generator, "concurrent"))
    execution = asyncio.create_task(manager.aexecute(SimpleNamespace(), choice))
    try:
        await entered.wait()
        execution.cancel("external cancellation")
        done, _ = await asyncio.wait({execution}, timeout=1)
        assert execution in done
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await execution
        assert exc_info.value.args == ("external cancellation",)
        assert generator.ag_frame is None
        assert asyncio.current_task().cancelling() == 0
    finally:
        release.set()
        with suppress(asyncio.CancelledError):
            await execution
        await generator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["aexecute_action", "aact"])
async def test_timed_out_close_precedes_agent_observers(monkeypatch, entrypoint):
    release = asyncio.Event()

    async def stream():
        try:
            yield 1
        finally:
            await release.wait()

    generator = stream()
    await anext(generator)
    manager, choice = _manager(await _wrap(generator, "concurrent"))
    agent = object.__new__(LLMAgent)
    agent.unique_id = 1
    agent._action_manager = manager
    agent.memory = SimpleNamespace(add_to_memory=Mock(), aadd_to_memory=AsyncMock())
    agent.recorder = Mock()
    agent.achoose_action = AsyncMock(return_value=choice)
    deepcopy = Mock(side_effect=AssertionError("invalid result reached deepcopy"))
    monkeypatch.setattr(agent_module, "copy", SimpleNamespace(deepcopy=deepcopy))
    operation = (
        agent.aexecute_action(choice)
        if entrypoint == "aexecute_action"
        else agent.aact("Choose once.")
    )
    execution = asyncio.create_task(operation)
    try:
        done, _ = await asyncio.wait({execution}, timeout=1)
        assert execution in done
        with pytest.raises(TypeError) as exc_info:
            execution.result()
        _assert_rejection(exc_info.value, unresolved=True)
        assert not release.is_set()
        deepcopy.assert_not_called()
        agent.memory.add_to_memory.assert_not_called()
        agent.memory.aadd_to_memory.assert_not_awaited()
        agent.recorder.record_event.assert_not_called()
    finally:
        release.set()
        if not execution.done():
            with suppress(TypeError):
                await execution
        await generator.aclose()


@pytest.mark.parametrize("wrapper_kind", ["direct", "concurrent", "asyncio", "task"])
@pytest.mark.parametrize("suppress_cancellation", [False, True])
def test_synchronous_rejection_leaves_started_finalizer_untouched(
    wrapper_kind, suppress_cancellation
):
    entered = threading.Event()
    finished = threading.Event()
    state = SimpleNamespace(error=None, loop=None, release=None, generator=None)

    def execute():
        async def stream():
            try:
                yield 1
            finally:
                state.loop = asyncio.get_running_loop()
                state.release = asyncio.Event()
                entered.set()
                try:
                    await state.release.wait()
                except asyncio.CancelledError:
                    if not suppress_cancellation:
                        raise
                    await state.release.wait()

        try:
            generator = stream()
            state.generator = generator
            # Prime outside a loop, without registering an automatic finalizer.
            with pytest.raises(StopIteration) as start:
                generator.__anext__().send(None)
            assert start.value.value == 1
            payload = generator
            if wrapper_kind in ("asyncio", "task"):
                loop = asyncio.new_event_loop()
                try:
                    payload = loop.run_until_complete(_wrap(generator, wrapper_kind))
                finally:
                    loop.close()
            if wrapper_kind != "direct":
                outer = concurrent.futures.Future()
                outer.set_result(payload)
                payload = outer
            manager, choice = _manager(payload, asynchronous=False)
            manager.execute(SimpleNamespace(), choice)
        except BaseException as error:
            state.error = error
        finally:
            finished.set()

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    try:
        assert finished.wait(1), "synchronous rejection or loop shutdown blocked"
        notes = _assert_rejection(state.error)
        assert "cleanup unresolved" in notes
        assert "left untouched" in notes
        assert not entered.is_set()
        assert state.release is None
        assert state.generator.ag_frame is not None
        assert state.loop is None
    finally:
        if not finished.is_set() and state.loop is not None:
            state.loop.call_soon_threadsafe(state.release.set)
        thread.join(timeout=2)
        assert not thread.is_alive()
        # Owner teardown follows rejection assertions. Without a running
        # loop, the finalizer fails at its first loop lookup before waiting.
        with suppress(RuntimeError, StopIteration):
            state.generator.aclose().send(None)


@pytest.mark.asyncio
async def test_synchronous_rejection_on_active_loop_does_not_start_close(monkeypatch):
    entered = []

    async def stream():
        try:
            yield 1
        finally:
            entered.append(True)
            await asyncio.sleep(0)

    generator = stream()
    await anext(generator)
    manager, choice = _manager(await _wrap(generator, "concurrent"), asynchronous=False)
    run = Mock(side_effect=AssertionError("must not nest the active loop"))
    monkeypatch.setattr(manager_module.asyncio, "run", run)
    try:
        with pytest.raises(TypeError) as exc_info:
            manager.execute(SimpleNamespace(), choice)
        assert "cannot await aclose" in _assert_rejection(exc_info.value)
        assert entered == []
        run.assert_not_called()
    finally:
        await generator.aclose()


@pytest.mark.asyncio
async def test_unstarted_and_closed_generators_do_not_run_the_body():
    events = []

    async def stream():
        events.append("body")
        yield 1

    generator = stream()
    for _ in range(2):
        manager, choice = _manager(await _wrap(generator, "concurrent"))
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(SimpleNamespace(), choice)
        assert _assert_rejection(exc_info.value) == ""
        assert generator.ag_frame is None
    assert events == []
    gc.collect()


@pytest.mark.asyncio
async def test_close_does_not_schedule_helper_tasks():
    events = []

    async def stream():
        try:
            yield 1
        finally:
            await asyncio.sleep(0)
            events.append("closed")

    generator = stream()
    await anext(generator)
    manager, choice = _manager(await _wrap(generator, "concurrent"))
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()

    def forbidden_factory(*args, **kwargs):
        raise AssertionError("async-generator cleanup scheduled a new Task")

    loop.set_task_factory(forbidden_factory)
    try:
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(SimpleNamespace(), choice)
        assert _assert_rejection(exc_info.value) == ""
        assert events == ["closed"]
    finally:
        loop.set_task_factory(original_factory)
        await generator.aclose()


@pytest.mark.asyncio
async def test_cleanup_failure_during_timeout_keeps_primary_rejection():
    async def stream():
        try:
            yield 1
        finally:
            try:
                await asyncio.Event().wait()
            finally:
                raise ValueError("failure during close interruption")

    generator = stream()
    await anext(generator)
    manager, choice = _manager(await _wrap(generator, "concurrent"))
    with pytest.raises(TypeError) as exc_info:
        await manager.aexecute(SimpleNamespace(), choice)
    notes = _assert_rejection(exc_info.value, unresolved=True)
    assert "failure during close interruption" in notes
    assert generator.ag_frame is None


@pytest.mark.asyncio
async def test_finalizer_that_refuses_interruption_is_diagnosed(recwarn):
    events = []
    release = asyncio.Event()

    async def stream():
        try:
            yield 1
        finally:
            try:
                await release.wait()
            except GeneratorExit:
                events.append("refused interruption")
                await release.wait()
            events.append("resumed")

    generator = stream()
    await anext(generator)
    manager, choice = _manager(await _wrap(generator, "concurrent"))
    before = asyncio.all_tasks()
    with pytest.raises(TypeError) as exc_info:
        await manager.aexecute(SimpleNamespace(), choice)
    notes = _assert_rejection(exc_info.value, unresolved=True)
    assert "ignored generatorexit" in notes
    assert asyncio.all_tasks() == before
    # No scheduled close can resume this deliberately non-cooperative finalizer.
    release.set()
    await asyncio.sleep(0)
    assert events == ["refused interruption"]
    manager.actions.clear()
    del exc_info, generator
    gc.collect()
    await asyncio.sleep(0)
    assert not [
        warning for warning in recwarn if "was never awaited" in str(warning.message)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_kind", ["concurrent", "asyncio", "task"])
async def test_finalizer_can_handle_its_awaited_future_being_cancelled(wrapper_kind):
    events = []

    async def stream():
        try:
            yield 1
        finally:
            waiter = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_soon(waiter.cancel)
            try:
                await waiter
            except asyncio.CancelledError:
                # Only the awaited Future was cancelled, not the execution Task.
                await asyncio.sleep(0)
                events.append("handled")

    generator = stream()
    await anext(generator)
    manager, choice = _manager(await _wrap(generator, wrapper_kind))
    with pytest.raises(TypeError) as exc_info:
        await manager.aexecute(SimpleNamespace(), choice)
    assert _assert_rejection(exc_info.value) == ""
    assert events == ["handled"]
    assert generator.ag_frame is None
