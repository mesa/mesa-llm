"""Rejected asyncio work must only be cancelled from its owning loop."""

import asyncio
import concurrent.futures
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import mesa_llm.actions.action_manager as manager_module
from mesa_llm.actions import ActionChoice, ActionManager, action


@pytest.fixture
def foreign_work():
    ready = concurrent.futures.Future()
    diagnostics = []

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_debug(True)
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        state = SimpleNamespace(loop=loop, finally_ran=False)
        state.waiter = loop.create_future()

        async def body():
            try:
                ready.set_result(state)
                await state.waiter
            finally:
                state.finally_ran = True

        state.task = loop.create_task(body())
        try:
            loop.run_forever()
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    state = ready.result(timeout=5)

    def on_owner(fn):
        result = concurrent.futures.Future()

        def invoke():
            try:
                result.set_result(fn())
            except BaseException as error:
                result.set_exception(error)

        state.loop.call_soon_threadsafe(invoke)
        return result.result(timeout=5)

    def snapshot():
        return on_owner(
            lambda: (
                state.task.done(),
                state.task.cancelling(),
                state.waiter.cancelled(),
                state.finally_ran,
            )
        )

    state.on_owner = on_owner
    state.snapshot = snapshot
    # A barrier ensures the Task has reached its awaited Future before testing.
    assert snapshot() == (False, 0, False, False)
    try:
        yield state
    finally:

        async def finish():
            if not state.waiter.done():
                state.waiter.set_result(None)
            done, _ = await asyncio.wait({state.task}, timeout=1)
            if state.task not in done:
                # Bounded teardown even when running this test against the
                # broken implementation, which can lose the Task's wakeup.
                state.task.get_coro().close()
                raise AssertionError("foreign Task lost its wakeup")
            state.task.result()

        try:
            asyncio.run_coroutine_threadsafe(finish(), state.loop).result(timeout=5)
        finally:
            state.loop.call_soon_threadsafe(state.loop.stop)
            thread.join(timeout=5)
        assert not thread.is_alive()
        assert diagnostics == []


def _configure(payload, asynchronous=False):
    manager = ActionManager()
    if asynchronous:

        @action(action_manager=manager)
        async def deferred_work(agent) -> object:
            """Return deferred work after the supported action await."""
            del agent
            return payload

    else:

        @action(action_manager=manager)
        def deferred_work(agent) -> object:
            """Return a deferred result for rejection."""
            del agent
            return payload

    return manager, ActionChoice(name=deferred_work.__name__, arguments={})


def _executor_wrapper(payload):
    wrapper = concurrent.futures.Future()
    wrapper.set_result(payload)
    return wrapper


def _assert_untouched(state, error):
    assert type(error) is TypeError
    notes = getattr(error, "__notes__", ())
    assert any(
        "Cleanup unresolved" in note and "event loop" in note for note in notes
    )
    assert state.snapshot() == (False, 0, False, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["task", "future"])
@pytest.mark.parametrize(
    "producer", ["post-await", "asyncio-wrapper", "executor-wrapper"]
)
async def test_async_rejection_leaves_running_foreign_loop_work_untouched(
    foreign_work, kind, producer
):
    payload = foreign_work.task if kind == "task" else foreign_work.waiter
    if producer == "asyncio-wrapper":
        wrapper = asyncio.get_running_loop().create_future()
        wrapper.set_result(payload)
        payload = wrapper
    elif producer == "executor-wrapper":
        payload = _executor_wrapper(payload)
    manager, choice = _configure(payload, asynchronous=producer == "post-await")
    with patch.object(manager_module.asyncio, "wait", new=AsyncMock()) as drain:
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(SimpleNamespace(), choice)
        drain.assert_not_awaited()
    _assert_untouched(foreign_work, exc_info.value)


@pytest.mark.parametrize("kind", ["task", "future"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("active_loop", [False, True])
def test_sync_rejection_leaves_foreign_work_untouched_with_or_without_a_loop(
    foreign_work, kind, wrapped, active_loop
):
    payload = foreign_work.task if kind == "task" else foreign_work.waiter
    if wrapped:
        payload = _executor_wrapper(payload)
    manager, choice = _configure(payload)

    def check():
        with pytest.raises(TypeError) as exc_info:
            manager.execute(SimpleNamespace(), choice)
        _assert_untouched(foreign_work, exc_info.value)

    if active_loop:

        async def run_check():
            check()

        asyncio.run(run_check(), debug=True)
    else:
        check()


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["execute", "aexecute"])
async def test_completed_foreign_wrappers_still_inspect_nested_payloads(
    foreign_work, entrypoint
):
    def make_wrapper():
        wrapper = foreign_work.loop.create_future()
        wrapper.set_result(foreign_work.task)
        return wrapper

    payload = _executor_wrapper(foreign_work.on_owner(make_wrapper))
    manager, choice = _configure(payload)
    with pytest.raises(TypeError) as exc_info:
        if entrypoint == "execute":
            manager.execute(SimpleNamespace(), choice)
        else:
            await manager.aexecute(SimpleNamespace(), choice)
    _assert_untouched(foreign_work, exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["execute", "aexecute"])
async def test_owned_pending_future_is_still_cancelled(entrypoint):
    future = asyncio.get_running_loop().create_future()
    manager, choice = _configure(_executor_wrapper(future))
    with pytest.raises(TypeError) as exc_info:
        if entrypoint == "execute":
            manager.execute(SimpleNamespace(), choice)
        else:
            await manager.aexecute(SimpleNamespace(), choice)
    assert future.cancelled()
    assert not any(
        "untouched" in note for note in getattr(exc_info.value, "__notes__", ())
    )


@pytest.mark.asyncio
async def test_owned_task_still_finishes_async_cleanup_before_rejection():
    started = asyncio.Event()
    finished = []

    async def body():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            finished.append(True)

    task = asyncio.create_task(body())
    try:
        await started.wait()
        manager, choice = _configure(task, asynchronous=True)
        with pytest.raises(TypeError):
            await manager.aexecute(SimpleNamespace(), choice)
        assert task.done()
        assert task.cancelled()
        assert finished == [True]
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["execute", "aexecute"])
async def test_completed_wrapper_keeps_current_task_protection(entrypoint):
    current = asyncio.current_task()
    cancelling = current.cancelling()
    manager, choice = _configure(_executor_wrapper(current))
    with pytest.raises(TypeError) as exc_info:
        if entrypoint == "execute":
            manager.execute(SimpleNamespace(), choice)
        else:
            await manager.aexecute(SimpleNamespace(), choice)
    assert current.cancelling() == cancelling
    assert any("current task" in note for note in exc_info.value.__notes__)
