"""Deadlines and external cancellation must not depend on child termination."""

import asyncio
import concurrent.futures
import inspect
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import mesa_llm.actions.action_manager as manager_module
from mesa_llm.actions import ActionChoice, ActionManager, action
from mesa_llm.actions._asyncgen_cleanup import (
    close_asyncgen_best_effort,
    close_asyncgen_without_loop,
)

BUDGET = 0.03
WATCHDOG = 1.0


def _notes(error):
    return "\n".join(getattr(error, "__notes__", ())).casefold()


async def _task_and_stream(already_cancelling=False):
    started = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    closing = asyncio.Event()
    waiter = asyncio.get_running_loop().create_future()

    async def worker():
        try:
            started.set()
            await waiter
        finally:
            cleaning.set()
            await release.wait()

    child = asyncio.create_task(worker())
    await started.wait()
    if already_cancelling:
        child.cancel()
        await cleaning.wait()

    async def stream():
        try:
            yield 1
        finally:
            closing.set()
            await child

    generator = stream()
    await anext(generator)
    return SimpleNamespace(
        child=child,
        generator=generator,
        waiter=waiter,
        release=release,
        closing=closing,
        cleaning=cleaning,
    )


async def _teardown(state, execution=None):
    # Cleanup belongs to the test owner and starts only after rejection assertions.
    state.release.set()
    if not state.waiter.done():
        state.waiter.set_result(None)
    with suppress(asyncio.CancelledError):
        await state.child
    if execution is not None:
        with suppress(TypeError, asyncio.CancelledError):
            await execution
    await state.generator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("already_cancelling", [False, True])
async def test_deadline_does_not_cancel_or_wait_for_child_finally(already_cancelling):
    state = await _task_and_stream(already_cancelling)
    primary = TypeError("invalid action result")
    cancellations = state.child.cancelling()
    waiter_cancelled = state.waiter.cancelled()
    execution = asyncio.create_task(
        close_asyncgen_best_effort(state.generator, primary, BUDGET)
    )
    try:
        done, _ = await asyncio.wait({execution}, timeout=WATCHDOG)
        assert execution in done, "cleanup deadline depended on child termination"
        execution.result()
        assert state.closing.is_set()
        assert not state.release.is_set()
        assert not state.child.done()
        assert state.child.cancelling() == cancellations
        assert state.waiter.cancelled() == waiter_cancelled
        assert state.cleaning.is_set() == already_cancelling
        assert "cleanup unresolved" in _notes(primary)
        assert "timeout" in _notes(primary)
        assert state.generator.ag_frame is None
    finally:
        await _teardown(state, execution)


class UnprintableFailureError(Exception):
    def __str__(self):
        raise ValueError("formatting failed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_type", [ValueError, TimeoutError, UnprintableFailureError]
)
@pytest.mark.parametrize("await_kind", ["event", "task"])
async def test_external_cancellation_survives_interruption_failure(
    failure_type, await_kind
):
    entered = asyncio.Event()
    release = asyncio.Event()
    child = asyncio.create_task(release.wait()) if await_kind == "task" else None
    primary = TypeError("invalid action result")

    async def stream():
        try:
            yield 1
        finally:
            try:
                entered.set()
                if child is None:
                    await release.wait()
                else:
                    await child
            except (asyncio.CancelledError, GeneratorExit):
                raise failure_type("secondary interruption failure") from None

    generator = stream()
    await anext(generator)
    execution = asyncio.create_task(close_asyncgen_best_effort(generator, primary, 10))
    try:
        await entered.wait()
        execution.cancel("external request")
        done, _ = await asyncio.wait({execution}, timeout=WATCHDOG)
        assert execution in done
        with pytest.raises(asyncio.CancelledError) as caught:
            await execution
        assert caught.value.args == ("external request",)
        assert execution.cancelled()
        assert execution.cancelling() == 1
        assert failure_type.__name__.casefold() in _notes(caught.value)
        assert _notes(caught.value) == _notes(primary)
        if child is not None:
            assert child.cancelling() == 0
            assert not child.done()
        assert generator.ag_frame is None
    finally:
        release.set()
        if child is not None:
            with suppress(asyncio.CancelledError):
                await child
        with suppress(asyncio.CancelledError):
            await execution
        await generator.aclose()


@pytest.mark.asyncio
async def test_timeout_interruption_failure_is_diagnostic_not_cancellation():
    primary = TypeError("invalid action result")

    async def stream():
        try:
            yield 1
        finally:
            try:
                await asyncio.Event().wait()
            except GeneratorExit:
                raise ValueError("secondary interruption failure") from None

    generator = stream()
    await anext(generator)
    await close_asyncgen_best_effort(generator, primary, BUDGET)
    assert "cleanup unresolved" in _notes(primary)
    assert "secondary interruption failure" in _notes(primary)
    assert asyncio.current_task().cancelling() == 0
    assert generator.ag_frame is None


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "cancelled", "failed"])
async def test_normal_child_outcomes_still_reach_finalizer(outcome):
    primary = TypeError("invalid action result")
    completed = []

    async def child_body():
        await asyncio.sleep(0)
        if outcome == "cancelled":
            raise asyncio.CancelledError
        if outcome == "failed":
            raise ValueError("child failure")
        return 7

    async def stream():
        try:
            yield 1
        finally:
            try:
                value = await asyncio.create_task(child_body())
            except asyncio.CancelledError:
                completed.append("cancelled")
            except ValueError:
                completed.append("failed")
            else:
                assert value == 7
                completed.append("success")

    generator = stream()
    await anext(generator)
    await close_asyncgen_best_effort(generator, primary, WATCHDOG)
    assert completed == [outcome]
    assert _notes(primary) == ""
    assert generator.ag_frame is None


@pytest.mark.parametrize("inert", [False, True])
def test_synchronous_cleanup_never_starts_a_finalizer_or_event_loop(monkeypatch, inert):
    called = []

    async def stream():
        try:
            yield 1
        finally:
            called.append(True)
            loop = asyncio.get_running_loop()
            await loop.create_task(asyncio.Event().wait())

    generator = stream()
    if not inert:
        with pytest.raises(StopIteration):
            generator.__anext__().send(None)
    primary = TypeError("invalid action result")
    run = Mock(side_effect=AssertionError("must not create a cleanup loop"))
    monkeypatch.setattr(asyncio, "run", run)
    try:
        close_asyncgen_without_loop(generator, primary)
        run.assert_not_called()
        assert called == []
        if inert:
            assert inspect.getasyncgenstate(generator) == inspect.AGEN_CLOSED
            assert _notes(primary) == ""
        else:
            assert inspect.getasyncgenstate(generator) == inspect.AGEN_SUSPENDED
            assert "cleanup unresolved" in _notes(primary)
            assert "left untouched" in _notes(primary)
    finally:
        # Owner teardown runs after the assertions. Without a running loop,
        # the finalizer fails at its loop lookup before it can create work.
        if generator.ag_frame is not None:
            close = generator.aclose()
            with suppress(RuntimeError, StopIteration):
                close.send(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper", ["direct", "concurrent", "asyncio", "task"])
async def test_manager_rejects_before_child_release(wrapper, monkeypatch):

    monkeypatch.setattr(
        manager_module, "_TASK_CANCELLATION_DRAIN_TIMEOUT_SECONDS", BUDGET
    )
    state = await _task_and_stream()
    payload = state.generator
    if wrapper == "concurrent":
        payload = concurrent.futures.Future()
        payload.set_result(state.generator)
    elif wrapper == "asyncio":
        payload = asyncio.get_running_loop().create_future()
        payload.set_result(state.generator)
    elif wrapper == "task":

        async def completed():
            return state.generator

        payload = asyncio.create_task(completed())
        await payload
    manager = ActionManager()

    @action(action_manager=manager)
    async def deferred_action(agent) -> object:
        """Return a started async-generator directly or behind a wrapper."""
        del agent
        return payload

    choice = ActionChoice(name="deferred_action", arguments={})
    execution = asyncio.create_task(manager.aexecute(SimpleNamespace(), choice))
    try:
        done, _ = await asyncio.wait({execution}, timeout=WATCHDOG)
        assert execution in done
        with pytest.raises(TypeError) as caught:
            execution.result()
        assert type(caught.value) is TypeError
        assert "cleanup unresolved" in _notes(caught.value)
        assert not state.release.is_set()
        assert not state.child.done()
        assert state.child.cancelling() == 0
    finally:
        await _teardown(state, execution)


@pytest.mark.asyncio
async def test_async_close_joins_its_driver_before_return():
    loop = asyncio.get_running_loop()
    before = asyncio.all_tasks()
    drivers = []
    primary = TypeError("invalid action result")
    closed = []

    async def stream():
        try:
            yield 1
        finally:
            drivers.append(asyncio.current_task())
            waiter = loop.create_future()
            loop.call_soon(waiter.set_result, None)
            await waiter
            await asyncio.sleep(0)
            closed.append(True)

    generator = stream()
    await anext(generator)

    try:
        await close_asyncgen_best_effort(generator, primary, WATCHDOG)
        assert closed == [True]
        assert _notes(primary) == ""
        assert drivers and all(task.done() for task in drivers)
        assert asyncio.all_tasks() == before
    finally:
        await generator.aclose()


@pytest.mark.asyncio
async def test_abandoned_waiter_is_not_cancelled_after_timeout():
    primary = TypeError("invalid action result")
    waiter = asyncio.get_running_loop().create_future()

    async def stream():
        try:
            yield 1
        finally:
            await waiter

    generator = stream()
    await anext(generator)
    await close_asyncgen_best_effort(generator, primary, BUDGET)
    assert "timeout" in _notes(primary)
    assert not waiter.done()
    assert generator.ag_frame is None
    # The owning caller may still resolve and await the same Future normally.
    waiter.set_result(4)
    assert await waiter == 4


@pytest.mark.asyncio
async def test_external_cancellation_object_is_preserved():
    caught_inside = []
    entered = asyncio.Event()
    primary = TypeError("invalid action result")

    async def stream():
        try:
            yield 1
        finally:
            try:
                entered.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                caught_inside.append(cancellation)
                raise ValueError("interrupted") from None

    generator = stream()
    await anext(generator)
    execution = asyncio.create_task(close_asyncgen_best_effort(generator, primary, 10))
    await entered.wait()
    execution.cancel("caller cancelled")
    with pytest.raises(asyncio.CancelledError) as caught:
        await execution
    assert caught.value is caught_inside[0]
    assert execution.cancelled()
    assert "interrupted" in _notes(caught.value)
