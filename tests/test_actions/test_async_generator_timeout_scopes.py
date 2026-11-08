"""Keep finalizer-local timeouts separate from action cancellation and deadlines."""

import asyncio
import concurrent.futures
import contextvars
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import mesa_llm.actions.action_manager as manager_module
import mesa_llm.llm_agent as agent_module
from mesa_llm.actions import ActionChoice, ActionManager, action
from mesa_llm.actions._asyncgen_cleanup import close_asyncgen_best_effort
from mesa_llm.llm_agent import LLMAgent

SUCCESS_BUDGET = 10.0
EXPIRY_BUDGET = 0.03
WATCHDOG = 5.0


def _notes(error):
    return "\n".join(getattr(error, "__notes__", ())).casefold()


async def _outcome(awaitable):
    try:
        await awaitable
    except BaseException as error:
        return error
    return None


async def _finish(task):
    done, _ = await asyncio.wait({task}, timeout=WATCHDOG)
    assert task in done, "cleanup depended on a finalizer or child being released"
    return task.result()


@pytest.mark.asyncio
@pytest.mark.parametrize("handled", [False, True])
@pytest.mark.parametrize("timeout_kind", ["relative", "absolute", "nested"])
@pytest.mark.parametrize("await_kind", ["event", "task"])
async def test_local_timeout_matches_native_close(handled, timeout_kind, await_kind):
    async def run(use_helper):
        events = []
        children = []
        release = asyncio.Event()
        primary = TypeError("invalid action result")

        async def stream():
            try:
                yield 1
            finally:
                loop = asyncio.get_running_loop()
                deadline = (
                    asyncio.timeout_at(loop.time() + 0.01)
                    if timeout_kind == "absolute"
                    else asyncio.timeout(0.01)
                )
                try:
                    async with asyncio.timeout(
                        SUCCESS_BUDGET if timeout_kind == "nested" else None
                    ):
                        async with deadline:
                            if await_kind == "task":
                                child = asyncio.create_task(release.wait())
                                children.append(child)
                                await child
                            else:
                                await release.wait()
                except TimeoutError:
                    if not handled:
                        raise
                    await asyncio.sleep(0)
                    events.append("handled")
                events.append("finished")

        generator = stream()
        await anext(generator)
        before = asyncio.current_task().cancelling()
        try:
            operation = (
                close_asyncgen_best_effort(generator, primary, SUCCESS_BUDGET)
                if use_helper
                else generator.aclose()
            )
            error = await _outcome(operation)
            assert asyncio.current_task().cancelling() == before
            assert generator.ag_frame is None
            assert _notes(primary) == ""
            return type(error), events
        finally:
            release.set()
            for child in children:
                with suppress(asyncio.CancelledError):
                    await child
            await generator.aclose()

    native = await _finish(asyncio.create_task(run(False)))
    actual = await _finish(asyncio.create_task(run(True)))
    assert actual == native
    assert actual == (
        (type(None), ["handled", "finished"]) if handled else (TimeoutError, [])
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("handled", [False, True])
@pytest.mark.parametrize("wrapper", ["direct", "concurrent", "asyncio", "task"])
async def test_manager_local_timeout_preserves_action_rejection(
    monkeypatch, handled, wrapper
):
    monkeypatch.setattr(
        manager_module, "_TASK_CANCELLATION_DRAIN_TIMEOUT_SECONDS", SUCCESS_BUDGET
    )
    handled_timeouts = []

    async def stream():
        try:
            yield 1
        finally:
            try:
                async with asyncio.timeout(0.01):
                    await asyncio.Event().wait()
            except TimeoutError:
                if not handled:
                    raise
                handled_timeouts.append(True)

    generator = stream()
    await anext(generator)
    payload = generator
    if wrapper == "concurrent":
        payload = concurrent.futures.Future()
        payload.set_result(generator)
    elif wrapper == "asyncio":
        payload = asyncio.get_running_loop().create_future()
        payload.set_result(generator)
    elif wrapper == "task":

        async def completed():
            return generator

        payload = asyncio.create_task(completed())
        await payload
    manager = ActionManager()

    @action(action_manager=manager)
    async def deferred_result(agent) -> object:
        """Return an invalid result whose finalizer has a local timeout."""
        del agent
        return payload

    agent = object.__new__(LLMAgent)
    agent.unique_id = 1
    agent._action_manager = manager
    agent.memory = SimpleNamespace(add_to_memory=Mock(), aadd_to_memory=AsyncMock())
    agent.recorder = Mock()
    deepcopy = Mock(side_effect=AssertionError("invalid result reached observers"))
    monkeypatch.setattr(agent_module, "copy", SimpleNamespace(deepcopy=deepcopy))
    execution = asyncio.create_task(
        agent.aexecute_action(ActionChoice(name="deferred_result", arguments={}))
    )
    try:
        with pytest.raises(TypeError) as caught:
            await _finish(execution)
        assert type(caught.value) is TypeError
        assert not execution.cancelled()
        assert execution.cancelling() == 0
        assert handled_timeouts == ([True] if handled else [])
        if handled:
            assert _notes(caught.value) == ""
        else:
            assert "timeouterror" in _notes(caught.value)
            assert "cleanup unresolved" not in _notes(caught.value)
        deepcopy.assert_not_called()
        agent.memory.add_to_memory.assert_not_called()
        agent.memory.aadd_to_memory.assert_not_awaited()
        agent.recorder.record_event.assert_not_called()
    finally:
        if not execution.done():
            execution.cancel()
        with suppress(TypeError, asyncio.CancelledError):
            await execution
        await generator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_local_timeout", [False, True])
@pytest.mark.parametrize("secondary_failure", [False, True])
async def test_external_cancellation_wins_over_active_local_timeout(
    after_local_timeout, secondary_failure
):
    entered = asyncio.Event()
    driver_tasks = []
    primary = TypeError("invalid action result")

    async def stream():
        try:
            yield 1
        finally:
            driver_tasks.append(asyncio.current_task())
            try:
                if after_local_timeout:
                    try:
                        async with asyncio.timeout(0):
                            await asyncio.sleep(0)
                    except TimeoutError:
                        pass
                async with asyncio.timeout(SUCCESS_BUDGET):
                    entered.set()
                    await asyncio.Event().wait()
            except (asyncio.CancelledError, GeneratorExit):
                if secondary_failure:
                    raise ValueError("interruption failure") from None
                raise

    generator = stream()
    await anext(generator)
    execution = asyncio.create_task(
        close_asyncgen_best_effort(generator, primary, SUCCESS_BUDGET)
    )
    try:
        ready = asyncio.create_task(entered.wait())
        await _finish(ready)
        execution.cancel("original external cancellation")
        with pytest.raises(asyncio.CancelledError) as caught:
            await _finish(execution)
        assert caught.value.args == ("original external cancellation",)
        assert execution.cancelled()
        assert execution.cancelling() == 1
        assert driver_tasks and all(task.done() for task in driver_tasks)
        if secondary_failure:
            assert "interruption failure" in _notes(caught.value)
        assert generator.ag_frame is None
    finally:
        if not execution.done():
            execution.cancel()
        with suppress(asyncio.CancelledError):
            await execution
        await generator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("local_timeout_first", [False, True])
async def test_framework_deadline_is_independent_of_finalizer_task_cleanup(
    local_timeout_first,
):
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    driver_tasks = []
    waiter = asyncio.get_running_loop().create_future()

    async def worker():
        try:
            entered.set()
            await waiter
        finally:
            cleaning.set()
            await release.wait()

    child = asyncio.create_task(worker())
    await entered.wait()

    async def stream():
        try:
            yield 1
        finally:
            driver_tasks.append(asyncio.current_task())
            if local_timeout_first:
                try:
                    async with asyncio.timeout(0):
                        await asyncio.sleep(0)
                except TimeoutError:
                    pass
            async with asyncio.timeout(SUCCESS_BUDGET):
                await child

    generator = stream()
    await anext(generator)
    primary = TypeError("invalid action result")
    before = asyncio.all_tasks()
    execution = asyncio.create_task(
        close_asyncgen_best_effort(generator, primary, EXPIRY_BUDGET)
    )
    try:
        await _finish(execution)
        assert "cleanup unresolved" in _notes(primary)
        assert "timeout" in _notes(primary)
        assert not release.is_set()
        assert not child.done()
        assert child.cancelling() == 0
        assert not waiter.cancelled()
        assert not cleaning.is_set()
        assert driver_tasks and all(task.done() for task in driver_tasks)
        assert asyncio.all_tasks() == before
        assert generator.ag_frame is None
    finally:
        release.set()
        if not waiter.done():
            waiter.set_result(None)
        with suppress(asyncio.CancelledError):
            await child
        await execution
        await generator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True])
async def test_finalizer_context_tokens_survive_cancellation_scope_isolation(eager):
    variable = contextvars.ContextVar("cleanup_test_value", default="initial")
    loop = asyncio.get_running_loop()
    factory = loop.get_task_factory()
    primary = TypeError("invalid action result")
    driver_tasks = []

    async def stream():
        token = variable.set("inside")
        try:
            yield 1
        finally:
            driver_tasks.append(asyncio.current_task())
            async with asyncio.timeout(SUCCESS_BUDGET):
                await asyncio.sleep(0)
            assert variable.get() == "inside"
            variable.reset(token)

    generator = stream()
    await anext(generator)
    before = asyncio.all_tasks()
    if eager:
        loop.set_task_factory(asyncio.eager_task_factory)
    try:
        await close_asyncgen_best_effort(generator, primary, SUCCESS_BUDGET)
        assert variable.get() == "initial"
        assert _notes(primary) == ""
        assert driver_tasks[0] is not asyncio.current_task()
        assert driver_tasks[0].done()
        assert asyncio.all_tasks() == before
    finally:
        loop.set_task_factory(factory)
        await generator.aclose()


@pytest.mark.asyncio
async def test_repeated_external_cancellation_joins_the_cleanup_driver():
    entered = asyncio.Event()
    driver_tasks = []
    observed = []
    primary = TypeError("invalid action result")

    async def stream():
        try:
            yield 1
        finally:
            driver_tasks.append(asyncio.current_task())
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                observed.append(cancellation)
                execution.cancel("second request")
                raise

    generator = stream()
    await anext(generator)
    execution = asyncio.create_task(
        close_asyncgen_best_effort(generator, primary, SUCCESS_BUDGET)
    )
    await entered.wait()
    execution.cancel("first request")
    with pytest.raises(asyncio.CancelledError) as caught:
        await _finish(execution)
    assert caught.value is observed[0]
    assert caught.value.args == ("first request",)
    assert execution.cancelling() == 2
    assert driver_tasks[0].done()
    assert generator.ag_frame is None


@pytest.mark.asyncio
async def test_local_timeout_cancels_child_without_defeating_framework_deadline():
    started = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    drivers = []

    async def worker():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    child = asyncio.create_task(worker())
    await started.wait()

    async def stream():
        try:
            yield 1
        finally:
            drivers.append(asyncio.current_task())
            async with asyncio.timeout(0):
                await child

    generator = stream()
    await anext(generator)
    primary = TypeError("invalid action result")
    execution = asyncio.create_task(
        close_asyncgen_best_effort(generator, primary, EXPIRY_BUDGET)
    )
    try:
        await _finish(execution)
        assert cleaning.is_set()
        assert not release.is_set()
        assert not child.done()
        assert child.cancelling() == 1
        assert "cleanup unresolved" in _notes(primary)
        assert drivers[0].done()
        assert not execution.cancelled()
        assert execution.cancelling() == 0
    finally:
        release.set()
        with suppress(asyncio.CancelledError):
            await child
        await execution
        await generator.aclose()


@pytest.mark.asyncio
async def test_local_timeout_keeps_native_child_cancellation_suppression():
    observed = []
    started = asyncio.Event()

    async def child_body():
        try:
            started.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            return 7

    child = asyncio.create_task(child_body())
    await started.wait()

    async def stream():
        try:
            yield 1
        finally:
            async with asyncio.timeout(0):
                observed.append(await child)

    generator = stream()
    await anext(generator)
    primary = TypeError("invalid action result")
    try:
        await _finish(
            asyncio.create_task(
                close_asyncgen_best_effort(generator, primary, SUCCESS_BUDGET)
            )
        )
        assert observed == [7]
        assert child.done()
        assert child.cancelling() == 1
        assert _notes(primary) == ""
        assert generator.ag_frame is None
    finally:
        if not child.done():
            child.cancel()
        with suppress(asyncio.CancelledError):
            await child
        await generator.aclose()


@pytest.mark.asyncio
async def test_finalizer_task_group_retains_its_exception_semantics():
    observed = []

    async def fail():
        await asyncio.sleep(0)
        raise ValueError("child failure")

    async def stream():
        try:
            yield 1
        finally:
            try:
                async with asyncio.TaskGroup() as group:
                    group.create_task(fail())
                    await asyncio.Event().wait()
            except* ValueError as error:
                observed.extend(str(item) for item in error.exceptions)

    generator = stream()
    await anext(generator)
    primary = TypeError("invalid action result")
    before = asyncio.all_tasks()
    await _finish(
        asyncio.create_task(
            close_asyncgen_best_effort(generator, primary, SUCCESS_BUDGET)
        )
    )
    assert observed == ["child failure"]
    assert _notes(primary) == ""
    assert generator.ag_frame is None
    assert asyncio.all_tasks() == before
    assert asyncio.current_task().cancelling() == 0


@pytest.mark.asyncio
async def test_callers_timeout_still_converts_its_own_cancellation():
    primary = TypeError("invalid action result")
    drivers = []

    async def stream():
        try:
            yield 1
        finally:
            drivers.append(asyncio.current_task())
            async with asyncio.timeout(SUCCESS_BUDGET):
                await asyncio.Event().wait()

    generator = stream()
    await anext(generator)
    before = asyncio.all_tasks()
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await close_asyncgen_best_effort(generator, primary, SUCCESS_BUDGET)
    assert _notes(primary) == ""
    assert asyncio.current_task().cancelling() == 0
    assert drivers and all(driver.done() for driver in drivers)
    assert asyncio.all_tasks() == before
    assert generator.ag_frame is None


@pytest.mark.asyncio
@pytest.mark.parametrize("handled", [False, True])
@pytest.mark.parametrize(
    "timeout_kind", ["relative", "absolute", "nested", "rescheduled"]
)
@pytest.mark.parametrize("async_unwind", [False, True])
async def test_pre_yield_timeout_matches_native_close(
    handled, timeout_kind, async_unwind
):
    async def run(use_helper):
        events = []
        loop = asyncio.get_running_loop()
        deadline = (
            asyncio.timeout_at(loop.time())
            if timeout_kind == "absolute"
            else asyncio.timeout(None if timeout_kind == "rescheduled" else 0)
        )

        async def stream():
            try:
                async with asyncio.timeout(
                    SUCCESS_BUDGET if timeout_kind == "nested" else None
                ):
                    async with deadline:
                        try:
                            yield 1
                        finally:
                            if timeout_kind == "rescheduled":
                                deadline.reschedule(loop.time())
                            try:
                                await asyncio.Event().wait()
                            finally:
                                if async_unwind:
                                    await asyncio.sleep(0)
                                    await asyncio.sleep(0)
            except TimeoutError:
                if not handled:
                    raise
                await asyncio.sleep(0)
                events.append("handled")

        generator = stream()
        assert await anext(generator) == 1
        primary = TypeError("invalid action result")
        before = asyncio.current_task().cancelling()
        operation = (
            close_asyncgen_best_effort(generator, primary, SUCCESS_BUDGET)
            if use_helper
            else generator.aclose()
        )
        try:
            error = await _outcome(operation)
            assert asyncio.current_task().cancelling() == before
            assert generator.ag_frame is None
            assert _notes(primary) == ""
            return type(error) if error is not None else None, events
        finally:
            await generator.aclose()

    # Priming and closure must stay in the same task in each comparison. A
    # watchdog wrapping only close() would hide the original ownership defect.
    native = await _finish(asyncio.create_task(run(False)))
    actual = await _finish(asyncio.create_task(run(True)))
    assert actual == native
    expected = (None, ["handled"]) if handled else (TimeoutError, [])
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry", ["before-yield", "during-close", "both"])
@pytest.mark.parametrize("handled", [False, True])
async def test_overlapping_timeout_scopes_do_not_resurrect_consumed_cancellation(
    expiry, handled
):
    async def run(use_helper):
        events = []
        outer = asyncio.timeout(None)

        async def stream():
            try:
                async with outer:
                    try:
                        yield 1
                    finally:
                        async with asyncio.timeout(None) as inner:
                            now = asyncio.get_running_loop().time()
                            if expiry in {"before-yield", "both"}:
                                outer.reschedule(now)
                            if expiry in {"during-close", "both"}:
                                inner.reschedule(now)
                            try:
                                await asyncio.Event().wait()
                            finally:
                                await asyncio.sleep(0)
            except TimeoutError:
                if not handled:
                    raise
                events.append("handled")

        generator = stream()
        await anext(generator)
        primary = TypeError("invalid action result")
        try:
            operation = (
                close_asyncgen_best_effort(generator, primary, SUCCESS_BUDGET)
                if use_helper
                else generator.aclose()
            )
            error = await _outcome(operation)
            assert asyncio.current_task().cancelling() == 0
            assert generator.ag_frame is None
            assert _notes(primary) == ""
            return type(error) if error is not None else None, events
        finally:
            await generator.aclose()

    assert await _finish(asyncio.create_task(run(True))) == await _finish(
        asyncio.create_task(run(False))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("handled", [False, True])
@pytest.mark.parametrize("wrapper", ["direct", "asyncio", "concurrent", "task"])
async def test_manager_preserves_pre_yield_timeout_rejection(
    monkeypatch, handled, wrapper
):
    monkeypatch.setattr(
        manager_module, "_TASK_CANCELLATION_DRAIN_TIMEOUT_SECONDS", SUCCESS_BUDGET
    )
    manager = ActionManager()
    handled_timeouts = []
    owners = []

    @action(action_manager=manager)
    async def return_started_stream(agent) -> object:
        """Prime a timeout-owning generator and return an invalid action result."""
        del agent
        scope = asyncio.timeout(None)

        async def stream():
            try:
                async with scope:
                    owners.append(asyncio.current_task())
                    try:
                        yield 1
                    finally:
                        scope.reschedule(asyncio.get_running_loop().time())
                        await asyncio.Event().wait()
            except TimeoutError:
                if not handled:
                    raise
                handled_timeouts.append(True)

        generator = stream()
        await anext(generator)
        if wrapper == "direct":
            return generator
        if wrapper == "task":

            async def completed():
                return generator

            result = asyncio.create_task(completed())
            await result
            return result
        result = (
            asyncio.get_running_loop().create_future()
            if wrapper == "asyncio"
            else concurrent.futures.Future()
        )
        result.set_result(generator)
        return result

    choice = ActionChoice(name="return_started_stream", arguments={})
    before = asyncio.all_tasks()
    execution = asyncio.create_task(manager.aexecute(SimpleNamespace(), choice))
    try:
        with pytest.raises(TypeError) as caught:
            await _finish(execution)
        assert type(caught.value) is TypeError
        assert owners == [execution]
        assert not execution.cancelled()
        assert execution.cancelling() == 0
        assert handled_timeouts == ([True] if handled else [])
        if handled:
            assert _notes(caught.value) == ""
        else:
            assert "timeouterror" in _notes(caught.value)
            assert "cleanup unresolved" not in _notes(caught.value)
        assert asyncio.all_tasks() == before
    finally:
        if not execution.done():
            execution.cancel()
        with suppress(TypeError, asyncio.CancelledError):
            await execution


@pytest.mark.asyncio
@pytest.mark.parametrize("expire_local", [False, True])
async def test_pre_yield_scope_does_not_consume_an_extra_external_request(expire_local):
    ready = asyncio.Event()
    scope = asyncio.timeout(None)

    async def run():
        async def stream():
            try:
                async with scope:
                    try:
                        yield 1
                    finally:
                        ready.set()
                        await asyncio.Event().wait()
            except TimeoutError:
                pytest.fail("an external cancellation request was lost")

        generator = stream()
        await anext(generator)
        try:
            await close_asyncgen_best_effort(
                generator, TypeError("invalid action result"), SUCCESS_BUDGET
            )
        finally:
            assert generator.ag_frame is None

    execution = asyncio.create_task(run())
    await _finish(asyncio.create_task(ready.wait()))
    if expire_local:
        # Both requests run before the cancellation relay can resume the scope.
        scope.reschedule(asyncio.get_running_loop().time())
        asyncio.get_running_loop().call_soon(execution.cancel, "external request")
    else:
        execution.cancel("external request")
    try:
        with pytest.raises(asyncio.CancelledError) as caught:
            await _finish(execution)
        assert execution.cancelled()
        assert execution.cancelling() == 1
        if not expire_local:
            assert caught.value.args == ("external request",)
    finally:
        if not execution.done():
            execution.cancel()
        with suppress(asyncio.CancelledError):
            await execution


@pytest.mark.asyncio
async def test_handled_pre_yield_timeout_does_not_restart_cleanup_deadline():
    release = asyncio.Event()

    async def run():
        handled = []

        async def stream():
            try:
                async with asyncio.timeout(0):
                    try:
                        yield 1
                    finally:
                        await asyncio.Event().wait()
            except TimeoutError:
                handled.append(True)
                await release.wait()

        generator = stream()
        await anext(generator)
        primary = TypeError("invalid action result")
        try:
            await close_asyncgen_best_effort(generator, primary, EXPIRY_BUDGET)
            assert handled == [True]
            assert "cleanup unresolved" in _notes(primary)
            assert "timeout" in _notes(primary)
            assert not release.is_set()
            assert asyncio.current_task().cancelling() == 0
            assert generator.ag_frame is None
        finally:
            release.set()
            await generator.aclose()

    await _finish(asyncio.create_task(run()))


@pytest.mark.asyncio
async def test_external_cancellation_survives_suspended_unwinding_and_forced_failure():
    ready = asyncio.Event()
    release = asyncio.Event()
    primary = TypeError("invalid action result")

    async def stream():
        try:
            yield 1
        finally:
            try:
                try:
                    execution.cancel("external request")
                    ready.set()
                    await release.wait()
                except asyncio.CancelledError:
                    # Genuine external cancellation has not been withdrawn.
                    await release.wait()
            except GeneratorExit:
                raise ValueError("forced interruption failed") from None

    generator = stream()
    await anext(generator)
    execution = asyncio.create_task(
        close_asyncgen_best_effort(generator, primary, EXPIRY_BUDGET)
    )
    try:
        await ready.wait()
        with pytest.raises(asyncio.CancelledError) as caught:
            await _finish(execution)
        assert caught.value.args == ("external request",)
        assert execution.cancelled()
        assert execution.cancelling() == 1
        assert "forced interruption failed" in _notes(caught.value)
        assert "cleanup unresolved" in _notes(caught.value)
        assert not release.is_set()
        assert generator.ag_frame is None
    finally:
        release.set()
        with suppress(asyncio.CancelledError):
            await execution
        await generator.aclose()


@pytest.mark.asyncio
async def test_pre_yield_timeout_diagnoses_unfinished_user_owned_child():
    release = asyncio.Event()
    child = asyncio.create_task(release.wait())

    async def run():
        handled = []
        scope = asyncio.timeout(None)

        async def stream():
            try:
                async with scope:
                    try:
                        yield 1
                    finally:
                        scope.reschedule(asyncio.get_running_loop().time())
                        await child
            except TimeoutError:
                handled.append(True)

        generator = stream()
        await anext(generator)
        primary = TypeError("invalid action result")
        try:
            await close_asyncgen_best_effort(generator, primary, SUCCESS_BUDGET)
            assert handled == [True]
            assert "cleanup unresolved" in _notes(primary)
            assert "remains live" in _notes(primary)
            assert child.cancelling() == 0
            assert not child.done()
            assert not release.is_set()
            assert asyncio.current_task().cancelling() == 0
            assert generator.ag_frame is None
        finally:
            await generator.aclose()

    try:
        await _finish(asyncio.create_task(run()))
    finally:
        release.set()
        await child
