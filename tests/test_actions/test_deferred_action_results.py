"""Exercise completed-wrapper cleanup and observer rejection boundaries."""

import asyncio
import concurrent.futures
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import mesa_llm.actions.action_manager as manager_module
import mesa_llm.llm_agent as agent_module
from mesa_llm.actions import ActionChoice, ActionManager, action
from mesa_llm.llm_agent import LLMAgent


class CompletedOnlyFuture(concurrent.futures.Future):
    """Fail a regression immediately if cleanup ever performs a blocking read."""

    def __init__(self):
        super().__init__()
        self.result_calls = 0

    def result(self, timeout=None):
        assert self.done(), "cleanup must not wait on an unfinished executor Future"
        self.result_calls += 1
        return super().result(timeout=timeout)


def _manager_for_payload(payload, asynchronous=False):
    manager = ActionManager()
    if asynchronous:

        @action(action_manager=manager)
        async def deferred_result(agent) -> object:
            """Return a deferred payload from an asynchronous action."""
            del agent
            return payload

    else:

        @action(action_manager=manager)
        def deferred_result(agent) -> object:
            """Return a deferred payload from a synchronous action."""
            del agent
            return payload

    return manager, ActionChoice(name=deferred_result.__name__, arguments={})


async def _execute(manager, choice, entrypoint, agent=None):
    actor = SimpleNamespace() if agent is None else agent
    if entrypoint == "execute":
        return manager.execute(actor, choice)
    return await manager.aexecute(actor, choice)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["execute", "aexecute", "post-await"])
@pytest.mark.parametrize("depth", [0, 1, 3])
async def test_concurrent_future_chains_cancel_leaf_without_blocking(entrypoint, depth):
    leaf = CompletedOnlyFuture()
    root = leaf
    wrappers = []
    for _ in range(depth):
        wrapper = CompletedOnlyFuture()
        wrapper.set_result(root)
        wrappers.append(wrapper)
        root = wrapper
    manager, choice = _manager_for_payload(root, entrypoint == "post-await")
    with pytest.raises(TypeError) as exc_info:
        await _execute(manager, choice, entrypoint)
    assert type(exc_info.value) is TypeError
    assert "one completed result" in str(exc_info.value)
    assert leaf.cancelled()
    assert leaf.result_calls == 0
    assert all(wrapper.result_calls == 1 for wrapper in wrappers)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["execute", "aexecute", "post-await"])
@pytest.mark.parametrize("length", [1, 2])
async def test_concurrent_future_identity_cycles_are_bounded(entrypoint, length):
    cycle = [CompletedOnlyFuture() for _ in range(length)]
    for index, future in enumerate(cycle):
        future.set_result(cycle[(index + 1) % length])
    manager, choice = _manager_for_payload(cycle[0], entrypoint == "post-await")
    with pytest.raises(TypeError) as exc_info:
        await _execute(manager, choice, entrypoint)
    assert type(exc_info.value) is TypeError
    assert any("cycle" in note for note in exc_info.value.__notes__)
    assert all(future.result_calls == 1 for future in cycle)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["execute", "aexecute", "post-await"])
async def test_running_concurrent_future_is_rejected_with_diagnostic(entrypoint):
    future = CompletedOnlyFuture()
    assert future.set_running_or_notify_cancel()
    manager, choice = _manager_for_payload(future, entrypoint == "post-await")
    try:
        with pytest.raises(TypeError) as exc_info:
            await _execute(manager, choice, entrypoint)
        assert type(exc_info.value) is TypeError
        assert future.running()
        assert future.result_calls == 0
        assert any("Cleanup unresolved" in note for note in exc_info.value.__notes__)
    finally:
        future.set_result(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["execute", "aexecute", "post-await"])
async def test_completed_concurrent_future_exception_keeps_rejection(entrypoint):
    future = CompletedOnlyFuture()
    future.set_exception(RuntimeError("worker failed"))
    manager, choice = _manager_for_payload(future, entrypoint == "post-await")
    with pytest.raises(TypeError) as exc_info:
        await _execute(manager, choice, entrypoint)
    assert type(exc_info.value) is TypeError
    assert future.result_calls == 1
    assert any("worker failed" in note for note in exc_info.value.__notes__)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["execute", "aexecute", "post-await"])
@pytest.mark.parametrize("payload_kind", ["coroutine", "generator"])
async def test_completed_executor_wrapper_closes_deferred_payload(
    entrypoint, payload_kind
):
    effects = []

    async def coroutine_body():
        effects.append("must not execute")

    def generator_body():
        try:
            yield "started"
        finally:
            effects.append("closed")

    payload = coroutine_body() if payload_kind == "coroutine" else generator_body()
    if payload_kind == "generator":
        assert next(payload) == "started"
    wrapper = CompletedOnlyFuture()
    wrapper.set_result(payload)
    manager, choice = _manager_for_payload(wrapper, entrypoint == "post-await")
    try:
        with pytest.raises(TypeError):
            await _execute(manager, choice, entrypoint)
        if payload_kind == "coroutine":
            assert payload.cr_frame is None
            assert effects == []
        else:
            assert payload.gi_frame is None
            assert effects == ["closed"]
    finally:
        payload.close()


@pytest.mark.asyncio
async def test_mixed_completed_wrappers_close_async_generator():
    closed = []

    async def generator_body():
        try:
            yield "started"
        finally:
            await asyncio.sleep(0)
            closed.append(True)

    payload = generator_body()
    assert await anext(payload) == "started"
    inner = asyncio.get_running_loop().create_future()
    inner.set_result(payload)
    outer = CompletedOnlyFuture()
    outer.set_result(inner)
    manager, choice = _manager_for_payload(outer, asynchronous=True)
    try:
        with pytest.raises(TypeError):
            await manager.aexecute(SimpleNamespace(), choice)
        assert payload.ag_frame is None
        assert closed == [True]
    finally:
        await payload.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["execute", "aexecute"])
async def test_mixed_wrappers_do_not_cancel_current_task(entrypoint):
    task = asyncio.current_task()
    before = task.cancelling()
    inner = asyncio.get_running_loop().create_future()
    inner.set_result(task)
    outer = CompletedOnlyFuture()
    outer.set_result(inner)
    manager, choice = _manager_for_payload(outer)
    with pytest.raises(TypeError) as exc_info:
        await _execute(manager, choice, entrypoint)
    assert task.cancelling() == before
    assert any("current task" in note for note in exc_info.value.__notes__)
    await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["execute_action", "aexecute_action", "aact"])
async def test_agent_rejects_completed_executor_payload_before_observers(
    monkeypatch, entrypoint
):
    leaf = CompletedOnlyFuture()
    outer = CompletedOnlyFuture()
    outer.set_result(leaf)
    manager, choice = _manager_for_payload(outer)
    agent = object.__new__(LLMAgent)
    agent.unique_id = 1
    agent._action_manager = manager
    agent.memory = SimpleNamespace(add_to_memory=Mock(), aadd_to_memory=AsyncMock())
    agent.recorder = Mock()
    agent.achoose_action = AsyncMock(return_value=choice)
    deepcopy = Mock(side_effect=AssertionError("deferred result reached deepcopy"))
    monkeypatch.setattr(agent_module, "copy", SimpleNamespace(deepcopy=deepcopy))
    with pytest.raises(TypeError) as exc_info:
        if entrypoint == "execute_action":
            agent.execute_action(choice)
        elif entrypoint == "aexecute_action":
            await agent.aexecute_action(choice)
        else:
            await agent.aact("Select one action.")
    assert type(exc_info.value) is TypeError
    assert leaf.cancelled()
    assert leaf.result_calls == 0
    assert outer.result_calls == 1
    deepcopy.assert_not_called()
    agent.memory.add_to_memory.assert_not_called()
    agent.memory.aadd_to_memory.assert_not_awaited()
    agent.recorder.record_event.assert_not_called()
    if entrypoint == "aact":
        agent.achoose_action.assert_awaited_once()


@pytest.mark.asyncio
async def test_already_cancelling_task_is_drained_without_second_cancel():
    started = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    finished = []

    async def body():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            finished.append(True)

    child = asyncio.create_task(body())
    try:
        await started.wait()
        child.cancel()
        await cleaning.wait()
        before = child.cancelling()
        asyncio.get_running_loop().call_soon(release.set)
        manager, choice = _manager_for_payload(child, asynchronous=True)
        with pytest.raises(TypeError):
            await manager.aexecute(SimpleNamespace(), choice)
        assert child.done()
        assert child.cancelled()
        assert child.cancelling() == before
        assert finished == [True]
    finally:
        release.set()
        with suppress(asyncio.CancelledError):
            await child


@pytest.mark.asyncio
async def test_stubborn_task_has_bounded_unresolved_cleanup(monkeypatch):
    monkeypatch.setattr(
        manager_module, "_TASK_CANCELLATION_DRAIN_TIMEOUT_SECONDS", 0.01
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def body():
        try:
            started.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    child = asyncio.create_task(body())
    try:
        await started.wait()
        manager, choice = _manager_for_payload(child, asynchronous=True)
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(SimpleNamespace(), choice)
        assert not child.done()
        assert child.cancelling() == 1
        assert any("Cleanup unresolved" in note for note in exc_info.value.__notes__)
    finally:
        release.set()
        await child


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, "completed", [1, 2], {"done": True}])
async def test_completed_results_still_pass_through_unchanged(payload):
    manager, choice = _manager_for_payload(payload, asynchronous=True)
    assert await manager.aexecute(SimpleNamespace(), choice) is payload
