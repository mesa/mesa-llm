import asyncio
import gc
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mesa_llm.actions import ActionChoice, ActionManager, action
from mesa_llm.llm_agent import LLMAgent


def _assert_nested_awaitable_error(error: BaseException, action_name: str) -> None:
    assert type(error) is TypeError
    message = str(error).casefold()
    assert action_name.casefold() in message
    assert "one completed result" in message
    assert "nested awaitable" in message


async def _await_cancelled(task: asyncio.Task) -> None:
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_completed_future_cleans_pending_task_result():
    release = asyncio.Event()
    started = asyncio.Event()
    state = SimpleNamespace(mutations=0, child=None)
    manager = ActionManager()

    @action(action_manager=manager)
    async def completed_wrapper_task(agent) -> object:
        """Return a completed Future containing a pending child Task."""

        async def mutate_after_release():
            started.set()
            await release.wait()
            agent.mutations += 1

        agent.child = asyncio.create_task(mutate_after_release())
        await started.wait()
        wrapper = asyncio.get_running_loop().create_future()
        wrapper.set_result(agent.child)
        return wrapper

    try:
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(
                state,
                ActionChoice(name="completed_wrapper_task", arguments={}),
            )

        _assert_nested_awaitable_error(
            exc_info.value,
            "completed_wrapper_task",
        )
        assert state.child.cancelling() > 0
        await _await_cancelled(state.child)
        release.set()
        await asyncio.sleep(0)
        assert state.mutations == 0
    finally:
        release.set()
        if state.child is not None and not state.child.done():
            state.child.cancel()
            await _await_cancelled(state.child)


@pytest.mark.asyncio
async def test_completed_future_chain_cleans_pending_task_result():
    release = asyncio.Event()
    state = SimpleNamespace(mutations=0, child=None)
    manager = ActionManager()

    @action(action_manager=manager)
    def completed_wrapper_chain(agent) -> object:
        """Return completed Future wrappers around a pending child Task."""

        async def mutate_after_release():
            await release.wait()
            agent.mutations += 1

        loop = asyncio.get_running_loop()
        agent.child = asyncio.create_task(mutate_after_release())
        inner = loop.create_future()
        inner.set_result(agent.child)
        outer = loop.create_future()
        outer.set_result(inner)
        return outer

    try:
        with pytest.raises(TypeError) as exc_info:
            await manager.aexecute(
                state,
                ActionChoice(name="completed_wrapper_chain", arguments={}),
            )

        _assert_nested_awaitable_error(
            exc_info.value,
            "completed_wrapper_chain",
        )
        assert state.child.cancelling() > 0
        await _await_cancelled(state.child)
        release.set()
        await asyncio.sleep(0)
        assert state.mutations == 0
    finally:
        release.set()
        if state.child is not None and not state.child.done():
            state.child.cancel()
            await _await_cancelled(state.child)


@pytest.mark.asyncio
async def test_completed_future_closes_native_coroutine_result(recwarn):
    state = SimpleNamespace(mutations=0, child=None)
    manager = ActionManager()

    @action(action_manager=manager)
    async def completed_wrapper_coroutine(agent) -> object:
        """Return a completed Future containing a native coroutine."""

        async def mutate_later():
            agent.mutations += 1

        agent.child = mutate_later()
        wrapper = asyncio.get_running_loop().create_future()
        wrapper.set_result(agent.child)
        return wrapper

    with pytest.raises(TypeError) as exc_info:
        await manager.aexecute(
            state,
            ActionChoice(name="completed_wrapper_coroutine", arguments={}),
        )

    _assert_nested_awaitable_error(
        exc_info.value,
        "completed_wrapper_coroutine",
    )
    assert state.child.cr_frame is None
    assert state.mutations == 0
    gc.collect()
    assert not [
        warning for warning in recwarn if "was never awaited" in str(warning.message)
    ]


@pytest.mark.asyncio
async def test_completed_future_identity_cycle_does_not_recurse_forever():
    manager = ActionManager()

    @action(action_manager=manager)
    async def completed_wrapper_cycle(agent) -> object:
        """Return a completed Future whose result is itself."""

        del agent
        wrapper = asyncio.get_running_loop().create_future()
        wrapper.set_result(wrapper)
        return wrapper

    with pytest.raises(TypeError) as exc_info:
        await manager.aexecute(
            SimpleNamespace(),
            ActionChoice(name="completed_wrapper_cycle", arguments={}),
        )

    error = exc_info.value
    _assert_nested_awaitable_error(error, "completed_wrapper_cycle")
    notes = getattr(error, "__notes__", ())
    assert any("cycle" in note.casefold() for note in notes)


@pytest.mark.asyncio
async def test_completed_future_current_task_is_not_self_cancelled():
    manager = ActionManager()
    execution_task = asyncio.current_task()
    assert execution_task is not None
    cancelling_before = execution_task.cancelling()

    @action(action_manager=manager)
    async def completed_wrapper_current_task(agent) -> object:
        """Return a completed Future containing the current Task."""

        del agent
        wrapper = asyncio.get_running_loop().create_future()
        wrapper.set_result(asyncio.current_task())
        return wrapper

    with pytest.raises(TypeError) as exc_info:
        await manager.aexecute(
            SimpleNamespace(),
            ActionChoice(name="completed_wrapper_current_task", arguments={}),
        )

    error = exc_info.value
    _assert_nested_awaitable_error(error, "completed_wrapper_current_task")
    assert execution_task.cancelling() == cancelling_before
    notes = getattr(error, "__notes__", ())
    assert any("current task" in note.casefold() for note in notes)


@pytest.mark.asyncio
async def test_llm_agent_does_not_record_completed_wrapper_child():
    release = asyncio.Event()
    started = asyncio.Event()
    manager = ActionManager()
    agent = object.__new__(LLMAgent)
    agent.unique_id = 1
    agent.mutations = 0
    agent.child = None
    agent._action_manager = manager
    agent.memory = SimpleNamespace(aadd_to_memory=AsyncMock())
    agent.recorder = Mock()

    @action(action_manager=manager)
    async def completed_wrapper_agent(agent) -> object:
        """Return a completed Future containing a pending child Task."""

        async def mutate_after_release():
            started.set()
            await release.wait()
            agent.mutations += 1

        agent.child = asyncio.create_task(mutate_after_release())
        await started.wait()
        wrapper = asyncio.get_running_loop().create_future()
        wrapper.set_result(agent.child)
        return wrapper

    try:
        with pytest.raises(TypeError) as exc_info:
            await agent.aexecute_action(
                ActionChoice(name="completed_wrapper_agent", arguments={}),
            )

        _assert_nested_awaitable_error(
            exc_info.value,
            "completed_wrapper_agent",
        )
        assert agent.child.cancelling() > 0
        await _await_cancelled(agent.child)
        assert agent.mutations == 0
        agent.memory.aadd_to_memory.assert_not_awaited()
        agent.recorder.record_event.assert_not_called()
    finally:
        release.set()
        if agent.child is not None and not agent.child.done():
            agent.child.cancel()
            await _await_cancelled(agent.child)
