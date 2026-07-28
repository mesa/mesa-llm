import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from mesa.agent import Agent, AgentSet
from mesa.model import Model

from mesa_llm.parallel_stepping import (
    disable_automatic_parallel_stepping,
    enable_automatic_parallel_stepping,
    step_agents_multithreaded,
    step_agents_parallel,
    step_agents_parallel_sync,
)

_CHILD_PROCESS_BASE_ENV = os.environ.copy()


@pytest.fixture(autouse=True)
def restore_shuffle_do():
    """Keep monkey-patched Mesa activation isolated between tests."""
    disable_automatic_parallel_stepping()
    yield
    disable_automatic_parallel_stepping()


def _run_import_side_effect_check(import_statement: str) -> dict[str, bool]:
    repo_root = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        f"""
        import json

        from mesa.agent import AgentSet

        original_shuffle_do = AgentSet.shuffle_do
        original_do_async = getattr(AgentSet, "do_async", None)
        original_do_async_exists = hasattr(AgentSet, "do_async")

        {import_statement}

        print(
            json.dumps(
                {{
                    "shuffle_do_unchanged": AgentSet.shuffle_do is original_shuffle_do,
                    "do_async_unchanged": (
                        hasattr(AgentSet, "do_async") == original_do_async_exists
                        and getattr(AgentSet, "do_async", None) is original_do_async
                    ),
                }}
            )
        )
        """
    )
    env = _CHILD_PROCESS_BASE_ENV.copy()
    env.update(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "Import side-effect child process failed.\n"
        f"return code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return json.loads(result.stdout.splitlines()[-1])


class DummyModel(Model):
    def __init__(self):
        super().__init__(rng=42)
        self.parallel_stepping = False


class SyncAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.counter = 0

    def step(self):
        self.counter += 1


class AsyncAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.counter = 0

    async def astep(self):
        self.counter += 1


def test_import_mesa_llm_does_not_patch_shuffle_do():
    result = _run_import_side_effect_check("import mesa_llm")

    assert result["shuffle_do_unchanged"]
    assert result["do_async_unchanged"]


def test_public_from_imports_do_not_patch_shuffle_do():
    result = _run_import_side_effect_check(
        "from mesa_llm import ("
        "ActionManager, "
        "ToolManager, "
        "default_actions, "
        "default_tools, "
        "enable_automatic_parallel_stepping"
        ")"
    )

    assert result["shuffle_do_unchanged"]
    assert result["do_async_unchanged"]


def test_import_side_effect_check_reports_child_output():
    with pytest.raises(AssertionError) as exc_info:
        _run_import_side_effect_check(
            'print("child stdout sentinel"); '
            'raise RuntimeError("child stderr sentinel")'
        )

    message = str(exc_info.value)
    assert "return code:" in message
    assert "child stdout sentinel" in message
    assert "child stderr sentinel" in message


@pytest.mark.asyncio
async def test_step_agents_parallel():
    m = DummyModel()
    a1 = SyncAgent(m)
    a2 = AsyncAgent(m)
    await step_agents_parallel([a1, a2])
    assert a1.counter == 1
    assert a2.counter == 1


def test_step_agents_multithreaded():
    m = DummyModel()
    a1 = SyncAgent(m)
    a2 = AsyncAgent(m)
    step_agents_multithreaded([a1, a2])
    assert a1.counter == 1
    assert a2.counter == 1


def test_automatic_parallel_shuffle_do():
    """
    verify that enable_automatic_parallel_stepping
    monkey patches AgentSet.shuffle_do and ends up
    using step_agents_parallel_sync
    """
    m = DummyModel()
    m.parallel_stepping = True

    # SyncAgent that will be called by AgentSet.shuffle_do
    a1 = SyncAgent(m)
    agents = AgentSet([a1], random=m.random)

    original_shuffle_do = AgentSet.shuffle_do

    # enable patch
    enable_automatic_parallel_stepping("asyncio")
    assert AgentSet.shuffle_do is not original_shuffle_do
    assert hasattr(AgentSet, "do_async")

    # shuffle_do should now call step_agents_parallel_sync
    # instead of individual step, so the counter still ends up 1
    agents.shuffle_do("step")
    assert a1.counter == 1

    # disable patch and check that shuffle_do calls default (and will step again)
    disable_automatic_parallel_stepping()
    assert AgentSet.shuffle_do is original_shuffle_do
    assert not hasattr(AgentSet, "do_async")
    agents.shuffle_do("step")
    assert a1.counter == 2


def test_step_agents_parallel_sync_in_running_loop():
    # ensure no exception is raised if we call the sync wrapper
    # while an event loop is already running
    m = DummyModel()
    a1 = SyncAgent(m)
    a2 = AsyncAgent(m)

    async def wrapper():
        # running inside an event loop
        step_agents_parallel_sync([a1, a2])

    asyncio.run(wrapper())
    assert a1.counter == 1
    assert a2.counter == 1
