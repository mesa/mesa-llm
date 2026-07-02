"""
Automatic parallel stepping for Mesa-LLM simulations.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
from typing import TYPE_CHECKING

from mesa.agent import Agent, AgentSet

if TYPE_CHECKING:
    from .llm_agent import LLMAgent

logger = logging.getLogger(__name__)

# Global variable to control parallel stepping mode
_PARALLEL_STEPPING_MODE = "asyncio"  # or "threading"


async def step_agents_parallel(agents: list[Agent | LLMAgent]) -> None:
    """Step all agents in parallel using async/await."""
    tasks = []
    for agent in agents:
        if hasattr(agent, "astep"):
            tasks.append(agent.astep())
        elif hasattr(agent, "step"):
            tasks.append(_sync_step(agent))
    await asyncio.gather(*tasks)


async def _sync_step(agent: Agent) -> None:
    """Run synchronous step in async context."""
    agent.step()


def step_agents_multithreaded(agents: list[Agent | LLMAgent]) -> None:
    """Step all agents in parallel using threads."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for agent in agents:
            if hasattr(agent, "astep"):
                # run async steps in the event loop in a thread
                futures.append(
                    executor.submit(lambda agent=agent: asyncio.run(agent.astep()))
                )
            elif hasattr(agent, "step"):
                futures.append(executor.submit(agent.step))

        for future in futures:
            future.result()


def step_agents_parallel_sync(agents: list[Agent | LLMAgent]) -> None:
    """Synchronous wrapper for parallel stepping using the global mode."""
    if _PARALLEL_STEPPING_MODE == "asyncio":
        try:
            asyncio.get_running_loop()
            # If in event loop, use thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(step_agents_parallel(agents))
                )
                future.result()
        except RuntimeError:
            # No event loop - create one
            asyncio.run(step_agents_parallel(agents))
    elif _PARALLEL_STEPPING_MODE == "threading":
        step_agents_multithreaded(agents)
    else:
        raise ValueError(f"Unknown parallel stepping mode: {_PARALLEL_STEPPING_MODE}")


# Patch Mesa's shuffle_do for automatic parallel detection
_original_shuffle_do = AgentSet.shuffle_do
_original_do_async = getattr(AgentSet, "do_async", None)
_original_do_async_exists = hasattr(AgentSet, "do_async")


def _enhanced_shuffle_do(self, method: str, *args, **kwargs):
    """Enhanced shuffle_do with automatic parallel stepping."""
    if method == "step" and self:
        agent = next(iter(self))
        if hasattr(agent, "model") and getattr(agent.model, "parallel_stepping", False):
            step_agents_parallel_sync(list(self))
            return
    _original_shuffle_do(self, method, *args, **kwargs)


def enable_automatic_parallel_stepping(mode: str = "asyncio"):
    """Enable automatic parallel stepping with selectable mode ('asyncio' or 'threading')."""
    global _PARALLEL_STEPPING_MODE  # noqa: PLW0603
    if mode not in ("asyncio", "threading"):
        raise ValueError("mode must be either 'asyncio' or 'threading'")
    _PARALLEL_STEPPING_MODE = mode
    AgentSet.shuffle_do = _enhanced_shuffle_do
    AgentSet.do_async = _agentset_do_async


def disable_automatic_parallel_stepping():
    """Restore original shuffle_do behavior."""
    AgentSet.shuffle_do = _original_shuffle_do
    if _original_do_async_exists:
        AgentSet.do_async = _original_do_async
    else:
        with contextlib.suppress(AttributeError):
            delattr(AgentSet, "do_async")


# --- Monkey-patch AgentSet with do_async for async parallel method calls ---


def _agentset_do_async(self, method: str, *args, **kwargs):
    """
    Call the given async method on all agents in the set in parallel.
    Usage: await agents.do_async("async_function")
    """
    logger.info("Running async method '%s' on %d agents", method, len(self))

    async def _run():
        tasks = []
        for agent in self:
            fn = getattr(agent, method, None)
            if fn is not None and asyncio.iscoroutinefunction(fn):
                tasks.append(fn(*args, **kwargs))
            else:
                raise AttributeError(
                    f"Agent {agent} does not have async method '{method}'"
                )
        return await asyncio.gather(*tasks)

    return _run()
