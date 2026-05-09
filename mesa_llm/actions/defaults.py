"""Explicit action-set factories for Mesa-LLM capability configuration."""

from collections.abc import Callable

from mesa_llm.actions.inbuilt_actions import wait


def default_actions() -> tuple[Callable, ...]:
    """Return the recommended default actions."""
    return (wait,)


__all__ = ["default_actions"]
