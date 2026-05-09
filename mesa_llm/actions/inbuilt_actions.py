from __future__ import annotations

from typing import Any

from mesa_llm.actions.action_decorator import action


@action
def wait(agent: Any) -> str:
    """Take no action for this turn.

    Returns:
        A confirmation that the agent waited.
    """
    return "waited"


__all__ = ["wait"]
