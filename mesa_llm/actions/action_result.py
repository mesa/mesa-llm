from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mesa_llm.actions.action_manager import ActionChoice


@dataclass(frozen=True)
class ActResult:
    """Result returned by ``LLMAgent.act(...)`` after a successful action."""

    action: ActionChoice
    result: Any
