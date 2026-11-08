from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mesa_llm.actions.action_manager import ActionChoice


class ActionPostCommitError(RuntimeError):
    """Report observer failures after an action has committed."""

    def __init__(
        self,
        action: ActionChoice,
        result: Any,
        observer_errors: dict[str, Exception],
    ) -> None:
        if not isinstance(observer_errors, dict):
            raise TypeError(
                "observer_errors must be a dict[str, Exception], "
                f"got {type(observer_errors).__name__}."
            )

        validated_observer_errors = dict(observer_errors)
        if not validated_observer_errors:
            raise ValueError("observer_errors must contain at least one error.")

        unsupported_observers = [
            observer
            for observer in validated_observer_errors
            if observer not in {"memory", "recorder"}
        ]
        if unsupported_observers:
            raise ValueError(
                "observer_errors keys must be 'memory' and/or 'recorder'; "
                f"got unsupported key(s): {unsupported_observers!r}."
            )

        for observer, error in validated_observer_errors.items():
            if not isinstance(error, Exception):
                raise TypeError(
                    f"observer_errors[{observer!r}] must be an Exception instance, "
                    f"got {type(error).__name__}."
                )

        failed_observers = ", ".join(validated_observer_errors)
        super().__init__(
            f"Action {action.name!r} committed, but post-commit observer(s) "
            f"failed: {failed_observers}."
        )
        self.committed = True
        self.action = action
        self.result = result
        self.observer_errors = validated_observer_errors


@dataclass(frozen=True)
class ActResult:
    """Result returned by ``LLMAgent.act(...)`` after a successful action."""

    action: ActionChoice
    result: Any
