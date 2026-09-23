from __future__ import annotations

from typing import TYPE_CHECKING

from examples.student.agent import StudentState, student_tool_manager
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=student_tool_manager)
def graduate(agent: LLMAgent) -> str:
    """
    Mark a student as graduated once they reach grade 12.

    Args:
        agent: Provided automatically.

    Returns:
        A message describing whether the student graduated.
    """
    if getattr(agent, "grade", 0) < 12:
        return f"Student {agent.unique_id} cannot graduate before grade 12."

    agent.state = StudentState.GRADUATE
    agent.current_school = None
    agent.refresh_internal_state()
    return f"Student {agent.unique_id} has graduated."


@tool(tool_manager=student_tool_manager)
def leave_school(agent: LLMAgent) -> str:
    """
    Remove a student from school and mark them as a dropout.

    Args:
        agent: Provided automatically.

    Returns:
        A message confirming the updated student status.
    """
    agent.state = StudentState.DROPOUT
    agent.current_school = None
    agent.refresh_internal_state()
    return f"Student {agent.unique_id} has left school and is now a dropout."
