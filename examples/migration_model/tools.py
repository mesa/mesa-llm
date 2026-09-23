from typing import TYPE_CHECKING

from examples.migration_model.agent import (
    Citizen_tool_manager,
    CitizenState,
)
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=Citizen_tool_manager)
def move_to_safe_zone(agent: "LLMAgent") -> str:
    """
    Move a migrating citizen one step toward the safe zone.

    Args:
        agent: The agent performing the action.

    Returns:
        Description of the movement result.
    """

    if agent.state != CitizenState.MIGRATE:
        return "Agent is not migrating."

    safe_x, safe_y = agent.model.safe_zone[0]
    x, y = agent.pos

    new_x = x + (1 if safe_x > x else -1 if safe_x < x else 0)
    new_y = y + (1 if safe_y > y else -1 if safe_y < y else 0)

    new_pos = (new_x, new_y)
    agent.model.grid.move_agent(agent, new_pos)

    return f"Agent {agent.unique_id} moved toward safe zone {new_pos}"
