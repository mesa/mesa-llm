from typing import TYPE_CHECKING

from examples.sugarscrap_g1mt.agents import (
    Resource,
    Trader,
)
from mesa_llm.actions import action

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@action
def move_to_best_resource(agent: "LLMAgent") -> str:
    """
    Move the agent to the best resource cell within its vision range.
        Args:
            agent: Provided automatically
        Returns:
            A string confirming the new position of the agent.
    """

    def calculate_potential_welfare(sugar_if_moved, spice_if_moved):
        m_total = agent.metabolism_sugar + agent.metabolism_spice
        w_sugar = agent.metabolism_sugar / m_total
        w_spice = agent.metabolism_spice / m_total

        return (sugar_if_moved**w_sugar) * (spice_if_moved**w_spice)

    best_cell = agent.pos

    current_welfare = calculate_potential_welfare(agent.sugar, agent.spice)
    max_welfare = current_welfare

    x, y = agent.pos
    vision = agent.vision

    for dx in range(-vision, vision + 1):
        for dy in range(-vision, vision + 1):
            nx, ny = x + dx, y + dy

            if not agent.model.grid.out_of_bounds((nx, ny)):
                cell_contents = agent.model.grid.get_cell_list_contents((nx, ny))

                potential_sugar = agent.sugar
                potential_spice = agent.spice

                for obj in cell_contents:
                    if isinstance(obj, Resource):
                        if obj.type == "sugar":
                            potential_sugar += obj.current_amount
                        elif obj.type == "spice":
                            potential_spice += obj.current_amount

                welfare_here = calculate_potential_welfare(
                    potential_sugar, potential_spice
                )

                if welfare_here > max_welfare:
                    max_welfare = welfare_here
                    best_cell = (nx, ny)

    if best_cell != agent.pos:
        agent.model.grid.move_agent(agent, best_cell)

        harvested_sugar = 0
        harvested_spice = 0

        cell_contents = agent.model.grid.get_cell_list_contents(best_cell)
        for obj in cell_contents:
            if isinstance(obj, Resource):
                amount = obj.current_amount
                if obj.type == "sugar":
                    harvested_sugar += amount
                elif obj.type == "spice":
                    harvested_spice += amount
                obj.current_amount = 0

        agent.sugar += harvested_sugar
        agent.spice += harvested_spice
        agent.model.total_sugar_harvested += harvested_sugar
        agent.model.total_spice_harvested += harvested_spice

        return (
            f"Agent {agent.unique_id} moved to {best_cell} to maximize welfare. "
            f"Harvested {harvested_sugar} sugar and {harvested_spice} spice. "
            f"New Welfare: {max_welfare:.2f}"
        )

    return (
        f"Agent {agent.unique_id} stayed put as no better welfare options were found."
    )


@action
def propose_trade(
    agent: "LLMAgent", other_agent_id: int, sugar_amount: int, spice_amount: int
) -> str:
    """
    Propose a trade to another agent.

        Args:
            other_agent_id: The unique id of the other agent to trade with.
            sugar_amount: The amount of sugar to offer.
            spice_amount: The amount of spice to offer.
            agent: Provided automatically

        Returns:
            A string confirming the trade proposal.
    """
    other_agent = next(
        (a for a in agent.model.agents if a.unique_id == other_agent_id), None
    )

    if other_agent is None:
        return f"Agent {other_agent_id} not found."

    if not isinstance(other_agent, Trader):
        return f"agent {other_agent_id} is not a valid trader."

    # Simple trade acceptance logic for demonstration
    if sugar_amount <= 0 or spice_amount <= 0:
        return "sugar_amount and spice_amount must be positive."

    if agent.sugar < sugar_amount or other_agent.spice < spice_amount:
        return (
            f"agent {agent.unique_id} or agent {other_agent_id} "
            "does not have enough resources for this trade."
        )

    if other_agent.calculate_mrs() > agent.calculate_mrs():
        agent.sugar -= sugar_amount
        agent.spice += spice_amount
        other_agent.sugar += sugar_amount
        other_agent.spice -= spice_amount
        return f"agent {agent.unique_id} traded {sugar_amount} sugar for {spice_amount} spice with agent {other_agent_id}."
    else:
        return f"agent {other_agent_id} rejected the trade proposal from agent {agent.unique_id}."
