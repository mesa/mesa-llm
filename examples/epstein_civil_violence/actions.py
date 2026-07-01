from typing import TYPE_CHECKING

from examples.epstein_civil_violence.agents import CitizenState
from mesa_llm.actions import action

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@action
def change_state(agent: "LLMAgent", state: str) -> str:
    """
    Change the state of the agent. The state can be "QUIET" or "ACTIVE"

        Args:
            state: The state to change the agent to. Must be one of the following: "QUIET" or "ACTIVE"
            agent: Provided automatically

        Returns:
            a string confirming the agent's new state.
    """
    state_map = {
        "QUIET": CitizenState.QUIET,
        "ACTIVE": CitizenState.ACTIVE,
    }
    if state not in state_map:
        raise ValueError(f"Invalid state: {state}")
    agent.state = state_map[state]
    return f"agent {agent.unique_id} changed state to {state}."


@action
def arrest_citizen(agent: "LLMAgent", citizen_id: int) -> str:
    """
    Arrest a citizen (only if they are active).

        Args:
            citizen_id: The unique id of the citizen to arrest.
            agent: Provided automatically

        Returns:
            a string confirming the citizen's arrest.
    """
    citizen = next(
        (agent for agent in agent.model.agents if agent.unique_id == citizen_id), None
    )
    citizen.state = CitizenState.ARRESTED
    citizen.jail_sentence_left = agent.random.randint(1, agent.max_jail_term)
    return f"agent {citizen_id} arrested by {agent.unique_id}."
