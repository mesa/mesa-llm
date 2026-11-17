from typing import TYPE_CHECKING

from examples.rumor_spreading.agents import (
    BeliefState,
    CitizenAgent,
    citizen_tool_manager,
    journalist_tool_manager,
)
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=citizen_tool_manager)
def update_belief(agent: "LLMAgent", new_state: str) -> str:
    """
    Update the citizen's personal belief about the rumor they have heard.

    Use this after deciding whether you believe, doubt, or want to debunk
    what you've been told. This is how you record your stance.

        Args:
            new_state: Your new belief. Must be exactly one of:
                "BELIEVER"  — you believe the rumor and are willing to spread it.
                "SKEPTIC"   — you doubt it but haven't proven it false.
                "DEBUNKED"  — you are confident it is misinformation.
            agent: Provided automatically.

        Returns:
            A confirmation string describing the state change.
    """
    valid_states = {
        "BELIEVER": BeliefState.BELIEVER,
        "SKEPTIC": BeliefState.SKEPTIC,
        "DEBUNKED": BeliefState.DEBUNKED,
    }

    if new_state not in valid_states:
        raise ValueError(
            f"Invalid state: '{new_state}'. Choose one of {list(valid_states.keys())}."
        )

    previous = agent.belief_state.value
    agent.belief_state = valid_states[new_state]

    return (
        f"Agent {agent.unique_id} ({agent.personality}) changed belief: "
        f"{previous} → {new_state.lower()}."
    )


@tool(tool_manager=journalist_tool_manager)
def publish_story(agent: "LLMAgent", headline: str, is_misinformation: bool) -> str:
    """
    Broadcast a news story to every citizen within the journalist's vision range.

    The story lands directly in each citizen's memory as a received message,
    where they can read and reason about it on their next turn.

        Args:
            headline: The headline or story content to publish. Be specific and realistic.
            is_misinformation: Set True if the story is fabricated, exaggerated, or misleading.
                               Set False if it is factual and verified.
            agent: Provided automatically.

        Returns:
            A summary of the broadcast — what was published and how many people received it.
    """
    nearby_citizens = [
        a
        for a in agent.model.grid.get_neighbors(
            agent.pos, moore=True, include_center=False, radius=agent.vision
        )
        if isinstance(a, CitizenAgent)
    ]

    tag = "[MISINFORMATION]" if is_misinformation else "[VERIFIED NEWS]"
    full_message = f"{tag} {headline}"

    for citizen in nearby_citizens:
        citizen.memory.add_to_memory(
            type="message",
            content={
                "message": full_message,
                "sender": agent.unique_id,
                "recipients": [c.unique_id for c in nearby_citizens],
            },
        )

    agent.stories_published += 1

    return (
        f"Journalist {agent.unique_id} ({agent.bias}) published: '{full_message}'. "
        f"Reached {len(nearby_citizens)} citizen(s). "
        f"Total stories this journalist has published: {agent.stories_published}."
    )
