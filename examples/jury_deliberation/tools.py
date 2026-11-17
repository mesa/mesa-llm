from typing import TYPE_CHECKING

from examples.jury_deliberation.agents import JurorAgent, juror_tool_manager
from examples.jury_deliberation.case_data import get_evidence_detail
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=juror_tool_manager)
def speak_to_room(agent: "LLMAgent", statement: str) -> str:
    """Broadcast a statement to the entire jury room. All jurors will hear this.

    Args:
        agent: The juror speaking (provided automatically).
        statement: The argument or comment to share with the jury.

    Returns:
        Confirmation of the broadcast.
    """
    name = agent.persona["name"]

    # Limit statement length to prevent token explosion and ensure concise arguments
    statement = statement[:400]

    # add to the shared discussion log so everyone can read it
    agent.model.discussion_log.append(
        {
            "juror_id": agent.unique_id,
            "name": name,
            "round": agent.model.current_round,
            "statement": statement,
        }
    )

    # also push into other jurors' memory so they can recall it later
    other_jurors = [
        juror
        for juror in agent.model.agents
        if isinstance(juror, JurorAgent) and juror.unique_id != agent.unique_id
    ]
    recipient_ids = [j.unique_id for j in other_jurors]

    for juror in other_jurors:
        juror.memory.add_to_memory(
            type="message",
            content={
                "message": statement,
                "sender": agent.unique_id,
                "recipients": recipient_ids,
            },
        )

    # nudge listeners' beliefs based on whether the statement leans guilty or innocent
    # simple heuristic: presence of guilt-leaning or innocence-leaning keywords
    persuasion = _estimate_persuasion_direction(statement)
    for juror in other_jurors:
        juror.update_belief(persuasion)
        # Protect belief range to prevent bugs if persuasion accumulates
        juror.guilt_belief = max(0.0, min(1.0, juror.guilt_belief))

    return f"{name} spoke to the jury: {statement[:60]}..."


def _estimate_persuasion_direction(statement: str) -> float:
    """Rough heuristic to guess if a statement argues for guilt or innocence.
    Returns positive (0.1) for guilt-leaning, negative (-0.1) for innocence-leaning.
    Gradual persuasion produces better deliberation dynamics."""
    text = statement.lower()

    guilt_signals = [
        "guilty",
        "fingerprint",
        "caught",
        "evidence against",
        "convicted",
        "prior record",
        "suspicious",
        "pawn shop",
        "broke in",
    ]
    innocence_signals = [
        "innocent",
        "alibi",
        "reasonable doubt",
        "not enough",
        "circumstantial",
        "biased",
        "explained",
        "legitimate",
    ]

    guilt_score = sum(1 for w in guilt_signals if w in text)
    innocence_score = sum(1 for w in innocence_signals if w in text)

    if "evidence" in text or "proof" in text:
        guilt_score += 1

    if guilt_score > innocence_score:
        return 0.1
    elif innocence_score > guilt_score:
        return -0.1
    return 0.0


@tool(tool_manager=juror_tool_manager)
def review_evidence(agent: "LLMAgent", evidence_id: str) -> str:
    """Look up a specific piece of evidence from the case file.
    Available IDs: E1, E2, E3, E4, E5, E6, E7.

    This avoids stuffing all evidence into every prompt — jurors request
    what they need on demand.

    Args:
        agent: The juror reviewing evidence (provided automatically).
        evidence_id: The ID of the evidence to review (e.g. "E2").

    Returns:
        The full evidence details, or an error if the ID is invalid.
    """
    return get_evidence_detail(evidence_id)


@tool(tool_manager=juror_tool_manager)
def cast_vote(agent: "LLMAgent", verdict: str) -> str:
    """Cast a formal vote during a voting round.

    Args:
        agent: The juror voting (provided automatically).
        verdict: Must be "guilty" or "not_guilty".

    Returns:
        Confirmation of the vote.
    """
    verdict = verdict.lower().strip()
    if verdict not in ("guilty", "not_guilty"):
        return f"Invalid verdict '{verdict}'. Must be 'guilty' or 'not_guilty'."

    agent.vote = verdict
    name = agent.persona["name"]

    # record in the model's vote history for tracking
    agent.model.current_votes[agent.unique_id] = verdict

    return f"{name} voted: {verdict}"
