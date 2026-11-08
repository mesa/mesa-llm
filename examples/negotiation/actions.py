import logging
from typing import TYPE_CHECKING

from mesa_llm.actions import ActionManager, action

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


_action_manager = ActionManager()
logger = logging.getLogger(__name__)


@action(action_manager=_action_manager)
def speak_to(
    agent: "LLMAgent",
    listener_agents_unique_ids: list[int],
    message: str,
) -> dict[str, list[int]]:
    """Send a message only to currently eligible negotiation recipients.

    Args:
        agent: Provided automatically.
        listener_agents_unique_ids: The requested recipient IDs.
        message: The message to send.

    Returns:
        The requested, delivered, skipped, and failed recipient IDs.
    """
    requested = [int(recipient_id) for recipient_id in listener_agents_unique_ids]
    eligible_ids = set(agent._get_eligible_speak_to_recipient_ids())
    recipients_by_id = {
        candidate.unique_id: candidate for candidate in agent.model.agents
    }

    delivered = []
    skipped = []
    failed = []
    for recipient_id in dict.fromkeys(requested):
        recipient = recipients_by_id.get(recipient_id)
        if recipient is None or recipient is agent:
            skipped.append(recipient_id)
            continue

        if recipient_id not in eligible_ids:
            skipped.append(recipient_id)
            continue

        try:
            memory = getattr(recipient, "memory", None)
            add_to_memory = getattr(memory, "add_to_memory", None)
            if not callable(add_to_memory):
                skipped.append(recipient_id)
                continue

            add_to_memory(
                type="message",
                content={
                    "message": message,
                    "sender": agent.unique_id,
                },
            )
        except Exception:
            logger.exception(
                "Failed to deliver speak_to message from agent %s to agent %s.",
                agent.unique_id,
                recipient_id,
            )
            failed.append(recipient_id)
        else:
            delivered.append(recipient_id)

    return {
        "requested": requested,
        "delivered": delivered,
        "skipped": skipped,
        "failed": failed,
    }


@action
def buy_product(agent: "LLMAgent", chosen_product: str, chosen_price: int) -> str:
    """
    An action to set the brand of choice of the buyer agent. The product must be one of:
    ["Brand A Shoes", "Brand A Track Suit", "Brand B Shoes", "Brand B Track Suit"].

    Args:
        agent : The buyer agent.
        chosen_product : The product chosen by the buyer, specifying brand and type.
        chosen_price : The price of the product chosen by the buyer.

    Returns:
        str: The brand of choice of the buyer agent, either "A" or "B".
    """
    valid_products = {
        "Brand A Shoes": 40,
        "Brand A Track Suit": 50,
        "Brand B Shoes": 35,
        "Brand B Track Suit": 47,
    }

    if chosen_product not in valid_products:
        raise ValueError(
            f"Invalid product choice: {chosen_product}. Must be one of {list(valid_products.keys())}."
        )

    price = valid_products[chosen_product]
    if agent.budget < price:
        raise ValueError(f"Insufficient budget: {agent.budget}. Product costs {price}.")

    agent.products.append(chosen_product)
    agent.internal_state.append(f"Owner of the following product: {chosen_product}")
    agent.budget -= price

    # Get model and identify seller agent
    model = agent.model
    brand = "A" if "Brand A" in chosen_product else "B"

    # Increment sales of appropriate seller
    if brand == "A":
        model.seller_a.sales += 1
    else:  # brand == "B"
        model.seller_b.sales += 1

    return f"The agent has chosen {chosen_product} as their brand of choice. Remaining budget: {agent.budget}."
