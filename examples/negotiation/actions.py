from typing import TYPE_CHECKING

from mesa_llm.actions import action

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


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
