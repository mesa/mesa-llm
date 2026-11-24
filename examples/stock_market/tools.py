from typing import TYPE_CHECKING

from examples.stock_market.agents import trader_tool_manager
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@tool(tool_manager=trader_tool_manager)
def execute_trade(agent: "LLMAgent", action: str, quantity: int = 1) -> str:
    """
    Execute a stock trade on behalf of a trader agent.

    Args:
        agent: The trader agent executing the trade.
        action: One of "BUY", "SELL", or "HOLD".
        quantity: Number of shares to buy or sell (default: 1).

    Returns:
        str: A summary of the trade outcome.
    """
    action = action.upper().strip()
    if action not in {"BUY", "SELL", "HOLD"}:
        raise ValueError(f"Invalid action '{action}'. Must be BUY, SELL, or HOLD.")

    price = agent.model.current_price

    if action == "BUY":
        total_cost = price * quantity
        if agent.budget < total_cost:
            return (
                f"Cannot BUY {quantity} share(s) at ${price:.2f} each "
                f"(total ${total_cost:.2f}). Insufficient budget: ${agent.budget:.2f}."
            )
        agent.budget -= total_cost
        agent.shares += quantity
        agent.trades += 1
        return (
            f"Bought {quantity} share(s) at ${price:.2f}. "
            f"Spent: ${total_cost:.2f}. Budget: ${agent.budget:.2f}. Shares: {agent.shares}."
        )

    elif action == "SELL":
        if agent.shares < quantity:
            return f"Cannot SELL {quantity} share(s). Only holding {agent.shares}."
        proceeds = price * quantity
        agent.shares -= quantity
        agent.budget += proceeds
        agent.trades += 1
        return (
            f"Sold {quantity} share(s) at ${price:.2f}. "
            f"Proceeds: ${proceeds:.2f}. Budget: ${agent.budget:.2f}. Shares: {agent.shares}."
        )

    else:
        return (
            f"Holding. Price: ${price:.2f}. "
            f"Shares: {agent.shares} (value: ${agent.shares * price:.2f}). "
            f"Budget: ${agent.budget:.2f}."
        )
