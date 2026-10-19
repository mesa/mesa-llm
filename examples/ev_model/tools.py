from typing import TYPE_CHECKING

from examples.ev_model.agent import HOUSEHOLD_TOOL_MANAGER, AgentState
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    pass


@tool(tool_manager=HOUSEHOLD_TOOL_MANAGER)
def buy_ev(agent):
    """
    Purchase an electric vehicle.

    Args:
        agent: HouseholdAgent making the EV purchase decision.

    Returns:
        str: Result of the EV purchase.
    """

    if agent.state != AgentState.NONE_HOLDER:
        return "Agent already owns a vehicle."

    if agent.utility_ev <= agent.utility_ice:
        return "EV not purchased because ICE utility is higher."

    agent.state = AgentState.EV_HOLDER
    agent.battery_level = agent.battery_capacity

    return "Agent purchased an EV."


@tool(tool_manager=HOUSEHOLD_TOOL_MANAGER)
def buy_ice(agent):
    """
    Purchase an internal combustion engine vehicle.

    Args:
        agent: HouseholdAgent making the ICE purchase decision.

    Returns:
        str: Result of the ICE purchase.
    """

    if agent.state != AgentState.NONE_HOLDER:
        return "Agent already owns a vehicle."

    if agent.utility_ice < agent.utility_ev:
        return "ICE not purchased because EV utility is higher."

    agent.state = AgentState.ICE_HOLDER

    return "Agent purchased an ICE vehicle."


@tool(tool_manager=HOUSEHOLD_TOOL_MANAGER)
def charge_ev(agent):
    """
    Charge EV battery at nearest charging station.

    Args:
        agent: HouseholdAgent owning an EV.

    Returns:
        str: Charging result.
    """

    if agent.state != AgentState.EV_HOLDER:
        return "Charging skipped: agent does not own EV."

    if agent.battery_level > 0.7 * agent.battery_capacity:
        return "Battery already sufficient."

    station = agent.find_nearest_station()

    if station.utilization_rate >= station.capacity:
        return "Charging station full."

    # occupy charger
    station.utilization_rate += 1

    energy = agent.battery_capacity - agent.battery_level
    cost = energy * station.price_per_kwh

    agent.battery_level = agent.battery_capacity

    # release charger
    station.utilization_rate -= 1

    return f"EV charged. Cost {cost}"
