from datetime import timedelta

from examples.agriculture_model.agent import FARMER_TOOL_MANAGER, CropState
from mesa_llm.tools.tool_decorator import tool


@tool(tool_manager=FARMER_TOOL_MANAGER)
def plant_crop(agent, days_to_harvest: int = 60):
    """
    Plant a crop in the field.

    Args:
        days_to_harvest: Number of days required for harvest.

    Returns:
        A status message.
    """
    try:
        days_to_harvest = int(days_to_harvest)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"days_to_harvest must be an integer, got {days_to_harvest!r}"
        ) from exc
    if days_to_harvest <= 0:
        raise ValueError("days_to_harvest must be greater than 0")
    if agent.crop_state != CropState.IDLE:
        return f"Cannot plant while crop is {agent.crop_state.value}"

    agent.crop_state = CropState.PLANTED
    agent.plant_date = agent.model.current_day
    agent.harvest_date = agent.model.current_day + timedelta(days=days_to_harvest)
    agent.fertilizer = 0.0
    return "Crop planted"


@tool(tool_manager=FARMER_TOOL_MANAGER)
def apply_fertilizer(agent, level: float = 0.5):
    """
    Apply fertilizer to the crop.

    Args:
        level: Fertilizer intensity from 0 to 1.

    Returns:
        A status message.
    """
    try:
        level = float(level)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"level must be numeric, got {level!r}") from exc
    if level < 0:
        raise ValueError("level must be non-negative")
    if agent.crop_state in [CropState.IDLE, CropState.READY]:
        return "Cannot apply fertilizer at this stage"

    agent.fertilizer += level
    if agent.crop_state == CropState.PLANTED:
        agent.crop_state = CropState.GROWING
    return f"Fertilizer applied: {level}"


@tool(tool_manager=FARMER_TOOL_MANAGER)
def harvest_crop(agent):
    """
    Harvest the crop if ready.

    Args:
        agent: Provided automatically.

    Returns:
        A harvest status message.
    """
    if (
        agent.harvest_date is not None
        and agent.crop_state in {CropState.PLANTED, CropState.GROWING}
        and agent.model.current_day >= agent.harvest_date
    ):
        agent.crop_state = CropState.READY

    if agent.crop_state != CropState.READY:
        return "Not ready"

    agent.compute_yield()
    agent.crop_state = CropState.IDLE
    agent.plant_date = None
    agent.harvest_date = None
    agent.fertilizer = 0.0
    return "Harvested"
