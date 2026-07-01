from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from typing import Any

from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.space import (
    ContinuousSpace,
    MultiGrid,
    SingleGrid,
)

from mesa_llm.actions.action_decorator import action

logger = logging.getLogger(__name__)

# Mapping directions to (dx, dy) for Cartesian-style spaces.
direction_map_xy = {
    "North": (0, 1),
    "South": (0, -1),
    "East": (1, 0),
    "West": (-1, 0),
    "NorthEast": (1, 1),
    "NorthWest": (-1, 1),
    "SouthEast": (1, -1),
    "SouthWest": (-1, -1),
}


# Mapping directions to (drow, dcol) for mesa.discrete_space orthogonal grids.
direction_map_row_col = {
    "North": (-1, 0),
    "South": (1, 0),
    "East": (0, 1),
    "West": (0, -1),
    "NorthEast": (-1, 1),
    "NorthWest": (-1, -1),
    "SouthEast": (1, 1),
    "SouthWest": (1, -1),
}


def _get_agent_position(agent: Any) -> Any:
    """Return the agent position across Mesa space APIs."""
    cell = getattr(agent, "cell", None)
    if cell is not None and getattr(cell, "coordinate", None) is not None:
        return cell.coordinate

    pos = getattr(agent, "pos", None)
    if pos is not None:
        return pos

    position = getattr(agent, "position", None)
    if position is not None:
        return position

    raise ValueError(
        "Could not infer agent position from `cell`, `pos`, or `position`."
    )


def _normalize_discrete_grid_coordinates(
    target_coordinates: tuple[Any, ...],
) -> tuple[int, ...]:
    """Normalize coordinates that Mesa discrete grids can index safely."""
    normalized_coordinates = []
    for coordinate in target_coordinates:
        if isinstance(coordinate, bool):
            raise ValueError(
                "Discrete grid coordinates must be finite integers; "
                f"got {target_coordinates}."
            )

        if isinstance(coordinate, int):
            normalized_coordinates.append(coordinate)
            continue

        if not (
            isinstance(coordinate, float)
            and math.isfinite(coordinate)
            and coordinate.is_integer()
        ):
            raise ValueError(
                "Discrete grid coordinates must be finite integers; "
                f"got {target_coordinates}."
            )

        normalized_coordinates.append(int(coordinate))

    return tuple(normalized_coordinates)


@action
def wait(agent: Any) -> str:
    """Take no action for this turn.

    Returns:
        A confirmation that the agent waited.
    """
    return "waited"


@action
def move_one_step(agent: Any, direction: str) -> str:
    """
    Moves agents one step in specified cardinal/diagonal directions.

    Automatically handles Mesa grid types including SingleGrid, MultiGrid,
    OrthogonalGrids, and ContinuousSpace.

    Args:
        agent: Provided automatically.
        direction: The direction to move in. Must be one of:
            'North', 'South', 'East', 'West', 'NorthEast', 'NorthWest',
            'SouthEast', or 'SouthWest'.

    Returns:
        A string confirming the result of the movement attempt.
    """
    if direction not in direction_map_xy:
        raise ValueError(
            f"Invalid direction '{direction}'."
            f"Must be one of {list(direction_map_xy.keys())}"
        )

    grid = getattr(agent.model, "grid", None)
    if isinstance(grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid):
        row, col = _get_agent_position(agent)
        drow, dcol = direction_map_row_col[direction]
        new_pos = (row + drow, col + dcol)

        if grid.torus:
            dimensions = grid.dimensions
            if len(dimensions) == len(new_pos):
                new_pos = tuple(coord % dim for coord, dim in zip(new_pos, dimensions))
        elif new_pos not in grid._cells:
            return (
                f"Agent {agent.unique_id} is at the boundary and cannot move "
                f"{direction}. Try a different direction."
            )

        target_cell = grid._cells.get(new_pos)
        if target_cell is None:
            return (
                f"Agent {agent.unique_id} is at the boundary and cannot move "
                f"{direction}. Try a different direction."
            )

        if target_cell.is_full:
            return (
                f"Agent {agent.unique_id} cannot move {direction} because "
                "the target cell is full."
            )

        target_coordinates = tuple(target_cell.coordinate)
        return teleport_to_location(agent, target_coordinates)

    space = getattr(agent.model, "space", None)
    grid_or_space = None
    if isinstance(grid, SingleGrid | MultiGrid):
        grid_or_space = grid
    elif isinstance(space, ContinuousSpace):
        grid_or_space = space

    if grid_or_space is not None:
        dx, dy = direction_map_xy[direction]
        x, y = _get_agent_position(agent)
        new_pos = (x + dx, y + dy)

        if grid_or_space.torus:
            new_pos = grid_or_space.torus_adj(new_pos)
        elif grid_or_space.out_of_bounds(new_pos):
            return (
                f"Agent {agent.unique_id} is at the boundary and cannot move "
                f"{direction}. Try a different direction."
            )

        if isinstance(grid_or_space, SingleGrid) and not grid_or_space.is_cell_empty(
            new_pos
        ):
            return (
                f"Agent {agent.unique_id} cannot move {direction} because "
                "the target cell is occupied."
            )

        target_coordinates = tuple(new_pos)
        return teleport_to_location(agent, target_coordinates)

    raise ValueError(
        "Unsupported environment for move_one_step. Expected SingleGrid, "
        "MultiGrid, OrthogonalMooreGrid, OrthogonalVonNeumannGrid, or "
        "ContinuousSpace."
    )


@action
def teleport_to_location(
    agent: Any,
    target_coordinates: tuple[int | float, int | float],
) -> str:
    """
    Instantly moves agents to specific [x, y] coordinates.

    Args:
        agent: Provided automatically.
        target_coordinates: Exactly two numeric coordinates in the form [x, y]
            that fall inside the current environment bounds.

    Returns:
        A string confirming the agent's new position.
    """
    target_coordinates = tuple(target_coordinates)
    grid = getattr(agent.model, "grid", None)
    space = getattr(agent.model, "space", None)

    if isinstance(grid, SingleGrid | MultiGrid):
        target_coordinates = _normalize_discrete_grid_coordinates(target_coordinates)
        if grid.torus:
            target_coordinates = grid.torus_adj(target_coordinates)
        elif grid.out_of_bounds(target_coordinates):
            raise ValueError(
                f"Target coordinates {target_coordinates} are out of bounds."
            )

        current_position = getattr(agent, "pos", None)
        target_is_current_position = (
            current_position is not None
            and tuple(current_position) == target_coordinates
        )
        if (
            isinstance(grid, SingleGrid)
            and not target_is_current_position
            and not grid.is_cell_empty(target_coordinates)
        ):
            raise ValueError(f"Target coordinates {target_coordinates} are occupied.")

        grid.move_agent(agent, target_coordinates)

    elif isinstance(grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid):
        target_cell = grid._cells.get(target_coordinates)
        if target_cell is None:
            raise ValueError(
                f"Target coordinates {target_coordinates} are out of bounds."
            )

        current_cell = getattr(agent, "cell", None)
        if target_cell is not current_cell and target_cell.is_full:
            raise ValueError(f"Target coordinates {target_coordinates} are full.")

        agent.cell = target_cell

    elif isinstance(space, ContinuousSpace):
        if space.torus:
            target_coordinates = space.torus_adj(target_coordinates)
        elif space.out_of_bounds(target_coordinates):
            raise ValueError(
                f"Target coordinates {target_coordinates} are out of bounds."
            )

        space.move_agent(agent, target_coordinates)

    else:
        raise ValueError(
            "Unsupported environment for teleport_to_location. Expected "
            "SingleGrid, MultiGrid, OrthogonalMooreGrid, "
            "OrthogonalVonNeumannGrid, or ContinuousSpace."
        )

    return f"agent {agent.unique_id} moved to {target_coordinates}."


@action
def speak_to(agent: Any, listener_agents_unique_ids: list[int], message: str) -> str:
    """
    Send a message to specified recipients.

    Messages are automatically added to recipients' memory systems for future
    reasoning context.

    Args:
        agent: Provided automatically.
        listener_agents_unique_ids: The unique ids of the agents receiving the
            message.
        message: The message to send.

    Returns:
        A string describing delivery status.
    """
    if isinstance(listener_agents_unique_ids, str):
        try:
            listener_agents_unique_ids = json.loads(listener_agents_unique_ids)
        except (json.JSONDecodeError, ValueError):
            listener_agents_unique_ids = [
                int(x.strip())
                for x in listener_agents_unique_ids.strip("[]").split(",")
                if x.strip()
            ]
    listener_agents_unique_ids = [
        int(uid) for uid in (listener_agents_unique_ids or [])
    ]

    listener_agents = [
        listener_agent
        for listener_agent in agent.model.agents
        if listener_agent.unique_id in listener_agents_unique_ids
        and listener_agent.unique_id != agent.unique_id
    ]

    delivered_ids = []
    skipped_ids = []

    for recipient in listener_agents:
        if not hasattr(recipient, "memory"):
            skipped_ids.append(recipient.unique_id)
            logger.warning(
                "Agent %s has no memory attribute; skipping speak_to.",
                recipient.unique_id,
            )
            continue
        delivered_ids.append(recipient.unique_id)
        recipient.memory.add_to_memory(
            type="message",
            content={
                "message": message,
                "sender": agent.unique_id,
            },
        )

    status_parts = []
    if delivered_ids:
        status_parts.append(f"sent message {message!r} to {delivered_ids}")
    if skipped_ids:
        status_parts.append(
            f"skipped {skipped_ids} because they have no `memory` attribute"
        )

    if not status_parts:
        return f"Could not send message {message!r}: no matching recipients found."

    return "; ".join(status_parts)


def default_actions() -> tuple[Callable, ...]:
    """Return the recommended default actions."""
    return (wait,)


def spatial_actions() -> tuple[Callable, ...]:
    """Return opt-in spatial movement actions."""
    return (move_one_step, teleport_to_location)


def social_actions() -> tuple[Callable, ...]:
    """Return opt-in social communication actions."""
    return (speak_to,)


__all__ = [
    "default_actions",
    "move_one_step",
    "social_actions",
    "spatial_actions",
    "speak_to",
    "teleport_to_location",
    "wait",
]
