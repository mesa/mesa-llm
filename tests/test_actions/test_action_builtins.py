from __future__ import annotations

import logging
import math
from random import Random
from types import SimpleNamespace

import numpy as np
import pytest
from mesa.discrete_space import OrthogonalMooreGrid, OrthogonalVonNeumannGrid
from mesa.space import ContinuousSpace, MultiGrid, SingleGrid

from mesa_llm.actions import (
    ActionChoice,
    ActionManager,
    default_actions,
    move_one_step,
    social_actions,
    spatial_actions,
    speak_to,
    teleport_to_location,
    wait,
)


class DummyModel:
    def __init__(self):
        self.grid = None
        self.space = None
        self.agents = []


class DummyAgent:
    def __init__(self, unique_id: int, model: DummyModel):
        self.unique_id = unique_id
        self.model = model
        self.pos = None


class _CellAwareDummyAgent(DummyAgent):
    def __init__(self, unique_id: int, model: DummyModel):
        super().__init__(unique_id, model)
        self._cell = None

    @property
    def cell(self):
        return self._cell

    @cell.setter
    def cell(self, cell):
        if self._cell is cell:
            return
        if self._cell is not None:
            self._cell.remove_agent(self)
        self._cell = cell
        if cell is not None:
            cell.add_agent(self)


def _execute(agent, name: str, arguments: dict, actions):
    manager = ActionManager(actions=actions)
    return manager.execute(agent, ActionChoice(name=name, arguments=arguments))


def _execute_spatial(agent, name: str, arguments: dict):
    return _execute(agent, name, arguments, spatial_actions())


def _execute_social(agent, arguments: dict):
    return _execute(agent, "speak_to", arguments, social_actions())


def _validate_spatial(agent, name: str, arguments: dict):
    manager = ActionManager(actions=spatial_actions())
    return manager.validate(agent, ActionChoice(name=name, arguments=arguments))


def _discrete_grid_agent(grid_type, start=(1, 1), unique_id=101):
    model = DummyModel()
    model.grid = grid_type(width=4, height=4, torus=False)

    agent = DummyAgent(unique_id=unique_id, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, start)

    return model, agent


def _assert_discrete_grid_agent_unchanged(model, agent, position):
    assert agent.pos == position
    assert agent in model.grid.get_cell_list_contents([position])


def _orthogonal_grid_agent(
    grid_type,
    start=(0, 0),
    unique_id=112,
    *,
    dimensions=(4, 4),
    torus=False,
):
    model = DummyModel()
    model.grid = grid_type(
        dimensions=dimensions,
        torus=torus,
        random=Random(0),
    )

    agent = _CellAwareDummyAgent(unique_id=unique_id, model=model)
    model.agents.append(agent)
    agent.cell = model.grid[start]

    return model, agent


def _assert_orthogonal_grid_agent_unchanged(model, agent, start_cell):
    assert agent.cell is start_cell
    assert agent in start_cell.agents
    assert sum(agent in cell.agents for cell in model.grid.all_cells.cells) == 1


def _assert_orthogonal_grid_agent_moved(model, agent, start_cell, target_cell):
    assert agent.cell is target_cell
    assert agent not in start_cell.agents
    assert agent in target_cell.agents
    assert sum(agent in cell.agents for cell in model.grid.all_cells.cells) == 1


def _continuous_space_agent(torus, start=(1.25, 2.5), unique_id=113):
    model = DummyModel()
    model.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=torus)

    agent = DummyAgent(unique_id=unique_id, model=model)
    model.agents.append(agent)
    model.space.place_agent(agent, start)
    model.space.get_neighbors(start, radius=0)

    return model, agent


def _assert_continuous_space_agent_unchanged(model, agent, start):
    assert agent.pos == start
    assert agent in model.space._agent_to_index
    assert agent in model.space.get_neighbors(start, radius=0)


def test_builtin_action_factories_are_explicit_immutable_tuples():
    assert default_actions() == (wait,)
    assert spatial_actions() == (move_one_step, teleport_to_location)
    assert social_actions() == (speak_to,)

    assert isinstance(default_actions(), tuple)
    assert isinstance(spatial_actions(), tuple)
    assert isinstance(social_actions(), tuple)


def test_migrated_action_schemas_omit_agent_and_keep_required_arguments():
    manager = ActionManager(actions=spatial_actions() + social_actions())
    schemas = {schema["name"]: schema for schema in manager.get_actions_schema()}

    assert set(schemas) == {
        "move_one_step",
        "teleport_to_location",
        "speak_to",
    }
    assert "agent" not in schemas["move_one_step"]["parameters"]["properties"]
    assert schemas["move_one_step"]["parameters"]["required"] == ["direction"]
    assert (
        schemas["move_one_step"]["parameters"]["properties"]["direction"]["type"]
        == "string"
    )

    teleport_properties = schemas["teleport_to_location"]["parameters"]["properties"]
    assert "agent" not in teleport_properties
    assert schemas["teleport_to_location"]["parameters"]["required"] == [
        "target_coordinates",
    ]
    assert teleport_properties["target_coordinates"]["type"] == "array"

    speak_properties = schemas["speak_to"]["parameters"]["properties"]
    assert "agent" not in speak_properties
    assert schemas["speak_to"]["parameters"]["required"] == [
        "listener_agents_unique_ids",
        "message",
    ]
    assert speak_properties["listener_agents_unique_ids"]["items"]["type"] == "integer"
    assert speak_properties["message"]["type"] == "string"


def test_speak_to_schema_requires_integer_ids_instead_of_agent_labels():
    manager = ActionManager(actions=[speak_to])
    schema = manager.get_actions_schema()[0]
    description = schema["parameters"]["properties"]["listener_agents_unique_ids"][
        "description"
    ]

    assert "[1, 2]" in description
    assert "integer IDs only" in description
    assert '["BuyerAgent 1"]' in description
    assert "never agent labels" in description


def test_move_one_step_on_singlegrid():
    model = DummyModel()
    model.grid = SingleGrid(width=5, height=5, torus=False)

    agent = DummyAgent(unique_id=1, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (2, 2))

    result = _execute_spatial(
        agent,
        "move_one_step",
        {"direction": "North"},
    )

    assert agent.pos == (2, 3)
    assert result == "agent 1 moved to (2, 3)."


def test_teleport_to_location_on_multigrid():
    model = DummyModel()
    model.grid = MultiGrid(width=4, height=4, torus=False)

    agent = DummyAgent(unique_id=7, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (0, 0))

    out = _execute_spatial(
        agent,
        "teleport_to_location",
        {"target_coordinates": [3, 2]},
    )

    assert agent.pos == (3, 2)
    assert out == "agent 7 moved to (3, 2)."


@pytest.mark.parametrize("grid_type", [SingleGrid, MultiGrid])
def test_teleport_to_location_discrete_grids_accept_valid_integer_coordinates(
    grid_type,
):
    model, agent = _discrete_grid_agent(grid_type, unique_id=102)

    out = _execute_spatial(
        agent,
        "teleport_to_location",
        {"target_coordinates": [3, 2]},
    )

    assert agent.pos == (3, 2)
    assert agent in model.grid.get_cell_list_contents([(3, 2)])
    assert agent not in model.grid.get_cell_list_contents([(1, 1)])
    assert out == "agent 102 moved to (3, 2)."


@pytest.mark.parametrize("grid_type", [SingleGrid, MultiGrid])
def test_teleport_to_location_discrete_grids_accept_numpy_integer_coordinates(
    grid_type,
):
    model, agent = _discrete_grid_agent(
        grid_type,
        start=(np.int64(1), np.int64(1)),
        unique_id=110,
    )

    out = teleport_to_location(agent, [np.int64(2), np.int64(3)])

    assert agent.pos == (2, 3)
    assert all(type(coordinate) is int for coordinate in agent.pos)
    assert agent in model.grid.get_cell_list_contents([(2, 3)])
    assert agent not in model.grid.get_cell_list_contents([(1, 1)])
    assert out == "agent 110 moved to (2, 3)."


@pytest.mark.parametrize("grid_type", [SingleGrid, MultiGrid])
def test_move_one_step_discrete_grids_accept_numpy_integer_positions(grid_type):
    model, agent = _discrete_grid_agent(
        grid_type,
        start=(np.int64(1), np.int64(1)),
        unique_id=111,
    )

    out = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    assert agent.pos == (1, 2)
    assert all(type(coordinate) is int for coordinate in agent.pos)
    assert agent in model.grid.get_cell_list_contents([(1, 2)])
    assert agent not in model.grid.get_cell_list_contents([(1, 1)])
    assert out == "agent 111 moved to (1, 2)."


def test_teleport_to_location_multigrid_rejects_fractional_coordinate_before_mutation():
    model, agent = _discrete_grid_agent(MultiGrid, unique_id=103)

    with pytest.raises(ValueError):
        teleport_to_location(agent, [2.5, 2])

    _assert_discrete_grid_agent_unchanged(model, agent, (1, 1))


def test_teleport_to_location_singlegrid_rejects_fractional_coordinate_before_mutation():
    model, agent = _discrete_grid_agent(SingleGrid, unique_id=104)

    with pytest.raises(ValueError):
        teleport_to_location(agent, [2, 2.5])

    _assert_discrete_grid_agent_unchanged(model, agent, (1, 1))


@pytest.mark.parametrize("grid_type", [SingleGrid, MultiGrid])
@pytest.mark.parametrize("target_coordinates", [[True, 2], [2, False]])
def test_teleport_to_location_discrete_grids_reject_boolean_coordinates(
    grid_type,
    target_coordinates,
):
    model, agent = _discrete_grid_agent(grid_type, unique_id=105)

    with pytest.raises(ValueError):
        teleport_to_location(agent, target_coordinates)

    _assert_discrete_grid_agent_unchanged(model, agent, (1, 1))


@pytest.mark.parametrize("grid_type", [SingleGrid, MultiGrid])
@pytest.mark.parametrize(
    "target_coordinates",
    [
        [math.nan, 2],
        [2, math.nan],
        [math.inf, 2],
        [-math.inf, 2],
    ],
)
def test_teleport_to_location_discrete_grids_reject_non_finite_coordinates(
    grid_type,
    target_coordinates,
):
    model, agent = _discrete_grid_agent(grid_type, unique_id=106)

    with pytest.raises(ValueError):
        teleport_to_location(agent, target_coordinates)

    _assert_discrete_grid_agent_unchanged(model, agent, (1, 1))


@pytest.mark.parametrize("grid_type", [SingleGrid, MultiGrid])
def test_teleport_to_location_discrete_grids_large_integer_out_of_bounds_no_mutation(
    grid_type,
):
    model, agent = _discrete_grid_agent(grid_type, unique_id=109)
    huge_coordinate = 10**400

    with pytest.raises(ValueError, match="out of bounds"):
        teleport_to_location(agent, [huge_coordinate, 2])

    _assert_discrete_grid_agent_unchanged(model, agent, (1, 1))


@pytest.mark.parametrize("grid_type", [SingleGrid, MultiGrid])
def test_teleport_to_location_discrete_grids_handle_int_like_float_coordinates_safely(
    grid_type,
):
    model, agent = _discrete_grid_agent(grid_type, unique_id=107)

    out = teleport_to_location(agent, [2.0, 3.0])

    assert agent.pos == (2, 3)
    assert all(type(coordinate) is int for coordinate in agent.pos)
    assert agent in model.grid.get_cell_list_contents([(2, 3)])
    assert agent not in model.grid.get_cell_list_contents([(1, 1)])
    assert out == "agent 107 moved to (2, 3)."


@pytest.mark.parametrize(
    "grid_type",
    [OrthogonalMooreGrid, OrthogonalVonNeumannGrid],
)
@pytest.mark.parametrize(
    "target_coordinates",
    [
        [True, 1],
        [1, False],
        [1.5, 1],
        [1, 1.5],
        [math.nan, 1],
        [1, math.inf],
        [-math.inf, 1],
    ],
    ids=[
        "boolean-row",
        "boolean-column",
        "fractional-row",
        "fractional-column",
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_teleport_to_location_orthogonal_grids_reject_invalid_coordinates_before_mutation(
    grid_type,
    target_coordinates,
):
    model, agent = _orthogonal_grid_agent(grid_type)
    start_cell = agent.cell

    with pytest.raises(ValueError):
        teleport_to_location(agent, target_coordinates)

    _assert_orthogonal_grid_agent_unchanged(model, agent, start_cell)


@pytest.mark.parametrize(
    "grid_type",
    [OrthogonalMooreGrid, OrthogonalVonNeumannGrid],
)
@pytest.mark.parametrize(
    ("target_coordinates", "expected_coordinates"),
    [
        ([2, 3], (2, 3)),
        ([np.int64(2), np.int32(3)], (2, 3)),
        ([2.0, 3.0], (2, 3)),
    ],
    ids=["python-integers", "numpy-integers", "integer-valued-floats"],
)
def test_teleport_to_location_orthogonal_grids_normalize_valid_coordinates(
    grid_type,
    target_coordinates,
    expected_coordinates,
):
    model, agent = _orthogonal_grid_agent(grid_type, unique_id=114)
    start_cell = agent.cell
    target_cell = model.grid._cells[expected_coordinates]

    out = teleport_to_location(agent, target_coordinates)

    assert agent.cell is target_cell
    assert agent in target_cell.agents
    assert agent not in start_cell.agents
    assert out == f"agent 114 moved to {expected_coordinates}."


def test_teleport_to_location_on_orthogonal_grid_without_constructor():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (3, 3)
    target = (1, 1)
    dummy_cell = SimpleNamespace(coordinate=target, agents=[], is_full=False)
    orth_grid._cells = {target: dummy_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=9, model=model)
    model.agents.append(agent)

    out = _execute_spatial(
        agent,
        "teleport_to_location",
        {"target_coordinates": [1, 1]},
    )

    assert getattr(agent, "cell", None) is dummy_cell
    assert out == "agent 9 moved to (1, 1)."


def test_move_one_step_on_orthogonal_grid_without_constructor():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (5, 5)
    start_target = (1, 1)
    end_target = (0, 1)
    start_cell = SimpleNamespace(
        coordinate=start_target, agents=[], connections={}, is_full=False
    )
    end_cell = SimpleNamespace(
        coordinate=end_target, agents=[], connections={}, is_full=False
    )
    start_cell.connections[(-1, 0)] = end_cell
    orth_grid._cells = {start_target: start_cell, end_target: end_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=10, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    out = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    assert getattr(agent, "cell", None) is end_cell
    assert out == "agent 10 moved to (0, 1)."


def test_move_one_step_east_on_orthogonal_grid_without_constructor():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (5, 5)
    start_target = (1, 1)
    end_target = (1, 2)
    start_cell = SimpleNamespace(
        coordinate=start_target, agents=[], connections={}, is_full=False
    )
    end_cell = SimpleNamespace(
        coordinate=end_target, agents=[], connections={}, is_full=False
    )
    start_cell.connections[(0, 1)] = end_cell
    orth_grid._cells = {start_target: start_cell, end_target: end_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=11, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    out = _execute_spatial(agent, "move_one_step", {"direction": "East"})

    assert getattr(agent, "cell", None) is end_cell
    assert out == "agent 11 moved to (1, 2)."


def test_speak_to_records_on_recipients(mocker):
    model = DummyModel()

    sender = DummyAgent(unique_id=10, model=model)
    r1 = DummyAgent(unique_id=11, model=model)
    r2 = DummyAgent(unique_id=12, model=model)

    r1.memory = SimpleNamespace(add_to_memory=mocker.Mock())
    r2.memory = SimpleNamespace(add_to_memory=mocker.Mock())

    model.agents = [sender, r1, r2]

    message = "Hello there"
    ret = _execute_social(
        sender,
        {
            "listener_agents_unique_ids": [10, 11, 12],
            "message": message,
        },
    )

    r1.memory.add_to_memory.assert_called_once()
    r2.memory.add_to_memory.assert_called_once()

    _, kwargs = r1.memory.add_to_memory.call_args
    assert kwargs["type"] == "message"
    content = kwargs["content"]
    assert content["message"] == message
    assert content["sender"] == sender.unique_id
    assert "recipients" not in content
    assert ret == "sent message 'Hello there' to [11, 12]"


def test_speak_to_rejects_single_free_text_id_before_execution(mocker):
    model = DummyModel()

    sender = DummyAgent(unique_id=1, model=model)
    recipients = [DummyAgent(unique_id=2, model=model)]

    for recipient in recipients:
        recipient.memory = SimpleNamespace(add_to_memory=mocker.Mock())

    model.agents = [sender, *recipients]

    with pytest.raises(
        ValueError,
        match=r"Invalid argument type.*listener_agents_unique_ids.*list\[int\]",
    ):
        _execute_social(
            sender,
            {
                "listener_agents_unique_ids": "Agent 2",
                "message": "ping",
            },
        )

    recipients[0].memory.add_to_memory.assert_not_called()


def test_teleport_to_location_coerces_string_json_coordinates_before_execution():
    model = DummyModel()
    model.grid = MultiGrid(width=4, height=4, torus=False)

    agent = DummyAgent(unique_id=43, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (0, 0))

    out = _execute_spatial(
        agent,
        "teleport_to_location",
        {"target_coordinates": "[3, 2]"},
    )

    assert agent.pos == (3, 2)
    assert out == "agent 43 moved to (3, 2)."


def test_move_one_step_invalid_direction_fails_before_mutation():
    model = DummyModel()
    model.grid = MultiGrid(width=4, height=4, torus=False)

    agent = DummyAgent(unique_id=3, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (2, 2))

    with pytest.raises(ValueError, match="Invalid direction"):
        _execute_spatial(agent, "move_one_step", {"direction": "north east"})

    assert agent.pos == (2, 2)


def test_move_one_step_unsupported_environment():
    model = DummyModel()
    agent = DummyAgent(unique_id=4, model=model)
    model.agents.append(agent)
    agent.pos = (1, 1)

    with pytest.raises(ValueError, match="Unsupported environment"):
        _execute_spatial(agent, "move_one_step", {"direction": "North"})

    assert agent.pos == (1, 1)


def test_move_one_step_unsupported_non_none_environment():
    class _UnsupportedGrid:
        pass

    class _UnsupportedSpace:
        pass

    model = DummyModel()
    model.grid = _UnsupportedGrid()
    model.space = _UnsupportedSpace()

    agent = DummyAgent(unique_id=32, model=model)
    model.agents.append(agent)
    agent.pos = (1, 1)

    with pytest.raises(ValueError, match="Unsupported environment"):
        _execute_spatial(agent, "move_one_step", {"direction": "North"})

    assert agent.pos == (1, 1)


def test_teleport_to_location_unsupported_environment():
    model = DummyModel()
    agent = DummyAgent(unique_id=8, model=model)
    model.agents.append(agent)
    agent.pos = (1, 1)

    with pytest.raises(ValueError, match="Unsupported environment"):
        _execute_spatial(
            agent,
            "teleport_to_location",
            {"target_coordinates": [2, 2]},
        )

    assert agent.pos == (1, 1)


def test_teleport_to_location_unsupported_non_none_environment():
    class _UnsupportedGrid:
        pass

    class _UnsupportedSpace:
        pass

    model = DummyModel()
    model.grid = _UnsupportedGrid()
    model.space = _UnsupportedSpace()

    agent = DummyAgent(unique_id=33, model=model)
    model.agents.append(agent)
    agent.pos = (1, 1)

    with pytest.raises(ValueError, match="Unsupported environment"):
        _execute_spatial(
            agent,
            "teleport_to_location",
            {"target_coordinates": [2, 2]},
        )

    assert agent.pos == (1, 1)


def test_teleport_to_location_on_continuousspace():
    model = DummyModel()
    model.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=False)

    agent = DummyAgent(unique_id=5, model=model)
    model.agents.append(agent)
    model.space.place_agent(agent, (1.0, 1.0))

    out = _execute_spatial(
        agent,
        "teleport_to_location",
        {"target_coordinates": [5.5, 7.25]},
    )

    assert agent.pos == (5.5, 7.25)
    assert out == "agent 5 moved to (5.5, 7.25)."


def test_teleport_to_location_on_continuousspace_without_grid_attribute():
    model = SimpleNamespace(
        space=ContinuousSpace(x_max=10.0, y_max=10.0, torus=False),
        agents=[],
    )

    agent = DummyAgent(unique_id=39, model=model)
    model.agents.append(agent)
    model.space.place_agent(agent, (1.0, 1.0))

    out = _execute_spatial(
        agent,
        "teleport_to_location",
        {"target_coordinates": [4.5, 6.5]},
    )

    assert agent.pos == (4.5, 6.5)
    assert out == "agent 39 moved to (4.5, 6.5)."


def test_teleport_to_location_continuousspace_accepts_finite_float_coordinates():
    model = DummyModel()
    model.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=False)

    agent = DummyAgent(unique_id=108, model=model)
    model.agents.append(agent)
    model.space.place_agent(agent, (1.0, 1.0))

    out = _execute_spatial(
        agent,
        "teleport_to_location",
        {"target_coordinates": [2.5, 3.75]},
    )

    assert agent.pos == (2.5, 3.75)
    assert out == "agent 108 moved to (2.5, 3.75)."


@pytest.mark.parametrize("managed", [False, True], ids=["direct", "managed"])
@pytest.mark.parametrize("torus", [False, True], ids=["non-torus", "torus"])
@pytest.mark.parametrize(
    "target_coordinates",
    [
        (math.nan, 2.0),
        (2.0, math.nan),
        (math.inf, 2.0),
        (-math.inf, 2.0),
    ],
    ids=[
        "nan-x",
        "nan-y",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_teleport_to_location_continuousspace_rejects_non_finite_coordinates_before_mutation(
    managed,
    torus,
    target_coordinates,
):
    start = (1.25, 2.5)
    model, agent = _continuous_space_agent(torus, start=start)

    with pytest.raises(ValueError):
        if managed:
            _execute_spatial(
                agent,
                "teleport_to_location",
                {"target_coordinates": target_coordinates},
            )
        else:
            teleport_to_location(agent, target_coordinates)

    _assert_continuous_space_agent_unchanged(model, agent, start)


@pytest.mark.parametrize("managed", [False, True], ids=["direct", "managed"])
@pytest.mark.parametrize("torus", [False, True], ids=["non-torus", "torus"])
@pytest.mark.parametrize(
    "target_coordinates",
    [
        (True, 3.0),
        (2.0, False),
    ],
    ids=["boolean-x", "boolean-y"],
)
def test_teleport_to_location_continuousspace_rejects_boolean_coordinates_before_mutation(
    managed,
    torus,
    target_coordinates,
):
    start = (1.25, 2.5)
    model, agent = _continuous_space_agent(torus, start=start)

    with pytest.raises(ValueError):
        if managed:
            _execute_spatial(
                agent,
                "teleport_to_location",
                {"target_coordinates": target_coordinates},
            )
        else:
            teleport_to_location(agent, target_coordinates)

    _assert_continuous_space_agent_unchanged(model, agent, start)


@pytest.mark.parametrize("torus", [False, True], ids=["non-torus", "torus"])
def test_teleport_to_location_continuousspace_rejects_non_numeric_coordinates_before_mutation(
    torus,
):
    start = (1.25, 2.5)
    model, agent = _continuous_space_agent(torus, start=start)

    with pytest.raises(ValueError):
        teleport_to_location(agent, ("not-a-number", 2.0))

    _assert_continuous_space_agent_unchanged(model, agent, start)


@pytest.mark.parametrize(
    "target_coordinates",
    [
        (2, 3),
        (2.5, 3.75),
        (np.float32(2.5), np.float32(3.75)),
        (np.float64(2.5), np.float64(3.75)),
    ],
    ids=[
        "python-integers",
        "python-floats",
        "numpy-float32",
        "numpy-float64",
    ],
)
def test_teleport_to_location_continuousspace_accepts_finite_numeric_coordinates(
    target_coordinates,
):
    model, agent = _continuous_space_agent(False, unique_id=115)

    out = teleport_to_location(agent, target_coordinates)

    assert agent.pos == target_coordinates
    assert agent in model.space._agent_to_index
    assert agent in model.space.get_neighbors(target_coordinates, radius=0)
    assert out.startswith("agent 115 moved to ")


@pytest.mark.parametrize("managed", [False, True], ids=["direct", "managed"])
def test_teleport_to_location_continuousspace_wraps_valid_finite_coordinates(
    managed,
):
    model, agent = _continuous_space_agent(True, unique_id=116)
    target_coordinates = (12.5, -1.25)

    if managed:
        out = _execute_spatial(
            agent,
            "teleport_to_location",
            {"target_coordinates": target_coordinates},
        )
    else:
        out = teleport_to_location(agent, target_coordinates)

    expected_coordinates = (2.5, 8.75)
    assert agent.pos == expected_coordinates
    assert agent in model.space._agent_to_index
    assert agent in model.space.get_neighbors(expected_coordinates, radius=0)
    assert out == f"agent 116 moved to {expected_coordinates}."


def test_teleport_to_location_singlegrid_occupied_target_raises_before_mutation():
    model = DummyModel()
    model.grid = SingleGrid(width=4, height=4, torus=False)

    moving_agent = DummyAgent(unique_id=34, model=model)
    blocking_agent = DummyAgent(unique_id=35, model=model)
    model.agents.extend([moving_agent, blocking_agent])
    model.grid.place_agent(moving_agent, (1, 1))
    model.grid.place_agent(blocking_agent, (1, 2))

    with pytest.raises(ValueError, match="occupied"):
        _execute_spatial(
            moving_agent,
            "teleport_to_location",
            {"target_coordinates": [1, 2]},
        )

    assert moving_agent.pos == (1, 1)
    assert blocking_agent.pos == (1, 2)


def test_teleport_to_location_singlegrid_out_of_bounds_raises_before_mutation():
    model = DummyModel()
    model.grid = SingleGrid(width=4, height=4, torus=False)

    agent = DummyAgent(unique_id=36, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (1, 1))

    with pytest.raises(ValueError, match="out of bounds"):
        _execute_spatial(
            agent,
            "teleport_to_location",
            {"target_coordinates": [-1, 1]},
        )

    assert agent.pos == (1, 1)


def test_teleport_to_location_orthogonal_missing_cell_raises_before_mutation():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (3, 3)
    start = (1, 1)
    start_cell = SimpleNamespace(coordinate=start, agents=[], is_full=False)
    orth_grid._cells = {start: start_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=37, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    with pytest.raises(ValueError, match="out of bounds"):
        _execute_spatial(
            agent,
            "teleport_to_location",
            {"target_coordinates": [0, 1]},
        )

    assert agent.cell is start_cell


def test_teleport_to_location_orthogonal_full_cell_raises_before_mutation():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (3, 3)
    start = (1, 1)
    target = (0, 1)
    start_cell = SimpleNamespace(coordinate=start, agents=[], is_full=False)
    full_cell = SimpleNamespace(
        coordinate=target,
        agents=[SimpleNamespace(unique_id=99)],
        is_full=True,
    )
    orth_grid._cells = {start: start_cell, target: full_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=40, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    with pytest.raises(ValueError, match="full"):
        _execute_spatial(
            agent,
            "teleport_to_location",
            {"target_coordinates": [0, 1]},
        )

    assert agent.cell is start_cell


@pytest.mark.parametrize(
    "bad_coordinates",
    [
        ["x", "y"],
        [1],
        [1, 2, 3],
    ],
)
def test_teleport_invalid_coordinate_args_fail_validation_before_mutation(
    bad_coordinates,
):
    model = DummyModel()
    model.grid = SingleGrid(width=4, height=4, torus=False)

    agent = DummyAgent(unique_id=41, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (1, 1))

    with pytest.raises(ValueError, match="Invalid argument type"):
        _validate_spatial(
            agent,
            "teleport_to_location",
            {"target_coordinates": bad_coordinates},
        )

    assert agent.pos == (1, 1)


def test_teleport_valid_coordinate_list_is_coerced_to_tuple():
    model = DummyModel()
    agent = DummyAgent(unique_id=42, model=model)

    validated = _validate_spatial(
        agent,
        "teleport_to_location",
        {"target_coordinates": [1, "2.5"]},
    )

    assert validated.arguments == {"target_coordinates": (1, 2.5)}


def test_move_one_step_on_continuousspace():
    model = DummyModel()
    model.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=False)

    agent = DummyAgent(unique_id=6, model=model)
    model.agents.append(agent)
    model.space.place_agent(agent, (2.0, 2.0))

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    assert agent.pos == (2.0, 3.0)
    assert result == "agent 6 moved to (2.0, 3.0)."


def test_move_one_step_boundary_on_continuousspace():
    model = DummyModel()
    model.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=False)

    agent = DummyAgent(unique_id=30, model=model)
    model.agents.append(agent)
    model.space.place_agent(agent, (2.0, 9.0))

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    assert agent.pos == (2.0, 9.0)
    assert "boundary" in result.lower()
    assert "North" in result


def test_move_one_step_torus_wrap_on_continuousspace():
    model = DummyModel()
    model.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=True)

    agent = DummyAgent(unique_id=31, model=model)
    model.agents.append(agent)
    model.space.place_agent(agent, (2.0, 9.0))

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    assert agent.pos == (2.0, 0.0)
    assert result == "agent 31 moved to (2.0, 0.0)."


def test_move_one_step_boundary_singlegrid_north():
    model = DummyModel()
    model.grid = SingleGrid(width=5, height=5, torus=False)

    agent = DummyAgent(unique_id=20, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (2, 4))

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    assert agent.pos == (2, 4)
    assert "boundary" in result.lower()
    assert "North" in result


def test_move_one_step_torus_wrap_singlegrid_north():
    model = DummyModel()
    model.grid = SingleGrid(width=5, height=5, torus=True)

    agent = DummyAgent(unique_id=23, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (2, 4))

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    assert agent.pos == (2, 0)
    assert result == "agent 23 moved to (2, 0)."


def test_move_one_step_boundary_multigrid_west():
    model = DummyModel()
    model.grid = MultiGrid(width=5, height=5, torus=False)

    agent = DummyAgent(unique_id=21, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (0, 2))

    result = _execute_spatial(agent, "move_one_step", {"direction": "West"})

    assert agent.pos == (0, 2)
    assert "boundary" in result.lower()
    assert "West" in result


def test_move_one_step_torus_wrap_multigrid_west():
    model = DummyModel()
    model.grid = MultiGrid(width=5, height=5, torus=True)

    agent = DummyAgent(unique_id=24, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (0, 2))

    result = _execute_spatial(agent, "move_one_step", {"direction": "West"})

    assert agent.pos == (4, 2)
    assert result == "agent 24 moved to (4, 2)."


def test_move_one_step_singlegrid_occupied_target():
    model = DummyModel()
    model.grid = SingleGrid(width=5, height=5, torus=False)

    moving_agent = DummyAgent(unique_id=25, model=model)
    blocking_agent = DummyAgent(unique_id=26, model=model)
    model.agents.extend([moving_agent, blocking_agent])
    model.grid.place_agent(moving_agent, (2, 2))
    model.grid.place_agent(blocking_agent, (2, 3))

    result = _execute_spatial(moving_agent, "move_one_step", {"direction": "North"})

    assert moving_agent.pos == (2, 2)
    assert blocking_agent.pos == (2, 3)
    assert "occupied" in result.lower()
    assert "North" in result


@pytest.mark.parametrize(
    "grid_type",
    [OrthogonalMooreGrid, OrthogonalVonNeumannGrid],
)
def test_move_one_step_boundary_on_real_orthogonal_grid(grid_type):
    model, agent = _orthogonal_grid_agent(
        grid_type,
        start=(0, 1),
        unique_id=22,
    )
    start_cell = agent.cell

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    _assert_orthogonal_grid_agent_unchanged(model, agent, start_cell)
    assert "unavailable" in result.lower()
    assert "no connection" in result.lower()
    assert "grid topology" in result.lower()
    assert "North" in result


def test_move_one_step_removed_torus_connection_is_not_a_boundary():
    model, agent = _orthogonal_grid_agent(
        OrthogonalMooreGrid,
        start=(0, 0),
        unique_id=38,
        dimensions=(3, 3),
        torus=True,
    )
    start_cell = agent.cell
    wrapped_cell = model.grid[(2, 0)]
    model.grid.remove_connection(start_cell, wrapped_cell)

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    _assert_orthogonal_grid_agent_unchanged(model, agent, start_cell)
    assert agent not in wrapped_cell.agents
    assert "unavailable" in result.lower()
    assert "no connection" in result.lower()
    assert "grid topology" in result.lower()
    assert "boundary" not in result.lower()
    assert "North" in result


def test_move_one_step_full_target_orthogonal_grid():
    model, agent = _orthogonal_grid_agent(
        OrthogonalMooreGrid,
        start=(1, 1),
        unique_id=27,
    )
    start_cell = agent.cell
    full_target_cell = model.grid[(0, 1)]
    full_target_cell.capacity = 1
    blocking_agent = _CellAwareDummyAgent(unique_id=99, model=model)
    model.agents.append(blocking_agent)
    blocking_agent.cell = full_target_cell

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    _assert_orthogonal_grid_agent_unchanged(model, agent, start_cell)
    assert blocking_agent.cell is full_target_cell
    assert blocking_agent in full_target_cell.agents
    assert "full" in result.lower()
    assert "North" in result


def test_move_one_step_diagonal_unavailable_on_real_vonneumann_grid():
    model, agent = _orthogonal_grid_agent(
        OrthogonalVonNeumannGrid,
        start=(2, 2),
        unique_id=28,
    )
    start_cell = agent.cell

    result = _execute_spatial(agent, "move_one_step", {"direction": "NorthEast"})

    _assert_orthogonal_grid_agent_unchanged(model, agent, start_cell)
    assert "unavailable" in result.lower()
    assert "no connection" in result.lower()
    assert "grid topology" in result.lower()
    assert "boundary" not in result.lower()
    assert "NorthEast" in result


def test_move_one_step_diagonal_available_on_real_moore_grid():
    model, agent = _orthogonal_grid_agent(
        OrthogonalMooreGrid,
        start=(2, 2),
        unique_id=29,
    )
    start_cell = agent.cell
    target_cell = model.grid[(1, 3)]

    result = _execute_spatial(agent, "move_one_step", {"direction": "NorthEast"})

    _assert_orthogonal_grid_agent_moved(
        model,
        agent,
        start_cell,
        target_cell,
    )
    assert result == "agent 29 moved to (1, 3)."


@pytest.mark.parametrize(
    "grid_type",
    [OrthogonalMooreGrid, OrthogonalVonNeumannGrid],
)
def test_move_one_step_cardinal_direction_succeeds_on_real_orthogonal_grids(
    grid_type,
):
    model, agent = _orthogonal_grid_agent(
        grid_type,
        start=(2, 2),
        unique_id=30,
    )
    start_cell = agent.cell
    target_cell = model.grid[(1, 2)]

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    _assert_orthogonal_grid_agent_moved(
        model,
        agent,
        start_cell,
        target_cell,
    )
    assert result == "agent 30 moved to (1, 2)."


def test_move_one_step_removed_connection_prevents_movement():
    model, agent = _orthogonal_grid_agent(
        OrthogonalMooreGrid,
        start=(2, 2),
        unique_id=31,
    )
    start_cell = agent.cell
    disconnected_cell = model.grid[(1, 2)]
    model.grid.remove_connection(start_cell, disconnected_cell)

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    _assert_orthogonal_grid_agent_unchanged(model, agent, start_cell)
    assert agent not in disconnected_cell.agents
    assert "unavailable" in result.lower()
    assert "no connection" in result.lower()
    assert "grid topology" in result.lower()
    assert "boundary" not in result.lower()
    assert "North" in result


def test_move_one_step_follows_custom_connection_target():
    model, agent = _orthogonal_grid_agent(
        OrthogonalMooreGrid,
        start=(2, 2),
        unique_id=32,
    )
    start_cell = agent.cell
    coordinate_derived_cell = model.grid[(1, 2)]
    connected_cell = model.grid[(3, 3)]
    start_cell.connections[(-1, 0)] = connected_cell

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    _assert_orthogonal_grid_agent_moved(
        model,
        agent,
        start_cell,
        connected_cell,
    )
    assert agent not in coordinate_derived_cell.agents
    assert result == "agent 32 moved to (3, 3)."


def test_move_one_step_boundary_connection_remains_authoritative():
    model, agent = _orthogonal_grid_agent(
        OrthogonalMooreGrid,
        start=(0, 1),
        unique_id=34,
    )
    start_cell = agent.cell
    connected_cell = model.grid[(3, 3)]
    start_cell.connections[(-1, 0)] = connected_cell

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    _assert_orthogonal_grid_agent_moved(
        model,
        agent,
        start_cell,
        connected_cell,
    )
    assert result == "agent 34 moved to (3, 3)."


def test_move_one_step_torus_wraps_through_real_grid_connection():
    model, agent = _orthogonal_grid_agent(
        OrthogonalMooreGrid,
        start=(0, 0),
        unique_id=33,
        dimensions=(3, 3),
        torus=True,
    )
    start_cell = agent.cell
    wrapped_cell = model.grid[(2, 2)]

    result = _execute_spatial(agent, "move_one_step", {"direction": "NorthWest"})

    _assert_orthogonal_grid_agent_moved(
        model,
        agent,
        start_cell,
        wrapped_cell,
    )
    assert result == "agent 33 moved to (2, 2)."


def test_speak_to_skips_non_llm_recipient(mocker):
    model = DummyModel()

    sender = DummyAgent(unique_id=1, model=model)
    llm_recipient = DummyAgent(unique_id=2, model=model)
    rule_recipient = DummyAgent(unique_id=3, model=model)

    llm_recipient.memory = SimpleNamespace(add_to_memory=mocker.Mock())

    model.agents = [sender, llm_recipient, rule_recipient]

    ret = _execute_social(
        sender,
        {
            "listener_agents_unique_ids": [2, 3],
            "message": "Hello both",
        },
    )

    llm_recipient.memory.add_to_memory.assert_called_once()
    call_kwargs = llm_recipient.memory.add_to_memory.call_args[1]
    assert call_kwargs["type"] == "message"
    assert call_kwargs["content"]["message"] == "Hello both"
    assert "recipients" not in call_kwargs["content"]

    assert ret == (
        "sent message 'Hello both' to [2]; "
        "skipped [3] because they have no `memory` attribute"
    )


def test_speak_to_warns_for_non_llm_recipient(mocker, caplog):
    model = DummyModel()
    sender = DummyAgent(unique_id=10, model=model)
    rule_recipient = DummyAgent(unique_id=11, model=model)

    model.agents = [sender, rule_recipient]

    with caplog.at_level(logging.WARNING, logger="mesa_llm.actions.builtins"):
        ret = _execute_social(
            sender,
            {
                "listener_agents_unique_ids": [11],
                "message": "Test message",
            },
        )

    assert any(
        "11" in record.message and "memory" in record.message
        for record in caplog.records
    )
    assert ret == "skipped [11] because they have no `memory` attribute"


def test_speak_to_returns_clear_message_when_no_valid_recipients():
    model = DummyModel()
    sender = DummyAgent(unique_id=20, model=model)

    model.agents = [sender]

    ret = _execute_social(
        sender,
        {
            "listener_agents_unique_ids": [20, 999],
            "message": "Anyone there?",
        },
    )

    assert (
        ret == "Could not send message 'Anyone there?': no matching recipients found."
    )


def test_migrated_actions_reject_missing_extra_and_narrowed_out_inputs():
    model = DummyModel()
    model.grid = MultiGrid(width=4, height=4, torus=False)
    agent = DummyAgent(unique_id=50, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (1, 1))
    manager = ActionManager(actions=spatial_actions() + social_actions())

    with pytest.raises(ValueError, match="Missing required argument"):
        manager.execute(agent, ActionChoice(name="move_one_step", arguments={}))
    with pytest.raises(ValueError, match="Unexpected argument"):
        manager.execute(
            agent,
            ActionChoice(
                name="speak_to",
                arguments={
                    "listener_agents_unique_ids": [51],
                    "message": "hello",
                    "volume": "loud",
                },
            ),
        )
    with pytest.raises(ValueError, match="Unknown action name"):
        manager.execute(
            agent,
            ActionChoice(name="speak_to", arguments={}),
            actions=spatial_actions(),
        )

    assert agent.pos == (1, 1)
