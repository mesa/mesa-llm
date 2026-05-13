from __future__ import annotations

import logging
from types import SimpleNamespace

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


def test_speak_to_coerces_single_free_text_id_before_execution(mocker):
    model = DummyModel()

    sender = DummyAgent(unique_id=1, model=model)
    recipients = [DummyAgent(unique_id=2, model=model)]

    for recipient in recipients:
        recipient.memory = SimpleNamespace(add_to_memory=mocker.Mock())

    model.agents = [sender, *recipients]

    ret = _execute_social(
        sender,
        {
            "listener_agents_unique_ids": "Agent 2",
            "message": "ping",
        },
    )

    recipients[0].memory.add_to_memory.assert_called_once()
    assert ret == "sent message 'ping' to [2]"


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


def test_move_one_step_boundary_orthogonal_grid():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (5, 5)
    start = (0, 1)
    start_cell = SimpleNamespace(
        coordinate=start, agents=[], connections={}, is_full=False
    )
    orth_grid._cells = {start: start_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=22, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    assert agent.cell is start_cell
    assert "boundary" in result.lower()
    assert "North" in result


def test_move_one_step_boundary_orthogonal_torus_missing_wrapped_cell():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = True
    orth_grid.dimensions = (3, 3)
    start = (0, 0)
    start_cell = SimpleNamespace(coordinate=start, agents=[], is_full=False)
    orth_grid._cells = {start: start_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=38, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    assert agent.cell is start_cell
    assert "boundary" in result.lower()
    assert "North" in result


def test_move_one_step_full_target_orthogonal_grid():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (5, 5)
    start = (1, 1)
    end = (0, 1)
    start_cell = SimpleNamespace(
        coordinate=start, agents=[], connections={}, is_full=False
    )
    full_target_cell = SimpleNamespace(
        coordinate=end,
        agents=[SimpleNamespace(unique_id=99)],
        connections={},
        is_full=True,
    )
    start_cell.connections[(-1, 0)] = full_target_cell
    orth_grid._cells = {start: start_cell, end: full_target_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=27, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    result = _execute_spatial(agent, "move_one_step", {"direction": "North"})

    assert agent.cell is start_cell
    assert "full" in result.lower()
    assert "North" in result


def test_move_one_step_diagonal_on_orthogonal_vonneumann_grid():
    class _DummyOrthogonalVonNeumannGrid(OrthogonalVonNeumannGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalVonNeumannGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (5, 5)
    start = (2, 2)
    end = (1, 3)
    start_cell = SimpleNamespace(coordinate=start, agents=[], is_full=False)
    end_cell = SimpleNamespace(coordinate=end, agents=[], is_full=False)
    orth_grid._cells = {start: start_cell, end: end_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=28, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    result = _execute_spatial(agent, "move_one_step", {"direction": "NorthEast"})

    assert agent.cell is end_cell
    assert result == "agent 28 moved to (1, 3)."


def test_move_one_step_torus_wrap_orthogonal_grid():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = True
    orth_grid.dimensions = (3, 3)
    start = (0, 0)
    end = (2, 2)
    start_cell = SimpleNamespace(coordinate=start, agents=[], is_full=False)
    wrapped_cell = SimpleNamespace(coordinate=end, agents=[], is_full=False)
    orth_grid._cells = {start: start_cell, end: wrapped_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=29, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    result = _execute_spatial(agent, "move_one_step", {"direction": "NorthWest"})

    assert agent.cell is wrapped_cell
    assert result == "agent 29 moved to (2, 2)."


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
