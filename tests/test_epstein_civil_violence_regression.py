import random as process_random
from unittest.mock import AsyncMock, Mock

import pytest
from mesa.model import Model
from mesa.space import MultiGrid

from examples.epstein_civil_violence.actions import arrest_citizen
from examples.epstein_civil_violence.agents import Citizen, CitizenState, Cop
from mesa_llm.reasoning.react import ReActReasoning


class NoOpMemory:
    def process_step(self, *args, **kwargs):
        pass

    async def aprocess_step(self, *args, **kwargs):
        pass

    def add_to_memory(self, *args, **kwargs):
        pass

    async def aadd_to_memory(self, *args, **kwargs):
        pass


class CivilViolenceTestModel(Model):
    def __init__(self, seed=42):
        super().__init__(rng=seed)
        self.grid = MultiGrid(5, 5, torus=False)


def make_citizen(model=None, *, jail_sentence_left=0):
    if model is None:
        model = CivilViolenceTestModel()
    citizen = Citizen(
        model=model,
        reasoning=ReActReasoning,
        llm_model="openai/test",
        system_prompt="",
        vision=1,
        internal_state=[],
        step_prompt="test citizen step",
    )
    model.grid.place_agent(citizen, (2, 2))
    citizen.memory = NoOpMemory()
    citizen.jail_sentence_left = jail_sentence_left
    citizen.update_estimated_arrest_probability = Mock()
    citizen.generate_obs = Mock(return_value="stubbed observation")
    citizen.act = Mock(return_value=None)
    citizen.aact = AsyncMock(return_value=None)
    return citizen


def make_cop_and_citizen(*, seed=42, max_jail_term=5):
    model = CivilViolenceTestModel(seed=seed)
    cop = Cop(
        model=model,
        reasoning=ReActReasoning,
        llm_model="openai/test",
        system_prompt="",
        vision=1,
        internal_state=[],
        step_prompt="test cop step",
        max_jail_term=max_jail_term,
    )
    citizen = make_citizen(model)
    citizen.state = CitizenState.ACTIVE
    cop.memory = NoOpMemory()
    model.grid.place_agent(cop, (1, 1))
    return cop, citizen


def assert_rejected_arrest_has_no_side_effects(cop, citizen_id, expected_error):
    citizen = next(
        (
            model_agent
            for model_agent in cop.model.agents
            if model_agent.unique_id == citizen_id
        ),
        None,
    )
    citizen_fields = (
        None
        if citizen is None
        else {
            "state": getattr(citizen, "state", None),
            "jail_sentence_left": getattr(citizen, "jail_sentence_left", None),
            "pos": citizen.pos,
            "internal_state": list(citizen.internal_state),
        }
    )
    rng_state = cop.random.getstate()
    cop_fields = {
        "pos": cop.pos,
        "max_jail_term": cop.max_jail_term,
        "internal_state": list(cop.internal_state),
    }
    cop.memory.add_to_memory = Mock()
    cop.memory.aadd_to_memory = AsyncMock()

    with pytest.raises(ValueError, match=expected_error):
        arrest_citizen(cop, citizen_id)

    assert cop.random.getstate() == rng_state
    assert {
        "pos": cop.pos,
        "max_jail_term": cop.max_jail_term,
        "internal_state": cop.internal_state,
    } == cop_fields
    if citizen is not None:
        assert {
            "state": getattr(citizen, "state", None),
            "jail_sentence_left": getattr(citizen, "jail_sentence_left", None),
            "pos": citizen.pos,
            "internal_state": citizen.internal_state,
        } == citizen_fields
    cop.memory.add_to_memory.assert_not_called()
    cop.memory.aadd_to_memory.assert_not_awaited()


def test_sync_step_runs_action_when_citizen_is_unjailed():
    citizen = make_citizen(jail_sentence_left=0)

    citizen.step()

    citizen.update_estimated_arrest_probability.assert_called_once_with()
    citizen.generate_obs.assert_called_once_with()
    citizen.act.assert_called_once_with(
        prompt=["test citizen step", "OBSERVATION:\nstubbed observation"],
        actions=["change_state", "move_one_step"],
    )
    assert citizen.jail_sentence_left == 0


def test_sync_step_treats_floating_point_zero_residue_as_released():
    citizen = make_citizen(jail_sentence_left=-2.7755575615628914e-17)

    citizen.step()

    citizen.update_estimated_arrest_probability.assert_called_once_with()
    citizen.act.assert_called_once()
    assert citizen.jail_sentence_left <= 0


def test_sync_jail_countdown_clamps_to_zero_instead_of_drifting_negative():
    citizen = make_citizen(jail_sentence_left=0.3)

    for _ in range(3):
        citizen.step()

    citizen.act.assert_not_called()
    assert citizen.jail_sentence_left == 0


def test_sync_citizen_leaves_jailed_branch_after_expected_countdown():
    citizen = make_citizen(jail_sentence_left=0.3)

    for _ in range(3):
        citizen.step()
    citizen.step()

    citizen.update_estimated_arrest_probability.assert_called_once_with()
    citizen.act.assert_called_once()
    assert citizen.jail_sentence_left == 0


@pytest.mark.asyncio
async def test_async_jail_countdown_clamps_and_leaves_jailed_branch():
    citizen = make_citizen(jail_sentence_left=0.3)

    for _ in range(3):
        await citizen.astep()
    await citizen.astep()

    citizen.update_estimated_arrest_probability.assert_called_once_with()
    citizen.aact.assert_awaited_once()
    assert citizen.jail_sentence_left == 0


def test_arrest_citizen_uses_agent_rng_not_process_global_random(monkeypatch):
    cop, citizen = make_cop_and_citizen()
    agent_randint = Mock(return_value=3)
    monkeypatch.setattr(cop.random, "randint", agent_randint)

    def fail_if_process_global_random_is_used(*args, **kwargs):
        raise AssertionError("arrest_citizen must use agent.random.randint")

    monkeypatch.setattr(
        process_random, "randint", fail_if_process_global_random_is_used
    )

    arrest_citizen(cop, citizen.unique_id)

    agent_randint.assert_called_once_with(1, cop.max_jail_term)
    assert citizen.state is CitizenState.ARRESTED
    assert citizen.jail_sentence_left == 3


def arrest_sentence_for_seed(*, model_seed, process_seed):
    process_random.seed(process_seed)
    cop, citizen = make_cop_and_citizen(seed=model_seed, max_jail_term=10)

    arrest_citizen(cop, citizen.unique_id)

    return citizen.jail_sentence_left


def test_arrest_sentence_reproducible_from_model_seed_not_process_global_seed():
    first_sentence = arrest_sentence_for_seed(model_seed=123, process_seed=1)
    second_sentence = arrest_sentence_for_seed(model_seed=123, process_seed=2)

    assert first_sentence == second_sentence


def test_arrest_rejects_missing_id_without_side_effects():
    cop, _ = make_cop_and_citizen()

    assert_rejected_arrest_has_no_side_effects(
        cop, 999_999, r"Agent 999999 does not exist\."
    )


def test_arrest_rejects_non_citizen_without_side_effects():
    cop, _ = make_cop_and_citizen()
    other_cop = Cop(
        model=cop.model,
        reasoning=ReActReasoning,
        llm_model="openai/test",
        system_prompt="",
        vision=1,
        internal_state=[],
        step_prompt="test cop step",
    )
    other_cop.memory = NoOpMemory()
    cop.model.grid.place_agent(other_cop, (1, 2))

    assert_rejected_arrest_has_no_side_effects(
        cop, other_cop.unique_id, rf"Agent {other_cop.unique_id} is not a citizen\."
    )


@pytest.mark.parametrize(
    ("state", "jail_sentence_left"),
    [
        pytest.param(CitizenState.QUIET, 0, id="quiet"),
        pytest.param(CitizenState.ARRESTED, 2, id="already-arrested"),
    ],
)
def test_arrest_rejects_non_active_citizen_without_side_effects(
    state, jail_sentence_left
):
    cop, citizen = make_cop_and_citizen()
    citizen.state = state
    citizen.jail_sentence_left = jail_sentence_left

    assert_rejected_arrest_has_no_side_effects(
        cop, citizen.unique_id, rf"Citizen {citizen.unique_id} is not active\."
    )


@pytest.mark.parametrize(
    ("target_position", "case"),
    [
        pytest.param((3, 3), "outside vision", id="outside-vision"),
        pytest.param((1, 1), "same cell", id="same-cell"),
    ],
)
def test_arrest_rejects_active_citizen_absent_from_local_state_without_side_effects(
    target_position, case
):
    cop, citizen = make_cop_and_citizen()
    cop.model.grid.move_agent(citizen, target_position)
    _, local_state = cop._build_observation()

    assert f"Citizen {citizen.unique_id}" not in local_state, case
    assert_rejected_arrest_has_no_side_effects(
        cop,
        citizen.unique_id,
        rf"Citizen {citizen.unique_id} is not visible to cop {cop.unique_id}\.",
    )


@pytest.mark.parametrize(
    "target_position",
    [
        pytest.param((1, 2), id="adjacent"),
        pytest.param((2, 2), id="diagonal"),
    ],
)
def test_arrest_visible_active_citizen_changes_only_arrest_fields_reproducibly(
    target_position,
):
    results = []
    for _ in range(2):
        cop, citizen = make_cop_and_citizen(seed=817, max_jail_term=10)
        cop.model.grid.move_agent(citizen, target_position)
        _, local_state = cop._build_observation()
        assert f"Citizen {citizen.unique_id}" in local_state
        cop.memory.add_to_memory = Mock()
        before = {
            "pos": citizen.pos,
            "internal_state": list(citizen.internal_state),
            "hardship": citizen.hardship,
            "risk_aversion": citizen.risk_aversion,
            "grievance": citizen.grievance,
        }

        result = arrest_citizen(cop, citizen.unique_id)

        assert result == f"agent {citizen.unique_id} arrested by {cop.unique_id}."
        assert citizen.state is CitizenState.ARRESTED
        assert 1 <= citizen.jail_sentence_left <= cop.max_jail_term
        assert {
            "pos": citizen.pos,
            "internal_state": citizen.internal_state,
            "hardship": citizen.hardship,
            "risk_aversion": citizen.risk_aversion,
            "grievance": citizen.grievance,
        } == before
        cop.memory.add_to_memory.assert_not_called()
        results.append(citizen.jail_sentence_left)

    assert results[0] == results[1]
