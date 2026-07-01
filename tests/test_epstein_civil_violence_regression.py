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
