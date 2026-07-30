import json
import random as process_random
from unittest.mock import AsyncMock, Mock

import pytest
from mesa.model import Model
from mesa.space import MultiGrid

from examples.epstein_civil_violence.actions import arrest_citizen
from examples.epstein_civil_violence.agents import Citizen, CitizenState, Cop
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.reasoning.reasoning import Observation

_ARREST_OR_MOVE_STEP_PROMPT = (
    "Use arrest_citizen for an eligible active citizen; otherwise use move_one_step."
)


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


def add_citizen(cop, *, state, position=(2, 2)):
    citizen = make_citizen(cop.model)
    cop.model.grid.move_agent(citizen, position)
    citizen.state = state
    return citizen


def make_cop_observation(cop, visible_citizens, *, internal_states=None):
    internal_states = internal_states or {}
    local_state = {}
    for citizen in visible_citizens:
        local_state[f"Citizen {citizen.unique_id}"] = {
            "position": citizen.pos,
            "internal_state": internal_states.get(
                citizen.unique_id, list(citizen.internal_state)
            ),
        }
    return Observation(
        step=cop.model.steps,
        self_state={
            "agent_unique_id": cop.unique_id,
            "location": cop.pos,
            "internal_state": list(cop.internal_state),
        },
        local_state=local_state,
    )


def configure_cop_workflow(cop, observation):
    cop.generate_obs = Mock(return_value=observation)
    cop.act = Mock(return_value=None)
    cop.aact = AsyncMock(return_value=None)


def action_call(cop, *, asynchronous=False):
    action_mock = cop.aact if asynchronous else cop.act
    return action_mock.await_args if asynchronous else action_mock.call_args


def eligible_ids_from_prompt(prompt):
    prompt_text = "\n".join(prompt)
    eligible_contexts = []
    for index, character in enumerate(prompt_text):
        if character != "{":
            continue
        try:
            context, _ = json.JSONDecoder().raw_decode(prompt_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(context, dict) and "eligible_active_citizen_ids" in context:
            eligible_contexts.append(context)

    assert len(eligible_contexts) == 1, prompt_text
    eligible_context = eligible_contexts[0]
    assert set(eligible_context) == {"eligible_active_citizen_ids"}
    eligible_ids = eligible_context["eligible_active_citizen_ids"]
    assert all(type(citizen_id) is int for citizen_id in eligible_ids)
    return eligible_ids


def prompt_text_from_call(call):
    return "\n".join(call.kwargs["prompt"])


def assert_movement_only_prompt(call, *, observation):
    prompt_text = prompt_text_from_call(call)
    assert f"OBSERVATION:\n{observation}" in prompt_text
    assert json.dumps({"eligible_active_citizen_ids": []}) in prompt_text
    assert eligible_ids_from_prompt(call.kwargs["prompt"]) == []
    assert "arrest_citizen" not in prompt_text
    assert _ARREST_OR_MOVE_STEP_PROMPT not in prompt_text
    assert "Move to a nearby cell." in prompt_text


def assert_eligible_arrest_prompt(call, *, observation, eligible_ids):
    prompt_text = prompt_text_from_call(call)
    assert f"OBSERVATION:\n{observation}" in prompt_text
    assert json.dumps({"eligible_active_citizen_ids": eligible_ids}) in prompt_text
    assert eligible_ids_from_prompt(call.kwargs["prompt"]) == eligible_ids
    assert "arrest_citizen" in prompt_text
    assert _ARREST_OR_MOVE_STEP_PROMPT in prompt_text


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


@pytest.mark.parametrize(
    ("state", "jail_sentence_left"),
    [
        pytest.param(CitizenState.QUIET, 0, id="quiet"),
        pytest.param(CitizenState.ARRESTED, 2, id="arrested"),
    ],
)
def test_cop_excludes_visible_non_active_citizens_from_arrest_selection(
    state, jail_sentence_left
):
    cop, citizen = make_cop_and_citizen()
    citizen.state = state
    citizen.jail_sentence_left = jail_sentence_left
    observation = make_cop_observation(cop, [citizen])
    configure_cop_workflow(cop, observation)

    result = cop.step()

    assert result is None
    cop.generate_obs.assert_called_once_with()
    cop.act.assert_called_once()
    call = action_call(cop)
    assert call.kwargs["actions"] == ["move_one_step"]
    assert eligible_ids_from_prompt(call.kwargs["prompt"]) == []


def test_cop_excludes_active_citizen_absent_from_observation():
    cop, citizen = make_cop_and_citizen()
    cop.step_prompt = _ARREST_OR_MOVE_STEP_PROMPT
    observation = make_cop_observation(cop, [])
    configure_cop_workflow(cop, observation)

    result = cop.step()

    assert result is None
    cop.generate_obs.assert_called_once_with()
    call = action_call(cop)
    assert call.kwargs["actions"] == ["move_one_step"]
    assert_movement_only_prompt(call, observation=observation)
    assert citizen.state is CitizenState.ACTIVE


def test_cop_includes_visible_active_citizen_in_arrest_selection():
    cop, citizen = make_cop_and_citizen()
    cop.step_prompt = _ARREST_OR_MOVE_STEP_PROMPT
    observation = make_cop_observation(cop, [citizen])
    configure_cop_workflow(cop, observation)

    result = cop.step()

    assert result is None
    cop.generate_obs.assert_called_once_with()
    call = action_call(cop)
    assert call.kwargs["actions"] == ["move_one_step", "arrest_citizen"]
    assert_eligible_arrest_prompt(
        call,
        observation=observation,
        eligible_ids=[citizen.unique_id],
    )


def test_cop_prompt_lists_only_sorted_visible_active_citizen_ids():
    cop, first_active = make_cop_and_citizen()
    quiet = add_citizen(cop, state=CitizenState.QUIET)
    second_active = add_citizen(cop, state=CitizenState.ACTIVE)
    arrested = add_citizen(cop, state=CitizenState.ARRESTED)
    observation = make_cop_observation(
        cop,
        [arrested, second_active, quiet, first_active],
    )
    configure_cop_workflow(cop, observation)

    cop.step()

    call = action_call(cop)
    assert call.kwargs["actions"] == ["move_one_step", "arrest_citizen"]
    assert eligible_ids_from_prompt(call.kwargs["prompt"]) == sorted(
        [first_active.unique_id, second_active.unique_id]
    )


def test_cop_uses_current_state_instead_of_stale_observation_internal_state():
    cop, citizen = make_cop_and_citizen()
    citizen.state = CitizenState.QUIET
    observation = make_cop_observation(
        cop,
        [citizen],
        internal_states={
            citizen.unique_id: [
                f"my current state in the simulation is {CitizenState.ACTIVE}"
            ]
        },
    )
    configure_cop_workflow(cop, observation)

    cop.step()

    call = action_call(cop)
    assert call.kwargs["actions"] == ["move_one_step"]
    assert eligible_ids_from_prompt(call.kwargs["prompt"]) == []


def test_citizen_becoming_quiet_before_cop_step_removes_arrest_action():
    cop, citizen = make_cop_and_citizen()
    observation = make_cop_observation(
        cop,
        [citizen],
        internal_states={
            citizen.unique_id: [
                f"my current state in the simulation is {CitizenState.ACTIVE}"
            ]
        },
    )
    citizen.state = CitizenState.QUIET
    configure_cop_workflow(cop, observation)

    cop.step()

    call = action_call(cop)
    assert call.kwargs["actions"] == ["move_one_step"]
    assert eligible_ids_from_prompt(call.kwargs["prompt"]) == []
    assert citizen.state is CitizenState.QUIET
    assert citizen.jail_sentence_left == 0


def test_cop_eligibility_check_does_not_mutate_state_or_memory():
    cop, active = make_cop_and_citizen()
    quiet = add_citizen(cop, state=CitizenState.QUIET)
    observation = make_cop_observation(cop, [active, quiet])
    configure_cop_workflow(cop, observation)
    cop.memory.add_to_memory = Mock()
    cop.memory.aadd_to_memory = AsyncMock()
    state_before = {
        agent.unique_id: {
            "pos": agent.pos,
            "internal_state": list(agent.internal_state),
            "state": getattr(agent, "state", None),
            "jail_sentence_left": getattr(agent, "jail_sentence_left", None),
        }
        for agent in cop.model.agents
    }

    result = cop.step()

    state_after = {
        agent.unique_id: {
            "pos": agent.pos,
            "internal_state": list(agent.internal_state),
            "state": getattr(agent, "state", None),
            "jail_sentence_left": getattr(agent, "jail_sentence_left", None),
        }
        for agent in cop.model.agents
    }
    assert result is None
    assert state_after == state_before
    cop.memory.add_to_memory.assert_not_called()
    cop.memory.aadd_to_memory.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "expected_actions", "expected_ids"),
    [
        pytest.param(
            CitizenState.QUIET,
            ["move_one_step"],
            [],
            id="no-eligible-citizen",
        ),
        pytest.param(
            CitizenState.ACTIVE,
            ["move_one_step", "arrest_citizen"],
            "visible-citizen",
            id="eligible-citizen",
        ),
    ],
)
async def test_async_cop_narrows_actions_from_current_eligibility(
    state, expected_actions, expected_ids
):
    cop, citizen = make_cop_and_citizen()
    cop.step_prompt = _ARREST_OR_MOVE_STEP_PROMPT
    citizen.state = state
    observation = make_cop_observation(cop, [citizen])
    configure_cop_workflow(cop, observation)

    result = await cop.astep()

    assert result is None
    cop.generate_obs.assert_called_once_with()
    cop.aact.assert_awaited_once()
    call = action_call(cop, asynchronous=True)
    assert call.kwargs["actions"] == expected_actions
    if expected_ids == "visible-citizen":
        expected_ids = [citizen.unique_id]
        assert_eligible_arrest_prompt(
            call,
            observation=observation,
            eligible_ids=expected_ids,
        )
    else:
        assert_movement_only_prompt(call, observation=observation)
    cop.act.assert_not_called()


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
