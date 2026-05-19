import math
from enum import Enum

import mesa

from mesa_llm.actions import move_one_step
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory


class CitizenState(Enum):
    QUIET = 1
    ACTIVE = 2
    ARRESTED = 3


class Citizen(LLMAgent, mesa.discrete_space.CellAgent):
    """
    A member of the general population, may or may not be in active rebellion.
    Summary of rule: If grievance - risk > threshold, rebel.

    Attributes:
        hardship: Agent's 'perceived hardship (i.e., physical or economic
            privation).' Exogenous, drawn from U(0,1).
        regime_legitimacy: Agent's perception of regime legitimacy, equal
            across agents.  Exogenous.
        risk_aversion: Exogenous, drawn from U(0,1).
        threshold: if (grievance - (risk_aversion * arrest_probability)) >
            threshold, go/remain Active
        vision: number of cells in each direction (N, S, E and W) that agent
            can inspect
        condition: Can be "Quiescent" or "Active;" deterministic function of
            greivance, perceived risk, and
        grievance: deterministic function of hardship and regime_legitimacy;
            how aggrieved is agent at the regime?
        arrest_probability: agent's assessment of arrest probability, given
            rebellion
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        step_prompt,
        arrest_prob_constant=0.5,
        regime_legitimacy=0.5,
        threshold=0.5,
        api_base=None,
    ):
        # Call the superclass constructor with updated internal state
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
            api_base=api_base,
            actions=[move_one_step, "change_state"],
        )

        self.hardship = self.random.random()
        self.risk_aversion = self.random.random()
        self.regime_legitimacy = regime_legitimacy
        self.state = CitizenState.QUIET
        self.vision = vision
        self.jail_sentence_left = 0  # A jail sentence of 1 implies that the agent cannot participate in the next 10 steps.
        self.grievance = self.hardship * (1 - self.regime_legitimacy)
        self.arrest_prob_constant = arrest_prob_constant
        self.arrest_probability = None

        self.memory = STLTMemory(
            agent=self,
            display=True,
            llm_model=llm_model,
            api_base=api_base,
        )

        self.threshold = threshold
        self.internal_state.append(
            f"tendency for risk aversion is {self.risk_aversion:.4f} on scale from 0 to 1"
        )
        self.internal_state.append(
            f"On a scale from 0 to 1, my threshold for suffering is {self.threshold:.4f}"
        )
        self.internal_state.append(
            f"On a scale of 0 to 1 my grievance due to current legitimacy of rule and personal hardships is {self.grievance:.4f}"
        )
        self.internal_state.append(
            f"tendency for risk aversion is {self.risk_aversion:.4f} on scale from 0 to 1"
        )
        self.internal_state.append(
            f"my current state in the simulation is {self.state}"
        )
        self.system_prompt = "You are a citizen in a country that is experiencing civil violence. You are a member of the general population, may or may not be in active rebellion. In general, more your suffering more the tendency for you to become active. You can move one step in a nearby cell or change your state."

    def update_estimated_arrest_probability(self):
        """
        Based on the ratio of cops to actives in my neighborhood, estimate the
        p(Arrest | I go active).
        """
        cops_in_vision = 0
        actives_in_vision = 1  # citizen counts herself

        neighbors = self.model.grid.get_neighbors(
            tuple(self.pos), moore=True, include_center=False, radius=self.vision
        )
        for i in neighbors:
            if isinstance(i, Cop):
                cops_in_vision += 1
            elif i.state == CitizenState.ACTIVE:
                actives_in_vision += 1
        # there is a body of literature on this equation
        # the round is not in the pnas paper but without it, its impossible to replicate
        # the dynamics shown there.
        self.arrest_probability = 1 - math.exp(
            -1 * self.arrest_prob_constant * round(cops_in_vision / actives_in_vision)
        )
        for item in self.internal_state:
            if item.lower().startswith("my arrest probability is"):
                self.internal_state.remove(item)
                break
        self.internal_state.append(
            f"my arrest probability is {self.arrest_probability:.4f}"
        )

    def step(self):
        if self.jail_sentence_left == 0:
            self.update_estimated_arrest_probability()
            observation = self.generate_obs()
            self.act(
                prompt=[self.step_prompt, f"OBSERVATION:\n{observation}"],
                actions=["change_state", "move_one_step"],
            )
        else:
            self.jail_sentence_left -= 0.1

    async def astep(self):
        if self.jail_sentence_left == 0:
            self.update_estimated_arrest_probability()
            observation = self.generate_obs()
            await self.aact(
                prompt=[self.step_prompt, f"OBSERVATION:\n{observation}"],
                actions=["change_state", "move_one_step"],
            )
        else:
            self.jail_sentence_left -= 0.1


class Cop(LLMAgent, mesa.discrete_space.CellAgent):
    """
    A cop for life.  No defection.
    Summary of rule: Inspect local vision and arrest a random active agent.

    Attributes:
        unique_id: unique int
        x, y: Grid coordinates
        vision: number of cells in each direction (N, S, E and W) that cop is
            able to inspect
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        step_prompt,
        max_jail_term=2,
        api_base=None,
    ):
        """
        Create a new Cop.
        Args:
            x, y: Grid coordinates
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance
        """
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
            api_base=api_base,
            actions=[move_one_step, "arrest_citizen"],
        )
        self.max_jail_term = max_jail_term
        self.system_prompt = "You are a cop in a country that is experiencing civil violence. You are a member of the police force and your job is to arrest active citizens. You can arrest a citizen ONLY if they are active. You can move one step in a nearby cell or arrest a citizen."

        self.memory = STLTMemory(
            agent=self,
            display=True,
            llm_model=llm_model,
            api_base=api_base,
        )

    def step(self):
        """
        Inspect local vision and arrest a random active agent. Move if
        applicable.
        """
        observation = self.generate_obs()
        self.act(
            prompt=[self.step_prompt, f"OBSERVATION:\n{observation}"],
            actions=["move_one_step", "arrest_citizen"],
        )

    async def astep(self):
        """
        Inspect local vision and arrest a random active agent. Move if
        applicable.
        """
        observation = self.generate_obs()
        await self.aact(
            prompt=[self.step_prompt, f"OBSERVATION:\n{observation}"],
            actions=["move_one_step", "arrest_citizen"],
        )
