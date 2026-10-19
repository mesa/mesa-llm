import math
from enum import Enum

import mesa

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager

Citizen_tool_manager = ToolManager()


class CitizenState(Enum):
    REST = 1
    MIGRATE = 2


class Citizen(LLMAgent, mesa.discrete_space.CellAgent):
    """
    A citizen living in a conflict-affected region who may remain or migrate.

    Summary of rule:
    If migration_probability exceeds a stochastic threshold, migrate.

    Attributes:
        risk_proneness: Agent’s sensitivity to perceived conflict risk.
            Exogenous, drawn from U(0,1).

        previous_perceived_risk: Risk accumulated from previous steps.

        migration_prob: Probability of migrating at current step.
            Deterministic function of perceived risk.

        state: Can be "REST" or "MIGRATE"; determined by
            migration_prob and stochastic draw.

        perceived_risk: Deterministic function of:
            - event impact
            - individual risk_proneness
            - memory_retention

        event_impact: Function of conflict intensity adjusted by
            spatial and temporal decay parameters.

        vision: Number of cells in each direction agent can inspect
            for local observation (used for LLM reasoning context).

    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        step_prompt,
        vision,
        internal_state=None,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            step_prompt=step_prompt,
            internal_state=internal_state,
        )

        self.state = CitizenState.REST
        self.household_id = self.random.randint(0, int(self.model.num_households) - 1)
        self.risk_proneness = self.random.random()
        self.memory_retention = self.random.random()
        self.distance = self.random.uniform(0, 1)
        self.time_difference = self.random.uniform(0, 1)
        self.previous_perceived_risk = 0.0
        self.migration_prob = None

        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/llama3.1:latest",
            display=True,
        )

        # Internal state context
        self.internal_state.append(f"My household ID is {self.household_id}")
        self.internal_state.append(f"My risk proneness is {self.risk_proneness:.3f}")
        self.internal_state.append(
            f"my current state in the simulation is {self.state}"
        )
        self.internal_state.append(
            f"my current risk_roneness in the simulation is {self.risk_proneness}"
        )

        self.tool_manager = Citizen_tool_manager

    def compute_event_impact(self):
        return self.model.intensity_of_event / (
            (1 + self.model.spatial_decay * self.distance)
            * (1 + self.model.temporal_decay * self.time_difference)
        )

    def update_migration_probability(self):
        event_impact = self.compute_event_impact()
        total_risk = event_impact

        # 3️⃣ Perceived Behavior Control (memory)
        perceived_risk = (
            self.risk_proneness * total_risk
            + self.memory_retention * self.previous_perceived_risk
        )

        self.previous_perceived_risk = perceived_risk

        self.migration_prob = 1 / (
            1
            + math.exp(
                -self.model.growth_rate * (perceived_risk - self.model.baseline_Q)
            )
        )

    def apply_migration(self):
        if self.random.random() <= self.migration_prob:
            self.state = CitizenState.MIGRATE
            self.internal_state.append(
                f"Agent {self.unique_id}: migration_prob={self.migration_prob:.3f}, state={self.state.name}"
            )

    def explain_decision(self):
        observation = self.generate_obs()
        prompt = f"""
        You are a civilian living in a conflict zone.

        Current perceived risk: {self.previous_perceived_risk:.3f}
        Migration probability: {self.migration_prob:.3f}
        Current state: {self.state.name}

        Rules:
        •⁠  ⁠If your state is MIGRATE, move toward the safe zone using move_to_safe_zone.
        •⁠  ⁠If your state is REST, do NOT move to the safe zone. You may stay or wander.
        - You can use speak_to tool also.

        Explain briefly why the migration probability is high or low.
        """

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["move_one_step", "move_to_safe_zone", "speak_to"],
        )

        self.apply_plan(plan)

    def step(self):
        self.compute_event_impact()
        self.update_migration_probability()
        self.apply_migration()

        self.explain_decision()
