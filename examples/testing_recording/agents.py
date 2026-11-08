import mesa

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory


class Trader(LLMAgent, mesa.discrete_space.CellAgent):
    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        step_prompt,
        sugar=0,
        spice=0,
        metabolism_sugar=1,
        metabolism_spice=1,
        api_base=None,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
            api_base=api_base,
            tools=["move_to_best_resource", "propose_trade"],
        )
        self.sugar = sugar
        self.spice = spice
        self.metabolism_sugar = metabolism_sugar
        self.metabolism_spice = metabolism_spice

        self.prices = []
        self.trade_partners = []

        self.memory = STLTMemory(
            agent=self,
            display=True,
            llm_model=llm_model,
            api_base=api_base,
        )

        self.system_prompt = (
            "You are a Trader agent in a Sugarscape simulation. "
            "You need Sugar and Spice to survive. "
            "If your MRS (Marginal Rate of Substitution) is high, you desperately need Sugar. "
            "If MRS is low, you need Spice. "
            "You can move to harvest resources or trade with neighbors."
        )

        self.update_internal_metrics()

    def calculate_mrs(self):
        if self.sugar == 0:
            return 100.0

        if self.metabolism_sugar == 0:
            return 100.0

        if self.metabolism_spice == 0:
            return 0.0

        return (self.spice / self.metabolism_spice) / (
            self.sugar / self.metabolism_sugar
        )

    def update_internal_metrics(self):
        mrs = self.calculate_mrs()

        self.internal_state = [
            s
            for s in self.internal_state
            if not any(x in s for x in ["Sugar", "Spice", "MRS", "WARNING_"])
        ]

        self.internal_state.append(f"My Sugar inventory is: {self.sugar}")
        self.internal_state.append(f"My Spice inventory is: {self.spice}")
        self.internal_state.append(
            f"My Marginal Rate of Substitution (MRS) is {mrs:.2f}"
        )

        if self.sugar < self.metabolism_sugar * 2:
            self.internal_state.append(
                "WARNING: I am in danger of starvation from lack of sugar"
            )
        if self.spice < self.metabolism_spice * 2:
            self.internal_state.append(
                "WARNING: I am in danger of starvation from lack of spice"
            )

    def get_trade(self):
        return self.trade_partners

    def step(self):
        self.sugar -= self.metabolism_sugar
        self.spice -= self.metabolism_spice

        if self.sugar <= 0 or self.spice <= 0:
            self.model.grid.remove_agent(self)
            self.remove()
            return

        self.update_internal_metrics()

        observation = self.generate_obs()

        plan = self.reasoning.plan(
            obs=observation,
            tools=["move_to_best_resource", "propose_trade"],
        )

        self.apply_plan(plan)

    async def astep(self):
        self.sugar -= self.metabolism_sugar
        self.spice -= self.metabolism_spice

        if self.sugar <= 0 or self.spice <= 0:
            self.model.grid.remove_agent(self)
            self.remove()
            return

        self.update_internal_metrics()
        observation = self.generate_obs()

        plan = await self.reasoning.aplan(
            obs=observation,
            tools=["move_to_best_resource", "propose_trade"],
        )
        self.apply_plan(plan)


class Resource(mesa.discrete_space.CellAgent):
    def __init__(
        self, model, max_capacity=10, current_amount=10, growback=1, type="sugar"
    ):
        super().__init__(model=model)

        self.max_capacity = max_capacity
        self.current_amount = current_amount
        self.growback = growback
        self.type = type

        self.internal_state = []

    def get_trade(self):
        return []

    def step(self):
        if self.current_amount < self.max_capacity:
            self.current_amount += self.growback
            if self.current_amount > self.max_capacity:
                self.current_amount = self.max_capacity

    async def astep(self):
        self.step()
