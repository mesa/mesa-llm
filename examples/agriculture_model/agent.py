from enum import Enum

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager

FARMER_TOOL_MANAGER = ToolManager()


class CropState(Enum):
    IDLE = "IDLE"
    PLANTED = "PLANTED"
    GROWING = "GROWING"
    READY = "READY"


class FarmerAgent(LLMAgent):
    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        step_prompt,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
        )

        self.tool_manager = FARMER_TOOL_MANAGER
        self.land_size = self.random.randint(1, 5)
        self.wealth = self.random.randint(1000, 50000)
        self._base_internal_state = list(self.internal_state)

        self.crop_type = self.random.choice(["wheat", "rice", "maize"])
        self.crop_state = CropState.IDLE

        self.plant_date = None
        self.harvest_date = None
        self.fertilizer = 0.0

        self.yield_output = 0.0
        self.profit = 0.0

        self.memory = STLTMemory(
            agent=self,
            llm_model=llm_model,
            display=True,
        )

    def observe_environment(self):
        self.internal_state = [
            *self._base_internal_state,
            f"My land size is {self.land_size}",
            f"On a scale of 1000 - 50000, my wealth is {self.wealth}",
            f"My crop type is {self.crop_type}",
            f"My crop state is {self.crop_state.value}",
            f"Rainfall condition is {self.model.rainfall}",
        ]

    def update_crop_state(self):
        if (
            self.crop_state in {CropState.PLANTED, CropState.GROWING}
            and self.harvest_date is not None
            and self.model.current_day >= self.harvest_date
        ):
            self.crop_state = CropState.READY

    def decide(self):
        observation = self.generate_obs()

        prompt = f"""
        You are a farmer.
        wealth: {self.wealth}
        Crop type: {self.crop_type}
        Crop state: {self.crop_state}
        land size: {self.land_size}

        Decide actions:
        - plant_crop(days_to_harvest)
        - apply_fertilizer(level)
        - harvest_crop()
        - speak_to()

        Use tools smartly.
        """

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=[
                "plant_crop",
                "apply_fertilizer",
                "harvest_crop",
                "speak_to",
            ],
        )

        self.apply_plan(plan)

    def compute_yield(self):
        base_yield = 1000 * self.land_size

        if self.model.rainfall == "LOW":
            rain_factor = 0.7
        elif self.model.rainfall == "HIGH":
            rain_factor = 1.3
        else:
            rain_factor = 1.0

        fert_factor = 1 + 0.3 * self.fertilizer
        noise = self.random.uniform(0.9, 1.1)

        self.profit = 0

        self.yield_output = base_yield * rain_factor * fert_factor * noise
        price = self.model.market_price[self.crop_type]
        self.profit += self.yield_output * price

    def step(self):
        self.observe_environment()
        self.update_crop_state()
        self.decide()
        self.update_crop_state()
