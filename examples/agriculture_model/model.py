from datetime import datetime, timedelta

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

from examples.agriculture_model.agent import CropState, FarmerAgent
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.record_model import record_model


@record_model(output_dir="recordings")
class FarmerModel(Model):
    def __init__(
        self,
        initial_farmers: int,
        rainfall: int,
        width: int,
        height: int,
        reasoning: type[Reasoning],
        llm_model: str,
        vision: int,
        parallel_stepping=True,
        seed=None,
    ):
        normalized_seed = None if seed in (None, "") else int(seed)
        super().__init__(seed=normalized_seed)
        self.width = width
        self.height = height
        self.parallel_stepping = parallel_stepping
        self.grid = MultiGrid(width, height, torus=False)

        self.start_date = datetime(2024, 6, 1)
        self.current_day = self.start_date

        self.rainfall = self._normalize_rainfall(rainfall=rainfall)
        self.market_price = {
            "wheat": 2.0,
            "rice": 3.0,
            "maize": 1.5,
        }

        for _ in range(initial_farmers):
            agent = FarmerAgent(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt="You are a smart farmer.",
                step_prompt="Decide actions.",
                vision=vision,
                internal_state=["hardworking"],
            )
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "Rice": lambda m: sum(
                    1
                    for a in m.agents
                    if isinstance(a, FarmerAgent) and a.crop_type == "rice"
                ),
                "Wheat": lambda m: sum(
                    1
                    for a in m.agents
                    if isinstance(a, FarmerAgent) and a.crop_type == "wheat"
                ),
                "Maize": lambda m: sum(
                    1
                    for a in m.agents
                    if isinstance(a, FarmerAgent) and a.crop_type == "maize"
                ),
                "Total Profit": lambda m: sum(
                    a.profit for a in m.agents if isinstance(a, FarmerAgent)
                ),
                "Average Profit": lambda m: (
                    sum(a.profit for a in m.agents if isinstance(a, FarmerAgent))
                    / max(
                        1,
                        sum(1 for a in m.agents if isinstance(a, FarmerAgent)),
                    )
                ),
                "Planted": lambda m: sum(
                    1
                    for a in m.agents
                    if isinstance(a, FarmerAgent) and a.crop_state == CropState.PLANTED
                ),
                "Growing": lambda m: sum(
                    1
                    for a in m.agents
                    if isinstance(a, FarmerAgent) and a.crop_state == CropState.GROWING
                ),
                "Ready": lambda m: sum(
                    1
                    for a in m.agents
                    if isinstance(a, FarmerAgent) and a.crop_state == CropState.READY
                ),
                "Idle": lambda m: sum(
                    1
                    for a in m.agents
                    if isinstance(a, FarmerAgent) and a.crop_state == CropState.IDLE
                ),
            }
        )

    def _normalize_rainfall(self, rainfall: float | str | None) -> str:
        if isinstance(rainfall, str):
            rainfall_upper = rainfall.upper()
            if rainfall_upper in {"LOW", "NORMAL", "HIGH"}:
                return rainfall_upper
            raise ValueError(f"Unsupported rainfall value: {rainfall}")

        if rainfall is None:
            return "NORMAL"
        if rainfall < 0.33:
            return "LOW"
        if rainfall > 0.66:
            return "HIGH"
        return "NORMAL"

    def step(self):
        self.current_day += timedelta(days=1)
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

        if (self.current_day - self.start_date).days >= 120:
            self.running = False


if __name__ == "__main__":
    from examples.agriculture_model.app import model

    for _ in range(5):
        model.step()
