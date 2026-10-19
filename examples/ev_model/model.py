from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid

from examples.ev_model.agent import ChargingStationAgent, HouseholdAgent


class EVModel(Model):
    def __init__(
        self,
        num_households,
        width,
        height,
        reasoning,
        llm_model,
        vision,
        seed=None,
    ):

        super().__init__(seed=seed)

        self.grid = MultiGrid(width, height, torus=False)

        # Utility weights
        self.alpha_financial = 0.4
        self.beta_social = 0.3
        self.gamma_infrastructure = 0.2
        self.delta_environment = 0.1
        self.theta_risk = 0.2

        # Costs
        self.purchase_price_ev = 3000
        self.purchase_price_ice = 2800

        self.maintenance_ev = 300
        self.maintenance_ice = 800

        self.fuel_price = 1.6
        self.electricity_price = 0.15

        self.fuel_efficiency = 15
        self.ev_efficiency = 5

        self.subsidy_amount = 50

        self.charging_stations = []

        # Create charging stations

        station1 = ChargingStationAgent(
            model=self,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="Charging station",
            step_prompt="Serve EV vehicles",
            capacity=5,
            price_per_kwh=0.15,
        )

        station2 = ChargingStationAgent(
            model=self,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="Charging station",
            step_prompt="Serve EV vehicles",
            capacity=2,
            price_per_kwh=0.20,
        )

        self.charging_stations.append(station1)
        self.charging_stations.append(station2)

        self.grid.place_agent(station1, (1, 1))
        self.grid.place_agent(station2, (8, 8))

        # Create households

        agents = HouseholdAgent.create_agents(
            self,
            n=num_households,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="You are a household deciding between EV and ICE vehicles.",
            step_prompt="Evaluate vehicle costs and decide.",
            vision=vision,
        )

        for agent in agents:
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)

            self.grid.place_agent(agent, (x, y))

        # Data collector

        self.datacollector = DataCollector(
            model_reporters={
                "ev": lambda m: sum(
                    1
                    for a in m.agents
                    if isinstance(a, HouseholdAgent) and a.state.name == "EV_HOLDER"
                ),
                "ice": lambda m: sum(
                    1
                    for a in m.agents
                    if isinstance(a, HouseholdAgent) and a.state.name == "ICE_HOLDER"
                ),
            }
        )

    def step(self):

        self.agents.shuffle_do("step")

        self.datacollector.collect(self)


if __name__ == "__main__":
    from examples.ev_model.app import model

    for _ in range(5):
        model.step()
