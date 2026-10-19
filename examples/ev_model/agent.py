import random
from enum import Enum

import mesa

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager

HOUSEHOLD_TOOL_MANAGER = ToolManager()


class AgentState(Enum):
    NONE_HOLDER = "none_holder"
    ICE_HOLDER = "ice_holder"
    EV_HOLDER = "ev_holder"


# -------------------------------
# Charging Station Agent
# -------------------------------


class ChargingStationAgent(LLMAgent, mesa.discrete_space.CellAgent):
    """
    Charging station providing electricity for EV vehicles.

    Charging stations represent the infrastructure layer
    that influences EV adoption decisions.

    Households evaluate the accessibility of charging stations
    when computing EV utility.

    Infrastructure score depends on:

        distance to the station
        station capacity
        current utilization (congestion)

    Infrastructure score formula:

        I = 1 / (1 + distance + congestion)

    Where:
        congestion = utilization_rate / capacity

    Attributes:
        capacity:
            Maximum number of EVs that can charge simultaneously.

        price_per_kwh:
            Electricity price charged per kWh.

        charging_speed:
            Charging rate of the station.

        utilization_rate:
            Current number of vehicles charging.

    Role in the model:
        Charging stations affect the infrastructure component
        of the EV adoption utility function used by households.
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        step_prompt,
        capacity=3,
        price_per_kwh=0.20,
        charging_speed=50,
    ):

        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            step_prompt=step_prompt,
        )

        self.capacity = capacity
        self.price_per_kwh = price_per_kwh
        self.charging_speed = charging_speed

        self.utilization_rate = 0

        self.memory = STLTMemory(
            agent=self,
            llm_model=llm_model,
            display=False,
        )

    def charging_cost(self, kwh):

        return kwh * self.price_per_kwh


class HouseholdAgent(LLMAgent, mesa.discrete_space.CellAgent):
    """
    Household agent deciding whether to adopt an Electric Vehicle (EV)
    or continue using an Internal Combustion Engine (ICE) vehicle.

    Summary of rule:
    A household compares the utility of EV and ICE vehicles.
    If EV utility exceeds ICE utility, the household adopts an EV.

    Decision model:
        U_EV = αF + βS + γI + δE − θR

    Where:
        F : Financial attractiveness
            Difference between EV and ICE total cost of ownership.

        S : Social influence
            Fraction of neighboring households that already adopted EV.

        I : Infrastructure convenience
            Accessibility and congestion level of nearby charging stations.

        E : Environmental motivation
            Household environmental awareness level.

        R : Risk perception
            Household hesitation toward adopting new technology.

    Attributes:
        income:
            Household annual income.

        env_awareness:
            Environmental concern level of the household.

        risk_aversion:
            Resistance to adopting new technology.

        annual_mileage:
            Distance traveled per year.

        total_cost_ev:
            Estimated total cost of owning an EV.

        total_cost_ice:
            Estimated total cost of owning an ICE vehicle.

        utility_ev:
            Utility score for EV adoption.

        utility_ice:
            Utility score for ICE ownership.

    State:
        NONE : household does not yet own a car
        ICE  : household owns an ICE vehicle
        EV   : household owns an electric vehicle
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        step_prompt,
        vision,
    ):

        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            step_prompt=step_prompt,
            vision=vision,
        )

        self.state = AgentState.NONE_HOLDER

        # socioeconomic attributes
        self.income = random.uniform(20000, 100000)
        self.env_awareness = random.random()
        self.risk_aversion = random.random()

        self.annual_mileage = random.uniform(8000, 20000)

        # costs
        self.total_cost_ev = 0
        self.total_cost_ice = 0

        # utilities
        self.utility_ev = 0
        self.utility_ice = 0
        self.battery_capacity = 60  # kWh
        self.battery_level = 60  # start full
        self.energy_consumption = 0.18  # kWh per km
        self.daily_distance = random.uniform(10, 50)

        self.memory = STLTMemory(
            agent=self,
            llm_model=llm_model,
            display=True,
        )

        self.tool_manager = HOUSEHOLD_TOOL_MANAGER

        self.internal_state.append(f"My income is {self.income}")

        self.internal_state.append(
            f"My environmental awareness is {self.env_awareness}"
        )

    # ---------------- COST CALCULATIONS ----------------

    def calculate_ice_cost(self):

        fuel_cost = (
            self.model.fuel_price * self.annual_mileage / self.model.fuel_efficiency
        )

        self.total_cost_ice = (
            self.model.purchase_price_ice + fuel_cost + self.model.maintenance_ice
        )

    def calculate_ev_cost(self):

        electricity_cost = (
            self.model.electricity_price
            * self.annual_mileage
            / self.model.ev_efficiency
        )

        self.total_cost_ev = (
            self.model.purchase_price_ev
            - self.model.subsidy_amount
            + electricity_cost
            + self.model.maintenance_ev
        )

    # ---------------- SOCIAL INFLUENCE ----------------

    def compute_social_influence(self):

    neighbors = self.model.grid.get_neighbors(
        self.pos, moore=True, include_center=False
    )

    household_neighbors = [
        n for n in neighbors if isinstance(n, HouseholdAgent)
    ]

    if len(household_neighbors) == 0:
        return 0

    ev_neighbors = sum(
        1
        for n in household_neighbors
        if n.state == AgentState.EV_HOLDER
    )

    return ev_neighbors / len(household_neighbors)

    # ---------------- INFRASTRUCTURE ----------------

    def compute_infrastructure_score(self):

        stations = self.model.charging_stations
        scores = []

        for s in stations:
            dx = self.pos[0] - s.pos[0]
            dy = self.pos[1] - s.pos[1]
            distance = (dx**2 + dy**2) ** 0.5

            congestion = s.utilization_rate / s.capacity

            score = 1 / (1 + distance + congestion)

            scores.append(score)

        return max(scores)

    # ---------------- UTILITY ----------------

    def compute_utility(self):

        financial = (self.total_cost_ice - self.total_cost_ev) / max(
            self.total_cost_ice, 1
        )
        social = self.compute_social_influence()
        infrastructure = self.compute_infrastructure_score()
        environment = self.env_awareness
        risk = self.risk_aversion

        self.utility_ev = (
            self.model.alpha_financial * financial
            + self.model.beta_social * social
            + self.model.gamma_infrastructure * infrastructure
            + self.model.delta_environment * environment
            - self.model.theta_risk * risk
        )

        self.utility_ice = (
            -self.model.alpha_financial * financial
            - self.model.beta_social * social
            - self.model.gamma_infrastructure * infrastructure
            - self.model.delta_environment * environment
        )

    def drive(self):
        if self.state == AgentState.EV_HOLDER:
            energy_used = self.daily_distance * self.energy_consumption
            self.battery_level -= energy_used
            if self.battery_level < 0:
                self.battery_level = 0

    def find_nearest_station(self):

        stations = self.model.charging_stations

        nearest = min(
            stations, key=lambda s: self.model.grid.get_distance(self.pos, s.pos)
        )
        return nearest

    # ---------------- LLM DECISION ----------------

    def make_decision(self):

        observation = self.generate_obs()

        prompt = f"""
        You are a household deciding transportation actions.

        Current vehicle state: {self.state}
        EV utility: {self.utility_ev:.3f}
        ICE utility: {self.utility_ice:.3f}
        Battery level: {self.battery_level:.2f}

        Decision rules:
        1. If you do not own a vehicle → choose buy_ev or buy_ice
        2. If you own an EV and battery < 30% → use charge_ev
        3. If you own an ICE vehicle → do nothing

        Available tools:
        - buy_ev
        - buy_ice
        - charge_ev
        """

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["buy_ev", "buy_ice", "charge_ev"],
        )

        self.apply_plan(plan)

    # ---------------- STEP ----------------

    def step(self):

        self.calculate_ice_cost()
        self.calculate_ev_cost()
        self.compute_utility()
        self.drive()
        self.make_decision()
