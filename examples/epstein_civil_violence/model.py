from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print

from examples.epstein_civil_violence.agents import Citizen, CitizenState, Cop
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.record_model import record_model


@record_model(output_dir="recordings")
class EpsteinModel(Model):
    def __init__(
        self,
        initial_cops: int,
        initial_citizens: int,
        width: int,
        height: int,
        reasoning: type[Reasoning],
        llm_model: str,
        vision: int,
        api_base: str | None = None,
        parallel_stepping=False,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.parallel_stepping = parallel_stepping
        self.grid = MultiGrid(self.height, self.width, torus=False)

        model_reporters = {
            "active": lambda m: sum(
                1
                for agent in m.agents
                if isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE
            ),
            "quiet": lambda m: sum(
                1
                for agent in m.agents
                if isinstance(agent, Citizen) and agent.state == CitizenState.QUIET
            ),
            "arrested": lambda m: sum(
                1
                for agent in m.agents
                if isinstance(agent, Citizen) and agent.state == CitizenState.ARRESTED
            ),
        }
        agent_reporters = {
            "jail_sentence": lambda a: getattr(a, "jail_sentence_left", None),
            "arrest_probability": lambda a: getattr(a, "arrest_probability", None),
        }
        self.datacollector = DataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )

        # ---------------------Create the cop agents---------------------
        cop_system_prompt = "You are a cop. You are tasked with arresting citizens if they are active and their arrest probability is high enough. You are also tasked with moving to a new location if there is no citizen in sight."

        agents = Cop.create_agents(
            self,
            n=initial_cops,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=cop_system_prompt,
            vision=vision,
            internal_state=None,
            step_prompt="Inspect your local vision and arrest a random active agent. Move if applicable.",
            api_base=api_base,
        )

        x = self.rng.integers(0, self.grid.width, size=(initial_cops,))
        y = self.rng.integers(0, self.grid.height, size=(initial_cops,))
        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

        # ---------------------Create the citizen agents---------------------
        agents = Citizen.create_agents(
            self,
            n=initial_citizens,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="",
            vision=vision,
            internal_state=None,
            step_prompt="Move around and change your state if the conditions indicate it.",
            api_base=api_base,
        )

        x = self.rng.integers(0, self.grid.width, size=(initial_citizens,))
        y = self.rng.integers(0, self.grid.height, size=(initial_citizens,))
        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

    def step(self):
        """
        Execute one step of the model.
        """

        print(
            f"\n[bold purple] step  {self.steps} ────────────────────────────────────────────────────────────────────────────────[/bold purple]"
        )
        self.agents.shuffle_do("step")

        self.datacollector.collect(self)


# ===============================================================
#                     RUN WITHOUT GRAPHICS
# ===============================================================

if __name__ == "__main__":
    """
    run the model without the solara integration with:
    conda activate mesa-llm && python -m examples.epstein_civil_violence.model
    """
    from examples.epstein_civil_violence.app import model

    for _ in range(5):
        model.step()
