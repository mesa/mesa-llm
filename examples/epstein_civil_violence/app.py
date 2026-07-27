# app.py (at the very top, before any other imports)
import logging
import warnings

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_plot_component,
    make_space_component,
)

from examples.epstein_civil_violence.agents import Citizen, CitizenState, Cop
from examples.epstein_civil_violence.model import EpsteinModel
from mesa_llm.reasoning.react import ReActReasoning

# Suppress Pydantic serialization warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
    message=r".*Pydantic serializer warnings.*",
)

# Also suppress through logging
logging.getLogger("pydantic").setLevel(logging.ERROR)

load_dotenv()

COP_COLOR = "#000000"

agent_colors = {
    CitizenState.ACTIVE: "#FE6100",
    CitizenState.QUIET: "#648FFF",
    CitizenState.ARRESTED: "#DB28A2",
}

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "initial_citizens": 20,
    "initial_cops": 5,
    "width": 10,
    "height": 10,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/llama3.2",
    "api_base": None,
    "vision": 5,
}


model = EpsteinModel(
    initial_citizens=model_params["initial_citizens"],
    initial_cops=model_params["initial_cops"],
    width=model_params["width"],
    height=model_params["height"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    api_base=model_params["api_base"],
    seed=model_params["seed"]["value"],
)


def citizen_cop_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "size": 50,
    }

    if isinstance(agent, Cop):
        portrayal["color"] = COP_COLOR

    elif isinstance(agent, Citizen):
        portrayal["color"] = agent_colors[agent.state]

    return portrayal


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(10, 10)


space_component = make_space_component(
    citizen_cop_portrayal, post_process=post_process, draw_grid=False
)

chart_component = make_plot_component(
    {state.name.lower(): agent_colors[state] for state in CitizenState}
)

if __name__ == "__main__":
    page = SolaraViz(
        model,
        components=[
            space_component,
            chart_component,
        ],  # Add ShowSalesButton here
        model_params=model_params,
        name="Espstein Civil Violence Model",
    )


"""run with:
cd examples/epstein_civil_violence
conda activate mesa-llm && solara run app.py
"""
