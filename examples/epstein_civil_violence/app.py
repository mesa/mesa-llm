# app.py (at the very top, before any other imports)
import logging
import warnings

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_plot_component,
    SpaceRenderer,
)

from examples.epstein_civil_violence.agents import Citizen, CitizenState, Cop
from examples.epstein_civil_violence.model import EpsteinModel
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping
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

enable_automatic_parallel_stepping(mode="threading")

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
    "initial_citizens": 5,
    "initial_cops": 5,
    "width": 10,
    "height": 10,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/llama3-groq-tool-use:8b", # openai/gpt-4o-mini
    "api_base": None,                  #
    "vision": 5,
    "parallel_stepping": False,
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
    parallel_stepping=model_params["parallel_stepping"],
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

renderer = SpaceRenderer(model, backend="altair").setup_agents(
    citizen_cop_portrayal,
)

chart_component = make_plot_component(
    {state.name.lower(): agent_colors[state] for state in CitizenState}
)

def post_process(chart):
    return (
        chart.properties(width=700, height=700)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
renderer.post_process = post_process


page = SolaraViz(
        model,
        renderer,
        components=[
            chart_component,
        ],  # Add ShowSalesButton here
        model_params=model_params,
        name="Espstein Civil Violence Model",
    )

if __name__ == "__main__":
    pass


"""run with:
cd examples/epstein_civil_violence
conda activate mesa-llm && solara run app.py
"""
