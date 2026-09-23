import logging
import warnings

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_plot_component,
    make_space_component,
)

from examples.ev_model.agent import AgentState, HouseholdAgent
from examples.ev_model.model import EVModel
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping
from mesa_llm.reasoning.react import ReActReasoning

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
)

logging.getLogger("pydantic").setLevel(logging.ERROR)

enable_automatic_parallel_stepping(mode="threading")

load_dotenv()


# -----------------------
# Agent Colors
# -----------------------

agent_colors = {
    AgentState.EV_HOLDER: "#2ecc71",  # green
    AgentState.ICE_HOLDER: "#e74c3c",  # red
}


# -----------------------
# Model Parameters
# -----------------------

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "num_households": 20,
    "width": 10,
    "height": 10,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/llama3.1:latest",
    "vision": 5,
}


model = EVModel(
    num_households=model_params["num_households"],
    width=model_params["width"],
    height=model_params["height"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    seed=model_params["seed"]["value"],
)


# -----------------------
# Agent Visualization
# -----------------------


def agent_portrayal(agent):

    if agent is None:
        return

    portrayal = {
        "size": 40,
        "layer": 1,
    }

    if isinstance(agent, HouseholdAgent):
        if agent.state == AgentState.EV_HOLDER:
            portrayal["color"] = agent_colors[AgentState.EV_HOLDER]
            portrayal["marker"] = "$EV$"

        elif agent.state == AgentState.ICE_HOLDER:
            portrayal["color"] = agent_colors[AgentState.ICE_HOLDER]
            portrayal["marker"] = "$ICE$"

        else:
            portrayal["color"] = "gray"
            portrayal["marker"] = "o"

    return portrayal


def post_process(ax):

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(8, 8)


space_component = make_space_component(
    agent_portrayal,
    post_process=post_process,
    draw_grid=False,
)


# -----------------------
# Graph (EV vs ICE)
# -----------------------

chart_component = make_plot_component(
    {
        "ev": agent_colors[AgentState.EV_HOLDER],
        "ice": agent_colors[AgentState.ICE_HOLDER],
    }
)


# -----------------------
# App
# -----------------------

if __name__ == "__main__":
    page = SolaraViz(
        model,
        components=[
            space_component,
            chart_component,
        ],
        model_params=model_params,
        name="EV Adoption Model",
    )
