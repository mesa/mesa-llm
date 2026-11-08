# app.py
import logging
import warnings

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_plot_component,
    make_space_component,
)

from examples.sugarscrap_g1mt.agents import Resource, Trader
from examples.sugarscrap_g1mt.model import SugarScapeModel
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

# enable_automatic_parallel_stepping(mode="threading")

load_dotenv()


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "initial_traders": 2,
    "initial_resources": 10,
    "width": 10,
    "height": 10,
    "reasoning": ReActReasoning,
    "llm_model": "gemini/gemini-2.5-flash",
    "api_base": None,
    "vision": 5,
    "parallel_stepping": False,
}

model = SugarScapeModel(
    initial_traders=model_params["initial_traders"],
    initial_resources=model_params["initial_resources"],
    width=model_params["width"],
    height=model_params["height"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    api_base=model_params["api_base"],
    seed=model_params["seed"]["value"],
    parallel_stepping=model_params["parallel_stepping"],
)


def agent_potrayal(agent):
    if agent is None:
        return

    portrayal = {
        "shape": "circle",
        "filled": True,
        "size": 20,
        "layer": 1,
    }

    if isinstance(agent, Trader):
        portrayal["color"] = "black"
        portrayal["size"] = 50
        portrayal["text"] = f"S:{agent.sugar} Sp:{agent.spice}"

    elif isinstance(agent, Resource):
        portrayal["layer"] = 0
        color = "green"
        if hasattr(agent, "type"):
            if agent.type == "sugar":
                color = "blue"
            elif agent.type == "spice":
                color = "red"

        portrayal["color"] = color
        portrayal["size"] = 15

        if agent.current_amount > 0:
            portrayal["opacity"] = agent.current_amount / agent.max_capacity
        else:
            portrayal["color"] = "white"

    return portrayal


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(10, 10)

    ax.text(0, 10.5, "● Trader (Black)", color="black", fontsize=10, fontweight="bold")
    ax.text(4, 10.5, "● Sugar (Blue)", color="blue", fontsize=10, fontweight="bold")
    ax.text(8, 10.5, "● Spice (Red)", color="red", fontsize=10, fontweight="bold")


space_component = make_space_component(
    agent_potrayal, post_process=post_process, draw_grid=False
)

chart_component = make_plot_component({"Total_Sugar": "blue", "Total_Spice": "red"})

if __name__ == "__main__":
    page = SolaraViz(
        model,
        components=[
            space_component,
            chart_component,
        ],
        model_params=model_params,
        name="SugarScape G1MT Example",
    )

    """
    run with
    cd examples/sugarscrap_g1mt
    conda activate mesa-llm && solara run app.py
    """
