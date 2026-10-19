import logging
import warnings

from dotenv import load_dotenv
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
    make_space_component,
)

from examples.agriculture_model.agent import CropState, FarmerAgent
from examples.agriculture_model.model import FarmerModel
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping
from mesa_llm.reasoning.react import ReActReasoning

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
    message=r".*Pydantic serializer warnings.*",
)
logging.getLogger("pydantic").setLevel(logging.ERROR)

enable_automatic_parallel_stepping()
load_dotenv()


CROP_COLORS = {
    CropState.IDLE: "#D7C7A3",
    CropState.PLANTED: "#A3D977",
    CropState.GROWING: "#6FBF73",
    CropState.READY: "#FFBD66",
}


model_params = {
    "initial_farmers": Slider("Number of Farmers", 10, 1, 50, 1),
    "rainfall": Slider("Rainfall Level", 0.5, 0.0, 1.0, 0.1),
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "width": 20,
    "height": 20,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/llama3.1:latest",
    "vision": 3,
    "parallel_stepping": False,
}

model = FarmerModel(
    initial_farmers=10,
    rainfall=0.5,
    width=model_params["width"],
    height=model_params["height"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    seed=model_params["seed"]["value"],
    parallel_stepping=model_params["parallel_stepping"],
)


def farmer_portrayal(agent):
    if agent is None:
        return None

    portrayal = {"size": 50}

    if isinstance(agent, FarmerAgent):
        portrayal["color"] = CROP_COLORS.get(agent.crop_state, "#888888")

    return portrayal


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(8, 8)


space_component = make_space_component(
    farmer_portrayal,
    post_process=post_process,
    draw_grid=False,
)

crop_chart = make_plot_component(
    {
        "Rice": "#4CAF50",
        "Wheat": "#FFC107",
        "Maize": "#FF5722",
    }
)

profit_chart = make_plot_component(
    {
        "Total Profit": "#2196F3",
        "Average Profit": "#9C27B0",
    }
)

state_chart = make_plot_component(
    {
        "Planted": "#A3D977",
        "Growing": "#6FBF73",
        "Ready": "#FFBD66",
        "Idle": "#D7C7A3",
    }
)


page = SolaraViz(
    model,
    components=[
        space_component,
        crop_chart,
        profit_chart,
        state_chart,
    ],
    model_params=model_params,
    name="Agriculture Decision Model (Mesa-LLM)",
)
