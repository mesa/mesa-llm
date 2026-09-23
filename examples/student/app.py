# app.py
import logging
import warnings

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_plot_component,
    make_space_component,
)

from examples.student.agent import SchoolAgent, StudentAgent, StudentState
from examples.student.model import StudentSchoolModel
from mesa_llm.reasoning.react import ReActReasoning

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
    message=r".*Pydantic serializer warnings.*",
)
logging.getLogger("pydantic").setLevel(logging.ERROR)

load_dotenv()


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "n_students": 20,
    "n_schools": 4,
    "width": 10,
    "height": 10,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/llama3.1:latest",
    "api_base": None,
    "vision": 5,
    "parallel_stepping": False,
}

model = StudentSchoolModel(
    width=model_params["width"],
    height=model_params["height"],
    n_students=model_params["n_students"],
    n_schools=model_params["n_schools"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    api_base=model_params["api_base"],
    seed=model_params["seed"]["value"],
    parallel_stepping=model_params["parallel_stepping"],
)


def agent_portrayal(agent):
    if agent is None:
        return None

    portrayal = {
        "filled": True,
        "layer": 1,
    }

    if isinstance(agent, StudentAgent):
        portrayal["shape"] = "circle"
        portrayal["size"] = 35
        portrayal["text"] = str(agent.grade)

        if agent.state == StudentState.DROPOUT:
            portrayal["color"] = "#c0392b"
        elif agent.state == StudentState.GRADUATE:
            portrayal["color"] = "#27ae60"
        else:
            portrayal["color"] = "#1f77b4"

    elif isinstance(agent, SchoolAgent):
        portrayal["shape"] = "rect"
        portrayal["size"] = 45
        portrayal["layer"] = 0
        portrayal["color"] = "#f39c12" if agent.selective else "#7f8c8d"
        portrayal["text"] = f"T:{int(agent.tuition)}"

    return portrayal


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(10, 10)


space_component = make_space_component(
    agent_portrayal,
    post_process=post_process,
    draw_grid=False,
)

chart_component = make_plot_component(
    {
        "Enrolled": "#1f77b4",
        "Dropout": "#c0392b",
        "Graduate": "#27ae60",
    }
)


if __name__ == "__main__":
    page = SolaraViz(
        model,
        components=[space_component, chart_component],
        model_params=model_params,
        name="School Enrollment Model",
    )

    """
    run with
    cd examples/student
    conda activate mesa-llm && solara run app.py
    """
