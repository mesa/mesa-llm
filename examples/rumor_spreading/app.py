# app.py
import logging
import warnings

from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_plot_component,
    make_space_component,
)
from mesa.visualization.components import AgentPortrayalStyle

from examples.rumor_spreading.agents import BeliefState, CitizenAgent, JournalistAgent
from examples.rumor_spreading.model import RumorSpreadingModel
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping
from mesa_llm.reasoning.react import ReActReasoning

# Suppress Pydantic serialization warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
    message=r".*Pydantic serializer warnings.*",
)

logging.getLogger("pydantic").setLevel(logging.ERROR)

enable_automatic_parallel_stepping(mode="threading")

load_dotenv()

# Visual identity — each belief state gets a distinct colour so you
# can see the cascade happening in real time on the grid.
BELIEF_COLORS = {
    BeliefState.UNINFORMED: "#648FFF",  # calm blue   — hasn't heard anything yet
    BeliefState.BELIEVER: "#FE6100",  # orange      — bought into it
    BeliefState.SKEPTIC: "#FFB000",  # amber       — doubtful, on the fence
    BeliefState.DEBUNKED: "#DC267F",  # pink/red    — actively calling it false
}
JOURNALIST_COLOR = "#FF0000"  # bright red — unmistakably visible

# Model parameters exposed to the Solara UI
OLLAMA_URL = "put_your_ollama_url_here"  # e.g. "http://localhost:11434"

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    # Increased counts/area for clearer dynamics
    "initial_citizens": 12,
    "initial_journalists": 2,
    "width": 8,
    "height": 8,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/llama3.1:8b",
    "vision": 2,  # slightly larger broadcast radius
    "parallel_stepping": False,
    "api_base": OLLAMA_URL,
}

model = RumorSpreadingModel(
    initial_citizens=model_params["initial_citizens"],
    initial_journalists=model_params["initial_journalists"],
    width=model_params["width"],
    height=model_params["height"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    seed=model_params["seed"]["value"],
    parallel_stepping=model_params["parallel_stepping"],
    api_base=model_params["api_base"],
)


# Agent portrayal — what each dot on the grid looks like
def agent_portrayal(agent):
    if agent is None:
        return

    if isinstance(agent, JournalistAgent):
        # Large red circle — unmissable on any background
        return AgentPortrayalStyle(color=JOURNALIST_COLOR, size=150, alpha=1.0)

    if isinstance(agent, CitizenAgent):
        return AgentPortrayalStyle(
            color=BELIEF_COLORS[agent.belief_state],
            size=50,
            alpha=0.9,
        )


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(10, 10)

    # Hand-drawn legend at the bottom of the plot
    legend_y = -0.9
    ax.text(
        0,
        legend_y,
        "● Uninformed",
        color=BELIEF_COLORS[BeliefState.UNINFORMED],
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        2.5,
        legend_y,
        "● Believer",
        color=BELIEF_COLORS[BeliefState.BELIEVER],
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        5,
        legend_y,
        "● Skeptic",
        color=BELIEF_COLORS[BeliefState.SKEPTIC],
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        7.5,
        legend_y,
        "● Debunked",
        color=BELIEF_COLORS[BeliefState.DEBUNKED],
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        0,
        legend_y - 0.6,
        "● Journalist (fixed)",
        color=JOURNALIST_COLOR,
        fontsize=9,
        fontweight="bold",
    )


# Solara components
space_component = make_space_component(
    agent_portrayal, post_process=post_process, draw_grid=False
)

chart_component = make_plot_component(
    {
        "Uninformed": BELIEF_COLORS[BeliefState.UNINFORMED],
        "Believers": BELIEF_COLORS[BeliefState.BELIEVER],
        "Skeptics": BELIEF_COLORS[BeliefState.SKEPTIC],
        "Debunked": BELIEF_COLORS[BeliefState.DEBUNKED],
    }
)

if __name__ == "__main__":
    page = SolaraViz(
        model,
        components=[space_component, chart_component],
        model_params=model_params,
        name="Rumor Spreading Model",
    )

"""
run with:
    solara run examples/rumor_spreading/app.py
"""
