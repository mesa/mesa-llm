# app.py (at the very top, before any other imports)
import logging
import warnings

import pandas as pd
import solara
from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    SpaceRenderer,
)

from examples.negotiation.agents import BuyerAgent, SellerAgent
from examples.negotiation.model import NegotiationModel
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


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "initial_buyers": 5,
    "width": 5,
    "height": 5,
    "reasoning": ReActReasoning,
    "llm_model": "ollama/llama3-groq-tool-use:8b",
    "vision": 5,
    "api_base": None,
}


model = NegotiationModel(
    initial_buyers=model_params["initial_buyers"],
    width=model_params["width"],
    height=model_params["height"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    api_base=model_params["api_base"],
    seed=model_params["seed"]["value"],
)


def model_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "size": 25,
    }

    if isinstance(agent, SellerAgent):
        portrayal["color"] = "tab:red"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2
    elif isinstance(agent, BuyerAgent):
        portrayal["color"] = "tab:blue"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 1

    return portrayal

@solara.component
def ShowSalesButton(*args, **kwargs):
    show = solara.use_reactive(False)
    df = solara.use_memo(
        lambda: (
            model.datacollector.get_model_vars_dataframe()
            if show.value
            else pd.DataFrame()
        ),
        [show.value],
    )

    def on_click():
        show.set(True)

    solara.Button(label="Show Sales Data", on_click=on_click)
    if show.value and not df.empty:
        solara.DataFrame(df)
renderer = (
    SpaceRenderer(model, backend="matplotlib")
    .setup_agents(model_portrayal)
    .render()
)

if __name__ == "__main__":
    
    page = SolaraViz(
        model,
        renderer,
        components=[
            ShowSalesButton,
        ],
        model_params=model_params,
        name="Negotiation",
    )


"""run with:
conda activate mesa-llm && solara run examples/negotiation/app.py
"""
