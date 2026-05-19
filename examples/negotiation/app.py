# app.py (at the very top, before any other imports)
import logging
import warnings

import pandas as pd
import solara
from dotenv import load_dotenv
from mesa.visualization import (
    SolaraViz,
    make_space_component,
)

from examples.negotiation.agents import BuyerAgent, SellerAgent
from examples.negotiation.model import NegotiationModel
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
    "llm_model": "openai/gpt-4o",
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

if __name__ == "__main__":

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

    page = SolaraViz(
        model,
        components=[
            make_space_component(model_portrayal),
            ShowSalesButton,
        ],  # Add ShowSalesButton here
        model_params=model_params,
        name="Negotiation",
    )


"""run with:
conda activate mesa-llm && solara run examples/negotiation/app.py
"""
