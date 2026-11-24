import logging
import warnings

import pandas as pd
import solara
from dotenv import load_dotenv
from mesa.visualization import SolaraViz, make_space_component

import examples.stock_market.tools  # noqa: F401, registers tools
from examples.stock_market.agents import AnalystAgent, TraderAgent
from examples.stock_market.model import StockMarketModel
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping
from mesa_llm.reasoning.react import ReActReasoning

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")
logging.getLogger("pydantic").setLevel(logging.ERROR)

enable_automatic_parallel_stepping(mode="threading")
load_dotenv()

model_params = {
    "seed": {"type": "InputText", "value": 42, "label": "Random Seed"},
    "initial_traders": 4,
    "n_analysts": 2,
    "width": 5,
    "height": 5,
    "reasoning": ReActReasoning,
    "llm_model": "openai/gpt-4o",
    "vision": 3,
    "initial_price": 100.0,
    "api_base": None,
}

model = StockMarketModel(
    initial_traders=model_params["initial_traders"],
    n_analysts=model_params["n_analysts"],
    width=model_params["width"],
    height=model_params["height"],
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    vision=model_params["vision"],
    initial_price=model_params["initial_price"],
    api_base=model_params["api_base"],
    seed=model_params["seed"]["value"],
)

if __name__ == "__main__":

    def model_portrayal(agent):
        if agent is None:
            return
        portrayal = {"size": 25}
        if isinstance(agent, AnalystAgent):
            portrayal["color"] = "tab:orange"
            portrayal["marker"] = "D"
            portrayal["zorder"] = 3
        elif isinstance(agent, TraderAgent):
            portrayal["color"] = "tab:green" if agent.budget > 500.0 else "tab:red"
            portrayal["marker"] = "o"
            portrayal["zorder"] = 2
        return portrayal

    @solara.component
    def MarketStatsPanel(*args, **kwargs):
        show = solara.use_reactive(False)
        df = solara.use_memo(
            lambda: (
                model.datacollector.get_model_vars_dataframe()
                if show.value
                else pd.DataFrame()
            ),
            [show.value],
        )
        solara.Button(label="Show Market Data", on_click=lambda: show.set(True))
        if show.value and not df.empty:
            solara.DataFrame(df)

    page = SolaraViz(
        model,
        components=[make_space_component(model_portrayal), MarketStatsPanel],
        model_params=model_params,
        name="Stock Market",
    )

"""
Run with:
conda activate mesa-llm && solara run examples/stock_market/app.py
"""
