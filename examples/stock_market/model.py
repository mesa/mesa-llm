import math

import numpy as np
from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print

from examples.stock_market.agents import AnalystAgent, TraderAgent


class StockMarketModel(Model):
    """
    A stock market simulation where LLM-powered trader agents buy and sell
    shares based on price data and analyst signals.

    Args:
        initial_traders (int): Number of trader agents.
        n_analysts (int): Number of analyst agents.
        width (int): Grid width.
        height (int): Grid height.
        reasoning: Reasoning class for all agents.
        llm_model (str): LiteLLM model string e.g. "openai/gpt-4o".
        vision (int): Agent observation radius.
        initial_price (float): Starting stock price.
        api_base (str | None): Optional custom API base URL.
        seed: Random seed.
    """

    def __init__(
        self,
        initial_traders,
        n_analysts,
        width,
        height,
        reasoning,
        llm_model,
        vision,
        initial_price=100.0,
        api_base=None,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.current_price = initial_price
        self.price_history = [initial_price]
        self.grid = MultiGrid(self.height, self.width, torus=False)

        # Analyst agents
        analysts = AnalystAgent.create_agents(
            self,
            n=n_analysts,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="You are a quantitative market analyst. Observe price trends, RSI, and volatility. Send clear BUY, HOLD, or SELL signals to nearby traders. Be concise and data-driven.",
            vision=vision,
            internal_state=["analytical", "data-driven", "calm"],
            api_base=api_base,
        )
        ax = self.rng.integers(0, self.grid.width, size=(n_analysts,))
        ay = self.rng.integers(0, self.grid.height, size=(n_analysts,))
        for a, i, j in zip(analysts, ax, ay):
            self.grid.place_agent(a, (i, j))

        # Conservative traders
        n_conservative = math.ceil(initial_traders / 2)
        conservative = TraderAgent.create_agents(
            self,
            n=n_conservative,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="You are a conservative trader. Only buy when RSI is below 35 and trend is rising. Sell quickly if you sense risk. Protect capital above all else.",
            vision=vision,
            internal_state=["risk-averse", "patient", "disciplined"],
            budget=500.0,
            api_base=api_base,
        )
        cx = self.rng.integers(0, self.grid.width, size=(n_conservative,))
        cy = self.rng.integers(0, self.grid.height, size=(n_conservative,))
        for a, i, j in zip(conservative, cx, cy):
            self.grid.place_agent(a, (i, j))

        # Aggressive traders
        n_aggressive = math.floor(initial_traders / 2)
        aggressive = TraderAgent.create_agents(
            self,
            n=n_aggressive,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="You are an aggressive momentum trader. Buy when price is rising, sell fast to lock in gains. Act quickly on analyst signals. Maximize profit.",
            vision=vision,
            internal_state=["risk-tolerant", "fast", "opportunistic"],
            budget=500.0,
            api_base=api_base,
        )
        agx = self.rng.integers(0, self.grid.width, size=(n_aggressive,))
        agy = self.rng.integers(0, self.grid.height, size=(n_aggressive,))
        for a, i, j in zip(aggressive, agx, agy):
            self.grid.place_agent(a, (i, j))

        self.datacollector = DataCollector(
            model_reporters={
                "Stock_Price": lambda m: round(m.current_price, 2),
                "Total_Trades": lambda m: sum(
                    a.trades for a in m.agents if isinstance(a, TraderAgent)
                ),
                "Avg_Trader_Budget": lambda m: round(
                    np.mean([a.budget for a in m.agents if isinstance(a, TraderAgent)]),
                    2,
                ),
            }
        )

    def price_trend(self) -> str:
        if len(self.price_history) < 3:
            return "insufficient data"
        delta = self.price_history[-1] - self.price_history[-3]
        if delta > 1.5:
            return f"strongly rising (+${delta:.2f})"
        elif delta > 0:
            return f"slightly rising (+${delta:.2f})"
        elif delta < -1.5:
            return f"strongly falling (-${abs(delta):.2f})"
        elif delta < 0:
            return f"slightly falling (-${abs(delta):.2f})"
        return "flat"

    def rsi(self, period: int = 6) -> float:
        if len(self.price_history) < period + 1:
            return 50.0
        deltas = np.diff(self.price_history[-period - 1 :])
        gains = deltas[deltas > 0].sum()
        losses = abs(deltas[deltas < 0].sum())
        if losses == 0:
            return 100.0
        return round(100 - (100 / (1 + gains / losses)), 2)

    def volatility(self, window: int = 10) -> float:
        if len(self.price_history) < 2:
            return 0.0
        prices = np.array(self.price_history[-window:])
        return round(float(np.std(np.diff(np.log(prices)))), 4)

    def _update_price(self):
        shock = self.rng.normal(0, 0.015)
        mean_reversion = 0.01 * (100.0 - self.current_price) / 100.0
        self.current_price *= 1 + shock + mean_reversion
        self.current_price = max(self.current_price, 1.0)
        self.price_history.append(round(self.current_price, 4))

    def step(self):
        self._update_price()
        self.datacollector.collect(self)
        print(
            f"\n[bold cyan] step {self.steps} | price ${self.current_price:.2f} "
            f"| RSI {self.rsi():.1f} | {self.price_trend()} ────────────────[/bold cyan]"
        )
        self.agents.shuffle_do("step")


if __name__ == "__main__":
    from examples.stock_market.app import model

    for _ in range(10):
        model.step()
