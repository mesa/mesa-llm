"""
Stock Market with Rumors Model

Simulates how information asymmetry, insider trading, and rumors
affect stock prices. LLM-powered agents trade, spread rumors,
publish reports, and provide liquidity.

Author: Adarsh Kumar (GSoC 2026)
"""

from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print

from examples.stock_market.agents import (
    Analyst,
    InsiderTrader,
    MarketMaker,
    RetailTrader,
)
from mesa_llm.reasoning.reasoning import Reasoning


def avg_stock_price(model):
    prices = [s["price"] for s in model.stocks.values()]
    return sum(prices) / len(prices) if prices else 0


def total_market_volume(model):
    return sum(s["volume"] for s in model.stocks.values())


def total_rumors(model):
    return len(model.rumors)


def manipulation_rumors(model):
    return sum(1 for r in model.rumors.values() if r["is_manipulation"])


def market_volatility(model):
    volatilities = []
    for data in model.stocks.values():
        history = data["price_history"]
        if len(history) >= 2:
            changes = [
                abs(history[i] - history[i - 1]) / history[i - 1] * 100
                for i in range(1, len(history))
                if history[i - 1] > 0
            ]
            if changes:
                volatilities.append(sum(changes) / len(changes))
    return sum(volatilities) / len(volatilities) if volatilities else 0


def insider_profit(model):
    insiders = [a for a in model.agents if isinstance(a, InsiderTrader)]
    total = 0
    for insider in insiders:
        portfolio_value = sum(
            insider.portfolio.get(t, 0) * model.stocks[t]["price"] for t in model.stocks
        )
        total += insider.cash + portfolio_value
    return total


def detect_bubble(model):
    for data in model.stocks.values():
        history = data["price_history"]
        if len(history) >= 5:
            initial = history[-5]
            current = history[-1]
            if initial > 0 and (current - initial) / initial > 0.3:
                return True
    return False


class StockMarketModel(Model):
    """
    A model simulating stock market dynamics with information asymmetry.

    Features:
    - Multiple stocks with supply/demand price mechanics
    - Insider traders who manipulate through rumors
    - Analysts providing research reports
    - Market makers stabilizing volatility
    - Bubble and crash detection
    """

    def __init__(
        self,
        n_retail: int = 8,
        n_insiders: int = 1,
        n_analysts: int = 2,
        n_makers: int = 1,
        width: int = 8,
        height: int = 8,
        reasoning: type[Reasoning] = Reasoning,
        llm_model: str = "gpt-4o-mini",
        vision: int = 2,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.grid = MultiGrid(width, height, torus=True)

        # Initialize stocks
        self.stocks = {
            "TECH": {
                "price": 100.0,
                "price_history": [100.0],
                "demand": 0,
                "supply": 0,
                "volume": 0,
                "base_volatility": 0.03,
            },
            "HEALTH": {
                "price": 75.0,
                "price_history": [75.0],
                "demand": 0,
                "supply": 0,
                "volume": 0,
                "base_volatility": 0.02,
            },
            "ENERGY": {
                "price": 50.0,
                "price_history": [50.0],
                "demand": 0,
                "supply": 0,
                "volume": 0,
                "base_volatility": 0.04,
            },
        }

        self.rumors = {}
        self._rumor_counter = 0
        self.reports = {}
        self._report_counter = 0

        # Create Retail Traders
        risk_levels = ["conservative", "moderate", "aggressive"]
        for i in range(n_retail):
            risk = risk_levels[i % len(risk_levels)]
            cash = self.rng.integers(500, 2000)
            agent = RetailTrader(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt=(
                    f"You are a {risk} retail trader in a stock market. "
                    "Analyze market data, listen to rumors and reports, "
                    "and make trading decisions. Be careful of manipulation! "
                    "Use tools for all actions."
                ),
                vision=vision,
                internal_state=[risk],
                starting_cash=float(cash),
                risk_tolerance=risk,
            )
            pos = (self.rng.integers(0, width), self.rng.integers(0, height))
            self.grid.place_agent(agent, pos)

        # Create Insider Traders
        target_stocks = list(self.stocks.keys())
        for i in range(n_insiders):
            target = target_stocks[i % len(target_stocks)]
            agent = InsiderTrader(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt=(
                    "You are an insider trader with secret knowledge. "
                    "Your goal is to profit by buying stocks cheaply, "
                    "spreading positive rumors to pump the price, "
                    "then selling at the top. Be subtle and strategic. "
                    "Use tools for all actions."
                ),
                vision=vision,
                internal_state=["cunning", "greedy"],
                starting_cash=5000.0,
                target_stock=target,
            )
            pos = (self.rng.integers(0, width), self.rng.integers(0, height))
            self.grid.place_agent(agent, pos)

        # Create Analysts
        specialties = ["technology", "healthcare", "energy"]
        for i in range(n_analysts):
            spec = specialties[i % len(specialties)]
            agent = Analyst(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt=(
                    f"You are a market analyst specializing in {spec}. "
                    "Analyze stock performance and publish research reports "
                    "with honest ratings. Your reputation depends on accuracy. "
                    "Use tools for all actions."
                ),
                vision=vision,
                internal_state=["analytical", "objective"],
                specialty=spec,
            )
            pos = (self.rng.integers(0, width), self.rng.integers(0, height))
            self.grid.place_agent(agent, pos)

        # Create Market Makers
        for _ in range(n_makers):
            agent = MarketMaker(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt=(
                    "You are a market maker providing liquidity. "
                    "Monitor stocks for extreme volatility and provide "
                    "stabilizing liquidity. Calm panicking traders. "
                    "Use tools for all actions."
                ),
                vision=vision,
                internal_state=["stable", "cautious"],
            )
            pos = (self.rng.integers(0, width), self.rng.integers(0, height))
            self.grid.place_agent(agent, pos)

        # Data Collector
        self.datacollector = DataCollector(
            model_reporters={
                "Avg_Price": avg_stock_price,
                "Market_Volume": total_market_volume,
                "Total_Rumors": total_rumors,
                "Manipulation_Rumors": manipulation_rumors,
                "Volatility_%": market_volatility,
                "Insider_Total_Wealth": insider_profit,
                "Bubble_Detected": detect_bubble,
            }
        )

    def next_rumor_id(self) -> int:
        self._rumor_counter += 1
        return self._rumor_counter

    def next_report_id(self) -> int:
        self._report_counter += 1
        return self._report_counter

    def _update_prices(self):
        """Update stock prices based on supply, demand, and randomness."""
        for _ticker, data in self.stocks.items():
            demand = data["demand"]
            supply = data["supply"]
            volume = demand + supply
            data["volume"] = volume

            # Price impact from supply/demand imbalance
            if volume > 0:
                imbalance = (demand - supply) / volume
                price_impact = imbalance * data["base_volatility"] * data["price"]
            else:
                price_impact = 0

            # Random walk component
            random_change = self.rng.normal(0, data["base_volatility"]) * data["price"]

            # Update price (minimum $1)
            new_price = max(1.0, data["price"] + price_impact + random_change)
            data["price"] = round(new_price, 2)
            data["price_history"].append(data["price"])

            # Reset supply/demand for next step
            data["demand"] = 0
            data["supply"] = 0

    def step(self):
        self.datacollector.collect(self)

        # Print market status
        bubble = detect_bubble(self)
        bubble_text = " [bold red blink]BUBBLE![/bold red blink]" if bubble else ""
        print(
            f"\n[bold blue]$$$ STEP {self.steps} "
            f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$${bubble_text}[/bold blue]"
        )
        for _ticker, data in self.stocks.items():
            prev = (
                data["price_history"][-2]
                if len(data["price_history"]) > 1
                else data["price"]
            )
            change = ((data["price"] - prev) / prev * 100) if prev > 0 else 0
            color = "green" if change >= 0 else "red"
            print(
                f"  [{color}]{ticker}: ${data['price']:.2f} ({change:+.1f}%)[/{color}]"
            )

        # Agents act
        self.agents.shuffle_do("step")

        # Update prices based on trading activity
        self._update_prices()


if __name__ == "__main__":
    """
    Run without visualization:
    python -m examples.stock_market.model
    """

    model = StockMarketModel(
        n_retail=6,
        n_insiders=1,
        n_analysts=1,
        n_makers=1,
        width=8,
        height=8,
        reasoning=Reasoning,
        llm_model="gpt-4o-mini",
        vision=2,
    )

    for _ in range(5):
        model.step()

    print("\n[bold cyan]$$$ FINAL MARKET STATE $$$[/bold cyan]")
    for ticker, data in model.stocks.items():
        start = data["price_history"][0]
        end = data["price"]
        total_change = ((end - start) / start * 100) if start > 0 else 0
        color = "green" if total_change >= 0 else "red"
        print(
            f"  [{color}]{ticker}: ${start:.2f} -> ${end:.2f} "
            f"({total_change:+.1f}%)[/{color}]"
        )

    print(
        f"\n  Rumors: {len(model.rumors)} "
        f"({sum(1 for r in model.rumors.values() if r['is_manipulation'])} manipulative)"
    )
    print(f"  Reports: {len(model.reports)}")
