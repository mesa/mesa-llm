"""
Stock Market with Rumors - Agent Definitions

A simulation of how information asymmetry and rumors affect
stock markets. LLM-powered traders independently decide when
to buy, sell, or hold based on news, rumors, and analysis.

Agent Types:
- RetailTrader: Regular investor making decisions from public info
- InsiderTrader: Plants rumors to manipulate prices for profit
- Analyst: Publishes research reports influencing market sentiment
- MarketMaker: Provides liquidity and stabilizes extreme moves

Author: Adarsh Kumar (GSoC 2026)
"""

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_decorator import tool
from mesa_llm.tools.tool_manager import ToolManager

retail_tools = ToolManager()
insider_tools = ToolManager()
analyst_tools = ToolManager()
maker_tools = ToolManager()


@retail_tools.register
@tool
def place_order(agent: "RetailTrader", stock: str, action: str, quantity: int) -> str:
    """
    Place a buy or sell order for a stock.

    Args:
        stock: The stock ticker symbol (e.g. 'TECH', 'HEALTH', 'ENERGY')
        action: Either 'buy' or 'sell'
        quantity: Number of shares to trade (1-100)
        agent: Provided automatically
    """
    action = action.lower().strip()
    if action not in ("buy", "sell"):
        return f"Invalid action '{action}'. Must be 'buy' or 'sell'."

    stock = stock.upper().strip()
    if stock not in agent.model.stocks:
        return (
            f"Stock '{stock}' not found. Available: {list(agent.model.stocks.keys())}"
        )

    quantity = max(1, min(100, quantity))
    price = agent.model.stocks[stock]["price"]
    total_cost = price * quantity

    if action == "buy":
        if agent.cash < total_cost:
            return (
                f"Insufficient funds. Need ${total_cost:.2f} but have ${agent.cash:.2f}"
            )
        agent.cash -= total_cost
        agent.portfolio[stock] = agent.portfolio.get(stock, 0) + quantity
        agent.model.stocks[stock]["demand"] += quantity
        agent.trades.append(
            {
                "action": "buy",
                "stock": stock,
                "qty": quantity,
                "price": price,
                "step": agent.model.steps,
            }
        )
        return (
            f"Agent {agent.unique_id} BOUGHT {quantity} {stock} @ ${price:.2f} "
            f"(total: ${total_cost:.2f}). Cash remaining: ${agent.cash:.2f}"
        )
    else:
        held = agent.portfolio.get(stock, 0)
        if held < quantity:
            return f"Cannot sell {quantity} {stock}. Only holding {held} shares."
        agent.cash += total_cost
        agent.portfolio[stock] -= quantity
        agent.model.stocks[stock]["supply"] += quantity
        agent.trades.append(
            {
                "action": "sell",
                "stock": stock,
                "qty": quantity,
                "price": price,
                "step": agent.model.steps,
            }
        )
        return (
            f"Agent {agent.unique_id} SOLD {quantity} {stock} @ ${price:.2f} "
            f"(earned: ${total_cost:.2f}). Cash: ${agent.cash:.2f}"
        )


@retail_tools.register
@tool
def check_portfolio(agent: "RetailTrader") -> str:
    """
    Check your current portfolio value and holdings.

    Args:
        agent: Provided automatically
    """
    lines = [f"Cash: ${agent.cash:.2f}"]
    total_value = agent.cash
    for stock, qty in agent.portfolio.items():
        if qty > 0:
            price = agent.model.stocks[stock]["price"]
            value = price * qty
            total_value += value
            lines.append(f"  {stock}: {qty} shares @ ${price:.2f} = ${value:.2f}")
    lines.append(f"Total Portfolio Value: ${total_value:.2f}")
    return "\n".join(lines)


@retail_tools.register
@tool
def check_market(agent: "RetailTrader") -> str:
    """
    Check current stock prices, trends, and recent activity.

    Args:
        agent: Provided automatically
    """
    lines = []
    for ticker, data in agent.model.stocks.items():
        price = data["price"]
        prev = data["price_history"][-2] if len(data["price_history"]) > 1 else price
        change = ((price - prev) / prev * 100) if prev > 0 else 0
        direction = "\u2191" if change > 0 else ("\u2193" if change < 0 else "\u2192")
        lines.append(
            f"  {ticker}: ${price:.2f} {direction} ({change:+.1f}%) "
            f"| Volume: {data['volume']}"
        )
    return "MARKET STATUS:\n" + "\n".join(lines)


@insider_tools.register
@tool
def spread_rumor(agent: "InsiderTrader", stock: str, rumor: str) -> str:
    """
    Spread a market rumor to nearby traders to influence stock prices.

    Args:
        stock: The stock ticker the rumor is about
        rumor: The rumor content (can be true, exaggerated, or false)
        agent: Provided automatically
    """
    stock = stock.upper().strip()
    rumor_id = agent.model.next_rumor_id()
    agent.model.rumors[rumor_id] = {
        "id": rumor_id,
        "stock": stock,
        "content": rumor,
        "source": agent.unique_id,
        "is_manipulation": True,
        "spread_count": 0,
        "step_created": agent.model.steps,
    }
    agent.rumors_spread += 1

    neighbors = agent.model.grid.get_neighbors(
        agent.pos, moore=True, include_center=False, radius=agent.vision
    )
    recipient_ids = []
    for neighbor in neighbors:
        if hasattr(neighbor, "receive_rumor"):
            neighbor.receive_rumor(rumor_id, agent.unique_id)
            recipient_ids.append(neighbor.unique_id)

    return (
        f"Agent {agent.unique_id} spread rumor about {stock} to {recipient_ids}: "
        f"'{rumor}'"
    )


# Register place_order and check tools for insider too
@insider_tools.register
@tool
def insider_place_order(
    agent: "InsiderTrader", stock: str, action: str, quantity: int
) -> str:
    """
    Place a buy or sell order using insider knowledge.

    Args:
        stock: The stock ticker symbol
        action: Either 'buy' or 'sell'
        quantity: Number of shares (1-100)
        agent: Provided automatically
    """
    action = action.lower().strip()
    if action not in ("buy", "sell"):
        return f"Invalid action '{action}'."
    stock = stock.upper().strip()
    if stock not in agent.model.stocks:
        return f"Stock '{stock}' not found."
    quantity = max(1, min(100, quantity))
    price = agent.model.stocks[stock]["price"]
    total_cost = price * quantity

    if action == "buy":
        if agent.cash < total_cost:
            return "Insufficient funds."
        agent.cash -= total_cost
        agent.portfolio[stock] = agent.portfolio.get(stock, 0) + quantity
        agent.model.stocks[stock]["demand"] += quantity
        agent.trades.append(
            {
                "action": "buy",
                "stock": stock,
                "qty": quantity,
                "price": price,
                "step": agent.model.steps,
            }
        )
        return f"Insider {agent.unique_id} BOUGHT {quantity} {stock} @ ${price:.2f}"
    else:
        held = agent.portfolio.get(stock, 0)
        if held < quantity:
            return f"Cannot sell {quantity} {stock}. Holding {held}."
        agent.cash += total_cost
        agent.portfolio[stock] -= quantity
        agent.model.stocks[stock]["supply"] += quantity
        agent.trades.append(
            {
                "action": "sell",
                "stock": stock,
                "qty": quantity,
                "price": price,
                "step": agent.model.steps,
            }
        )
        return f"Insider {agent.unique_id} SOLD {quantity} {stock} @ ${price:.2f}"


@analyst_tools.register
@tool
def publish_report(agent: "Analyst", stock: str, rating: str, analysis: str) -> str:
    """
    Publish a research report with a rating for a stock.

    Args:
        stock: The stock ticker to analyze
        rating: Your rating - 'strong_buy', 'buy', 'hold', 'sell', or 'strong_sell'
        analysis: Your detailed analysis and reasoning
        agent: Provided automatically
    """
    stock = stock.upper().strip()
    valid_ratings = ["strong_buy", "buy", "hold", "sell", "strong_sell"]
    if rating.lower() not in valid_ratings:
        return f"Invalid rating. Use one of: {valid_ratings}"

    report_id = agent.model.next_report_id()
    agent.model.reports[report_id] = {
        "id": report_id,
        "stock": stock,
        "rating": rating.lower(),
        "analysis": analysis,
        "author": agent.unique_id,
        "step": agent.model.steps,
    }
    agent.reports_published += 1

    neighbors = agent.model.grid.get_neighbors(
        agent.pos, moore=True, include_center=False, radius=agent.vision
    )
    for neighbor in neighbors:
        if hasattr(neighbor, "receive_report"):
            neighbor.receive_report(report_id, agent.unique_id)

    return (
        f"Analyst {agent.unique_id} published {rating.upper()} on {stock}: "
        f"'{analysis[:80]}...'"
    )


@maker_tools.register
@tool
def provide_liquidity(agent: "MarketMaker", stock: str, quantity: int) -> str:
    """
    Provide liquidity by placing balanced buy and sell orders to stabilize price.

    Args:
        stock: The stock ticker
        quantity: Shares to offer on each side
        agent: Provided automatically
    """
    stock = stock.upper().strip()
    if stock not in agent.model.stocks:
        return f"Stock '{stock}' not found."

    agent.model.stocks[stock]["demand"] += quantity
    agent.model.stocks[stock]["supply"] += quantity
    agent.liquidity_provided += quantity * 2

    return (
        f"MarketMaker {agent.unique_id} provided {quantity} shares liquidity "
        f"on each side for {stock}"
    )


def get_rumor_feed(agent) -> str:
    if not agent.rumor_inbox:
        return "No new rumors."
    lines = []
    for item in agent.rumor_inbox[-5:]:
        rumor = agent.model.rumors.get(item["rumor_id"])
        if rumor:
            lines.append(
                f"- Rumor about {rumor['stock']}: '{rumor['content']}' "
                f"(from Agent {item['from']})"
            )
    return "\n".join(lines) if lines else "No new rumors."


def get_report_feed(agent) -> str:
    if not agent.report_inbox:
        return "No new analyst reports."
    lines = []
    for item in agent.report_inbox[-5:]:
        report = agent.model.reports.get(item["report_id"])
        if report:
            lines.append(
                f"- {report['rating'].upper()} on {report['stock']}: "
                f"'{report['analysis'][:60]}...' (Analyst {report['author']})"
            )
    return "\n".join(lines) if lines else "No new analyst reports."


class RetailTrader(LLMAgent):
    """A regular investor who trades based on public info and rumors."""

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        starting_cash,
        risk_tolerance,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )
        self.tool_manager = retail_tools
        self.cash = starting_cash
        self.portfolio = {}
        self.trades = []
        self.risk_tolerance = risk_tolerance
        self.rumor_inbox = []
        self.report_inbox = []

    def receive_rumor(self, rumor_id, from_agent_id):
        self.rumor_inbox.append({"rumor_id": rumor_id, "from": from_agent_id})

    def receive_report(self, report_id, from_agent_id):
        self.report_inbox.append({"report_id": report_id, "from": from_agent_id})

    def step(self):
        observation = self.generate_obs()
        rumors = get_rumor_feed(self)
        reports = get_report_feed(self)

        prompt = (
            f"MARKET RUMORS:\n{rumors}\n\n"
            f"ANALYST REPORTS:\n{reports}\n\n"
            f"YOUR RISK TOLERANCE: {self.risk_tolerance}\n"
            f"YOUR CASH: ${self.cash:.2f}\n"
            f"YOUR HOLDINGS: {self.portfolio}\n\n"
            "INSTRUCTIONS:\n"
            "1. Use check_market to see current prices and trends.\n"
            "2. Consider rumors and analyst reports but be cautious.\n"
            "3. Use place_order to buy or sell stocks.\n"
            "4. Use check_portfolio to review your position.\n"
            "5. Move around to hear more rumors and find information."
        )
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=[
                "place_order",
                "check_portfolio",
                "check_market",
                "speak_to",
                "move_one_step",
            ],
        )
        self.apply_plan(plan)
        self.rumor_inbox.clear()
        self.report_inbox.clear()


class InsiderTrader(LLMAgent):
    """A trader with secret info who manipulates prices through rumors."""

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        starting_cash,
        target_stock,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )
        self.tool_manager = insider_tools
        self.cash = starting_cash
        self.portfolio = {}
        self.trades = []
        self.target_stock = target_stock
        self.rumors_spread = 0
        self.rumor_inbox = []
        self.report_inbox = []

    def receive_rumor(self, rumor_id, from_agent_id):
        self.rumor_inbox.append({"rumor_id": rumor_id, "from": from_agent_id})

    def receive_report(self, report_id, from_agent_id):
        self.report_inbox.append({"report_id": report_id, "from": from_agent_id})

    def step(self):
        observation = self.generate_obs()
        stock_price = self.model.stocks[self.target_stock]["price"]

        prompt = (
            f"YOUR TARGET STOCK: {self.target_stock} (current: ${stock_price:.2f})\n"
            f"YOUR CASH: ${self.cash:.2f}\n"
            f"YOUR HOLDINGS: {self.portfolio}\n"
            f"RUMORS SPREAD: {self.rumors_spread}\n\n"
            "SECRET STRATEGY:\n"
            "1. Buy your target stock quietly using insider_place_order.\n"
            "2. Then spread positive rumors using spread_rumor to pump the price.\n"
            "3. When price rises enough, sell for profit.\n"
            "4. Be subtle - don't spread too many rumors at once.\n"
            "5. Move around to reach more traders."
        )
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=[
                "insider_place_order",
                "spread_rumor",
                "speak_to",
                "move_one_step",
            ],
        )
        self.apply_plan(plan)
        self.rumor_inbox.clear()
        self.report_inbox.clear()


class Analyst(LLMAgent):
    """Publishes research reports that influence market sentiment."""

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        specialty,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )
        self.tool_manager = analyst_tools
        self.specialty = specialty
        self.reports_published = 0
        self.rumor_inbox = []
        self.report_inbox = []

    def receive_rumor(self, rumor_id, from_agent_id):
        self.rumor_inbox.append({"rumor_id": rumor_id, "from": from_agent_id})

    def receive_report(self, report_id, from_agent_id):
        self.report_inbox.append({"report_id": report_id, "from": from_agent_id})

    def step(self):
        observation = self.generate_obs()
        market_summary = []
        for ticker, data in self.model.stocks.items():
            price = data["price"]
            history = data["price_history"][-5:]
            market_summary.append(f"  {ticker}: ${price:.2f} | Recent: {history}")

        prompt = (
            f"YOUR SPECIALTY: {self.specialty}\n"
            f"REPORTS PUBLISHED: {self.reports_published}\n\n"
            f"MARKET DATA:\n" + "\n".join(market_summary) + "\n\n"
            "INSTRUCTIONS:\n"
            "1. Analyze market trends for stocks in your specialty.\n"
            "2. Publish a research report using publish_report.\n"
            "3. Base ratings on price trends and fundamentals.\n"
            "4. Use speak_to to discuss findings with nearby agents.\n"
            "5. Move around to gather more information."
        )
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["publish_report", "speak_to", "move_one_step"],
        )
        self.apply_plan(plan)
        self.rumor_inbox.clear()
        self.report_inbox.clear()


class MarketMaker(LLMAgent):
    """Provides liquidity and stabilizes extreme price movements."""

    def __init__(
        self, model, reasoning, llm_model, system_prompt, vision, internal_state
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )
        self.tool_manager = maker_tools
        self.liquidity_provided = 0
        self.rumor_inbox = []
        self.report_inbox = []

    def receive_rumor(self, rumor_id, from_agent_id):
        self.rumor_inbox.append({"rumor_id": rumor_id, "from": from_agent_id})

    def receive_report(self, report_id, from_agent_id):
        self.report_inbox.append({"report_id": report_id, "from": from_agent_id})

    def step(self):
        observation = self.generate_obs()
        volatile_stocks = []
        for ticker, data in self.model.stocks.items():
            if len(data["price_history"]) >= 2:
                prev = data["price_history"][-2]
                curr = data["price"]
                change = abs((curr - prev) / prev * 100) if prev > 0 else 0
                if change > 5:
                    volatile_stocks.append(f"  {ticker}: {change:.1f}% move!")

        volatility_text = (
            "\n".join(volatile_stocks)
            if volatile_stocks
            else "  No extreme moves detected."
        )

        prompt = (
            f"VOLATILE STOCKS:\n{volatility_text}\n"
            f"LIQUIDITY PROVIDED: {self.liquidity_provided} shares total\n\n"
            "INSTRUCTIONS:\n"
            "1. Monitor stocks for extreme price movements.\n"
            "2. Use provide_liquidity on volatile stocks to stabilize.\n"
            "3. Use speak_to to calm panicking traders.\n"
            "4. Move around the market floor."
        )
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["provide_liquidity", "speak_to", "move_one_step"],
        )
        self.apply_plan(plan)
        self.rumor_inbox.clear()
        self.report_inbox.clear()
