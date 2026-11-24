import examples.stock_market.tools  # noqa: F401, to register tools
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_manager import ToolManager

trader_tool_manager = ToolManager()
analyst_tool_manager = ToolManager()


def get_trading_history(agent, max_messages: int = 5) -> str:
    history = []
    memory_source = None
    if hasattr(agent.memory, "short_term_memory"):
        memory_source = agent.memory.short_term_memory
    elif hasattr(agent.memory, "memory_entries"):
        memory_source = agent.memory.memory_entries

    if memory_source:
        entries_to_check = min(len(memory_source), max_messages * 2)
        for entry in reversed(list(memory_source)[-entries_to_check:]):
            if len(history) >= max_messages:
                break
            if isinstance(entry.content, dict) and "message" in entry.content:
                sender = entry.content.get("sender", "Unknown")
                msg = entry.content.get("message", "")
                if hasattr(sender, "unique_id"):
                    sender_name = f"{type(sender).__name__} {sender.unique_id}"
                elif isinstance(sender, int):
                    try:
                        agent_obj = next(
                            a for a in agent.model.agents if a.unique_id == sender
                        )
                        sender_name = f"{type(agent_obj).__name__} {sender}"
                    except StopIteration:
                        sender_name = f"Agent {sender}"
                else:
                    sender_name = str(sender)
                history.append(f"- {sender_name}: {msg}")

    history.reverse()
    return "\n".join(history) if history else "No recent activity."


class TraderAgent(LLMAgent):
    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        budget,
        api_base=None,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            api_base=api_base,
            vision=vision,
            internal_state=internal_state,
        )
        self.tool_manager = trader_tool_manager
        self.budget = budget
        self.shares = 0
        self.trades = 0

    def step(self):
        observation = self.generate_obs()
        history = get_trading_history(self)
        price = self.model.current_price
        prompt = (
            f"MARKET DATA:\n"
            f"- Price: ${price:.2f}\n"
            f"- Trend: {self.model.price_trend()}\n"
            f"- RSI: {self.model.rsi():.1f} (>70 overbought, <30 oversold)\n"
            f"- Budget: ${self.budget:.2f} | Shares: {self.shares}\n\n"
            f"RECENT ACTIVITY:\n{history}\n\n"
            "Use execute_trade to BUY, SELL, or HOLD. Justify briefly."
        )
        plan = self.reasoning.plan(
            prompt=prompt, obs=observation, selected_tools=["execute_trade", "speak_to"]
        )
        self.apply_plan(plan)

    async def astep(self):
        observation = self.generate_obs()
        history = get_trading_history(self)
        price = self.model.current_price
        prompt = (
            f"MARKET DATA:\n"
            f"- Price: ${price:.2f}\n"
            f"- Trend: {self.model.price_trend()}\n"
            f"- RSI: {self.model.rsi():.1f} (>70 overbought, <30 oversold)\n"
            f"- Budget: ${self.budget:.2f} | Shares: {self.shares}\n\n"
            f"RECENT ACTIVITY:\n{history}\n\n"
            "Use execute_trade to BUY, SELL, or HOLD. Justify briefly."
        )
        plan = await self.reasoning.aplan(
            prompt=prompt, obs=observation, selected_tools=["execute_trade", "speak_to"]
        )
        self.apply_plan(plan)


class AnalystAgent(LLMAgent):
    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        api_base=None,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            api_base=api_base,
            vision=vision,
            internal_state=internal_state,
        )
        self.tool_manager = analyst_tool_manager
        self.recommendations_sent = 0

    def step(self):
        observation = self.generate_obs()
        prompt = (
            f"MARKET SUMMARY:\n"
            f"- Price: ${self.model.current_price:.2f}\n"
            f"- Trend: {self.model.price_trend()}\n"
            f"- RSI: {self.model.rsi():.1f}\n"
            f"- Volatility: {self.model.volatility():.4f}\n\n"
            "Broadcast a BUY/HOLD/SELL signal with brief reasoning to nearby traders using speak_to."
        )
        plan = self.reasoning.plan(
            prompt=prompt, obs=observation, selected_tools=["speak_to"]
        )
        self.apply_plan(plan)

    async def astep(self):
        observation = self.generate_obs()
        prompt = (
            f"MARKET SUMMARY:\n"
            f"- Price: ${self.model.current_price:.2f}\n"
            f"- Trend: {self.model.price_trend()}\n"
            f"- RSI: {self.model.rsi():.1f}\n"
            f"- Volatility: {self.model.volatility():.4f}\n\n"
            "Broadcast a BUY/HOLD/SELL signal with brief reasoning to nearby traders using speak_to."
        )
        plan = await self.reasoning.aplan(
            prompt=prompt, obs=observation, selected_tools=["speak_to"]
        )
        self.apply_plan(plan)
