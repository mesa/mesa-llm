import random

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.reasoning.cot import CoTReasoning


class TraderAgent(LLMAgent):
    def __init__(self, model, llm_model="deepseek/deepseek-chat"):
        super().__init__(
            model=model,
            llm_model=llm_model,
            reasoning=CoTReasoning,
            system_prompt=(
                "You are a quantitative trader in a highly volatile market. "
                "You act purely on market rumors and optimize for survival. "
                "CRITICAL: You exist purely on a trading network, NOT on a physical grid. "
                "NEVER use the 'move_one_step' tool. Your ONLY available tool is 'speak_to'."
            ),
        )

        self.step_prompt = (
            "Analyze the latest market rumors from your memory. "
            "Decide whether to BUY, SELL, or HOLD your tech stocks."
        )

        self.memory = STLTMemory(
            agent=self,
            short_term_capacity=1,
            consolidation_capacity=1,
            llm_model=llm_model,
        )

    def step(self):
        obs = self.generate_obs()

        # Broadcast rumors to neighbors
        neighbors = [a for a in self.model.agents if a != self]
        if neighbors:
            target = random.choice(neighbors)
            uid = getattr(self, "unique_id", "Unknown")
            message = (
                f"URGENT ALERT: Algorithmic sell-off detected in tech sector. "
                f"Liquidate positions immediately! - From Trader {uid}"
            )
            self.send_message(message, recipients=[target])

        plan = self.reasoning.plan(obs=obs)
        self.apply_plan(plan)
