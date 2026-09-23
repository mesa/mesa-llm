import sys
import traceback

from agents import TraderAgent
from mesa import Model


class MarketPanicModel(Model):
    def __init__(self, num_agents=4, llm_model="deepseek/deepseek-chat", **kwargs):
        print("MarketPanicModel __init__", file=sys.stderr)

        try:
            super().__init__(**kwargs)
            self.running = True

            print(f"Building {num_agents} agents...", file=sys.stderr)
            for _ in range(num_agents):
                TraderAgent(model=self, llm_model=llm_model)

        except Exception as e:
            print("Error", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise e

    def step(self):
        self.agents.shuffle_do("step")
