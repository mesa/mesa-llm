import math

from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print

from examples.negotiation.agents import BuyerAgent, SellerAgent
from mesa_llm.reasoning.reasoning import Reasoning


# @record_model
class NegotiationModel(Model):
    """
    A model for a negotiation game between a seller and a buyer.

    Args:
        initial_buyers (int): The number of initial buyers in the model.
        initial_sellers (int): The number of initial sellers in the model.
        width (int): The width of the grid.
        height (int): The height of the grid.
    """

    def __init__(
        self,
        initial_buyers: int,
        width: int,
        height: int,
        reasoning: type[Reasoning],
        llm_model: str,
        vision: int,
        api_base: str | None = None,
        seed=None,
        parallel_stepping=False,
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.parallel_stepping = parallel_stepping
        self.grid = MultiGrid(self.height, self.width, torus=False)

        # ---------------------Create the buyer agents---------------------
        buyer_system_prompt = "You are a buyer in a negotiation game. You are interested in buying a product from a seller. You are also interested in negotiating with the seller. Prefer speaking over changing location as long as you have a seller in sight. If no seller is in sight, move around randomly until yous see one"
        buyer_internal_state = ""

        agents = BuyerAgent.create_agents(
            self,
            n=initial_buyers - math.floor(initial_buyers / 2),
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=buyer_system_prompt,
            vision=vision,
            internal_state=buyer_internal_state,
            budget=50,  # Each buyer has a budget of $50
            api_base=api_base,
        )

        x = self.rng.integers(0, self.grid.width, size=(initial_buyers,))
        y = self.rng.integers(0, self.grid.height, size=(initial_buyers,))
        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

        agents = BuyerAgent.create_agents(
            self,
            n=math.floor(initial_buyers / 2),
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=buyer_system_prompt,
            vision=vision,
            internal_state=buyer_internal_state,
            budget=100,
            api_base=api_base,
        )

        x = self.rng.integers(0, self.grid.width, size=(initial_buyers,))
        y = self.rng.integers(0, self.grid.height, size=(initial_buyers,))
        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

        # ---------------------Create the seller agents---------------------
        seller_a = SellerAgent(
            model=self,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="You are a Seller in a negotiation game trying to sell shoes($40) and track suit($50) of brand A. You are trying to pitch your product to the Buyer type Agents. You are extremely good at persuading, and have good sales skills. You are also hardworking and dedicated to your work. To change simulation state, choose one of the actions provided to you.",
            vision=vision,
            internal_state=["hardworking", "dedicated", "persuasive"],
            api_base=api_base,
        )
        self.grid.place_agent(
            seller_a,
            (math.floor(self.grid.width / 2), math.floor(self.grid.height / 2)),
        )
        self.seller_a = seller_a  # Store reference to seller A

        # Just for testing purposes, we can add more seller agents later
        seller_b = SellerAgent(
            model=self,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="You are a Seller in a negotiation game trying to sell shoes($35) and track suit($47) of brand B. You are trying to pitch your product to the Buyer type Agents. You are not interested in your work and are doing it for the sake of doing. To change simulation state, choose one of the actions provided to you.",
            vision=vision,
            internal_state=["lazy", "unmotivated"],
            api_base=api_base,
        )
        self.grid.place_agent(
            seller_b,
            (math.floor(self.grid.width / 2), math.floor(self.grid.height / 2) + 1),
        )
        self.seller_b = seller_b  # Store reference to seller B

        # Initialize DataCollector to track sales of both sellers
        self.datacollector = DataCollector(
            model_reporters={
                "SellerA_Sales": lambda m: m.seller_a.sales,
                "SellerB_Sales": lambda m: m.seller_b.sales,
            }
        )

    def step(self):
        """
        Execute one step of the model.
        """
        self.datacollector.collect(self)
        print(
            f"\n[bold purple] step  {self.steps} ────────────────────────────────────────────────────────────────────────────────[/bold purple]"
        )
        self.agents.shuffle_do("step")


# ===============================================================
#                     RUN WITHOUT GRAPHICS
# ===============================================================

if __name__ == "__main__":
    """
    run the model without the solara integration with:
    conda activate mesa-llm && python -m examples.negotiation.model
    """

    from examples.negotiation.app import model

    # Run the model for 10 steps
    for _ in range(10):
        model.step()
