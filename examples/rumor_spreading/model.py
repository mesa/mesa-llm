from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print

from examples.rumor_spreading.agents import BeliefState, CitizenAgent, JournalistAgent
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.record_model import record_model

# Personality pool — citizens are created round-robin from this list
# so every run gets a natural mix of personality types.
_PERSONALITIES = ["credulous", "skeptic", "critical_thinker", "conformist"]

# Two journalists, one on each side of the grid. Biases differ so their
# stories compete — biased journalist vs. objective journalist.
_JOURNALIST_SETUPS = [
    {"bias": "biased", "position_fn": lambda w, h: (w // 4, h // 4)},
    {"bias": "objective", "position_fn": lambda w, h: (3 * w // 4, 3 * h // 4)},
]


@record_model(output_dir="recordings")
class RumorSpreadingModel(Model):
    """
    A model where journalists publish stories and citizens decide whether
    to believe, spread, doubt, or debunk what they hear.

    The interesting tension:
        - A biased journalist seeds misinformation from one corner.
        - An objective journalist publishes verified news from another.
        - Citizens with different personalities process information differently.
        - Believers spread via speak_to; skeptics and critical thinkers push back.

    Watch the chart: will misinformation cascade, or will skeptics slow it down?

    Args:
        initial_citizens:    How many citizens to populate the grid with.
        initial_journalists: How many journalists to place (max 2, one per corner).
        width:               Grid width.
        height:              Grid height.
        reasoning:           Reasoning class (e.g. ReActReasoning).
        llm_model:           LLM backend string, e.g. "openai/gpt-4o-mini".
        vision:              How many cells an agent can see in every direction.
        parallel_stepping:   Run agent steps concurrently (faster but needs threading).
        seed:                Random seed for reproducibility.
    """

    def __init__(
        self,
        initial_citizens: int,
        initial_journalists: int,
        width: int,
        height: int,
        reasoning: type[Reasoning],
        llm_model: str,
        vision: int,
        parallel_stepping: bool = True,
        seed=None,
        api_base: str | None = None,
    ):
        super().__init__(seed=seed)

        # Remote LLM endpoint (e.g. remote Ollama URL)
        self.api_base = api_base

        self.width = width
        self.height = height
        self.parallel_stepping = parallel_stepping
        self.grid = MultiGrid(self.width, self.height, torus=False)

        # Data collection — the chart will show how belief spreads over time
        model_reporters = {
            "Uninformed": lambda m: sum(
                1
                for a in m.agents
                if isinstance(a, CitizenAgent)
                and a.belief_state == BeliefState.UNINFORMED
            ),
            "Believers": lambda m: sum(
                1
                for a in m.agents
                if isinstance(a, CitizenAgent)
                and a.belief_state == BeliefState.BELIEVER
            ),
            "Skeptics": lambda m: sum(
                1
                for a in m.agents
                if isinstance(a, CitizenAgent) and a.belief_state == BeliefState.SKEPTIC
            ),
            "Debunked": lambda m: sum(
                1
                for a in m.agents
                if isinstance(a, CitizenAgent)
                and a.belief_state == BeliefState.DEBUNKED
            ),
            "Stories_Published": lambda m: sum(
                a.stories_published for a in m.agents if isinstance(a, JournalistAgent)
            ),
        }

        agent_reporters = {
            "belief_state": lambda a: getattr(a, "belief_state", None),
            "personality": lambda a: getattr(a, "personality", None),
        }

        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
        )

        # ------------------------------------------------------------------
        # Citizens — scattered across the grid with varied personalities.
        # Citizens are placed FIRST so that journalists, placed afterwards,
        # are always the last entry in any shared cell. Mesa's coord_iter()
        # renders agents in insertion order, so journalists (placed last)
        # are drawn on top and never hidden by citizen dots.
        # ------------------------------------------------------------------
        citizen_system_prompt = (
            "You are a person living in a community. "
            "You move around the neighbourhood, talk to people, and hear stories. "
            "How you react to information depends on who you are — your personality. "
            "You have three tools: move_one_step to explore, speak_to to share what "
            "you heard, and update_belief to record your personal stance on a rumor."
        )

        # Collect journalist positions upfront so we can keep citizens away
        # from those cells entirely — avoids overlap in the first place.
        n_journalists = min(initial_journalists, len(_JOURNALIST_SETUPS))
        journalist_positions = {
            setup["position_fn"](self.width, self.height)
            for setup in _JOURNALIST_SETUPS[:n_journalists]
        }

        placed = 0
        attempts = 0
        max_attempts = initial_citizens * 20
        while placed < initial_citizens and attempts < max_attempts:
            attempts += 1
            x = int(self.rng.integers(0, self.width))
            y = int(self.rng.integers(0, self.height))
            # Keep journalist cells clear AND avoid stacking citizens on the same cell
            if (x, y) in journalist_positions or not self.grid.is_cell_empty((x, y)):
                continue

            personality = _PERSONALITIES[placed % len(_PERSONALITIES)]
            citizen = CitizenAgent(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt=citizen_system_prompt,
                vision=vision,
                internal_state=[],
                step_prompt="Think about what you've heard. Believe it, question it, or debunk it.",
                personality=personality,
                api_base=api_base,
            )
            self.grid.place_agent(citizen, (x, y))
            placed += 1

        # Journalists — placed LAST so they render on top of any co-located
        # agent when Mesa's visualization iterates the cell list in order.
        for setup in _JOURNALIST_SETUPS[:n_journalists]:
            bias = setup["bias"]
            pos = setup["position_fn"](self.width, self.height)

            system_prompt = (
                f"You are a {bias} journalist working for a local news outlet. "
                "You sit at your desk and publish stories to inform — or mislead — "
                "the community around you. Use publish_story every single turn."
            )

            journalist = JournalistAgent(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt=system_prompt,
                vision=vision,
                internal_state=[],
                step_prompt="Write and publish a news story about something happening in the community.",
                bias=bias,
                api_base=api_base,
            )
            self.grid.place_agent(journalist, pos)

        # Collect initial state (Step 0: all citizens uninformed, no stories published yet)
        self.datacollector.collect(self)

    # Model step

    def step(self):
        print(
            f"\n[bold purple] step  {self.steps} "
            "────────────────────────────────────────────────────────────────"
            "[/bold purple]"
        )

        # Journalists go first so their stories land in citizen memory
        # before citizens decide what to do this turn.
        journalists = [a for a in self.agents if isinstance(a, JournalistAgent)]
        citizens = [a for a in self.agents if isinstance(a, CitizenAgent)]

        for journalist in journalists:
            try:
                journalist.step()
            except Exception as e:
                print(
                    f"[yellow]  Journalist {journalist.unique_id} step failed: {e}[/yellow]"
                )

        self.random.shuffle(citizens)
        for citizen in citizens:
            try:
                citizen.step()
            except Exception as e:
                print(
                    f"[yellow]  Citizen {citizen.unique_id} step failed: {e}[/yellow]"
                )

        self.datacollector.collect(self)
