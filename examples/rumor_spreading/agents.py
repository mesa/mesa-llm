import contextlib
from enum import Enum

import mesa

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.module_llm import ModuleLLM
from mesa_llm.tools.tool_manager import ToolManager

# Each agent type gets its own tool manager.
# Inbuilt tools (move_one_step, speak_to, teleport_to_location) are automatically
# included because mesa_llm.__init__ registers them before any ToolManager is created.
citizen_tool_manager = ToolManager()
journalist_tool_manager = ToolManager()


class BeliefState(Enum):
    UNINFORMED = "uninformed"
    BELIEVER = "believer"
    SKEPTIC = "skeptic"
    DEBUNKED = "debunked"


class CitizenAgent(LLMAgent, mesa.discrete_space.CellAgent):
    """
    An ordinary person living in the community.

    They wander around, bump into neighbors, and hear things — from journalists,
    from friends, from strangers. Whether they believe what they hear depends
    entirely on who they are as a person.

    Personality options:
        - credulous:        Tends to believe and repeat almost anything they hear.
        - skeptic:          Doubts first, asks questions, rarely spreads unverified info.
        - critical_thinker: Evaluates claims carefully; needs logic and evidence to believe.
        - conformist:       Goes along with whatever the majority around them believes.
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        step_prompt,
        personality: str = "credulous",
        api_base: str | None = None,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
        )

        self.personality = personality
        self.belief_state = BeliefState.UNINFORMED

        # Override the LLM to ensure it hits the configured api_base
        self.llm = ModuleLLM(llm_model=llm_model, api_base=api_base)

        self.memory = STLTMemory(
            agent=self,
            display=True,
            llm_model=llm_model,
            consolidation_capacity=100,  # never consolidate — no extra LLM calls
        )

        # Ensure memory LLM hits same remote endpoint when provided
        if api_base:
            with contextlib.suppress(Exception):
                self.memory.llm.api_base = api_base

        # Attach the citizen-specific tool manager so tools like
        # `update_belief` and `speak_to` are available to the agent.
        self.tool_manager = citizen_tool_manager

        self.internal_state.append(f"My personality is: {self.personality}")
        self.internal_state.append(
            f"My current belief state is: {self.belief_state.value}"
        )

    # Helpers
    def _sync_belief_in_internal_state(self):
        """Keep the internal_state belief line in sync with self.belief_state."""
        self.internal_state = [
            s for s in self.internal_state if "belief state" not in s.lower()
        ]
        self.internal_state.append(
            f"My current belief state is: {self.belief_state.value}"
        )

    def _get_recent_rumors(self) -> str:
        """
        Dig through short-term memory and pull out any messages that were
        sent to this agent — those are the rumors worth thinking about.
        """
        rumors = []

        if hasattr(self.memory, "short_term_memory"):
            # Read the last 8 entries and pick out messages
            for entry in list(self.memory.short_term_memory)[-8:]:
                if isinstance(entry.content, dict) and "message" in entry.content:
                    sender_id = entry.content.get("sender", "Unknown")
                    msg = entry.content.get("message", "")
                    rumors.append(f'  - From Agent {sender_id}: "{msg}"')

        return "\n".join(rumors) if rumors else "  (nothing yet)"

    def _count_believers_nearby(self) -> int:
        """How many neighbors in radius-1 are already believers?"""
        return sum(
            1
            for a in self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False, radius=1
            )
            if isinstance(a, CitizenAgent) and a.belief_state == BeliefState.BELIEVER
        )

    def _get_nearby_citizen_ids(self) -> list[int]:
        """IDs of all citizens within vision that could receive a message."""
        return [
            a.unique_id
            for a in self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False, radius=self.vision
            )
            if isinstance(a, CitizenAgent)
        ]

    # Step
    def _build_step_prompt(self) -> str:
        recent_rumors = self._get_recent_rumors()
        believers_nearby = self._count_believers_nearby()
        nearby_citizen_ids = self._get_nearby_citizen_ids()

        return (
            f"RUMORS HEARD:\n{recent_rumors}\n"
            f"BELIEVERS NEARBY: {believers_nearby} | "
            f"CITIZENS YOU CAN TALK TO: {nearby_citizen_ids}\n"
            f"YOUR PERSONALITY: {self.personality} | "
            f"YOUR BELIEF: {self.belief_state.value}\n\n"
            "Pick ONE action:\n"
            "  - Heard something + believe it → speak_to neighbours, then update_belief BELIEVER\n"
            "  - Doubt it → update_belief SKEPTIC\n"
            "  - Sure it is false → update_belief DEBUNKED\n"
            "  - Nothing heard → move_one_step to explore\n"
        )

    def step(self):
        self._sync_belief_in_internal_state()
        observation = self.generate_obs()
        prompt = self._build_step_prompt()

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["update_belief", "speak_to", "move_one_step"],
        )
        self.apply_plan(plan)

    async def astep(self):
        self._sync_belief_in_internal_state()
        observation = self.generate_obs()
        prompt = self._build_step_prompt()

        plan = await self.reasoning.aplan(
            prompt=prompt,
            obs=observation,
            selected_tools=["update_belief", "speak_to", "move_one_step"],
        )
        self.apply_plan(plan)


class JournalistAgent(LLMAgent, mesa.discrete_space.CellAgent):
    """
    A journalist sitting at a fixed desk, broadcasting stories to the public.

    They never move — their job is to publish. What they publish, and how
    honestly, depends on whether they're biased or objective.

    Bias options:
        - biased:    Sensational, emotionally charged, sometimes misleading.
        - objective: Factual, measured, evidence-based.
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        step_prompt,
        bias: str = "objective",
        api_base: str | None = None,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
        )

        self.bias = bias
        self.stories_published = 0

        # Override the LLM to ensure it hits the configured api_base
        self.llm = ModuleLLM(llm_model=llm_model, api_base=api_base)

        self.memory = STLTMemory(
            agent=self,
            display=True,
            llm_model=llm_model,
            consolidation_capacity=100,  # never consolidate — no extra LLM calls
        )

        if api_base:
            with contextlib.suppress(Exception):
                self.memory.llm.api_base = api_base

        # Attach the journalist-specific tool manager so `publish_story`
        # is available to this agent.
        self.tool_manager = journalist_tool_manager

        self.internal_state.append(f"I am a {self.bias} journalist.")
        self.internal_state.append(
            f"Total stories I have published so far: {self.stories_published}"
        )

    # Helpers

    def _sync_stories_in_internal_state(self):
        self.internal_state = [
            s
            for s in self.internal_state
            if "stories i have published" not in s.lower()
        ]
        self.internal_state.append(
            f"Total stories I have published so far: {self.stories_published}"
        )

    def _get_nearby_citizen_ids(self) -> list[int]:
        return [
            a.unique_id
            for a in self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False, radius=self.vision
            )
            if isinstance(a, CitizenAgent)
        ]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _build_step_prompt(self) -> str:
        nearby_ids = self._get_nearby_citizen_ids()

        return (
            f"CITIZENS IN RANGE: {nearby_ids}\n"
            f"YOUR BIAS: {self.bias}\n\n"
            "Use publish_story. "
            "If biased: write a sensational or misleading headline, set is_misinformation=True. "
            "If objective: write a factual headline, set is_misinformation=False."
        )

    def step(self):
        self._sync_stories_in_internal_state()
        observation = self.generate_obs()
        prompt = self._build_step_prompt()

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["publish_story"],
        )
        self.apply_plan(plan)

    async def astep(self):
        self._sync_stories_in_internal_state()
        observation = self.generate_obs()
        prompt = self._build_step_prompt()

        plan = await self.reasoning.aplan(
            prompt=prompt,
            obs=observation,
            selected_tools=["publish_story"],
        )
        self.apply_plan(plan)
