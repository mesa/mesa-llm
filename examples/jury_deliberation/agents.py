from mesa.agent import Agent

from examples.jury_deliberation.case_data import get_case_brief
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_manager import ToolManager

juror_tool_manager = ToolManager()

# 12 juror personas with distinct backgrounds and reasoning tendencies
JUROR_PERSONAS = [
    {
        "name": "Linda Park",
        "occupation": "Retired Engineer",
        "traits": "analytical, methodical, demands hard evidence before deciding",
    },
    {
        "name": "James Whitfield",
        "occupation": "High School Teacher",
        "traits": "patient, empathetic, considers human circumstances",
    },
    {
        "name": "Rosa Gutierrez",
        "occupation": "Nurse",
        "traits": "compassionate but practical, trusts expert testimony",
    },
    {
        "name": "Derek Thompson",
        "occupation": "Small Business Owner",
        "traits": "blunt, results-oriented, skeptical of excuses",
    },
    {
        "name": "Aisha Patel",
        "occupation": "College Student",
        "traits": "idealistic, values fairness, questions authority",
    },
    {
        "name": "Frank Morrison",
        "occupation": "Former Military",
        "traits": "disciplined, values order, respects the legal process",
    },
    {
        "name": "Diane Kowalski",
        "occupation": "Social Worker",
        "traits": "understanding of difficult backgrounds, wary of systemic bias",
    },
    {
        "name": "Robert Chen",
        "occupation": "Accountant",
        "traits": "detail-oriented, follows logical chains, dislikes speculation",
    },
    {
        "name": "Carmen Reyes",
        "occupation": "Restaurant Manager",
        "traits": "street-smart, reads people well, trusts gut instinct",
    },
    {
        "name": "William Hayes",
        "occupation": "Retired Police Officer",
        "traits": "experienced with criminal cases, trusts law enforcement procedures",
    },
    {
        "name": "Megan O'Brien",
        "occupation": "Freelance Artist",
        "traits": "open-minded, emotionally perceptive, dislikes rigid thinking",
    },
    {
        "name": "Howard Kim",
        "occupation": "Pharmacist",
        "traits": "cautious, evidence-driven, uncomfortable with uncertainty",
    },
]


def _build_system_prompt(persona):
    case_brief = get_case_brief()
    return (
        f"You are {persona['name']}, a {persona['occupation']} serving on a jury. "
        f"Your personality: {persona['traits']}.\n\n"
        "You are deliberating with 11 other jurors. You should argue your position, "
        "listen to others, and be willing to change your mind if persuaded — but stay "
        "true to your personality.\n\n"
        f"CASE INFORMATION:\n{case_brief}\n\n"
        "Keep your statements concise (2-3 sentences max). "
        "Use the review_evidence tool if you want to examine specific evidence. "
        "Use the speak_to_room tool to share your argument with the jury."
    )


def get_recent_discussion(model, max_statements=6):
    """Pull the last few statements from the model's discussion log."""
    recent = model.discussion_log[-max_statements:]
    if not recent:
        return "No discussion yet."
    lines = []
    for entry in recent:
        lines.append(f"{entry['name']}: {entry['statement']}")
    return "\n".join(lines)


class ForepersonAgent(Agent):
    """Manages deliberation flow — selects speakers, calls votes, tracks rounds.

    This is a rule-based agent, not LLM-powered.
    """

    def __init__(self, model):
        super().__init__(model=model)
        self.rounds_since_vote = 0
        self.speaker_history = []
        # LLMAgents expect all observable agents to have internal_state
        self.internal_state = []

    def select_speakers(self, jurors, num_speakers=3):
        """Pick jurors to speak this round, favoring those who haven't spoken recently
        and those who disagree with the majority."""

        # figure out current majority leaning
        beliefs = [j.guilt_belief for j in jurors]
        avg_belief = sum(beliefs) / len(beliefs) if beliefs else 0.5

        scored = []
        for juror in jurors:
            score = 0.0

            # boost jurors who haven't spoken recently
            rounds_silent = 0
            for past_id in reversed(self.speaker_history):
                if past_id == juror.unique_id:
                    break
                rounds_silent += 1
            else:
                rounds_silent = len(self.speaker_history)
            score += min(rounds_silent, 5) * 2.0

            # boost jurors who disagree with the majority
            disagreement = abs(juror.guilt_belief - avg_belief)
            score += disagreement * 3.0

            # small random factor so it doesn't feel deterministic
            score += self.model.random.random() * 1.5
            scored.append((score, juror))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [j for _, j in scored[:num_speakers]]

        # update history
        for j in selected:
            self.speaker_history.append(j.unique_id)
        # keep history from growing forever
        if len(self.speaker_history) > len(jurors) * 3:
            self.speaker_history = self.speaker_history[-len(jurors) * 2 :]

        return selected

    def should_call_vote(self):
        """Call a vote every 3 rounds of discussion."""
        self.rounds_since_vote += 1
        if self.rounds_since_vote >= 3:
            self.rounds_since_vote = 0
            return True
        return False


class JurorAgent(LLMAgent):
    def __init__(self, model, reasoning, llm_model, persona, vision=-1):
        system_prompt = _build_system_prompt(persona)
        internal_state = [
            f"name: {persona['name']}",
            f"occupation: {persona['occupation']}",
            f"traits: {persona['traits']}",
        ]

        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )

        self.tool_manager = juror_tool_manager
        self.persona = persona
        self.guilt_belief = 0.5  # start neutral
        self.vote = "undecided"
        self.has_spoken_this_round = False

    def __repr__(self):
        return f"JurorAgent({self.persona['name']})"

    def build_prompt(self):
        # cap at 6 recent statements to keep token usage under control
        discussion = get_recent_discussion(self.model, max_statements=6)
        return (
            f"RECENT DISCUSSION:\n{discussion}\n\n"
            f"Your current belief about guilt: {self.guilt_belief:.1f} "
            "(0=innocent, 1=guilty)\n\n"
            "Based on the discussion so far and the evidence, make your argument. "
            "You can review specific evidence with review_evidence if needed. "
            "Then speak to the room with your position."
        )

    def step(self):
        if self.has_spoken_this_round:
            return

        prompt = self.build_prompt()
        observation = self.generate_obs()
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["speak_to_room", "review_evidence"],
        )
        self.apply_plan(plan)
        self.has_spoken_this_round = True

    async def astep(self):
        if self.has_spoken_this_round:
            return

        prompt = self.build_prompt()
        observation = await self.agenerate_obs()
        plan = await self.reasoning.aplan(
            prompt=prompt,
            obs=observation,
            selected_tools=["speak_to_room", "review_evidence"],
        )
        self.apply_plan(plan)
        self.has_spoken_this_round = True

    def update_belief(self, persuasion_direction):
        """Shift belief based on what was argued. persuasion_direction is positive
        for guilt arguments, negative for innocence arguments."""
        shift = persuasion_direction * 0.1
        # majority pressure — nudge slightly toward where most jurors lean
        all_jurors = [a for a in self.model.agents if isinstance(a, JurorAgent)]
        avg = sum(j.guilt_belief for j in all_jurors) / len(all_jurors)
        conformity_nudge = (avg - self.guilt_belief) * 0.05
        self.guilt_belief = max(
            0.0, min(1.0, self.guilt_belief + shift + conformity_nudge)
        )

    def cast_formal_vote(self):
        """Called during voting rounds. Updates vote based on current belief."""
        if self.guilt_belief >= 0.55:
            self.vote = "guilty"
        elif self.guilt_belief <= 0.45:
            self.vote = "not_guilty"
        else:
            self.vote = "undecided"
        return self.vote

    def reset_round(self):
        self.has_spoken_this_round = False
