from mesa.datacollection import DataCollector
from mesa.model import Model

# tools must be imported so they register with the juror_tool_manager
import examples.jury_deliberation.tools  # noqa: F401
from examples.jury_deliberation.agents import (
    JUROR_PERSONAS,
    ForepersonAgent,
    JurorAgent,
)
from mesa_llm.reasoning.reasoning import Reasoning


class JuryDeliberationModel(Model):
    """Jury deliberation model without spatial grid."""

    def __init__(
        self,
        reasoning: type[Reasoning],
        llm_model: str,
        num_speakers: int = 1,
        max_rounds: int = 15,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.num_speakers = num_speakers
        self.max_rounds = max_rounds
        self.current_round = 0
        self.verdict = None

        # shared deliberation state
        self.discussion_log = []
        self.current_votes = {}
        self.vote_history = []

        # foreperson agent (rule-based)
        self.foreperson = ForepersonAgent(self)

        # create 12 jurors with distinct personas
        self.jurors = []
        for persona in JUROR_PERSONAS:
            juror = JurorAgent(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                persona=persona,
                vision=-1,  # see all agents, no grid needed
            )
            self.jurors.append(juror)

        self.datacollector = DataCollector(
            model_reporters={
                "Guilty_Votes": lambda m: sum(
                    1 for v in m.current_votes.values() if v == "guilty"
                ),
                "Not_Guilty_Votes": lambda m: sum(
                    1 for v in m.current_votes.values() if v == "not_guilty"
                ),
                "Undecided": lambda m: sum(
                    1 for v in m.current_votes.values() if v == "undecided"
                ),
                "Avg_Guilt_Belief": lambda m: (
                    sum(j.guilt_belief for j in m.jurors) / len(m.jurors)
                    if m.jurors
                    else 0
                ),
                "Total_Statements": lambda m: len(m.discussion_log),
                "Statements_Last_Round": lambda m: len(
                    [s for s in m.discussion_log if s["round"] == m.current_round]
                ),
            }
        )

    def _check_verdict(self):
        """Check if the jury has reached consensus or run out of rounds."""
        if len(self.current_votes) == len(self.jurors):
            votes = list(self.current_votes.values())
            if all(v == "guilty" for v in votes):
                self.verdict = "Guilty"
                self.running = False
            elif all(v == "not_guilty" for v in votes):
                self.verdict = "Not Guilty"
                self.running = False

        if self.current_round >= self.max_rounds:
            self.verdict = "Hung Jury"
            self.running = False

    def step(self):
        if not self.running or self.verdict is not None:
            return

        self.current_round += 1
        print(f"\nRound {self.current_round}")
        print("-" * 50)

        # reset round flags
        for juror in self.jurors:
            juror.reset_round()

        # foreperson picks who speaks this round
        speakers = self.foreperson.select_speakers(
            self.jurors, num_speakers=self.num_speakers
        )
        names = ", ".join(j.persona["name"] for j in speakers)
        print(f"Speakers this round ({len(speakers)}): {names}\n")

        # only selected jurors speak this round
        for speaker in speakers:
            speaker.step()

        # every few rounds the foreperson calls a vote
        if self.foreperson.should_call_vote():
            print("\n-- VOTE CALLED --")
            self.current_votes = {}
            for juror in self.jurors:
                vote = juror.cast_formal_vote()
                self.current_votes[juror.unique_id] = vote

            guilty_votes = sum(1 for v in self.current_votes.values() if v == "guilty")
            not_guilty_votes = sum(
                1 for v in self.current_votes.values() if v == "not_guilty"
            )
            undecided_votes = sum(
                1 for v in self.current_votes.values() if v == "undecided"
            )

            tally = {
                "round": self.current_round,
                "guilty": guilty_votes,
                "not_guilty": not_guilty_votes,
                "undecided": undecided_votes,
            }
            self.vote_history.append(tally)

            # foreperson announces the result into the discussion log
            summary = (
                f"[Vote Result] Guilty: {guilty_votes}, "
                f"Not Guilty: {not_guilty_votes}, Undecided: {undecided_votes}"
            )
            self.discussion_log.append(
                {
                    "juror_id": "FOREPERSON",
                    "name": "Foreperson",
                    "round": self.current_round,
                    "statement": summary,
                }
            )
            print(summary)

            self._check_verdict()
            if not self.running:
                print(f"\nDeliberation over. Verdict: {self.verdict}")

        self.datacollector.collect(self)


# Run without visualization

if __name__ == "__main__":
    from mesa_llm.reasoning.cot import CoTReasoning

    print("Starting jury deliberation...\n")
    model = JuryDeliberationModel(
        reasoning=CoTReasoning,
        llm_model="ollama/llama3.1",
        seed=42,
    )

    while model.running:
        model.step()

    print(f"\nFinal verdict: {model.verdict}")
    print(f"Total rounds: {model.current_round}")
    print(f"Total statements: {len(model.discussion_log)}")
