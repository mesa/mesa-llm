import logging
import warnings

import solara
from dotenv import load_dotenv
from mesa.visualization import SolaraViz, make_plot_component

from examples.jury_deliberation.model import JuryDeliberationModel
from mesa_llm.reasoning.cot import CoTReasoning

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
    message=r".*Pydantic serializer warnings.*",
)
logging.getLogger("pydantic").setLevel(logging.ERROR)

load_dotenv()

model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "reasoning": CoTReasoning,
    "llm_model": "ollama/llama3.1",
    "num_speakers": 1,
    "max_rounds": 15,
}

model = JuryDeliberationModel(
    reasoning=model_params["reasoning"],
    llm_model=model_params["llm_model"],
    num_speakers=model_params["num_speakers"],
    max_rounds=model_params["max_rounds"],
    seed=model_params["seed"]["value"],
)


# vote distribution chart — tracks guilty vs not guilty over rounds
vote_chart = make_plot_component(
    {"Guilty_Votes": "#e74c3c", "Not_Guilty_Votes": "#2ecc71", "Undecided": "#f39c12"}
)

# average belief trajectory
belief_chart = make_plot_component({"Avg_Guilt_Belief": "#3498db"})


@solara.component
def DiscussionLog(model):
    """Shows the last few entries from the discussion log."""
    show = solara.use_reactive(False)

    def toggle():
        show.set(not show.value)

    solara.Button(
        label="Show Discussion Log" if not show.value else "Hide Discussion Log",
        on_click=toggle,
    )

    if show.value and hasattr(model, "discussion_log"):
        recent = model.discussion_log[-8:]  # show last 8 statements
        if recent:
            lines = []
            for entry in recent:
                lines.append(
                    f"**{entry['name']}** (Round {entry['round']}): {entry['statement']}"
                )
            solara.Markdown("\n\n".join(lines))
        else:
            solara.Text("No discussion yet.")


@solara.component
def VerdictStatus(model):
    """Displays the current round, verdict status, and juror beliefs."""
    round_text = f"**Round {model.current_round} / {model.max_rounds}**"

    if model.verdict:
        verdict_text = f"**Verdict: {model.verdict}**"
        solara.Markdown(f"{round_text} \u2014 {verdict_text}")
    else:
        solara.Markdown(f"{round_text} \u2014 Deliberation in progress...")

    # show individual juror beliefs as a compact table
    if hasattr(model, "jurors") and model.jurors:
        rows = []
        for j in model.jurors:
            belief_bar = "\u2588" * int(j.guilt_belief * 10) + "\u2591" * (
                10 - int(j.guilt_belief * 10)
            )
            rows.append(
                f"| {j.persona['name']} | {j.persona['occupation']} | "
                f"{j.guilt_belief:.2f} | `{belief_bar}` | {j.vote} |"
            )
        header = "| Juror | Occupation | Belief | | Vote |\n|---|---|---|---|---|"
        table = header + "\n" + "\n".join(rows)
        solara.Markdown(table)


if __name__ == "__main__":
    page = SolaraViz(
        model,
        components=[
            VerdictStatus,
            vote_chart,
            belief_chart,
            DiscussionLog,
        ],
        model_params=model_params,
        name="Jury Deliberation",
    )


"""run with:
solara run examples/jury_deliberation/app.py
"""
