import io
import re
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout

import solara
from dotenv import load_dotenv
from model import MarketPanicModel

warnings.filterwarnings("ignore", category=RuntimeWarning)

load_dotenv()

sim_model = solara.reactive(None)
error_msg = solara.reactive(None)
step_count = solara.reactive(0)

simulation_logs = solara.reactive("Waiting for simulation to start...\n")
ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


@solara.component
def Page():
    if error_msg.value:
        solara.Error(f"CRASH DETECTED:\n\n{error_msg.value}")
        return

    if sim_model.value is None:
        try:
            sim_model.value = MarketPanicModel(num_agents=4)
        except Exception:
            error_msg.value = traceback.format_exc()
            return

    solara.Style("""
    .terminal-logs {
        max-height: 650px;
        overflow: auto;
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #333;
    }
    .terminal-logs pre {
        font-family: 'Consolas', 'Courier New', monospace !important;
        font-size: 12px !important;
        color: #10B981 !important;
        white-space: pre !important;
        line-height: 1.2 !important;
        margin: 0 !important;
    }
    """)

    solara.Markdown("# 📈 Information Cascade Benchmark")

    with solara.Row(justify="space-between"):
        solara.Markdown(f"### Current Simulation Step: {step_count.value}")

        def on_step():
            try:
                f = io.StringIO()
                with redirect_stdout(f), redirect_stderr(f):
                    sim_model.value.step()

                raw_output = f.getvalue()
                clean_output = ansi_escape.sub("", raw_output)

                if clean_output.strip():
                    new_log = f"==== SIMULATION STEP {step_count.value + 1} ====\n{clean_output}\n\n"
                    simulation_logs.value = new_log + simulation_logs.value

                step_count.value += 1
            except Exception:
                error_msg.value = traceback.format_exc()

        solara.Button("Run 1 Step", on_click=on_step, color="primary")

    solara.Markdown("### 🧠 Agents' Thought Process & System Logs")

    with solara.VBox(classes=["terminal-logs"]):
        solara.Preformatted(simulation_logs.value)


page = Page
