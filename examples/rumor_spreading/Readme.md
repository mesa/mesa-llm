# Rumor Spreading Model

**Disclaimer**: This is an original model created for Mesa-LLM. It is inspired by classic information-diffusion and epidemic-spreading research but is not a direct port of any single paper.

## Summary

This model simulates how **misinformation and verified news compete** to shape public belief inside a community. Two journalists sit at fixed positions on the grid and broadcast stories every step — one publishes sensational, potentially fabricated headlines; the other publishes calm, evidence-based facts. Citizens wander around the neighbourhood, bump into each other, hear what the journalists broadcast, and decide — based on who they are as people — whether to **believe it, doubt it, debunk it, or simply pass it on**.

The central question the model asks:

> *Does misinformation cascade through a community, or do skeptics and critical thinkers slow it down — and how does the mix of personalities in the population determine the outcome?*

This is a question you cannot answer with a simple probability rule. The answer depends on the **content** of the message and the **character** of the person receiving it — exactly the kind of nuanced, language-level reasoning that makes LLM-powered agents the right tool here.

---

## Agents

### `CitizenAgent`

Ordinary people who move around the grid, hear stories, and form opinions.

Every citizen has a **personality** that shapes how they reason about information:

| Personality | Behaviour |
|---|---|
| `credulous` | Trusts almost anything they hear; spreads quickly without much evaluation. |
| `skeptic` | Questions claims before accepting them; often stays on the fence. |
| `critical_thinker` | Demands logic and evidence; most likely to debunk misinformation. |
| `conformist` | Follows the crowd; goes along with whatever most neighbours already believe. |

Citizens are distributed round-robin across personalities so every run starts with a natural mix.

Each citizen tracks a **`BeliefState`**:

| State | Meaning |
|---|---|
| `UNINFORMED` | Hasn't heard anything meaningful yet. |
| `BELIEVER` | Has accepted the rumor and is willing to spread it. |
| `SKEPTIC` | Heard something, doubts it, but hasn't proven it false. |
| `DEBUNKED` | Is confident the story is false and will say so if asked. |

---

### `JournalistAgent`

Journalists sit at **fixed positions** on the grid and never move. Their only job is to publish stories every single step. There are two of them placed in opposite corners:

| Journalist | Bias | Behaviour |
|---|---|---|
| Journalist 1 | `biased` | Publishes sensational, emotionally charged, sometimes fabricated stories. Sets `is_misinformation=True`. |
| Journalist 2 | `objective` | Publishes verified, factual, calm news. Sets `is_misinformation=False`. |

Their stories are broadcast directly into the memory of every citizen within their `vision` radius — those citizens will read and reason about the story on their next turn.

---

## Tools

| Tool | Used by | What it does |
|---|---|---|
| `update_belief` | `CitizenAgent` | Records the citizen's personal stance: `BELIEVER`, `SKEPTIC`, or `DEBUNKED`. |
| `publish_story` | `JournalistAgent` | Writes a headline and broadcasts it to all nearby citizens' memories. Tags it as `[MISINFORMATION]` or `[VERIFIED NEWS]`. |
| `speak_to` | `CitizenAgent` | Sends a message to one or more nearby citizens to spread what they believe. |
| `move_one_step` | `CitizenAgent` | Moves one cell in any cardinal or diagonal direction to explore the grid. |

---

## How Information Flows (Step by Step)

```
Step N (with current defaults: 12 citizens, 2 journalists, 8×8 grid, vision=2):

  1. Journalist (biased)   → publishes "[MISINFORMATION] Shocking scandal uncovered!"
                              → drops into memory of every citizen within vision=2

  2. Journalist (objective) → publishes "[VERIFIED NEWS] New park opens downtown."
                              → drops into memory of every citizen within vision=2

  3. Citizens take turns (shuffled):
       - Each citizen reads its short-term memory (recent messages received)
       - Counts how many neighbours are already believers (social pressure)
       - Builds a reasoning prompt with: personality + heard rumor + social context
       - LLM reasons (ReActReasoning by default) → calls tools:
           → credulous citizen: speak_to(neighbours, "Did you hear? Scandal!") + update_belief("BELIEVER")
           → skeptic citizen:   update_belief("SKEPTIC")
           → critical_thinker:  update_belief("DEBUNKED") + speak_to(neighbours, "That's false.")
           → conformist:        looks at neighbour count → follows majority
       - If nothing heard yet: move_one_step("North") to explore

  4. DataCollector records counts of each BeliefState.
```

---

## What Makes LLM Genuinely Useful Here

In a classical rule-based epidemic model (like SIR), spread is a coin flip: `if random() < spread_chance: infect()`. The content of the rumor is irrelevant.

Here, the LLM actually **reads the headline**. A `critical_thinker` receiving `"[MISINFORMATION] Government hiding alien contact"` will reason about the claim itself — recognise the sensational framing, weigh it against what it knows, and likely call `update_belief("DEBUNKED")`. Give that same agent `"[VERIFIED NEWS] City council approves budget"` and they'll accept it calmly.

This means the **quality of the story matters**, the **framing matters**, and the **personality of the recipient matters** — none of which a probability-based rule could capture.

---

## Data Collection

The model tracks the following metrics every step:

| Metric | Description |
|---|---|
| `Uninformed` | Citizens who haven't formed a belief yet. |
| `Believers` | Citizens who accepted the rumor. |
| `Skeptics` | Citizens who doubt but haven't debunked. |
| `Debunked` | Citizens actively calling the story false. |
| `Stories_Published` | Total stories published by all journalists so far. |

The Solara chart shows all four belief curves over time — you can watch the cascade or the resistance unfold live.

---

## Visual Guide (Grid)

| Symbol | Colour | Meaning |
|---|---|---|
| ● blue `#648FFF` | Uninformed citizen |
| ● orange `#FE6100` | Believer citizen |
| ● amber `#FFB000` | Skeptic citizen |
| ● pink `#DC267F` | Debunked citizen |
| ● red `#FF0000` | Journalist (fixed, never moves) |

---

## Configuration

### Default Parameters (in `app.py`)

| Parameter | Default | Purpose |
|---|---|---|
| `initial_citizens` | 12 | Number of citizen agents; more = richer dynamics. |
| `initial_journalists` | 2 | Number of journalists (max 2, opposite corners). |
| `width` / `height` | 8 | Grid size; larger = more room to wander. |
| `vision` | 2 | Broadcast and observation radius (cells). |
| `reasoning` | `ReActReasoning` | LLM reasoning strategy; switch to `CoTReasoning` for fewer LLM calls. |
| `llm_model` | `ollama/llama3.1:8b` | Change to `openai/gpt-4o-mini`, etc. |
| `api_base` | (Remote Ollama URL) | Set to your Ollama server; omit for local/default. |

### Tuning for Performance

**Fewer LLM calls:**
- Reduce `initial_citizens` (e.g., 6 instead of 12)
- Reduce `vision` (e.g., 1 instead of 2)
- Switch `reasoning` to `CoTReasoning` (1 call/agent vs 3–5 for ReAct)

**Richer dynamics:**
- Increase `initial_citizens` (e.g., 20)
- Increase grid size (`width`, `height`)
- Increase `vision` for faster information spread

If you have cloned the repo into your local machine, ensure you run the following command from the root of the library: ``pip install -e .``. Then, you will need an API key from an LLM provider of your choice. Once you have obtained the API key, follow the steps below to set it up for this model.

### Option 1: Local LLM (Ollama)

If running Ollama on your **local machine**:
1. Start Ollama: `ollama serve` (default: `http://localhost:11434`)
2. Pull a model: `ollama pull llama3.1:8b`
3. Run the example: `solara run examples/rumor_spreading/app.py`

If running Ollama on a **remote machine**:
1. Edit `app.py`: set `OLLAMA_URL = "http://your-remote-machine:port"` (e.g., the pinggy URL you received)
2. Run the example from your local machine: `solara run examples/rumor_spreading/app.py`

### Option 2: Cloud LLM (OpenAI, Gemini, etc.)

1. Ensure the `dotenv` package is installed. If not, run ``pip install python-dotenv``.
2. In the root folder of the project, create a file named `.env`.
3. If you are using OpenAI's API key, add: ``OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx``. If you are using Gemini, add: ``GEMINI_API_KEY=your-gemini-api-key-here``.
4. In `app.py`, change:
   - `llm_model` to match your provider: e.g. `"openai/gpt-4o-mini"` or `"gemini/gemini-1.5-flash"`
   - `OLLAMA_URL` can be removed or left as-is (it won't be used)

Once set up, run from the **root of the repository**:

```bash
solara run examples/rumor_spreading/app.py
```

---

## Files

* `model.py` — Core model logic: grid setup, agent creation, step ordering, data collection.
* `agents.py` — `CitizenAgent` and `JournalistAgent` classes with `BeliefState` enum.
* `tools.py` — Custom tools: `update_belief` (citizen) and `publish_story` (journalist).
* `app.py` — Solara interactive visualization: grid display and live belief-state chart.
* `__init__.py` — Registers tools at import time.

---


This model draws conceptual inspiration from:

- **SIR / SEIR epidemic models** — the canonical framework for information-spreading simulations.
- Vosoughi, S., Roy, D., & Aral, S. (2018). *The spread of true and false news online*. **Science**, 359(6380), 1146–1151. — Empirical study showing false news spreads faster than true news on Twitter.
- Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023) — the foundation for LLM-powered agent memory and reasoning used in Mesa-LLM.

Mesa's original epidemic-spreading example (without LLMs) can be found here:
https://github.com/mesa/mesa/tree/main/mesa/examples/basic/virus_on_network
