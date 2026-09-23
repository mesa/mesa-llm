# Jury Deliberation

A small Mesa-LLM example that simulates a 12-person jury deliberating over a criminal case.
Built using Mesa for simulation and Mesa-LLM for language-based agents.

Each juror is an LLM-powered agent with a simple personality — a retired engineer, a social worker, a nurse, a skeptic, and so on. They argue, listen to each other, update their beliefs, and eventually vote. A lightweight Foreperson agent (no LLM) manages the whole process: deciding who speaks, when to vote, and when deliberation ends.

---

## Why this model is interesting

Most agent-based models don't handle persuasion very well. Traditional ABMs use probability tables to decide if one agent "convinces" another. This model uses actual language — agents construct arguments based on evidence and respond to what others said.

That introduces a few practical challenges:

- **Context window pressure**: 12 agents speaking across multiple rounds fills up memory fast. This model uses a rolling window — each juror only reads the last 6 statements, not the full history.
- **LLM cost control**: Instead of running all 12 agents each round, the Foreperson selects 2-3 speakers. Total LLM calls drop from 12 to ~3 per round.
- **Persona consistency**: Each juror has a fixed personality injected into their system prompt every step, so they don't drift.

---

## The case

*State v. Marcus Rivera* — a burglary charge with 7 pieces of evidence. The case is designed to be ambiguous: there's a fingerprint on the window, but Rivera visited the store legitimately two days before. There's a security video, but it's grainy. There's an alibi witness, but he's a close friend.

Jurors genuinely disagree. Some will lean guilty early; others will push back. That's the point.

---

## How it works

```
Each round:
  1. Foreperson picks 2-3 jurors to speak (based on who's been quiet and who disagrees with the majority)
  2. Selected jurors read the last few statements and generate an argument
  3. Arguments are broadcast to all jurors and influence their belief (0=innocent, 1=guilty)
  4. Every 3 rounds, a formal vote is held
  5. If all 12 agree → verdict. If rounds run out → hung jury.
```

The model stops automatically when a unanimous verdict is reached or after `max_rounds` (default 15).

---

## Project structure

```
jury_deliberation/
├── case_data.py    # the court case, evidence, and facts
├── agents.py       # ForepersonAgent + JurorAgent (12 personas)
├── tools.py        # speak_to_room, review_evidence, cast_vote
├── model.py        # orchestration, voting, termination
├── app.py          # Solara visualization
└── README.md
```

---

## Running it

**With visualization:**
```bash
solara run examples/jury_deliberation/app.py
```

**Headless (terminal only):**
```bash
python -m examples.jury_deliberation.model
```

The default LLM backend is `ollama/llama3.1`. You can change `llm_model` in `app.py` or `model.py` to use any LiteLLM-compatible model (e.g. `openai/gpt-4o`, `gemini/gemini-2.0-flash`).

---

## What you can observe

- How quickly jurors converge or polarize
- Which personalities tend to hold their position vs. be persuaded
- The juror belief table with simple progress bars (`█░`)
- How the rolling memory window affects the flow of argument (a friction point worth exploring)
- Whether the Foreperson's speaker selection strategy produces more realistic debates than random selection

---

## Limitations and friction points

A few things this model intentionally surfaces as open problems:

1. **Belief updates are heuristic** — juror beliefs shift based on keyword detection, not actual semantic understanding. A better version would use the LLM itself to rate persuasiveness.
2. **Memory compression** — the rolling window helps, but long deliberations still accumulate context. Summarization-based memory could help here.
3. **Scaling** — adding more jurors works, but the Foreperson's speaker selection gets harder to tune. This is worth experimenting with.
