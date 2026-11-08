# Epstein Civil Violence Model

## Summary

This model is based on Joshua Epstein's simulation of how civil unrest grows and is suppressed. Citizen agents wander the grid randomly, and are endowed with individual risk aversion and hardship levels; there is also a universal regime legitimacy value. There are also Cop agents, who work on behalf of the regime. Cops arrest Citizens who are actively rebelling; Citizens decide whether to rebel based on their hardship and the regime legitimacy, and their perceived probability of arrest.

The model generates mass uprising as self-reinforcing processes: if enough agents are rebelling, the probability of any individual agent being arrested is reduced, making more agents more likely to join the uprising. However, the more rebelling Citizens the Cops arrest, the less likely additional agents become to join.

This model is implemented using Mesa-LLM unlike the original Mesa and NetLogo versions. All agents have the ability to choose actions (arresting citizens for cops, changing their state, etc.) depending on their reasoning method.

## Technical Details

The **Epstein Civil Violence Model** simulates the dynamics of civil unrest using two types of agents: **Citizens** and **Cops**.

Each **Citizen** agent is characterized by individual attributes such as:

- `hardship`
- `risk_aversion`
- `threshold` (for rebellion)

Additionally, all Citizens share a common perception of `regime_legitimacy`.

---

### Rebellion Rule

A Citizen becomes (or remains) *active* (i.e., rebels) if the following condition is met:

```
grievance - (risk_aversion * arrest_probability) > threshold
```

Where:

```
grievance = hardship * (1 - regime_legitimacy)
```

---

### Arrest Probability

The perceived probability of arrest is calculated as:

```
arrest_probability = 1 - exp(-k * round(cops_in_vision / actives_in_vision))
```

Where:

- `k` is a constant
- `cops_in_vision` is the number of cops within the agent’s vision
- `actives_in_vision` is the number of active Citizens (including the agent itself)

---

### Cops' Behavior

**Cops** patrol the grid and attempt to **arrest active Citizens** within their vision.

---

### LLM-Powered Agents

Both Citizens and Cops are implemented as **LLM-powered agents**, meaning:

- Their actions (e.g., `move`, `change state`, `arrest`) are selected by an **action workflow**.
- This workflow takes as input:
  - The agent’s internal state
  - Local observations
  - A set of available actions

This design enables **flexible, context-aware decision-making** that incorporates both:

- Quantitative attributes (e.g., `hardship`, `risk_aversion`)
- Qualitative reasoning (e.g., situational awareness, adaptive strategy)




## How to Run

If you have cloned the repo into your local machine, run ``pip install -e .`` from the project root. Then obtain an API key for an LLM provider of your choice and follow the steps below to configure this model. This model makes a large number of calls per minute, so a paid API key with higher rate limits is recommended.
1) Create a `.env` file in the project root.
2) In `.env`, set the API key variable that matches the provider prefix in `llm_model`. For example: ``OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx`` for `openai/...`, or ``GEMINI_API_KEY=your-gemini-api-key-here`` for `gemini/...`. The app uses `load_dotenv()` to load this automatically. If you use `ollama/...`, no API key is required, but you may need to configure `api_base` instead.
3) Update the ``llm_model`` attribute in `app.py` to a model you have access to. Use the format ``{provider}/{model_name}``, for example ``openai/gpt-4o-mini``.

Once you have configured `.env` and `llm_model`, run the following command from this directory:

```
    $ solara run app.py
```

## Files

* ``model.py``: Core model code.
* ``agent.py``: Agent classes.
* ``app.py``: Sets up the interactive visualization.
* ``actions.py``: Custom actions for the llm-agents to use.

## Further Reading

This model is based adapted from:

[Epstein, J. “Modeling civil violence: An agent-based computational approach”, Proceedings of the National Academy of Sciences, Vol. 99, Suppl. 3, May 14, 2002](http://www.pnas.org/content/99/suppl.3/7243.short)

A similar model is also included with NetLogo:

Wilensky, U. (2004). NetLogo Rebellion model. http://ccl.northwestern.edu/netlogo/models/Rebellion. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

You can also find Mesa's version of the model without using LLMs here:
https://github.com/mesa/mesa/tree/main/mesa/examples/advanced/epstein_civil_violence
