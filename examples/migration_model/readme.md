# Conflict-Driven Migration Model

## Summary

This model simulates how individuals in a conflict-affected environment decide whether to migrate, based on their perceived risk and interactions with neighbors.

In this version of the model, citizens evaluate perceived risk and their own risk proneness, computing a migration probability at each step. Agents update their state (REST or MIGRATE) based on this probability. The model also uses **LLM-powered reasoning** to provide contextual explanations of agent behavior.

Unlike traditional CA or threshold models, this model combines quantitative risk dynamics with narrative reasoning to produce emergent patterns of migration behavior.

---

## Technical Details

The **Conflict-Driven Migration Model** uses a grid environment to simulate spatial dynamics of individuals:

### Agent Types

Each **Citizen** agent has:

- `risk_proneness` — how sensitive the agent is to risk
- `migration_prob` — computed based on perceived risk
- `state` — either `REST` or `MIGRATE`

Agents remain in place if resting, or change to a migrating state when migration probability crosses a threshold. In this version, agents do not physically relocate; instead, they change state and the system visualization reflects their status.

LLM reasoning is used for descriptive explanations of each agent’s reasoning at each step, but the underlying state transitions remain deterministic and mathematically driven.

---

### Migration Probability

At each step, a citizen computes:
```
perceived_risk = own_risk_proneness + (influence_of_neighbors) migration_prob = sigmoid(growth_rate * (perceived_risk - baseline_threshold))
```
This calculation creates a **positive feedback effect**:

- Higher perceived risk increases migration probability
- As more neighbors migrate, perceived risk increases further

This dynamic can lead to a rapid cascade of migrating agents.

---

## LLM-Powered Reasoning

Agents can use a reasoning module (typically LLM-based via mesa-llm) to produce human-like justifications for their decisions. The model does *not* allow arbitrary tool actions by the LLM, so reasoning is explanatory rather than autonomous.

Each agent’s reasoning process consists of:

1. Observing its local environment and internal state
2. Using its memory (short-term or long-term)
3. Producing a `reasoning` text along with a symbolic `action`

In the basic version, actions are explanations only; movement or location updates are not handled by the LLM.

---

## How to Run

If you have cloned the repo into your local machine, ensure you run the following command from the root of the library: ``pip install -e . ``. Then, you will need an api key of an LLM-provider of your choice. You can also use Ollama Model. Once you have obtained the api-key follow the below steps to set it up for this model.
1) Ensure the dotenv package is installed. If not, run ``pip install python-dotenv``.
2) In the root folder of the project, create a file named .env.
3) If you are using openAI's api key, add the following command in the .env file: ``OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx``. If you have the paid version of Gemini, use this line instead: ``GEMINI_API_KEY=your-gemini-api-key-here``(the free ones tend to not work with this model).
4) Change the  ``api_key`` specification in app.py according to the provider you have chosen.
5) Similarly change the ``llm_model`` attribute as well in app.py to the name of a model you have access to. Ensure it is in the form of {provider}/{model_name}. For e.g. ``openai/gpt-4o-mini``.

Once you have set up the api-key in your system, run the following command from this directory:

```
    solara run app.py
```
## Files

* ``model.py``: Core model code.
* ``agent.py``: Agent classes.
* ``app.py``: Sets up the interactive visualization.
* ``tools.py``: Tools for the llm to use.

## Further Reading

This model is inspired by the study:
[Dolmas et al., "Title of Paper", PNAS Nexus (2023)](https://academic.oup.com/pnasnexus/article/3/3/pgae080/7624910?login=false)

Key ideas from that research include:
- Conflict intensity influences migration decisions
- Migration exhibits threshold/cascade dynamics
- Spatial proximity and social influence affect migration likelihood
- This prototype combines those ideas with LLM-assisted decision explanations.

### Note:-
This module is currently a simplified prototype. Advanced features such as custom tools, relocation mechanics, and extended agent interactions can be added in future versions. The current design focuses on validating the core migration dynamics and hybrid LLM reasoning architecture.
