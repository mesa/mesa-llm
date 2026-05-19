# Sugarscape Constant Growback Model with Traders

## Summary

This model is based on Epstein & Axtell's classic "Sugarscape" simulation from Growing Artificial Societies (1996), specifically the G1MT (Growback 1, Metabolism, Trade) variation. Trader agents wander a grid populated with two unevenly distributed resources: Sugar and Spice. Agents are endowed with individual metabolic rates for each resource and a vision range; there are also Resource agents, which represent the landscape and regenerate food over time.

The model generates emergent economic dynamics through decentralized interactions. Traders must constantly harvest resources to satisfy their metabolic needs; if they run out of either sugar or spice, they starve. Crucially, agents can trade with neighbors. Decisions are governed by the Marginal Rate of Substitution (MRS); agents rich in sugar but poor in spice will trade sugar to acquire spice, and vice versa. Over time, this decentralized trading allows for the emergence of a price equilibrium and wealth distribution patterns.

This model is implemented using Mesa-LLM, unlike the original deterministic versions. All Trader agents use Large Language Models to "think" about their survival. They observe their internal inventory and MRS, then autonomously choose actions to move to high-value resource tiles or propose trades to neighbors to ensure their continued existence.

## Technical Details

### Agents

- `Trader (LLMAgent):` The primary actor equipped with STLTMemory and ReActReasoning.

        Internal State: Dynamically updates a context string with current inventory (Sugar, Spice) and hunger warnings to guide the LLM.

        Metabolism: Consumes a fixed amount of resources per step. Zero inventory results in agent removal (death).

        MRS Calculation: Computes the Marginal Rate of Substitution (MRS) using the Cobb-Douglas formula to value Sugar vs. Spice relative to biological needs.

- `Resource (CellAgent):` A passive environmental agent that acts as a container for resources. It regenerates its current_amount by 1 unit per step up to a max_capacity.

### Actions

- `move_to_best_resource:`

        Function: Scans the local grid within the agent's vision radius.

        Action: Identifies the cell with the highest current_amount of resources, moves the agent there, and automatically harvests the full amount into the agent's inventory.

-  `propose_trade:`

        Function: Targets a specific neighbor by unique_id.

        Logic: Executes a trade only if the partner's MRS is higher than the proposer's (indicating the partner values Sugar more highly). This ensures trades are mathematically rational and mutually beneficial.


### Movement Rule (Rule M)

A Trader agent moves to a new location and harvests resources if the following logic, executed by the *move_to_best_resource action*, is satisfied:

Scan: The agent inspects all cells within its *vision range* (von Neumann neighborhood).

Identify: It identifies the cell containing a *Resource* agent with the highest *current_amount* of sugar/spice.

Harvest: The agent moves to that cell, sets the resource's amount to 0, and adds the harvested amount to its own inventory.

### Trade Rule (Rule T)

Agents determine whether to trade based on their Marginal Rate of Substitution (MRS). A trade is proposed via the propose_trade action and accepted if it is mutually beneficial:

```
Trade occurs if: Partner_MRS > Agent_MRS
```

Where the MRS is calculated using the agent's inventory and metabolism:

```
MRS = (spice_inventory / spice_metabolism) / (sugar_inventory / sugar_metabolism)
```

In this implementation:

-    `Agent (Proposer):` Gives Sugar, Receives Spice.

-    `Partner (Receiver):` Receives Sugar, Gives Spice.

-    This flow ensures resources move from agents who value them less to agents who value them more.

### Resource Behavior

Resource Agents represent the landscape. They are passive agents that regenerate wealth over time:

-    `Growback:` At every step, a Resource agent increases its current_amount by growback (default: 1).

-    `Capacity:` This growth is capped at the agent's max_capacity.

### LLM-Powered Agents

Both Traders and the simulation logic are driven by LLM-powered agents, meaning:

-    Their actions (e.g., `move_to_best_resource`, `propose_trade`) are selected by an action workflow.

-    This workflow takes as input:

        The agent’s internal state (current inventory, metabolic warnings, and calculated MRS).

        Local observations of the grid.

-        A set of available actions registered by the example.



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
* ``agents.py``: Agent classes.
* ``app.py``: Sets up the interactive visualization.
* ``actions.py``: Custom actions for the llm-agents to use.


## Further Reading

[Growing Artificial Societies](https://mitpress.mit.edu/9780262550253/growing-artificial-societies/)
[Complexity Explorer Sugarscape with Traders Tutorial](https://www.complexityexplorer.org/courses/172-agent-based-models-with-python-an-introduction-to-mesa#gsc.tab=0)
