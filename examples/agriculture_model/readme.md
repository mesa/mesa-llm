## Summary

This model simulates how farmers make agricultural decisions under uncertainty such as rainfall variability, market prices, and social influence.

Farmer agents are placed on a grid and manage their land over a growing season. Each farmer decides when to plant, apply fertilizer, and harvest crops. These decisions directly impact yield and profit.

The model incorporates environmental factors such as rainfall conditions (low, normal, high) and economic factors such as crop market prices. Additionally, farmers can observe neighboring agents, introducing social influence into decision-making.

This model is implemented using **Mesa-LLM**, meaning agents are capable of reasoning and tool usage rather than relying purely on fixed rules. Farmers can dynamically decide actions like planting or waiting based on context.

---

## Technical Details

The **Agriculture Decision Model** simulates farming behavior using a population of **Farmer agents**.

Each **FarmerAgent** is characterized by:

- `land_size`
- `wealth`
- `education_level`
- `crop_type` (wheat, rice, maize)
- `fertilizer` level
- `crop_state`
- `yield_output`
- `profit`

---

### Crop Lifecycle

Each farmer follows a simplified crop lifecycle:
``` bash REST → PLANTED → READY → HARVESTED → REST
```

---

### Yield Function

Crop yield depends on rainfall and fertilizer usage:
``` bash
yield = base_yield × rainfall_factor × fertilizer_factor

Where:

- `rainfall_factor` depends on environmental condition (LOW / NORMAL / HIGH)
- `fertilizer_factor` increases yield when fertilizer is applied

```

### Profit Calculation
``` bash profit = yield × market_price

Where:

- `market_price` depends on crop type

```

### Environment

- The model uses a **MultiGrid** environment
- Time progresses in **daily steps**
- Each simulation runs for ~120 days (one farming season)

---

### Farmer Behavior

Each Farmer agent:

1. Observes:
   - Rainfall condition
   - Current crop state
   - Neighbor behavior

2. Decides:
   - Plant crop
   - Apply fertilizer
   - Harvest
   - Wait

3. Executes actions using tools

---

### Tools

Farmer agents use tool-based actions:

- `plant_crop` → plants crop and sets harvest date
- `apply_fertilizer` → increases yield potential
- `harvest_crop` → harvests when ready

---

### LLM-Powered Agents

Farmers are implemented as **LLM agents**, meaning:

- Decisions are generated via a reasoning module
- Agents use:
  - Internal state (wealth, crop, etc.)
  - Observations (environment + neighbors)
  - Available tools

This allows:

- Context-aware decision making
- Adaptive strategies
- Non-deterministic behavior

---

## How to Run

If you have cloned the repo into your local machine, ensure you run the following command from the root of the library:
``` bash pip install -e .
```

Then, set up your LLM API key.

### Setup Steps

1. Install dotenv if not already installed:
pip install python-dotenv
Copy code

2. Create a `.env` file in the root directory

3. Add your API key:

For OpenAI:
``` python OPENAI_API_KEY=your-api-key
```


For Gemini:
``` python GEMINI_API_KEY=your-api-key
```

4. In `app.py`, set:
``` python
- `llm_model = "openai/gpt-4o-mini"` (or your model)
```

---

### Run the Model
``` python
solara run app.py
```


Open in browser:
``` bash http://localhost:8765⁠�
```
---

## Files

- `model.py` → Core simulation logic
- `agent.py` → Farmer agent definition
- `tools.py` → Actions available to agents
- `app.py` → Visualization and UI

---

## Metrics

The model tracks:

- **Total Yield**
- **Average Profit**
- **Crop Distribution**

---

## Further Reading

This model is inspired by:

> [Agent-Based Modeling in Agricultural Productivity in Rural Area of Bahir Dar](https://www.mdpi.com/2571-9394/4/1/20)

Related work on Agent-Based Modeling in agriculture:

- Agent-based crop decision models
- Climate adaptation simulations
- Socio-environmental systems

---

## Notes

- This model simplifies crop growth dynamics
- Focus is on **decision-making behavior**, not full biological simulation
- Can be extended with:
  - Climate models
  - Policy interventions
  - Multi-season dynamics

