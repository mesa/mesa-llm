# Examples

Mesa-LLM includes working example models demonstrating how to integrate LLM-powered reasoning into agent-based simulations. These serve as both learning materials and starting points for your own models.

Each example is self-contained with its own README explaining the model concepts and how to run it.

## Epstein Civil Violence Model

Joshua Epstein's [model](https://www.pnas.org/doi/10.1073/pnas.092080199) of how a decentralized uprising can be suppressed or reach a critical mass of support.

**Location:** [examples/epstein_civil_violence](https://github.com/projectmesa/mesa-llm/tree/main/examples/epstein_civil_violence)

**What You'll Learn**:

- Complex social reasoning with LLMs
- Multiple agent types with different strategies
- Tool use for environment interaction (rebel, arrest)
- Memory systems for temporal context

**Key Concepts**: ReAct reasoning, grid perception, multi-agent dynamics

**Run it**:

```bash
cd examples/epstein_civil_violence
python app.py
```

## Negotiation Model

Implementation of a negotiation model where two types of agents (buyer and seller) negotiate over a product.

**Location:** [examples/negotiation](https://github.com/projectmesa/mesa-llm/tree/main/examples/negotiation)

**What You'll Learn**:

- Agent-to-agent communication in natural language
- Negotiation strategies and reasoning
- Conversation memory and context
- Emergent agreement patterns

**Key Concepts**: Dialogue between agents, strategic reasoning, negotiation loops

**Run it**:

```bash
cd examples/negotiation
python app.py
```

## Sugarscrap-g1mt Model

Implementation of Epstein & Axtell’s classic Sugarscape (G1MT) model, where Trader agents harvest resources (Sugar & Spice), consume them according to metabolic needs, and trade with neighbors based on economic rationality (MRS).

**Location:** [examples/sugarscrap_g1mt](https://github.com/mesa/mesa-llm/tree/main/examples/sugarscrap_g1mt)

**What You'll Learn**:

- Economic reasoning with LLMs
- Multi-resource management
- Emergent market dynamics
- Grid-based trading

**Key Concepts**: Economic optimization, resource trading, emergent equilibrium

**Run it**:

```bash
cd examples/sugarscrap_g1mt
python app.py
```

---

## How to Run Any Example

1. **Install Mesa-LLM** (if not done):

   ```bash
   pip install -U mesa-llm
   ```

2. **Set up LLM** (choose one):
   - **Local:** `ollama pull llama2 && ollama serve`
   - **Cloud:** Create `.env` with `OPENAI_API_KEY="..."`

3. **Run the example**:
   ```bash
   cd examples/MODEL_NAME
   python app.py
   ```

---

## Next Steps

- **Explore the code** - Each model has commented source files
- **Modify and experiment** - Try changing prompts, reasoning strategies, memory settings
- **Create your own** - Use these as templates for your research
