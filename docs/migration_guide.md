# Migration Guide: From Mesa / Mesa-Geo to Mesa-LLM

## Who this is for

If you already have a working Mesa model — or a geospatial one built on Mesa-Geo — and you're wondering what it takes to give your agents language-based reasoning, this guide is for you. The short version: **you're not rebuilding your model, you're upgrading how your agents think.** Mesa-LLM doesn't replace Mesa's scheduling, spaces, or data collection. It slots a reasoning layer into the agents you already know how to write.

> **Before you start:** Mesa-LLM is still under active development, and its maintainers are upfront that the API may shift as the project heads toward a first stable release. Pin your version once you find one that works (`pip install mesa-llm==<version>`), and re-check this guide against the [official docs](https://mesa-llm.readthedocs.io/) if something doesn't line up.

---

## 1. Key Differences Between Mesa, Mesa-Geo, and Mesa-LLM

It helps to think of these three as layers, not competitors. Mesa is the foundation. Mesa-Geo and Mesa-LLM are both *extensions* that add one specific capability on top of it — and, importantly, they add different capabilities, which means they can be combined.

| | **Mesa** | **Mesa-Geo** | **Mesa-LLM** |
|---|---|---|---|
| **What it is** | The core agent-based modeling (ABM) framework | A GIS/geospatial extension to Mesa | A language-model reasoning extension to Mesa |
| **What it adds** | Agents, models, schedulers, spaces, data collection, visualization | Real-world geometry, coordinate reference systems (CRS), shapefiles/GeoJSON support | Natural-language reasoning, memory, and tool-use inside agents |
| **Core agent class** | `Agent` | `GeoAgent` (adds `geometry` and `crs`) | `LLMAgent` (adds `reasoning`, `memory`, `llm_model`) |
| **Decision-making** | Hard-coded rules you write in `step()` | Same as Mesa — rules, just spatially aware | Delegated (fully or partially) to an LLM via a reasoning module |
| **Space** | Abstract grids, continuous space, networks | `GeoSpace` — real coordinates, shapefiles, GeoDataFrames | Whatever space Mesa/Mesa-Geo already gives you — Mesa-LLM adds no new space types |
| **Key dependency** | None beyond Python | `geopandas`, `shapely`, `pyproj`, `rtree` | An LLM provider (OpenAI, Anthropic, Ollama, etc.) via LiteLLM |
| **Execution model** | `model.step()` → `agent.step()` | Identical to Mesa | Identical to Mesa — `step()` is still what gets called |

The one-sentence version the Mesa-LLM team uses themselves is worth repeating: *agents in Mesa-LLM are still standard Mesa agents — the only difference is how decisions get made, not how the model runs.* The same is true of Mesa-Geo agents. Neither extension touches Mesa's scheduling or execution loop, which is exactly what makes migrating (or combining both) tractable.

---

## 2. Mapping Mesa Concepts to Mesa-LLM

If you know Mesa, most of what you already know carries over directly. Here's the concept-by-concept translation:

| Mesa concept | Stays the same? | Mesa-LLM equivalent / addition |
|---|---|---|
| `Model` | ✅ Unchanged | Your model class still inherits from `mesa.model.Model` |
| `Agent` | ✅ Unchanged, but... | Inherit from `LLMAgent` instead when you want that agent to reason via an LLM |
| `AgentSet` / `create_agents()` | ✅ Unchanged | `LLMAgent.create_agents()` works exactly like Mesa's, plus reasoning/LLM kwargs |
| `model.step()` scheduling (`shuffle_do`, etc.) | ✅ Unchanged | No changes needed |
| Hard-coded `if/else` decision logic in `step()` | ⚠️ Optional to keep | Replace with (or supplement using) `self.reasoning.plan(prompt, obs)` |
| Agent internal state (plain attributes) | ✅ Unchanged | Can also be surfaced to the LLM as `internal_state` or folded into the observation dict |
| `DataCollector` | ✅ Unchanged | Reasoning traces and actions can be logged the same way as any other agent attribute |
| Spaces (`MultiGrid`, `ContinuousSpace`, `GeoSpace`, etc.) | ✅ Unchanged | Mesa-LLM has no space abstractions of its own — it relies entirely on Mesa's |

The practical takeaway: **you migrate agent by agent, not model by model.** A model can mix ordinary rule-based `Agent`s and LLM-powered `LLMAgent`s side by side, since both are scheduled the exact same way.

---

## 3. Migrating an Existing Mesa Model

Let's walk through a minimal but realistic example: a rule-based agent that decides what to do based on a hard-coded condition, migrated to reason about that same decision in natural language.

### Before: plain Mesa

```python
from mesa import Model, Agent

class VillagerAgent(Agent):
    def __init__(self, model, food_level=5):
        super().__init__(model)
        self.food_level = food_level

    def step(self):
        if self.food_level < 3:
            self.food_level += 2  # forage
        else:
            self.food_level -= 1  # idle, consume reserves


class VillageModel(Model):
    def __init__(self, n=5, seed=None):
        super().__init__(seed=seed)
        VillagerAgent.create_agents(model=self, n=n)

    def step(self):
        self.agents.shuffle_do("step")
```

### After: Mesa-LLM

The model class barely changes. The agent class swaps its hard-coded branch for a reasoning call:

```python
from mesa.model import Model
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.memory.st_lt_memory import STLTMemory

class VillagerAgent(LLMAgent):
    def __init__(self, *args, food_level=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.food_level = food_level
        self.memory = STLTMemory(agent=self, llm_model="ollama/llama3")

    def step(self):
        observation = {"food_level": self.food_level}
        prompt = (
            "You are a villager managing your food supply. "
            "Decide whether to forage for more food or rest and "
            "conserve energy, and briefly explain why."
        )
        plan = self.reasoning.plan(prompt=prompt, obs=observation, tools=[])
        print(plan)


class VillageModel(Model):
    def __init__(self, n=5, seed=None):
        super().__init__(seed=seed)
        VillagerAgent.create_agents(
            model=self,
            n=n,
            reasoning=ReActReasoning,
            llm_model="ollama/llama3",
            system_prompt="You are a villager agent in a small settlement simulation.",
            internal_state="",
        )

    def step(self):
        self.agents.shuffle_do("step")
```

What actually changed:

- `Agent` → `LLMAgent`
- The `if/else` block became a natural-language `prompt` fed to `self.reasoning.plan()`
- `create_agents()` now takes `reasoning`, `llm_model`, and `system_prompt` — everything else about agent creation is identical to vanilla Mesa
- `model.step()` and the scheduling call (`self.agents.shuffle_do("step")`) **did not change at all**

Note that `ReActReasoning` always returns both a reasoning trace *and* a suggested action. In an introductory migration like this, that action is just printed — actually executing it against your model's state (e.g., calling `self.forage()`) is a deliberate next step you add once you're comfortable with the reasoning output.

---

## 4. Migrating Mesa-Geo Models and Geospatial Workflows

This is the case most people hit next: you have real coordinates, shapefiles, or GeoDataFrames driving your model, and you want your `GeoAgent`s to reason about their spatial situation in language rather than pure rule logic.

The good news is that **nothing about your geospatial setup needs to change.** `GeoSpace`, `AgentCreator`, CRS handling, `from_GeoDataFrame()`, `from_GeoJSON()` — all of that stays exactly as it was. Mesa-LLM adds reasoning to the *agent*, not to the space.

### Before: plain Mesa-Geo

```python
import mesa_geo as mg
from mesa import Model

class ResidentAgent(mg.GeoAgent):
    def __init__(self, model, geometry, crs, risk_tolerance=0.5):
        super().__init__(model, geometry, crs)
        self.risk_tolerance = risk_tolerance
        self.evacuated = False

    def step(self):
        flood_risk = self.model.get_local_flood_risk(self.geometry)
        if flood_risk > self.risk_tolerance:
            self.evacuated = True


class FloodModel(Model):
    def __init__(self, geojson_data, seed=None):
        super().__init__(seed=seed)
        self.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(agent_class=ResidentAgent, model=self)
        agents = creator.from_GeoJSON(GeoJSON=geojson_data, unique_id="id")
        self.space.add_agents(agents)

    def step(self):
        self.agents.shuffle_do("step")
```

### After: Mesa-LLM + Mesa-Geo combined

Because `GeoAgent` and `LLMAgent` are both thin, well-behaved wrappers around Mesa's base `Agent`, you can compose them with multiple inheritance. This is the recommended pattern for geospatial + reasoning agents:

```python
import mesa_geo as mg
from mesa import Model
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.memory.st_lt_memory import STLTMemory

class ResidentAgent(mg.GeoAgent, LLMAgent):
    def __init__(self, model, geometry, crs, risk_tolerance=0.5, **kwargs):
        super().__init__(model=model, geometry=geometry, crs=crs, **kwargs)
        self.risk_tolerance = risk_tolerance
        self.evacuated = False
        self.memory = STLTMemory(agent=self, llm_model="ollama/llama3")

    def step(self):
        flood_risk = self.model.get_local_flood_risk(self.geometry)
        observation = {
            "flood_risk": flood_risk,
            "risk_tolerance": self.risk_tolerance,
            "already_evacuated": self.evacuated,
        }
        prompt = (
            "You are a resident deciding whether to evacuate given the "
            "current flood risk near your home. Reason about the tradeoffs "
            "and state your decision clearly."
        )
        plan = self.reasoning.plan(prompt=prompt, obs=observation, tools=[])
        print(plan)


class FloodModel(Model):
    def __init__(self, geojson_data, seed=None):
        super().__init__(seed=seed)
        self.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(
            agent_class=ResidentAgent,
            model=self,
            agent_kwargs={
                "reasoning": ReActReasoning,
                "llm_model": "ollama/llama3",
                "system_prompt": "You are a resident in a flood-prone coastal town.",
                "internal_state": "",
            },
        )
        agents = creator.from_GeoJSON(GeoJSON=geojson_data, unique_id="id")
        self.space.add_agents(agents)

    def step(self):
        self.agents.shuffle_do("step")
```

A few things worth calling out when you combine the two:

- **Watch your `__init__` signature and MRO.** With multiple inheritance, keyword arguments need to flow cleanly to both parent constructors. Passing everything through `**kwargs` and calling `super().__init__(...)` once (rather than calling each parent explicitly) is the safest pattern.
- **Geometry stays geometry.** The LLM doesn't reason over raw `Shapely` objects — pull out the human-readable values you want it to consider (distance, risk score, neighbor count, coordinates as text) and put them in the `obs` dictionary, same as any other prompt.
- **CRS conversions are unaffected.** `to_crs()`, `distance()`, `get_neighbors_within_distance()`, and friends all work exactly as before, because Mesa-LLM never touches `GeoSpace`.
- **Vectorized geospatial pipelines (bulk GeoDataFrame operations) don't reason per-row automatically.** If you're processing thousands of agents from a GeoDataFrame, decide deliberately which subset actually needs LLM reasoning per step — calling an LLM for every agent, every step, at GIS scale gets expensive and slow fast (see the troubleshooting section below).

---

## 5. Adapting Agents to Use LLM-Powered Reasoning and Memory

### Reasoning modules

Reasoning is deliberately decoupled from the agent class so you can swap strategies without restructuring your model. Mesa-LLM currently ships `ReActReasoning` (reason-then-act, always returns a suggested action alongside its reasoning) with other strategies such as `CoTReasoning` (chain-of-thought) also available. You attach one via the `reasoning` argument when creating agents, and call it inside `step()`:

```python
plan = self.reasoning.plan(prompt=prompt, obs=observation, tools=[])
```

### Memory

Memory is optional but recommended for anything beyond a single-step demo — without it, every reasoning call starts from zero context. Mesa-LLM provides a few memory backends:

- **`STLTMemory`** — short-term/long-term memory, the default used in tutorials
- **`EpisodicMemory`** — remembers discrete past events/episodes
- **`LongTermMemory`** — persistent memory across the full run

Attach memory in your agent's `__init__`, not in `step()`, so it persists across the simulation:

```python
self.memory = STLTMemory(
    agent=self,
    llm_model="ollama/llama3",
    display=True,  # prints memory contents as they update — useful while migrating
)
```

If you're migrating an agent that previously tracked history manually (e.g., a list of past states you appended to each step), memory is where that logic now belongs — let the memory module own it instead of hand-rolling it.

### Tools

If your old Mesa agent's `step()` called out to helper methods (`self.move()`, `self.attack()`, `self.trade()`), those map naturally onto Mesa-LLM **tools** — functions the LLM can choose to invoke as part of its reasoning:

```python
from mesa_llm.tools import tool

@tool
def forage(agent) -> str:
    """Forage for food near the agent's current location."""
    agent.food_level += 2
    return f"{agent.unique_id} foraged and gained 2 food."
```

Pass tools in at agent creation (`tools=[forage]`) and Mesa-LLM handles exposing the schema to the LLM and routing the call back to your function.

---

## 6. Configuring LLM Providers and Models

Mesa-LLM talks to providers through LiteLLM, so switching providers is a one-line change, not a rewrite. Supported providers as of this writing: **OpenAI, Anthropic, xAI, Hugging Face, Ollama, OpenRouter, Novita AI, and Google Gemini.**

### Local development (no API key needed)

Tutorials default to **Ollama** running locally, which is the easiest way to migrate and test without incurring API costs:

```bash
ollama pull llama3
ollama serve
```

```python
llm_model = "ollama/llama3"
```

### Cloud providers

Set your key in a `.env` file:

```bash
# .env
OPENAI_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-api-key
```

Then reference the model in the `{provider}/{model}` format everywhere Mesa-LLM expects an `llm_model`:

```python
llm_model = "openai/gpt-4o"
# or
llm_model = "anthropic/claude-3-sonnet"
```

### Custom / self-hosted endpoints

If you're running a remote Ollama instance, vLLM, or LM Studio, pass `api_base` explicitly:

```python
from mesa_llm.module_llm import ModuleLLM

llm = ModuleLLM(
    llm_model="ollama_chat/llama3.2",
    system_prompt="You are a helpful agent.",
    api_base="http://192.168.1.100:11434",
)
```

`api_base` is supported consistently across `ModuleLLM`, `LLMAgent`, and every memory subclass, so you only need to set it in one place per agent type.

For direct, low-level LLM calls outside the agent reasoning loop (e.g., a quick one-off generation), `ModuleLLM` is the class to reach for:

```python
from mesa_llm.module_llm import ModuleLLM

llm = ModuleLLM(llm_model="openai/gpt-4o", system_prompt="You are a helpful simulation agent.")
response = llm.generate("What should I do next in this situation?")
```

---

## 7. Common Migration Issues and Troubleshooting

**"My agent throws an error about `llm_model` format."**
`llm_model` must always be `"{provider}/{model}"` — e.g. `"openai/gpt-4o"`, not just `"gpt-4o"`. This is the single most common first-time error.

**"It says my API key is missing, but I set it."**
Double-check the key is in `.env` (not just exported in your shell for a different session) and that the provider prefix in `llm_model` matches the key name (`OPENAI_API_KEY` for `openai/...`, `ANTHROPIC_API_KEY` for `anthropic/...`).

**"Ollama connection refused."**
The local server needs to actually be running at `http://localhost:11434` before you run your model — `ollama serve` in one terminal, your script in another. Run `ollama ls` to confirm the model you referenced was actually pulled.

**"My agent's reasoning output includes an action, but nothing happens in the model."**
This is expected with `ReActReasoning` in its default form — it *suggests* an action as part of the reasoning trace, but doesn't execute it automatically unless you wire that up yourself (typically via `tools` or by parsing `plan` and calling the matching method). This trips people up because it looks like a bug when it's actually the documented, staged design.

**"My model got extremely slow / expensive once I migrated."**
Every `step()` call that reaches `reasoning.plan()` is a real network round-trip to an LLM. If you migrated a model with hundreds of agents running many steps, you likely don't want *every* agent reasoning via LLM on *every* step. Common mitigations:
- Only give `LLMAgent` behavior to the subset of agents whose decisions actually benefit from language reasoning; leave the rest as plain rule-based `Agent`s.
- Use `agenerate()` / async batching to parallelize calls across agents within a step instead of reasoning sequentially.
- Reason less frequently (e.g., every N steps) rather than every step, if your model design allows it.

**"Combining `GeoAgent` and `LLMAgent` raises a `TypeError` about arguments."**
This is almost always a multiple-inheritance `__init__` ordering issue. Pass shared arguments through `**kwargs` and call `super().__init__()` exactly once, letting Python's MRO route arguments to both parent classes, rather than calling each parent's `__init__` directly.

**"Output isn't deterministic between runs, and I need reproducibility for my study."**
This is inherent to LLM-based reasoning, not a bug. If reproducibility matters for your research, log full reasoning traces via your `DataCollector` (or the memory module's `display=True` output) so runs are at least fully auditable, and consider fixing `seed` on the Mesa side for everything *except* the LLM calls themselves.

**"The API changed between versions and my migrated code broke."**
Mesa-LLM explicitly warns that it's under active development pre-1.0. Pin your version (`mesa-llm==x.y.z`) once your migration works, and check the [changelog/discussions](https://github.com/mesa/mesa-llm/discussions) before upgrading.

---

## 8. Further Reading

- [Mesa-LLM Getting Started](https://mesa-llm.readthedocs.io/en/latest/getting_started.html)
- [Mesa-LLM Overview (core concepts)](https://mesa-llm.readthedocs.io/en/latest/overview.html)
- [Creating Your First Mesa-LLM Model (tutorial)](https://mesa-llm.readthedocs.io/en/latest/tutorials/first_model.html)
- [Negotiation Model Tutorial (multi-agent LLM communication)](https://mesa-llm.readthedocs.io/en/latest/tutorials/negotiation_model_tutorial.html)
- [ModuleLLM / provider setup reference](https://mesa-llm.readthedocs.io/en/latest/apis/module_llm.html)
- [Mesa-LLM API Documentation](https://mesa-llm.readthedocs.io/en/latest/apis/api_main.html)
- [Mesa-Geo Documentation](https://mesa-geo.readthedocs.io/) — for anything geospatial that this guide didn't cover
- [Mesa-LLM GitHub Discussions](https://github.com/mesa/mesa-llm/discussions) and [Discord](https://discord.gg/fa5pEv3NxY) — for questions specific to your migration

---

## Summary

Migrating to Mesa-LLM is less "port your model" and more "decide which agents should think in language instead of `if/else`." Your model class, scheduling, spaces, and (if you're using Mesa-Geo) all your GIS plumbing stay put. What changes is scoped tightly to the agent: swap the base class, replace hard-coded decision logic with a `reasoning.plan()` call, and optionally add memory and tools where they genuinely add value. Start with one agent type, get it reasoning sensibly on a local Ollama model where mistakes are free, and only then decide which other agents — and how much of your compute budget — are worth handing over to an LLM.