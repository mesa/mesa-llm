# Epstein Civil Violence

## About this Tutorial
This tutorial is inspired by official [Mesa LLM Epstein Civil Violence Model](https://github.com/mesa/mesa-llm/tree/main/examples/epstein_civil_violence)

The goal of this tutorial is **not to replicate** the full original example, but to provide a simplified and beginner-friendly version. It focuses on explaining how agents interact and how the reasoning process works, helping new users understand how the model functions step by step.

#### Key Difference from original Tutorial
- Reduced no. of agent
- No SolarViz (UI)

This makes the example easier to follow for new users while preserving the core concept of the library.

### Model Description
This model shows how protests can start and spread in a society.

There are two types of people:
- `Citizens` – normal people in society
- `Cops` – police who try to control protests

Citizens move around randomly and decide whether to `protest` or stay `quiet`. Their decision depends on:
- How difficult their life is (`hardship`)
- How much they trust the government (`legitimacy`)
- How scared they are of getting arrested (`risk`)

If a `citizen` feels unhappy and thinks it is safe, they may start protesting. Cops move around and look for people who are protesting. When they find one, they arrest them.

### Citizen Class
Each citizen is an agent that inherits from `LLMAgent`, meaning they can observe, think, and take actions.

Every citizen has the following characteristics:
- `Hardship` – how difficult their life is
- `Risk Aversion` – how scared they are of punishment
- `Regime Legitimacy` – how much they trust the government
- `Grievance` – overall dissatisfaction (based on hardship and legitimacy)

Estimated Arrest Probability – chance of getting arrested, based on:
- Number of cops nearby
- Number of active (rebelling) **citizens** nearby

A citizen decides to rebel when:
- Their **grievance** is high, and
- The risk of being arrested is low

If the risk is **high**, they choose to stay quiet even if they are unhappy.

### Cop Class
Each cop also inherit from `LLMAgent` whose job is to control protests. Cops move around and watch nearby areas.

If they find a protesting `citizen`:
- They arrest them
- The citizen goes to **jail**
- The cop moves to that position

### Tutorial Setup
Ensure you are using python 3.14 or later

## Model Execution
At each model step:
- The model advances one step
- All agents are activated using `shuffle_do("step")` and moving in a grid.
- Each agent generate an reasoning plan and choose an tool using `ReactReasoning` and applies it

## Important Dependencies (for this tutorial)
``` python
from enum import Enum
from mesa_llm.tools.tool_decorator import tool
from mesa_llm.tools.tool_manager import ToolManager
import math
import mesa
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
import random
from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from mesa_llm.reasoning.reasoning import Reasoning
```

## Creating the CitizenState
Using the previously dependencies, We define `CitizenState` Class. Each agent have a state, `QUIET` is the default state, State can be change during execution.

``` python
class CitizenState(Enum):
    QUIET = 1
    ACTIVE = 2
    ARRESTED = 3
```

## Custom Tools
In this tutorial, we use custom tools that agents can execute during simulation, as shown in the output.
First, you need to define a `tool manager` like this:
``` python
citizen_tool_manager = ToolManager()
```
The `@tool` decorator is used to convert Python functions into **LLM-compatible tools**. It automatically generates the required JSON schema using the function’s type hints and `docstrings`.

#### While creating custom tools, keep these things in mind:
- Always create and assign a `Tool Manager` as you seen above
- Use the `@tool` decorator to convert a normal Python function into an LLM-compatible tool
- Always include a `docstring` in your function
👉 Without a `docstring`, the model may throw an error

### Tools used in this Tutorial

``` python
@tool(tool_manager=citizen_tool_manager)
def change_state(agent, state: str) -> str:
    """
    Change the citizen state to either QUIET or ACTIVE.

    Args:
        state: The new state for the citizen. Must be either "QUIET" or "ACTIVE".
        agent: Provided automatically.

    Returns:
        A confirmation message describing the citizen's new state.
    """

    # Normalize input to avoid case issues (e.g., "active", "Active")
    normalized_state = state.upper()

    # Map string input to actual Enum values
    state_map = {
        "QUIET": CitizenState.QUIET,
        "ACTIVE": CitizenState.ACTIVE,
    }

    # Validate input state
    if normalized_state not in state_map:
        raise ValueError(f"Invalid state: {state}")

    # Update agent state
    agent.state = state_map[normalized_state]

    # Sync internal_state so LLM has updated context
    # (important for correct reasoning in next step)
    agent.sync_state_note()

    # Return confirmation (used in LLM tool feedback loop)
    return f"agent {agent.unique_id} changed state to {normalized_state}."

cop_tool_manager = ToolManager()
@tool(tool_manager=cop_tool_manager)
def arrest_citizen(agent: "LLMAgent", citizen_id: int) -> str:
    """
    Arrest a citizen (only if they are active).

        Args:
            citizen_id: The unique id of the citizen to arrest.
            agent: Provided automatically

        Returns:
            a string confirming the citizen's arrest.
    """
    citizen = next(
        (agent for agent in agent.model.agents if agent.unique_id == citizen_id), None
    )
    citizen.state = CitizenState.ARRESTED
    citizen.jail_sentence_left = random.randint(1, agent.max_jail_term)
    citizen.sync_state_note()
    return f"agent {citizen_id} arrested by {agent.unique_id}."
```

## Creating Citizen Class
Each agent can calculate their **arrest probability**. If their arrest probability is greater then `threshold` they can be arrested by **cop**. The default **Neighbours** state is `ACTIVE`. All agent can move freely in the **grid**.
``` python
class Citizen(LLMAgent, mesa.discrete_space.CellAgent):
    # A citizen agent that can either remain quiet or become active (rebel)
    # Uses BOTH mathematical model (Epstein rules) + LLM reasoning

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        step_prompt,
        arrest_prob_constant=0.5,   # constant used in arrest probability formula
        regime_legitimacy=0.5,     # global perception of government legitimacy
        threshold=0.5,             # rebellion threshold
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
        )

        # --- Core Epstein parameters ---
        self.hardship = self.random.random()   # personal suffering (0–1)
        self.risk_aversion = self.random.random()  # fear of being arrested (0–1)

        self.regime_legitimacy = regime_legitimacy
        self.threshold = threshold

        # --- State variables ---
        self.state = CitizenState.QUIET   # default state
        self.vision = vision              # how far agent can see
        self.jail_sentence_left = 0       # jail timer

        # --- Derived values ---
        # grievance = how angry the agent is at the system
        self.grievance = self.hardship * (1 - self.regime_legitimacy)

        self.arrest_prob_constant = arrest_prob_constant
        self.arrest_probability = None   # will be updated dynamically

        # --- LLM Memory ---
        self.memory = STLTMemory(
            agent=self,
            display=True,
            llm_model=llm_model,
        )

        # --- Internal state (used by LLM for reasoning context) ---
        self.internal_state.append(
            f"risk aversion is {self.risk_aversion:.4f} (0 to 1)"
        )
        self.internal_state.append(
            f"rebellion threshold is {self.threshold:.4f}"
        )
        self.internal_state.append(
            f"grievance level is {self.grievance:.4f}"
        )
        self.internal_state.append(
            f"current state is {self.state}"
        )

        # --- Tool manager for actions ---
        self.tool_manager = citizen_tool_manager

    def update_estimated_arrest_probability(self):
        # Estimate probability of being arrested based on local neighborhood

        cops_in_vision = 0
        actives_in_vision = 1  # include self

        # Get nearby agents within vision radius
        neighbors = self.model.grid.get_neighbors(
            tuple(self.pos), moore=True, include_center=False, radius=self.vision
        )

        # Count cops and active rebels
        for i in neighbors:
            if isinstance(i, Cop):
                cops_in_vision += 1
            elif i.state == CitizenState.ACTIVE:
                actives_in_vision += 1

        # Epstein formula for arrest probability
        self.arrest_probability = 1 - math.exp(
            -1 * self.arrest_prob_constant * round(cops_in_vision / actives_in_vision)
        )

        # Update internal_state for LLM context
        self.internal_state = [
            item for item in self.internal_state
            if not item.lower().startswith("my arrest probability is")
        ]

        self.internal_state.append(
            f"my arrest probability is {self.arrest_probability:.4f}"
        )

    def sync_state_note(self):
        # Keep the citizen's textual internal state aligned with the Enum state
        self.internal_state = [
            item
            for item in self.internal_state
            if not item.lower().startswith("current state is")
        ]
        self.internal_state.append(f"current state is {self.state}")

    def step(self):
        # Main simulation step for each agent

        if self.jail_sentence_left == 0:
            # Agent is free → can act

            # Update perceived arrest probability
            self.update_estimated_arrest_probability()

            # Generate observation for LLM
            observation = self.generate_obs()

            # LLM decides action using available tools
            plan = self.reasoning.plan(
                prompt=self.step_prompt,
                obs=observation,
                selected_tools=["change_state", "move_one_step"],
            )

            # Execute the chosen plan
            self.apply_plan(plan)

        else:
            # Agent is in jail → cannot act
            self.jail_sentence_left -= 1

            # If sentence completed → release
            if self.jail_sentence_left <= 0:
                self.jail_sentence_left = 0
                self.state = CitizenState.QUIET

                # Sync state info for LLM
                self.sync_state_note()
```
## Cop class
`Cops` move around and look for nearby `citizens` who are actively **rebelling**. If they find one, they arrest them and move to that position. In simple terms, cops control protests by catching and **jailing** active rebels.

``` python
class Cop(LLMAgent, mesa.discrete_space.CellAgent):
    """
    A cop agent responsible for maintaining order.
    Rule: Arrest nearby active citizens, otherwise move.
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        step_prompt,
        max_jail_term=2,  # maximum jail duration for arrested citizens
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
        )

        # Maximum jail time a cop can assign to a citizen
        self.max_jail_term = max_jail_term

        # Define behavior/personality for LLM reasoning
        # (overrides passed system_prompt for strict role behavior)
        self.system_prompt = (
            "You are a cop in a country experiencing civil violence. "
            "Your job is to arrest ACTIVE citizens only. "
            "You can move or arrest."
        )

        # Memory module for storing past observations/actions
        self.memory = STLTMemory(
            agent=self,
            display=True,
            llm_model=llm_model,
        )
        self.tool_manager = cop_tool_manager

    def step(self):
        """
        Main logic:
        1. Look for active citizens nearby
        2. Arrest if found (rule-based)
        3. Otherwise use LLM to decide movement
        """

        # Get all nearby agents within vision radius
        neighbors = self.model.grid.get_neighbors(
            tuple(self.pos), moore=True, include_center=False, radius=self.vision
        )

        # Filter only ACTIVE citizens
        active_citizens = [
            agent
            for agent in neighbors
            if isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE
        ]

        # --- Rule-based arrest (no LLM call → cost efficient) ---
        if active_citizens:
            # Randomly pick one active citizen
            citizen = self.random.choice(active_citizens)

            # Change citizen state to ARRESTED
            citizen.state = CitizenState.ARRESTED

            # Assign jail time (random between 1 and max_jail_term)
            citizen.jail_sentence_left = self.random.randint(1, self.max_jail_term)

            # Update citizen's internal state for LLM consistency
            citizen.sync_state_note()

            return  # skip LLM reasoning when arrest happens

        # --- LLM reasoning (only when no arrest possible) ---
        observation = self.generate_obs()

        # Decide movement using LLM
        plan = self.reasoning.plan(
            prompt=self.step_prompt,   # gives context to LLM
            obs=observation,
            selected_tools=["move_one_step","arrest_citizen"],  # restrict tools for control
        )

        # Execute chosen action
        self.apply_plan(plan)

```
## Creating EpsteinModel Class
The EpsteinModel sets up and runs the civil unrest simulation between citizens and cops.
It creates the agents, places them in the environment, and controls how the simulation progresses step by step.

### How it works:
- Inherits from Mesa’s `Model` class.
- Creates multiple citizen agents with different `hardship`, `risk`, and `grievance levels`.
- Creates cop agents to control and suppress rebellion.
- Places all agents randomly on the `grid`.
- Assigns `ReActReasoning` to agents so they can make decisions using an `LLM`.
- Updates all agents each step so they can move, act, or interact.
- Runs the simulation step by step to show how protests **emerge** and **spread**.

``` python
class EpsteinModel(Model):
    def __init__(
        self,
        initial_cops: int = 2,
        initial_citizens: int = 6,
        width: int = 10,
        height: int = 10,
        reasoning: type[Reasoning] | None = None,
        llm_model: str = "ollama/llama3.1:latest",
        vision: int = 3,
        seed=None,
    ):
        # If no reasoning is provided, default to ReAct (LLM reasoning strategy)
        if reasoning is None:
            from mesa_llm.reasoning.react import ReActReasoning
            reasoning = ReActReasoning

        # Initialize base Mesa model with seed (for reproducibility)
        super().__init__(seed=seed)

        # --- Grid setup ---
        self.width = width
        self.height = height

        # MultiGrid: agents can share same cell
        self.grid = MultiGrid(self.height, self.width, torus=False)

        # --- Data Collection (Model-level metrics) ---
        model_reporters = {
            # Count number of active (rebelling) citizens
            "active": lambda m: sum(
                1
                for agent in m.agents
                if isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE
            ),

            # Count number of quiet citizens
            "quiet": lambda m: sum(
                1
                for agent in m.agents
                if isinstance(agent, Citizen) and agent.state == CitizenState.QUIET
            ),

            # Count number of arrested citizens
            "arrested": lambda m: sum(
                1
                for agent in m.agents
                if isinstance(agent, Citizen) and agent.state == CitizenState.ARRESTED
            ),
        }

        # --- Agent-level metrics ---
        agent_reporters = {
            # Remaining jail time for each agent
            "jail_sentence": lambda a: getattr(a, "jail_sentence_left", None),

            # Agent’s perceived arrest probability
            "arrest_probability": lambda a: getattr(a, "arrest_probability", None),
        }

        # Initialize DataCollector
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters
        )

        # --------------------- Create Cop Agents ---------------------

        # System prompt defines behavior/personality of cops
        cop_system_prompt = (
            "You are a cop. You are tasked with arresting citizens if they are active "
            "and their arrest probability is high enough. You are also tasked with "
            "moving to a new location if there is no citizen in sight."
        )

        # Create multiple Cop agents using helper method
        agents = Cop.create_agents(
            self,
            n=initial_cops,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=cop_system_prompt,
            vision=vision,
            internal_state=None,
            step_prompt="Inspect your local vision and arrest a random active agent. Move if applicable.",
        )

        # Randomly place cops on grid (using model RNG for reproducibility)
        x = self.rng.integers(0, self.grid.width, size=(initial_cops,))
        y = self.rng.integers(0, self.grid.height, size=(initial_cops,))
        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

        # --------------------- Create Citizen Agents ---------------------

        # Create Citizen agents
        agents = Citizen.create_agents(
            self,
            n=initial_citizens,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="",
            vision=vision,
            internal_state=None,
            step_prompt="Move around and change your state if the conditions indicate it.",
        )

        # Random placement of citizens
        x = self.rng.integers(0, self.grid.width, size=(initial_citizens,))
        y = self.rng.integers(0, self.grid.height, size=(initial_citizens,))
        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

    def step(self):
        # Print step info (for debugging / visualization)
        print(
            f"\n[bold purple] step {self.steps} ─────────────────────────────────────────────────────────[/bold purple]"
        )

        # Shuffle activation → random order of agent execution
        # Prevents bias (important in ABM)
        self.agents.shuffle_do("step")

        # Collect metrics after each step
        self.datacollector.collect(self)
```
## Running the Model
Running the model for a few steps
``` python
if __name__ == "__main__":
    # Create an instance of the EpsteinModel
    model = EpsteinModel()

    # Run the model for a fixed number of steps
    # Each step activates all agents once and prints their reasoning output
    for _ in range(3):
        model.step()
```
## Output
``` bash
╭─ Step 3 | Citizen 4 ─────────────────────────────────────────────╮
│ [Observation]                                                    │
│ └── self_state:                                                  │
│     ├── agent_unique_id : 4                                      │
│     ├── system_prompt :                                          │
│     ├── location : (0, 2)                                        │
│     └── internal_state:                                          │
│         ├── risk aversion : 0.0499 (0 to 1)                      │
│         ├── rebellion threshold : 0.5000                         │
│         ├── grievance level : 0.1833                             │
│         ├── current state : CitizenState.ACTIVE                  │
│         └── arrest probability : 0.0000                          │
│                                                                 │
│ [Plan]                                                          │
│ ├── reasoning : My grievance level (0.1833) is below the         │
│ │   rebellion threshold (0.5000), so no unrest.                 │
│ └── action : change_state(state='ACTIVE')                        │
│                                                                 │
│ [Action]                                                        │
│ └── tool_call : change_state → agent 4 changed state to ACTIVE   │
╰─────────────────────────────────────────────────────────────────╯
```
## Exercises
Try the following exercises to better understand Mesa-llm:

1. **Increase the number;s of steps**
   Run the `model` for a larger number of steps and observe how agent behavior evolves over time.

2. **Create More Custom Tools**
   Extend the system by adding your own `tools` for agents.

3. **Change Citizen & Cop Parameter**
   Experiment with different `parameters` to see how behavior changes.

4. **Change Prompt**
   Modify the system prompt given to agents and Understand how prompts influence agent `reasoning` and decisions















