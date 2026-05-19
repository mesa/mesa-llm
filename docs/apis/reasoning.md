# Reasoning System

The reasoning system in Mesa-LLM provides different cognitive strategies for agents to analyze situations, make decisions, and plan next steps. It forms the core intelligence layer that transforms observations and memory context into structured deliberation through different cognitive frameworks.

Use `tools=` for read-only deliberation helpers, such as inspecting local state or computing derived context for the LLM. Committed state-changing behavior should be modeled as actions and executed through `act(...)`, `choose_action(...)`, or `execute_action(...)`, not as provider tool calls.

## Usage in Mesa Simulations

```python
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.actions import action
from mesa_llm.reasoning.cot import CoTReasoning
from mesa_llm.tools import tool

@tool
def inspect_neighborhood(agent) -> str:
   """Return read-only local context for deliberation.
   Args:
      agent: The agent making the request (provided automatically).
   Returns:
      A text summary of nearby state.
   """
   return agent.describe_neighborhood()

@action
def arrest_citizen(agent, citizen_id: int) -> str:
   """Arrest a citizen in this model.
   Args:
      agent: The agent making the request (provided automatically).
      citizen_id: Citizen to arrest.
   Returns:
      Arrest status.
   """
   return agent.arrest_citizen(citizen_id)

class MyAgent(LLMAgent):
   def __init__(self, model, **kwargs):
      super().__init__(
            model=model,
            reasoning=CoTReasoning,  # Specify reasoning strategy
            tools=[inspect_neighborhood],
            actions=[arrest_citizen],
            **kwargs
      )

   def step(self):
      # Generate observation and deliberate using read-only tools.
      obs = self.generate_obs()
      plan = self.plan(
            prompt="Review the local situation before choosing an action.",
            obs=obs,
            tools=["inspect_neighborhood"]
      )

      # Commit one validated local action. Actions are not provider tools.
      self.act(
            prompt=[f"OBSERVATION:\n{obs}", f"PLAN:\n{plan}"],
            actions=["arrest_citizen"]
      )

# Strategy-specific configurations
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.reasoning.rewoo import ReWOOReasoning

# For ReWOO with multi-step planning
plan = self.reasoning.plan(obs=obs, ttl=3)  # Plan valid for 3 steps

# Async reasoning execution
async def astep(self):
   obs = self.generate_obs()
   plan = await self.reasoning.aplan(
      prompt="Review the local situation before choosing an action.",
      obs=obs,
      tools=["inspect_neighborhood"]
   )
   await self.aact(
      prompt=[f"OBSERVATION:\n{obs}", f"PLAN:\n{plan}"],
      actions=["arrest_citizen"]
   )
```

Omitting per-call `tools` inherits the tools configured on the agent. Passing `tools=None` or `tools=[]` exposes no tools for that reasoning call. Passing `tools=[...]` narrows the configured set and fails fast if a named or callable tool was not configured on the agent first.

Omitting per-call `actions` on action workflow methods inherits the actions configured on the agent. Passing `actions=None` or `actions=[]` exposes no actions for that call. Passing `actions=[...]` narrows the configured action set and fails fast if a named or callable action was not configured on the agent first.

## Base abstractions

```{eval-rst}
.. automodule:: mesa_llm.reasoning.reasoning
   :members:
   :undoc-members:
   :show-inheritance:
```

## Reasoning strategies

```{eval-rst}
.. automodule:: mesa_llm.reasoning.cot
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mesa_llm.reasoning.react
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mesa_llm.reasoning.rewoo
   :members:
   :undoc-members:
   :show-inheritance:
```
