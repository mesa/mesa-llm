# Reasoning System

The reasoning system in Mesa-LLM provides different cognitive strategies for agents to analyze situations, make decisions, and plan actions. It forms the core intelligence layer that transforms observations into actionable plans using structured thinking approaches. The reasoning module enables agents to process environmental observations and memory context into executable action plans through various cognitive frameworks.

## Usage in Mesa Simulations

```python
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.cot import CoTReasoning
from mesa_llm.tools import tool

@tool
def arrest_citizen(agent, citizen_id: int) -> str:
   """Arrest a citizen in this model.
   Args:
      agent: The agent making the request (provided automatically)
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
            tools=[arrest_citizen],
            **kwargs
      )

   def step(self):
      # Generate observation and create plan using reasoning strategy
      obs = self.generate_obs()
      plan = self.reasoning.plan(
            obs=obs,
            tools=["arrest_citizen"]
      )
      self.apply_plan(plan)

# Strategy-specific configurations
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.reasoning.rewoo import ReWOOReasoning

# For ReWOO with multi-step planning
plan = self.reasoning.plan(obs=obs, ttl=3)  # Plan valid for 3 steps

# Parallel reasoning execution
async def astep(self):
   obs = self.generate_obs()
   plan = await self.reasoning.aplan(
      prompt=self.step_prompt,
      obs=obs,
      tools=["arrest_citizen"]
   )
   self.apply_plan(plan)
```

Omitting per-call `tools` inherits the tools configured on the agent. Passing `tools=None` or `tools=[]` exposes no tools for that reasoning call. Passing `tools=[...]` narrows the configured set and fails fast if a named or callable tool was not configured on the agent first.

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
