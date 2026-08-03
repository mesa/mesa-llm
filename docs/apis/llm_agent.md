# LLMAgent

LLMAgent is the core agent class in Mesa-LLM that extends Mesa's base Agent class with Large Language Model capabilities. It provides a complete framework for creating intelligent agents that can reason, remember, communicate, and act in simulations using natural language processing.

## Basic Agent Implementation

```python
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.cot import CoTReasoning

class MyAgent(LLMAgent):
   def __init__(self, model, **kwargs):
      super().__init__(
            model=model,
            reasoning=CoTReasoning,
            llm_model="openai/gpt-4o",
            system_prompt="You are a helpful agent in a simulation.",
            vision=2,  # See 2 cells in each direction
            internal_state=["curious", "cooperative"],
            step_prompt="Decide what to do next based on your observations.",
            api_base=None,  # Set to a custom URL for self-hosted LLMs (e.g., "http://192.168.1.100:11434")
      )

      # You can override default memory with EpisodicMemory (default is STLTMemory)
      self.memory = EpisodicMemory(
            agent=self,
            llm_model="openai/gpt-4o-mini",
            max_memory=20
      )

   def step(self):
      # Generate current observation
      obs = self.generate_obs()

      # Use reasoning to create plan
      plan = self.reasoning.plan(obs=obs)

      # Execute the plan
      self.apply_plan(plan)
```

## Parallel Execution Example

```python
class ParallelAgent(LLMAgent):
   async def astep(self): # Use 'async' with astep method to enable parallel execution
      """Asynchronous step for parallel execution"""
      obs = self.generate_obs()
      plan = await self.reasoning.aplan( # Use 'await' with aplan method to enable parallel execution
            prompt=self.step_prompt,
            obs=obs
      )
      self.apply_plan(plan)
```

## Agent Communication Example

```python
def step(self):
   obs = self.generate_obs()

   # Find nearby agents
   neighbors = [agent for agent in obs.local_state.keys()]
   if neighbors:
      # Send message to neighbors
      neighbor_ids = [int(name.split()[-1]) for name in neighbors]
      self.send_message("Hello neighbors!",
                        [agent for agent in self.model.agents
                        if agent.unique_id in neighbor_ids])

   plan = self.reasoning.plan(obs=obs)
   self.apply_plan(plan)
```

## Post-commit observer failures

A normal return from an action commits its state change before Mesa-LLM records the result in memory and an optional recorder. If either post-commit observer raises an `Exception`, Mesa-LLM still attempts the other configured observer once, then raises `ActionPostCommitError`, available from `mesa_llm.actions` and top-level `mesa_llm`. The error reports `committed=True`, the validated `action`, the committed `result`, and the original exception objects in `observer_errors` under `memory` and/or `recorder`.

This error means the action is already committed but may not be fully recorded. Do not retry the action in response: Mesa-LLM does not retry or roll back the action or its observers.

## API Reference

```{eval-rst}
.. automodule:: mesa_llm.llm_agent
   :members:
   :undoc-members:
   :show-inheritance:
```
