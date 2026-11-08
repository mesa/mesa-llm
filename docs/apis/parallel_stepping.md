# Parallel Stepping

Parallel stepping is an advanced opt-in API for simulations whose agent steps
are safe to overlap. For standard Mesa-LLM examples and state-mutating
simulations, prefer normal sequential Mesa stepping:

```python
from mesa import Model


class MyModel(Model):
   def step(self):
      self.agents.shuffle_do("step")
```

Sequential stepping gives each agent a clear activation order. When agents
change shared state such as grid positions, messages, resources, memory, or
model-level counters, the next agent observes the effects of earlier agents in
the same model step.

## Why Parallel Stepping Is Advanced

Parallel execution can be useful for carefully designed workloads, but it is
not a drop-in default for state-mutating LLM simulations.

- Concurrent LLM calls can overload local Ollama servers, remote Ollama
  deployments, or other model providers. Large batches can cause timeouts,
  rate-limit errors, excessive memory use, or slower overall throughput.
- Concurrent choice can observe stale state. One agent may choose an action
  based on an environment snapshot that another agent changes before the first
  action is committed.
- The automatic mode monkey-patches Mesa's `AgentSet.shuffle_do()` method
  process-wide. That makes it convenient for experiments, but it also makes
  scheduling behavior less explicit in notebooks, tests, and larger apps.
- Parallel stepping does not add conflict resolution, snapshot isolation, or a
  deterministic commit protocol for simultaneous state changes.

Use tools only for read-only deliberation helpers, such as inspecting or
summarizing state for the LLM. State-changing behavior should be modeled as
actions and committed through `act(...)` or `execute_action(...)`, not as
provider tool calls. Use `choose_action(...)` when you need to inspect or veto
the selected action before committing it.

## Recommended State-Mutating Pattern

For ordinary agent-based models, keep Mesa's sequential scheduler and use
actions for committed behavior:

```python
from mesa import Model
from mesa_llm.actions import move_one_step, wait
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.cot import CoTReasoning


class MyAgent(LLMAgent):
   def __init__(self, model, **kwargs):
      super().__init__(
            model=model,
            reasoning=CoTReasoning,
            actions=[wait, move_one_step],
            **kwargs,
      )

   def step(self):
      obs = self.generate_obs()
      self.act(
            prompt=f"Choose one validated action for this observation:\n{obs}",
      )


class MyModel(Model):
   def step(self):
      self.agents.shuffle_do("step")
```

If an agent should deliberate before acting, compose that explicitly: run
`plan(...)` for read-only deliberation, then pass the resulting context into
`act(...)`. Keep the actual mutation at the action boundary.

## Advanced Automatic Mode

Automatic mode patches `AgentSet.shuffle_do()` and activates only when the
model has `parallel_stepping = True`.

```python
from mesa import Model
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping


class ParallelSafeModel(Model):
   def __init__(self):
      super().__init__()
      self.parallel_stepping = True
      enable_automatic_parallel_stepping(mode="asyncio")

   def step(self):
      # Advanced opt-in: only use when agent steps are safe to overlap.
      self.agents.shuffle_do("step")
```

Use this only when overlapping steps have clear semantics for your model. Good
candidates are read-only probes, independent external requests with explicit
rate limiting, or simulations that collect choices first and commit them later
using your own deterministic rules.

## Manual Parallel Control

Manual helpers make concurrency more visible than the automatic monkey-patch,
but they have the same state-safety and provider-capacity constraints.

```python
from mesa import Model
from mesa_llm.parallel_stepping import step_agents_parallel_sync


class CustomModel(Model):
   def step(self):
      citizen_agents = [a for a in self.agents if isinstance(a, Citizen)]
      cop_agents = [a for a in self.agents if isinstance(a, Cop)]

      # Advanced opt-in: ensure these steps are safe to overlap first.
      step_agents_parallel_sync(citizen_agents)
      step_agents_parallel_sync(cop_agents)
```

## Mode Selection

```python
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping

# Uses asyncio tasks for agents with async step methods.
enable_automatic_parallel_stepping(mode="asyncio")

# Uses ThreadPoolExecutor for thread-based execution.
enable_automatic_parallel_stepping(mode="threading")
```

`asyncio` mode is useful for I/O-bound experiments, including LLM requests, but
it still needs provider-side rate limits and model-level semantics for stale
observations. `threading` mode can help with compatibility or CPU-bound work,
but shared mutable model state still needs the same care.

## Future Scheduler Work

Future concurrency support for state-mutating simulations should be explicit
opt-in scheduler work with clear semantics: when observations are captured, how
actions are selected, how conflicts are resolved, and in what deterministic
order validated actions are committed. Prefer that kind of explicit scheduler
over transparent monkey-patching for simulations where correctness depends on
shared state.

## API Reference

```{eval-rst}
.. automodule:: mesa_llm.parallel_stepping
   :members:
   :undoc-members:
   :show-inheritance:
```
