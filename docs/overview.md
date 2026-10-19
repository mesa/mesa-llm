# Overview: Mesa-LLM Architecture & Features

[Mesa-LLM](https://github.com/mesa/mesa-llm) extends [Mesa](https://github.com/mesa) to enable LLM-based reasoning inside agents without modifying Mesa's core execution model, scheduling, or environment systems.

**Key insight**: Mesa-LLM changes HOW agents decide (using LLMs), not HOW models run (still using Mesa's standard workflow).

## Architecture: What's Added

| Component           | Mesa      | Mesa-LLM        |
| ------------------- | --------- | --------------- |
| Agents              | ✅        | ✅ LLMAgent     |
| Scheduling          | ✅        | ✅ Same         |
| Environments        | ✅        | ✅ Same         |
| **Decision Making** | Hardcoded | **LLM-powered** |
| **Memory**          | ❌        | **✅**          |
| **Tools**           | ❌        | **✅**          |

## Core Mesa-LLM Concepts

### 1. LLMAgent

`LLMAgent` is a Mesa Agent with LLM-powered reasoning. Standard Mesa lifecycle, but decisions delegated to LLMs.

#### 2. Reasoning

A reasoning module defines how an agent thinks, given:

- A prompt.
- Observations from the model or environment.
- Optional memory and tools.

Reasoning modules are attached to agents to encapsulate how language-based reasoning is performed. While Mesa-LLM currently provides `ReActReasoning`, the reasoning logic is kept separate from the agent to allow extensibility without changing agent structure.

#### 3. Memory

Memory allows agents to retain information across simulation steps.
In Mesa-LLM, memory is useful for reasoning-based agents. It enables agents to:

- Reference past observations or interactions.
- Maintain more consistent behavior over time.
- Produce more coherent and contextual reasoning.

Memory is optional in Mesa-LLM. When present, it allows agents to retain information across steps and reason with context. When absent, agents reason purely from the current observation and prompt.

#### 4. Tools

Tools provide a structured way for agents to express or request actions when the simulation supports action execution.

Tools are optional and typically become relevant only when:

- An environment exists (e.g. a grid).
- Actions can be meaningfully executed.

In introductory models, action suggestions may appear only as part of the reasoning trace and are not executed. Mesa-LLM does not introduce new space or environment abstractions; it relies entirely on Mesa’s existing space modules.

#### 5. Execution Model

Mesa-LLM follows Mesa’s standard execution model.
The simulation advances through calls to `model.step()`, and agents are activated using Mesa’s scheduling mechanisms.

When an agent is activated, its `step()` method is called.
In Mesa-LLM, this method typically includes language-based reasoning before deciding what to do next.
