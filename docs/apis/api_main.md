# API Reference

Complete reference documentation for all Mesa-LLM classes, functions, and modules.

## Core Modules

### LLMAgent (`llm_agent`)

The main agent class that integrates LLM reasoning into Mesa agents.

**Key Classes**:

- `LLMAgent` - Base agent with LLM-powered decision making

**Usage**: Inherit from `LLMAgent` to create intelligent agents that reason using LLMs.

```{toctree}
---
maxdepth: 2
---
llm_agent
```

### ModuleLLM (`module_llm`)

Unified interface to LLM providers through LiteLLM integration.

**Key Classes**:

- `ModuleLLM` - Provider abstraction layer

**Supports**: OpenAI, Anthropic, Gemini, Ollama, HuggingFace, OpenRouter, xAI, Novita AI

```{toctree}
---
maxdepth: 2
---
module_llm
```

## Reasoning Strategies

How agents think and make decisions.

### Reasoning Base (`reasoning`)

Abstract reasoning framework and base classes.

**Key Classes**:

- `Reasoning` - Abstract base class
- `Observation` - Environmental state
- `Plan` - Agent's planned action

```{toctree}
---
maxdepth: 2
---
reasoning
```

**Implementations**:

- `ReActReasoning` - Reason-Act loops
- `ChainOfThoughtReasoning` - Step-by-step planning
- `ReWOOReasoning` - Plan-first approach

## Memory Systems

Enable agents to learn and remember across steps.

### Memory (`memory`)

Agent memory systems for context retention.

**Key Classes**:

- `Memory` - Base interface
- `ShortTermMemory` - Recent observations
- `LongTermMemory` - Consolidated facts
- `STLTMemory` - Combined short + long term

```{toctree}
---
maxdepth: 2
---
memory
```

## Tools & Actions

Structured agent interactions with environments.

### Tools (`tools`)

Tool/action management system for agents.

**Key Classes**:

- `ToolManager` - Registry and tool management
- `@tool` - Decorator for tool functions

```{toctree}
---
maxdepth: 2
---
tools
```

## Analysis & Recording

Capture and analyze agent behavior.

### Recording (`recording`)

Automatic recording of agent reasoning and actions.

**Key Classes**:

- `@record_model` - Decorator for simulation recording
- `SimulationRecorder` - Record manager
- `AgentAnalysis` - Agent behavior analysis

```{toctree}
---
maxdepth: 2
---
recording
```

## Performance

Optimize simulation execution.

### Parallel Stepping (`parallel_stepping`)

Asynchronous agent execution for performance.

```{toctree}
---
maxdepth: 2
---
parallel_stepping
```

## Quick Reference

| Module              | Purpose                | Key Class                     |
| ------------------- | ---------------------- | ----------------------------- |
| `llm_agent`         | Base LLM agent         | `LLMAgent`                    |
| `module_llm`        | LLM provider interface | `ModuleLLM`                   |
| `reasoning`         | Thinking strategies    | `Reasoning`, `ReActReasoning` |
| `memory`            | Context retention      | `STLTMemory`                  |
| `tools`             | Agent actions          | `ToolManager`                 |
| `recording`         | Behavior capture       | `@record_model`               |
| `parallel_stepping` | Fast execution         | ParallelStepper               |
