# Getting Started with Mesa-LLM

Mesa-LLM is an extension of the Mesa agent-based modeling framework that enables language-model-based reasoning inside agents, while preserving Mesa's execution model, scheduling, and environments.

Mesa-LLM allows agents to reason using natural language prompts, enabling more flexible, interpretable, and adaptive decision-making within structured simulations. Agents in Mesa-LLM remain standard Mesa agents—the only difference is how decisions are made, not how models run.

## Installation

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- An LLM provider account or local setup (optional for Ollama)

### Install from PyPI (Recommended)

```bash
pip install -U mesa-llm
```

### Install Development Version

```bash
pip install -U -e git+https://github.com/mesa/mesa-llm.git#egg=mesa-llm
```

### Setting Up an LLM Provider

Mesa-LLM supports 8+ providers. Start with a local free option:

**Option 1: Local LLM (Ollama - Free)**

```bash
# Install from https://ollama.ai
ollama pull llama2
# Then use: llm_model="ollama/llama2"
```

**Option 2: Cloud LLM (OpenAI - requires API key)**

```bash
# Create .env file with:
# OPENAI_API_KEY="your-key-here"
# Then use: llm_model="openai/gpt-4o-mini"
```

Other providers: Anthropic, Google Gemini, HuggingFace, xAI, Novita AI, OpenRouter.

## Your First Model (2 Minutes)

Create `first_model.py`:

```python
from mesa.model import Model
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning

class MyAgent(LLMAgent):
    def step(self):
        obs = {"step": self.model.steps}
        plan = self.reasoning.plan("What should I do?", obs, [])
        print(plan.response)

class MyModel(Model):
    def __init__(self):
        super().__init__()
        for _ in range(2):
            MyAgent(self, ReActReasoning, "ollama/llama2")

    def step(self):
        for agent in self.agents:
            agent.step()
        self.steps += 1

model = MyModel()
model.step()
```

Run: `python first_model.py`

## Overview

If you want a high-level understanding of Mesa-LLM structure and capabilities, start here:

- [Overview of Mesa-LLM Library](overview.md)

## LLM Backend Setup
Mesa-LLM leverages various LLM providers through the LiteLLM library. To run the examples and tutorials, you typically need one of the following:

- **Local LLM (Default in Tutorials)**: [Ollama](https://ollama.com) is used for local inference. It must be installed, and the local server must be running at `http://localhost:11434`.

- **Cloud LLM**: Providers like OpenAI, Anthropic, or Gemini require an API key set in your environment variables.

Refer to the [Tutorial Setup section](tutorials/first_model.md#tutorial-setup) in the first tutorial for more details on Ollama setup.

For provider-specific setup, including cloud API keys and custom Ollama endpoints, see [Basic LLM Setup](apis/module_llm.md#basic-llm-setup) and [Custom API Endpoints](apis/module_llm.md#custom-api-endpoints).

## Tutorials

If you want to learn Mesa-LLM step by step, follow these tutorials:

- [Creating your First Mesa-LLM Model](tutorials/first_model.md)
  Learn how to define a minimal LLMAgent that reasons using an LLM while remaining a standard Mesa agent.

- [Negotiation Model Tutorial](tutorials/negotiation_model_tutorial.md)
  Learn how multiple LLM-powered agents reason, communicate, and negotiate within a shared model.

## Examples

Mesa-LLM ships with example models demonstrating how language-based reasoning integrates with classic agent-based modeling patterns.

These examples are useful if you are already familiar with Mesa and want to see how LLM-powered agents behave in practice. You can find them here:
[Mesa-LLM Examples](examples.md)

## Troubleshooting

### `ModuleNotFoundError: No module named 'mesa_llm'`

```bash
pip install -U mesa-llm
```

### `LLM provider authentication failed`

1. Check your API key in `.env`
2. For Ollama: start with `ollama serve`
3. Restart Python after changing `.env`

### `Timeout or connection refused`

```bash
# Start Ollama (if using local LLM)
ollama serve
```

## Performance Tips

- **Development**: Use `ollama/llama2` (free, private)
- **Production**: Use `openai/gpt-4o-mini` (fast, cheap)
- **Accuracy**: Use larger models like `openai/gpt-4o`

## Source Code

- [Mesa-LLM Github Repository](https://github.com/mesa/mesa-llm)

## Community and Support

- [Mesa-LLM Discussion](https://github.com/mesa/mesa-llm/discussions)
- [Discord](https://discord.gg/fa5pEv3NxY)
