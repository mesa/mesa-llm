# Getting Started
Mesa-LLM is an extension of the Mesa agent-based modeling framework that enables language-model-based reasoning inside agents, while preserving Mesa’s execution model, scheduling, and environments.

Mesa-LLM allows agents to reason using natural language prompts, enabling more flexible, interpretable, and adaptive decision-making within structured simulations.

Agents in Mesa-LLM are still standard Mesa agents. The only difference is how decisions are made, not how models run.

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

## Source Code
- [Mesa-LLM Github Repository](https://github.com/mesa/mesa-llm)

## Community and Support
- [Mesa-LLM Discussion](https://github.com/mesa/mesa-llm/discussions)
- [Discord](https://discord.gg/fa5pEv3NxY)
