# Developer guide
This guide explains how `mesa-llm` is designed, how core modules interact during a simulation, and the engineering standards contributors should follow.

Take a look at (CONTRIBUTING_GUIDE)[] to take first Step

## 1. Design Principle
* **Mesa-First Compatibility:** `mesa-llm` extends standard Mesa primitives (`Model`, `Agent`, `AgentSet`) rather than replacing them.
* **Provider Agnosticism:** All LLM interactions route through a unified backend (LiteLLM) so simulations stay decoupled from specific vendors (OpenAI, Anthropic, Ollama, Gemini).
* **Deterministic Environments:** Spatial state, scheduling order, and non-LLM randomness remain reproducible via `model.rng`.
* **Cost-Safe Testing:** Core unit tests must execute offline using mocked LLM responses rather than live paid API endpoints.

Go to (Overview)[], get to know about Core Component.

## Before Begin

- Look at
- Look at
- Look at

## How Mesa-llm is Structured

