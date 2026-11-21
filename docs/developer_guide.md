# Developer Guide

This guide explains how **Mesa-LLM** is designed, how its core components interact during a simulation, and how developers can extend, test, and contribute to the project.

Whether you are fixing a bug, introducing a new reasoning strategy, integrating an LLM provider, or improving memory capabilities, this guide will help you understand the codebase and follow the project's engineering practices.

**Before contributing:** Start with the [Contributing Guide](CONTRIBUTING.md) to understand the contribution process.

### What Problem Does Mesa-LLM Solve?

`Mesa` often implement agent behavior through predefined rules and decision-making logic. Although this works well for many simulations, designing agents that can respond flexibly to complex situations, communicate naturally, or make context-dependent decisions can be challenging.

**Mesa-LLM** extends the Mesa framework by introducing LLM-powered reasoning into agents. Instead of relying exclusively on predefined decision rules, agents can use language models to interpret observations, reason about situations, and generate decisions.

This makes it possible to explore richer agent behaviors, including communication, negotiation, and context-aware decision-making.

For a broader understanding of Mesa's architecture, see the [Mesa Developer Documentation](https://mesa.readthedocs.io/en/latest/development/).

## 2. Architecture Overview

`Mesa-LLM` introduces LLM-powered reasoning through LLMAgent and its associated components, while retaining `Mesa` existing simulation infrastructure.

It builds on Mesa's existing simulation infrastructure, including:

- Models and agents
- AgentSet and scheduling
- Spatial environments
- Simulation time progression
- Data collection

Mesa-LLM adds LLM-based reasoning as a separate component that can be used by Mesa agents.

For a detailed explanation of the architecture and component responsibilities, see the [Mesa-LLM Overview](https://github.com/mesa/mesa-llm/blob/main/docs/overview.md).

## 3. Development Practices

- **Mesa-First Compatibility:** Extend standard Mesa primitives rather than replacing the existing simulation architecture.
- **Provider Agnosticism:** Keep LLM interactions independent of specific providers through the existing provider integration layer.
- **Separation of Concerns:** Keep agent behavior, reasoning, memory, and provider communication as distinct responsibilities.
- **Reproducibility:** Keep simulation logic and tests reproducible wherever possible. Use controlled inputs and mocked LLM responses when testing LLM-dependent functionality. Remember that external LLM responses may be nondeterministic.
- **Cost-Safe Testing:** Avoid unnecessary external API calls in unit tests. Mock LLM responses where appropriate to make tests faster, more reliable, and less dependent on provider availability or API costs.

## Choosing the right contribution path

1. **Bug Fix** - Identify and reproduce the issue, investigate its root cause, create or refer to a GitHub issue, add a regression test, and submit a PR.

2. **Small Feature** - Explain the use case, discuss the proposed approach when necessary, then implement and test it.

3. **Major feature or API change** - Open a (GitHub Discussion)[https://github.com/mesa/mesa-llm/discussions] or issue first. Explain the problem, proposed design, compatibility implications, and alternatives. Get maintainer feedback before substantial implementation.

4. **New reasoning or memory approach** - Explain how it improves existing capabilities, its limitations, expected API changes, and how you will evaluate it.

5. **Tools and Integrations** - Propose new built-in tools, tool integrations, or improvements to the existing tool-execution mechanism.

- If an implementation is still in progress and early feedback would be useful, consider opening a Draft PR.

## Using AI Coding Tools
AI coding assistants can help contributors understanding the codebase, explore implementation approaches, debug issues, and write tests. We welcome the use of AI tools, but AI should assist your development process, not replace your understanding and engineering judgment.


