# Examples

The examples folder contains LLM agent-based models implemented using Mesa-LLM. They serve as  demonstrations of Mesa-LLM's capabilities and starting points for your own models.

This folder contains both basic and advanced examples that demonstrate different aspects of Mesa-LLM.

## Epstein Civil Violence Model

Joshua Epstein's [model](https://www.pnas.org/doi/10.1073/pnas.092080199) of how a decentralized uprising can be suppressed or reach a critical mass of support.

**Location:** [examples/epstein_civil_violence](https://github.com/projectmesa/mesa-llm/tree/main/examples/epstein_civil_violence)

This example demonstrates:
- Agent decision-making based on social dynamics
- LLM-powered agent reasoning in conflict scenarios
- Integration of multiple agent types with different behaviors

## Negotiation Model

Implementation of a negotiation model where two types of agents (buyer and seller) negotiate over a product.

**Location:** [examples/negotiation](https://github.com/projectmesa/mesa-llm/tree/main/examples/negotiation)

This example demonstrates:
- Agent-to-agent communication
- LLM-based negotiation strategies
- Dynamic pricing and offer-counteroffer mechanisms

## Sugarscrap-g1mt Model

Implementation of Epstein & Axtell’s classic Sugarscape (G1MT) model, where Trader agents harvest resources (Sugar & Spice), consume them according to metabolic needs, and trade with neighbors based on economic rationality (MRS).

**Location:** [examples/sugarscrap_g1mt](https://github.com/mesa/mesa-llm/tree/main/examples/sugarscrap_g1mt)

 This example demonstrates:
- LLM-powered survival decision-making
- Resource regeneration and environmental dynamics
- Marginal Rate of Substitution (MRS)-driven economic behavior
- Emergent price equilibrium and wealth distribution

## Information Cascade Model

Implementation of a financial market simulation where LLM-powered quantitative traders make buy/sell/hold decisions based on market rumors and memory consolidation.

**Location:** [examples/information_cascade](https://github.com/projectmesa/mesa-llm/tree/main/examples/information_cascade)

This example demonstrates:
- Emergent herd behavior and "Information Cascades" in financial markets
- High-frequency `STLTMemory` consolidation stress testing
- Latency benchmarking for LLM-powered memory modules
- Custom robust Solara-based UI for asynchronous agent interactions