# Fake News Epidemic Simulation

A mesa-llm example that simulates how misinformation spreads through a social network of LLM-powered agents.

## Overview

This model explores one of the most pressing challenges of the information age: **how does fake news spread, and what can stop it?**

Unlike traditional agent-based models that use hard-coded rules for information sharing, this simulation uses LLM-powered agents that **independently reason** about what news to believe, share, create, and flag.

## Agent Types

| Agent | Role |
|-------|------|
| **RegularCitizen** | Reads, evaluates, believes/rejects, and shares news. Each has a unique critical thinking level and evolving trust scores. |
| **Propagandist** | Deliberately creates convincing fake news to advance a hidden agenda. Mixes truth with lies. |
| **FactChecker** | Analyzes news for misinformation signs. Flags fake news and warns others. |
| **NewsCreator** | Legitimate journalist who creates factual news covering a specific beat. |

## Key Mechanics

### Trust Network
- Every citizen maintains trust scores for other agents (0.0 to 1.0)
- Trust evolves based on whether information from a source turns out to be true or flagged
- Propagandists start with neutral trust and must earn credibility

### Belief System
- Citizens decide what to believe based on source trust, content analysis, and fact-checker flags
- Critical thinking level (varies per agent) determines the threshold for believing news
- Once a fact-checker flags news, believers may reconsider

### Information Spread
- News spreads through the spatial grid — agents share with neighbors
- Each share is tracked, building a spread chain
- Fake news and real news compete for attention

## Metrics Tracked

- **Fake vs Real News Count** — Articles of each type
- **Share Rates** — How often fake vs real news gets shared
- **Belief Rate** — Percentage of citizens believing fake news
- **Trust in Propagandists** — Average trust in malicious agents
- **Fact-Checker Accuracy** — Correct flag percentage

## Running
```bash
python -m examples.fake_news.model
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_citizens | 10 | Regular citizens |
| n_propagandists | 2 | Malicious agents |
| n_factcheckers | 2 | Fact-checker agents |
| n_creators | 2 | Legitimate journalists |
| width/height | 10 | Grid dimensions |
| vision | 2 | Agent interaction radius |
| avg_critical_thinking | 0.5 | Average critical thinking (0-1) |
| llm_model | gpt-4o-mini | LLM model to use |

## Author

**Adarsh Kumar** — GSoC 2026 Contributor Candidate
- GitHub: [@adarshkumar23](https://github.com/adarshkumar23)
- IIT Patna | AI/ML Engineer
