# Stock Market with Rumors Simulation

A mesa-llm example simulating how information asymmetry, insider trading, and rumors create market bubbles and crashes.

## Overview

This model demonstrates how **LLM-powered agents** interact in a simulated stock market where information is power. Insider traders manipulate prices through planted rumors, analysts publish research, and retail traders try to navigate the chaos.

## Agent Types

| Agent | Role |
|-------|------|
| **RetailTrader** | Regular investor trading based on public info, rumors, and reports. Each has different risk tolerance. |
| **InsiderTrader** | Buys stocks cheaply, spreads positive rumors to pump prices, then sells at the top. |
| **Analyst** | Publishes research reports with honest ratings that influence trader decisions. |
| **MarketMaker** | Monitors volatility and provides stabilizing liquidity during extreme moves. |

## Market Mechanics

### Price Engine
- Prices move based on supply/demand imbalance from agent trading
- Random walk component simulates market noise
- Minimum price floor of $1 prevents negative prices

### Information Flow
- Insiders spread rumors to nearby traders on the grid
- Analysts publish reports visible to neighbors
- Traders share tips through speak_to communication
- Information advantage decays with distance

### Stocks
- **TECH** ($100) — High growth, moderate volatility
- **HEALTH** ($75) — Stable, low volatility
- **ENERGY** ($50) — Cyclical, high volatility

## Metrics Tracked

- **Average Stock Price** — Market health indicator
- **Market Volume** — Trading activity level
- **Total/Manipulation Rumors** — Information quality
- **Volatility %** — Price stability
- **Insider Wealth** — Whether manipulation is profitable
- **Bubble Detection** — Flags 30%+ price increases over 5 steps

## Running
```bash
python -m examples.stock_market.model
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_retail | 8 | Retail traders |
| n_insiders | 1 | Insider/manipulator agents |
| n_analysts | 2 | Research analysts |
| n_makers | 1 | Market makers |
| width/height | 8 | Grid dimensions |
| vision | 2 | Agent interaction radius |
| llm_model | gpt-4o-mini | LLM model to use |

## Author

**Adarsh Kumar** — GSoC 2026 Contributor Candidate
- GitHub: [@adarshkumar23](https://github.com/adarshkumar23)
- IIT Patna | AI/ML Engineer
