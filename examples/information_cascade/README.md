# Information Cascade (Financial Rumor Mill)

## Summary
This model demonstrates the socio-economic phenomenon of an "Information Cascade" or "Echo Chamber" within a highly volatile financial market.

Quantitative Trader agents rely purely on market rumors and their internal reasoning to decide whether to BUY, SELL, or HOLD. By passing alarming messages to one another, agents begin to hallucinate their own generated rumors as external market signals, breaking objective reality and causing a market panic.

**Crucially, this model serves as a stress-test for Mesa-LLM's memory architecture.** By deliberately constraining `short_term_capacity=1`, the model forces constant, rapid consolidation via `STLTMemory`. This exposes the synchronous API blocking bottleneck (Issue #200), providing developers with a measurable benchmark for latency spikes during multi-agent simulations.

## How to Run

1. Ensure you have the `dotenv` package installed: `pip install python-dotenv`.
2. In the root folder of the project, create a file named `.env`.
3. Add your LLM API key: `DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx` (or `OPENAI_API_KEY` etc.)
4. Make sure you are in the `examples/information_cascade` directory.

### To run the Interactive UI:
```bash
solara run app.py
```
*(Note: Because of the synchronous LLM memory consolidation, the UI may experience significant freezing during steps. This is the intended stress-test behavior.)*

### To run the Latency Benchmark (Terminal):
To objectively measure the I/O blocking delay without UI overhead, run the headless benchmark script:
```bash
python run_benchmark.py
```

## Files
* `model.py`: Core model environment setup.
* `agents.py`: Trader agent logic featuring `STLTMemory` and `CoTReasoning`.
* `run_benchmark.py`: A headless script to benchmark the exact execution time of memory consolidation.
* `app.py`: Launches the SolaraViz visualization.