import os
import time

from dotenv import load_dotenv
from model import MarketPanicModel

load_dotenv()

if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set your API KEY in your .env file")
        exit(1)

    print("Starting Information Cascade Benchmark...")
    print(
        "Testing STLTMemory consolidation latency bottleneck (short_term_capacity=1)."
    )

    # Initialize the model
    model = MarketPanicModel(num_agents=4)

    for step in range(1, 4):
        start_time = time.perf_counter()
        print(f"\n--- Simulating Step {step} ---")

        model.step()

        step_duration = time.perf_counter() - start_time
        print(f"⚠️ Step {step} completed in: {step_duration:.2f} seconds")

        if step_duration > 5:
            print(
                "🚨 CRITICAL BOTTLENECK DETECTED: Latency spiked massively "
                "due to LLM memory summarization blocking the main thread!"
            )
