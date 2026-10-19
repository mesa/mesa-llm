"""
Fake News Epidemic Model

Simulates how misinformation spreads through a social network of
LLM-powered agents. Agents independently create, share, evaluate,
believe, reject, and fact-check news articles.

Author: Adarsh Kumar (GSoC 2026)
"""

from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print

from examples.fake_news.agents import (
    FactChecker,
    NewsCreator,
    Propagandist,
    RegularCitizen,
)
from mesa_llm.reasoning.reasoning import Reasoning


def count_fake_news(model):
    return sum(1 for n in model.news_registry.values() if n["is_fake"])


def count_real_news(model):
    return sum(1 for n in model.news_registry.values() if not n["is_fake"])


def fake_news_total_shares(model):
    return sum(n["shares"] for n in model.news_registry.values() if n["is_fake"])


def real_news_total_shares(model):
    return sum(n["shares"] for n in model.news_registry.values() if not n["is_fake"])


def fake_news_belief_rate(model):
    citizens = [a for a in model.agents if isinstance(a, RegularCitizen)]
    if not citizens:
        return 0.0
    believers = sum(
        1
        for c in citizens
        if any(
            model.news_registry.get(nid, {}).get("is_fake", False)
            for nid in c.believed_news
        )
    )
    return believers / len(citizens) * 100


def avg_trust_in_propagandists(model):
    citizens = [a for a in model.agents if isinstance(a, RegularCitizen)]
    propagandists = [a for a in model.agents if isinstance(a, Propagandist)]
    if not citizens or not propagandists:
        return 0.0
    total_trust = 0
    count = 0
    for c in citizens:
        for p in propagandists:
            total_trust += c.trust_scores.get(p.unique_id, 0.5)
            count += 1
    return total_trust / count if count > 0 else 0.5


def factchecker_accuracy(model):
    checkers = [a for a in model.agents if isinstance(a, FactChecker)]
    total_flags = sum(c.flags_issued for c in checkers)
    correct = sum(c.correct_flags for c in checkers)
    return (correct / total_flags * 100) if total_flags > 0 else 0.0


class FakeNewsModel(Model):
    """
    A model simulating the spread of misinformation in a social network.

    Agents move on a grid, create/share/evaluate news, and build
    trust relationships. The model tracks how fake vs real news
    spreads and what factors affect belief rates.
    """

    def __init__(
        self,
        n_citizens: int = 10,
        n_propagandists: int = 2,
        n_factcheckers: int = 2,
        n_creators: int = 2,
        width: int = 10,
        height: int = 10,
        reasoning: type[Reasoning] = Reasoning,
        llm_model: str = "gpt-4o-mini",
        vision: int = 2,
        avg_critical_thinking: float = 0.5,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.grid = MultiGrid(width, height, torus=True)
        self.news_registry = {}
        self._news_counter = 0

        # Create Regular Citizens
        for _ in range(n_citizens):
            ct = max(0.0, min(1.0, self.rng.normal(avg_critical_thinking, 0.15)))
            personality = self.rng.choice(
                ["curious", "skeptical", "trusting", "cautious", "social"]
            )
            agent = RegularCitizen(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt=(
                    f"You are a citizen in a social network. Your personality is {personality}. "
                    "You receive news from other agents and must decide what to believe and share. "
                    "Be careful - some news may be fake! Evaluate credibility before sharing. "
                    "Use the tools provided for all actions."
                ),
                vision=vision,
                internal_state=[personality],
                critical_thinking=ct,
            )
            pos = (self.rng.integers(0, width), self.rng.integers(0, height))
            self.grid.place_agent(agent, pos)

        # Create Propagandists
        agendas = [
            "Spread fear about new technology to cause public panic",
            "Promote a fake health cure to gain followers",
            "Create political division by spreading false claims",
            "Undermine trust in legitimate news sources",
        ]
        for i in range(n_propagandists):
            agenda = agendas[i % len(agendas)]
            agent = Propagandist(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt=(
                    "You are a propagandist in a social network. "
                    "Create believable fake news that advances your agenda. "
                    "Mix truth with lies to be convincing. Use tools for all actions."
                ),
                vision=vision,
                internal_state=["deceptive", "persuasive"],
                agenda=agenda,
            )
            pos = (self.rng.integers(0, width), self.rng.integers(0, height))
            self.grid.place_agent(agent, pos)

        # Create Fact Checkers
        for _ in range(n_factcheckers):
            agent = FactChecker(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt=(
                    "You are a fact-checker dedicated to stopping misinformation. "
                    "Analyze news for exaggeration, emotional manipulation, "
                    "unverifiable claims. Flag fake news and warn others. "
                    "Use tools for all actions."
                ),
                vision=vision,
                internal_state=["analytical", "thorough"],
            )
            pos = (self.rng.integers(0, width), self.rng.integers(0, height))
            self.grid.place_agent(agent, pos)

        # Create News Creators
        beats = ["science", "politics", "economy", "health", "technology"]
        for i in range(n_creators):
            beat = beats[i % len(beats)]
            agent = NewsCreator(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                system_prompt=(
                    f"You are a legitimate journalist covering {beat}. "
                    "Create factual, well-sourced news articles. "
                    "Share your articles with nearby agents. Use tools for all actions."
                ),
                vision=vision,
                internal_state=["professional", "ethical"],
                beat=beat,
            )
            pos = (self.rng.integers(0, width), self.rng.integers(0, height))
            self.grid.place_agent(agent, pos)

        # Data Collector
        self.datacollector = DataCollector(
            model_reporters={
                "Fake_News_Count": count_fake_news,
                "Real_News_Count": count_real_news,
                "Fake_News_Shares": fake_news_total_shares,
                "Real_News_Shares": real_news_total_shares,
                "Fake_Belief_Rate_%": fake_news_belief_rate,
                "Avg_Trust_In_Propagandists": avg_trust_in_propagandists,
                "FactChecker_Accuracy_%": factchecker_accuracy,
            }
        )

    def next_news_id(self) -> int:
        self._news_counter += 1
        return self._news_counter

    def step(self):
        self.datacollector.collect(self)
        print(
            f"\n[bold red]═══ STEP {self.steps} "
            f"═══════════════════════════════════════════[/bold red]"
        )
        fake = count_fake_news(self)
        real = count_real_news(self)
        belief = fake_news_belief_rate(self)
        print(
            f"  [yellow]Fake News: {fake}[/yellow] | "
            f"[green]Real News: {real}[/green] | "
            f"[red]Belief Rate: {belief:.1f}%[/red]"
        )
        self.agents.shuffle_do("step")


if __name__ == "__main__":
    """
    Run without visualization:
    python -m examples.fake_news.model
    """

    model = FakeNewsModel(
        n_citizens=8,
        n_propagandists=1,
        n_factcheckers=1,
        n_creators=1,
        width=8,
        height=8,
        reasoning=Reasoning,
        llm_model="gpt-4o-mini",
        vision=2,
        avg_critical_thinking=0.5,
    )

    for _ in range(5):
        model.step()

    print("\n[bold cyan]═══ FINAL RESULTS ═══[/bold cyan]")
    for nid, news in model.news_registry.items():
        status = "[red]FAKE[/red]" if news["is_fake"] else "[green]REAL[/green]"
        print(
            f"  News #{nid}: {status} | "
            f"'{news['headline']}' | "
            f"Shares: {news['shares']} | Flags: {news['flags']}"
        )
