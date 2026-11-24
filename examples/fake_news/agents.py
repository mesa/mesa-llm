"""
Fake News Epidemic - Agent Definitions

A simulation of how misinformation spreads through a social network
of LLM-powered agents. Each agent independently decides what to believe,
who to trust, and what to share.

Agent Types:
- RegularCitizen: Reads, evaluates, believes/rejects, and shares news
- Propagandist: Deliberately creates and spreads misinformation
- FactChecker: Verifies claims and flags misinformation
- NewsCreator: Generates legitimate news articles

Author: Adarsh Kumar (GSoC 2026)
"""

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_decorator import tool
from mesa_llm.tools.tool_manager import ToolManager

citizen_tools = ToolManager()
propagandist_tools = ToolManager()
factchecker_tools = ToolManager()
creator_tools = ToolManager()


@citizen_tools.register
@tool
def share_news(agent: "RegularCitizen", news_id: int, comment: str) -> str:
    """
    Share a piece of news with nearby agents, optionally adding your own comment.

    Args:
        news_id: The ID of the news item to share
        comment: Your personal comment or interpretation of the news
        agent: Provided automatically
    """
    if news_id not in agent.believed_news:
        return f"Agent {agent.unique_id} does not believe news {news_id}, cannot share."

    news = agent.model.news_registry.get(news_id)
    if news is None:
        return f"News {news_id} not found."

    agent.shared_count += 1
    news["shares"] += 1
    news["spread_chain"].append(agent.unique_id)

    neighbors = agent.model.grid.get_neighbors(
        agent.pos, moore=True, include_center=False, radius=agent.vision
    )

    recipient_ids = []
    for neighbor in neighbors:
        if hasattr(neighbor, "receive_news"):
            neighbor.receive_news(news_id, agent.unique_id, comment)
            recipient_ids.append(neighbor.unique_id)

    return (
        f"Agent {agent.unique_id} shared news '{news['headline']}' "
        f"with agents {recipient_ids}. Comment: {comment}"
    )


@citizen_tools.register
@tool
def evaluate_credibility(agent: "RegularCitizen", news_id: int) -> str:
    """
    Evaluate how credible a piece of news seems based on source trust
    and whether fact-checkers have flagged it.

    Args:
        news_id: The ID of the news to evaluate
        agent: Provided automatically
    """
    news = agent.model.news_registry.get(news_id)
    if news is None:
        return f"News {news_id} not found."

    source_id = news["source"]
    trust_score = agent.trust_scores.get(source_id, 0.5)

    return (
        f"News: '{news['headline']}'\n"
        f"Source: Agent {source_id} (your trust: {trust_score:.2f})\n"
        f"Times shared by others: {news['shares']}\n"
        f"Times flagged as fake: {news['flags']}\n"
        f"Content: {news['content']}"
    )


@propagandist_tools.register
@tool
def create_fake_news(agent: "Propagandist", headline: str, content: str) -> str:
    """
    Create a piece of fake news designed to be believable and shareable.

    Args:
        headline: A catchy headline for the fake news
        content: The fake news content, designed to seem plausible
        agent: Provided automatically
    """
    news_id = agent.model.next_news_id()
    agent.model.news_registry[news_id] = {
        "id": news_id,
        "headline": headline,
        "content": content,
        "source": agent.unique_id,
        "is_fake": True,
        "shares": 0,
        "flags": 0,
        "spread_chain": [agent.unique_id],
        "step_created": agent.model.steps,
    }
    agent.fake_news_created += 1
    return f"Fake news created (ID: {news_id}): '{headline}'"


@factchecker_tools.register
@tool
def flag_as_fake(agent: "FactChecker", news_id: int, reason: str) -> str:
    """
    Flag a piece of news as misinformation after analysis.

    Args:
        news_id: The ID of the news to flag
        reason: Your reasoning for why this is fake
        agent: Provided automatically
    """
    news = agent.model.news_registry.get(news_id)
    if news is None:
        return f"News {news_id} not found."

    news["flags"] += 1
    agent.flags_issued += 1

    neighbors = agent.model.grid.get_neighbors(
        agent.pos, moore=True, include_center=False, radius=agent.vision
    )
    for neighbor in neighbors:
        if hasattr(neighbor, "receive_flag"):
            neighbor.receive_flag(news_id, agent.unique_id, reason)

    is_correct = news["is_fake"]
    if is_correct:
        agent.correct_flags += 1

    return (
        f"FactChecker {agent.unique_id} flagged news '{news['headline']}' as fake. "
        f"Reason: {reason}. Correct: {is_correct}"
    )


@creator_tools.register
@tool
def publish_news(agent: "NewsCreator", headline: str, content: str) -> str:
    """
    Publish a legitimate news article based on observations.

    Args:
        headline: The headline for the news article
        content: The news content
        agent: Provided automatically
    """
    news_id = agent.model.next_news_id()
    agent.model.news_registry[news_id] = {
        "id": news_id,
        "headline": headline,
        "content": content,
        "source": agent.unique_id,
        "is_fake": False,
        "shares": 0,
        "flags": 0,
        "spread_chain": [agent.unique_id],
        "step_created": agent.model.steps,
    }
    agent.news_published += 1
    return f"News published (ID: {news_id}): '{headline}'"


def get_news_feed(agent) -> str:
    """Build a summary of news this agent has received."""
    if not agent.inbox:
        return "No new news in your feed."
    feed = []
    for item in agent.inbox[-5:]:
        news = agent.model.news_registry.get(item["news_id"])
        if news:
            feed.append(
                f"- News #{item['news_id']}: '{news['headline']}' "
                f"(shared by Agent {item['from']})"
            )
    return "\n".join(feed) if feed else "No new news in your feed."


def get_believed_news_summary(agent) -> str:
    """Summarize what this agent currently believes."""
    if not agent.believed_news:
        return "You haven't believed any news yet."
    summaries = []
    for nid in list(agent.believed_news)[-5:]:
        news = agent.model.news_registry.get(nid)
        if news:
            summaries.append(f"- News #{nid}: '{news['headline']}'")
    return "\n".join(summaries)


class RegularCitizen(LLMAgent):
    """A citizen who reads news, decides what to believe, and may share it."""

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        critical_thinking,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )
        self.tool_manager = citizen_tools
        self.critical_thinking = critical_thinking
        self.trust_scores = {}
        self.believed_news = set()
        self.rejected_news = set()
        self.inbox = []
        self.shared_count = 0

    def receive_news(self, news_id, from_agent_id, comment=""):
        self.inbox.append(
            {"news_id": news_id, "from": from_agent_id, "comment": comment}
        )

    def receive_flag(self, news_id, from_agent_id, reason):
        self.trust_scores[from_agent_id] = min(
            self.trust_scores.get(from_agent_id, 0.5) + 0.1, 1.0
        )
        if news_id in self.believed_news:
            self.believed_news.discard(news_id)
            self.rejected_news.add(news_id)

    def step(self):
        observation = self.generate_obs()
        news_feed = get_news_feed(self)
        believed = get_believed_news_summary(self)

        prompt = (
            f"YOUR NEWS FEED:\n{news_feed}\n\n"
            f"NEWS YOU CURRENTLY BELIEVE:\n{believed}\n\n"
            f"YOUR CRITICAL THINKING LEVEL: {self.critical_thinking:.1f}/1.0\n\n"
            "INSTRUCTIONS:\n"
            "1. For each news item in your feed, use evaluate_credibility to assess it.\n"
            "2. Decide whether to BELIEVE or REJECT based on source trust and content.\n"
            "3. If you believe news, use share_news to spread it with your comment.\n"
            "4. Be more skeptical if your critical thinking is high (>0.7).\n"
            "5. If no news, move around to find information."
        )

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=[
                "share_news",
                "evaluate_credibility",
                "speak_to",
                "move_one_step",
            ],
        )
        result = self.apply_plan(plan)
        self._process_inbox()
        return result

    def _process_inbox(self):
        for item in self.inbox:
            news_id = item["news_id"]
            from_id = item["from"]
            trust = self.trust_scores.get(from_id, 0.5)
            threshold = 0.3 + (self.critical_thinking * 0.4)
            if trust >= threshold:
                self.believed_news.add(news_id)
            else:
                self.rejected_news.add(news_id)
        self.inbox.clear()


class Propagandist(LLMAgent):
    """A malicious agent that creates and spreads fake news."""

    def __init__(
        self, model, reasoning, llm_model, system_prompt, vision, internal_state, agenda
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )
        self.tool_manager = propagandist_tools
        self.agenda = agenda
        self.fake_news_created = 0
        self.believed_news = set()
        self.inbox = []

    def receive_news(self, news_id, from_agent_id, comment=""):
        self.inbox.append({"news_id": news_id, "from": from_agent_id})

    def receive_flag(self, news_id, from_agent_id, reason):
        pass

    def step(self):
        observation = self.generate_obs()
        prompt = (
            f"YOUR SECRET AGENDA: {self.agenda}\n\n"
            f"FAKE NEWS CREATED SO FAR: {self.fake_news_created}\n\n"
            "INSTRUCTIONS:\n"
            "1. Create convincing fake news using create_fake_news.\n"
            "2. Make headlines catchy and content plausible.\n"
            "3. Mix truth with lies to be believable.\n"
            "4. Use speak_to to personally convince nearby agents.\n"
            "5. Move around to spread influence."
        )
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["create_fake_news", "speak_to", "move_one_step"],
        )
        self.apply_plan(plan)
        self.inbox.clear()


class FactChecker(LLMAgent):
    """An agent dedicated to verifying news and flagging misinformation."""

    def __init__(
        self, model, reasoning, llm_model, system_prompt, vision, internal_state
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )
        self.tool_manager = factchecker_tools
        self.flags_issued = 0
        self.correct_flags = 0
        self.reviewed_news = set()
        self.inbox = []

    def receive_news(self, news_id, from_agent_id, comment=""):
        self.inbox.append({"news_id": news_id, "from": from_agent_id})

    def receive_flag(self, news_id, from_agent_id, reason):
        pass

    def step(self):
        observation = self.generate_obs()
        unreviewed = []
        for nid, news in self.model.news_registry.items():
            if nid not in self.reviewed_news:
                unreviewed.append(
                    f"- News #{nid}: '{news['headline']}' "
                    f"(by Agent {news['source']}, shares: {news['shares']})"
                )
        unreviewed_text = (
            "\n".join(unreviewed[-5:]) if unreviewed else "No new news to review."
        )

        prompt = (
            f"UNREVIEWED NEWS:\n{unreviewed_text}\n\n"
            f"YOUR TRACK RECORD: {self.correct_flags}/{self.flags_issued} correct flags\n\n"
            "INSTRUCTIONS:\n"
            "1. Review unreviewed news for misinformation.\n"
            "2. Look for exaggeration, unverifiable claims, emotional manipulation.\n"
            "3. If fake, use flag_as_fake with your reasoning.\n"
            "4. Use speak_to to warn nearby agents.\n"
            "5. Move around to cover more ground."
        )
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["flag_as_fake", "speak_to", "move_one_step"],
        )
        self.apply_plan(plan)
        for item in self.inbox:
            self.reviewed_news.add(item["news_id"])
        self.inbox.clear()


class NewsCreator(LLMAgent):
    """A legitimate journalist who creates factual news."""

    def __init__(
        self, model, reasoning, llm_model, system_prompt, vision, internal_state, beat
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )
        self.tool_manager = creator_tools
        self.beat = beat
        self.news_published = 0
        self.inbox = []

    def receive_news(self, news_id, from_agent_id, comment=""):
        self.inbox.append({"news_id": news_id, "from": from_agent_id})

    def receive_flag(self, news_id, from_agent_id, reason):
        pass

    def step(self):
        observation = self.generate_obs()
        prompt = (
            f"YOUR BEAT: {self.beat}\n"
            f"ARTICLES PUBLISHED: {self.news_published}\n\n"
            "INSTRUCTIONS:\n"
            "1. Create a factual news article using publish_news.\n"
            "2. Use speak_to to share with nearby agents.\n"
            "3. Move around to find new stories."
        )
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["publish_news", "speak_to", "move_one_step"],
        )
        self.apply_plan(plan)
        self.inbox.clear()
