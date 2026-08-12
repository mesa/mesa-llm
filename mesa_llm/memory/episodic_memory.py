import json
from collections import deque
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from mesa_llm.memory.memory import Memory, MemoryEntry, _format_message_entry

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class EventGrade(BaseModel):
    grade: int = Field(
        description="Integer score representing the importance of the event"
    )


def normalize_dict_values(scores: dict, min_target: float, max_target: float) -> dict:
    """
    Normalize dictionary values to a target range with min-max scaling.

    This mirrors the min-max helper used in the Generative Agents reference
    retrieval implementation:
    https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/retrieve.py
    """
    if not scores:
        return {}

    vals = list(scores.values())
    min_val = min(vals)
    max_val = max(vals)

    range_val = max_val - min_val

    if range_val == 0:
        midpoint = (max_target - min_target) / 2 + min_target
        for key in scores:
            scores[key] = midpoint
    else:
        for key, val in scores.items():
            scores[key] = (val - min_val) * (
                max_target - min_target
            ) / range_val + min_target

    return scores


class EpisodicMemory(Memory):
    """
    Event-level memory with LLM-based importance scoring and recency-aware retrieval.

    Credit / references:
    - Paper: Generative Agents: Interactive Simulacra of Human Behavior
      https://arxiv.org/abs/2304.03442
    - Reference retrieval code:
      https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/retrieve.py

    This implementation is inspired by the paper's retrieval scoring design
    (component-wise min-max normalization, then weighted combination). It is
    not a strict copy of the original code: relevance scoring via embeddings is
    not implemented yet, and recency is computed from step age.
    """

    def __init__(
        self,
        agent: "LLMAgent",
        llm_model: str | None = None,
        display: bool = True,
        max_capacity: int = 200,
        considered_entries: int = 30,
        recency_decay: float = 0.995,
        api_base: str | None = None,
    ):
        """
        Initialize the EpisodicMemory.

        Args:
            agent : the agent that owns this memory
            llm_model : the model used to grade event importance
            display : whether to display memory entries in the console
            max_capacity : maximum number of finalized episodic entries to keep
            considered_entries : number of entries to consider during retrieval
            recency_decay : exponential decay factor for recency scoring
            api_base : the API base URL to use for the LLM provider
        """
        if not llm_model:
            raise ValueError(
                "llm_model must be provided for the usage of episodic memory"
            )

        super().__init__(
            agent,
            llm_model=llm_model,
            api_base=api_base,
            display=display,
        )

        self.max_capacity = max_capacity
        self.memory_entries = deque(maxlen=self.max_capacity)
        self.considered_entries = considered_entries
        self.recency_decay = recency_decay

        self.system_prompt = """
            You are an assistant that evaluates memory entries on a scale from 1 to 5, based on their importance to a specific problem or task. Your goal is to assign a score that reflects how much each entry contributes to understanding, solving, or advancing the task. Use the following grading scale:

            5 - Critical: Introduces essential, novel information that significantly impacts problem-solving or decision-making.

            4 - High: Provides important context or clarification that meaningfully improves understanding or direction.

            3 - Moderate: Adds somewhat useful information that may assist but is not essential.

            2 - Low: Offers minimal relevance or slight redundancy; impact is marginal.

            1 - Irrelevant: Contains no useful or applicable information for the current problem.

            Only assess based on the entry's content and its value to the task at hand. Ignore style, grammar, or tone.
            """

    def _extract_importance(self, entry) -> int:
        """
        Safely extracts importance score regardless of data structure.
        Handles:
        - Nested: {"msg": {"importance": 5}}
        - Flat:   {"importance": 5}
        """
        if "importance" in entry.content:
            val = entry.content["importance"]
            return val if isinstance(val, (int, float)) else 1

        for value in entry.content.values():
            if isinstance(value, dict) and "importance" in value:
                val = value["importance"]
                return val if isinstance(val, (int, float)) else 1

        return 1

    def _build_grade_prompt(self, type: str, content: dict) -> str:
        """
        This helper assembles a prompt that includes the event type, event content,
        and up to the five most recent memory entries for contextual grounding.
        It is shared by both synchronous and asynchronous grading methods to
        avoid duplicated prompt-construction logic.
        """
        if len(self.memory_entries) > 0:
            entries = list(self.memory_entries)[-5:]
            previous_entries = "previous memory entries:\n\n".join(
                [str(entry) for entry in entries]
            )
        else:
            previous_entries = "No previous memory entries"

        return f"""
            grade the importance of the following event on a scale from 1 to 5:
            {type}: {content}
            ------------------------------
            {previous_entries}
            """

    def grade_event_importance(self, type: str, content: dict) -> float:
        """
        Grade this event based on the content respect to the previous memory entries
        """
        prompt = self._build_grade_prompt(type, content)
        self.llm.system_prompt = self.system_prompt

        rsp = self.llm.generate(
            prompt=prompt,
            response_format=EventGrade,
        )

        formatted_response = json.loads(rsp.choices[0].message.content)
        return formatted_response["grade"]

    async def agrade_event_importance(self, type: str, content: dict) -> float:
        """
        Asynchronous version of grade_event_importance
        """
        prompt = self._build_grade_prompt(type, content)
        self.llm.system_prompt = self.system_prompt

        rsp = await self.llm.agenerate(
            prompt=prompt,
            response_format=EventGrade,
        )

        formatted_response = json.loads(rsp.choices[0].message.content)
        return formatted_response["grade"]

    def retrieve_top_k_entries(self, k: int) -> list[MemoryEntry]:
        """
        Retrieve the top-k entries using normalized importance and recency.

        Notes:
        - Inspired by Generative Agents retrieval scoring:
          recency/importance/relevance are normalized separately and combined.
        - This implementation currently combines importance + recency only.
          Relevance (embedding cosine similarity with a focal query) is pending.
        """
        if not self.memory_entries:
            return []

        importance_dict = {}
        recency_dict = {}

        entries = list(self.memory_entries)
        current_step = self.agent.model.steps

        for i, entry in enumerate(entries):
            importance_dict[i] = self._extract_importance(entry)

            age = current_step - entry.step
            recency_dict[i] = self.recency_decay**age

        importance_scaled = normalize_dict_values(importance_dict, 0, 1)
        recency_scaled = normalize_dict_values(recency_dict, 0, 1)

        final_scores = []
        for i in range(len(entries)):
            total_score = importance_scaled[i] + recency_scaled[i]
            final_scores.append((total_score, entries[i]))

        final_scores.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in final_scores[:k]]

    def _finalize_entry(self, type: str, graded_content: dict):
        """Create and persist a finalized episodic entry."""
        new_entry = MemoryEntry(
            agent=self.agent,
            content={type: graded_content},
            step=self.agent.model.steps,
        )
        new_entry._event_order = [type]
        self.memory_entries.append(new_entry)

    def add_to_memory(self, type: str, content: dict):
        """
        grading logic + adding to memory function call
        """
        graded_content = {
            **content,
            "importance": self.grade_event_importance(type, content),
        }
        self._finalize_entry(type, graded_content)

    async def aadd_to_memory(self, type: str, content: dict):
        """
        Async version of add_to_memory + grading logic
        """
        graded_content = {
            **content,
            "importance": await self.agrade_event_importance(type, content),
        }
        self._finalize_entry(type, graded_content)

    def get_prompt_ready(self) -> str:
        return f"Top {self.considered_entries} memory entries:\n\n" + "\n".join(
            [
                str(entry)
                for entry in self.retrieve_top_k_entries(self.considered_entries)
            ]
        )

    def get_communication_history(self) -> str:
        """
        Get the communication history
        """
        lines = []
        for entry in self.memory_entries:
            if "message" not in entry.content:
                continue
            msgs = entry.content["message"]
            if isinstance(msgs, list):
                for msg in msgs:
                    lines.append(f"Step {entry.step}: {_format_message_entry(msg)}\n\n")
            else:
                lines.append(f"Step {entry.step}: {_format_message_entry(msgs)}\n\n")
        return "\n".join(lines)

    async def aprocess_step(self, pre_step: bool = False):
        """
        Asynchronous version of process_step.

        EpisodicMemory persists entries at add-time and does not use two-phase
        pre/post-step buffering.
        """
        return

    def process_step(self, pre_step: bool = False):
        """
        Process step hook (no-op for episodic memory).

        EpisodicMemory persists entries at add-time and does not use two-phase
        pre/post-step buffering.
        """
        return
