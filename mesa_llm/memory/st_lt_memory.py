from collections import deque
from itertools import islice
from typing import TYPE_CHECKING

from mesa_llm.memory.memory import Memory, MemoryEntry, _format_message_entry

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class STLTMemory(Memory):
    """
    Implements a dual-memory system where recent experiences are stored in short-term memory with limited capacity, and older memories are consolidated into long-term summaries using LLM-based summarization.

    Attributes:
        agent : the agent that the memory belongs to

    Memory is composed of
        - A short term memory who stores the n (int) most recent interactions (observations, planning, discussions)
        - A long term memory that is a summary of the memories that are removed from short term memory (summary
        completed/refactored as it goes)
        - Event types in ``additive_event_types`` accumulate within a step.
          Defaults to ``{"message", "action"}``.

    Logic behind the implementation
        - **Short-term capacity**: Configurable number of recent memory entries (default: short_term_capacity = 5)
        - **Consolidation**: When capacity is exceeded, oldest entries are summarized into long-term memory (number of entries to summarize is configurable, default: consolidation_capacity = 3)
        - **LLM Summarization**: Uses a separate LLM instance to create meaningful summaries of past experiences

    """

    def __init__(
        self,
        agent: "LLMAgent",
        short_term_capacity: int = 5,
        consolidation_capacity: int = 2,
        display: bool = True,
        llm_model: str | None = None,
        fallback_model: str | None = None,
        api_base: str | None = None,
        additive_event_types: list[str] | set[str] | tuple[str, ...] | None = None,
    ):
        """
        Initialize the memory

        Args:
            short_term_capacity : the number of interactions to store in the short term memory
            llm_model : the model to use for the summarization
            api_base : the API base URL to use for the LLM provider
            agent : the agent that the memory belongs to
            additive_event_types : event types that accumulate multiple values
                within a step instead of overwriting. Defaults to
                ``{"message", "action"}``.
        """
        if not llm_model:
            raise ValueError(
                "llm_model must be provided for the usage of st/lt memory. You can use the pre-built 'short-term-only' memory without a model."
            )

        super().__init__(
            agent=agent,
            llm_model=llm_model,
            fallback_model=fallback_model,
            api_base=api_base,
            display=display,
            additive_event_types=additive_event_types,
        )

        self.capacity = short_term_capacity
        self.consolidation_capacity = (
            consolidation_capacity if consolidation_capacity > 0 else None
        )

        self.short_term_memory = deque()
        self.long_term_memory = ""
        self.system_prompt = """
            You are a helpful assistant that summarizes the short term memory into a long term memory.
            The long term memory should be a summary of the short term memory that is concise and informative.
            If the short term memory is empty, return the long term memory unchanged.
            If the long term memory is not empty, update it to include the new information from the short term memory.
            """

        if self.agent.step_prompt:
            self.system_prompt += f" This is the prompt of the problem you will be tackling:{self.agent.step_prompt}, ensure you summarize the short-term memory into long-term a way that is relevant to the problem at hand."

        self.llm.system_prompt = self.system_prompt

    def _build_consolidation_prompt(self, evicted_entries: list[MemoryEntry]) -> str:
        """
        Build a prompt that asks the LLM to integrate *evicted* memories
        into the existing long-term summary.

        Args:
            evicted_entries: the oldest short-term entries selected for
                consolidation. They are still in the deque while the summary
                is generated and are removed only after it has been stored.
        """
        evicted_text = "\n".join(
            f"Step {e.step}: \n{e.content}" for e in evicted_entries
        )
        return (
            "Memories to consolidate (oldest entries being removed "
            "from short-term memory):\n"
            f"{evicted_text}\n\n"
            f"Existing long term memory:\n{self.long_term_memory}\n\n"
            "Please integrate the above memories into a concise, updated "
            "long-term memory summary."
        )

    @staticmethod
    def _extract_summary(response) -> str:
        """
        Return the summary text of an LLM response, or raise ``ValueError``.

        Long-term memory is only overwritten with a valid, non-empty string, so
        a bad response can never replace a good summary.
        """
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("Consolidation response has no message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Consolidation response must be a non-empty string")
        return content

    def _discard_consolidated(self, evicted_entries: list[MemoryEntry]) -> None:
        """
        Remove the entries that were just consolidated from short-term memory.

        Only call this after the summary has been stored in long-term memory.
        Entries are matched by identity, oldest first.
        """
        for entry in evicted_entries:
            if self.short_term_memory and self.short_term_memory[0] is entry:
                self.short_term_memory.popleft()

    def _update_long_term_memory(self, evicted_entries: list[MemoryEntry]):
        """
        Update the long term memory by summarizing the evicted entries.

        Raises if the LLM call fails or returns an invalid response. In both
        cases long-term memory is left unchanged.
        """
        prompt = self._build_consolidation_prompt(evicted_entries)
        response = self.llm.generate(prompt)
        self.long_term_memory = self._extract_summary(response)

    async def _aupdate_long_term_memory(self, evicted_entries: list[MemoryEntry]):
        """
        Async version of _update_long_term_memory
        """
        prompt = self._build_consolidation_prompt(evicted_entries)
        response = await self.llm.agenerate(prompt)
        self.long_term_memory = self._extract_summary(response)

    def _process_step_core(self, pre_step: bool):
        """
        Shared core logic for process_step and aprocess_step.

        Update short-term memory and decide if consolidation is needed.
        When consolidation is needed, the oldest entries are selected and
        returned so the caller can pass them to the LLM for summarization.
        They are *not* removed here: the caller removes them with
        ``_discard_consolidated`` only after the summary has been stored, so a
        failed LLM call cannot lose them.

        Returns:
            ``(new_entry, evicted_entries)`` where *evicted_entries* is a
            (possibly empty) list of MemoryEntry objects, still present at the
            left of short-term memory, that should be consolidated.
        """
        if pre_step:
            new_entry = MemoryEntry(
                agent=self.agent,
                content=self.step_content,
                step=None,
            )
            new_entry._event_order = list(self._step_event_order)
            self.short_term_memory.append(new_entry)
            self.step_content = {}
            self._reset_step_event_order()
            return None, []

        if not self.short_term_memory or self.short_term_memory[-1].step is not None:
            return None, []

        pre_step_entry = self.short_term_memory.pop()
        merged_content = self._merge_step_contents(
            self.step_content, pre_step_entry.content
        )
        new_entry = MemoryEntry(
            agent=self.agent,
            content=merged_content,
            step=self.agent.model.steps,
        )
        staged_event_order = self._normalized_step_event_order(
            pre_step_entry.content,
            getattr(pre_step_entry, "_event_order", ()),
        )
        current_event_order = self._normalized_step_event_order(
            self.step_content,
            self._step_event_order,
        )
        new_entry._event_order = [
            *staged_event_order,
            *current_event_order,
        ]
        self.short_term_memory.append(new_entry)
        self.step_content = {}
        self._reset_step_event_order()

        evicted: list[MemoryEntry] = []

        if (
            len(self.short_term_memory)
            > self.capacity + (self.consolidation_capacity or 0)
            and self.consolidation_capacity
        ):
            # Select (do not remove) the oldest entries for summarization.
            # They are removed after a successful consolidation.
            evicted = list(islice(self.short_term_memory, self.consolidation_capacity))

        elif (
            len(self.short_term_memory) > self.capacity
            and not self.consolidation_capacity
        ):
            # No consolidation configured — just discard the oldest entry
            self.short_term_memory.popleft()

        return new_entry, evicted

    def process_step(self, pre_step: bool = False):
        """
        Synchronous memory step handler
        """
        new_entry, evicted = self._process_step_core(pre_step)

        if evicted:
            # Raises on failure, leaving the entries in short-term memory
            self._update_long_term_memory(evicted)
            self._discard_consolidated(evicted)

        if new_entry and self.display:
            new_entry.display()

    async def aprocess_step(self, pre_step: bool = False):
        """
        Async memory step handler (non-blocking consolidation)
        """
        new_entry, evicted = self._process_step_core(pre_step)

        if evicted:
            # Raises on failure, leaving the entries in short-term memory
            await self._aupdate_long_term_memory(evicted)
            self._discard_consolidated(evicted)

        if new_entry and self.display:
            new_entry.display()

    def format_long_term(self) -> str:
        """
        Get the long term memory
        """
        return str(self.long_term_memory)

    def format_short_term(self) -> str:
        """
        Get the short term memory
        """
        if not self.short_term_memory:
            return "No recent memory."

        else:
            lines = []
            for st_memory_entry in self.short_term_memory:
                lines.append(
                    f"Step {st_memory_entry.step}: \n{st_memory_entry.content}"
                )
            return "\n".join(lines)

    def get_prompt_ready(self) -> str:
        return (
            f"Short term memory:\n {self.format_short_term()}\n\n"
            f"Long term memory: \n{self.format_long_term()}"
        )

    def get_communication_history(self) -> str:
        """
        Get the communication history
        """
        lines = []
        for entry in self.short_term_memory:
            if "message" not in entry.content:
                continue
            msgs = entry.content["message"]
            if isinstance(msgs, list):
                for msg in msgs:
                    lines.append(f"Step {entry.step}: {_format_message_entry(msg)}\n\n")
            else:
                lines.append(f"Step {entry.step}: {_format_message_entry(msgs)}\n\n")
        return "\n".join(lines)
