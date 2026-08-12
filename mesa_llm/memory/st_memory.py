from collections import deque
from typing import TYPE_CHECKING

from mesa_llm.memory.memory import Memory, MemoryEntry, _format_message_entry

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class ShortTermMemory(Memory):
    """
    Simple short-term memory implementation without consolidation (stores recent entries up to capacity limit). Same functionality as `STLTMemory` but without the long-term memory and consolidation mechanism.

    Attributes:
        agent : the agent that the memory belongs to
        n : positive number of short-term memories to remember
        display : whether to display the memory
        llm_model : the model to use for the summarization
        additive_event_types : event types accumulated as lists within a step.
            Defaults to ``{"message", "action"}``.
    """

    def __init__(
        self,
        agent: "LLMAgent",
        n: int = 5,
        display: bool = True,
        additive_event_types: list[str] | set[str] | tuple[str, ...] | None = None,
    ):
        """
        Initialize short-term memory.

        Args:
            agent : the agent that owns this memory
            n : maximum number of finalized short-term entries to keep
            display : whether memory entries should be displayed
            additive_event_types : event types that accumulate multiple values
                within a step instead of overwriting. Defaults to
                ``{"message", "action"}``.
        """
        if n < 1:
            raise ValueError("n must be >= 1 for ShortTermMemory")

        super().__init__(
            agent=agent,
            display=display,
            additive_event_types=additive_event_types,
        )
        self.n = n
        self.short_term_memory = deque(maxlen=self.n)
        self._current_step_entry: MemoryEntry | None = None

    async def aprocess_step(self, pre_step: bool = False):
        """
        Asynchronous version of process_step
        """
        return self.process_step(pre_step=pre_step)

    def process_step(self, pre_step: bool = False):
        """
        Process the step of the agent :
        - Capture pre-step content into the current in-progress step entry
        - Merge current and post-step content into one finalized entry
        - Display the new entry
        """

        # Save a temporary pre-step snapshot. This entry is not persisted in deque.
        if pre_step:
            self._current_step_entry = MemoryEntry(
                agent=self.agent,
                content=self.step_content,
                step=None,
            )
            self._current_step_entry._event_order = list(self._step_event_order)
            self.step_content = {}
            self._step_event_order = []
            return

        new_entry = None
        if self._current_step_entry is not None:
            merged_content = self._merge_step_contents(
                self.step_content, self._current_step_entry.content
            )
            new_entry = MemoryEntry(
                agent=self.agent,
                content=merged_content,
                step=self.agent.model.steps,
            )
            staged_event_order = getattr(self._current_step_entry, "_event_order", ())
            if not isinstance(staged_event_order, list | tuple):
                staged_event_order = ()
            new_entry._event_order = [
                *staged_event_order,
                *self._step_event_order,
            ]
            self.short_term_memory.append(new_entry)
            self._current_step_entry = None
            self.step_content = {}
            self._step_event_order = []

        # Display the new entry
        if self.display and new_entry is not None:
            new_entry.display()

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
        return f"Short term memory:\n {self.format_short_term()}\n"

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
