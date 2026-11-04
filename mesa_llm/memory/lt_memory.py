from collections import deque
from typing import TYPE_CHECKING, Any

from mesa_llm.memory.memory import Memory, MemoryEntry, _format_message_entry

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

_DEFAULT_COMMUNICATION_HISTORY_CAPACITY = 50


class LongTermMemory(Memory):
    """
    Purely long-term memory class that tries to store everything the agent experiences.

    Attributes:
        agent : the agent that the memory belongs to
        display : whether to display the memory
        llm_model : the model to use for the summarization
        additive_event_types : event types accumulated as lists within a step.
            Defaults to ``{"message", "action"}``.
        communication_history_capacity : number of most recent messages kept
            verbatim for ``get_communication_history()``. Defaults to 50.

    """

    def __init__(
        self,
        agent: "LLMAgent",
        display: bool = True,
        llm_model: str = "openai/gpt-4o-mini",
        api_base: str | None = None,
        additive_event_types: list[str] | set[str] | tuple[str, ...] | None = None,
        communication_history_capacity: int = _DEFAULT_COMMUNICATION_HISTORY_CAPACITY,
    ):
        """
        Initialize long-term memory.

        Args:
            agent : the agent that owns this memory
            display : whether memory entries should be displayed
            llm_model : the model used for long-term summarization
            api_base : the API base URL to use for the LLM provider
            additive_event_types : event types that accumulate multiple values
                within a step instead of overwriting. Defaults to
                ``{"message", "action"}``.
            communication_history_capacity : number of most recent messages
                kept verbatim for ``get_communication_history()``
        """
        if not llm_model:
            raise ValueError(
                "llm_model must be provided for the usage of long term memory"
            )
        if (
            not isinstance(communication_history_capacity, int)
            or isinstance(communication_history_capacity, bool)
            or communication_history_capacity < 1
        ):
            raise ValueError(
                "communication_history_capacity must be a positive integer, "
                f"got {communication_history_capacity!r}"
            )

        super().__init__(
            agent=agent,
            llm_model=llm_model,
            api_base=api_base,
            display=display,
            additive_event_types=additive_event_types,
        )

        self.long_term_memory = ""
        # Consolidation folds messages into a prose summary, so keep the most
        # recent ones verbatim as (step, message) pairs.
        self._messages: deque[tuple[int, Any]] = deque(
            maxlen=communication_history_capacity
        )
        self.system_prompt = """
            You are a helpful assistant that summarizes all memory entries and stores it into long-term.
            The long term memory should be a summary of the individual memory entries such that it is concise and informative.
            """
        self.buffer = None
        if self.agent.step_prompt:
            self.system_prompt += f" This is the prompt of the problem you will be tackling:{self.agent.step_prompt}, ensure you summarize the memory entries into long-term a way that is relevant to the problem at hand."

        self.llm.system_prompt = self.system_prompt

    @property
    def _message_log(self) -> deque[tuple[int, Any]]:
        # Instances unpickled from older versions predate the message log.
        log = self.__dict__.get("_messages")
        if log is None:
            log = self._messages = deque(maxlen=_DEFAULT_COMMUNICATION_HISTORY_CAPACITY)
        return log

    def add_to_memory(self, type: str, content: dict):
        super().add_to_memory(type, content)
        if type == "message":
            self._message_log.append((self.agent.model.steps, content))

    def _build_consolidation_prompt(self) -> str:
        """
        Common function for both _update_long_term_memory() and _aupdate_long_term_memory()
        that provides it with a common prompt to redeuce code redundancy
        """
        return f"""
            This is the current Long term memory:
                {self.long_term_memory}
            This is the new memory entry:
                {self.buffer}

            """

    def _update_long_term_memory(self):
        """
        Update the long term memory by summarizing the short term memory with a LLM
        """
        prompt = self._build_consolidation_prompt()
        response = self.llm.generate(prompt)
        self.long_term_memory = response.choices[0].message.content

    async def _aupdate_long_term_memory(self):
        """
        Asynchronous version of _update_long_term_memory
        """
        prompt = self._build_consolidation_prompt()
        response = await self.llm.agenerate(prompt)
        self.long_term_memory = response.choices[0].message.content

    def _prepare_consolidation(self):
        """Install a complete candidate while retaining rollback aliases."""
        staged_entry = self.buffer
        current_content = self.step_content
        current_event_order = self._step_event_order
        old_long_term_memory = self.long_term_memory

        merged_content = self._merge_step_contents(
            current_content, staged_entry.content
        )
        candidate = MemoryEntry(
            agent=self.agent,
            content=merged_content,
            step=self.agent.model.steps,
        )
        staged_event_order = self._normalized_step_event_order(
            staged_entry.content,
            getattr(staged_entry, "_event_order", ()),
        )
        normalized_current_event_order = self._normalized_step_event_order(
            current_content,
            current_event_order,
        )
        candidate._event_order = [
            *staged_event_order,
            *normalized_current_event_order,
        ]

        self.buffer, self.step_content = candidate, {}
        self._reset_step_event_order()
        return (
            staged_entry,
            current_content,
            current_event_order,
            old_long_term_memory,
            candidate,
        )

    def _restore_failed_consolidation(self, transaction) -> None:
        """Roll back a failed consolidation without losing reentrant arrivals."""
        (
            staged_entry,
            current_content,
            current_event_order,
            old_long_term_memory,
            _,
        ) = transaction
        fresh_content = self.step_content
        fresh_event_order = self._step_event_order
        normalized_current_order = self._normalized_step_event_order(
            current_content,
            current_event_order,
        )
        normalized_fresh_order = self._normalized_step_event_order(
            fresh_content,
            fresh_event_order,
        )

        try:
            for event_type, value in fresh_content.items():
                if event_type not in self.additive_event_types:
                    current_content[event_type] = value
                    continue

                if event_type not in current_content:
                    current_content[event_type] = value
                    continue

                existing = current_content[event_type]
                if isinstance(existing, list):
                    if isinstance(value, list):
                        existing.extend(value)
                    else:
                        existing.append(value)
                elif isinstance(value, list):
                    current_content[event_type] = [existing, *value]
                else:
                    current_content[event_type] = [existing, value]
            current_event_order[:] = [
                *normalized_current_order,
                *normalized_fresh_order,
            ]
        finally:
            staged_entry.step = None
            self.long_term_memory = old_long_term_memory
            self.buffer = staged_entry
            self.step_content = current_content
            self._step_event_order = current_event_order

    def process_step(self, pre_step: bool = False):
        """
        Process the step of the agent:
        - Merge the new entry into long term memory
        - Display the new entry (Will display it only when a new entry is created in this call)
        """
        created = False

        if pre_step:
            new_entry = MemoryEntry(
                agent=self.agent,
                content=self.step_content,
                step=None,
            )
            new_entry._event_order = list(self._step_event_order)
            self.buffer = new_entry
            self.step_content = {}
            self._reset_step_event_order()
            return

        elif self.buffer and self.buffer.step is None:
            transaction = self._prepare_consolidation()
            try:
                self._update_long_term_memory()
            except BaseException:
                self._restore_failed_consolidation(transaction)
                raise
            created = True

        if self.display and created:
            transaction[-1].display()

    async def aprocess_step(self, pre_step: bool = False):
        """
        Asynchronous version of process_step (non-blocking)
        """
        created = False

        if pre_step:
            new_entry = MemoryEntry(
                agent=self.agent,
                content=self.step_content,
                step=None,
            )
            new_entry._event_order = list(self._step_event_order)
            self.buffer = new_entry
            self.step_content = {}
            self._reset_step_event_order()
            return

        elif self.buffer and self.buffer.step is None:
            transaction = self._prepare_consolidation()
            try:
                await self._aupdate_long_term_memory()
            except BaseException:
                self._restore_failed_consolidation(transaction)
                raise
            created = True

        if self.display and created:
            transaction[-1].display()

    def format_long_term(self) -> str:
        """
        Get the long term memory
        """
        return str(self.long_term_memory)

    def get_prompt_ready(self) -> str:
        return f"Long term memory: \n{self.format_long_term()}"

    def get_communication_history(self) -> str:
        """
        Get the communication history
        """
        return "\n".join(
            f"Step {step}: {_format_message_entry(msg)}\n\n"
            for step, msg in self._message_log
        )
