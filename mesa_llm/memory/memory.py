from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

from mesa_llm.module_llm import ModuleLLM

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


def _format_message_entry(msg_value) -> str:
    """Render a message memory value as a readable string.

    Handles the nested dict produced by the speak_to action:
        {"message": "<text>", "sender": <id>, "recipients": [...]}
    as well as plain strings stored by legacy or test code.
    """
    if isinstance(msg_value, dict):
        text = msg_value.get("message", str(msg_value))
        sender = msg_value.get("sender")
        if sender is not None:
            return f"Agent {sender} says: {text}"
        return str(text)
    return str(msg_value)


def _ordered_event_payloads(
    content: dict,
    event_order,
    event_types: tuple[str, ...],
) -> list[tuple[str, object]]:
    """Return selected grouped events in their captured insertion order.

    Older entries and malformed sidecars use the historical deterministic
    mapping order. Validation happens before replay so a partially valid
    sidecar can never drop or duplicate part of a content object.
    """
    selected_types = set(event_types)
    ordered_types = (
        {marker for marker in event_order if isinstance(marker, str)}
        if isinstance(event_order, list | tuple)
        else set()
    )
    grouped_values = {
        event_type: value if isinstance(value, list) else [value]
        for event_type, value in content.items()
        if event_type in selected_types or event_type in ordered_types
    }
    fallback = [
        (event_type, payload)
        for event_type, payloads in grouped_values.items()
        if event_type in selected_types
        for payload in payloads
    ]

    if not isinstance(event_order, list | tuple) or not event_order:
        return fallback

    if any(
        not isinstance(marker, str) or marker not in grouped_values
        for marker in event_order
    ):
        return fallback
    marker_counts = Counter(event_order)
    if any(
        marker_counts[event_type] != len(payloads)
        for event_type, payloads in grouped_values.items()
    ):
        return fallback

    cursors = dict.fromkeys(grouped_values, 0)
    ordered = []
    for event_type in event_order:
        cursor = cursors[event_type]
        payload = grouped_values[event_type][cursor]
        cursors[event_type] = cursor + 1
        if event_type in selected_types:
            ordered.append((event_type, payload))
    return ordered


@dataclass
class MemoryEntry:
    """
    A data structure that stores individual memory records with content, step number, and agent reference. Each entry includes `rich` formatting for display. Content is a nested dictionary of arbitrary depth containing the entry's information. Each entry is designed to hold all the information of a given step for an agent, but can also be used to store a single event if needed.
    """

    content: dict
    step: int | None
    agent: "LLMAgent"

    def __post_init__(self):
        self._event_order: list[str] = []

    def __str__(self) -> str:
        """
        Format the memory entry as a string.
        Note : 'content' is a dict that can have nested dictionaries of arbitrary depth
        """

        def format_nested_dict(data, indent_level=0):
            lines = []
            indent = "   " * indent_level

            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"{indent}[blue]└──[/blue] [cyan]{key} :[/cyan]")
                    lines.extend(format_nested_dict(value, indent_level + 1))
                elif isinstance(value, list):
                    lines.append(f"{indent}[blue]└──[/blue] [cyan]{key} :[/cyan]")
                    next_indent = "   " * (indent_level + 1)
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            lines.append(
                                f"{next_indent}[blue]├──[/blue] [cyan]({i + 1})[/cyan]"
                            )
                            lines.extend(format_nested_dict(item, indent_level + 2))
                        else:
                            lines.append(
                                f"{next_indent}[blue]├──[/blue] [cyan]{item}[/cyan]"
                            )
                else:
                    lines.append(
                        f"{indent}[blue]└──[/blue] [cyan]{key} : [/cyan]{value}"
                    )

            return lines

        lines = []
        for key, value in self.content.items():
            if not value:
                continue

            lines.append(f"\n[bold cyan][{key.title()}][/bold cyan]")
            if isinstance(value, list):
                for i, item in enumerate(value, 1):
                    lines.append(f"   [blue]({i})[/blue]")
                    if isinstance(item, dict):
                        lines.extend(format_nested_dict(item, 2))
                    else:
                        lines.append(f"      [blue]└──[/blue] [cyan]{item}[/cyan]")
            elif isinstance(value, dict):
                lines.extend(format_nested_dict(value, 1))
            else:
                lines.append(f"   [blue]└──[/blue] [cyan]{value}[/cyan]")

        content = "\n".join(lines)

        return content

    def display(self):
        if self.agent and hasattr(self.agent, "memory") and self.agent.memory.display:
            title = f"Step [bold purple]{self.agent.model.steps}[/bold purple] [bold]|[/bold] {type(self.agent).__name__} [bold purple]{self.agent.unique_id}[/bold purple]"
            panel = Panel(
                self.__str__(),
                title=title,
                title_align="left",
                border_style="bright_blue",
                padding=(0, 1),
            )
            console = Console()
            console.print(panel)


class Memory(ABC):
    """
    Generic parent class for memory backends.

    Attributes:
        agent : the agent that the memory belongs to
        llm_model : the model to use for the summarization if used
        display : whether to display the memory
        additive_event_types : event types that accumulate multiple values
            within a step. Defaults to ``{"message", "action"}``.

    Content Addition
        - Before each agent step, the agent can add new events to the memory through `add_to_memory(type, content)` so that the memory can be used to reason about the most recent events as well as the past events.
        - During the step, content for types in ``additive_event_types`` is accumulated as a list; all other types overwrite the previous value for that step.
        - At the end of the step, the memory is processed via `process_step()`, managing when memory entries are added,consolidated, displayed, or removed

    Default behavior
        - By default, ``additive_event_types == {"message", "action"}``.
        - Repeated ``message`` or ``action`` entries within one step are accumulated as a list.
        - Repeated ``observation`` or ``plan`` entries within one step overwrite the previous value unless configured otherwise.
    """

    _STEP_EVENT_ORDER_STORAGE_KEY = "_step_event_order_storage"

    @property
    def _step_event_order(self) -> list[str]:
        """Return this instance's event-order buffer.

        Older and partially initialized memory instances may not have the
        backing state introduced for ordered additive events. Lazily creating
        it here keeps every read safe without sharing mutable class state. A
        legacy value stored under the descriptor's attribute name is migrated by
        identity when possible.
        """
        instance_state = self.__dict__
        storage_key = Memory._STEP_EVENT_ORDER_STORAGE_KEY
        event_order = instance_state.get(storage_key)

        if isinstance(event_order, list):
            instance_state.pop("_step_event_order", None)
            return event_order

        legacy_event_order = instance_state.get("_step_event_order")
        if storage_key not in instance_state and isinstance(legacy_event_order, list):
            event_order = legacy_event_order
        else:
            event_order = []

        instance_state[storage_key] = event_order
        instance_state.pop("_step_event_order", None)
        return event_order

    @_step_event_order.setter
    def _step_event_order(self, event_order: list[str]) -> None:
        """Store an event-order buffer while preserving valid list identity."""
        if not isinstance(event_order, list):
            event_order = []
        self.__dict__[Memory._STEP_EVENT_ORDER_STORAGE_KEY] = event_order
        self.__dict__.pop("_step_event_order", None)

    def __init__(
        self,
        agent: "LLMAgent",
        llm_model: str | None = None,
        display: bool = True,
        api_base: str | None = None,
        additive_event_types: list[str] | set[str] | tuple[str, ...] | None = None,
    ):
        """
        Initialize the memory

        Args:
            agent : the agent that the memory belongs to
            llm_model : the model to use for summarization
            display : whether to display memory entries in the console
            api_base : the API base URL to use for the LLM provider
            additive_event_types : event types that should accumulate multiple
                values within the same step instead of overwriting. Defaults to
                ``{"message", "action"}``. For example, ``message`` and
                ``action`` accumulate by default, while ``observation`` and
                ``plan`` overwrite unless explicitly included here.
        """
        self.agent = agent
        if llm_model:
            self.llm = ModuleLLM(llm_model=llm_model, api_base=api_base)

        self.display = display

        self.step_content: dict = {}
        self._step_event_order: list[str] = []
        if additive_event_types is None:
            additive_event_types = {"message", "action"}
        self.additive_event_types = set(additive_event_types)

    @abstractmethod
    def get_prompt_ready(self) -> str:
        """
        Get the memory in a format that can be used for reasoning
        """

    @abstractmethod
    def get_communication_history(self) -> str:
        """
        Get the communication history in a format that can be used for reasoning
        """

    @abstractmethod
    def process_step(self, pre_step: bool = False):
        r"""
        A function that is called before and after the step of the agent is called.
        It is implemented to ensure that the memory is up to date when the agent is starting a new step.

        /!\ If you consider that you do not need this function, you can write "pass" in its implementation.
        """

    # Async Function implemented as a wrapper to the sync process_step()
    async def aprocess_step(self, pre_step: bool = False):
        return self.process_step(pre_step)

    @staticmethod
    def _coerce_additive_values(value):
        if isinstance(value, list):
            return list(value)
        return [value]

    def _normalized_step_event_order(
        self,
        content: dict,
        event_order,
    ) -> list[str]:
        """Return a complete order for one staged or current step segment.

        A valid sidecar is preserved exactly. Legacy, missing, or malformed
        metadata falls back to the segment's deterministic grouped-content
        order, without using unrelated list-valued fields as event markers.
        """
        grouped_counts = {
            event_type: len(self._coerce_additive_values(value))
            for event_type, value in content.items()
            if event_type in self.additive_event_types
        }
        fallback = [
            event_type
            for event_type, value in content.items()
            if event_type in self.additive_event_types
            for _ in self._coerce_additive_values(value)
        ]

        if not isinstance(event_order, list | tuple):
            return fallback

        normalized = list(event_order)
        if any(
            not isinstance(marker, str) or marker not in grouped_counts
            for marker in normalized
        ):
            return fallback
        if Counter(normalized) != Counter(grouped_counts):
            return fallback
        return normalized

    def _merge_step_contents(self, current_content: dict, staged_content: dict) -> dict:
        """
        Merge the current step buffer with staged pre-step content.

        Non-additive keys keep the staged value, matching the previous
        overwrite semantics during finalization. Additive event types are
        concatenated in chronological order so events from both halves of the
        step are preserved.
        """
        merged = dict(current_content)
        for key, staged_value in staged_content.items():
            if key in self.additive_event_types and key in merged:
                merged[key] = self._coerce_additive_values(
                    staged_value
                ) + self._coerce_additive_values(merged[key])
            else:
                merged[key] = staged_value
        return merged

    def add_to_memory(self, type: str, content: dict):
        """
        Add a new entry to the memory.

        Event types in ``self.additive_event_types`` accumulate multiple values
        within the same step. All other types use overwrite semantics.
        By default, ``self.additive_event_types == {"message", "action"}``.
        For example, repeated ``message`` entries are stored as a list, while
        repeated ``observation`` entries overwrite the previous value.
        """
        if not isinstance(content, dict):
            raise TypeError(
                "Expected 'content' to be dict, "
                f"got {content.__class__.__name__}: {content!r}"
            )

        if type in self.additive_event_types:
            # Accumulate discrete events so concurrent entries are preserved
            existing = self.step_content.get(type)
            if existing is None:
                self.step_content[type] = [content]
            elif isinstance(existing, list):
                existing.append(content)
            else:
                # Migrate a legacy single-dict entry into a list
                self.step_content[type] = [existing, content]
            self._step_event_order.append(type)
        else:
            self.step_content[type] = content

    # Async Function wrapper for add_to_memory()
    async def aadd_to_memory(self, type: str, content: dict):
        return self.add_to_memory(type, content)
