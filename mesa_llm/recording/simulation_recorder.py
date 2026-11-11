"""
Comprehensive simulation recorder for mesa-llm simulations.

This module provides tools to record all simulation events for post-analysis,
including agent observations, plans, actions, messages, and state changes.
"""

import contextlib
import json
import logging
import os
import pickle
import tempfile
import uuid
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SimulationEvent:
    """
    Dataclass representing a single recorded event in the simulation with complete context and metadata.

    Attributes:
        - **event_id** (*str*) - Unique identifier for this event
        - **timestamp** (*datetime*) - UTC timestamp when event occurred
        - **step** (*int*) - Simulation step number
        - **agent_id** (*int | None*) - Agent associated with event (None for model events)
        - **event_type** (*str*) - Type of event (observation, plan, action, message, state_change, etc.)
        - **content** (*dict*) - Event-specific data and information
        - **metadata** (*dict*) - Additional contextual metadata
    """

    event_id: str
    timestamp: datetime
    step: int
    agent_id: int | None
    event_type: str
    content: dict[str, Any]
    metadata: dict[str, Any]


class SimulationRecorder:
    """
    Centralized recorder for capturing all simulation events for post-analysis.
    It captures agent observations, plans, actions, messages, state changes, etc.
    as well as model-level events and transitions.

    Attributes:
        - **model** - Reference to the Mesa model being recorded
        - **events** - List of all recorded SimulationEvent objects
        - **simulation_id** - Unique identifier for this recording session
        - **start_time** - Recording start timestamp
        - **simulation_metadata** - Recording metadata and statistics
    """

    def __init__(
        self,
        model,
        output_dir: str = "recordings",
        record_state_changes: bool = True,
        auto_save_interval: int | None = None,
    ):
        """
        Initialize the simulation recorder.

        Parameters:
            - **model** (*Model*) - Mesa model instance to record
            - **output_dir** (*str*) - Directory for saving recordings (default: "recordings")
            - **record_state_changes** (*bool*) - Whether to track agent state changes (default: True)
            - **auto_save_interval** (*int | None*) - Automatic save frequency in events (default: None)
        """

        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Recording configuration
        self.record_state_changes = record_state_changes
        self.auto_save_interval = auto_save_interval

        # Internal state
        self.events: list[SimulationEvent] = []
        self.simulation_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now(UTC)

        # Agent state tracking for change detection
        self.previous_agent_states: dict[int, dict[str, Any]] = {}

        # Auto-save counter
        self.events_since_save = 0

        # Create a unique directory for this simulation run while keeping
        # `output_dir` as the parent directory for backward compatibility.
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        model_name = self.model.__class__.__name__
        self.session_dir = self.output_dir / (
            f"{model_name}_{timestamp}_{self.simulation_id}"
        )
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.simulation_dir = self.session_dir

        # Initialize simulation metadata
        self.simulation_metadata = {
            "simulation_id": self.simulation_id,
            "start_time": self.start_time.isoformat(),
            "model_class": self.model.__class__.__name__,
            "recording_directory": str(self.session_dir),
        }

    def _atomic_write(self, filepath: Path, payload: Any, *, format: str):
        """Write a file atomically to avoid corruption during a save."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w" if format == "json" else "wb",
                dir=str(filepath.parent),
                delete=False,
                prefix=f".{filepath.name}.",
            ) as tmp_file:
                temp_path = Path(tmp_file.name)
                if format == "json":
                    json.dump(payload, tmp_file, indent=2, default=str)
                else:
                    pickle.dump(payload, tmp_file)
            os.replace(temp_path, filepath)
        except Exception:
            if temp_path is not None:
                with contextlib.suppress(OSError):
                    temp_path.unlink(missing_ok=True)
            raise

    def _resolve_save_path(self, filename: str | None, *, format: str) -> Path:
        """Resolve the final output path for a recording inside the simulation directory."""
        if filename is None:
            default_name = "recording.json" if format == "json" else "recording.pkl"
            return self.session_dir / default_name

        path = Path(filename)
        if path.is_absolute():
            return path
        return self.session_dir / path

    def record_event(
        self,
        event_type: str,
        content: dict[str, Any] | str | None = None,
        agent_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        recipient_ids: list[int] | None = None,
    ):
        """Record a simulation event.

        Args:
            event_type: Type of event to record (observation, plan, action, message, state_change, etc.)
            content: Event content as dict or string
            agent_id: ID of the agent associated with this event
            metadata: Additional metadata for the event
            recipient_ids: List of recipient IDs for message events
        """

        # Handle different content formats based on event type
        if event_type == "message":
            if isinstance(content, str | dict | list):
                formatted_content = {
                    "message": content,
                    "recipient_ids": recipient_ids or [],
                }
            else:
                formatted_content = {
                    "message": content,
                    "recipient_ids": recipient_ids or [],
                }
        else:
            if isinstance(content, dict):
                formatted_content = content
            else:
                formatted_content = {"data": content}

        # If recording continues after a save/checkpoint, discard the previous
        # terminal marker so newly recorded events stay chronologically valid.
        if (
            event_type != "simulation_end"
            and self.events
            and self.events[-1].event_type == "simulation_end"
        ):
            self.events.pop()

        # Create the event
        event_id = f"{self.simulation_id}_{len(self.events):06d}"

        event = SimulationEvent(
            event_id=event_id,
            timestamp=datetime.now(UTC),
            step=self.model.steps,
            agent_id=agent_id,
            event_type=event_type,
            content=formatted_content,
            metadata=metadata,
        )

        self.events.append(event)
        self.events_since_save += 1

        # Auto-save if configured
        if (
            self.auto_save_interval
            and self.events_since_save >= self.auto_save_interval
        ):
            self.save(filename="checkpoint.json", format="json")
            self.events_since_save = 0

    def record_model_event(self, event_type: str, content: dict[str, Any]):
        """Record a model-level event."""
        self.record_event(
            event_type=event_type,
            content=content,
            agent_id=None,
            metadata={"source": "model"},
        )

    def get_agent_events(self, agent_id: int) -> list[SimulationEvent]:
        """Get all events for a specific agent."""
        return [event for event in self.events if event.agent_id == agent_id]

    def get_events_by_type(self, event_type: str) -> list[SimulationEvent]:
        """Get all events of a specific type."""
        return [event for event in self.events if event.event_type == event_type]

    def get_events_by_step(self, step: int) -> list[SimulationEvent]:
        """Get all events from a specific simulation step."""
        return [event for event in self.events if event.step == step]

    def export_agent_memory(self, agent_id: int) -> dict[str, Any]:
        """Export agent memory state for external analysis."""
        agent_events = self.get_agent_events(agent_id)

        return {
            "agent_id": agent_id,
            "events": [asdict(event) for event in agent_events],
            "summary": {
                "total_events": len(agent_events),
                "event_types": list({event.event_type for event in agent_events}),
                "active_steps": list({event.step for event in agent_events}),
                "first_event": (
                    agent_events[0].timestamp.isoformat() if agent_events else None
                ),
                "last_event": (
                    agent_events[-1].timestamp.isoformat() if agent_events else None
                ),
            },
        }

    def save(self, filename: str | None = None, format: str = "json"):
        """Save complete simulation recording.

        Args:
            filename: Optional filename. If None, auto-generates based on format.
            format: Save format, either "json" or deprecated "pickle".
        """
        if format not in ["json", "pickle"]:
            raise ValueError("Format must be 'json' or 'pickle'")

        filepath = self._resolve_save_path(filename, format=format)

        if self.events and self.events[-1].event_type == "simulation_end":
            self.events.pop()

        # Update metadata with final state
        self.simulation_metadata.update(
            {
                "end_time": datetime.now(UTC).isoformat(),
                "total_steps": self.model.steps,
                "total_events": len(self.events),
                "total_agents": len(self.model.agents),
                "duration_minutes": (
                    datetime.now(UTC) - self.start_time
                ).total_seconds()
                / 60,
                # Determine completion status gracefully when `max_steps` is absent
                "completion_status": (
                    "unknown"
                    if getattr(self.model, "max_steps", None) is None
                    else (
                        "interrupted"
                        if self.model.steps < self.model.max_steps
                        else "completed"
                    )
                ),
            }
        )

        # Replace the terminal "simulation_end" marker on each save. `save()`
        # is also used for autosave checkpoints, so the first "simulation_end"
        # marker must not be frozen forever.
        simulation_end_event = SimulationEvent(
            event_id=f"{self.simulation_id}_{len(self.events):06d}",
            timestamp=datetime.now(UTC),
            step=self.model.steps,
            agent_id=None,
            event_type="simulation_end",
            content={
                "status": (
                    "unknown"
                    if getattr(self.model, "max_steps", None) is None
                    else (
                        "interrupted"
                        if self.model.steps < self.model.max_steps
                        else "completed"
                    )
                ),
                "final_step": self.model.steps,
                "total_events": len(self.events),
            },
            metadata={"source": "model"},
        )
        self.events.append(simulation_end_event)

        # Prepare export data
        export_data = {
            "metadata": self.simulation_metadata,
            "events": [asdict(event) for event in self.events],
            "agent_summaries": {
                agent_id: self.export_agent_memory(agent_id)["summary"]
                for agent_id in {
                    event.agent_id
                    for event in self.events
                    if event.agent_id is not None
                }
            },
        }

        if format == "json":
            self._atomic_write(filepath, export_data, format="json")
        else:
            warnings.warn(
                "Pickle recording support is deprecated and will be removed "
                "in a future release. Pickle files can execute arbitrary code "
                "when loaded. Use JSON recordings instead.",
                FutureWarning,
                stacklevel=2,
            )
            self._atomic_write(filepath, export_data, format="pickle")

        logger.info("Simulation recording saved to: %s", filepath)
        return filepath

    def get_stats(self) -> dict[str, Any]:
        """Get recording statistics."""
        agent_ids = {
            event.agent_id for event in self.events if event.agent_id is not None
        }

        return {
            "total_events": len(self.events),
            "unique_agents": len(agent_ids),
            "event_types": list({event.event_type for event in self.events}),
            "simulation_steps": self.model.steps,
            "recording_duration_minutes": (
                datetime.now(UTC) - self.start_time
            ).total_seconds()
            / 60,
            "events_per_agent": {
                agent_id: len(self.get_agent_events(agent_id)) for agent_id in agent_ids
            },
        }
