"""Tests for the SimulationRecorder class."""

import json
import pickle
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mesa_llm.recording.simulation_recorder import SimulationEvent, SimulationRecorder


class TestSimulationEvent:
    """Test the SimulationEvent dataclass."""

    def test_simulation_event_creation(self):
        """Test creating a SimulationEvent."""
        event = SimulationEvent(
            event_id="test_001",
            timestamp=datetime.now(UTC),
            step=1,
            agent_id=123,
            event_type="test_event",
            content={"data": "test"},
            metadata={"source": "test"},
        )

        assert event.event_id == "test_001"
        assert event.step == 1
        assert event.agent_id == 123
        assert event.event_type == "test_event"
        assert event.content == {"data": "test"}
        assert event.metadata == {"source": "test"}


class TestSimulationRecorder:
    """Test the SimulationRecorder class."""

    def test_initialization(self, mock_model, temp_dir):
        """Test recorder initialization."""
        recorder = SimulationRecorder(
            model=mock_model,
            output_dir=str(temp_dir),
            record_state_changes=True,
            auto_save_interval=10,
        )

        assert recorder.model == mock_model
        assert recorder.output_dir == temp_dir
        assert recorder.record_state_changes is True
        assert recorder.auto_save_interval == 10
        assert recorder.events == []
        assert len(recorder.simulation_id) == 8
        assert isinstance(recorder.start_time, datetime)
        assert recorder.previous_agent_states == {}
        assert recorder.events_since_save == 0

        # Check simulation metadata
        assert recorder.simulation_metadata["simulation_id"] == recorder.simulation_id
        assert recorder.simulation_metadata["model_class"] == "TestModel"
        assert "start_time" in recorder.simulation_metadata

    def test_record_event_basic(self, recorder):
        """Test recording a basic event."""
        recorder.record_event(
            event_type="test_event",
            content={"data": "test_data"},
            agent_id=123,
            metadata={"source": "test"},
        )

        assert len(recorder.events) == 1
        event = recorder.events[0]

        assert event.event_type == "test_event"
        assert event.content == {"data": "test_data"}
        assert event.agent_id == 123
        assert event.metadata == {"source": "test"}
        assert event.step == 0  # mock_model.steps = 0
        assert recorder.events_since_save == 1

    def test_record_event_string_content(self, recorder):
        """Test recording an event with string content."""
        recorder.record_event(
            event_type="test_event",
            content="test_string",
            agent_id=123,
        )

        assert len(recorder.events) == 1
        event = recorder.events[0]
        assert event.content == {"data": "test_string"}

    def test_record_message_event(self, recorder):
        """Test recording a message event."""
        recorder.record_event(
            event_type="message",
            content="Hello world",
            agent_id=123,
            recipient_ids=[456, 789],
        )

        assert len(recorder.events) == 1
        event = recorder.events[0]

        assert event.event_type == "message"
        assert event.content == {"message": "Hello world", "recipient_ids": [456, 789]}

    def test_record_message_event_dict_content(self, recorder):
        """Test recording a message event with dict content."""
        message_content = {"text": "Hello", "priority": "high"}
        recorder.record_event(
            event_type="message",
            content=message_content,
            agent_id=123,
            recipient_ids=[456],
        )

        event = recorder.events[0]
        assert event.content == {"message": message_content, "recipient_ids": [456]}

    def test_record_model_event(self, recorder):
        """Test recording a model-level event."""
        recorder.record_model_event(event_type="step_start", content={"step": 1})

        assert len(recorder.events) == 1
        event = recorder.events[0]

        assert event.event_type == "step_start"
        assert event.content == {"step": 1}
        assert event.agent_id is None
        assert event.metadata == {"source": "model"}

    def test_auto_save_functionality(self, mock_model, temp_dir):
        """Test auto-save functionality."""
        recorder = SimulationRecorder(
            model=mock_model,
            output_dir=str(temp_dir),
            auto_save_interval=2,
        )

        with patch.object(recorder, "save") as mock_save:
            # Record first event - should not trigger save
            recorder.record_event("test1", {"data": "1"})
            mock_save.assert_not_called()
            assert recorder.events_since_save == 1

            # Record second event - should trigger save
            recorder.record_event("test2", {"data": "2"})
            mock_save.assert_called_once()
            assert recorder.events_since_save == 0

    def test_get_agent_events(self, recorder):
        """Test filtering events by agent ID."""
        recorder.record_event("event1", {"data": "1"}, agent_id=123)
        recorder.record_event("event2", {"data": "2"}, agent_id=456)
        recorder.record_event("event3", {"data": "3"}, agent_id=123)

        agent_123_events = recorder.get_agent_events(123)
        assert len(agent_123_events) == 2
        assert all(event.agent_id == 123 for event in agent_123_events)
        assert agent_123_events[0].content == {"data": "1"}
        assert agent_123_events[1].content == {"data": "3"}

    def test_get_events_by_type(self, recorder):
        """Test filtering events by type."""
        recorder.record_event("observation", {"data": "obs1"})
        recorder.record_event("action", {"data": "act1"})
        recorder.record_event("observation", {"data": "obs2"})

        obs_events = recorder.get_events_by_type("observation")
        assert len(obs_events) == 2
        assert all(event.event_type == "observation" for event in obs_events)

    def test_get_events_by_step(self, recorder, mock_model):
        """Test filtering events by simulation step."""
        mock_model.steps = 1
        recorder.record_event("event1", {"data": "1"})

        mock_model.steps = 2
        recorder.record_event("event2", {"data": "2"})
        recorder.record_event("event3", {"data": "3"})

        step_2_events = recorder.get_events_by_step(2)
        assert len(step_2_events) == 2
        assert all(event.step == 2 for event in step_2_events)

    def test_export_agent_memory(self, recorder):
        """Test exporting agent memory data."""
        # Record some events for agent 123
        recorder.record_event("observation", {"data": "obs1"}, agent_id=123)
        recorder.record_event("plan", {"data": "plan1"}, agent_id=123)
        recorder.record_event("action", {"data": "act1"}, agent_id=123)

        memory_export = recorder.export_agent_memory(123)

        assert memory_export["agent_id"] == 123
        assert len(memory_export["events"]) == 3

        summary = memory_export["summary"]
        assert summary["total_events"] == 3
        assert set(summary["event_types"]) == {"observation", "plan", "action"}
        assert summary["active_steps"] == [0]  # mock_model.steps = 0
        assert summary["first_event"] is not None
        assert summary["last_event"] is not None

    def test_export_agent_memory_no_events(self, recorder):
        """Test exporting memory for agent with no events."""
        memory_export = recorder.export_agent_memory(999)

        assert memory_export["agent_id"] == 999
        assert memory_export["events"] == []

        summary = memory_export["summary"]
        assert summary["total_events"] == 0
        assert summary["event_types"] == []
        assert summary["active_steps"] == []
        assert summary["first_event"] is None
        assert summary["last_event"] is None

    def test_save_json_format(self, recorder, mock_model, temp_dir):
        """Test saving recording in JSON format."""
        # Add some test data
        mock_model.steps = 5
        mock_model.agents = [Mock(), Mock()]  # 2 agents
        mock_model.max_steps = 10  # Add max_steps to avoid comparison error

        recorder.record_event("test_event", {"data": "test"}, agent_id=123)

        # Save with custom filename
        filepath = recorder.save(filename="test_recording.json", format="json")

        assert filepath == recorder.session_dir / "test_recording.json"
        assert filepath.exists()

        # Load and verify content
        with open(filepath) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "events" in data
        assert "agent_summaries" in data

        # Check metadata
        metadata = data["metadata"]
        assert metadata["simulation_id"] == recorder.simulation_id
        assert metadata["model_class"] == "TestModel"
        assert metadata["total_steps"] == 5
        assert metadata["total_agents"] == 2
        assert "start_time" in metadata
        assert "end_time" in metadata
        assert "duration_minutes" in metadata

        # Check events (includes our test event plus simulation_end event)
        assert len(data["events"]) == 2

        # Check agent summaries
        assert "123" in data["agent_summaries"]

    def test_save_auto_filename(self, recorder, temp_dir):
        """Test auto-generating filename when saving."""
        # Add max_steps to mock model
        recorder.model.max_steps = 10

        # Instead of mocking datetime, just test that auto-filename generation works
        filepath = recorder.save(format="json")

        # Check that the filename follows the expected pattern
        assert filepath.parent == recorder.session_dir
        assert filepath.name == "recording.json"
        assert filepath.exists()

    def test_save_auto_filename_uses_session_dir(self, recorder, mock_model, temp_dir):
        """Saving without a filename should create recording.json inside the session dir."""
        mock_model.max_steps = 10
        filepath = recorder.save(format="json")

        assert filepath == recorder.session_dir / "recording.json"
        assert filepath.parent == recorder.session_dir
        assert filepath.exists()

    def test_autosave_reuses_checkpoint_in_session_dir(self, mock_model, temp_dir):
        """Autosave must replace a single checkpoint instead of creating a chain of files."""
        mock_model.max_steps = 10
        recorder = SimulationRecorder(
            model=mock_model,
            output_dir=str(temp_dir),
            auto_save_interval=2,
        )

        recorder.record_event("event_1", {"data": "1"})
        recorder.record_event("event_2", {"data": "2"})

        checkpoint = recorder.session_dir / "checkpoint.json"
        assert checkpoint.exists()
        assert list(recorder.session_dir.iterdir()) == [checkpoint]

        with open(checkpoint) as handle:
            data = json.load(handle)

        event_types = [event["event_type"] for event in data["events"]]
        assert event_types.count("simulation_end") == 1
        assert event_types[-1] == "simulation_end"

    def test_session_dir_is_unique_per_simulation(self, mock_model, temp_dir):
        """Each simulation should own its own unique directory and retain previous folders."""
        first = SimulationRecorder(model=mock_model, output_dir=str(temp_dir))
        second = SimulationRecorder(model=mock_model, output_dir=str(temp_dir))

        assert first.session_dir != second.session_dir
        assert first.session_dir.parent == temp_dir
        assert second.session_dir.parent == temp_dir
        assert first.session_dir.name.startswith("TestModel_")
        assert second.session_dir.name.startswith("TestModel_")
        assert len(list(temp_dir.iterdir())) == 2

    def test_custom_filename_stays_inside_session_dir(self, recorder):
        """Relative custom filenames should still be stored inside the simulation session."""
        recorder.model.max_steps = 10
        filepath = recorder.save(filename="custom.json", format="json")

        assert filepath == recorder.session_dir / "custom.json"
        assert filepath.parent == recorder.session_dir

    def test_save_pickle_format_warns(self, recorder, temp_dir):
        """Test saving recording in pickle format emits a deprecation warning."""
        recorder.model.max_steps = 10
        recorder.record_event("test_event", {"data": "test"})

        with pytest.warns(
            FutureWarning,
            match="Pickle recording support is deprecated",
        ):
            filepath = recorder.save(filename="test_recording.pkl", format="pickle")

        assert filepath == recorder.session_dir / "test_recording.pkl"
        assert filepath.exists()

        with open(filepath, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        assert "metadata" in data
        assert "events" in data
        assert "agent_summaries" in data

    def test_save_json_format(self, recorder, mock_model, temp_dir):
        """Test saving recording in JSON format."""
        # Add some test data
        mock_model.steps = 5
        mock_model.agents = [Mock(), Mock()]  # 2 agents
        mock_model.max_steps = 10  # Add max_steps to avoid comparison error

        recorder.record_event("test_event", {"data": "test"}, agent_id=123)

        # Save with custom filename
        filepath = recorder.save(filename="test_recording.json", format="json")

        assert filepath == recorder.session_dir / "test_recording.json"
        assert filepath.exists()

        # Load and verify content
        with open(filepath) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "events" in data
        assert "agent_summaries" in data

        # Check metadata
        metadata = data["metadata"]
        assert metadata["simulation_id"] == recorder.simulation_id
        assert metadata["model_class"] == "TestModel"
        assert metadata["total_steps"] == 5
        assert metadata["total_agents"] == 2
        assert "start_time" in metadata
        assert "end_time" in metadata
        assert "duration_minutes" in metadata

        # Check events (includes our test event plus simulation_end event)
        assert len(data["events"]) == 2

        # Check agent summaries
        assert "123" in data["agent_summaries"]

    def test_save_auto_filename(self, recorder, temp_dir):
        """Test auto-generating filename when saving."""
        # Add max_steps to mock model
        recorder.model.max_steps = 10

        # Instead of mocking datetime, just test that auto-filename generation works
        filepath = recorder.save(format="json")

        # Check that the filename follows the expected pattern
        assert filepath.parent == recorder.session_dir
        assert filepath.name == "recording.json"
        assert filepath.exists()

    def test_save_invalid_format(self, recorder):
        """Test saving with invalid format raises error."""
        with pytest.raises(ValueError, match="Format must be 'json' or 'pickle'"):
            recorder.save(format="xml")

    def test_save_twice_produces_single_simulation_end(self, recorder, temp_dir):
        """Calling save() twice must not duplicate simulation_end and must write the file both times."""
        recorder.model.max_steps = 10
        recorder.record_event("test_event", {"data": "test"})

        filepath1 = recorder.save(filename="test_double_save.json", format="json")
        filepath2 = recorder.save(filename="test_double_save.json", format="json")

        assert filepath1 == filepath2
        assert filepath2.exists()

        with open(filepath2) as f:
            data = json.load(f)

        simulation_end_events = [
            e for e in data["events"] if e["event_type"] == "simulation_end"
        ]
        assert len(simulation_end_events) == 1

        recorded_simulation_end_events = [
            e for e in recorder.events if e.event_type == "simulation_end"
        ]
        assert len(recorded_simulation_end_events) == 1
        assert recorder.events[-1].event_type == "simulation_end"

    def test_auto_save_checkpoint_does_not_freeze_simulation_end(
        self, mock_model, temp_dir
    ):
        """Autosave checkpoints should not leave stale simulation_end data in later exports."""
        mock_model.max_steps = 10
        mock_model.steps = 1
        recorder = SimulationRecorder(
            model=mock_model,
            output_dir=str(temp_dir),
            auto_save_interval=2,
        )

        recorder.record_event("early_event_1", {"data": "early_1"})
        recorder.record_event("early_event_2", {"data": "early_2"})
        assert recorder.events[-1].event_type == "simulation_end"

        mock_model.steps = 10
        recorder.record_event("late_event", {"data": "late"})
        assert all(event.event_type != "simulation_end" for event in recorder.events)

        filepath = recorder.save(filename="final.json", format="json")

        with open(filepath) as f:
            data = json.load(f)

        event_types = [event["event_type"] for event in data["events"]]
        assert event_types.count("simulation_end") == 1
        assert event_types[-1] == "simulation_end"

        simulation_end = data["events"][-1]
        assert simulation_end["content"]["final_step"] == 10
        assert simulation_end["content"]["status"] == "completed"
        assert recorder.events[-1].event_type == "simulation_end"
        assert recorder.events[-1].content["final_step"] == 10
        assert recorder.events[-1].content["status"] == "completed"

    def test_get_stats(self, recorder, mock_model):
        """Test getting recording statistics."""
        # Add some test events
        mock_model.steps = 3
        recorder.record_event("observation", {"data": "obs1"}, agent_id=123)
        recorder.record_event("action", {"data": "act1"}, agent_id=123)
        recorder.record_event("observation", {"data": "obs2"}, agent_id=456)

        stats = recorder.get_stats()

        assert stats["total_events"] == 3
        assert stats["unique_agents"] == 2
        assert set(stats["event_types"]) == {"observation", "action"}
        assert stats["simulation_steps"] == 3
        assert "recording_duration_minutes" in stats
        assert stats["events_per_agent"] == {123: 2, 456: 1}

    def test_completion_status_with_max_steps(self, recorder, mock_model):
        """Test completion status determination when max_steps is available."""
        mock_model.max_steps = 10
        mock_model.steps = 5

        recorder.save(filename="checkpoint.json", format="json")

        # Check that completion status is "interrupted" since steps < max_steps
        assert recorder.simulation_metadata["completion_status"] == "interrupted"

        # Test completed status
        mock_model.steps = 10
        filepath = recorder.save(filename="final.json", format="json")
        assert recorder.simulation_metadata["completion_status"] == "completed"

        with open(filepath) as f:
            data = json.load(f)

        event_types = [event["event_type"] for event in data["events"]]
        assert event_types.count("simulation_end") == 1
        assert event_types[-1] == "simulation_end"

        simulation_end_events = [
            event for event in data["events"] if event["event_type"] == "simulation_end"
        ]
        assert simulation_end_events[0]["content"]["status"] == "completed"
        assert simulation_end_events[0]["content"]["final_step"] == 10
        assert recorder.events[-1].event_type == "simulation_end"
        assert recorder.events[-1].content["status"] == "completed"
        assert recorder.events[-1].content["final_step"] == 10

    def test_completion_status_without_max_steps(self, recorder, mock_model):
        """Test completion status when max_steps is not available."""
        # Ensure max_steps is not set
        if hasattr(mock_model, "max_steps"):
            delattr(mock_model, "max_steps")

        recorder.save()

        assert recorder.simulation_metadata["completion_status"] == "unknown"

    def test_event_id_generation(self, recorder):
        """Test that event IDs are generated correctly."""
        recorder.record_event("event1", {"data": "1"})
        recorder.record_event("event2", {"data": "2"})

        event1 = recorder.events[0]
        event2 = recorder.events[1]

        assert event1.event_id == f"{recorder.simulation_id}_000000"
        assert event2.event_id == f"{recorder.simulation_id}_000001"

    def test_output_directory_creation(self, mock_model):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = Path(tmpdir) / "recordings" / "subdir"

            recorder = SimulationRecorder(
                model=mock_model, output_dir=str(nonexistent_dir)
            )

            assert nonexistent_dir.exists()
            assert recorder.output_dir == nonexistent_dir
