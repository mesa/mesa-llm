"""Tests for the AgentViewer class and agent analysis functionality."""

import json
from unittest.mock import Mock, patch

import pytest

from mesa_llm.recording.agent_analysis import AgentViewer, quick_agent_view


class TestAgentViewer:
    """Test the AgentViewer class."""

    def test_init_with_json_file(self, temp_recording_file, sample_recording_data):
        """Test initializing AgentViewer with JSON file."""
        json_path, _ = temp_recording_file

        viewer = AgentViewer(str(json_path))

        assert viewer.recording_path == json_path
        assert viewer.data == sample_recording_data
        assert len(viewer.events) == 7
        assert viewer.metadata == sample_recording_data["metadata"]
        assert viewer.agent_summaries == sample_recording_data["agent_summaries"]
        assert 123 in viewer.agent_events
        assert 456 in viewer.agent_events

    def test_init_with_pickle_file_warns(
        self, temp_recording_file, sample_recording_data
    ):
        """Test pickle recordings load with a deprecation warning."""
        _, pkl_path = temp_recording_file

        with pytest.warns(
            FutureWarning,
            match="Pickle recording support is deprecated",
        ):
            viewer = AgentViewer(str(pkl_path))

        assert viewer.data == sample_recording_data
        assert len(viewer.events) == 7

    def test_organize_events_by_agent(self, temp_recording_file):
        """Test organizing events by agent ID."""
        json_path, _ = temp_recording_file
        viewer = AgentViewer(str(json_path))

        assert len(viewer.agent_events[123]) == 4
        assert len(viewer.agent_events[456]) == 3

        # Check that events are sorted by timestamp
        agent_123_events = viewer.agent_events[123]
        timestamps = [event["timestamp"] for event in agent_123_events]
        assert timestamps == sorted(timestamps)

    def test_format_event_message(self, temp_recording_file):
        """Test formatting message events."""
        json_path, _ = temp_recording_file
        viewer = AgentViewer(str(json_path))

        # Get message event
        message_event = viewer.agent_events[123][3]  # Last event for agent 123
        formatted = viewer._format_event(message_event)

        assert "MESSAGE to [456]: Hello, agent 456!" in formatted

    def test_format_event_observation_with_self_state(self, temp_recording_file):
        """Test formatting observation events with self_state."""
        json_path, _ = temp_recording_file
        viewer = AgentViewer(str(json_path))

        # Get observation event with self_state
        obs_event = viewer.agent_events[123][0]
        formatted = viewer._format_event(obs_event)

        assert "OBSERVATION" in formatted
        assert "Position: [1, 2]" in formatted
        assert "Internal State: happy, energetic" in formatted

    def test_format_event_observation_with_data(self, temp_recording_file):
        """Test formatting observation events with data field."""
        json_path, _ = temp_recording_file
        viewer = AgentViewer(str(json_path))

        # Get observation event with data
        obs_event = viewer.agent_events[456][1]  # Second event for agent 456
        formatted = viewer._format_event(obs_event)

        assert "OBSERVATION" in formatted
        assert "I see agent 123 nearby" in formatted

    def test_format_event_plan(self, temp_recording_file):
        """Test formatting plan events."""
        json_path, _ = temp_recording_file
        viewer = AgentViewer(str(json_path))

        # Get plan event
        plan_event = viewer.agent_events[123][1]
        formatted = viewer._format_event(plan_event)

        assert "PLANNING" in formatted
        assert "Reasoning: I will move north to explore" in formatted

    def test_format_event_action(self, temp_recording_file):
        """Test formatting action events."""
        json_path, _ = temp_recording_file
        viewer = AgentViewer(str(json_path))

        # Get action event
        action_event = viewer.agent_events[123][2]
        formatted = viewer._format_event(action_event)

        assert "ACTION: move" in formatted

    @pytest.mark.parametrize("rationale", [None, "The listener is nearby."])
    def test_format_event_action_with_structured_choice(
        self, temp_recording_file, rationale
    ):
        """Test formatting the structured action events recorded by LLMAgent."""
        json_path, _ = temp_recording_file
        viewer = AgentViewer(str(json_path))
        action = {
            "name": "speak_to",
            "arguments": {
                "listener_agents_unique_ids": [456],
                "message": "Hello, neighbor!",
            },
        }
        if rationale is not None:
            action["rationale"] = rationale
        event = {
            "event_type": "action",
            "content": {
                "action": action,
                "result": {"delivered": True},
            },
        }

        formatted = viewer._format_event(event)
        normalized = " ".join(formatted.split())

        assert "speak_to" in normalized
        assert "listener_agents_unique_ids" in normalized
        assert "456" in normalized
        assert "message" in normalized
        assert "Hello, neighbor!" in normalized
        assert "delivered" in normalized
        assert "True" in normalized
        assert formatted.strip() not in {"ACTION", "ACTION:"}

    def test_format_event_action_with_legacy_data(self, temp_recording_file):
        """Test that legacy data-based action events remain readable."""
        json_path, _ = temp_recording_file
        viewer = AgentViewer(str(json_path))
        event = {
            "event_type": "action",
            "content": {"data": {"direction": "north"}},
        }

        formatted = viewer._format_event(event)

        assert "ACTION" in formatted
        assert "direction" in formatted
        assert "north" in formatted

    @pytest.mark.parametrize("content", ["wait", 42, None])
    def test_format_event_action_with_non_dict_content(
        self, temp_recording_file, content
    ):
        """Test that non-dictionary action content remains readable."""
        json_path, _ = temp_recording_file
        viewer = AgentViewer(str(json_path))

        formatted = viewer._format_event({"event_type": "action", "content": content})

        assert "ACTION" in formatted
        assert str(content) in formatted
        assert "ERROR formatting" not in formatted

    @pytest.mark.parametrize(
        ("content", "expected_fragment"),
        [
            (
                {
                    "action": {"arguments": {"distance": 2}},
                    "result": "moved",
                },
                "moved",
            ),
            ({"action": "wait", "result": "finished"}, "wait"),
            ({"action": None, "result": "not executed"}, "not executed"),
            (
                {
                    "action": {
                        "name": "move_one_step",
                        "arguments": object(),
                    },
                    "result": object(),
                },
                "move_one_step",
            ),
            ({"action": {}}, "Unknown action"),
        ],
    )
    def test_format_event_action_with_partial_or_malformed_content(
        self, temp_recording_file, content, expected_fragment
    ):
        """Test that partial action structures degrade without raising."""
        json_path, _ = temp_recording_file
        viewer = AgentViewer(str(json_path))

        formatted = viewer._format_event({"event_type": "action", "content": content})

        assert "ACTION" in formatted
        assert expected_fragment in formatted
        assert "ERROR formatting" not in formatted
        assert formatted.strip() not in {"ACTION", "ACTION:"}

    def test_format_event_state_change(self, temp_recording_file):
        """Test formatting state_change events."""
        json_path, _ = temp_recording_file
        viewer = AgentViewer(str(json_path))

        # Get state_change event
        state_event = viewer.agent_events[456][2]
        formatted = viewer._format_event(state_event)

        assert "STATE CHANGE" in formatted
        assert "position: [2, 3]" in formatted
        assert "energy: 95" in formatted

    def test_format_event_unknown_type(self, temp_recording_file):
        """Test formatting unknown event types."""
        json_path, _ = temp_recording_file
        viewer = AgentViewer(str(json_path))

        # Create fake event with unknown type
        unknown_event = {"event_type": "unknown_type", "content": {"data": "test_data"}}

        formatted = viewer._format_event(unknown_event)
        assert "UNKNOWN_TYPE: test_data" in formatted

    def test_format_event_error_handling(self, temp_recording_file):
        """Test that formatting handles None content gracefully."""
        json_path, _ = temp_recording_file
        viewer = AgentViewer(str(json_path))

        # Create event with None content
        bad_event = {
            "event_type": "message",
            "content": None,  # This should be handled gracefully
        }

        formatted = viewer._format_event(bad_event)
        # The method handles None content gracefully, so we expect it to work
        assert "MESSAGE to []: None" in formatted

    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_show_simulation_info(self, mock_console_class, temp_recording_file):
        """Test showing simulation information."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        viewer = AgentViewer(str(json_path))
        viewer.show_simulation_info()

        # Check that console.print was called
        assert mock_console.print.call_count > 0

        # Check that tables were created and printed
        calls = mock_console.print.call_args_list
        call_args = [call[0][0] if call[0] else None for call in calls]

        # Should print simulation info and agent overview
        assert any("Simulation Information" in str(arg) for arg in call_args)

    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_list_agents(self, mock_console_class, temp_recording_file):
        """Test listing agents."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        viewer = AgentViewer(str(json_path))
        viewer.list_agents()

        # Check that console.print was called
        assert mock_console.print.call_count > 0

    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_view_agent_timeline(self, mock_console_class, temp_recording_file):
        """Test viewing agent timeline."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        viewer = AgentViewer(str(json_path))
        viewer.view_agent_timeline(123)

        # Should print timeline information
        assert mock_console.print.call_count > 0

    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_view_agent_timeline_nonexistent_agent(
        self, mock_console_class, temp_recording_file
    ):
        """Test viewing timeline for non-existent agent."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        viewer = AgentViewer(str(json_path))
        viewer.view_agent_timeline(999)

        # Should print error message
        mock_console.print.assert_called_with("Agent 999 not found.", style="red")

    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_view_agent_conversations(self, mock_console_class, temp_recording_file):
        """Test viewing agent conversations."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        viewer = AgentViewer(str(json_path))
        viewer.view_agent_conversations(123)

        # Should print conversation information
        assert mock_console.print.call_count > 0

    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_view_agent_conversations_no_messages(
        self, mock_console_class, temp_recording_file
    ):
        """Test viewing conversations for agent with no messages."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        # Create viewer and remove message events
        viewer = AgentViewer(str(json_path))
        viewer.agent_events[123] = [
            e for e in viewer.agent_events[123] if e["event_type"] != "message"
        ]
        viewer.agent_events[456] = [
            e for e in viewer.agent_events[456] if e["event_type"] != "message"
        ]

        viewer.view_agent_conversations(123)

        # Should indicate no conversations found
        calls = mock_console.print.call_args_list
        assert any(
            "No conversations found" in str(call[0][0]) for call in calls if call[0]
        )

    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_view_agent_decisions(self, mock_console_class, temp_recording_file):
        """Test viewing agent decision-making process."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        viewer = AgentViewer(str(json_path))
        viewer.view_agent_decisions(123)

        # Should print decision information
        assert mock_console.print.call_count > 0

    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_view_agent_summary(self, mock_console_class, temp_recording_file):
        """Test viewing agent summary."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        viewer = AgentViewer(str(json_path))
        viewer.view_agent_summary(123)

        # Should print summary information
        assert mock_console.print.call_count > 0

    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_view_agent_summary_with_precomputed_data(
        self, mock_console_class, temp_recording_file
    ):
        """Test viewing agent summary with precomputed summary data."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        viewer = AgentViewer(str(json_path))
        viewer.view_agent_summary(123)

        # Should use precomputed summary data
        assert mock_console.print.call_count > 0

    @patch("mesa_llm.recording.agent_analysis.Prompt.ask")
    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_interactive_mode_quit(
        self, mock_console_class, mock_prompt, temp_recording_file
    ):
        """Test interactive mode quit command."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        mock_prompt.return_value = "quit"

        viewer = AgentViewer(str(json_path))
        viewer.interactive_mode()

        # Should print goodbye message
        calls = mock_console.print.call_args_list
        assert any("Goodbye!" in str(call[0][0]) for call in calls if call[0])

    @patch("mesa_llm.recording.agent_analysis.Prompt.ask")
    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_interactive_mode_info_command(
        self, mock_console_class, mock_prompt, temp_recording_file
    ):
        """Test interactive mode info command."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        mock_prompt.side_effect = ["info", "quit"]

        viewer = AgentViewer(str(json_path))
        with patch.object(viewer, "show_simulation_info") as mock_show_info:
            viewer.interactive_mode()
            mock_show_info.assert_called_once()

    @patch("mesa_llm.recording.agent_analysis.Prompt.ask")
    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_interactive_mode_list_command(
        self, mock_console_class, mock_prompt, temp_recording_file
    ):
        """Test interactive mode list command."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        mock_prompt.side_effect = ["list", "quit"]

        viewer = AgentViewer(str(json_path))
        with patch.object(viewer, "list_agents") as mock_list:
            viewer.interactive_mode()
            mock_list.assert_called_once()

    @patch("mesa_llm.recording.agent_analysis.Prompt.ask")
    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_interactive_mode_agent_commands(
        self, mock_console_class, mock_prompt, temp_recording_file
    ):
        """Test interactive mode agent-specific commands."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        mock_prompt.side_effect = ["timeline 123", "summary 456", "quit"]

        viewer = AgentViewer(str(json_path))
        with (
            patch.object(viewer, "view_agent_timeline") as mock_timeline,
            patch.object(viewer, "view_agent_summary") as mock_summary,
        ):
            viewer.interactive_mode()
            mock_timeline.assert_called_once_with(123)
            mock_summary.assert_called_once_with(456)

    @patch("mesa_llm.recording.agent_analysis.Prompt.ask")
    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_interactive_mode_invalid_agent_id(
        self, mock_console_class, mock_prompt, temp_recording_file
    ):
        """Test interactive mode with invalid agent ID."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        mock_prompt.side_effect = ["timeline abc", "quit"]

        viewer = AgentViewer(str(json_path))
        viewer.interactive_mode()

        # Should print error message about invalid agent ID
        calls = mock_console.print.call_args_list
        assert any("Invalid agent ID" in str(call[0][0]) for call in calls if call[0])

    @patch("mesa_llm.recording.agent_analysis.Prompt.ask")
    @patch("mesa_llm.recording.agent_analysis.Console")
    def test_interactive_mode_unknown_command(
        self, mock_console_class, mock_prompt, temp_recording_file
    ):
        """Test interactive mode with unknown command."""
        json_path, _ = temp_recording_file
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        mock_prompt.side_effect = ["unknown_command 123", "quit"]

        viewer = AgentViewer(str(json_path))
        viewer.interactive_mode()

        # Should print error message about unknown command
        calls = mock_console.print.call_args_list
        assert any("Unknown command" in str(call[0][0]) for call in calls if call[0])

    def test_load_recording_missing_metadata(self, temp_recording_file):
        """Test loading recording with missing metadata."""
        json_path, _ = temp_recording_file

        # Modify the file to remove metadata
        with open(json_path) as f:
            data = json.load(f)

        del data["metadata"]

        with open(json_path, "w") as f:
            json.dump(data, f)

        viewer = AgentViewer(str(json_path))

        # Should handle missing metadata gracefully
        assert viewer.metadata == {}

    def test_load_recording_missing_agent_summaries(self, temp_recording_file):
        """Test loading recording with missing agent summaries."""
        json_path, _ = temp_recording_file

        # Modify the file to remove agent_summaries
        with open(json_path) as f:
            data = json.load(f)

        del data["agent_summaries"]

        with open(json_path, "w") as f:
            json.dump(data, f)

        viewer = AgentViewer(str(json_path))

        # Should handle missing agent summaries gracefully
        assert viewer.agent_summaries == {}


class TestQuickAgentView:
    """Test the quick_agent_view function."""

    @patch("mesa_llm.recording.agent_analysis.AgentViewer")
    def test_quick_agent_view_info(self, mock_viewer_class, temp_recording_file):
        """Test quick_agent_view with info view type."""
        json_path, _ = temp_recording_file
        mock_viewer = Mock()
        mock_viewer_class.return_value = mock_viewer

        quick_agent_view(str(json_path), view_type="info")

        mock_viewer_class.assert_called_once_with(str(json_path))
        mock_viewer.show_simulation_info.assert_called_once()

    @patch("mesa_llm.recording.agent_analysis.AgentViewer")
    def test_quick_agent_view_agent_timeline(
        self, mock_viewer_class, temp_recording_file
    ):
        """Test quick_agent_view with timeline view type."""
        json_path, _ = temp_recording_file
        mock_viewer = Mock()
        mock_viewer_class.return_value = mock_viewer

        quick_agent_view(str(json_path), agent_id=123, view_type="timeline")

        mock_viewer.view_agent_timeline.assert_called_once_with(123)

    @patch("mesa_llm.recording.agent_analysis.AgentViewer")
    def test_quick_agent_view_conversations(
        self, mock_viewer_class, temp_recording_file
    ):
        """Test quick_agent_view with conversations view type."""
        json_path, _ = temp_recording_file
        mock_viewer = Mock()
        mock_viewer_class.return_value = mock_viewer

        quick_agent_view(str(json_path), agent_id=123, view_type="conversations")

        mock_viewer.view_agent_conversations.assert_called_once_with(123)

    @patch("mesa_llm.recording.agent_analysis.AgentViewer")
    def test_quick_agent_view_decisions(self, mock_viewer_class, temp_recording_file):
        """Test quick_agent_view with decisions view type."""
        json_path, _ = temp_recording_file
        mock_viewer = Mock()
        mock_viewer_class.return_value = mock_viewer

        quick_agent_view(str(json_path), agent_id=123, view_type="decisions")

        mock_viewer.view_agent_decisions.assert_called_once_with(123)

    @patch("mesa_llm.recording.agent_analysis.AgentViewer")
    def test_quick_agent_view_default_summary(
        self, mock_viewer_class, temp_recording_file
    ):
        """Test quick_agent_view with default summary view type."""
        json_path, _ = temp_recording_file
        mock_viewer = Mock()
        mock_viewer_class.return_value = mock_viewer

        quick_agent_view(str(json_path), agent_id=123)

        mock_viewer.view_agent_summary.assert_called_once_with(123)

    @patch("mesa_llm.recording.agent_analysis.AgentViewer")
    def test_quick_agent_view_no_agent_id(self, mock_viewer_class, temp_recording_file):
        """Test quick_agent_view with no agent ID shows simulation info."""
        json_path, _ = temp_recording_file
        mock_viewer = Mock()
        mock_viewer_class.return_value = mock_viewer

        quick_agent_view(str(json_path), agent_id=None)

        mock_viewer.show_simulation_info.assert_called_once()
