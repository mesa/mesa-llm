"""
Enhanced agent viewer for recorded mesa-llm simulations with rich formatting.

This module provides comprehensive analysis and visualization tools for exploring
recorded simulation data, including agent behavior, conversations, decision-making
processes, and simulation metadata. Compatible with both legacy and enhanced
simulation recorder formats.
"""

import json
import pickle
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table


class AgentViewer:
    """
    Interactive analysis tool for exploring recorded simulation data with rich terminal formatting and comprehensive agent behavior insights.
    """

    def __init__(self, recording_path: str):
        self.recording_path = Path(recording_path)
        self.data = self._load_recording()
        self.events = self.data["events"]
        self.metadata = self.data.get("metadata", {})
        self.agent_summaries = self.data.get("agent_summaries", {})
        self.agent_events = self._organize_events_by_agent()
        self.console = Console()

    def _load_recording(self):
        """Load simulation recording from file."""
        if self.recording_path.suffix == ".pkl":
            warnings.warn(
                "Pickle recording support is deprecated and will be removed "
                "in a future release. Loading .pkl files can execute "
                "arbitrary code. Use JSON recordings instead.",
                FutureWarning,
                stacklevel=3,
            )
            with open(self.recording_path, "rb") as f:
                return pickle.load(f)  # noqa: S301

        with open(self.recording_path) as f:
            return json.load(f)

    def _organize_events_by_agent(self):
        """Organize events by agent ID."""
        agent_events = defaultdict(list)
        for event in self.events:
            if event.get("agent_id") is not None:
                agent_events[event["agent_id"]].append(event)

        # Sort by timestamp
        for agent_id in agent_events:
            agent_events[agent_id].sort(key=lambda x: x["timestamp"])

        return dict(agent_events)

    @staticmethod
    def _format_action_event(content):
        """Format current and legacy action event content."""
        if not isinstance(content, dict):
            return f"ACTION: {content}"

        if "action" not in content:
            action = content.get("action_type", content.get("data", ""))
            return f"ACTION: {action}"

        action = content["action"]
        if isinstance(action, dict):
            name = action.get("name")
            lines = [f"ACTION: {name}" if name else "ACTION: Unknown action"]
            if "arguments" in action:
                lines.append(f"Arguments: {action['arguments']}")
        else:
            lines = [f"ACTION: {action}" if action else "ACTION: Unknown action"]

        if "result" in content:
            lines.append(f"Result: {content['result']}")

        return "\n".join(lines)

    def _format_event(self, event):
        """Format event content for rich display."""
        try:
            content = event.get("content", {})
            event_type = event.get("event_type", "unknown")

            if event_type == "message":
                msg = (
                    content.get("message", "")
                    if isinstance(content, dict)
                    else str(content)
                )
                recipients = (
                    content.get("recipient_ids", [])
                    if isinstance(content, dict)
                    else []
                )
                return f"MESSAGE to {recipients}: {msg}"

            elif event_type == "observation":
                lines = ["OBSERVATION"]
                if isinstance(content, dict):
                    if "self_state" in content:
                        self_state = content["self_state"]
                        lines.append(
                            f"Position: {self_state.get('location', 'Unknown')}"
                        )
                        if "internal_state" in self_state:
                            lines.append(
                                f"Internal State: {', '.join(map(str, self_state['internal_state']))}"
                            )
                    elif "data" in content:
                        lines.append(str(content["data"]))
                    else:
                        lines.append(str(content))
                else:
                    lines.append(str(content))
                return "\n".join(lines)

            elif event_type == "plan":
                lines = ["PLANNING"]
                if isinstance(content, dict):
                    if "plan_content" in content:
                        plan = content["plan_content"].get("content", "")
                        lines.append(f"Reasoning: {plan}")
                    elif "data" in content:
                        lines.append(str(content["data"]))
                    else:
                        lines.append(str(content))
                else:
                    lines.append(str(content))
                return "\n".join(lines)

            elif event_type == "action":
                return self._format_action_event(content)

            elif event_type == "state_change":
                lines = ["STATE CHANGE"]
                if isinstance(content, dict):
                    for key, value in content.items():
                        lines.append(f"{key}: {value}")
                else:
                    lines.append(str(content))
                return "\n".join(lines)

            elif event_type in ["simulation_start", "simulation_end"]:
                lines = [event_type.upper().replace("_", " ")]
                if isinstance(content, dict):
                    for key, value in content.items():
                        lines.append(f"{key}: {value}")
                else:
                    lines.append(str(content))
                return "\n".join(lines)

            else:
                # Handle any other event types
                if isinstance(content, dict):
                    if "data" in content:
                        return f"{event_type.upper()}: {content['data']}"
                    else:
                        return f"{event_type.upper()}: {content}"
                else:
                    return f"{event_type.upper()}: {content}"

        except Exception as e:
            # Fallback for any formatting errors
            return f"ERROR formatting {event.get('event_type', 'unknown')} event: {e}"

    def show_simulation_info(self):
        """Show simulation metadata and overview."""
        self.console.print("\nSimulation Information", style="bold blue")

        if self.metadata:
            info_table = Table(title="Simulation Metadata")
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value", style="green")

            # Display key metadata fields
            for key, value in self.metadata.items():
                if key in [
                    "simulation_id",
                    "start_time",
                    "end_time",
                    "model_class",
                    "total_steps",
                    "total_events",
                    "total_agents",
                    "duration_minutes",
                    "completion_status",
                ]:
                    if key == "duration_minutes" and isinstance(value, int | float):
                        v = f"{value:.2f} minutes"
                    else:
                        v = str(value)
                    info_table.add_row(key.replace("_", " ").title(), v)

            self.console.print(info_table)

        # Show agent overview
        self.console.print("\nAgent Overview", style="bold blue")

        agent_table = Table(show_header=True, header_style="bold magenta")
        agent_table.add_column("Agent ID", style="dim", width=12)
        agent_table.add_column("Total Events", justify="right")
        agent_table.add_column("Event Types", style="green")

        for agent_id in sorted(self.agent_events.keys()):
            events = self.agent_events[agent_id]
            event_types = {e["event_type"] for e in events}
            agent_table.add_row(
                str(agent_id), str(len(events)), ", ".join(sorted(event_types))
            )

        self.console.print(agent_table)

    def list_agents(self):
        """Show all agents."""
        self.console.print("\nAvailable Agents", style="bold blue")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Agent ID", style="dim", width=12)
        table.add_column("Total Events", justify="right")
        table.add_column("Event Types", style="green")

        for agent_id in sorted(self.agent_events.keys()):
            events = self.agent_events[agent_id]
            event_types = {e["event_type"] for e in events}
            table.add_row(
                str(agent_id), str(len(events)), ", ".join(sorted(event_types))
            )

        self.console.print(table)

    ############################### displaying of agent events ##################################

    def view_agent_timeline(self, agent_id):
        """Show agent timeline."""
        if agent_id not in self.agent_events:
            self.console.print(f"Agent {agent_id} not found.", style="red")
            return

        events = self.agent_events[agent_id]
        self.console.print(f"\nTimeline for Agent {agent_id}", style="bold blue")
        self.console.print(f"Showing {len(events)} events\n", style="dim")

        for event in events:
            timestamp = datetime.fromisoformat(event["timestamp"]).strftime("%H:%M:%S")
            title = (
                f"Step {event['step']} | {timestamp} | {event['event_type'].title()}"
            )
            formatted = self._format_event(event)

            panel = Panel(
                formatted,
                title=title,
                title_align="left",
                border_style="bright_blue"
                if event["event_type"] == "message"
                else "white",
            )
            self.console.print(panel)

    def view_agent_conversations(self, agent_id):
        """Show agent conversations."""
        if agent_id not in self.agent_events:
            self.console.print(f"Agent {agent_id} not found.", style="red")
            return

        # Get sent and received messages
        sent_messages = [
            e for e in self.agent_events[agent_id] if e["event_type"] == "message"
        ]

        received_messages = []
        for other_id, other_events in self.agent_events.items():
            if other_id == agent_id:
                continue
            for event in other_events:
                if event["event_type"] == "message" and agent_id in event[
                    "content"
                ].get("recipient_ids", []):
                    received_messages.append((other_id, event))

        self.console.print(f"\nConversations for Agent {agent_id}", style="bold blue")

        if not sent_messages and not received_messages:
            self.console.print("No conversations found for this agent.", style="yellow")
            return

        # Combine and sort by timestamp
        all_messages = []
        for msg in sent_messages:
            all_messages.append(("SENT", agent_id, msg))
        for sender_id, msg in received_messages:
            all_messages.append(("RECEIVED", sender_id, msg))

        all_messages.sort(key=lambda x: x[2]["timestamp"])

        for direction, sender_id, event in all_messages:
            timestamp = datetime.fromisoformat(event["timestamp"]).strftime("%H:%M:%S")
            message = event["content"].get("message", "")

            if direction == "SENT":
                recipients = event["content"].get("recipient_ids", [])
                content = f"To agents {recipients}: {message}"
                title = f"SENT Step {event['step']} | {timestamp}"
                style = "green"
            else:
                content = f"From agent {sender_id}: {message}"
                title = f"RECEIVED Step {event['step']} | {timestamp}"
                style = "blue"

            panel = Panel(content, title=title, title_align="left", border_style=style)
            self.console.print(panel)

    def view_agent_decisions(self, agent_id):
        """Show agent decision-making process."""
        if agent_id not in self.agent_events:
            self.console.print(f"Agent {agent_id} not found.", style="red")
            return

        events = self.agent_events[agent_id]
        decision_events = [
            e for e in events if e["event_type"] in ["observation", "plan", "action"]
        ]

        self.console.print(f"\nDecision-Making for Agent {agent_id}", style="bold blue")

        # Group by step
        steps = defaultdict(list)
        for event in decision_events:
            steps[event["step"]].append(event)

        for step in sorted(steps.keys()):
            self.console.print(f"\nStep {step} Decision Cycle", style="bold yellow")
            step_events = sorted(
                steps[step],
                key=lambda x: ["observation", "plan", "action"].index(x["event_type"]),
            )
            for event in step_events:
                formatted = self._format_event(event)
                panel = Panel(
                    formatted,
                    title=f"{event['event_type'].title()}",
                    border_style="cyan",
                )
                self.console.print(panel)

    def view_agent_summary(self, agent_id):
        """Show agent summary."""
        if agent_id not in self.agent_events:
            self.console.print(f"Agent {agent_id} not found.", style="red")
            return

        events = self.agent_events[agent_id]
        self.console.print(f"\nAgent {agent_id} Summary", style="bold blue")

        # Check if we have precomputed summary from new recorder format
        if str(agent_id) in self.agent_summaries:
            summary = self.agent_summaries[str(agent_id)]

            # Display precomputed summary info
            summary_table = Table(title="Summary Information")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green")

            summary_table.add_row("Total Events", str(summary.get("total_events", 0)))
            summary_table.add_row(
                "Event Types", ", ".join(summary.get("event_types", []))
            )
            summary_table.add_row(
                "Active Steps", str(len(summary.get("active_steps", [])))
            )

            if summary.get("first_event"):
                first_time = datetime.fromisoformat(
                    summary["first_event"].replace("Z", "+00:00")
                ).strftime("%Y-%m-%d %H:%M:%S")
                summary_table.add_row("First Event", first_time)

            if summary.get("last_event"):
                last_time = datetime.fromisoformat(
                    summary["last_event"].replace("Z", "+00:00")
                ).strftime("%Y-%m-%d %H:%M:%S")
                summary_table.add_row("Last Event", last_time)

            self.console.print(summary_table)

        # Detailed statistics table (computed from events)
        event_counts = defaultdict(int)
        for event in events:
            event_counts[event["event_type"]] += 1

        # Count received messages
        received_count = 0
        for other_id, other_events in self.agent_events.items():
            if other_id == agent_id:
                continue
            for event in other_events:
                if event["event_type"] == "message" and agent_id in event[
                    "content"
                ].get("recipient_ids", []):
                    received_count += 1

        # Activity statistics table
        activity_table = Table(title="Activity Statistics")
        activity_table.add_column("Metric", style="cyan")
        activity_table.add_column("Value", style="green")

        activity_table.add_row("Total Events", str(len(events)))
        activity_table.add_row("Messages Sent", str(event_counts["message"]))
        activity_table.add_row("Messages Received", str(received_count))
        activity_table.add_row("Observations", str(event_counts["observation"]))
        activity_table.add_row("Plans", str(event_counts["plan"]))
        activity_table.add_row("Actions", str(event_counts["action"]))

        # Add other event types if they exist
        other_types = [
            etype
            for etype in event_counts
            if etype not in ["message", "observation", "plan", "action"]
        ]
        for etype in sorted(other_types):
            activity_table.add_row(etype.title(), str(event_counts[etype]))

        self.console.print(activity_table)

    def interactive_mode(self):
        """Interactive mode for exploring agents."""
        self.console.print("Welcome to the Mesa-LLM Agent Viewer!", style="bold green")
        self.console.print(
            "Explore individual agent behavior from your recorded simulation.\n"
        )

        commands = {
            "info": "Show simulation information",
            "list": "Show all agents",
            "timeline": "View agent timeline",
            "conversations": "View agent conversations",
            "decisions": "View agent decision-making",
            "summary": "View agent summary",
            "quit": "Exit viewer",
        }

        while True:
            self.console.print("\nAvailable Commands:", style="bold blue")
            for command, description in commands.items():
                self.console.print(f"• {command} - {description}")

            command = Prompt.ask("\nEnter command").strip().lower()

            if command in ["quit", "q"]:
                self.console.print("Goodbye!", style="yellow")
                break
            elif command == "info":
                self.show_simulation_info()
            elif command == "list":
                self.list_agents()
            else:
                parts = command.split()
                if len(parts) >= 2:
                    try:
                        agent_id = int(parts[1])
                        cmd = parts[0]
                        if cmd == "timeline":
                            self.view_agent_timeline(agent_id)
                        elif cmd == "conversations":
                            self.view_agent_conversations(agent_id)
                        elif cmd == "decisions":
                            self.view_agent_decisions(agent_id)
                        elif cmd == "summary":
                            self.view_agent_summary(agent_id)
                        else:
                            self.console.print(f"Unknown command: {cmd}", style="red")
                    except ValueError:
                        self.console.print(
                            "Invalid agent ID. Please enter a number.", style="red"
                        )
                else:
                    self.console.print("Usage: <command> <agent_id>", style="red")


def quick_agent_view(
    recording_path: str, agent_id: int | None = None, view_type: str = "summary"
):
    """Quick view of a specific agent or simulation info."""
    viewer = AgentViewer(recording_path)

    if agent_id is None or view_type == "info":
        viewer.show_simulation_info()
    elif view_type == "timeline":
        viewer.view_agent_timeline(agent_id)
    elif view_type == "conversations":
        viewer.view_agent_conversations(agent_id)
    elif view_type == "decisions":
        viewer.view_agent_decisions(agent_id)
    else:
        viewer.view_agent_summary(agent_id)


if __name__ == "__main__":
    """
    run the model with:
    conda activate mesa-llm && python -m mesa_llm.recording.agent_analysis
    """
    path = "recordings/simulation_ca45dffb_20250623_050620.json"
    viewer = AgentViewer(path)
    viewer.interactive_mode()
