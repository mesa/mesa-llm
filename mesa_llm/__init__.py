import datetime

from .actions import (
    ActionChoice,
    ActionManager,
    ActionMetadata,
    ActionPostCommitError,
    ActResult,
    action,
    default_actions,
    move_one_step,
    social_actions,
    spatial_actions,
    speak_to,
    teleport_to_location,
    wait,
)
from .parallel_stepping import (
    enable_automatic_parallel_stepping,
    step_agents_parallel,
    step_agents_parallel_sync,
)
from .reasoning.reasoning import Observation, Plan
from .recording.record_model import record_model
from .tools import (
    ToolManager,
    default_tools,
    environment_tools,
    external_tools,
    math_tools,
    social_query_tools,
    spatial_tools,
)

__all__ = [
    "ActResult",
    "ActionChoice",
    "ActionManager",
    "ActionMetadata",
    "ActionPostCommitError",
    "Observation",
    "Plan",
    "ToolManager",
    "action",
    "default_actions",
    "default_tools",
    "enable_automatic_parallel_stepping",
    "environment_tools",
    "external_tools",
    "math_tools",
    "move_one_step",
    "record_model",
    "social_actions",
    "social_query_tools",
    "spatial_actions",
    "spatial_tools",
    "speak_to",
    "step_agents_parallel",
    "step_agents_parallel_sync",
    "teleport_to_location",
    "wait",
]

__title__ = "Mesa-LLM"
__version__ = "0.3.2"
__license__ = "MIT"
_this_year = datetime.datetime.now(tz=datetime.UTC).date().year
__copyright__ = f"Copyright {_this_year} Project Mesa Team"
