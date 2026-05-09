import datetime

import mesa_llm.tools.inbuilt_tools  # noqa: F401, to register inbuilt tools

from .actions.action_decorator import ActionMetadata, action
from .actions.action_manager import ActionChoice, ActionManager
from .actions.defaults import default_actions
from .actions.inbuilt_actions import wait
from .parallel_stepping import (
    enable_automatic_parallel_stepping,
    step_agents_parallel,
    step_agents_parallel_sync,
)
from .reasoning.reasoning import Observation, Plan
from .recording.record_model import record_model
from .tools.defaults import (
    default_tools,
    environment_tools,
    external_tools,
    legacy_tools,
    math_tools,
    social_query_tools,
    spatial_tools,
)
from .tools.tool_manager import ToolManager

# Enable automatic parallel stepping when mesa_llm is imported
enable_automatic_parallel_stepping()

__all__ = [
    "ActionChoice",
    "ActionManager",
    "ActionMetadata",
    "Observation",
    "Plan",
    "ToolManager",
    "action",
    "default_actions",
    "default_tools",
    "enable_automatic_parallel_stepping",
    "environment_tools",
    "external_tools",
    "legacy_tools",
    "math_tools",
    "record_model",
    "social_query_tools",
    "spatial_tools",
    "step_agents_parallel",
    "step_agents_parallel_sync",
    "wait",
]

__title__ = "Mesa-LLM"
__version__ = "0.3.2"
__license__ = "MIT"
_this_year = datetime.datetime.now(tz=datetime.UTC).date().year
__copyright__ = f"Copyright {_this_year} Project Mesa Team"
