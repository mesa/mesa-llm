from __future__ import annotations

import pytest

import mesa_llm.tools as tool_exports
from mesa_llm.tools.tool_decorator import _GLOBAL_TOOL_REGISTRY, _TOOL_CALLBACKS
from mesa_llm.tools.tool_manager import ToolManager


@pytest.fixture(autouse=True)
def restore_global_tool_registry():
    """Keep migration assertions isolated from ad hoc tool registrations."""
    original_registry = dict(_GLOBAL_TOOL_REGISTRY)
    original_callbacks = list(_TOOL_CALLBACKS)
    _GLOBAL_TOOL_REGISTRY.clear()
    _TOOL_CALLBACKS.clear()
    ToolManager.instances.clear()
    yield
    _GLOBAL_TOOL_REGISTRY.clear()
    _GLOBAL_TOOL_REGISTRY.update(original_registry)
    _TOOL_CALLBACKS.clear()
    _TOOL_CALLBACKS.extend(original_callbacks)
    ToolManager.instances.clear()


def test_builtin_tools_no_longer_export_mutating_actions():
    for migrated_name in (
        "move_one_step",
        "teleport_to_location",
        "speak_to",
    ):
        assert migrated_name not in tool_exports.__all__
        assert not hasattr(tool_exports, migrated_name)


def test_tool_factories_do_not_include_mutating_builtins_or_legacy_tools():
    assert not hasattr(tool_exports, "legacy_tools")

    assert tool_exports.default_tools() == ()
    assert tool_exports.math_tools() == ()
    assert tool_exports.spatial_tools() == ()
    assert tool_exports.environment_tools() == ()
    assert tool_exports.social_query_tools() == ()
    assert tool_exports.external_tools() == ()


@pytest.mark.parametrize(
    "migrated_name",
    ["move_one_step", "teleport_to_location", "speak_to"],
)
def test_migrated_builtin_action_names_are_not_registered_tools(migrated_name):
    with pytest.raises(ValueError, match="Unknown tool name"):
        ToolManager(tools=[migrated_name])
