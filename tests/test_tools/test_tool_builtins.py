from __future__ import annotations

import importlib

import pytest

import mesa_llm
import mesa_llm.tools as tool_exports
import mesa_llm.tools.defaults as canonical_defaults
from mesa_llm.tools.defaults import (
    default_tools,
    environment_tools,
    external_tools,
    math_tools,
    social_query_tools,
    spatial_tools,
)
from mesa_llm.tools.tool_decorator import _GLOBAL_TOOL_REGISTRY, _TOOL_CALLBACKS
from mesa_llm.tools.tool_manager import ToolManager

CANONICAL_FACTORIES = {
    "default_tools": default_tools,
    "math_tools": math_tools,
    "spatial_tools": spatial_tools,
    "environment_tools": environment_tools,
    "social_query_tools": social_query_tools,
    "external_tools": external_tools,
}
MIGRATED_MUTATING_TOOL_NAMES = (
    "move_one_step",
    "teleport_to_location",
    "speak_to",
)
REMOVED_TOOL_NAMES = (*MIGRATED_MUTATING_TOOL_NAMES, "legacy_tools")
REGISTERED_TOOL_NAMES_AFTER_PACKAGE_IMPORT = frozenset(_GLOBAL_TOOL_REGISTRY)


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


def test_defaults_has_exact_canonical_factory_exports():
    assert set(canonical_defaults.__all__) == set(CANONICAL_FACTORIES)


@pytest.mark.parametrize(("factory_name", "factory"), CANONICAL_FACTORIES.items())
def test_canonical_factory_returns_current_empty_tuple(factory_name, factory):
    assert factory.__name__ == factory_name
    assert factory.__module__ == "mesa_llm.tools.defaults"

    result = factory()

    assert isinstance(result, tuple)
    assert result == ()


@pytest.mark.parametrize(("factory_name", "factory"), CANONICAL_FACTORIES.items())
def test_tools_and_package_roots_reexport_canonical_factory(factory_name, factory):
    assert getattr(canonical_defaults, factory_name) is factory
    assert getattr(tool_exports, factory_name) is factory
    assert getattr(mesa_llm, factory_name) is factory


def test_builtin_tools_no_longer_export_mutating_actions():
    for migrated_name in MIGRATED_MUTATING_TOOL_NAMES:
        assert migrated_name not in tool_exports.__all__
        assert not hasattr(tool_exports, migrated_name)
        assert migrated_name not in canonical_defaults.__all__
        assert not hasattr(canonical_defaults, migrated_name)
        assert migrated_name not in _GLOBAL_TOOL_REGISTRY


def test_package_import_does_not_register_removed_tools():
    assert set(REMOVED_TOOL_NAMES).isdisjoint(
        REGISTERED_TOOL_NAMES_AFTER_PACKAGE_IMPORT
    )


def test_tool_factories_do_not_include_mutating_builtins_or_legacy_tools():
    assert "legacy_tools" not in tool_exports.__all__
    assert not hasattr(tool_exports, "legacy_tools")
    assert "legacy_tools" not in canonical_defaults.__all__
    assert not hasattr(canonical_defaults, "legacy_tools")
    assert "legacy_tools" not in _GLOBAL_TOOL_REGISTRY

    assert tool_exports.default_tools() == ()
    assert tool_exports.math_tools() == ()
    assert tool_exports.spatial_tools() == ()
    assert tool_exports.environment_tools() == ()
    assert tool_exports.social_query_tools() == ()
    assert tool_exports.external_tools() == ()


@pytest.mark.parametrize(
    "migrated_name",
    MIGRATED_MUTATING_TOOL_NAMES,
)
def test_migrated_builtin_action_names_are_not_registered_tools(migrated_name):
    with pytest.raises(ValueError, match="Unknown tool name"):
        ToolManager(tools=[migrated_name])


def test_legacy_tools_name_is_not_registered():
    with pytest.raises(ValueError, match="Unknown tool name"):
        ToolManager(tools=["legacy_tools"])


def test_removed_tools_builtins_module_is_not_a_compatibility_path():
    with pytest.raises(ModuleNotFoundError) as exc_info:
        importlib.import_module("mesa_llm.tools.builtins")

    assert exc_info.value.name == "mesa_llm.tools.builtins"
