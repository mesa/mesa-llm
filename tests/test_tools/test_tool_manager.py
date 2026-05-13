from unittest.mock import Mock

import pytest

from mesa_llm.tools import (
    default_tools,
    environment_tools,
    external_tools,
    math_tools,
    social_query_tools,
    spatial_tools,
)
from mesa_llm.tools.tool_decorator import (
    _GLOBAL_TOOL_REGISTRY,
    _TOOL_CALLBACKS,
    add_tool_callback,
    tool,
)
from mesa_llm.tools.tool_manager import ToolManager


class TestToolManager:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear global registry to start fresh
        _GLOBAL_TOOL_REGISTRY.clear()
        _TOOL_CALLBACKS.clear()
        # Clear instances list
        ToolManager.instances.clear()

    def teardown_method(self):
        """Clean up after each test method."""
        _GLOBAL_TOOL_REGISTRY.clear()
        _TOOL_CALLBACKS.clear()
        ToolManager.instances.clear()

    def test_init_empty(self):
        """Test initialization with no tools."""
        manager = ToolManager()
        assert isinstance(manager.tools, dict)
        assert len(manager.tools) == 0
        assert manager in ToolManager.instances

        none_manager = ToolManager(tools=None)
        list_manager = ToolManager(tools=[])
        assert none_manager.tools == {}
        assert list_manager.tools == {}

    def test_tool_set_factories(self):
        """Tool-set factories are explicit immutable tuples."""
        assert default_tools() == ()
        assert math_tools() == ()
        assert spatial_tools() == ()
        assert environment_tools() == ()
        assert social_query_tools() == ()
        assert external_tools() == ()
        assert all(
            isinstance(pack, tuple)
            for pack in (
                default_tools(),
                math_tools(),
                spatial_tools(),
                environment_tools(),
                social_query_tools(),
                external_tools(),
            )
        )

    def test_init_does_not_copy_global_tools(self):
        """Bare managers do not copy global tools implicitly."""

        @tool
        def test_global_tool(agent, param1: str) -> str:
            """Test global tool.
            Args:
                agent: The agent making the request (provided automatically)
                param1: A test parameter.
            Returns:
                The input parameter.
            """
            return param1

        manager = ToolManager()
        assert manager.tools == {}

        explicit_manager = ToolManager(tools=["test_global_tool"])
        assert explicit_manager.tools == {"test_global_tool": test_global_tool}

    def test_init_with_explicit_callable_tools(self):
        """Explicit callables configure exactly those tools."""

        @tool
        def tool_a(agent, x: int) -> int:
            """Tool A.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def tool_b(agent, y: int) -> int:
            """Tool B.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        manager = ToolManager(tools=[tool_a])

        assert manager.tools == {"tool_a": tool_a}
        assert "tool_b" not in manager.tools

    def test_late_bare_tool_does_not_leak_into_explicit_managers(self):
        """Bare @tool registrations after construction do not mutate managers."""
        no_tool_manager = ToolManager()

        @tool
        def initial_tool(agent, x: int) -> int:
            """Initial tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        explicit_manager = ToolManager(tools=[initial_tool])

        @tool
        def late_tool(agent, y: int) -> int:
            """Late tool.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        assert no_tool_manager.tools == {}
        assert explicit_manager.tools == {"initial_tool": initial_tool}
        assert "late_tool" not in explicit_manager.tools

    def test_late_bare_tool_does_not_leak_through_deprecated_callbacks(self):
        """Deprecated global callbacks cannot opt managers into bare tools."""
        manager = ToolManager()

        def add_to_all_callback(fn):
            ToolManager.add_tool_to_all(fn)

        with pytest.warns(DeprecationWarning, match="add_tool_callback"):
            add_tool_callback(add_to_all_callback)

        @tool
        def callback_tool(agent, x: int) -> int:
            """Callback tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        assert manager.tools == {}
        assert callback_tool is _GLOBAL_TOOL_REGISTRY["callback_tool"]

    def test_init_with_extra_tools(self):
        """Test initialization with extra tools."""

        def extra_tool(agent, x: int) -> int:
            """Extra tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input number.
            Returns:
                The input number.
            """
            return x

        extra_tool.__tool_schema__ = {
            "type": "function",
            "function": {
                "name": "extra_tool",
                "description": "Extra tool returns: The input number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "Input number."}
                    },
                    "required": ["x"],
                },
            },
        }

        extra_tools = {"alias_name": extra_tool}
        with pytest.warns(DeprecationWarning, match="extra_tools"):
            manager = ToolManager(extra_tools=extra_tools)

        assert "alias_name" in manager.tools
        assert "extra_tool" not in manager.tools

        schemas = manager.get_tools_schema()
        selected_schemas = manager.get_tools_schema(tools="alias_name")
        callable_schemas = manager.get_tools_schema(tools=extra_tool)
        assert schemas[0]["function"]["name"] == "alias_name"
        assert selected_schemas[0]["function"]["name"] == "alias_name"
        assert callable_schemas[0]["function"]["name"] == "alias_name"

        mock_tool_call = Mock()
        mock_tool_call.id = "call_alias"
        mock_tool_call.function.name = "alias_name"
        mock_tool_call.function.arguments = '{"x": 4}'
        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(Mock(), mock_response, tools="alias_name")
        assert result[0]["name"] == "alias_name"
        assert result[0]["response"] == "4"

    def test_register_tool(self):
        """Test registering a tool manually."""
        manager = ToolManager()

        def manual_tool(agent, text: str) -> str:
            """Manual tool.
            Args:
                agent: The agent making the request (provided automatically)
                text: Input text.
            Returns:
                The input text.
            """
            return text

        manager.register(manual_tool)
        assert "manual_tool" in manager.tools
        assert manager.tools["manual_tool"] == manual_tool

    def test_add_tool_to_all(self):
        """Test adding a tool to all manager instances."""
        manager1 = ToolManager()
        manager2 = ToolManager()

        def shared_tool(agent, value: str) -> str:
            """Shared tool.
            Args:
                agent: The agent making the request (provided automatically)
                value: Input value.
            Returns:
                The input value.
            """
            return value

        ToolManager.add_tool_to_all(shared_tool)

        assert "shared_tool" in manager1.tools
        assert "shared_tool" in manager2.tools

    def test_get_tool_schema_deprecated_alias(self):
        """Deprecated single-tool schema alias still works."""
        manager = ToolManager()

        @tool
        def schema_test_tool(agent, param: str) -> str:
            """Schema test tool.
            Args:
                agent: The agent making the request (provided automatically)
                param: A parameter.
            Returns:
                The parameter.
            """
            return param

        manager.register(schema_test_tool)
        with pytest.warns(DeprecationWarning, match="get_tool_schema"):
            schema = manager.get_tool_schema(schema_test_tool)

        assert "type" in schema
        assert "function" in schema
        assert schema["function"]["name"] == "schema_test_tool"

    def test_get_tool_schema_deprecated_alias_missing(self):
        """Deprecated single-tool schema alias handles missing schema."""
        manager = ToolManager()

        def no_schema_tool():
            return "test"

        with pytest.warns(DeprecationWarning, match="get_tool_schema"):
            schema = manager.get_tool_schema(no_schema_tool, "no_schema_tool")
        assert "error" in schema

    def test_private_get_tool_schema_missing(self):
        """Private single-tool helper handles missing schema."""
        manager = ToolManager()

        def no_schema_tool():
            return "test"

        schema = manager._get_tool_schema(no_schema_tool)
        assert "error" in schema

    def test_get_tools_schema(self):
        """Test getting all tools schemas."""

        @tool
        def tool1(agent, x: int) -> int:
            """Tool 1.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def tool2(agent, y: str) -> str:
            """Tool 2.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        manager = ToolManager(tools=[tool1, tool2])
        schemas = manager.get_tools_schema()

        assert len(schemas) == 2
        assert all("function" in schema for schema in schemas)

    def test_get_all_tools_schema_deprecated_alias(self):
        """Deprecated all-tools schema alias delegates to get_tools_schema."""

        @tool
        def alias_tool(agent, x: int) -> int:
            """Alias tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        manager = ToolManager(tools=[alias_tool])
        with pytest.warns(DeprecationWarning, match="get_all_tools_schema"):
            schemas = manager.get_all_tools_schema()

        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "alias_tool"

    def test_deprecated_schema_aliases_none_inherits_configured_tools(self):
        """Deprecated None selectors keep old all-configured semantics."""

        @tool
        def compatibility_tool(agent, x: int) -> int:
            """Compatibility tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        manager = ToolManager(tools=[compatibility_tool])

        assert manager.get_tools_schema(tools=None) == []

        with pytest.warns(DeprecationWarning, match="get_all_tools_schema"):
            positional_none_schemas = manager.get_all_tools_schema(None)

        with pytest.warns(DeprecationWarning, match="get_all_tools_schema"):
            selected_none_schemas = manager.get_all_tools_schema(selected_tools=None)

        with pytest.warns(DeprecationWarning, match="selected_tools"):
            selected_alias_schemas = manager.get_tools_schema(selected_tools=None)

        for schemas in (
            positional_none_schemas,
            selected_none_schemas,
            selected_alias_schemas,
        ):
            assert len(schemas) == 1
            assert schemas[0]["function"]["name"] == "compatibility_tool"

    def test_get_tools_schema_with_selected_tools(self):
        """Test getting schemas for selected tools only."""

        @tool
        def tool_a(agent, x: int) -> int:
            """Tool A.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def tool_b(agent, y: str) -> str:
            """Tool B.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        @tool
        def tool_c(agent, z: float) -> float:
            """Tool C.
            Args:
                agent: The agent making the request (provided automatically)
                z: Input.
            Returns:
                Output.
            """
            return z

        manager = ToolManager(tools=[tool_a, tool_b, tool_c])

        # Test selecting specific tools
        selected_tools = ["tool_a", "tool_c"]
        schemas = manager.get_tools_schema(tools=selected_tools)

        assert len(schemas) == 2
        tool_names = [schema["function"]["name"] for schema in schemas]
        assert "tool_a" in tool_names
        assert "tool_c" in tool_names
        assert "tool_b" not in tool_names

    def test_get_tools_schema_empty_list(self):
        """Test that empty list returns no tools (selected_tools=[] means 'no tools')."""

        @tool
        def test_tool(agent, x: int) -> int:
            """Test tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        manager = ToolManager()

        # Empty list should return no tools — the user explicitly asked for none
        empty_list_schemas = manager.get_tools_schema(tools=[])

        assert len(empty_list_schemas) == 0

    def test_get_tools_schema_none(self):
        """Test that explicit None returns no tools."""

        @tool
        def test_tool(agent, x: int) -> int:
            """Test tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        manager = ToolManager(tools=[test_tool])

        all_schemas = manager.get_tools_schema()
        none_schemas = manager.get_tools_schema(tools=None)

        assert len(all_schemas) == 1
        assert none_schemas == []

    def test_get_tools_schema_nonexistent_tools(self):
        """Test that requesting nonexistent tools raises appropriate errors."""

        @tool
        def existing_tool(agent, x: int) -> int:
            """Existing tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        manager = ToolManager()

        # Test with nonexistent tools
        selected_tools = ["existing_tool", "nonexistent_tool"]

        with pytest.raises(ValueError, match="Unknown tool name"):
            manager.get_tools_schema(tools=selected_tools)

    def test_get_tools_schema_single_tool_sequence(self):
        """Test selecting a single tool in a sequence."""

        @tool
        def single_tool(agent, x: int) -> int:
            """Single tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def other_tool(agent, y: str) -> str:
            """Other tool.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        manager = ToolManager(tools=[single_tool])

        schemas = manager.get_tools_schema(tools=["single_tool"])

        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "single_tool"

    def test_get_tools_schema_single_string_selector(self):
        """A single configured tool name narrows the schema selection."""

        @tool
        def single_name_tool(agent, x: int) -> int:
            """Single name tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def other_name_tool(agent, y: str) -> str:
            """Other name tool.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        manager = ToolManager(tools=[single_name_tool, other_name_tool])

        schemas = manager.get_tools_schema(tools="single_name_tool")

        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "single_name_tool"

    def test_get_tools_schema_single_callable_selector(self):
        """A single configured callable narrows the schema selection."""

        @tool
        def single_callable_tool(agent, x: int) -> int:
            """Single callable tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def other_callable_tool(agent, y: str) -> str:
            """Other callable tool.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        manager = ToolManager(tools=[single_callable_tool, other_callable_tool])

        schemas = manager.get_tools_schema(tools=single_callable_tool)

        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "single_callable_tool"

    def test_get_tools_schema_rejects_unconfigured_callable(self):
        """Per-call callable selectors cannot add unconfigured tools."""

        @tool
        def configured_tool(agent, x: int) -> int:
            """Configured tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def unconfigured_tool(agent, y: int) -> int:
            """Unconfigured tool.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        manager = ToolManager(tools=[configured_tool])

        with pytest.raises(ValueError, match="Unknown tool name"):
            manager.get_tools_schema(tools=[unconfigured_tool])

        with pytest.raises(ValueError, match="Unknown tool name"):
            manager.get_tools_schema(tools=unconfigured_tool)

    def test_get_tools_schema_rejects_unconfigured_registered_name(self):
        """Per-call string selectors only narrow configured tools."""

        @tool
        def configured_name_tool(agent, x: int) -> int:
            """Configured name tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def unconfigured_name_tool(agent, y: int) -> int:
            """Unconfigured name tool.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        manager = ToolManager(tools=[configured_name_tool])

        with pytest.raises(ValueError, match="Unknown tool name"):
            manager.get_tools_schema(tools=["unconfigured_name_tool"])

        with pytest.raises(ValueError, match="Unknown tool name"):
            manager.get_tools_schema(tools="unconfigured_name_tool")

    def test_call_tool_success(self):
        """Test successfully calling a tool."""
        manager = ToolManager()

        def callable_tool(agent, message: str) -> str:
            """Callable tool.
            Args:
                agent: The agent making the request (provided automatically)
                message: Input message.
            Returns:
                The message with prefix.
            """
            return f"Result: {message}"

        manager.register(callable_tool)
        result = manager.call("callable_tool", {"agent": Mock(), "message": "test"})
        assert result == "Result: test"

    def test_call_tool_not_found(self):
        """Test calling a non-existent tool."""
        manager = ToolManager()

        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            manager.call("nonexistent", {})

    def test_has_tool(self):
        """Test checking if a tool exists."""
        manager = ToolManager()

        def existing_tool():
            return "test"

        manager.register(existing_tool)

        assert manager.has_tool("existing_tool") is True
        assert manager.has_tool("nonexistent_tool") is False

    def test_call_tools_no_tool_calls(self):
        """Test call_tools with response that has no tool_calls."""
        manager = ToolManager()
        mock_agent = Mock()

        mock_response = Mock()
        mock_response.tool_calls = None

        result = manager.call_tools(mock_agent, mock_response)
        assert result == []

    def test_call_tools_empty_tool_calls(self):
        """Test call_tools with empty tool_calls list."""
        manager = ToolManager()
        mock_agent = Mock()

        mock_response = Mock()
        mock_response.tool_calls = []

        result = manager.call_tools(mock_agent, mock_response)
        assert result == []

    def test_call_tools_success(self):
        """Test successful tool calling."""

        @tool
        def test_tool(agent, param1: str) -> str:
            """Test tool for call_tools.
            Args:
                agent: The agent making the request (provided automatically)
                param1: Test parameter.
            Returns:
                Processed parameter.
            """
            return f"Processed: {param1}"

        manager = ToolManager(tools=[test_tool])

        # Mock agent
        mock_agent = Mock()

        # Mock LLM response with tool calls
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"param1": "test_value"}'

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["role"] == "tool"
        assert result[0]["name"] == "test_tool"
        assert "Processed: test_value" in result[0]["response"]

    def test_call_tools_single_selectors(self):
        """Execution accepts a single configured callable or name selector."""

        @tool
        def configured_execution_tool(agent, value: int) -> int:
            """Configured execution tool.
            Args:
                agent: The agent making the request (provided automatically)
                value: Input.
            Returns:
                Output.
            """
            return value + 1

        manager = ToolManager(tools=[configured_execution_tool])
        mock_agent = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_single_selector"
        mock_tool_call.function.name = "configured_execution_tool"
        mock_tool_call.function.arguments = '{"value": 3}'

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        name_result = manager.call_tools(
            mock_agent,
            mock_response,
            tools="configured_execution_tool",
        )
        callable_result = manager.call_tools(
            mock_agent,
            mock_response,
            tools=configured_execution_tool,
        )

        assert name_result[0]["response"] == "4"
        assert callable_result[0]["response"] == "4"

    def test_call_tools_rejects_unconfigured_per_call_callable(self):
        """Execution selectors cannot add tools outside the configured set."""

        @tool
        def configured_execution_tool(agent, value: int) -> int:
            """Configured execution tool.
            Args:
                agent: The agent making the request (provided automatically)
                value: Input.
            Returns:
                Output.
            """
            return value

        @tool
        def unconfigured_execution_tool(agent, value: int) -> int:
            """Unconfigured execution tool.
            Args:
                agent: The agent making the request (provided automatically)
                value: Input.
            Returns:
                Output.
            """
            return value

        manager = ToolManager(tools=[configured_execution_tool])
        mock_agent = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_unconfigured"
        mock_tool_call.function.name = "unconfigured_execution_tool"
        mock_tool_call.function.arguments = '{"value": 3}'

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        with pytest.raises(ValueError, match="Unknown tool name"):
            manager.call_tools(
                mock_agent,
                mock_response,
                tools=[unconfigured_execution_tool],
            )

        with pytest.raises(ValueError, match="Unknown tool name"):
            manager.call_tools(
                mock_agent,
                mock_response,
                tools=unconfigured_execution_tool,
            )

        with pytest.raises(ValueError, match="Unknown tool name"):
            manager.call_tools(
                mock_agent,
                mock_response,
                tools="unconfigured_execution_tool",
            )

    def test_call_tools_function_not_found(self):
        """Test call_tools with non-existent function."""
        manager = ToolManager()
        mock_agent = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "nonexistent_function"
        mock_tool_call.function.arguments = '{"param": "value"}'

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert "Error:" in result[0]["response"]

    def test_call_tools_invalid_json(self):
        """Test call_tools with invalid JSON arguments."""

        @tool
        def test_tool(agent, param1: str) -> str:
            """Test tool.
            Args:
                agent: The agent making the request (provided automatically)
                param1: Test parameter.
            Returns:
                Processed parameter.
            """
            return f"Processed: {param1}"

        manager = ToolManager(tools=[test_tool])
        mock_agent = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"param1": invalid_json}'

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert "Error:" in result[0]["response"]

    def test_call_tools_successful_argument_filtering(self):
        """Test call_tools with argument filtering when function signature doesn't match."""
        manager = ToolManager()

        def simple_tool(required_param: str) -> str:
            """Simple tool that only takes required_param.
            Args:
                required_param: The only parameter this function accepts.
            Returns:
                Processed parameter.
            """
            return f"Simple: {required_param}"

        # Register tool manually without using decorator to test filtering
        manager.register(simple_tool)

        mock_agent = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "simple_tool"
        # Include extra parameters that the function doesn't accept
        mock_tool_call.function.arguments = (
            '{"required_param": "test", "extra_param": "ignored"}'
        )

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert "Simple: test" in result[0]["response"]

    def test_call_tools_type_coercion_float(self):
        """Test coercion of float arguments passed as JSON strings."""

        @tool
        def float_tool(agent, amount: float) -> str:
            """Float tool.
            Args:
                agent: The agent making the request
                amount: Amount to format.
            Returns:
                Formatted amount.
            """
            return f"{amount:.2f}"

        manager = ToolManager(tools=[float_tool])
        mock_agent = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_float"
        mock_tool_call.function.name = "float_tool"
        mock_tool_call.function.arguments = '{"amount": "35.0"}'

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_float"
        assert result[0]["response"] == "35.00"

    def test_call_tools_type_coercion_float_with_string_annotation(self):
        """Test coercion when annotations are stored as strings."""

        @tool
        def float_tool(agent, amount: "float") -> "str":
            """Float tool.
            Args:
                agent: The agent making the request
                amount: Amount to format.
            Returns:
                Formatted amount.
            """
            return f"{amount:.2f}"

        manager = ToolManager(tools=[float_tool])
        mock_agent = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_float_string_annotation"
        mock_tool_call.function.name = "float_tool"
        mock_tool_call.function.arguments = '{"amount": "35.0"}'

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_float_string_annotation"
        assert result[0]["response"] == "35.00"

    def test_call_tools_type_coercion_int(self):
        """Test coercion of int arguments passed as JSON strings."""

        @tool
        def int_tool(agent, count: int) -> int:
            """Int tool.
            Args:
                agent: The agent making the request
                count: Count to increment.
            Returns:
                Incremented count.
            """
            return count + 1

        manager = ToolManager(tools=[int_tool])
        mock_agent = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_int"
        mock_tool_call.function.name = "int_tool"
        mock_tool_call.function.arguments = '{"count": "5"}'

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_int"
        assert result[0]["response"] == "6"

    def test_call_tools_no_response(self):
        """Test call_tools when tool returns None."""

        @tool
        def silent_tool(agent) -> None:
            """Tool that returns None.
            Args:
                agent: The agent making the request (provided automatically)
            Returns:
                None
            """
            return None

        manager = ToolManager(tools=[silent_tool])
        mock_agent = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "silent_tool"
        mock_tool_call.function.arguments = "{}"

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert "silent_tool executed successfully" in result[0]["response"]

    def test_call_tools_general_exception(self):
        """Test call_tools handling of general exceptions."""
        manager = ToolManager()
        mock_agent = Mock()

        # Create a mock response that will cause an AttributeError
        mock_response = Mock()
        # Remove the tool_calls attribute to cause AttributeError
        del mock_response.tool_calls

        result = manager.call_tools(mock_agent, mock_response)
        assert result == []

    def test_selected_tools_consistency(self):
        """Test that selected_tools parameter works consistently."""

        @tool
        def consistency_tool_a(agent, x: int) -> int:
            """Consistency tool A.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def consistency_tool_b(agent, y: str) -> str:
            """Consistency tool B.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        @tool
        def consistency_tool_c(agent, z: float) -> float:
            """Consistency tool C.
            Args:
                agent: The agent making the request (provided automatically)
                z: Input.
            Returns:
                Output.
            """
            return z

        manager = ToolManager(
            tools=[consistency_tool_a, consistency_tool_b, consistency_tool_c]
        )

        # Test that same selected_tools always returns same schemas
        selected_tools = ["consistency_tool_a", "consistency_tool_c"]

        schemas1 = manager.get_tools_schema(tools=selected_tools)
        schemas2 = manager.get_tools_schema(tools=selected_tools)

        names1 = sorted([schema["function"]["name"] for schema in schemas1])
        names2 = sorted([schema["function"]["name"] for schema in schemas2])

        assert names1 == names2
        assert len(schemas1) == len(schemas2) == 2

        # Test order independence
        reversed_tools = list(reversed(selected_tools))
        schemas3 = manager.get_tools_schema(tools=reversed_tools)
        names3 = sorted([schema["function"]["name"] for schema in schemas3])

        assert names1 == names3

    def test_selected_tools_duplicate_handling(self):
        """Test how selected_tools handles duplicates."""

        @tool
        def duplicate_test_tool(agent, x: int) -> int:
            """Duplicate test tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        manager = ToolManager(tools=[duplicate_test_tool])

        # Test with duplicate tool names
        selected_tools = ["duplicate_test_tool", "duplicate_test_tool"]
        schemas = manager.get_tools_schema(tools=selected_tools)

        # Should return schemas for each request (may include duplicates)
        assert len(schemas) == 2

    def test_multiple_managers_selected_tools(self):
        """Test selected_tools functionality with multiple ToolManager instances."""

        @tool
        def shared_tool_1(agent, x: int) -> int:
            """Shared tool 1.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def shared_tool_2(agent, y: str) -> str:
            """Shared tool 2.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        manager1 = ToolManager(tools=[shared_tool_1])
        manager2 = ToolManager(tools=[shared_tool_1, shared_tool_2])

        # Bare managers no longer copy global registrations; explicit managers
        # expose only their configured maximum capability sets.
        all_schemas_1 = manager1.get_tools_schema()
        all_schemas_2 = manager2.get_tools_schema()

        assert len(all_schemas_1) == 1
        assert len(all_schemas_2) == 2

        # Selected tools should work the same on both managers
        selected_tools = ["shared_tool_1"]
        schemas_1 = manager1.get_tools_schema(tools=selected_tools)
        schemas_2 = manager2.get_tools_schema(tools=selected_tools)

        assert len(schemas_1) == len(schemas_2) == 1
        assert schemas_1[0]["function"]["name"] == schemas_2[0]["function"]["name"]

    @pytest.mark.asyncio
    async def test_acall_tools_success(self, monkeypatch):
        """
        This test validates the full async tool execution pipeline by ensuring that:

        - Tool calls are correctly extracted from the LLM response object.
        - JSON-formatted arguments are parsed without error.
        - The `agent` parameter is automatically injected when required by the tool's function signature.
        - The tool function is executed successfully.
        - The result is wrapped in the expected structured response format:
            {
                "tool_call_id": <tool_call_id>,
                "role": "tool",
                "name": <tool_name>,
                "response": <stringified_result>
            }
        - The asynchronous execution path using `asyncio.gather`
        returns the correct results.
        """
        mock_agent = Mock()

        @tool
        def async_test_tool(agent, value: str) -> str:
            """Async test tool.

            Args:
                agent: The agent making the request (provided automatically)
                value: Input value.

            Returns:
                Processed value.
            """
            return f"Async: {value}"

        manager = ToolManager(tools=[async_test_tool])

        mock_tool_call = Mock()
        mock_tool_call.id = "call_async"
        mock_tool_call.function.name = "async_test_tool"
        mock_tool_call.function.arguments = '{"value": "hello"}'

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = await manager.acall_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_async"
        assert "Async: hello" in result[0]["response"]
