import math
from enum import Enum
from typing import ClassVar, Final, Literal

import pytest

from mesa_llm.tools.tool_decorator import (
    _GLOBAL_TOOL_REGISTRY,
    DocstringParsingError,
    _parse_docstring,
    _python_to_json_type,
    tool,
)


class _ExampleLiteralEnum(Enum):
    VALUE = "value"


class TestToolDecoractor:
    def test_parse_docstring(self):
        def sample_func(agent, count: int, name: str) -> str:
            """Short summary.

            Args:
                agent: The agent making the request (provided automatically)
                count: Number of items.
                name: The name to process.

            Returns:
                A processed string.
            """

            return f"{name}:{count}"

        summary, param_desc, return_desc = _parse_docstring(sample_func)

        assert summary == "Short summary."
        assert set(param_desc.keys()) == {"agent", "count", "name"}
        assert param_desc["count"].startswith("Number of items")
        assert param_desc["name"].startswith("The name to process")
        assert return_desc is not None and "processed" in return_desc

        # Error case: missing Args for parameters in signature
        def bad_func(a, b):
            """No args section."""

            return a, b

        with pytest.raises(DocstringParsingError):
            _parse_docstring(bad_func)

    def test_python_to_json_type(self):
        # Basic types
        assert _python_to_json_type(int) == {"type": "integer"}
        assert _python_to_json_type(str) == {"type": "string"}
        assert _python_to_json_type(float) == {"type": "number"}
        assert _python_to_json_type(bool) == {"type": "boolean"}

        # Collections and generics
        assert _python_to_json_type(list[int]) == {
            "type": "array",
            "items": {"type": "integer"},
        }

        # Tuple with mixed types yields ordered anyOf
        assert _python_to_json_type(tuple[str, int]) == {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"},
                ]
            },
        }

        # Optional type (int | None) includes null
        assert _python_to_json_type(int | None) == {
            "type": ["integer", "null"],
        }

        # Non-null unions preserve declaration order
        assert _python_to_json_type(int | str) == {
            "anyOf": [
                {"type": "integer"},
                {"type": "string"},
            ]
        }

        # Dict with value types
        dict_schema = _python_to_json_type(dict[str, int])
        assert dict_schema["type"] == "object"
        assert dict_schema["additionalProperties"] == {"type": "integer"}

        # Set maps to array
        set_schema = _python_to_json_type(set[str])
        assert set_schema == {"type": "array", "items": {"type": "string"}}

    @pytest.mark.parametrize(
        ("annotation", "item_schema", "length"),
        [
            pytest.param(
                tuple[int, int],
                {"type": "integer"},
                2,
                id="two-integers",
            ),
            pytest.param(
                tuple[str, str, str],
                {"type": "string"},
                3,
                id="three-strings",
            ),
            pytest.param(
                tuple[int | float, int | float],
                {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "number"},
                    ]
                },
                2,
                id="two-numeric-unions",
            ),
        ],
    )
    def test_python_to_json_type_homogeneous_fixed_tuple_has_exact_length(
        self,
        annotation,
        item_schema,
        length,
    ):
        assert _python_to_json_type(annotation) == {
            "type": "array",
            "items": item_schema,
            "minItems": length,
            "maxItems": length,
        }

    @pytest.mark.parametrize(
        ("annotation", "item_schema"),
        [
            (tuple[int, ...], {"type": "integer"}),
            (tuple[str, ...], {"type": "string"}),
            (
                tuple[list[int], ...],
                {"type": "array", "items": {"type": "integer"}},
            ),
        ],
        ids=["integers", "strings", "nested-lists"],
    )
    def test_python_to_json_type_variadic_tuple_uses_homogeneous_items(
        self,
        annotation,
        item_schema,
    ):
        assert _python_to_json_type(annotation) == {
            "type": "array",
            "items": item_schema,
        }

    def test_python_to_json_type_literal_and_nullable_union_schemas(self):
        assert _python_to_json_type(Literal["a", "b"]) == {
            "type": "string",
            "enum": ["a", "b"],
        }
        assert _python_to_json_type(Literal["b", "a"]) == {
            "type": "string",
            "enum": ["b", "a"],
        }
        assert _python_to_json_type(int | str | None) == {
            "anyOf": [
                {"type": "integer"},
                {"type": "string"},
                {"type": "null"},
            ]
        }
        assert _python_to_json_type(Literal["a", "b"] | None) == {
            "anyOf": [
                {
                    "type": "string",
                    "enum": ["a", "b"],
                },
                {"type": "null"},
            ]
        }

    def test_python_to_json_type_homogeneous_literal_schemas(self):
        assert _python_to_json_type(Literal[1, 2]) == {
            "type": "integer",
            "enum": [1, 2],
        }
        assert _python_to_json_type(Literal[True, False]) == {
            "type": "boolean",
            "enum": [True, False],
        }
        assert _python_to_json_type(Literal[1.5, 2.5]) == {
            "type": "number",
            "enum": [1.5, 2.5],
        }
        assert _python_to_json_type(Literal[1.0]) == {
            "type": "number",
            "enum": [1.0],
        }

    @pytest.mark.parametrize(
        "literal_type",
        [Literal[1, 1.0], Literal[1.0, 1]],
        ids=["integer-first", "float-first"],
    )
    def test_python_to_json_type_rejects_ambiguous_numeric_literals(
        self,
        literal_type,
    ):
        with pytest.raises(
            TypeError,
            match=r"JSON-equivalent|ambiguous|indistinguishable",
        ):
            _python_to_json_type(literal_type)

    @pytest.mark.parametrize(
        "literal_type",
        [
            Literal[1] | Literal[1.0],
            list[Literal[1] | Literal[1.0]],
        ],
        ids=["split-union", "nested-split-union"],
    )
    def test_python_to_json_type_rejects_split_ambiguous_numeric_literals(
        self,
        literal_type,
    ):
        with pytest.raises(
            TypeError,
            match=r"JSON-equivalent|ambiguous|indistinguishable",
        ):
            _python_to_json_type(literal_type)

    @pytest.mark.parametrize(
        "literal_type",
        [
            list[Literal[1]] | set[Literal[1.0]],
            dict[str, list[Literal[1]]] | dict[str, set[Literal[1.0]]],
        ],
        ids=["collection-branches", "nested-dictionary-branches"],
    )
    def test_python_to_json_type_rejects_structurally_ambiguous_numeric_literals(
        self,
        literal_type,
    ):
        with pytest.raises(
            TypeError,
            match=r"JSON-equivalent|ambiguous|indistinguishable",
        ):
            _python_to_json_type(literal_type)

    @pytest.mark.parametrize(
        "annotation",
        [
            type[Literal[1] | Literal[1.0]],
            ClassVar[Literal[1] | Literal[1.0]],
            Final[Literal[1] | Literal[1.0]],
        ],
        ids=["type", "class-var", "final"],
    )
    def test_python_to_json_type_preserves_unsupported_generic_fallback(
        self,
        annotation,
    ):
        assert _python_to_json_type(annotation) == {"type": "object"}

    def test_python_to_json_type_heterogeneous_json_literal_omits_type(self):
        assert _python_to_json_type(Literal["a", 2, False, None, 1.5]) == {
            "enum": ["a", 2, False, None, 1.5],
        }

    @pytest.mark.parametrize(
        "literal_value",
        [
            b"bytes",
            _ExampleLiteralEnum.VALUE,
            1 + 2j,
            math.nan,
            math.inf,
            -math.inf,
        ],
        ids=[
            "bytes",
            "enum-member",
            "complex",
            "nan",
            "positive-infinity",
            "negative-infinity",
        ],
    )
    def test_python_to_json_type_rejects_non_json_literal_values(
        self,
        literal_value,
    ):
        with pytest.raises(
            TypeError,
            match="Literal schemas support only finite JSON scalar values",
        ):
            _python_to_json_type(Literal[literal_value])

    def test_tool_schema_exposes_literal_and_nullable_union(self):
        @tool
        def select_value(
            agent,
            mode: Literal["a", "b"],
            optional_mode: Literal["a", "b"] | None,
            value: int | str | None,
        ) -> str:
            """Select a typed value.

            Args:
                agent: The agent making the request.
                mode: Selection mode.
                optional_mode: Optional selection mode.
                value: Optional selected value.

            Returns:
                Selection confirmation.
            """
            del agent, optional_mode, value
            return mode

        try:
            properties = select_value.__tool_schema__["function"]["parameters"][
                "properties"
            ]

            assert properties["mode"] == {
                "type": "string",
                "enum": ["a", "b"],
                "description": "Selection mode.",
            }
            assert properties["optional_mode"] == {
                "anyOf": [
                    {
                        "type": "string",
                        "enum": ["a", "b"],
                    },
                    {"type": "null"},
                ],
                "description": "Optional selection mode.",
            }
            assert properties["value"] == {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                    {"type": "null"},
                ],
                "description": "Optional selected value.",
            }
        finally:
            _GLOBAL_TOOL_REGISTRY.pop("select_value", None)

    def test_tool_schema_rejects_ambiguous_numeric_literal_without_registration(self):
        _GLOBAL_TOOL_REGISTRY.pop("select_numeric_value", None)

        with pytest.raises(
            TypeError,
            match=r"JSON-equivalent|ambiguous|indistinguishable",
        ):

            @tool
            def select_numeric_value(agent, value: Literal[1, 1.0]) -> int | float:
                """Select a numeric value.

                Args:
                    agent: The agent making the request.
                    value: Numeric value to select.

                Returns:
                    The selected value.
                """
                del agent
                return value

        assert "select_numeric_value" not in _GLOBAL_TOOL_REGISTRY

    @pytest.mark.parametrize(
        "annotation",
        [
            Literal[1] | Literal[1.0],
            list[Literal[1] | Literal[1.0]],
        ],
        ids=["split-union", "nested-split-union"],
    )
    def test_tool_schema_rejects_split_ambiguous_numeric_literal_without_registration(
        self,
        annotation,
    ):
        _GLOBAL_TOOL_REGISTRY.pop("select_numeric_value", None)

        def select_numeric_value(agent, value) -> int | float:
            """Select a numeric value.

            Args:
                agent: The agent making the request.
                value: Numeric value to select.

            Returns:
                The selected value.
            """
            del agent
            return value

        select_numeric_value.__annotations__["value"] = annotation

        with pytest.raises(
            TypeError,
            match=r"JSON-equivalent|ambiguous|indistinguishable",
        ):
            tool(select_numeric_value)

        assert "select_numeric_value" not in _GLOBAL_TOOL_REGISTRY

    @pytest.mark.parametrize(
        "annotation",
        [
            list[Literal[1]] | set[Literal[1.0]],
            dict[str, list[Literal[1]]] | dict[str, set[Literal[1.0]]],
        ],
        ids=["collection-branches", "nested-dictionary-branches"],
    )
    def test_tool_schema_rejects_structurally_ambiguous_literal_without_side_effects(
        self,
        annotation,
    ):
        _GLOBAL_TOOL_REGISTRY.pop("select_numeric_value", None)
        mutations = []

        def select_numeric_value(agent, value) -> int | float:
            """Select a numeric value.

            Args:
                agent: The agent making the request.
                value: Numeric value to select.

            Returns:
                The selected value.
            """
            mutations.append((agent, value))
            return value

        select_numeric_value.__annotations__["value"] = annotation

        with pytest.raises(
            TypeError,
            match=r"JSON-equivalent|ambiguous|indistinguishable",
        ):
            tool(select_numeric_value)

        assert "select_numeric_value" not in _GLOBAL_TOOL_REGISTRY
        assert mutations == []

    @pytest.mark.parametrize(
        "annotation",
        [
            type[Literal[1] | Literal[1.0]],
            ClassVar[Literal[1] | Literal[1.0]],
            Final[Literal[1] | Literal[1.0]],
        ],
        ids=["type", "class-var", "final"],
    )
    def test_tool_schema_preserves_unsupported_generic_object_fallback(
        self,
        annotation,
    ):
        _GLOBAL_TOOL_REGISTRY.pop("inspect_legacy_value", None)

        def inspect_legacy_value(agent, value) -> str:
            """Inspect a legacy generic value.

            Args:
                agent: The agent making the request.
                value: Value to inspect.

            Returns:
                Inspection confirmation.
            """
            del agent, value
            return "inspected"

        inspect_legacy_value.__annotations__["value"] = annotation
        decorated_tool = tool(inspect_legacy_value)

        try:
            properties = decorated_tool.__tool_schema__["function"]["parameters"][
                "properties"
            ]
            assert properties["value"] == {
                "type": "object",
                "description": "Value to inspect.",
            }
        finally:
            _GLOBAL_TOOL_REGISTRY.pop("inspect_legacy_value", None)

    def test_tool(self):
        _GLOBAL_TOOL_REGISTRY.clear()

        @tool
        def greet(agent, name: str, times: int) -> str:
            """Greet someone.

            Args:
                agent: The agent making the request (provided automatically)
                name: Person name.
                times: Number of repetitions.

            Returns:
                Concatenated greeting.
            """

            return " ".join([f"Hi {name}!" for _ in range(times)])

        # Registered globally
        assert "greet" in _GLOBAL_TOOL_REGISTRY

        schema = greet.__tool_schema__
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "greet"
        assert "Greet someone." in fn["description"]
        assert "returns: Concatenated greeting." in fn["description"]

        params = fn["parameters"]
        assert params["type"] == "object"

        # 'agent' should be ignored in the schema (not required, not a property)
        assert set(params["required"]) == {"name", "times"}
        assert set(params["properties"].keys()) == {"name", "times"}

        # Types and descriptions propagated
        assert params["properties"]["name"] == {
            "type": "string",
            "description": "Person name.",
        }
        assert params["properties"]["times"]["type"] == "integer"
        assert params["properties"]["times"]["description"].startswith(
            "Number of repetitions"
        )

        _GLOBAL_TOOL_REGISTRY.clear()

    def test_parse_docstring_ignore_agent_true(self):
        def tool_func(agent, x: int) -> str:
            """Short summary.

            Args:
                x: An integer input.

            Returns:
                A string.
            """
            return str(x)

        summary, param_desc, _ = _parse_docstring(tool_func, ignore_agent=True)
        assert summary == "Short summary."
        assert "x" in param_desc
        assert "agent" not in param_desc

    def test_parse_docstring_ignore_agent_false(self):
        def tool_func(agent, x: int) -> str:
            """Short summary.

            Args:
                x: An integer input.

            Returns:
                A string.
            """
            return str(x)

        with pytest.raises(DocstringParsingError):
            _parse_docstring(tool_func, ignore_agent=False)

    def test_tool_agent_docstring_not_required(self):
        _GLOBAL_TOOL_REGISTRY.clear()

        @tool
        def move(agent, direction: str) -> str:
            """Move in a direction.

            Args:
                direction: The direction to move.

            Returns:
                A result string.
            """
            return direction

        assert "move" in _GLOBAL_TOOL_REGISTRY
        schema = move.__tool_schema__
        params = schema["function"]["parameters"]
        assert "agent" not in params["properties"]
        assert "agent" not in params["required"]

        _GLOBAL_TOOL_REGISTRY.clear()
