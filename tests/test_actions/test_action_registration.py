from __future__ import annotations

import gc
import inspect
import weakref
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pytest

import mesa_llm.actions.action_manager as action_manager_module
from mesa_llm.actions import (
    ActionChoice,
    ActionManager,
    ActionMetadata,
    action,
    default_actions,
    wait,
)
from mesa_llm.actions.action_decorator import (
    _GLOBAL_ACTION_REGISTRY,
    ActionAnnotationContractError,
    ActionSignatureError,
)

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class _UnsupportedActionPayload:
    pass


@pytest.fixture(autouse=True)
def restore_global_action_registry():
    """Keep bare @action registrations local to each test."""
    original_registry = dict(_GLOBAL_ACTION_REGISTRY)
    yield
    _GLOBAL_ACTION_REGISTRY.clear()
    _GLOBAL_ACTION_REGISTRY.update(original_registry)


def _make_named_managed_action(
    name: str,
    marker: str,
    execution_calls: list[str],
):
    def generated_action(agent) -> str:
        """Return a deterministic marker.

        Returns:
            The configured marker.
        """
        del agent
        execution_calls.append(marker)
        return marker

    generated_action.__name__ = name
    return action(action_manager=ActionManager())(generated_action)


_DECLARED_GENERATOR_KINDS = ("generator", "async-generator")


def _make_declared_generator_action(
    kind: str,
    name: str,
    body_calls: list[object],
):
    if kind == "generator":

        def generated_action(agent):
            """Yield one action result.

            Returns:
                One yielded result.
            """
            body_calls.append(agent)
            yield "generated"

    else:

        async def generated_action(agent):
            """Yield one asynchronous action result.

            Returns:
                One yielded result.
            """
            body_calls.append(agent)
            yield "generated"

    generated_action.__name__ = name
    return generated_action


def _assert_declared_generator_error(exc_info, action_name: str):
    assert type(exc_info.value) is ActionSignatureError
    assert str(exc_info.value) == (
        f"Action {action_name!r} must return one completed result; generator and "
        "async-generator functions are not supported as actions."
    )


_DUPLICATE_AGENT_PARAMETER_CASES = (
    pytest.param(
        "mixed-kinds",
        ("agent", "Agent"),
        id="positional-agent-and-keyword-only-Agent",
    ),
    pytest.param(
        "capitalization-variants",
        ("AgEnT", "AGENT", "aGeNt"),
        id="three-capitalization-variants",
    ),
)


def _make_duplicate_agent_action(
    case: str,
    name: str,
    body_calls: list[tuple[object, ...]],
):
    if case == "mixed-kinds":

        def duplicate_agent_action(agent, *, Agent) -> str:  # noqa: N803
            """Return a completion marker.

            Returns:
                The completion marker.
            """
            body_calls.append((agent, Agent))
            return "completed"

    else:

        def duplicate_agent_action(
            AgEnT,  # noqa: N803
            AGENT,  # noqa: N803
            *,
            aGeNt,  # noqa: N803
        ) -> str:
            """Return a completion marker.

            Returns:
                The completion marker.
            """
            body_calls.append((AgEnT, AGENT, aGeNt))
            return "completed"

    duplicate_agent_action.__name__ = name
    return duplicate_agent_action


def _assert_duplicate_agent_error(
    exc_info,
    action_name: str,
    parameter_names: tuple[str, ...],
):
    assert type(exc_info.value) is ActionSignatureError
    message = str(exc_info.value)
    assert action_name in message
    assert all(repr(name) in message for name in parameter_names)
    assert any(
        phrase in message.casefold()
        for phrase in ("multiple", "at most one", "more than one", "duplicate")
    )


@pytest.mark.parametrize("kind", _DECLARED_GENERATOR_KINDS)
def test_action_rejects_declared_generator_before_metadata_schema_or_global_registration(
    kind,
):
    body_calls = []
    action_name = f"direct_{kind.replace('-', '_')}_action"
    action_fn = _make_declared_generator_action(kind, action_name, body_calls)
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)

    assert inspect.isgeneratorfunction(action_fn) is (kind == "generator")
    assert inspect.isasyncgenfunction(action_fn) is (kind == "async-generator")

    with pytest.raises(ActionSignatureError) as exc_info:
        action(action_fn)

    _assert_declared_generator_error(exc_info, action_name)
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert body_calls == []


@pytest.mark.parametrize("kind", _DECLARED_GENERATOR_KINDS)
def test_action_manager_decorator_rejects_declared_generator_atomically(kind):
    body_calls = []
    manager = ActionManager()
    retained_action = _make_named_managed_action(
        f"retained_decorator_{kind.replace('-', '_')}_action",
        "retained",
        body_calls,
    )
    manager.register(retained_action)
    manager_before = dict(manager.actions)
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)
    action_name = f"managed_{kind.replace('-', '_')}_action"
    action_fn = _make_declared_generator_action(kind, action_name, body_calls)

    with pytest.raises(ActionSignatureError) as exc_info:
        action(action_manager=manager)(action_fn)

    _assert_declared_generator_error(exc_info, action_name)
    assert manager.actions == manager_before
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert body_calls == []


@pytest.mark.parametrize("kind", _DECLARED_GENERATOR_KINDS)
def test_action_manager_constructor_rejects_declared_generator_atomically(kind):
    body_calls = []
    suffix = kind.replace("-", "_")
    early_action = _make_named_managed_action(
        f"early_constructor_{suffix}_action",
        "early",
        body_calls,
    )
    action_name = f"constructor_{suffix}_action"
    action_fn = _make_declared_generator_action(kind, action_name, body_calls)
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)
    failed_manager = ActionManager.__new__(ActionManager)

    with pytest.raises(ActionSignatureError) as exc_info:
        failed_manager.__init__(actions=[early_action, action_fn])

    _assert_declared_generator_error(exc_info, action_name)
    assert failed_manager.actions == {}
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert body_calls == []


@pytest.mark.parametrize("kind", _DECLARED_GENERATOR_KINDS)
def test_action_manager_register_rejects_declared_generator_atomically(kind):
    body_calls = []
    suffix = kind.replace("-", "_")
    retained_action = _make_named_managed_action(
        f"retained_register_{suffix}_action",
        "retained",
        body_calls,
    )
    manager = ActionManager(actions=[retained_action])
    manager_before = dict(manager.actions)
    action_name = f"register_{suffix}_action"
    action_fn = _make_declared_generator_action(kind, action_name, body_calls)
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)

    with pytest.raises(ActionSignatureError) as exc_info:
        manager.register(action_fn)

    _assert_declared_generator_error(exc_info, action_name)
    assert manager.actions == manager_before
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert body_calls == []


@pytest.mark.parametrize("kind", _DECLARED_GENERATOR_KINDS)
def test_action_manager_register_many_rejects_declared_generator_atomically(kind):
    body_calls = []
    suffix = kind.replace("-", "_")
    retained_action = _make_named_managed_action(
        f"retained_batch_{suffix}_action",
        "retained",
        body_calls,
    )
    early_action = _make_named_managed_action(
        f"early_batch_{suffix}_action",
        "early",
        body_calls,
    )
    manager = ActionManager(actions=[retained_action])
    manager_before = dict(manager.actions)
    action_name = f"batch_{suffix}_action"
    action_fn = _make_declared_generator_action(kind, action_name, body_calls)
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)

    with pytest.raises(ActionSignatureError) as exc_info:
        manager.register_many([early_action, action_fn])

    _assert_declared_generator_error(exc_info, action_name)
    assert manager.actions == manager_before
    assert early_action.__name__ not in manager.actions
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert body_calls == []


def test_normal_action_returning_generator_remains_a_deferred_runtime_contract():
    body_calls = []
    manager = ActionManager()

    @action(action_manager=manager)
    def return_generator(agent) -> object:
        """Return a generator object from an ordinary action function.

        Returns:
            A deferred generator result.
        """
        body_calls.append(agent)
        return (value for value in ("deferred",))

    agent = object()

    result = manager.execute(
        agent,
        ActionChoice(name="return_generator", arguments={}),
    )

    assert not inspect.isgeneratorfunction(return_generator)
    assert inspect.isgenerator(result)
    assert body_calls == [agent]
    assert list(result) == ["deferred"]


@pytest.mark.parametrize(
    ("case", "parameter_names"),
    _DUPLICATE_AGENT_PARAMETER_CASES,
)
def test_action_rejects_duplicate_injected_agent_parameters_before_mutation(
    case,
    parameter_names,
):
    body_calls = []
    action_name = f"direct_duplicate_agent_{case.replace('-', '_')}"
    action_fn = _make_duplicate_agent_action(case, action_name, body_calls)
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)

    with pytest.raises(ActionSignatureError) as exc_info:
        action(action_fn)

    _assert_duplicate_agent_error(exc_info, action_name, parameter_names)
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert body_calls == []


@pytest.mark.parametrize(
    ("case", "parameter_names"),
    _DUPLICATE_AGENT_PARAMETER_CASES,
)
def test_action_manager_decorator_rejects_duplicate_injected_agent_parameters_atomically(
    case,
    parameter_names,
):
    body_calls = []
    retained_action = _make_named_managed_action(
        f"retained_duplicate_agent_decorator_{case.replace('-', '_')}",
        "retained",
        body_calls,
    )
    manager = ActionManager(actions=[retained_action])
    manager_before = dict(manager.actions)
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)
    action_name = f"managed_duplicate_agent_{case.replace('-', '_')}"
    action_fn = _make_duplicate_agent_action(case, action_name, body_calls)

    with pytest.raises(ActionSignatureError) as exc_info:
        action(action_manager=manager)(action_fn)

    _assert_duplicate_agent_error(exc_info, action_name, parameter_names)
    assert manager.actions == manager_before
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert body_calls == []


@pytest.mark.parametrize(
    ("case", "parameter_names"),
    _DUPLICATE_AGENT_PARAMETER_CASES,
)
def test_action_manager_register_rejects_raw_duplicate_injected_agent_parameters(
    case,
    parameter_names,
):
    body_calls = []
    retained_action = _make_named_managed_action(
        f"retained_duplicate_agent_register_{case.replace('-', '_')}",
        "retained",
        body_calls,
    )
    manager = ActionManager(actions=[retained_action])
    manager_before = dict(manager.actions)
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)
    action_name = f"raw_duplicate_agent_{case.replace('-', '_')}"
    action_fn = _make_duplicate_agent_action(case, action_name, body_calls)

    with pytest.raises(ActionSignatureError) as exc_info:
        manager.register(action_fn)

    _assert_duplicate_agent_error(exc_info, action_name, parameter_names)
    assert manager.actions == manager_before
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert body_calls == []


@pytest.mark.parametrize(
    ("case", "parameter_names"),
    _DUPLICATE_AGENT_PARAMETER_CASES,
)
def test_action_manager_constructor_rejects_duplicate_injected_agent_parameters_atomically(
    case,
    parameter_names,
):
    body_calls = []
    early_action = _make_named_managed_action(
        f"early_duplicate_agent_constructor_{case.replace('-', '_')}",
        "early",
        body_calls,
    )
    action_name = f"constructor_duplicate_agent_{case.replace('-', '_')}"
    action_fn = _make_duplicate_agent_action(case, action_name, body_calls)
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)
    failed_manager = ActionManager.__new__(ActionManager)

    with pytest.raises(ActionSignatureError) as exc_info:
        failed_manager.__init__(actions=[early_action, action_fn])

    _assert_duplicate_agent_error(exc_info, action_name, parameter_names)
    assert failed_manager.actions == {}
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert body_calls == []


@pytest.mark.parametrize(
    ("case", "parameter_names"),
    _DUPLICATE_AGENT_PARAMETER_CASES,
)
def test_action_manager_register_many_duplicate_injected_agent_parameters_is_atomic(
    case,
    parameter_names,
):
    body_calls = []
    retained_action = _make_named_managed_action(
        f"retained_duplicate_agent_batch_{case.replace('-', '_')}",
        "retained",
        body_calls,
    )
    early_action = _make_named_managed_action(
        f"early_duplicate_agent_batch_{case.replace('-', '_')}",
        "early",
        body_calls,
    )
    manager = ActionManager(actions=[retained_action])
    manager_before = dict(manager.actions)
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)
    action_name = f"batch_duplicate_agent_{case.replace('-', '_')}"
    action_fn = _make_duplicate_agent_action(case, action_name, body_calls)

    with pytest.raises(ActionSignatureError) as exc_info:
        manager.register_many([early_action, action_fn])

    _assert_duplicate_agent_error(exc_info, action_name, parameter_names)
    assert manager.actions == manager_before
    assert early_action.__name__ not in manager.actions
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert body_calls == []


def test_action_preserves_one_keyword_only_capitalized_agent_parameter():
    body_calls = []

    @action
    def keyword_only_agent(value: str, *, Agent) -> str:  # noqa: N803
        """Return a value.

        Args:
            value: Value to return.

        Returns:
            The provided value.
        """
        body_calls.append((Agent, value))
        return value

    manager = ActionManager(actions=[keyword_only_agent])
    injected_agent = object()
    result = manager.execute(
        injected_agent,
        ActionChoice(name="keyword_only_agent", arguments={"value": "accepted"}),
    )

    assert result == "accepted"
    assert keyword_only_agent.__action_schema__["parameters"]["properties"] == {
        "value": {"type": "string", "description": "Value to return."}
    }
    assert "Agent" not in keyword_only_agent.__action_metadata__.parameters
    assert body_calls == [(injected_agent, "accepted")]


def test_action_generates_metadata_and_schema_from_type_hints_and_docstring():
    @action
    def visit(agent, location: str, duration: int, tags: list[str]) -> str:
        """Visit a location.

        Args:
            location: Destination name.
            duration: Number of turns to stay.
            tags: Labels for the visit.

        Returns:
            Visit confirmation.
        """
        del agent
        return f"{location}:{duration}:{tags}"

    metadata = visit.__action_metadata__
    schema = visit.__action_schema__

    assert metadata == ActionMetadata(
        name="visit",
        description="Visit a location.",
        parameters={
            "location": {"type": "string", "description": "Destination name."},
            "duration": {
                "type": "integer",
                "description": "Number of turns to stay.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Labels for the visit.",
            },
        },
        required=["location", "duration", "tags"],
        return_description="Visit confirmation.",
    )
    assert schema == {
        "name": "visit",
        "description": "Visit a location.",
        "parameters": {
            "type": "object",
            "properties": metadata.parameters,
            "required": ["location", "duration", "tags"],
            "additionalProperties": False,
        },
        "returns": "Visit confirmation.",
    }


def test_action_schema_closes_outer_arguments_but_preserves_declared_dict_values():
    @action
    def record_scores(agent, scores: dict[str, int], round_number: int) -> None:
        """Record named scores for one round.

        Args:
            scores: Integer scores keyed by participant name.
            round_number: Round being recorded.
        """
        del agent, scores, round_number

    parameters = record_scores.__action_schema__["parameters"]

    assert parameters["additionalProperties"] is False
    assert parameters["properties"]["scores"] == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
        "description": "Integer scores keyed by participant name.",
    }


def test_action_schema_exposes_literal_and_nullable_union():
    @action
    def select_value(
        agent,
        mode: Literal["a", "b"],
        optional_mode: Literal["a", "b"] | None,
        value: int | str | None,
    ) -> str:
        """Select a typed value.

        Args:
            mode: Selection mode.
            optional_mode: Optional selection mode.
            value: Optional selected value.

        Returns:
            Selection confirmation.
        """
        del agent, optional_mode, value
        return mode

    properties = select_value.__action_schema__["parameters"]["properties"]

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


def test_action_schema_exposes_unambiguous_integral_numeric_literals():
    @action
    def select_numeric_value(
        agent,
        float_value: Literal[1.0],
        integer_value: Literal[1],
    ) -> tuple[float, int]:
        """Select numeric literal values.

        Args:
            float_value: Float literal to select.
            integer_value: Integer literal to select.

        Returns:
            The selected values.
        """
        del agent
        return float_value, integer_value

    properties = select_numeric_value.__action_schema__["parameters"]["properties"]

    assert properties["float_value"] == {
        "type": "number",
        "enum": [1.0],
        "description": "Float literal to select.",
    }
    assert properties["integer_value"] == {
        "type": "integer",
        "enum": [1],
        "description": "Integer literal to select.",
    }


_LIST_VARIADIC_NUMERIC_LITERAL_AMBIGUITIES = (
    pytest.param(
        list[Literal[1]] | tuple[Literal[1.0], ...],
        id="list-integer-first",
    ),
    pytest.param(
        tuple[Literal[1.0], ...] | list[Literal[1]],
        id="variadic-float-first",
    ),
)


def _make_numeric_container_action(name, annotation, mutations):
    def select_numeric_value(agent, value) -> object:
        """Select a numeric container value.

        Args:
            value: Value to select.

        Returns:
            The selected value.
        """
        mutations.append((agent, value))
        return value

    select_numeric_value.__name__ = name
    select_numeric_value.__annotations__["value"] = annotation
    return select_numeric_value


@pytest.mark.parametrize(
    "annotation",
    _LIST_VARIADIC_NUMERIC_LITERAL_AMBIGUITIES,
)
def test_action_rejects_list_variadic_tuple_numeric_literal_ambiguity_before_mutation(
    annotation,
):
    mutations = []
    action_name = "select_nested_numeric_value"
    action_fn = _make_numeric_container_action(action_name, annotation, mutations)
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)

    with pytest.raises(
        TypeError,
        match=r"JSON-equivalent|ambiguous|indistinguishable",
    ):
        action(action_fn)

    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert mutations == []


@pytest.mark.parametrize(
    "annotation",
    _LIST_VARIADIC_NUMERIC_LITERAL_AMBIGUITIES,
)
def test_action_manager_decorator_rejects_nested_numeric_literal_ambiguity_atomically(
    annotation,
):
    mutations = []
    retained_action = _make_named_managed_action(
        "retained_nested_numeric_action",
        "retained",
        mutations,
    )
    manager = ActionManager(actions=[retained_action])
    manager_before = dict(manager.actions)
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)
    action_fn = _make_numeric_container_action(
        "managed_nested_numeric_value",
        annotation,
        mutations,
    )

    with pytest.raises(
        TypeError,
        match=r"JSON-equivalent|ambiguous|indistinguishable",
    ):
        action(action_manager=manager)(action_fn)

    assert manager.actions == manager_before
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert mutations == []


def test_action_rejects_annotated_nested_dictionary_numeric_literal_ambiguity():
    annotation = (
        dict[
            str,
            Annotated[list[Literal[1]], "integer list branch"],
        ]
        | dict[str, tuple[Literal[1.0], ...]]
    )
    mutations = []
    action_fn = _make_numeric_container_action(
        "select_annotated_nested_numeric_value",
        annotation,
        mutations,
    )
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)

    with pytest.raises(
        TypeError,
        match=r"JSON-equivalent|ambiguous|indistinguishable",
    ):
        action(action_fn)

    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert mutations == []


@pytest.mark.parametrize(
    "annotation",
    (
        pytest.param(
            tuple[Literal[1], Literal[1]] | tuple[Literal[1.0], Literal[1.0]],
            id="same-fixed-length",
        ),
        pytest.param(
            tuple[Literal[1], Literal[1]] | tuple[Literal[1.0], ...],
            id="fixed-and-variadic-overlap",
        ),
    ),
)
def test_action_rejects_overlapping_homogeneous_tuple_numeric_literal_ambiguity(
    annotation,
):
    mutations = []
    action_fn = _make_numeric_container_action(
        "select_tuple_numeric_value",
        annotation,
        mutations,
    )
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)

    with pytest.raises(
        TypeError,
        match=r"JSON-equivalent|ambiguous|indistinguishable",
    ):
        action(action_fn)

    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert not hasattr(action_fn, "__action_metadata__")
    assert not hasattr(action_fn, "__action_schema__")
    assert mutations == []


@pytest.mark.parametrize(
    ("case_name", "annotation", "expected_any_of"),
    (
        pytest.param(
            "boolean_integer",
            list[Literal[True]] | tuple[Literal[1], ...],
            [
                {
                    "type": "array",
                    "items": {"type": "boolean", "enum": [True]},
                },
                {
                    "type": "array",
                    "items": {"type": "integer", "enum": [1]},
                },
            ],
            id="boolean-and-integer",
        ),
        pytest.param(
            "non_equivalent_numbers",
            list[Literal[1]] | tuple[Literal[2.0], ...],
            [
                {
                    "type": "array",
                    "items": {"type": "integer", "enum": [1]},
                },
                {
                    "type": "array",
                    "items": {"type": "number", "enum": [2.0]},
                },
            ],
            id="non-equivalent-numbers",
        ),
        pytest.param(
            "different_fixed_lengths",
            tuple[Literal[1], Literal[1]]
            | tuple[Literal[1.0], Literal[1.0], Literal[1.0]],
            [
                {
                    "type": "array",
                    "items": {"type": "integer", "enum": [1]},
                    "minItems": 2,
                    "maxItems": 2,
                },
                {
                    "type": "array",
                    "items": {"type": "number", "enum": [1.0]},
                    "minItems": 3,
                    "maxItems": 3,
                },
            ],
            id="different-fixed-lengths",
        ),
    ),
)
def test_action_accepts_non_ambiguous_nested_numeric_literal_branches(
    case_name,
    annotation,
    expected_any_of,
):
    mutations = []
    action_name = f"select_non_ambiguous_nested_numeric_value_{case_name}"
    action_fn = _make_numeric_container_action(action_name, annotation, mutations)

    decorated_action = action(action_fn)

    assert decorated_action is action_fn
    assert _GLOBAL_ACTION_REGISTRY[action_name] is action_fn
    assert action_fn.__action_schema__["parameters"]["properties"]["value"] == {
        "anyOf": expected_any_of,
        "description": "Value to select.",
    }
    assert mutations == []


def test_action_schema_rejects_ambiguous_numeric_literal_without_registration():
    with pytest.raises(
        TypeError,
        match=r"JSON-equivalent|ambiguous|indistinguishable",
    ):

        @action
        def select_numeric_value(agent, value: Literal[1, 1.0]) -> int | float:
            """Select a numeric value.

            Args:
                value: Numeric value to select.

            Returns:
                The selected value.
            """
            del agent
            return value

    assert "select_numeric_value" not in _GLOBAL_ACTION_REGISTRY


@pytest.mark.parametrize(
    "annotation",
    [
        Literal[1] | Literal[1.0],
        list[Literal[1] | Literal[1.0]],
    ],
    ids=["split-union", "nested-split-union"],
)
def test_action_schema_rejects_split_ambiguous_numeric_literal_without_registration(
    annotation,
):
    def select_numeric_value(agent, value) -> int | float:
        """Select a numeric value.

        Args:
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
        action(select_numeric_value)

    assert "select_numeric_value" not in _GLOBAL_ACTION_REGISTRY
    assert not hasattr(select_numeric_value, "__action_metadata__")
    assert not hasattr(select_numeric_value, "__action_schema__")


@pytest.mark.parametrize(
    "annotation",
    [
        list[Literal[1]] | set[Literal[1.0]],
        dict[str, list[Literal[1]]] | dict[str, set[Literal[1.0]]],
    ],
    ids=["collection-branches", "nested-dictionary-branches"],
)
def test_action_schema_rejects_structurally_ambiguous_literal_without_side_effects(
    annotation,
):
    mutations = []

    def select_numeric_value(agent, value) -> int | float:
        """Select a numeric value.

        Args:
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
        action(select_numeric_value)

    assert "select_numeric_value" not in _GLOBAL_ACTION_REGISTRY
    assert not hasattr(select_numeric_value, "__action_metadata__")
    assert not hasattr(select_numeric_value, "__action_schema__")
    assert mutations == []


def test_action_rejects_empty_literal_without_registration_or_mutation():
    mutations = []

    def select_value(agent, value) -> str:
        """Select a literal value.

        Args:
            value: Literal value to select.

        Returns:
            Selection confirmation.
        """
        mutations.append((agent, value))
        return "selected"

    select_value.__annotations__["value"] = Literal[()]

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(select_value)

    message = str(exc_info.value)
    assert "Literal" in message
    assert "empty" in message.lower() or "at least one" in message.lower()
    assert "select_value" not in _GLOBAL_ACTION_REGISTRY
    assert not hasattr(select_value, "__action_metadata__")
    assert not hasattr(select_value, "__action_schema__")
    assert mutations == []


def test_action_omits_agent_parameter_from_schema_by_default():
    @action
    def move(agent, direction: str) -> str:
        """Move in a direction.

        Args:
            direction: Direction to move.

        Returns:
            Move confirmation.
        """
        del agent
        return direction

    params = move.__action_schema__["parameters"]

    assert set(params["properties"]) == {"direction"}
    assert params["required"] == ["direction"]
    assert "agent" not in move.__action_metadata__.parameters

    manager = ActionManager(actions=[move])
    manager_params = manager.get_actions_schema()[0]["parameters"]

    assert set(manager_params["properties"]) == {"direction"}
    assert manager_params["required"] == ["direction"]


def test_action_schema_marks_only_non_default_non_agent_parameters_required():
    @action
    def send_message(
        agent,
        recipient_id: int,
        message: str,
        priority: int = 1,
        channel: str = "local",
    ) -> str:
        """Send a message.

        Args:
            recipient_id: Recipient agent id.
            message: Message body.
            priority: Delivery priority.
            channel: Delivery channel.

        Returns:
            Message confirmation.
        """
        del agent, priority, channel
        return message

    params = send_message.__action_schema__["parameters"]

    assert set(params["properties"]) == {
        "recipient_id",
        "message",
        "priority",
        "channel",
    }
    assert params["required"] == ["recipient_id", "message"]
    assert send_message.__action_metadata__.required == ["recipient_id", "message"]


def test_action_ignore_agent_false_is_disabled_with_clear_error():
    with pytest.raises(ValueError, match=r"ignore_agent=False.*not supported"):

        @action(ignore_agent=False)
        def expose_agent(agent) -> str:
            """Attempt to expose the framework-injected agent.

            Returns:
                Exposure confirmation.
            """
            del agent
            return "exposed"


def test_bare_action_global_registration_can_be_opted_into_by_name():
    @action
    def registered_action(agent, message: str) -> str:
        """Registered action.

        Args:
            message: Message to store.

        Returns:
            The message.
        """
        del agent
        return message

    assert _GLOBAL_ACTION_REGISTRY["registered_action"] is registered_action

    manager = ActionManager(actions=["registered_action"])

    assert manager.actions == {"registered_action": registered_action}
    assert manager.available_actions() == {"registered_action": registered_action}


def test_action_manager_registration_is_idempotent_for_identical_callable_and_name():
    execution_calls = []
    registered_action = _make_named_managed_action(
        "manager_idempotent_action",
        "registered",
        execution_calls,
    )
    manager = ActionManager(actions=[registered_action])
    original_items = list(manager.actions.items())

    manager.register(registered_action)

    assert list(manager.actions.items()) == original_items
    assert manager.actions["manager_idempotent_action"] is registered_action
    assert execution_calls == []


def test_action_manager_rejects_different_callable_with_occupied_name():
    execution_calls = []
    original_action = _make_named_managed_action(
        "manager_collision_action",
        "original",
        execution_calls,
    )
    conflicting_action = _make_named_managed_action(
        "manager_collision_action",
        "conflicting",
        execution_calls,
    )
    manager = ActionManager(actions=[original_action])
    original_items = list(manager.actions.items())

    with pytest.raises(ValueError) as exc_info:
        manager.register(conflicting_action)

    message = str(exc_info.value)
    assert "manager_collision_action" in message
    assert any(
        term in message.lower()
        for term in ("already", "collision", "conflict", "different")
    )
    assert list(manager.actions.items()) == original_items
    assert manager.actions["manager_collision_action"] is original_action
    assert execution_calls == []


def test_bare_action_registration_is_idempotent_when_redecorating_same_callable():
    execution_calls = []

    @action
    def global_idempotent_action(agent) -> str:
        """Return a marker.

        Returns:
            The marker.
        """
        del agent
        execution_calls.append("called")
        return "registered"

    registry_after_first_registration = dict(_GLOBAL_ACTION_REGISTRY)

    redecorated_action = action(global_idempotent_action)

    assert redecorated_action is global_idempotent_action
    assert registry_after_first_registration == _GLOBAL_ACTION_REGISTRY
    assert (
        _GLOBAL_ACTION_REGISTRY["global_idempotent_action"] is global_idempotent_action
    )
    assert execution_calls == []


def test_bare_action_registration_rejects_name_collision_and_retains_original():
    execution_calls = []

    @action
    def global_collision_action(agent) -> str:
        """Return the original marker.

        Returns:
            The original marker.
        """
        del agent
        execution_calls.append("original")
        return "original"

    original_action = global_collision_action
    registry_before_collision = dict(_GLOBAL_ACTION_REGISTRY)

    def global_collision_action(agent) -> str:
        """Return the conflicting marker.

        Returns:
            The conflicting marker.
        """
        del agent
        execution_calls.append("conflicting")
        return "conflicting"

    with pytest.raises(ValueError) as exc_info:
        action(global_collision_action)

    message = str(exc_info.value)
    assert "global_collision_action" in message
    assert any(
        term in message.lower()
        for term in ("already", "collision", "conflict", "different")
    )
    assert registry_before_collision == _GLOBAL_ACTION_REGISTRY
    assert _GLOBAL_ACTION_REGISTRY["global_collision_action"] is original_action
    assert execution_calls == []


def test_register_many_late_collision_is_atomic():
    execution_calls = []
    original_action = _make_named_managed_action(
        "batch_collision_action",
        "original",
        execution_calls,
    )
    early_new_action = _make_named_managed_action(
        "batch_early_new_action",
        "early",
        execution_calls,
    )
    conflicting_action = _make_named_managed_action(
        "batch_collision_action",
        "conflicting",
        execution_calls,
    )
    manager = ActionManager(actions=[original_action])
    original_items = list(manager.actions.items())

    with pytest.raises(ValueError) as exc_info:
        manager.register_many((early_new_action, conflicting_action))

    assert "batch_collision_action" in str(exc_info.value)
    assert list(manager.actions.items()) == original_items
    assert manager.actions["batch_collision_action"] is original_action
    assert "batch_early_new_action" not in manager.actions
    assert execution_calls == []


@pytest.mark.parametrize(
    ("late_ref_kind", "expected_exception", "expected_message"),
    [
        pytest.param(
            "invalid",
            TypeError,
            "callables or registered action names",
            id="invalid-reference",
        ),
        pytest.param(
            "undecorated",
            ActionAnnotationContractError,
            "@action",
            id="undecorated-callable",
        ),
        pytest.param(
            "unknown",
            ValueError,
            "Unknown action name",
            id="unknown-name",
        ),
    ],
)
def test_register_many_late_invalid_reference_is_atomic(
    late_ref_kind,
    expected_exception,
    expected_message,
):
    execution_calls = []
    existing_action = _make_named_managed_action(
        "batch_existing_action",
        "existing",
        execution_calls,
    )
    early_new_action = _make_named_managed_action(
        "batch_valid_early_action",
        "early",
        execution_calls,
    )

    if late_ref_kind == "invalid":
        late_ref = object()
    elif late_ref_kind == "undecorated":

        def late_ref(agent) -> str:
            del agent
            execution_calls.append("undecorated")
            return "undecorated"

    else:
        late_ref = "missing_batch_action"

    manager = ActionManager(actions=[existing_action])
    original_items = list(manager.actions.items())

    with pytest.raises(expected_exception, match=expected_message):
        manager.register_many([early_new_action, late_ref])

    assert list(manager.actions.items()) == original_items
    assert manager.actions["batch_existing_action"] is existing_action
    assert "batch_valid_early_action" not in manager.actions
    assert execution_calls == []


def test_action_manager_constructor_collision_does_not_partially_register():
    execution_calls = []
    early_action = _make_named_managed_action(
        "constructor_early_action",
        "early",
        execution_calls,
    )
    original_action = _make_named_managed_action(
        "constructor_collision_action",
        "original",
        execution_calls,
    )
    conflicting_action = _make_named_managed_action(
        "constructor_collision_action",
        "conflicting",
        execution_calls,
    )
    failed_manager = ActionManager.__new__(ActionManager)

    with pytest.raises(ValueError) as exc_info:
        failed_manager.__init__(
            actions=[early_action, original_action, conflicting_action]
        )

    assert "constructor_collision_action" in str(exc_info.value)
    assert failed_manager.actions == {}
    assert execution_calls == []


def test_register_many_resolves_repeated_names_and_staged_callable_by_identity():
    execution_calls = []

    @action
    def repeated_global_action(agent) -> str:
        """Return the global marker.

        Returns:
            The global marker.
        """
        del agent
        execution_calls.append("global")
        return "global"

    global_manager = ActionManager(
        actions=["repeated_global_action", "repeated_global_action"]
    )

    staged_action = _make_named_managed_action(
        "staged_named_action",
        "staged",
        execution_calls,
    )
    staged_manager = ActionManager(
        actions=[staged_action, "staged_named_action", "staged_named_action"]
    )

    assert global_manager.actions == {"repeated_global_action": repeated_global_action}
    assert global_manager.actions["repeated_global_action"] is repeated_global_action
    assert staged_manager.actions == {"staged_named_action": staged_action}
    assert staged_manager.actions["staged_named_action"] is staged_action
    assert execution_calls == []


def test_action_manager_rejects_undecorated_callable_before_contract_inference(
    monkeypatch,
):
    @action
    def decorated_action(agent, amount: int) -> int:
        """Return an amount.

        Args:
            amount: Amount to return.

        Returns:
            The amount.
        """
        del agent
        return amount

    def undecorated_action(agent, amount: int) -> int:
        del agent
        return amount

    manager = ActionManager()
    contract_inference_calls = []
    original_get_contract = action_manager_module._get_action_parameter_contract

    def record_contract_inference(*args, **kwargs):
        contract_inference_calls.append(args[0])
        return original_get_contract(*args, **kwargs)

    # Registration must reject on the missing decorator marker before attempting
    # to infer a callable contract from otherwise valid annotations.
    with monkeypatch.context() as patch:
        patch.setattr(
            action_manager_module,
            "_get_action_parameter_contract",
            record_contract_inference,
        )
        with pytest.raises(ActionAnnotationContractError) as exc_info:
            manager.register(undecorated_action)

    message = str(exc_info.value)
    assert "undecorated_action" in message
    assert "@action" in message or "decorated" in message.lower()
    assert manager.actions == {}
    assert contract_inference_calls == []
    assert not hasattr(undecorated_action, "__action_schema__")

    callable_manager = ActionManager(actions=[decorated_action])
    named_manager = ActionManager(actions=["decorated_action"])

    assert callable_manager.actions == {"decorated_action": decorated_action}
    assert named_manager.actions == {"decorated_action": decorated_action}


def test_action_manager_argument_registers_directly_with_that_manager():
    manager = ActionManager()

    @action(action_manager=manager)
    def direct_action(agent, score: float) -> str:
        """Direct action.

        Args:
            score: Score to record.

        Returns:
            Direct action confirmation.
        """
        del agent
        return f"score={score}"

    expected_schema = {
        "name": "direct_action",
        "description": "Direct action.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "description": "Score to record.",
                },
            },
            "required": ["score"],
            "additionalProperties": False,
        },
        "returns": "Direct action confirmation.",
    }

    assert manager.available_actions() == {"direct_action": direct_action}
    assert manager.get_actions_schema() == [expected_schema]
    assert manager.get_actions_schema(actions="direct_action") == [expected_schema]
    assert direct_action.__action_schema__ == expected_schema
    assert "direct_action" not in _GLOBAL_ACTION_REGISTRY


def test_action_manager_and_exclusively_owned_callable_can_be_garbage_collected():
    class ActionState:
        value = "performed"

    def register_owned_action(manager):
        state = ActionState()

        @action(action_manager=manager)
        def owned_action(agent) -> str:
            """Use state owned exclusively by this action.

            Returns:
                The owned state value.
            """
            del agent
            return state.value

        return weakref.ref(owned_action), weakref.ref(state)

    manager = ActionManager()
    action_ref, state_ref = register_owned_action(manager)
    manager_ref = weakref.ref(manager)

    assert manager.available_actions()["owned_action"](None) == "performed"
    assert action_ref() is manager.actions["owned_action"]
    assert state_ref() is not None

    del manager
    gc.collect()

    assert manager_ref() is None
    assert action_ref() is None
    assert state_ref() is None


def test_action_schema_resolves_postponed_annotations_with_type_checking_agent_import():
    @action
    def notify_neighbor(
        agent: LLMAgent,
        listener_agent_ids: list[int],
        message: str,
    ) -> str:
        """Notify a neighboring agent.

        Args:
            listener_agent_ids: Listener ids.
            message: Message body.

        Returns:
            Notification confirmation.
        """
        del agent
        return message

    params = notify_neighbor.__action_schema__["parameters"]

    assert set(params["properties"]) == {"listener_agent_ids", "message"}
    assert params["required"] == ["listener_agent_ids", "message"]
    assert params["properties"]["listener_agent_ids"] == {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Listener ids.",
    }
    assert params["properties"]["message"] == {
        "type": "string",
        "description": "Message body.",
    }
    assert "agent" not in notify_neighbor.__action_metadata__.parameters


def test_action_schema_ignores_string_llm_agent_parameter_annotation():
    @action
    def inspect_agent(agent: LLMAgent, note: str) -> str:
        """Inspect the injected agent safely.

        Args:
            note: Inspection note.

        Returns:
            Inspection confirmation.
        """
        del agent
        return note

    params = inspect_agent.__action_schema__["parameters"]

    assert set(params["properties"]) == {"note"}
    assert params["required"] == ["note"]
    assert "agent" not in inspect_agent.__action_metadata__.parameters


def test_action_rejects_missing_exposed_parameter_annotation_without_registration():
    def apply_payload(agent, payload) -> str:
        """Apply a payload.

        Args:
            payload: Payload to apply.

        Returns:
            Payload confirmation.
        """
        del agent, payload
        return "applied"

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(apply_payload)

    message = str(exc_info.value)
    assert "apply_payload" in message
    assert "payload" in message
    assert "annotation" in message.lower()
    assert "apply_payload" not in _GLOBAL_ACTION_REGISTRY
    assert not hasattr(apply_payload, "__action_metadata__")
    assert not hasattr(apply_payload, "__action_schema__")


@pytest.mark.parametrize(
    "annotation",
    [
        Any,
        object,
        _UnsupportedActionPayload,
        list[Any],
        list[_UnsupportedActionPayload],
        dict[str, object],
        list,
        dict,
        tuple,
        set,
    ],
    ids=[
        "any",
        "object",
        "custom",
        "nested-any",
        "nested-custom",
        "nested-object",
        "bare-list",
        "bare-dict",
        "bare-tuple",
        "bare-set",
    ],
)
def test_action_rejects_unsupported_or_unconstrained_parameter_annotation(
    annotation,
):
    def apply_payload(agent, payload) -> str:
        """Apply a payload.

        Args:
            payload: Payload to apply.

        Returns:
            Payload confirmation.
        """
        del agent, payload
        return "applied"

    apply_payload.__annotations__["payload"] = annotation

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(apply_payload)

    message = str(exc_info.value)
    assert "apply_payload" in message
    assert "payload" in message
    assert "annotation" in message.lower()
    assert "apply_payload" not in _GLOBAL_ACTION_REGISTRY
    assert not hasattr(apply_payload, "__action_metadata__")
    assert not hasattr(apply_payload, "__action_schema__")


@pytest.mark.parametrize(
    "annotation",
    [
        set[int],
        set[tuple[int, int]],
        list[set[int]],
    ],
    ids=[
        "scalar-set",
        "tuple-set",
        "nested-set",
    ],
)
def test_action_rejects_set_annotations_before_metadata_or_registration(annotation):
    mutations = []

    def apply_values(agent, values) -> str:
        """Apply a set of values.

        Args:
            values: Values to apply.

        Returns:
            Application confirmation.
        """
        mutations.append((agent, values))
        return "applied"

    apply_values.__annotations__["values"] = annotation
    manager = ActionManager()
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(action_manager=manager)(apply_values)

    message = str(exc_info.value)
    assert "apply_values" in message
    assert "values" in message
    assert "set" in message.lower()
    assert manager.actions == {}
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert not hasattr(apply_values, "__action_metadata__")
    assert not hasattr(apply_values, "__action_schema__")
    assert mutations == []


def test_action_exempts_injected_agent_and_return_annotations_from_contract():
    @action
    def inspect_payload(
        agent: Any,
        payload: list[int],
    ) -> _UnsupportedActionPayload:
        """Inspect a payload.

        Args:
            payload: Integer payload to inspect.

        Returns:
            Inspection result.
        """
        del agent, payload
        return _UnsupportedActionPayload()

    params = inspect_payload.__action_schema__["parameters"]

    assert params["properties"] == {
        "payload": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Integer payload to inspect.",
        }
    }
    assert params["required"] == ["payload"]


@pytest.mark.parametrize(
    ("annotation", "item_schema", "length"),
    [
        pytest.param(tuple[int, int], {"type": "integer"}, 2, id="two-integers"),
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
def test_action_schema_preserves_homogeneous_fixed_tuple_length(
    annotation,
    item_schema,
    length,
):
    def collect_values(agent, values) -> None:
        """Collect fixed tuple values.

        Args:
            values: Values to collect.
        """
        del agent, values

    collect_values.__annotations__["values"] = annotation
    manager = ActionManager()
    collect_values = action(action_manager=manager)(collect_values)

    assert collect_values.__action_schema__["parameters"]["properties"]["values"] == {
        "type": "array",
        "items": item_schema,
        "minItems": length,
        "maxItems": length,
        "description": "Values to collect.",
    }


@pytest.mark.parametrize(
    "annotation",
    [
        tuple[int, str],
        list[tuple[int, str]],
    ],
    ids=["direct", "nested"],
)
def test_action_rejects_heterogeneous_fixed_tuple_before_metadata_or_registration(
    annotation,
):
    mutations = []

    def apply_values(agent, values) -> str:
        """Apply tuple values.

        Args:
            values: Values to apply.

        Returns:
            Application confirmation.
        """
        mutations.append((agent, values))
        return "applied"

    apply_values.__annotations__["values"] = annotation
    manager = ActionManager()
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(action_manager=manager)(apply_values)

    message = str(exc_info.value)
    assert "apply_values" in message
    assert "values" in message
    assert "tuple" in message.lower()
    assert "homogeneous" in message.lower() or "heterogeneous" in message.lower()
    assert manager.actions == {}
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert not hasattr(apply_values, "__action_metadata__")
    assert not hasattr(apply_values, "__action_schema__")
    assert mutations == []


def test_action_schema_preserves_homogeneous_variadic_tuple_annotations():
    @action
    def collect_values(
        agent,
        integer_values: tuple[int, ...],
        text_values: tuple[str, ...],
        batches: tuple[list[int], ...],
    ) -> None:
        """Collect homogeneous tuple values.

        Args:
            integer_values: Integer values to collect.
            text_values: Text values to collect.
            batches: Integer batches to collect.

        Returns:
            Nothing.
        """
        del agent, integer_values, text_values, batches

    properties = collect_values.__action_schema__["parameters"]["properties"]

    assert properties == {
        "integer_values": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Integer values to collect.",
        },
        "text_values": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Text values to collect.",
        },
        "batches": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "description": "Integer batches to collect.",
        },
    }
    for property_schema in properties.values():
        assert "minItems" not in property_schema
        assert "maxItems" not in property_schema


@pytest.mark.parametrize(
    ("annotation", "default_value"),
    [
        pytest.param(int, "4", id="integer-string"),
        pytest.param(int, True, id="integer-boolean"),
        pytest.param(float, float("nan"), id="float-nan"),
        pytest.param(float, float("inf"), id="float-infinity"),
        pytest.param(list[int], [1, "2"], id="list-item"),
        pytest.param(list[int], (1, 2), id="list-container"),
        pytest.param(tuple[int, int], (1, "2"), id="tuple-item"),
        pytest.param(tuple[int, int], [1, 2], id="tuple-container"),
        pytest.param(dict[str, int], {"one": "1"}, id="dict-item"),
        pytest.param(dict[str, int], [("one", 1)], id="dict-container"),
        pytest.param(Literal["safe", "fast"], "other", id="literal"),
    ],
)
def test_action_rejects_incompatible_default_before_metadata_or_registration(
    annotation,
    default_value,
):
    mutations = []

    def apply_default(agent, value) -> object:
        """Apply a default value.

        Args:
            value: Value to apply.

        Returns:
            The applied value.
        """
        mutations.append((agent, value))
        return value

    apply_default.__annotations__["value"] = annotation
    apply_default.__defaults__ = (default_value,)
    manager = ActionManager()
    registry_before = dict(_GLOBAL_ACTION_REGISTRY)

    with pytest.raises((TypeError, ValueError)) as exc_info:
        action(action_manager=manager)(apply_default)

    message = str(exc_info.value)
    assert "apply_default" in message
    assert "value" in message
    assert "default" in message.lower()
    assert manager.actions == {}
    assert registry_before == _GLOBAL_ACTION_REGISTRY
    assert not hasattr(apply_default, "__action_metadata__")
    assert not hasattr(apply_default, "__action_schema__")
    assert mutations == []


def test_action_rejects_non_agent_positional_only_parameter():
    with pytest.raises(ValueError) as exc_info:

        @action
        def target_position(position: str, /, speed: int) -> str:
            """Target a position.

            Args:
                position: Position to target.
                speed: Movement speed.

            Returns:
                Target confirmation.
            """
            return f"{position}:{speed}"

    message = str(exc_info.value)

    assert "target_position" in message
    assert "position" in message
    assert "positional-only" in message


def test_action_rejects_non_agent_varargs_parameter():
    with pytest.raises(ValueError) as exc_info:

        @action
        def collect_items(agent, *items: str) -> str:
            """Collect items.

            Args:
                items: Items to collect.

            Returns:
                Collection confirmation.
            """
            del agent
            return ",".join(items)

    message = str(exc_info.value)

    assert "collect_items" in message
    assert "items" in message
    assert "*args" in message or "variadic positional" in message


def test_action_rejects_non_agent_varkwargs_parameter():
    with pytest.raises(ValueError) as exc_info:

        @action
        def configure_agent(agent, **options: str) -> str:
            """Configure an agent.

            Args:
                options: Configuration options.

            Returns:
                Configuration confirmation.
            """
            del agent
            return ",".join(sorted(options))

    message = str(exc_info.value)

    assert "configure_agent" in message
    assert "options" in message
    assert "**kwargs" in message or "variadic keyword" in message


def test_action_rejects_positional_only_agent_parameter():
    with pytest.raises(ValueError) as exc_info:

        @action
        def inspect_agent(agent, /, note: str) -> str:
            """Inspect an agent.

            Args:
                note: Inspection note.

            Returns:
                Inspection confirmation.
            """
            del agent
            return note

    message = str(exc_info.value)

    assert "inspect_agent" in message
    assert "agent" in message
    assert "positional-only" in message


def test_action_rejects_varargs_agent_parameter():
    with pytest.raises(ValueError) as exc_info:

        @action
        def capture_agent_args(*agent) -> str:
            """Capture variadic agent arguments.

            Returns:
                Capture confirmation.
            """
            return repr(agent)

    message = str(exc_info.value)

    assert "capture_agent_args" in message
    assert "agent" in message
    assert "*args" in message or "variadic positional" in message


def test_action_rejects_varkwargs_agent_parameter():
    with pytest.raises(ValueError) as exc_info:

        @action
        def capture_agent_kwargs(**agent) -> str:
            """Capture variadic agent keyword arguments.

            Returns:
                Capture confirmation.
            """
            return repr(agent)

    message = str(exc_info.value)

    assert "capture_agent_kwargs" in message
    assert "agent" in message
    assert "**kwargs" in message or "variadic keyword" in message


def test_empty_action_managers_expose_no_actions():
    assert ActionManager().available_actions() == {}
    assert ActionManager(actions=None).available_actions() == {}
    assert ActionManager(actions=[]).available_actions() == {}

    manager = ActionManager()

    assert manager.get_actions_schema() == []
    assert manager.get_actions_schema(actions=None) == []
    assert manager.get_actions_schema(actions=[]) == []


def test_callable_configuration_exposes_exactly_that_action():
    @action
    def first_action(agent, amount: int) -> int:
        """First action.

        Args:
            amount: Amount to return.

        Returns:
            The amount.
        """
        del agent
        return amount

    @action
    def second_action(agent, amount: int) -> int:
        """Second action.

        Args:
            amount: Amount to return.

        Returns:
            The amount.
        """
        del agent
        return amount

    manager = ActionManager(actions=[first_action])

    assert manager.actions == {"first_action": first_action}
    assert manager.available_actions() == {"first_action": first_action}
    assert "second_action" not in manager.actions


def test_available_actions_and_schema_selector_behavior():
    @action
    def action_a(agent, value: int) -> int:
        """Action A.

        Args:
            value: Value to return.

        Returns:
            The value.
        """
        del agent
        return value

    @action
    def action_b(agent, name: str) -> str:
        """Action B.

        Args:
            name: Name to return.

        Returns:
            The name.
        """
        del agent
        return name

    manager = ActionManager(actions=[action_a, action_b])

    assert manager.available_actions() == {
        "action_a": action_a,
        "action_b": action_b,
    }
    assert manager.available_actions(actions=None) == {}
    assert manager.available_actions(actions=[]) == {}
    assert manager.available_actions(actions="action_a") == {"action_a": action_a}
    assert manager.available_actions(actions=action_b) == {"action_b": action_b}

    assert [schema["name"] for schema in manager.get_actions_schema()] == [
        "action_a",
        "action_b",
    ]
    assert manager.get_actions_schema(actions=None) == []
    assert manager.get_actions_schema(actions=[]) == []
    action_a_schema_names = [
        schema["name"] for schema in manager.get_actions_schema(actions="action_a")
    ]
    action_b_schema_names = [
        schema["name"] for schema in manager.get_actions_schema(actions=action_b)
    ]

    assert action_a_schema_names == ["action_a"]
    assert action_b_schema_names == ["action_b"]


def test_unknown_or_unconfigured_names_and_callables_fail_fast():
    @action
    def configured_action(agent, value: int) -> int:
        """Configured action.

        Args:
            value: Value to return.

        Returns:
            The value.
        """
        del agent
        return value

    @action
    def unconfigured_action(agent, value: int) -> int:
        """Unconfigured action.

        Args:
            value: Value to return.

        Returns:
            The value.
        """
        del agent
        return value

    manager = ActionManager(actions=[configured_action])

    with pytest.raises(ValueError, match="Unknown action name"):
        ActionManager(actions=["missing_action"])

    with pytest.raises(ValueError, match="Unknown action name"):
        manager.available_actions(actions="unconfigured_action")

    with pytest.raises(ValueError, match="Unknown action name"):
        manager.get_actions_schema(actions=["unconfigured_action"])

    with pytest.raises(ValueError, match="Unknown action name"):
        manager.available_actions(actions=unconfigured_action)

    with pytest.raises(ValueError, match="Unknown action name"):
        manager.get_actions_schema(actions=[unconfigured_action])


def test_default_actions_returns_exactly_wait():
    defaults = default_actions()

    assert defaults == (wait,)
    assert isinstance(defaults, tuple)


def test_wait_has_action_metadata_schema_and_is_explicitly_configurable():
    assert wait.__action_metadata__ == ActionMetadata(
        name="wait",
        description="Take no action for this turn.",
        parameters={},
        required=[],
        return_description="A confirmation that the agent waited.",
    )
    assert wait.__action_schema__ == {
        "name": "wait",
        "description": "Take no action for this turn.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "returns": "A confirmation that the agent waited.",
    }

    callable_manager = ActionManager(actions=[wait])
    named_manager = ActionManager(actions=["wait"])

    assert callable_manager.available_actions() == {"wait": wait}
    assert named_manager.available_actions() == {"wait": wait}
    assert callable_manager.get_actions_schema() == [wait.__action_schema__]
    assert named_manager.get_actions_schema(actions="wait") == [
        wait.__action_schema__,
    ]
