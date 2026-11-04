from __future__ import annotations

import copy
import inspect
import json
import logging
import warnings
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from mesa.agent import Agent
from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.model import Model
from mesa.space import (
    ContinuousSpace,
    MultiGrid,
    SingleGrid,
)

from mesa_llm.actions.action_manager import (
    _UNSET as _ACTIONS_UNSET,
)
from mesa_llm.actions.action_manager import (
    ActionChoice,
    ActionManager,
    ActionSelection,
)
from mesa_llm.actions.action_result import ActionPostCommitError, ActResult
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.module_llm import ModuleLLM
from mesa_llm.reasoning.reasoning import (
    _UNSET,
    Observation,
    Plan,
    Reasoning,
    ToolSelection,
)
from mesa_llm.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)


def _log_step_finalizer_failure(
    body_error: BaseException,
    finalizer_error: BaseException,
    finalizer_name: str,
) -> None:
    finalizer_error_is_log_safe = True
    try:
        finalizer_error_type = type(finalizer_error).__name__
    except BaseException:
        finalizer_error_type = "<unknown type>"
        finalizer_error_is_log_safe = False

    try:
        finalizer_error_message = str(finalizer_error)
    except BaseException:
        finalizer_error_message = "<message unavailable>"
        finalizer_error_is_log_safe = False

    try:
        diagnostic = f"{finalizer_name} finalizer failed: {finalizer_error_type}"
        if finalizer_error_message:
            diagnostic = f"{diagnostic}: {finalizer_error_message}"
    except BaseException:
        diagnostic = "Step finalizer failed while preserving the original exception."

    # Diagnostics in this secondary-error path must never mask the body error.
    with suppress(BaseException):
        body_error.add_note(diagnostic)

    if finalizer_error_is_log_safe:
        try:
            logger.error(
                "%s while preserving the original step exception.",
                diagnostic,
                exc_info=(
                    type(finalizer_error),
                    finalizer_error,
                    finalizer_error.__traceback__,
                ),
            )
        except BaseException:
            with suppress(BaseException):
                logger.error(
                    "%s while preserving the original step exception.",
                    diagnostic,
                )
    else:
        with suppress(BaseException):
            logger.error(
                "%s while preserving the original step exception.",
                diagnostic,
            )


def _run_step_body_with_post(
    body: Callable[[], Any],
    post: Callable[[], Any],
    post_name: str,
) -> Any:
    try:
        result = body()
    except BaseException as body_error:
        try:
            post()
        except BaseException as finalizer_error:
            _log_step_finalizer_failure(body_error, finalizer_error, post_name)
        raise

    post()
    return result


async def _arun_step_body_with_post(
    body: Callable[[], Any],
    post: Callable[[], Any],
    post_name: str,
) -> Any:
    try:
        result = await body()
    except BaseException as body_error:
        try:
            await post()
        except BaseException as finalizer_error:
            _log_step_finalizer_failure(body_error, finalizer_error, post_name)
        raise

    await post()
    return result


class _AmbiguousActionChoiceJSONError(ValueError):
    """Raised when an LLM response contains more than one action JSON object."""


def _load_action_choice_json(content: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key, value in pairs:
            if key in data:
                raise ValueError(
                    f"LLM action choice response contains duplicate JSON key {key!r}."
                )
            data[key] = value
        return data

    return json.loads(content, object_pairs_hook=reject_duplicate_keys)


if TYPE_CHECKING:  # pragma: no cover
    from mesa_llm.recording.simulation_recorder import SimulationRecorder


class LLMAgent(Agent):
    """
    LLMAgent manages an LLM backend and optionally connects to a memory module.

    Parameters:
        model (Model): The Mesa model the agent belongs to.
        reasoning (type[Reasoning]): The reasoning strategy used by the agent.
        llm_model (str): The model to use for the LLM in the format
            ``provider/model``. Defaults to ``gemini/gemini-2.0-flash``.
        system_prompt (str | None): Optional system prompt for the LLM.
        vision (float | None): Observation radius for nearby agents. Use ``-1``
            to observe all agents in the simulation.
        internal_state (list[str] | str | None): Optional internal state facts
            exposed to the reasoning module.
        step_prompt (str | None): Optional task-specific prompt used to guide
            the agent each step.
        api_base (str | None): Optional custom LiteLLM-compatible base URL for
            self-hosted or remote inference endpoints.
        tools (list[Callable | str] | tuple[Callable | str, ...] | None):
            Explicit tools exposed to this agent. ``None`` and ``[]`` expose
            no tools; pass explicit read-only tool callables or tool-set
            factories to opt in to tool capabilities.
        actions (list[Callable | str] | tuple[Callable | str, ...] | None):
            Explicit actions exposed to this agent. ``None`` and ``[]`` expose
            no actions; pass an explicit action list or action-set factory to
            opt in to action capabilities.

    Attributes:
        llm (ModuleLLM): The internal LLM interface used by the agent.
        memory (Memory | None): The memory module attached to this agent, if any.
    """

    def __init__(
        self,
        model: Model,
        reasoning: type[Reasoning],
        llm_model: str = "gemini/gemini-2.0-flash",
        fallback_model: str | None = None,
        system_prompt: str | None = None,
        vision: float | None = None,
        internal_state: list[str] | str | None = None,
        step_prompt: str | None = None,
        api_base: str | None = None,
        tools: list[Callable | str] | tuple[Callable | str, ...] | None = None,
        actions: list[Callable | str] | tuple[Callable | str, ...] | None = None,
    ):
        try:
            super().__init__(model=model)

            self.model = model
            self.step_prompt = step_prompt
            self.llm = ModuleLLM(
                llm_model=llm_model,
                fallback_model=fallback_model,
                system_prompt=system_prompt,
                api_base=api_base,
            )

            self.memory = STLTMemory(
                agent=self,
                short_term_capacity=5,
                consolidation_capacity=2,
                llm_model=llm_model,
                fallback_model=fallback_model,
                api_base=api_base,
            )

            self._tool_manager = ToolManager(tools=tools)
            self._action_manager = ActionManager(actions=actions)
            self.vision = vision
            self.reasoning = reasoning(agent=self)
            self.system_prompt = system_prompt
            self._current_plan = None  # Store current plan for formatting

            # display coordination
            self._step_display_data = {}

            # Placeholder so @record_model can attach the SimulationRecorder
            self.recorder: SimulationRecorder | None = None

            if isinstance(internal_state, str):
                internal_state = [internal_state]
            elif internal_state is None:
                internal_state = []

            self.internal_state = internal_state
        except Exception as initialization_error:
            try:
                LLMAgent._remove_after_failed_initialization(self, model)
            except Exception as cleanup_error:
                initialization_error.add_note(
                    "Failed to remove the partially initialized LLMAgent: "
                    f"{cleanup_error!r}"
                )
            raise

    def _remove_after_failed_initialization(self, model: Model) -> None:
        if any(agent is self for agent in model.agents):
            super().remove()

    def __str__(self):
        return f"LLMAgent {self.unique_id}"

    @property
    def system_prompt(self) -> str | None:
        return self.llm.system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str | None):
        self.llm.system_prompt = value

    @property
    def tool_manager(self) -> ToolManager:
        warnings.warn(
            "`agent.tool_manager` is deprecated; configure tools with "
            "`LLMAgent(..., tools=...)` and treat the manager as internal.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._tool_manager

    @tool_manager.setter
    def tool_manager(self, value: ToolManager):
        warnings.warn(
            "`agent.tool_manager = ...` is deprecated; pass tools explicitly "
            "or assign `agent._tool_manager` only inside framework internals.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._tool_manager = value

    def plan(
        self,
        prompt: str | None = None,
        obs: Observation | None = None,
        ttl: int = 1,
        tools: ToolSelection | object = _UNSET,
        tool_calls: str | None = "auto",
        selected_tools: ToolSelection | object = _UNSET,
    ) -> Plan:
        """Run the configured reasoning module without exposing actions."""
        plan_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "obs": obs,
            "ttl": ttl,
            "tools": tools,
            "tool_calls": tool_calls,
        }
        if selected_tools is not _UNSET:
            plan_kwargs["selected_tools"] = selected_tools
        return self.reasoning.plan(**plan_kwargs)

    def act(
        self,
        prompt: str | list[str],
        actions: ActionSelection | object = _ACTIONS_UNSET,
        system_prompt: str | None = None,
    ) -> ActResult:
        """Choose one action, execute it, and return the successful result."""
        action_choice = self.choose_action(
            prompt,
            actions=actions,
            system_prompt=system_prompt,
        )
        result = self.execute_action(action_choice, actions=actions)
        return ActResult(action=action_choice, result=result)

    async def aact(
        self,
        prompt: str | list[str],
        actions: ActionSelection | object = _ACTIONS_UNSET,
        system_prompt: str | None = None,
    ) -> ActResult:
        """Asynchronously choose one action, execute it, and return the result."""
        action_choice = await self.achoose_action(
            prompt,
            actions=actions,
            system_prompt=system_prompt,
        )
        result = await self.aexecute_action(action_choice, actions=actions)
        return ActResult(action=action_choice, result=result)

    def execute_action(
        self,
        action_choice: ActionChoice | dict[str, Any],
        actions: ActionSelection | object = _ACTIONS_UNSET,
    ) -> Any:
        """Validate and execute one configured action locally."""
        validated_choice = self._action_manager.validate(
            self,
            action_choice,
            actions=actions,
        )
        result = self._action_manager.execute(self, validated_choice, actions=actions)
        self._record_successful_action_event(validated_choice, result)
        return result

    async def aexecute_action(
        self,
        action_choice: ActionChoice | dict[str, Any],
        actions: ActionSelection | object = _ACTIONS_UNSET,
    ) -> Any:
        """Validate and asynchronously execute one configured action locally."""
        validated_choice = self._action_manager.validate(
            self,
            action_choice,
            actions=actions,
        )
        result = await self._action_manager.aexecute(
            self,
            validated_choice,
            actions=actions,
        )
        await self._arecord_successful_action_event(validated_choice, result)
        return result

    def choose_action(
        self,
        prompt: str | list[str],
        actions: ActionSelection | object = _ACTIONS_UNSET,
        system_prompt: str | None = None,
    ) -> ActionChoice:
        """Choose one structured action without executing or mutating state."""
        action_schemas = self._action_manager.get_actions_schema(
            agent=self,
            actions=actions,
        )
        if not action_schemas:
            raise ValueError(
                "No actions are available for this call. Configure actions on "
                "the agent or pass a non-empty configured action selector."
            )

        action_prompt = self._build_action_choice_prompt(prompt, action_schemas)
        with self.llm._one_shot_completion():
            response = self.llm.generate(
                prompt=action_prompt,
                tool_schema=None,
                tool_choice="none",
                response_format=self._action_choice_response_format(),
                system_prompt=system_prompt,
                suppress_thinking=True,
            )
        action_choice = self._parse_action_choice_response(response)
        return self._action_manager.validate(self, action_choice, actions=actions)

    async def achoose_action(
        self,
        prompt: str | list[str],
        actions: ActionSelection | object = _ACTIONS_UNSET,
        system_prompt: str | None = None,
    ) -> ActionChoice:
        """Asynchronously choose one structured action without mutating state."""
        action_schemas = self._action_manager.get_actions_schema(
            agent=self,
            actions=actions,
        )
        if not action_schemas:
            raise ValueError(
                "No actions are available for this call. Configure actions on "
                "the agent or pass a non-empty configured action selector."
            )

        action_prompt = self._build_action_choice_prompt(prompt, action_schemas)
        with self.llm._one_shot_completion():
            response = await self.llm.agenerate(
                prompt=action_prompt,
                tool_schema=None,
                tool_choice="none",
                response_format=self._action_choice_response_format(),
                system_prompt=system_prompt,
                suppress_thinking=True,
            )
        action_choice = self._parse_action_choice_response(response)
        return self._action_manager.validate(self, action_choice, actions=actions)

    def _action_choice_response_format(self) -> dict[str, str] | type[ActionChoice]:
        if self.llm.llm_model.startswith("openai/"):
            return {"type": "json_object"}
        return ActionChoice

    def _build_action_choice_prompt(
        self,
        prompt: str | list[str],
        action_schemas: list[dict[str, Any]],
    ) -> list[str]:
        action_context = (
            "Choose exactly one action from the available action specs below. "
            "Return exactly one JSON object matching this shape: "
            '{"name": str, "arguments": object, "rationale": str | null}. '
            "The `name` value must be one of the listed action names. "
            "The `arguments` value must be a JSON object for that action. "
            "The `arguments` object may contain only properties declared for "
            "the selected action. "
            "Start your response with `{` and end it with `}`. "
            "Do not include any other JSON objects. "
            "Do not include Markdown fences, prose, or hidden reasoning. "
            "Do not call tools and do not execute the action.\n\n"
            f"Available actions:\n{json.dumps(action_schemas, indent=2)}"
        )
        output_contract = (
            "Respond now with exactly one JSON object and nothing else. "
            "Required keys: `name`, `arguments`, `rationale`."
        )
        if isinstance(prompt, str):
            return [action_context, prompt, output_contract]
        if isinstance(prompt, list):
            return [action_context, *prompt, output_contract]
        raise TypeError(
            f"Invalid prompt type '{type(prompt).__name__}'. Expected str or list[str]."
        )

    def _parse_action_choice_response(self, response: Any) -> ActionChoice:
        message = response.choices[0].message
        missing_content = object()
        content = getattr(message, "content", missing_content)
        has_raw_content = content is not missing_content and content is not None
        if isinstance(content, str):
            has_raw_content = bool(content.strip())

        if has_raw_content:
            return self._action_choice_from_content(content)

        parsed = getattr(message, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, ActionChoice):
                return parsed
            if isinstance(parsed, dict):
                return ActionChoice(**parsed)
            if hasattr(parsed, "model_dump"):
                return ActionChoice(**parsed.model_dump())

        if content is missing_content:
            content = message
        return self._action_choice_from_content(content)

    def _action_choice_from_content(self, content: Any) -> ActionChoice:
        if isinstance(content, ActionChoice):
            return content
        if isinstance(content, dict):
            return ActionChoice(**content)

        if isinstance(content, str):
            data = self._extract_single_action_choice_json(content)
            return ActionChoice(**data)

        raise TypeError(
            "LLM action choice response must be an ActionChoice, dict, or JSON "
            f"object string, got {type(content).__name__}."
        )

    def _extract_single_action_choice_json(
        self,
        content: str,
    ) -> dict[str, Any]:
        candidates = self._action_choice_json_candidates(content)

        if not candidates:
            raise ValueError(
                "LLM action choice response did not include a JSON object matching "
                "ActionChoice."
            )
        if len(candidates) > 1:
            raise _AmbiguousActionChoiceJSONError(
                "LLM action choice response must include exactly one JSON object "
                "matching ActionChoice."
            )
        return candidates[0]

    def _action_choice_json_candidates(self, content: str) -> list[dict[str, Any]]:
        stripped_content = content.strip()
        if not stripped_content:
            return []

        try:
            data = _load_action_choice_json(stripped_content)
        except json.JSONDecodeError:
            pass
        else:
            if not isinstance(data, dict):
                raise ValueError(
                    "LLM action choice response must be a JSON object matching "
                    "ActionChoice."
                )
            return [data]

        candidates: list[dict[str, Any]] = []
        for object_text in self._iter_json_object_strings(stripped_content):
            data = _load_action_choice_json(object_text)
            if isinstance(data, dict):
                candidates.append(data)
        return candidates

    def _iter_json_object_strings(self, content: str) -> list[str]:
        objects: list[str] = []
        start: int | None = None
        depth = 0
        in_string = False
        escaped = False

        for index, char in enumerate(content):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if depth and char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}":
                if depth == 0:
                    raise ValueError(
                        "LLM action choice response contains an unmatched closing "
                        "brace."
                    )
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(content[start : index + 1])
                    start = None

        if start is not None:
            if in_string:
                detail = (
                    "an unterminated escape sequence in a JSON string"
                    if escaped
                    else "an unterminated JSON string"
                )
            else:
                detail = "an unclosed JSON object"
            raise ValueError(f"LLM action choice response contains {detail}.")

        return objects

    def _record_successful_action_event(
        self,
        action_choice: ActionChoice,
        result: Any,
    ) -> None:
        content = {
            "action": action_choice.model_dump(),
            "result": result,
        }
        observer_errors: dict[str, Exception] = {}

        memory_content: dict[str, Any] | None = None
        try:
            memory_content = copy.deepcopy(content)
        except Exception as error:
            observer_errors["memory"] = error

        recorder = self.recorder
        recorder_content: dict[str, Any] | None = None
        if recorder is not None:
            try:
                recorder_content = copy.deepcopy(content)
            except Exception as error:
                observer_errors["recorder"] = error

        if memory_content is not None:
            try:
                self.memory.add_to_memory(type="action", content=memory_content)
            except Exception as error:
                observer_errors["memory"] = error
        if recorder is not None and recorder_content is not None:
            try:
                recorder.record_event(
                    "action",
                    content=recorder_content,
                    agent_id=self.unique_id,
                    metadata={"source": "LLMAgent.execute_action"},
                )
            except Exception as error:
                observer_errors["recorder"] = error
        if observer_errors:
            raise ActionPostCommitError(action_choice, result, observer_errors)

    async def _arecord_successful_action_event(
        self,
        action_choice: ActionChoice,
        result: Any,
    ) -> None:
        content = {
            "action": action_choice.model_dump(),
            "result": result,
        }
        observer_errors: dict[str, Exception] = {}

        memory_content: dict[str, Any] | None = None
        try:
            memory_content = copy.deepcopy(content)
        except Exception as error:
            observer_errors["memory"] = error

        recorder = self.recorder
        recorder_content: dict[str, Any] | None = None
        if recorder is not None:
            try:
                recorder_content = copy.deepcopy(content)
            except Exception as error:
                observer_errors["recorder"] = error

        if memory_content is not None:
            try:
                await self.memory.aadd_to_memory(
                    type="action",
                    content=memory_content,
                )
            except Exception as error:
                observer_errors["memory"] = error
        if recorder is not None and recorder_content is not None:
            try:
                recorder.record_event(
                    "action",
                    content=recorder_content,
                    agent_id=self.unique_id,
                    metadata={"source": "LLMAgent.execute_action"},
                )
            except Exception as error:
                observer_errors["recorder"] = error
        if observer_errors:
            raise ActionPostCommitError(action_choice, result, observer_errors)

    def _plan_message_recipients(
        self, recipients: list[Agent]
    ) -> list[tuple[int, Agent]]:
        """Return unique non-self recipients in first-occurrence order."""
        planned_recipients = []
        seen_ids = set()
        for recipient in recipients:
            if recipient is self:
                continue
            recipient_id = recipient.unique_id
            if recipient_id in seen_ids:
                continue
            seen_ids.add(recipient_id)
            planned_recipients.append((recipient_id, recipient))
        return planned_recipients

    def _format_message_status(
        self,
        message: str,
        delivered_ids: list[int],
        skipped_ids: list[int],
        *,
        failed_ids: list[int],
        sender_history_failed: bool,
    ) -> str:
        """Format direct-message delivery and sender-history status."""
        status_parts = []
        if delivered_ids:
            status_parts.append(f"sent message {message!r} to {delivered_ids}")
        if skipped_ids:
            status_parts.append(
                f"skipped {skipped_ids} because they have no usable memory method"
            )
        if failed_ids:
            status_parts.append(f"failed to deliver to {failed_ids}")
        if not delivered_ids and not skipped_ids and not failed_ids:
            status_parts.append(
                f"Could not send message {message!r}: no matching recipients found."
            )
        if sender_history_failed:
            status_parts.append(
                "sender history recording failed after recipient delivery"
            )
        return "; ".join(status_parts)

    async def aapply_plan(self, plan: Plan) -> list[dict]:
        """
        Asynchronous version of apply_plan.
        """
        self._current_plan = plan

        plan_tools = getattr(plan, "tools", _UNSET)
        if plan_tools is _UNSET:
            tool_call_resp = await self._tool_manager.acall_tools(
                agent=self, llm_response=plan.llm_plan
            )
        else:
            tool_call_resp = await self._tool_manager.acall_tools(
                agent=self, llm_response=plan.llm_plan, tools=plan_tools
            )

        await self.memory.aadd_to_memory(
            type="action",
            content={
                "tool_calls": [
                    {k: v for k, v in tc.items() if k not in ["tool_call_id", "role"]}
                    for tc in tool_call_resp
                ]
            },
        )

        return tool_call_resp

    def apply_plan(self, plan: Plan) -> list[dict]:
        """
        Execute the plan in the simulation.
        """
        # Store current plan for display
        self._current_plan = plan

        # Execute tool calls
        plan_tools = getattr(plan, "tools", _UNSET)
        if plan_tools is _UNSET:
            tool_call_resp = self._tool_manager.call_tools(
                agent=self, llm_response=plan.llm_plan
            )
        else:
            tool_call_resp = self._tool_manager.call_tools(
                agent=self, llm_response=plan.llm_plan, tools=plan_tools
            )

        # Add to memory
        self.memory.add_to_memory(
            type="action",
            content={
                "tool_calls": [
                    {k: v for k, v in tc.items() if k not in ["tool_call_id", "role"]}
                    for tc in tool_call_resp
                ]
            },
        )

        return tool_call_resp

    def _build_observation(self):
        """
        Construct the observation data visible to the agent at the current model step.

        This method encapsulates the shared logic used by both sync and
        async observation generation.
        This method constructs the agent's self state and determines which other
        agents are observable based on the configured vision:

        - vision > 0:
            The agent observes all agents within the specified vision radius.
        - vision == -1:
            The agent observes all agents present in the simulation.
        - vision == 0 or vision is None:
            The agent observes no other agents.

        The method supports grid-based and continuous spaces and builds a local
        state representation for all visible neighboring agents.

        Returns self_state and local_state of the agent
        """
        self_state = {
            "agent_unique_id": self.unique_id,
            "location": (
                self.pos
                if self.pos is not None
                else (
                    getattr(self, "cell", None).coordinate
                    if getattr(self, "cell", None) is not None
                    else None
                )
            ),
            "internal_state": self.internal_state,
        }
        if self.vision is not None and self.vision > 0:
            # Early return: agent has no position and no cell — cannot query neighbors
            if self.pos is None and getattr(self, "cell", None) is None:
                return self_state, {}

            # Check which type of space/grid the model uses
            grid = getattr(self.model, "grid", None)
            space = getattr(self.model, "space", None)

            if grid and isinstance(grid, SingleGrid | MultiGrid):
                neighbors = grid.get_neighbors(
                    tuple(self.pos),
                    moore=True,
                    include_center=False,
                    radius=self.vision,
                )
            elif grid and isinstance(
                grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid
            ):
                agent_cell = next(
                    (cell for cell in grid.all_cells if self in cell.agents),
                    None,
                )
                if agent_cell:
                    neighborhood = agent_cell.get_neighborhood(radius=self.vision)
                    neighbors = [a for cell in neighborhood for a in cell.agents]
                else:
                    neighbors = []

            elif space and isinstance(space, ContinuousSpace):
                all_nearby = space.get_neighbors(
                    self.pos, radius=self.vision, include_center=True
                )
                neighbors = [a for a in all_nearby if a is not self]

            else:
                # No recognized grid/space type
                neighbors = []

        elif self.vision == -1:
            all_agents = list(self.model.agents)
            neighbors = [agent for agent in all_agents if agent is not self]

        else:
            neighbors = []

        local_state = {}
        for i in neighbors:
            local_state[i.__class__.__name__ + " " + str(i.unique_id)] = {
                "position": (
                    i.pos
                    if i.pos is not None
                    else (
                        getattr(i, "cell", None).coordinate
                        if getattr(i, "cell", None) is not None
                        else None
                    )
                ),
                "internal_state": [
                    s for s in getattr(i, "internal_state", []) if not s.startswith("_")
                ],
            }
        return self_state, local_state

    async def agenerate_obs(self) -> Observation:
        """
        This method builds the agent's observation using the shared observation
        construction logic, stores it in the agent's memory module using
        async memory operations, and returns it as an Observation instance.
        """
        step = self.model.steps
        self_state, local_state = self._build_observation()
        await self.memory.aadd_to_memory(
            type="observation",
            content={
                "self_state": self_state,
                "local_state": local_state,
            },
        )

        return Observation(step=step, self_state=self_state, local_state=local_state)

    def generate_obs(self) -> Observation:
        """
        This method delegates observation construction to the shared observation
        builder, stores the resulting observation in the agent's memory module,
        and returns it as an Observation instance.
        """
        step = self.model.steps
        self_state, local_state = self._build_observation()
        # Add to memory (memory handles its own display separately)
        self.memory.add_to_memory(
            type="observation",
            content={
                "self_state": self_state,
                "local_state": local_state,
            },
        )

        return Observation(step=step, self_state=self_state, local_state=local_state)

    async def asend_message(self, message: str, recipients: list[Agent]) -> str:
        """
        Asynchronous version of send_message.
        """
        delivered_ids = []
        skipped_ids = []
        failed_ids = []
        planned_recipients = self._plan_message_recipients(recipients)
        for recipient_id, recipient in planned_recipients:
            try:
                try:
                    recipient_memory = recipient.memory
                except AttributeError:
                    try:
                        inspect.getattr_static(recipient, "memory")
                    except AttributeError:
                        recipient_memory = None
                    else:
                        raise
                if recipient_memory is None:
                    skipped_ids.append(recipient_id)
                    logger.warning(
                        "Recipient %s has no usable async memory method for "
                        "send_message; "
                        "skipping delivery from sender %s.",
                        recipient_id,
                        self.unique_id,
                    )
                    continue
                try:
                    writer = recipient_memory.aadd_to_memory
                except AttributeError:
                    try:
                        inspect.getattr_static(recipient_memory, "aadd_to_memory")
                    except AttributeError:
                        writer = None
                    else:
                        raise
                if not callable(writer):
                    skipped_ids.append(recipient_id)
                    logger.warning(
                        "Recipient %s has no usable async memory method for "
                        "send_message; "
                        "skipping delivery from sender %s.",
                        recipient_id,
                        self.unique_id,
                    )
                    continue
                await writer(
                    type="message",
                    content={
                        "message": message,
                        "sender": self.unique_id,
                    },
                )
            except Exception:
                failed_ids.append(recipient_id)
                logger.exception(
                    "Message delivery from sender %s to recipient %s failed.",
                    self.unique_id,
                    recipient_id,
                )
                continue
            delivered_ids.append(recipient_id)

        sender_history_failed = False
        try:
            sender_memory = self.memory
            if sender_memory is None:
                raise AttributeError("Sender has no memory.")
            sender_writer = sender_memory.aadd_to_memory
            if not callable(sender_writer):
                raise TypeError("Sender async memory writer is not callable.")
            await sender_writer(
                type="message",
                content={
                    "message": message,
                    "sender": self.unique_id,
                    "recipients": delivered_ids.copy(),
                },
            )
        except Exception:
            sender_history_failed = True
            logger.exception(
                "Sender %s history recording failed after recipient delivery.",
                self.unique_id,
            )

        return self._format_message_status(
            message,
            delivered_ids,
            skipped_ids,
            failed_ids=failed_ids,
            sender_history_failed=sender_history_failed,
        )

    def send_message(self, message: str, recipients: list[Agent]) -> str:
        """
        Send a message to the recipients.
        """
        delivered_ids = []
        skipped_ids = []
        failed_ids = []
        planned_recipients = self._plan_message_recipients(recipients)
        for recipient_id, recipient in planned_recipients:
            try:
                try:
                    recipient_memory = recipient.memory
                except AttributeError:
                    try:
                        inspect.getattr_static(recipient, "memory")
                    except AttributeError:
                        recipient_memory = None
                    else:
                        raise
                if recipient_memory is None:
                    skipped_ids.append(recipient_id)
                    logger.warning(
                        "Recipient %s has no usable sync memory method for "
                        "send_message; "
                        "skipping delivery from sender %s.",
                        recipient_id,
                        self.unique_id,
                    )
                    continue
                try:
                    writer = recipient_memory.add_to_memory
                except AttributeError:
                    try:
                        inspect.getattr_static(recipient_memory, "add_to_memory")
                    except AttributeError:
                        writer = None
                    else:
                        raise
                if not callable(writer):
                    skipped_ids.append(recipient_id)
                    logger.warning(
                        "Recipient %s has no usable sync memory method for "
                        "send_message; "
                        "skipping delivery from sender %s.",
                        recipient_id,
                        self.unique_id,
                    )
                    continue
                writer(
                    type="message",
                    content={
                        "message": message,
                        "sender": self.unique_id,
                    },
                )
            except Exception:
                failed_ids.append(recipient_id)
                logger.exception(
                    "Message delivery from sender %s to recipient %s failed.",
                    self.unique_id,
                    recipient_id,
                )
                continue
            delivered_ids.append(recipient_id)

        sender_history_failed = False
        try:
            sender_memory = self.memory
            if sender_memory is None:
                raise AttributeError("Sender has no memory.")
            sender_writer = sender_memory.add_to_memory
            if not callable(sender_writer):
                raise TypeError("Sender sync memory writer is not callable.")
            sender_writer(
                type="message",
                content={
                    "message": message,
                    "sender": self.unique_id,
                    "recipients": delivered_ids.copy(),
                },
            )
        except Exception:
            sender_history_failed = True
            logger.exception(
                "Sender %s history recording failed after recipient delivery.",
                self.unique_id,
            )

        return self._format_message_status(
            message,
            delivered_ids,
            skipped_ids,
            failed_ids=failed_ids,
            sender_history_failed=sender_history_failed,
        )

    async def apre_step(self):
        """
        Asynchronous version of pre_step.
        """
        await self.memory.aprocess_step(pre_step=True)

    async def apost_step(self):
        """
        Asynchronous version of post_step.
        """
        await self.memory.aprocess_step()

    def pre_step(self):
        """
        This is some code that is executed before the step method of the child agent is called.
        """
        self.memory.process_step(pre_step=True)

    def post_step(self):
        """
        This is some code that is executed after the step method of the child agent is called.
        It functions because of the __init_subclass__ method that creates a wrapper around the step method of the child agent.
        """
        self.memory.process_step()

    async def astep(self):
        """
        Default asynchronous step method for parallel agent execution.
        Subclasses should override this method for custom async behavior.
        If not overridden, falls back to calling the synchronous step() method.
        """

        async def run_raw_step():
            raw_step = getattr(self.__class__, "_raw_user_step", None)
            if raw_step is None:
                return None
            if not getattr(self.__class__, "_warned_sync_astep_fallback", False):
                warnings.warn(
                    (
                        f"{self.__class__.__name__}.astep() is falling back to "
                        "synchronous step(), which may block the asyncio event "
                        "loop. Override astep() for non-blocking behavior or use "
                        "threading parallel stepping."
                    ),
                    RuntimeWarning,
                    stacklevel=4,
                )
                self.__class__._warned_sync_astep_fallback = True
            return raw_step(self)

        await self.apre_step()
        return await _arun_step_body_with_post(
            run_raw_step,
            self.apost_step,
            "apost_step",
        )

    def __init_subclass__(cls, **kwargs):
        """
        Wrapper - allows to automatically integrate code to be executed after the step method of the child agent (created by the user) is called.
        """
        super().__init_subclass__(**kwargs)
        # only wrap if subclass actually defines its own step
        user_step = cls.__dict__.get("step")
        user_astep = cls.__dict__.get("astep")

        if user_step:
            # Store the raw user step for the default astep() to call
            # without the pre/post wrapper (astep handles its own pre/post)
            cls._raw_user_step = user_step

            def wrapped(self, *args, **kwargs):
                """
                This is the wrapper that is used to integrate the pre_step and post_step methods into the step method of the child agent.
                """
                LLMAgent.pre_step(self, *args, **kwargs)
                return _run_step_body_with_post(
                    lambda: user_step(self, *args, **kwargs),
                    lambda: LLMAgent.post_step(self, *args, **kwargs),
                    "post_step",
                )

            cls.step = wrapped

        if user_astep:

            async def awrapped(self, *args, **kwargs):
                """
                Async wrapper for astep method.
                """
                await self.apre_step()
                return await _arun_step_body_with_post(
                    lambda: user_astep(self, *args, **kwargs),
                    self.apost_step,
                    "apost_step",
                )

            cls.astep = awrapped
