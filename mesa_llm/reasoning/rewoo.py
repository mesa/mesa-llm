import copy
from typing import TYPE_CHECKING

from mesa_llm.reasoning.reasoning import (
    _UNSET,
    Observation,
    Plan,
    Reasoning,
    ToolSelection,
)

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class ReWOOReasoning(Reasoning):
    """
    Reasoning Without Observation for multi-step planning without environmental feedback. Enables multi-step planning without requiring immediate environmental feedback. Plans remain valid across multiple simulation steps with extended TTL. Reduces computational overhead through strategic long-term thinking.

    Attributes:
        - **agent** (LLMAgent reference)
        - **remaining_tool_calls** (int) - Number of tool calls remaining in current plan
        - **current_plan** (Plan) - Currently active multi-step plan
        - **current_obs** (Observation) - Last observation used for planning

    Methods:
        - **plan(prompt, obs=None, ttl=1, tools=<inherit>, tool_calls="auto")** → *Plan* - Generate synchronous plan with ReWOO reasoning
        - **async aplan(prompt, obs=None, ttl=1, tools=<inherit>, tool_calls="auto")** → *Plan* - Generate asynchronous plan
    """

    def __init__(self, agent: "LLMAgent"):
        super().__init__(agent=agent)
        self.remaining_tool_calls = 0  # Initialize remaining tool calls
        self.current_plan: Plan | None = None
        self.current_obs: Observation | None = None
        self.current_tools: ToolSelection | object = _UNSET

    def get_rewoo_system_prompt(self, obs: Observation) -> str:
        memory = getattr(self.agent, "memory", None)
        agent_persona = getattr(self.agent, "system_prompt", None)
        persona_section = ""
        if isinstance(agent_persona, str) and agent_persona.strip():
            persona_section = (
                "\n        ---\n\n"
                "        # Agent Persona\n"
                f"        {agent_persona.strip()}\n"
            )

        long_term_memory = ""
        if (
            memory
            and hasattr(memory, "format_long_term")
            and callable(memory.format_long_term)
        ):
            long_term_memory = memory.format_long_term()

        short_term_memory = ""
        if (
            memory
            and hasattr(memory, "format_short_term")
            and callable(memory.format_short_term)
        ):
            short_term_memory = memory.format_short_term()

        memory_section = ""
        if (
            memory is not None
            and not callable(getattr(memory, "format_long_term", None))
            and not callable(getattr(memory, "format_short_term", None))
            and callable(getattr(memory, "get_prompt_ready", None))
        ):
            memory_section = (
                "\n        # Memory\n"
                f"        {memory.get_prompt_ready()}\n\n        ---\n"
            )

        system_prompt = f"""
        You are an autonomous agent that creates multi-step plans without re-observing during execution.
        Using the ReWOO (Reasoning WithOut Observation) approach, you will create a comprehensive plan
        that anticipates multiple steps ahead based on your current observation and memory.
{persona_section}

        ---

        # Long-Term Memory
        {long_term_memory}

        ---

        # Short-Term Memory (Recent History)
        {short_term_memory}

        ---

{memory_section}
        # Current Observation
        {obs}

        ---

        # Instructions
        Create a detailed multi-step plan that can be executed without needing new observations.
        Your plan should anticipate likely scenarios and include contingencies.

        Determine the optimal number of steps (1-5) based on the complexity of the task and available tools.
        Use this format:


            "plan": "Describe your overall strategy and reasoning",
            "step_1": "First action with expected outcome",
            "step_2": "Second action building on Step 1 (optional)",
            "step_3": "Third action if needed (optional)",
            "step_4": "Fourth action if needed (optional)",
            "step_5": "Final action if needed (optional)",
            "contingency": "What to do if things don't go as expected"


        Only include the steps you need (step_1 is required, step_2 through step_5 are optional).
        Set unused step fields to null. The plan should be comprehensive enough to execute
        for multiple simulation steps without requiring new environmental observations.
        Refer to available tools when planning actions.

        ---
        """
        return system_prompt

    def plan(
        self,
        prompt: str | None = None,
        obs: Observation | None = None,
        ttl: int = 1,
        tools: ToolSelection | object = _UNSET,
        tool_calls: str | None = "auto",
        selected_tools: ToolSelection | object = _UNSET,
    ) -> Plan:
        """
        Plan the next (ReWOO) action based on the current observation and the
        agent's memory.

        ``tools`` controls provider tool exposure. Omitting it inherits the
        agent's configured tools. Explicit ``None`` or ``[]`` exposes no tools.
        A callable, string name, or sequence exposes exactly those configured
        tools.

        ``tool_calls`` controls the execution-phase LiteLLM ``tool_choice``.
        The planning pass still keeps tool use disabled with ``"none"``.

        Supported values in Mesa-LLM are:
        - ``None``: defer to LiteLLM/provider default behavior. In practice,
          this usually means no tool calls when no tools are provided and
          behavior similar to ``"auto"`` when tools are available.
        - ``"none"``: never return tool calls; return a normal assistant
          message instead.
        - ``"auto"``: allow the model to either return a normal assistant
          message or call one or more tools.
        - ``"required"``: require the model to call one or more tools.
        """
        # If we have remaining tool calls, skip observation and plan generation
        if self.remaining_tool_calls > 0:
            index_of_tool = (
                len(self.current_plan.tool_calls) - self.remaining_tool_calls
            )
            self.remaining_tool_calls -= 1
            tool_call = [self.current_plan.tool_calls[index_of_tool]]
            current_plan = copy.copy(self.current_plan)
            current_plan.tool_calls = tool_call
            return Plan(
                llm_plan=current_plan,
                step=self.current_obs.step,
                ttl=ttl,
                tools=self.current_tools,
            )

        # If no prompt is provided, use the agent's default step prompt
        if prompt is None:
            if self.agent.step_prompt is not None:
                prompt = self.agent.step_prompt
            else:
                raise ValueError("No prompt provided and agent.step_prompt is None.")

        if obs is None:
            self.current_obs = self.agent.generate_obs()
        else:
            self.current_obs = obs
        llm = self.agent.llm
        system_prompt = self.get_rewoo_system_prompt(self.current_obs)
        tools = self._resolve_tools_argument(tools, selected_tools)

        rsp = llm.generate(
            prompt=prompt,
            tool_schema=self._get_tools_schema(tools),
            tool_choice="none",
            system_prompt=system_prompt,
        )

        self.agent.memory.add_to_memory(
            type="plan", content={"content": rsp.choices[0].message.content}
        )

        execute_kwargs = {"ttl": ttl, "tool_calls": tool_calls}
        if tools is not _UNSET:
            execute_kwargs["tools"] = tools
        rewoo_plan = self.execute_tool_call(
            rsp.choices[0].message.content, **execute_kwargs
        )
        # Count the number of tool calls in the response and set remaining_tool_calls
        self.remaining_tool_calls = len(
            getattr(rewoo_plan.llm_plan, "tool_calls", None) or []
        )
        self.current_plan = rewoo_plan.llm_plan
        self.current_tools = tools

        return rewoo_plan

    async def aplan(
        self,
        prompt: str | None = None,
        obs: Observation | None = None,
        ttl: int = 1,
        tools: ToolSelection | object = _UNSET,
        tool_calls: str | None = "auto",
        selected_tools: ToolSelection | object = _UNSET,
    ) -> Plan:
        """
        Asynchronous version of plan() method for parallel planning.

        ``tools`` follows the same contract as ``plan()``.

        ``tool_calls`` controls the execution-phase LiteLLM ``tool_choice``.
        The planning pass still keeps tool use disabled with ``"none"``.

        Supported values in Mesa-LLM are:
        - ``None``: defer to LiteLLM/provider default behavior. In practice,
          this usually means no tool calls when no tools are provided and
          behavior similar to ``"auto"`` when tools are available.
        - ``"none"``: never return tool calls; return a normal assistant
          message instead.
        - ``"auto"``: allow the model to either return a normal assistant
          message or call one or more tools.
        - ``"required"``: require the model to call one or more tools.
        """
        # If we have remaining tool calls, skip observation and plan generation
        if self.remaining_tool_calls > 0:
            index_of_tool = (
                len(self.current_plan.tool_calls) - self.remaining_tool_calls
            )
            self.remaining_tool_calls -= 1
            tool_call = [self.current_plan.tool_calls[index_of_tool]]
            current_plan = copy.copy(self.current_plan)
            current_plan.tool_calls = tool_call
            return Plan(
                llm_plan=current_plan,
                step=self.current_obs.step,
                ttl=ttl,
                tools=self.current_tools,
            )

        # If no prompt is provided, use the agent's default step prompt
        if prompt is None:
            if self.agent.step_prompt is not None:
                prompt = self.agent.step_prompt
            else:
                raise ValueError("No prompt provided and agent.step_prompt is None.")

        if obs is None:
            self.current_obs = await self.agent.agenerate_obs()
        else:
            self.current_obs = obs
        llm = self.agent.llm
        system_prompt = self.get_rewoo_system_prompt(self.current_obs)
        tools = self._resolve_tools_argument(tools, selected_tools)

        rsp = await llm.agenerate(
            prompt=prompt,
            tool_schema=self._get_tools_schema(tools),
            tool_choice="none",
            system_prompt=system_prompt,
        )

        await self.agent.memory.aadd_to_memory(
            type="plan", content={"content": rsp.choices[0].message.content}
        )

        execute_kwargs = {"ttl": ttl, "tool_calls": tool_calls}
        if tools is not _UNSET:
            execute_kwargs["tools"] = tools
        rewoo_plan = await self.aexecute_tool_call(
            rsp.choices[0].message.content, **execute_kwargs
        )
        # Count the number of tool calls in the response and set remaining_tool_calls
        self.remaining_tool_calls = len(
            getattr(rewoo_plan.llm_plan, "tool_calls", None) or []
        )
        self.current_plan = rewoo_plan.llm_plan
        self.current_tools = tools

        return rewoo_plan
