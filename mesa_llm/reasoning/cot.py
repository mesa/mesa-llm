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


class CoTReasoning(Reasoning):
    """
    Chain of Thought reasoning with explicit step-by-step analysis before action execution. Uses structured numbered thoughts followed by tool execution. Integrates memory context for informed decision-making.

    Attributes:
        - **agent** (LLMAgent reference)

    Methods:
        - **plan(obs, ttl=1, prompt=None, tools=<inherit>, tool_calls="auto")** → *Plan* - Generate synchronous plan with CoT reasoning
        - **async aplan(obs, ttl=1, prompt=None, tools=<inherit>, tool_calls="auto")** → *Plan* - Generate asynchronous plan with CoT reasoning

    Reasoning Format:
        Thought 1: [Initial reasoning based on observation]
        Thought 2: [How memory informs the situation]
        Thought 3: [Possible alternatives or risks]
        Thought 4: [Final decision and justification]
        Action: [The action you decide to take]
    """

    def __init__(self, agent: "LLMAgent"):
        super().__init__(agent=agent)

    def get_cot_system_prompt(self, obs: Observation) -> str:
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

        obs_str = str(obs)

        system_prompt = f"""
        You are an autonomous agent operating in a simulation.
        Use a detailed step-by-step reasoning process (Chain-of-Thought) to decide your next action.
        Your memory contains information from past experiences, and your observation provides the current context.
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
        {obs_str}

        ---

        # Instructions
        First think through the situation step-by-step, and explain it in the format given below.
        ------------------------------------------------------
        Thought 1: [Initial reasoning based on the observation]
        Thought 2: [How memory informs the situation]
        Thought 3: [Possible alternatives or risks]
        Thought 4: [Final decision and justification]
        Action: [The action you decide to take]
        ------------------------------------------------------
        Keep the reasoning grounded in the current context and relevant history.


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
        Plan the next (CoT) action based on the current observation and the
        agent's memory.

        ``tools`` controls provider tool exposure. Omitting it inherits the
        agent's configured tools. Explicit ``None`` or ``[]`` exposes no tools.
        A callable, string name, or sequence exposes exactly those configured
        tools.

        ``tool_calls`` controls the execution-phase LiteLLM ``tool_choice``.
        The reasoning pass still keeps tool use disabled with ``"none"``.

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
        # If no prompt is provided, use the agent's default step prompt
        if prompt is None:
            if self.agent.step_prompt is not None:
                prompt = self.agent.step_prompt
            else:
                raise ValueError("No prompt provided and agent.step_prompt is None.")

        if obs is None:
            obs = self.agent.generate_obs()

        tools = self._resolve_tools_argument(tools, selected_tools)
        llm = self.agent.llm
        system_prompt = self.get_cot_system_prompt(obs)

        rsp = llm.generate(
            prompt=prompt,
            tool_schema=self._get_tools_schema(tools),
            tool_choice="none",
            system_prompt=system_prompt,
        )

        chaining_message = rsp.choices[0].message.content
        self.agent.memory.add_to_memory(
            type="plan", content={"content": chaining_message}
        )

        # Pass plan content to agent for display
        if hasattr(self.agent, "_step_display_data"):
            self.agent._step_display_data["plan_content"] = chaining_message
        execute_kwargs = {"ttl": ttl, "tool_calls": tool_calls}
        if tools is not _UNSET:
            execute_kwargs["tools"] = tools
        cot_plan = self.execute_tool_call(chaining_message, **execute_kwargs)

        return cot_plan

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
        The reasoning pass still keeps tool use disabled with ``"none"``.

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
        # If no prompt is provided, use the agent's default step prompt
        if prompt is None:
            if self.agent.step_prompt is not None:
                prompt = self.agent.step_prompt
            else:
                raise ValueError("No prompt provided and agent.step_prompt is None.")

        if obs is None:
            obs = await self.agent.agenerate_obs()

        tools = self._resolve_tools_argument(tools, selected_tools)
        llm = self.agent.llm
        system_prompt = self.get_cot_system_prompt(obs)

        rsp = await llm.agenerate(
            prompt=prompt,
            tool_schema=self._get_tools_schema(tools),
            tool_choice="none",
            system_prompt=system_prompt,
        )

        chaining_message = rsp.choices[0].message.content
        await self.agent.memory.aadd_to_memory(
            type="plan", content={"content": chaining_message}
        )

        # Pass plan content to agent for display
        if hasattr(self.agent, "_step_display_data"):
            self.agent._step_display_data["plan_content"] = chaining_message
        execute_kwargs = {"ttl": ttl, "tool_calls": tool_calls}
        if tools is not _UNSET:
            execute_kwargs["tools"] = tools
        cot_plan = await self.aexecute_tool_call(chaining_message, **execute_kwargs)

        return cot_plan
