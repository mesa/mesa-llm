from mesa_llm.actions import social_actions, teleport_to_location
from mesa_llm.llm_agent import LLMAgent


def get_dialogue_history(agent, max_messages: int = 5) -> str:
    """Extract and format recent dialogue from an agent's memory.

    This helper function supports both STLTMemory (short_term_memory) and
    EpisodicMemory (memory_entries). It efficiently extracts the last N
    dialogue messages by iterating in reverse order.

    Args:
        agent: The LLMAgent whose memory to extract dialogue from
        max_messages: Maximum number of dialogue messages to return (default: 5)

    Returns:
        Formatted dialogue history string, or "No recent dialogue." if empty
    """
    dialogue = []

    # Support both STLTMemory and EpisodicMemory
    memory_source = None
    if hasattr(agent.memory, "short_term_memory"):
        memory_source = agent.memory.short_term_memory
    elif hasattr(agent.memory, "memory_entries"):
        memory_source = agent.memory.memory_entries

    if memory_source:
        # Iterate in reverse to efficiently get last N messages
        # We check at most max_messages * 2 recent entries to account for
        # non-dialogue entries (observations, movements, etc.)
        entries_to_check = min(len(memory_source), max_messages * 2)

        for entry in reversed(list(memory_source)[-entries_to_check:]):
            # Stop if we already have enough dialogue messages
            if len(dialogue) >= max_messages:
                break

            # Check if entry.content is a dict and has 'message'
            if isinstance(entry.content, dict) and "message" in entry.content:
                sender = entry.content.get("sender", "Unknown")
                msg = entry.content.get("message", "")

                # Handle both agent objects and agent IDs
                if hasattr(sender, "unique_id"):
                    # sender is an agent object (from send_message())
                    sender_name = f"{type(sender).__name__} {sender.unique_id}"
                elif isinstance(sender, int):
                    # sender is an ID (from speak_to action)
                    # Try to find the agent by ID to get its type
                    try:
                        agent_obj = next(
                            a for a in agent.model.agents if a.unique_id == sender
                        )
                        sender_name = f"{type(agent_obj).__name__} {sender}"
                    except StopIteration:
                        sender_name = f"Agent {sender}"
                else:
                    sender_name = str(sender)

                dialogue.append(f"- {sender_name}: {msg}")

    # Reverse to get chronological order (oldest first)
    dialogue.reverse()
    return "\n".join(dialogue) if dialogue else "No recent dialogue."


class SellerAgent(LLMAgent):
    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        api_base=None,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            api_base=api_base,
            vision=vision,
            internal_state=internal_state,
            actions=social_actions(),
        )

        self.sales = 0

    def step(self):
        observation = self.generate_obs()
        dialogue_history = get_dialogue_history(self)

        prompt = (
            f"DIALOGUE HISTORY:\n{dialogue_history}\n\n"
            "INSTRUCTIONS:\n"
            "Don't move around. If there are any buyers in your cell or in the neighboring cells, "
            "pitch them your product using the speak_to action. "
            "Talk to them until they agree or definitely refuse to buy your product. "
            "Use the dialogue history to inform your next response (e.g., if you already offered a price, stick to it or negotiate)."
        )

        self.act(
            prompt=[f"OBSERVATION:\n{observation}", prompt],
            actions=["speak_to"],
        )

    async def astep(self):
        observation = self.generate_obs()
        dialogue_history = get_dialogue_history(self)

        prompt = (
            f"DIALOGUE HISTORY:\n{dialogue_history}\n\n"
            "INSTRUCTIONS:\n"
            "Don't move around. If there are any buyers in your cell or in the neighboring cells, "
            "pitch them your product using the speak_to action. "
            "Talk to them until they agree or definitely refuse to buy your product. "
            "Use the dialogue history to inform your next response."
        )

        await self.aact(
            prompt=[f"OBSERVATION:\n{observation}", prompt],
            actions=["speak_to"],
        )


class BuyerAgent(LLMAgent):
    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        budget,
        api_base=None,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            api_base=api_base,
            vision=vision,
            internal_state=internal_state,
            actions=[teleport_to_location, *social_actions(), "buy_product"],
        )
        self.budget = budget
        self.products = []

    def _buyer_step_prompt_and_actions(self, observation, dialogue_history):
        visible_sellers = [
            agent_label
            for agent_label in observation.local_state
            if agent_label.startswith("SellerAgent ")
        ]
        has_dialogue = dialogue_history != "No recent dialogue."

        base_prompt = (
            f"DIALOGUE HISTORY:\n{dialogue_history}\n\n"
            "INSTRUCTIONS:\n"
            f"Your budget is ${self.budget}. "
            "Seller agents around you might try to pitch their product by "
            "sending you messages; get as much information as possible. "
            "When you have enough information, decide what product to buy. "
            "Refer to the dialogue history to recall previous prices offered. "
        )

        if visible_sellers or has_dialogue:
            seller_context = (
                f"Visible sellers: {', '.join(visible_sellers)}. "
                if visible_sellers
                else ""
            )
            next_action_instruction = (
                "Use speak_to to ask or answer sellers, or use buy_product if "
                "you are ready to purchase."
                if has_dialogue
                else "Use speak_to to ask a visible seller about their products and prices."
            )
            actions = ["speak_to", "buy_product"] if has_dialogue else ["speak_to"]
            prompt = (
                base_prompt
                + seller_context
                + "A seller or recent seller dialogue is available, so do not "
                f"move this turn. {next_action_instruction}"
            )
            return prompt, actions

        target_x = int(self.model.rng.integers(0, self.model.grid.width))
        target_y = int(self.model.rng.integers(0, self.model.grid.height))
        prompt = (
            base_prompt
            + "No seller is visible yet, so you may explore with teleport_to_location. "
            f"Grid dimensions are {self.model.grid.width} x {self.model.grid.height}; "
            "coordinates must be inside the grid with 0 <= x < width and "
            "0 <= y < height. If you choose teleport_to_location, set "
            f"target_coordinates to exactly [{target_x}, {target_y}]. "
            "Never use null, None, an empty value, or an omitted "
            "target_coordinates value."
        )
        return prompt, ["teleport_to_location"]

    def step(self):
        observation = self.generate_obs()
        dialogue_history = get_dialogue_history(self)
        prompt, actions = self._buyer_step_prompt_and_actions(
            observation, dialogue_history
        )
        self.act(
            prompt=[f"OBSERVATION:\n{observation}", prompt],
            actions=actions,
        )

    async def astep(self):
        observation = self.generate_obs()
        dialogue_history = get_dialogue_history(self)
        prompt, actions = self._buyer_step_prompt_and_actions(
            observation, dialogue_history
        )
        await self.aact(
            prompt=[f"OBSERVATION:\n{observation}", prompt],
            actions=actions,
        )
