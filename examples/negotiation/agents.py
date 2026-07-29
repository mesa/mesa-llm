import json

from mesa_llm.actions import social_actions, teleport_to_location
from mesa_llm.llm_agent import LLMAgent


def _memory_event_payloads(agent, event_type: str):
    """Yield structured recent-memory payloads for one event type."""
    memory = agent.memory
    entries = getattr(memory, "short_term_memory", None)
    if entries is None:
        entries = getattr(memory, "memory_entries", ())

    contents = [entry.content for entry in entries]
    step_content = getattr(memory, "step_content", None)
    if isinstance(step_content, dict):
        contents.append(step_content)

    for content in contents:
        if not isinstance(content, dict) or event_type not in content:
            continue
        payloads = content[event_type]
        if not isinstance(payloads, list):
            payloads = [payloads]
        yield from (payload for payload in payloads if isinstance(payload, dict))


def _recent_dialogue_partner_ids(agent) -> set:
    """Return structured sender/recipient IDs from recent dialogue events."""
    partner_ids = set()

    for message in _memory_event_payloads(agent, "message"):
        sender = message.get("sender")
        if hasattr(sender, "unique_id"):
            sender = sender.unique_id
        if sender is not None and not isinstance(sender, bool):
            partner_ids.add(sender)

        recipients = message.get("recipients", ())
        if isinstance(recipients, list | tuple):
            partner_ids.update(
                recipient for recipient in recipients if not isinstance(recipient, bool)
            )

    for event in _memory_event_payloads(agent, "action"):
        action_choice = event.get("action")
        if (
            not isinstance(action_choice, dict)
            or action_choice.get("name") != "speak_to"
        ):
            continue
        arguments = action_choice.get("arguments")
        if not isinstance(arguments, dict):
            continue
        recipients = arguments.get("listener_agents_unique_ids", ())
        if isinstance(recipients, list | tuple):
            partner_ids.update(
                recipient for recipient in recipients if not isinstance(recipient, bool)
            )

    return partner_ids


def get_eligible_recipients(agent, observation, recipient_type: str) -> list[dict]:
    """Return visible and recent dialogue partners with explicit integer IDs."""
    visible_labels = set(observation.local_state)
    recent_partner_ids = _recent_dialogue_partner_ids(agent)
    recipients = []

    for candidate in agent.model.agents:
        label = f"{type(candidate).__name__} {candidate.unique_id}"
        is_visible = label in visible_labels
        is_recent_partner = candidate.unique_id in recent_partner_ids
        if (
            candidate is agent
            or type(candidate).__name__ != recipient_type
            or not (is_visible or is_recent_partner)
        ):
            continue
        recipients.append(
            {
                "label": label,
                "unique_id": int(candidate.unique_id),
                "currently_visible": is_visible,
                "recent_dialogue_partner": is_recent_partner,
            }
        )

    return sorted(recipients, key=lambda recipient: recipient["unique_id"])


def get_recipient_prompt(recipients: list[dict]) -> str:
    """Format visible identities and the strict speak_to argument contract."""
    if not recipients:
        return (
            "Eligible speak_to recipients: []. No eligible recipient is available. "
            "Do not invent a recipient ID. If speak_to is the only available action, "
            "set listener_agents_unique_ids to []. "
        )

    example_recipient = recipients[0]
    example_ids = json.dumps([example_recipient["unique_id"]])
    return (
        f"Eligible speak_to recipients: {json.dumps(recipients)}. "
        "When using speak_to, set listener_agents_unique_ids to a JSON list "
        "of the raw integer unique_id values shown above. For example, to target "
        f"{example_recipient['label']}, use {example_ids}. "
        "Never pass agent labels or names. "
    )


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

    def _seller_step_prompt(self, observation, dialogue_history):
        eligible_buyers = get_eligible_recipients(self, observation, "BuyerAgent")
        return (
            f"DIALOGUE HISTORY:\n{dialogue_history}\n\n"
            "INSTRUCTIONS:\n"
            f"{get_recipient_prompt(eligible_buyers)}"
            "Don't move around. If there are any buyers in your cell or in the neighboring cells, "
            "pitch them your product using the speak_to action. "
            "Talk to them until they agree or definitely refuse to buy your product. "
            "Use the dialogue history to inform your next response (e.g., if you already offered a price, stick to it or negotiate)."
        )

    def step(self):
        observation = self.generate_obs()
        dialogue_history = get_dialogue_history(self)
        prompt = self._seller_step_prompt(observation, dialogue_history)

        self.act(
            prompt=[f"OBSERVATION:\n{observation}", prompt],
            actions=["speak_to"],
        )

    async def astep(self):
        observation = self.generate_obs()
        dialogue_history = get_dialogue_history(self)
        prompt = self._seller_step_prompt(observation, dialogue_history)

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
        eligible_sellers = get_eligible_recipients(self, observation, "SellerAgent")
        visible_sellers = [
            seller for seller in eligible_sellers if seller["currently_visible"]
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
                get_recipient_prompt(eligible_sellers)
                if eligible_sellers
                else get_recipient_prompt([])
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
