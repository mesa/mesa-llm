from types import SimpleNamespace

from examples.negotiation.agents import (
    BuyerAgent,
    SellerAgent,
    get_eligible_recipients,
    get_recipient_prompt,
)


def _example_agent(agent_type: str, unique_id: int):
    agent_class = type(agent_type, (), {})
    agent = agent_class()
    agent.unique_id = unique_id
    return agent


def _memory(*contents, step_content=None):
    return SimpleNamespace(
        short_term_memory=[SimpleNamespace(content=content) for content in contents],
        step_content=step_content,
    )


def _observation(*visible_labels: str):
    return SimpleNamespace(local_state={label: {} for label in visible_labels})


def test_eligible_recipients_resolve_structured_ids_against_model_agents():
    self_agent = _example_agent("SellerAgent", 50)
    visible_seller = _example_agent("SellerAgent", 9)
    recent_sender = _example_agent("SellerAgent", 3)
    recent_message_recipient = _example_agent("SellerAgent", 7)
    recent_action_recipient = _example_agent("SellerAgent", 5)
    prose_only_seller = _example_agent("SellerAgent", 11)
    wrong_type = _example_agent("BuyerAgent", 4)
    self_agent.model = SimpleNamespace(
        agents=[
            visible_seller,
            self_agent,
            prose_only_seller,
            wrong_type,
            recent_message_recipient,
            recent_sender,
            recent_action_recipient,
        ]
    )
    self_agent.memory = _memory(
        {
            "message": {
                "sender": recent_sender.unique_id,
                "recipients": [
                    recent_message_recipient.unique_id,
                    recent_sender.unique_id,
                    wrong_type.unique_id,
                    99,
                    True,
                    "SellerAgent 11",
                ],
            }
        },
        {
            "action": {
                "action": {
                    "name": "speak_to",
                    "arguments": {
                        "listener_agents_unique_ids": [
                            recent_action_recipient.unique_id,
                            recent_message_recipient.unique_id,
                        ]
                    },
                }
            }
        },
        {"message": {"sender": "SellerAgent 11", "message": "prose only"}},
    )
    observation = _observation(
        "SellerAgent 9",
        "SellerAgent 50",
        "BuyerAgent 4",
        "SellerAgent 99",
    )

    recipients = get_eligible_recipients(
        self_agent,
        observation,
        "SellerAgent",
    )

    assert recipients == [
        {
            "label": "SellerAgent 3",
            "unique_id": 3,
            "currently_visible": False,
            "recent_dialogue_partner": True,
        },
        {
            "label": "SellerAgent 5",
            "unique_id": 5,
            "currently_visible": False,
            "recent_dialogue_partner": True,
        },
        {
            "label": "SellerAgent 7",
            "unique_id": 7,
            "currently_visible": False,
            "recent_dialogue_partner": True,
        },
        {
            "label": "SellerAgent 9",
            "unique_id": 9,
            "currently_visible": True,
            "recent_dialogue_partner": False,
        },
    ]
    assert all(type(recipient["unique_id"]) is int for recipient in recipients)


def test_recipient_prompt_uses_first_actual_eligible_id_as_example():
    recipients = [
        {
            "label": "BuyerAgent 42",
            "unique_id": 42,
            "currently_visible": True,
            "recent_dialogue_partner": False,
        }
    ]

    prompt = get_recipient_prompt(recipients)

    assert "to target BuyerAgent 42, use [42]" in prompt
    assert "use [1]" not in prompt
    assert "raw integer unique_id values" in prompt
    assert "Never pass agent labels or names" in prompt


def test_empty_recipient_prompt_requires_empty_ids_and_forbids_invention():
    prompt = get_recipient_prompt([])

    assert "Eligible speak_to recipients: []" in prompt
    assert "set listener_agents_unique_ids to []" in prompt
    assert "Do not invent a recipient ID" in prompt
    assert "use [1]" not in prompt


def test_seller_prompt_uses_visible_buyer_integer_id_dynamically():
    buyer = _example_agent("BuyerAgent", 42)
    seller = SimpleNamespace(
        model=SimpleNamespace(agents=[buyer]),
        memory=_memory(),
    )
    observation = _observation("BuyerAgent 42")

    prompt = SellerAgent._seller_step_prompt(
        seller,
        observation,
        "No recent dialogue.",
    )

    assert '"label": "BuyerAgent 42", "unique_id": 42' in prompt
    assert "to target BuyerAgent 42, use [42]" in prompt
    assert "use [1]" not in prompt
    assert "listener_agents_unique_ids" in prompt


def test_buyer_dialogue_without_visible_seller_gets_integer_recipient_mapping():
    seller = _example_agent("SellerAgent", 7)
    buyer = SimpleNamespace(
        budget=50,
        model=SimpleNamespace(agents=[seller]),
        memory=_memory(
            {
                "message": {
                    "sender": seller.unique_id,
                    "message": "The price is 40.",
                }
            }
        ),
    )
    observation = _observation()

    prompt, actions = BuyerAgent._buyer_step_prompt_and_actions(
        buyer,
        observation,
        "- SellerAgent 7: The price is 40.",
    )

    assert '"label": "SellerAgent 7", "unique_id": 7' in prompt
    assert '"currently_visible": false' in prompt
    assert '"recent_dialogue_partner": true' in prompt
    assert "to target SellerAgent 7, use [7]" in prompt
    assert "use [1]" not in prompt
    assert actions == ["speak_to", "buy_product"]


def test_buyer_dialogue_without_structured_partner_forbids_invented_ids():
    seller = _example_agent("SellerAgent", 7)
    buyer = SimpleNamespace(
        budget=50,
        model=SimpleNamespace(agents=[seller]),
        memory=_memory(
            {
                "message": {
                    "sender": "SellerAgent 7",
                    "message": "Unstructured sender label.",
                }
            }
        ),
    )
    observation = _observation()

    prompt, actions = BuyerAgent._buyer_step_prompt_and_actions(
        buyer,
        observation,
        "- SellerAgent 7: Unstructured sender label.",
    )

    assert "Eligible speak_to recipients: []" in prompt
    assert "set listener_agents_unique_ids to []" in prompt
    assert "Do not invent a recipient ID" in prompt
    assert actions == ["speak_to", "buy_product"]
