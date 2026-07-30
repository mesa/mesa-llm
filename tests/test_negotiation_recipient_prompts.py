from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mesa.agent import Agent
from mesa.model import Model

from examples.negotiation.agents import (
    BuyerAgent,
    SellerAgent,
    get_eligible_recipients,
    get_recipient_prompt,
)
from mesa_llm.reasoning.react import ReActReasoning


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


def _seller_agent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    model = Model(rng=42)
    seller = SellerAgent(
        model=model,
        reasoning=ReActReasoning,
        llm_model="openai/test",
        system_prompt="",
        vision=1,
        internal_state=["persuasive"],
    )
    seller.memory.display = False
    return seller


def _add_buyer(model):
    buyer_class = type("BuyerAgent", (Agent,), {})
    return buyer_class(model)


def _forbid_action_selection(seller):
    seller.act = Mock(side_effect=AssertionError("act() must not be called"))
    seller.aact = AsyncMock(side_effect=AssertionError("aact() must not be called"))
    seller.choose_action = Mock(
        side_effect=AssertionError("choose_action() must not be called")
    )
    seller.achoose_action = AsyncMock(
        side_effect=AssertionError("achoose_action() must not be called")
    )
    seller.llm.generate = Mock(
        side_effect=AssertionError("selection LLM must not be called")
    )
    seller.llm.agenerate = AsyncMock(
        side_effect=AssertionError("selection LLM must not be called")
    )
    seller.memory.llm.generate = Mock(
        side_effect=AssertionError("memory LLM must not be called")
    )
    seller.memory.llm.agenerate = AsyncMock(
        side_effect=AssertionError("memory LLM must not be called")
    )


def _assert_wait_only_memory(seller):
    content = seller.memory.short_term_memory[-1].content

    assert "message" not in content
    assert len(content["action"]) == 1
    assert content["action"][0] == {
        "action": {
            "name": "wait",
            "arguments": {},
            "rationale": None,
        },
        "result": "waited",
    }


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
    eligible_buyers = get_eligible_recipients(seller, observation, "BuyerAgent")

    prompt = SellerAgent._seller_step_prompt(
        seller,
        "No recent dialogue.",
        eligible_buyers,
    )

    assert '"label": "BuyerAgent 42", "unique_id": 42' in prompt
    assert "to target BuyerAgent 42, use [42]" in prompt
    assert "use [1]" not in prompt
    assert "listener_agents_unique_ids" in prompt


def test_seller_sync_step_waits_locally_without_eligible_buyer(monkeypatch):
    seller = _seller_agent(monkeypatch)
    seller.generate_obs = Mock(return_value=_observation())
    _forbid_action_selection(seller)
    execute_action = Mock(wraps=seller.execute_action)
    seller.execute_action = execute_action
    state_before = (seller.sales, list(seller.internal_state), seller.model.steps)

    result = seller.step()

    assert result is None
    execute_action.assert_called_once()
    action_choice = execute_action.call_args.args[0]
    assert action_choice.name == "wait"
    assert action_choice.arguments == {}
    assert execute_action.call_args.kwargs == {"actions": ["wait"]}
    assert set(seller._action_manager.available_actions(seller)) == {
        "speak_to",
        "wait",
    }
    seller.act.assert_not_called()
    seller.aact.assert_not_awaited()
    seller.choose_action.assert_not_called()
    seller.achoose_action.assert_not_awaited()
    seller.llm.generate.assert_not_called()
    seller.llm.agenerate.assert_not_awaited()
    seller.memory.llm.generate.assert_not_called()
    seller.memory.llm.agenerate.assert_not_awaited()
    assert (seller.sales, seller.internal_state, seller.model.steps) == state_before
    _assert_wait_only_memory(seller)


@pytest.mark.asyncio
async def test_seller_async_step_awaits_local_wait_without_eligible_buyer(monkeypatch):
    seller = _seller_agent(monkeypatch)
    seller.generate_obs = Mock(return_value=_observation())
    _forbid_action_selection(seller)
    seller.execute_action = Mock(
        side_effect=AssertionError("sync execute_action() must not be called")
    )
    aexecute_action = AsyncMock(wraps=seller.aexecute_action)
    seller.aexecute_action = aexecute_action
    state_before = (seller.sales, list(seller.internal_state), seller.model.steps)

    result = await seller.astep()

    assert result is None
    aexecute_action.assert_awaited_once()
    action_choice = aexecute_action.call_args.args[0]
    assert action_choice.name == "wait"
    assert action_choice.arguments == {}
    assert aexecute_action.call_args.kwargs == {"actions": ["wait"]}
    seller.execute_action.assert_not_called()
    seller.act.assert_not_called()
    seller.aact.assert_not_awaited()
    seller.choose_action.assert_not_called()
    seller.achoose_action.assert_not_awaited()
    seller.llm.generate.assert_not_called()
    seller.llm.agenerate.assert_not_awaited()
    seller.memory.llm.generate.assert_not_called()
    seller.memory.llm.agenerate.assert_not_awaited()
    assert (seller.sales, seller.internal_state, seller.model.steps) == state_before
    _assert_wait_only_memory(seller)


def test_seller_with_visible_buyer_preserves_speak_to_path(monkeypatch):
    seller = _seller_agent(monkeypatch)
    buyer = _add_buyer(seller.model)
    observation = _observation(f"BuyerAgent {buyer.unique_id}")
    seller.generate_obs = Mock(return_value=observation)
    seller.act = Mock(return_value="ignored action result")
    seller.execute_action = Mock(
        side_effect=AssertionError("visible buyer must not select wait")
    )

    result = seller.step()

    assert result is None
    seller.execute_action.assert_not_called()
    seller.act.assert_called_once()
    assert seller.act.call_args.kwargs["actions"] == ["speak_to"]
    assert f'"unique_id": {buyer.unique_id}' in seller.act.call_args.kwargs["prompt"][1]


def test_seller_with_structured_recent_buyer_preserves_speak_to_path(monkeypatch):
    seller = _seller_agent(monkeypatch)
    buyer = _add_buyer(seller.model)
    seller.memory.short_term_memory.append(
        SimpleNamespace(
            content={
                "message": {
                    "sender": buyer.unique_id,
                    "message": "What is your price?",
                }
            }
        )
    )
    seller.generate_obs = Mock(return_value=_observation())
    seller.act = Mock(return_value="ignored action result")
    seller.execute_action = Mock(
        side_effect=AssertionError("recent buyer must not select wait")
    )

    result = seller.step()

    assert result is None
    seller.execute_action.assert_not_called()
    seller.act.assert_called_once()
    assert seller.act.call_args.kwargs["actions"] == ["speak_to"]
    prompt = seller.act.call_args.kwargs["prompt"][1]
    assert f'"unique_id": {buyer.unique_id}' in prompt
    assert '"recent_dialogue_partner": true' in prompt


@pytest.mark.asyncio
async def test_seller_async_visible_buyer_preserves_speak_to_path(monkeypatch):
    seller = _seller_agent(monkeypatch)
    buyer = _add_buyer(seller.model)
    observation = _observation(f"BuyerAgent {buyer.unique_id}")
    seller.generate_obs = Mock(return_value=observation)
    seller.act = Mock(
        side_effect=AssertionError("async seller must not call sync act()")
    )
    seller.aact = AsyncMock(return_value="ignored action result")
    seller.aexecute_action = AsyncMock(
        side_effect=AssertionError("visible buyer must not select wait")
    )

    result = await seller.astep()

    assert result is None
    seller.act.assert_not_called()
    seller.aexecute_action.assert_not_awaited()
    seller.aact.assert_awaited_once()
    assert seller.aact.call_args.kwargs["actions"] == ["speak_to"]
    assert (
        f'"unique_id": {buyer.unique_id}' in seller.aact.call_args.kwargs["prompt"][1]
    )


@pytest.mark.asyncio
async def test_seller_async_structured_recent_buyer_preserves_speak_to_path(
    monkeypatch,
):
    seller = _seller_agent(monkeypatch)
    buyer = _add_buyer(seller.model)
    seller.memory.short_term_memory.append(
        SimpleNamespace(
            content={
                "message": {
                    "sender": buyer.unique_id,
                    "message": "What is your price?",
                }
            }
        )
    )
    seller.generate_obs = Mock(return_value=_observation())
    seller.act = Mock(
        side_effect=AssertionError("async seller must not call sync act()")
    )
    seller.aact = AsyncMock(return_value="ignored action result")
    seller.aexecute_action = AsyncMock(
        side_effect=AssertionError("recent buyer must not select wait")
    )

    result = await seller.astep()

    assert result is None
    seller.act.assert_not_called()
    seller.aexecute_action.assert_not_awaited()
    seller.aact.assert_awaited_once()
    assert seller.aact.call_args.kwargs["actions"] == ["speak_to"]
    prompt = seller.aact.call_args.kwargs["prompt"][1]
    assert f'"unique_id": {buyer.unique_id}' in prompt
    assert '"recent_dialogue_partner": true' in prompt


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
