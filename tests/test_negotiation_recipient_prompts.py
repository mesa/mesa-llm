from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mesa.agent import Agent
from mesa.model import Model

from examples.negotiation.agents import (
    BuyerAgent,
    SellerAgent,
    get_dialogue_history,
    get_eligible_recipients,
    get_recipient_prompt,
)
from mesa_llm.actions import ActionChoice
from mesa_llm.actions import speak_to as builtin_speak_to
from mesa_llm.memory.episodic_memory import EpisodicMemory
from mesa_llm.memory.st_lt_memory import STLTMemory
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


def _dialogue_agent(memory_kind: str = "stlt"):
    agent = SimpleNamespace(
        model=SimpleNamespace(agents=[], steps=0),
        step_prompt="",
    )
    if memory_kind == "stlt":
        memory = STLTMemory(
            agent=agent,
            short_term_capacity=20,
            consolidation_capacity=0,
            display=False,
            llm_model="openai/test",
        )
    else:
        memory = EpisodicMemory(
            agent=agent,
            display=False,
            llm_model="openai/test",
        )
        memory.grade_event_importance = Mock(return_value=3)
    agent.memory = memory
    return agent


def _add_message(agent, sender, message: str):
    agent.memory.add_to_memory(
        "message",
        {"sender": sender, "message": message, "recipients": []},
    )


def _finalize_stlt_step(agent, step: int):
    agent.model.steps = step
    agent.memory.process_step(pre_step=True)
    agent.memory.process_step(pre_step=False)


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


def _buyer_agent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    model = Model(rng=42)
    buyer = BuyerAgent(
        model=model,
        reasoning=ReActReasoning,
        llm_model="openai/test",
        system_prompt="",
        vision=1,
        internal_state=["careful"],
        budget=50,
    )
    buyer.memory.display = False
    return buyer


def _add_peer(model, agent_type: str):
    peer_class = type(agent_type, (Agent,), {})
    peer = peer_class(model)
    peer.memory = SimpleNamespace(add_to_memory=Mock())
    return peer


def _set_visible_peers(agent, *peers, side_effect=None):
    local_state = {f"{type(peer).__name__} {peer.unique_id}": {} for peer in peers}

    def build_observation():
        if side_effect is not None:
            side_effect()
        return {}, local_state

    agent._build_observation = Mock(side_effect=build_observation)
    return local_state


def _speak_choice(requested_ids, message="Offer is 40."):
    return ActionChoice(
        name="speak_to",
        arguments={
            "listener_agents_unique_ids": requested_ids,
            "message": message,
        },
    )


def _assert_delivery_partition(result, requested, delivered, skipped, failed=()):
    assert result == {
        "requested": requested,
        "delivered": delivered,
        "skipped": skipped,
        "failed": list(failed),
    }
    assert list(result) == ["requested", "delivered", "skipped", "failed"]


def _assert_action_result_recorded(agent, choice, result):
    expected_event = {
        "action": choice.model_dump(),
        "result": result,
    }
    assert agent.memory.step_content["action"][-1] == expected_event
    agent.recorder.record_event.assert_called_once_with(
        "action",
        content=expected_event,
        agent_id=agent.unique_id,
        metadata={"source": "LLMAgent.execute_action"},
    )


class _NegotiationDeliveryAbort(BaseException):
    pass


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


@pytest.mark.parametrize("memory_kind", ("stlt", "episodic"))
def test_dialogue_history_flattens_one_message_added_through_real_memory_api(
    memory_kind,
):
    sender = _example_agent("SellerAgent", 7)
    agent = _dialogue_agent(memory_kind)
    agent.model.agents = [sender]

    _add_message(agent, sender.unique_id, "The price is 40.")
    if memory_kind == "stlt":
        _finalize_stlt_step(agent, 1)

    assert get_dialogue_history(agent) == "- SellerAgent 7: The price is 40."


def test_dialogue_history_preserves_chronology_across_entries_and_current_buffer():
    matched_sender = _example_agent("SellerAgent", 7)
    object_sender = _example_agent("BuyerAgent", 3)
    agent = _dialogue_agent()
    agent.model.agents = [matched_sender]

    _add_message(agent, object_sender, "Object sender.")
    _finalize_stlt_step(agent, 1)
    _add_message(agent, matched_sender.unique_id, "Matching integer sender.")
    _add_message(agent, 99, "Unknown integer sender.")
    _finalize_stlt_step(agent, 2)
    _add_message(agent, "external-feed", "Stable fallback sender.")

    assert len(agent.memory.short_term_memory[0].content["message"]) == 1
    assert len(agent.memory.short_term_memory[1].content["message"]) == 2
    assert len(agent.memory.step_content["message"]) == 1
    assert get_dialogue_history(agent) == "\n".join(
        (
            "- BuyerAgent 3: Object sender.",
            "- SellerAgent 7: Matching integer sender.",
            "- Agent 99: Unknown integer sender.",
            "- external-feed: Stable fallback sender.",
        )
    )


def test_dialogue_history_supports_legacy_single_dictionary_message_entry():
    agent = _dialogue_agent()
    agent.memory.step_content["message"] = {
        "sender": 41,
        "message": "Legacy shape.",
        "recipients": [],
    }
    _finalize_stlt_step(agent, 1)

    assert isinstance(agent.memory.short_term_memory[0].content["message"], dict)
    assert get_dialogue_history(agent) == "- Agent 41: Legacy shape."


@pytest.mark.parametrize("memory_kind", ("stlt", "episodic"))
def test_dialogue_history_applies_max_messages_after_flattening(memory_kind):
    agent = _dialogue_agent(memory_kind)
    for number in range(1, 6):
        _add_message(agent, "sender", f"message {number}")

    assert get_dialogue_history(agent, max_messages=3) == "\n".join(
        (
            "- sender: message 3",
            "- sender: message 4",
            "- sender: message 5",
        )
    )


def test_dialogue_history_does_not_double_count_shared_current_content():
    agent = _dialogue_agent()
    _add_message(agent, 5, "Count this once.")
    _finalize_stlt_step(agent, 1)

    finalized_content = agent.memory.short_term_memory[-1].content
    agent.memory.step_content = finalized_content

    assert agent.memory.step_content is finalized_content
    assert get_dialogue_history(agent) == "- Agent 5: Count this once."


@pytest.mark.parametrize("memory_kind", ("stlt", "episodic"))
def test_dialogue_history_returns_empty_result_for_empty_real_memory(memory_kind):
    agent = _dialogue_agent(memory_kind)

    assert get_dialogue_history(agent) == "No recent dialogue."


@pytest.mark.parametrize("max_messages", (0, -1))
def test_dialogue_history_non_positive_limit_returns_empty_result(max_messages):
    agent = _dialogue_agent()
    _add_message(agent, 2, "Excluded by limit.")

    assert (
        get_dialogue_history(agent, max_messages=max_messages) == "No recent dialogue."
    )


def test_seller_execute_action_filters_and_reports_complete_recipient_partition(
    monkeypatch,
):
    seller = _seller_agent(monkeypatch)
    visible_buyer = _add_peer(seller.model, "BuyerAgent")
    recent_buyer = _add_peer(seller.model, "BuyerAgent")
    wrong_type = _add_peer(seller.model, "SellerAgent")
    off_topology_buyer = _add_peer(seller.model, "BuyerAgent")
    unusable_buyer = _add_peer(seller.model, "BuyerAgent")
    unusable_buyer.memory = SimpleNamespace(add_to_memory=None)
    _set_visible_peers(seller, visible_buyer, unusable_buyer)
    seller.memory.add_to_memory(
        "message",
        {
            "sender": recent_buyer.unique_id,
            "message": "What is your price?",
            "recipients": [seller.unique_id],
        },
    )
    _forbid_action_selection(seller)

    delivery_order = []
    for recipient in (visible_buyer, recent_buyer):
        recipient.memory.add_to_memory.side_effect = (
            lambda *, type, content, recipient_id=recipient.unique_id: (
                delivery_order.append(recipient_id)
            )
        )

    nonexistent_id = 999_999
    requested = [
        recent_buyer.unique_id,
        visible_buyer.unique_id,
        recent_buyer.unique_id,
        seller.unique_id,
        wrong_type.unique_id,
        unusable_buyer.unique_id,
        off_topology_buyer.unique_id,
        nonexistent_id,
        visible_buyer.unique_id,
        unusable_buyer.unique_id,
        wrong_type.unique_id,
    ]

    seller.recorder = SimpleNamespace(record_event=Mock())
    choice = _speak_choice(requested)
    result = seller.execute_action(choice, actions=["speak_to"])

    delivered = [recent_buyer.unique_id, visible_buyer.unique_id]
    skipped = [
        seller.unique_id,
        wrong_type.unique_id,
        unusable_buyer.unique_id,
        off_topology_buyer.unique_id,
        nonexistent_id,
    ]
    _assert_delivery_partition(result, requested, delivered, skipped)
    _assert_action_result_recorded(seller, choice, result)
    assert seller._action_manager.available_actions(seller)["speak_to"] is not (
        builtin_speak_to
    )
    seller._build_observation.assert_called_once_with()
    assert delivery_order == delivered
    for recipient in (visible_buyer, recent_buyer):
        recipient.memory.add_to_memory.assert_called_once_with(
            type="message",
            content={"message": "Offer is 40.", "sender": seller.unique_id},
        )
    wrong_type.memory.add_to_memory.assert_not_called()
    off_topology_buyer.memory.add_to_memory.assert_not_called()
    assert unusable_buyer.memory.add_to_memory is None
    seller.llm.generate.assert_not_called()
    seller.llm.agenerate.assert_not_awaited()


def test_seller_execute_action_reports_no_delivery_without_recipient_mutation(
    monkeypatch,
):
    seller = _seller_agent(monkeypatch)
    wrong_type = _add_peer(seller.model, "SellerAgent")
    off_topology_buyer = _add_peer(seller.model, "BuyerAgent")
    _set_visible_peers(seller)
    _forbid_action_selection(seller)
    nonexistent_id = 999_999
    requested = [
        seller.unique_id,
        wrong_type.unique_id,
        off_topology_buyer.unique_id,
        nonexistent_id,
        off_topology_buyer.unique_id,
    ]

    result = seller.execute_action(
        _speak_choice(requested, message="Anyone there?"),
        actions=["speak_to"],
    )

    skipped = [
        seller.unique_id,
        wrong_type.unique_id,
        off_topology_buyer.unique_id,
        nonexistent_id,
    ]
    _assert_delivery_partition(result, requested, [], skipped)
    assert result["delivered"] == []
    wrong_type.memory.add_to_memory.assert_not_called()
    off_topology_buyer.memory.add_to_memory.assert_not_called()
    assert "message" not in seller.memory.step_content


@pytest.mark.asyncio
async def test_seller_aexecute_action_uses_same_recipient_authorization(monkeypatch):
    seller = _seller_agent(monkeypatch)
    visible_buyer = _add_peer(seller.model, "BuyerAgent")
    off_topology_buyer = _add_peer(seller.model, "BuyerAgent")
    _set_visible_peers(seller, visible_buyer)
    _forbid_action_selection(seller)
    requested = [
        visible_buyer.unique_id,
        off_topology_buyer.unique_id,
        visible_buyer.unique_id,
    ]

    seller.recorder = SimpleNamespace(record_event=Mock())
    choice = _speak_choice(requested, message="Async offer.")
    result = await seller.aexecute_action(choice, actions=["speak_to"])

    _assert_delivery_partition(
        result,
        requested,
        [visible_buyer.unique_id],
        [off_topology_buyer.unique_id],
    )
    _assert_action_result_recorded(seller, choice, result)
    visible_buyer.memory.add_to_memory.assert_called_once_with(
        type="message",
        content={"message": "Async offer.", "sender": seller.unique_id},
    )
    off_topology_buyer.memory.add_to_memory.assert_not_called()
    seller.llm.generate.assert_not_called()
    seller.llm.agenerate.assert_not_awaited()


@pytest.mark.parametrize(
    ("outcomes", "delivered_indexes", "failed_indexes"),
    [
        pytest.param(("success", "success"), (0, 1), (), id="both-succeed"),
        pytest.param(("success", "failure"), (0,), (1,), id="second-fails"),
        pytest.param(("failure", "success"), (1,), (0,), id="first-fails"),
        pytest.param(("failure", "failure"), (), (0, 1), id="both-fail"),
    ],
)
def test_negotiation_speak_to_attempts_unique_recipients_and_reports_failures(
    monkeypatch,
    outcomes,
    delivered_indexes,
    failed_indexes,
):
    seller = _seller_agent(monkeypatch)
    created_first = _add_peer(seller.model, "BuyerAgent")
    created_second = _add_peer(seller.model, "BuyerAgent")
    recipients = [created_second, created_first]
    _set_visible_peers(seller, *recipients)
    _forbid_action_selection(seller)
    attempts = []
    message_events = []

    def delivery_effect(recipient, outcome):
        def add_to_memory(*, type, content):
            attempts.append(recipient.unique_id)
            if outcome == "failure":
                raise RuntimeError(f"memory {recipient.unique_id} failed")
            message_events.append(
                {
                    "recipient": recipient.unique_id,
                    "type": type,
                    "content": content,
                }
            )

        return add_to_memory

    for recipient, outcome in zip(recipients, outcomes, strict=True):
        recipient.memory.add_to_memory.side_effect = delivery_effect(
            recipient,
            outcome,
        )

    requested = [
        recipients[0].unique_id,
        recipients[1].unique_id,
        recipients[0].unique_id,
        recipients[1].unique_id,
    ]
    choice = _speak_choice(requested, message="Attempt every offer.")

    result = seller.execute_action(choice, actions=["speak_to"])

    delivered = [recipients[index].unique_id for index in delivered_indexes]
    failed = [recipients[index].unique_id for index in failed_indexes]
    _assert_delivery_partition(result, requested, delivered, [], failed)
    assert attempts == [recipient.unique_id for recipient in recipients]
    assert [event["recipient"] for event in message_events] == delivered
    assert all(event["type"] == "message" for event in message_events)
    assert all(
        event["content"]
        == {"message": "Attempt every offer.", "sender": seller.unique_id}
        for event in message_events
    )
    for recipient in recipients:
        recipient.memory.add_to_memory.assert_called_once()
    assert seller.memory.step_content["action"][-1] == {
        "action": choice.model_dump(),
        "result": result,
    }


def test_negotiation_speak_to_propagates_base_exception_without_recording_action(
    monkeypatch,
):
    seller = _seller_agent(monkeypatch)
    aborting_buyer = _add_peer(seller.model, "BuyerAgent")
    later_buyer = _add_peer(seller.model, "BuyerAgent")
    _set_visible_peers(seller, aborting_buyer, later_buyer)
    _forbid_action_selection(seller)
    seller.recorder = SimpleNamespace(record_event=Mock())
    abort = _NegotiationDeliveryAbort("recipient aborted")
    aborting_buyer.memory.add_to_memory.side_effect = abort
    choice = _speak_choice(
        [
            aborting_buyer.unique_id,
            later_buyer.unique_id,
            aborting_buyer.unique_id,
        ],
        message="This offer aborts.",
    )

    with pytest.raises(_NegotiationDeliveryAbort) as exc_info:
        seller.execute_action(choice, actions=["speak_to"])

    assert exc_info.value is abort
    aborting_buyer.memory.add_to_memory.assert_called_once()
    later_buyer.memory.add_to_memory.assert_not_called()
    assert "action" not in seller.memory.step_content
    seller.recorder.record_event.assert_not_called()


def test_buyer_execute_action_uses_seller_specific_authorization(monkeypatch):
    buyer = _buyer_agent(monkeypatch)
    visible_seller = _add_peer(buyer.model, "SellerAgent")
    wrong_type = _add_peer(buyer.model, "BuyerAgent")
    off_topology_seller = _add_peer(buyer.model, "SellerAgent")
    _set_visible_peers(buyer, visible_seller)
    _forbid_action_selection(buyer)
    nonexistent_id = 999_999
    requested = [
        visible_seller.unique_id,
        wrong_type.unique_id,
        off_topology_seller.unique_id,
        nonexistent_id,
        visible_seller.unique_id,
    ]

    result = buyer.execute_action(
        _speak_choice(requested, message="What is the price?"),
        actions=["speak_to"],
    )

    _assert_delivery_partition(
        result,
        requested,
        [visible_seller.unique_id],
        [
            wrong_type.unique_id,
            off_topology_seller.unique_id,
            nonexistent_id,
        ],
    )
    visible_seller.memory.add_to_memory.assert_called_once_with(
        type="message",
        content={"message": "What is the price?", "sender": buyer.unique_id},
    )
    wrong_type.memory.add_to_memory.assert_not_called()
    off_topology_seller.memory.add_to_memory.assert_not_called()
    buyer.llm.generate.assert_not_called()
    buyer.llm.agenerate.assert_not_awaited()


def test_recipient_partition_is_complete_before_first_delivery_mutation(monkeypatch):
    seller = _seller_agent(monkeypatch)
    first_buyer = _add_peer(seller.model, "BuyerAgent")
    second_buyer = _add_peer(seller.model, "BuyerAgent")
    off_topology_buyer = _add_peer(seller.model, "BuyerAgent")
    events = []
    local_state = _set_visible_peers(
        seller,
        first_buyer,
        second_buyer,
        side_effect=lambda: events.append("authorize"),
    )
    _forbid_action_selection(seller)

    def mutate_topology_after_first_delivery(*, type, content):
        del type, content
        events.append(f"deliver:{first_buyer.unique_id}")
        local_state[f"BuyerAgent {off_topology_buyer.unique_id}"] = {}

    first_buyer.memory.add_to_memory.side_effect = mutate_topology_after_first_delivery
    second_buyer.memory.add_to_memory.side_effect = lambda *, type, content: (
        events.append(f"deliver:{second_buyer.unique_id}")
    )
    requested = [
        first_buyer.unique_id,
        off_topology_buyer.unique_id,
        second_buyer.unique_id,
    ]

    result = seller.execute_action(
        _speak_choice(requested),
        actions=["speak_to"],
    )

    _assert_delivery_partition(
        result,
        requested,
        [first_buyer.unique_id, second_buyer.unique_id],
        [off_topology_buyer.unique_id],
    )
    assert events == [
        "authorize",
        f"deliver:{first_buyer.unique_id}",
        f"deliver:{second_buyer.unique_id}",
    ]
    seller._build_observation.assert_called_once_with()
    off_topology_buyer.memory.add_to_memory.assert_not_called()


def test_recent_partner_tracking_uses_delivered_partition_not_raw_requested_ids(
    monkeypatch,
):
    seller = _seller_agent(monkeypatch)
    delivered_buyer = _add_peer(seller.model, "BuyerAgent")
    unauthorized_buyer = _add_peer(seller.model, "BuyerAgent")
    _set_visible_peers(seller, delivered_buyer)
    _forbid_action_selection(seller)
    requested = [delivered_buyer.unique_id, unauthorized_buyer.unique_id]

    result = seller.execute_action(
        _speak_choice(requested),
        actions=["speak_to"],
    )
    recent_recipients = get_eligible_recipients(
        seller,
        _observation(),
        "BuyerAgent",
    )

    _assert_delivery_partition(
        result,
        requested,
        [delivered_buyer.unique_id],
        [unauthorized_buyer.unique_id],
    )
    assert [recipient["unique_id"] for recipient in recent_recipients] == result[
        "delivered"
    ]
    assert unauthorized_buyer.unique_id in result["skipped"]
    delivered_buyer.memory.add_to_memory.assert_called_once()
    unauthorized_buyer.memory.add_to_memory.assert_not_called()


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
