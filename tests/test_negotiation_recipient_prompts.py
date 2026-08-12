import asyncio
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
from mesa_llm.memory.lt_memory import LongTermMemory
from mesa_llm.memory.memory import MemoryEntry
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.memory.st_memory import ShortTermMemory
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


def _dialogue_agent(
    memory_kind: str = "stlt",
    *,
    agent_type: str = "DialogueAgent",
    unique_id: int = 1,
):
    agent = _example_agent(agent_type, unique_id)
    agent.model = SimpleNamespace(agents=[], steps=0)
    agent.step_prompt = ""
    if memory_kind == "short_term":
        memory = ShortTermMemory(agent=agent, n=20, display=False)
    elif memory_kind == "stlt":
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


def _finalize_buffered_step(agent, step: int):
    agent.model.steps = step
    agent.memory.process_step(pre_step=True)
    agent.memory.process_step(pre_step=False)


def _add_finalized_memory_event(agent, step: int, event_type: str, content: dict):
    agent.model.steps = step
    agent.memory.add_to_memory(event_type, content)
    if isinstance(agent.memory, STLTMemory):
        _finalize_stlt_step(agent, step)


def _successful_speak_action_event(
    message: str,
    *,
    requested=(2,),
    delivered=(2,),
    skipped=(),
    failed=(),
):
    requested_ids = list(requested)
    choice = _speak_choice(requested_ids, message=message)
    return {
        "action": choice.model_dump(),
        "result": {
            "requested": requested_ids,
            "delivered": list(delivered),
            "skipped": list(skipped),
            "failed": list(failed),
        },
    }


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


@pytest.mark.parametrize("memory_kind", ("short_term", "stlt"))
def test_final_a12_staged_incoming_precedes_current_outgoing_after_finalization(
    memory_kind,
):
    agent = _dialogue_agent(
        memory_kind,
        agent_type="BuyerAgent",
        unique_id=4,
    )
    seller = _example_agent("SellerAgent", 7)
    agent.model.agents = [agent, seller]

    _add_message(agent, seller.unique_id, "The opening price is 40.")
    agent.memory.process_step(pre_step=True)
    agent.memory.add_to_memory(
        "action",
        _successful_speak_action_event(
            "Can you lower it?",
            requested=(seller.unique_id,),
            delivered=(seller.unique_id,),
        ),
    )
    agent.model.steps = 1
    agent.memory.process_step(pre_step=False)

    assert get_dialogue_history(agent) == "\n".join(
        (
            "- SellerAgent 7: The opening price is 40.",
            "- BuyerAgent 4 to [SellerAgent 7]: Can you lower it?",
        )
    )


def test_final_a12_short_term_unfinalized_state_includes_staged_then_current():
    buyer = _dialogue_agent(
        "short_term",
        agent_type="BuyerAgent",
        unique_id=4,
    )
    seller = _example_agent("SellerAgent", 7)
    buyer.model.agents = [buyer, seller]

    _add_message(buyer, seller.unique_id, "The opening price is 40.")
    buyer.memory.process_step(pre_step=True)
    buyer.memory.add_to_memory(
        "action",
        _successful_speak_action_event(
            "Can you lower it?",
            requested=(seller.unique_id,),
            delivered=(seller.unique_id,),
        ),
    )

    assert get_dialogue_history(buyer) == "\n".join(
        (
            "- SellerAgent 7: The opening price is 40.",
            "- BuyerAgent 4 to [SellerAgent 7]: Can you lower it?",
        )
    )


@pytest.mark.asyncio
async def test_final_a12_atomic_long_term_history_is_exact_while_async_commit_waits(
    llm_response_factory,
):
    seller = _dialogue_agent(
        "short_term",
        agent_type="SellerAgent",
        unique_id=5,
    )
    buyer = _example_agent("BuyerAgent", 7)
    seller.model.agents = [seller, buyer]
    seller.memory = LongTermMemory(
        agent=seller,
        llm_model="openai/test",
        display=False,
    )
    incoming_one = {
        "sender": buyer.unique_id,
        "message": "Gated incoming one.",
    }
    outgoing_two = _successful_speak_action_event(
        "Gated outgoing two.",
        requested=(buyer.unique_id,),
        delivered=(buyer.unique_id,),
    )
    incoming_three = {
        "sender": buyer.unique_id,
        "message": "Gated incoming three.",
    }
    outgoing_four = _successful_speak_action_event(
        "Gated outgoing four.",
        requested=(buyer.unique_id,),
        delivered=(buyer.unique_id,),
    )

    seller.memory.add_to_memory("message", incoming_one)
    seller.memory.add_to_memory("action", outgoing_two)
    await seller.memory.aprocess_step(pre_step=True)
    seller.memory.add_to_memory("message", incoming_three)
    seller.memory.add_to_memory("action", outgoing_four)
    seller.model.steps = 1

    entered = asyncio.Event()
    release = asyncio.Event()

    async def consolidate(_prompt):
        entered.set()
        await release.wait()
        return llm_response_factory("summary")

    seller.memory.llm.generate = Mock(
        side_effect=AssertionError("sync provider must not run")
    )
    seller.memory.llm.agenerate = AsyncMock(side_effect=consolidate)
    task = asyncio.create_task(seller.memory.aprocess_step(pre_step=False))
    await entered.wait()

    expected = "\n".join(
        (
            "- BuyerAgent 7: Gated incoming one.",
            "- SellerAgent 5 to [BuyerAgent 7]: Gated outgoing two.",
            "- BuyerAgent 7: Gated incoming three.",
            "- SellerAgent 5 to [BuyerAgent 7]: Gated outgoing four.",
        )
    )
    in_flight_history = get_dialogue_history(seller)
    assert in_flight_history == expected
    assert in_flight_history.count("Gated incoming three.") == 1
    assert in_flight_history.count("Gated outgoing four.") == 1
    assert seller.memory.buffer.step == 1
    assert seller.memory.step_content == {}
    assert seller.memory._step_event_order == []

    release.set()
    await task

    assert get_dialogue_history(seller) == expected
    seller.memory.llm.agenerate.assert_awaited_once()
    seller.memory.llm.generate.assert_not_called()


@pytest.mark.parametrize("memory_kind", ("short_term", "stlt"))
@pytest.mark.parametrize("finalize", (False, True), ids=("current", "finalized"))
def test_final_a12_same_step_message_action_interleaving_is_exact(
    memory_kind,
    finalize,
):
    seller = _dialogue_agent(
        memory_kind,
        agent_type="SellerAgent",
        unique_id=5,
    )
    buyer = _example_agent("BuyerAgent", 7)
    seller.model.agents = [seller, buyer]
    events = (
        ("message", {"sender": buyer.unique_id, "message": "Incoming one."}),
        (
            "action",
            _successful_speak_action_event(
                "Outgoing two.",
                requested=(buyer.unique_id,),
                delivered=(buyer.unique_id,),
            ),
        ),
        ("message", {"sender": buyer.unique_id, "message": "Incoming three."}),
        (
            "action",
            _successful_speak_action_event(
                "Outgoing four.",
                requested=(buyer.unique_id,),
                delivered=(buyer.unique_id,),
            ),
        ),
    )
    for event_type, content in events:
        seller.memory.add_to_memory(event_type, content)
    if finalize:
        _finalize_buffered_step(seller, 1)

    assert get_dialogue_history(seller) == "\n".join(
        (
            "- BuyerAgent 7: Incoming one.",
            "- SellerAgent 5 to [BuyerAgent 7]: Outgoing two.",
            "- BuyerAgent 7: Incoming three.",
            "- SellerAgent 5 to [BuyerAgent 7]: Outgoing four.",
        )
    )


def test_final_a12_dialogue_projection_ignores_valid_third_additive_type_marker():
    seller = _dialogue_agent(
        "short_term",
        agent_type="SellerAgent",
        unique_id=5,
    )
    seller.memory = ShortTermMemory(
        agent=seller,
        n=20,
        display=False,
        additive_event_types={"action", "message", "audit"},
    )
    buyer = _example_agent("BuyerAgent", 7)
    seller.model.agents = [seller, buyer]
    action_one = _successful_speak_action_event(
        "Action one.",
        requested=(buyer.unique_id,),
        delivered=(buyer.unique_id,),
    )
    message_two = {"sender": buyer.unique_id, "message": "Message two."}
    audit_event = {"event": "delivery audited"}
    action_three = _successful_speak_action_event(
        "Action three.",
        requested=(buyer.unique_id,),
        delivered=(buyer.unique_id,),
    )

    seller.memory.add_to_memory("action", action_one)
    seller.memory.add_to_memory("message", message_two)
    seller.memory.add_to_memory("audit", audit_event)
    seller.memory.add_to_memory("action", action_three)
    _finalize_buffered_step(seller, 1)

    entry = seller.memory.short_term_memory[-1]
    assert entry.content == {
        "action": [action_one, action_three],
        "message": [message_two],
        "audit": [audit_event],
    }
    assert entry._event_order == ["action", "message", "audit", "action"]
    assert get_dialogue_history(seller) == "\n".join(
        (
            "- SellerAgent 5 to [BuyerAgent 7]: Action one.",
            "- BuyerAgent 7: Message two.",
            "- SellerAgent 5 to [BuyerAgent 7]: Action three.",
        )
    )


def test_final_a12_finalized_entries_then_current_buffer_keep_step_order_and_limit():
    seller = _dialogue_agent(
        "short_term",
        agent_type="SellerAgent",
        unique_id=5,
    )
    buyer = _example_agent("BuyerAgent", 7)
    seller.model.agents = [seller, buyer]

    _add_message(seller, buyer.unique_id, "Step one incoming.")
    seller.memory.add_to_memory(
        "action",
        _successful_speak_action_event(
            "Step one outgoing.",
            requested=(buyer.unique_id,),
            delivered=(buyer.unique_id,),
        ),
    )
    _finalize_buffered_step(seller, 1)

    seller.memory.add_to_memory(
        "action",
        _successful_speak_action_event(
            "Step two outgoing.",
            requested=(buyer.unique_id,),
            delivered=(buyer.unique_id,),
        ),
    )
    _add_message(seller, buyer.unique_id, "Step two incoming.")

    assert get_dialogue_history(seller, max_messages=3) == "\n".join(
        (
            "- SellerAgent 5 to [BuyerAgent 7]: Step one outgoing.",
            "- SellerAgent 5 to [BuyerAgent 7]: Step two outgoing.",
            "- BuyerAgent 7: Step two incoming.",
        )
    )


@pytest.mark.parametrize(
    ("event_order", "expected"),
    (
        (
            None,
            (
                "- Agent 7: Legacy incoming.",
                "- SellerAgent 5 to [Agent 7]: Legacy outgoing.",
            ),
        ),
        (
            ["message"],
            (
                "- Agent 7: Legacy incoming.",
                "- SellerAgent 5 to [Agent 7]: Legacy outgoing.",
            ),
        ),
        (
            ["message", "unknown", "action"],
            (
                "- Agent 7: Legacy incoming.",
                "- SellerAgent 5 to [Agent 7]: Legacy outgoing.",
            ),
        ),
        (
            ["message", "action", "action"],
            (
                "- Agent 7: Legacy incoming.",
                "- SellerAgent 5 to [Agent 7]: Legacy outgoing.",
            ),
        ),
    ),
    ids=("missing", "short", "unknown", "excess"),
)
def test_final_a12_legacy_or_malformed_sidecar_falls_back_for_whole_entry(
    event_order,
    expected,
):
    seller = _dialogue_agent(agent_type="SellerAgent", unique_id=5)
    content = {
        "message": {"sender": 7, "message": "Legacy incoming."},
        "action": _successful_speak_action_event(
            "Legacy outgoing.", requested=(7,), delivered=(7,)
        ),
    }
    entry = MemoryEntry(content=content, step=1, agent=seller)
    if event_order is None:
        del entry._event_order
    else:
        entry._event_order = event_order
    seller.memory.short_term_memory.append(entry)

    assert get_dialogue_history(seller) == "\n".join(expected)


def test_final_a12_short_term_hidden_staged_sender_remains_recent_and_eligible():
    seller = _dialogue_agent(
        "short_term",
        agent_type="SellerAgent",
        unique_id=5,
    )
    buyer = _example_agent("BuyerAgent", 7)
    seller.model.agents = [seller, buyer]
    _add_message(seller, buyer.unique_id, "What is your price?")
    seller.memory.process_step(pre_step=True)

    recipients = get_eligible_recipients(seller, _observation(), "BuyerAgent")

    assert recipients == [
        {
            "label": "BuyerAgent 7",
            "unique_id": 7,
            "currently_visible": False,
            "recent_dialogue_partner": True,
        }
    ]
    assert get_dialogue_history(seller) == "- BuyerAgent 7: What is your price?"


def test_final_a12_multiple_deliveries_use_actual_partition_and_keep_equal_events():
    seller = _dialogue_agent(
        "short_term",
        agent_type="SellerAgent",
        unique_id=5,
    )
    delivered_buyer = _example_agent("BuyerAgent", 7)
    unknown_delivered_id = 404
    seller.model.agents = [seller, delivered_buyer]
    event = _successful_speak_action_event(
        "Same delivered offer.",
        requested=(delivered_buyer.unique_id, 8, 9, unknown_delivered_id),
        delivered=(delivered_buyer.unique_id, unknown_delivered_id),
        skipped=(8,),
        failed=(9,),
    )
    seller.memory.add_to_memory("action", event)
    seller.memory.add_to_memory("action", event.copy())

    line = "- SellerAgent 5 to [BuyerAgent 7, Agent 404]: Same delivered offer."
    assert get_dialogue_history(seller) == "\n".join((line, line))
    assert "Agent 8" not in get_dialogue_history(seller)
    assert "Agent 9" not in get_dialogue_history(seller)


def test_final_a12_hidden_staged_and_persisted_alias_is_counted_once_by_identity():
    seller = _dialogue_agent(
        "short_term",
        agent_type="SellerAgent",
        unique_id=5,
    )
    _add_message(seller, 7, "Count this staged message once.")
    seller.memory.process_step(pre_step=True)
    seller.memory.short_term_memory.append(seller.memory._current_step_entry)

    assert get_dialogue_history(seller) == (
        "- Agent 7: Count this staged message once."
    )


def test_final_a12_episodic_events_keep_write_order_without_extra_grading():
    seller = _dialogue_agent(
        "episodic",
        agent_type="SellerAgent",
        unique_id=5,
    )
    buyer = _example_agent("BuyerAgent", 7)
    seller.model.agents = [seller, buyer]
    seller.memory.llm.generate = Mock(
        side_effect=AssertionError("provider grading must remain mocked")
    )
    seller.memory.llm.agenerate = AsyncMock(
        side_effect=AssertionError("async provider grading must not run")
    )
    events = (
        ("message", {"sender": buyer.unique_id, "message": "Incoming one."}),
        (
            "action",
            _successful_speak_action_event(
                "Outgoing two.",
                requested=(buyer.unique_id,),
                delivered=(buyer.unique_id,),
            ),
        ),
        ("message", {"sender": buyer.unique_id, "message": "Incoming three."}),
        (
            "action",
            _successful_speak_action_event(
                "Outgoing four.",
                requested=(buyer.unique_id,),
                delivered=(buyer.unique_id,),
            ),
        ),
    )
    for event_type, content in events:
        seller.memory.add_to_memory(event_type, content)

    assert get_dialogue_history(seller) == "\n".join(
        (
            "- BuyerAgent 7: Incoming one.",
            "- SellerAgent 5 to [BuyerAgent 7]: Outgoing two.",
            "- BuyerAgent 7: Incoming three.",
            "- SellerAgent 5 to [BuyerAgent 7]: Outgoing four.",
        )
    )
    assert seller.memory.grade_event_importance.call_count == len(events)
    seller.memory.llm.generate.assert_not_called()
    seller.memory.llm.agenerate.assert_not_awaited()


@pytest.mark.parametrize("memory_kind", ("stlt", "episodic"))
def test_rf_a6_dialogue_history_merges_incoming_and_delivered_speech_in_order(
    memory_kind,
):
    buyer = _dialogue_agent(
        memory_kind,
        agent_type="BuyerAgent",
        unique_id=4,
    )
    seller = _example_agent("SellerAgent", 7)
    buyer.model.agents = [buyer, seller]

    _add_finalized_memory_event(
        buyer,
        1,
        "message",
        {
            "sender": seller.unique_id,
            "message": "The opening price is 40.",
            "recipients": [buyer.unique_id],
        },
    )
    _add_finalized_memory_event(
        buyer,
        2,
        "action",
        _successful_speak_action_event(
            "Can you lower it?",
            requested=(seller.unique_id, 404, 8, 9),
            delivered=(seller.unique_id, 404),
            skipped=(8,),
            failed=(9,),
        ),
    )
    _add_finalized_memory_event(
        buyer,
        3,
        "message",
        {
            "sender": 99,
            "message": "An external counteroffer.",
            "recipients": [buyer.unique_id],
        },
    )
    buyer.model.steps = 4
    buyer.memory.add_to_memory(
        "action",
        _successful_speak_action_event(
            "I can pay 35.",
            requested=(seller.unique_id,),
            delivered=(seller.unique_id,),
        ),
    )

    assert get_dialogue_history(buyer) == "\n".join(
        (
            "- SellerAgent 7: The opening price is 40.",
            "- BuyerAgent 4 to [SellerAgent 7, Agent 404]: Can you lower it?",
            "- Agent 99: An external counteroffer.",
            "- BuyerAgent 4 to [SellerAgent 7]: I can pay 35.",
        )
    )


@pytest.mark.parametrize("memory_kind", ("stlt", "episodic"))
def test_rf_a6_dialogue_history_applies_limit_after_merged_traversal(memory_kind):
    seller = _dialogue_agent(
        memory_kind,
        agent_type="SellerAgent",
        unique_id=5,
    )
    buyer = _example_agent("BuyerAgent", 7)
    seller.model.agents = [seller, buyer]
    events = (
        (
            "message",
            {"sender": buyer.unique_id, "message": "Incoming one."},
        ),
        (
            "action",
            _successful_speak_action_event(
                "Outgoing two.",
                requested=(buyer.unique_id,),
                delivered=(buyer.unique_id,),
            ),
        ),
        (
            "message",
            {"sender": "external-feed", "message": "Incoming three."},
        ),
        (
            "action",
            _successful_speak_action_event(
                "Outgoing four.",
                requested=(buyer.unique_id,),
                delivered=(buyer.unique_id,),
            ),
        ),
        (
            "message",
            {"sender": buyer.unique_id, "message": "Incoming five."},
        ),
    )
    for step, (event_type, content) in enumerate(events, start=1):
        if memory_kind == "stlt" and step == len(events):
            seller.model.steps = step
            seller.memory.add_to_memory(event_type, content)
        else:
            _add_finalized_memory_event(seller, step, event_type, content)

    assert get_dialogue_history(seller, max_messages=3) == "\n".join(
        (
            "- external-feed: Incoming three.",
            "- SellerAgent 5 to [BuyerAgent 7]: Outgoing four.",
            "- BuyerAgent 7: Incoming five.",
        )
    )


@pytest.mark.parametrize("memory_kind", ("stlt", "episodic"))
@pytest.mark.parametrize(
    "scenario",
    (
        "skipped-only",
        "failed-only",
        "empty-delivery",
        "malformed-action",
        "missing-message",
        "malformed-result",
        "malformed-delivered",
        "unrelated-action",
        "legacy-missing-result",
        "legacy-unknown-result",
    ),
)
def test_rf_a6_dialogue_history_excludes_non_delivered_action_snapshots(
    scenario,
    memory_kind,
):
    agent = _dialogue_agent(
        memory_kind,
        agent_type="SellerAgent",
        unique_id=5,
    )
    event = _successful_speak_action_event(
        f"Excluded {scenario}.",
        requested=(7,),
        delivered=(),
    )

    if scenario == "skipped-only":
        event["result"]["skipped"] = [7]
    elif scenario == "failed-only":
        event["result"]["failed"] = [7]
    elif scenario == "malformed-action":
        event["action"] = "speak_to"
        event["result"]["delivered"] = [7]
    elif scenario == "missing-message":
        del event["action"]["arguments"]["message"]
        event["result"]["delivered"] = [7]
    elif scenario == "malformed-result":
        event["result"] = ["delivered", 7]
    elif scenario == "malformed-delivered":
        event["result"]["delivered"] = 7
    elif scenario == "unrelated-action":
        event["action"]["name"] = "wait"
        event["result"]["delivered"] = [7]
    elif scenario == "legacy-missing-result":
        del event["result"]
    elif scenario == "legacy-unknown-result":
        event["result"] = "sent message to recipient"

    _add_finalized_memory_event(agent, 1, "action", event)

    assert get_dialogue_history(agent) == "No recent dialogue."


@pytest.mark.parametrize("memory_kind", ("stlt", "episodic"))
def test_final_a6_equal_actions_remain_distinct_while_current_alias_is_deduplicated(
    memory_kind,
):
    agent = _dialogue_agent(
        memory_kind,
        agent_type="SellerAgent",
        unique_id=5,
    )
    for step in (1, 2):
        _add_finalized_memory_event(
            agent,
            step,
            "action",
            _successful_speak_action_event(
                "The price is still 40.",
                requested=(7,),
                delivered=(7,),
            ),
        )

    entries = (
        agent.memory.short_term_memory
        if memory_kind == "stlt"
        else agent.memory.memory_entries
    )
    assert entries[0].content == entries[1].content
    assert entries[0].content is not entries[1].content
    agent.memory.step_content = entries[-1].content

    assert get_dialogue_history(agent) == "\n".join(
        (
            "- SellerAgent 5 to [Agent 7]: The price is still 40.",
            "- SellerAgent 5 to [Agent 7]: The price is still 40.",
        )
    )


@pytest.mark.parametrize("memory_kind", ("stlt", "episodic"))
@pytest.mark.parametrize(
    ("actor_kind", "peer_type", "message"),
    (
        pytest.param(
            "seller",
            "BuyerAgent",
            "My delivered offer is 40.",
            id="seller",
        ),
        pytest.param(
            "buyer",
            "SellerAgent",
            "My delivered counteroffer is 35.",
            id="buyer",
        ),
    ),
)
def test_final_a6_next_prompt_includes_own_speech_without_sender_message_or_llm_call(
    monkeypatch,
    actor_kind,
    peer_type,
    message,
    memory_kind,
):
    actor = (
        _seller_agent(monkeypatch)
        if actor_kind == "seller"
        else _buyer_agent(monkeypatch)
    )
    grading = None
    if memory_kind == "episodic":
        actor.memory = EpisodicMemory(
            agent=actor,
            display=False,
            llm_model="openai/test",
        )
        grading = Mock(return_value=3)
        actor.memory.grade_event_importance = grading
    else:
        assert isinstance(actor.memory, STLTMemory)

    peer = _add_peer(actor.model, peer_type)
    _set_visible_peers(actor, peer)
    _forbid_action_selection(actor)
    result = actor.execute_action(
        _speak_choice([peer.unique_id], message=message),
        actions=["speak_to"],
    )

    _assert_delivery_partition(
        result,
        [peer.unique_id],
        [peer.unique_id],
        [],
    )
    if memory_kind == "stlt":
        assert list(actor.memory.step_content) == ["action"]
        assert len(actor.memory.step_content["action"]) == 1
    else:
        assert len(actor.memory.memory_entries) == 1
        assert set(actor.memory.memory_entries[0].content) == {"action"}
        assert grading.call_count == 1

    observation = _observation(f"{peer_type} {peer.unique_id}")
    actor.generate_obs = Mock(return_value=observation)
    actor.act = Mock(return_value="ignored action result")

    actor.step()

    own_line = (
        f"- {type(actor).__name__} {actor.unique_id} "
        f"to [{peer_type} {peer.unique_id}]: {message}"
    )
    prompt = actor.act.call_args.kwargs["prompt"][1]
    assert prompt.count(own_line) == 1
    if memory_kind == "stlt":
        assert "message" not in actor.memory.step_content
    else:
        assert all(
            "message" not in entry.content for entry in actor.memory.memory_entries
        )
        assert grading.call_count == 1
    actor.llm.generate.assert_not_called()
    actor.llm.agenerate.assert_not_awaited()
    actor.memory.llm.generate.assert_not_called()
    actor.memory.llm.agenerate.assert_not_awaited()


@pytest.mark.parametrize("actor_kind", ("seller", "buyer"))
def test_final_a12_next_prompt_displays_staged_incoming_before_own_reply_without_llm(
    monkeypatch,
    actor_kind,
):
    actor = (
        _seller_agent(monkeypatch)
        if actor_kind == "seller"
        else _buyer_agent(monkeypatch)
    )
    peer_type = "BuyerAgent" if actor_kind == "seller" else "SellerAgent"
    peer = _add_peer(actor.model, peer_type)
    _set_visible_peers(actor, peer)
    _forbid_action_selection(actor)

    actor.memory.add_to_memory(
        "message",
        {"sender": peer.unique_id, "message": "Incoming offer."},
    )
    actor.memory.process_step(pre_step=True)
    result = actor.execute_action(
        _speak_choice([peer.unique_id], message="Outgoing reply."),
        actions=["speak_to"],
    )
    actor.model.steps = 1
    actor.memory.process_step(pre_step=False)

    _assert_delivery_partition(
        result,
        [peer.unique_id],
        [peer.unique_id],
        [],
    )
    dialogue = get_dialogue_history(actor)
    assert dialogue.index("Incoming offer.") < dialogue.index("Outgoing reply.")

    observation = _observation(f"{peer_type} {peer.unique_id}")
    if actor_kind == "seller":
        prompt = actor._seller_step_prompt(
            dialogue,
            get_eligible_recipients(actor, observation, peer_type),
        )
    else:
        prompt, actions = actor._buyer_step_prompt_and_actions(observation, dialogue)
        assert actions == ["speak_to", "buy_product"]

    assert prompt.index("Incoming offer.") < prompt.index("Outgoing reply.")
    actor.llm.generate.assert_not_called()
    actor.llm.agenerate.assert_not_awaited()
    actor.memory.llm.generate.assert_not_called()
    actor.memory.llm.agenerate.assert_not_awaited()


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
