import httpx
import pytest

from main import app


def _base(**overrides):
    payload = {
        "request_id": "rid-governance",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
        "recent_messages": [],
    }
    payload.update(overrides)
    return payload


async def _post(path: str, payload: dict[str, object]):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post(path, json=payload)


async def _get(path: str):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_tense_failure_report_suppresses_humor_and_commentary_and_uses_tactical_posture():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text="I think I broke the server and prod is failing"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "tense_debugging"
    assert result["humor_allowed"] is False
    assert result["commentary_allowed"] is False
    assert result["response_posture"] == "tactical"


@pytest.mark.asyncio
async def test_normal_question_classifies_as_question_and_stays_non_actioning():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text="What does this function do?"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "question"
    assert result["action_allowed"] is False
    assert result["humor_allowed"] is False
    assert result["commentary_allowed"] is False


@pytest.mark.asyncio
async def test_recent_messages_latest_user_text_is_used_when_current_user_text_is_omitted():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            recent_messages=[
                {"role": "user", "content": "Can you take a look?"},
                {"role": "assistant", "content": "What seems wrong?"},
                {
                    "role": "user",
                    "content": "I think I broke the server and prod is failing",
                },
            ],
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "tense_debugging"
    assert result["humor_allowed"] is False
    assert result["commentary_allowed"] is False
    assert result["response_posture"] == "tactical"


@pytest.mark.asyncio
async def test_playful_low_risk_message_allows_humor_without_forcing_commentary():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text="lol roast my tiny todo list"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "joke_or_playful"
    assert result["humor_allowed"] is True
    assert result["commentary_allowed"] is False


@pytest.mark.asyncio
async def test_destructive_frustrated_phrase_requires_confirmation_and_stays_non_actioning():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text="nuke this"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "ambiguous"
    assert result["action_allowed"] is False
    assert result["requires_confirmation"] is True


@pytest.mark.asyncio
async def test_direct_low_risk_command_classifies_as_command_but_is_not_actionable():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text="rename this variable to count"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "command"
    assert result["literal_command_confidence"] >= 0.75
    assert result["action_allowed"] is False


@pytest.mark.asyncio
async def test_runtime_event_payload_is_summarized_only():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text="I think I broke the server and prod is failing badly"),
    )

    assert response.status_code == 200
    runtime_session_id = response.json()["runtime_session_id"]

    diagnostics = await _get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    events = diagnostics.json()["events"]
    governance_event = next(
        event for event in events if event["event_type"] == "interaction_governance_evaluated"
    )
    payload = governance_event["event_payload_json"]
    assert set(payload.keys()) == {
        "request_id",
        "interaction_kind",
        "response_posture",
        "commentary_allowed",
        "humor_allowed",
        "action_allowed",
        "requires_confirmation",
        "reason_summary",
    }
    assert "current_user_text" not in str(payload)
    assert "broke the server" not in str(payload)


@pytest.mark.asyncio
async def test_ambiguous_fallback_is_conservative_and_non_actioning():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "ambiguous"
    assert result["action_allowed"] is False
    assert result["humor_allowed"] is False
    assert result["commentary_allowed"] is False


@pytest.mark.asyncio
async def test_runtime_turn_integration_updates_intent_class_and_records_event():
    started = await _post(
        "/v1/runtime/turns/start",
        {
            "request_id": "rid-turn-start",
            "owner_id": "owner",
            "conversation_id": "conv-1",
            "surface": "dev",
            "input_message_id": "msg-1",
        },
    )
    assert started.status_code == 200
    runtime_session_id = started.json()["runtime_session"]["runtime_session_id"]
    runtime_turn_id = started.json()["runtime_turn"]["runtime_turn_id"]

    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            request_id="rid-turn-governance",
            runtime_session_id=runtime_session_id,
            runtime_turn_id=runtime_turn_id,
            current_user_text="rename this variable to count",
        ),
    )

    assert response.status_code == 200
    diagnostics = await _get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    body = diagnostics.json()
    assert body["latest_turn"]["intent_class"] == "action_command"
    assert any(
        event["event_type"] == "interaction_governance_evaluated" for event in body["events"]
    )


async def _governance_turn_diagnostics(
    *,
    request_id: str,
    current_user_text: str,
    recent_messages: list[dict[str, str]] | None = None,
):
    started = await _post(
        "/v1/runtime/turns/start",
        {
            "request_id": f"{request_id}-start",
            "owner_id": "owner",
            "conversation_id": "conv-1",
            "surface": "dev",
            "input_message_id": "msg-1",
        },
    )
    assert started.status_code == 200
    runtime_session_id = started.json()["runtime_session"]["runtime_session_id"]
    runtime_turn_id = started.json()["runtime_turn"]["runtime_turn_id"]

    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            request_id=request_id,
            runtime_session_id=runtime_session_id,
            runtime_turn_id=runtime_turn_id,
            current_user_text=current_user_text,
            recent_messages=recent_messages or [],
        ),
    )

    assert response.status_code == 200
    diagnostics = await _get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    return response.json()["result"], diagnostics.json()


@pytest.mark.asyncio
async def test_clarification_request_projection_uses_question_form_clarification_markers():
    result, diagnostics = await _governance_turn_diagnostics(
        request_id="rid-clarification",
        current_user_text="What do you mean by policy hints?",
    )

    assert result["interaction_kind"] == "question"
    assert diagnostics["latest_turn"]["intent_class"] == "clarification_request"


@pytest.mark.asyncio
async def test_confirmation_response_projection_requires_assistant_question_context():
    result, diagnostics = await _governance_turn_diagnostics(
        request_id="rid-confirmation",
        current_user_text="Yes.",
        recent_messages=[
            {"role": "assistant", "content": "Do you want me to proceed?"},
        ],
    )

    assert result["interaction_kind"] == "ambiguous"
    assert diagnostics["latest_turn"]["intent_class"] == "confirmation_response"


@pytest.mark.asyncio
async def test_continuation_projection_requires_live_conversation_context():
    result, diagnostics = await _governance_turn_diagnostics(
        request_id="rid-continuation",
        current_user_text="Go on.",
        recent_messages=[
            {"role": "assistant", "content": "The first issue is in the retry path."},
        ],
    )

    assert result["interaction_kind"] == "ambiguous"
    assert diagnostics["latest_turn"]["intent_class"] == "continuation"


@pytest.mark.asyncio
async def test_unsupported_ambiguous_reply_still_falls_back_to_low_confidence_unclear():
    result, diagnostics = await _governance_turn_diagnostics(
        request_id="rid-ambiguous-fallback",
        current_user_text="Okay.",
        recent_messages=[
            {"role": "assistant", "content": "Do you want me to proceed?"},
        ],
    )

    assert result["interaction_kind"] == "ambiguous"
    assert diagnostics["latest_turn"]["intent_class"] == "low_confidence_unclear"


@pytest.mark.asyncio
async def test_existing_supported_intent_class_mappings_remain_unchanged():
    cases = [
        (
            "rid-information-request",
            "What does this function do?",
            [],
            "information_request",
        ),
        (
            "rid-action-command",
            "rename this variable to count",
            [],
            "action_command",
        ),
        (
            "rid-correction",
            "Actually, the wrong service failed; I meant the worker queue.",
            [],
            "correction",
        ),
        (
            "rid-venting",
            "Ugh, this sucks and I'm upset.",
            [],
            "venting_signal",
        ),
    ]

    for request_id, current_user_text, recent_messages, expected_intent_class in cases:
        _, diagnostics = await _governance_turn_diagnostics(
            request_id=request_id,
            current_user_text=current_user_text,
            recent_messages=recent_messages,
        )
        assert diagnostics["latest_turn"]["intent_class"] == expected_intent_class
