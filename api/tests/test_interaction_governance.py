from fastapi.testclient import TestClient
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


def test_tense_failure_report_suppresses_humor_and_commentary_and_uses_tactical_posture():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/interaction-governance/evaluate",
        json=_base(current_user_text="I think I broke the server and prod is failing"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "tense_debugging"
    assert result["humor_allowed"] is False
    assert result["commentary_allowed"] is False
    assert result["response_posture"] == "tactical"


def test_normal_question_classifies_as_question_and_stays_non_actioning():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/interaction-governance/evaluate",
        json=_base(current_user_text="What does this function do?"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "question"
    assert result["action_allowed"] is False
    assert result["humor_allowed"] is False
    assert result["commentary_allowed"] is False


def test_recent_messages_latest_user_text_is_used_when_current_user_text_is_omitted():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/interaction-governance/evaluate",
        json=_base(
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


def test_playful_low_risk_message_allows_humor_without_forcing_commentary():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/interaction-governance/evaluate",
        json=_base(current_user_text="lol roast my tiny todo list"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "joke_or_playful"
    assert result["humor_allowed"] is True
    assert result["commentary_allowed"] is False


def test_destructive_frustrated_phrase_requires_confirmation_and_stays_non_actioning():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/interaction-governance/evaluate",
        json=_base(current_user_text="nuke this"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "ambiguous"
    assert result["action_allowed"] is False
    assert result["requires_confirmation"] is True


def test_direct_low_risk_command_classifies_as_command_but_is_not_actionable():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/interaction-governance/evaluate",
        json=_base(current_user_text="rename this variable to count"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "command"
    assert result["literal_command_confidence"] >= 0.75
    assert result["action_allowed"] is False


def test_runtime_event_payload_is_summarized_only():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/interaction-governance/evaluate",
        json=_base(current_user_text="I think I broke the server and prod is failing badly"),
    )

    assert response.status_code == 200
    runtime_session_id = response.json()["runtime_session_id"]

    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
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


def test_ambiguous_fallback_is_conservative_and_non_actioning():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/interaction-governance/evaluate",
        json=_base(),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "ambiguous"
    assert result["action_allowed"] is False
    assert result["humor_allowed"] is False
    assert result["commentary_allowed"] is False


def test_runtime_turn_integration_updates_intent_class_and_records_event():
    client = TestClient(app)

    started = client.post(
        "/v1/runtime/turns/start",
        json={
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

    response = client.post(
        "/v1/runtime/interaction-governance/evaluate",
        json=_base(
            request_id="rid-turn-governance",
            runtime_session_id=runtime_session_id,
            runtime_turn_id=runtime_turn_id,
            current_user_text="rename this variable to count",
        ),
    )

    assert response.status_code == 200
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    body = diagnostics.json()
    assert body["latest_turn"]["intent_class"] == "action_command"
    assert any(
        event["event_type"] == "interaction_governance_evaluated" for event in body["events"]
    )
