from __future__ import annotations

from fastapi.testclient import TestClient

from main import app


def _base(**overrides):
    payload = {
        "request_id": "rid-restraint",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
        "recent_messages": [],
    }
    payload.update(overrides)
    return payload


def test_direct_prompt_request_prefers_short_answer():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="give me the prompt"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["restraint_policy"] == "short_answer"
    assert "output" in result["domains"]
    assert result["brevity_preferred"] is True
    assert "brief" in result["prompt_overlay"].lower()


def test_recent_messages_latest_user_text_is_used_when_current_user_text_is_omitted():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(
            recent_messages=[
                {"role": "user", "content": "Can you check this?"},
                {"role": "assistant", "content": "What should I look at?"},
                {"role": "user", "content": "give me the prompt"},
            ],
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["restraint_policy"] == "short_answer"
    assert "output" in result["domains"]
    assert result["brevity_preferred"] is True


def test_tense_debugging_preserves_tactical_help_with_affect_restraint():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="I broke production and prod is failing"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["restraint_policy"] == "short_answer"
    assert {"output", "affect"}.issubset(set(result["domains"]))
    assert "tactical" in result["prompt_overlay"].lower()
    assert "humor" not in result["prompt_overlay"].lower()


def test_venting_does_not_force_problem_solving_or_dependency_framing():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="this week has been exhausting"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["restraint_policy"] == "defer_expansion"
    assert {"personalization", "affect"}.issubset(set(result["domains"]))
    assert "problem-solving" in result["prompt_overlay"].lower()
    payload_text = str(result).lower()
    assert "dependency" not in payload_text
    assert "attachment" not in payload_text


def test_ambiguous_request_prefers_clarifying_question_without_filling_gaps():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="fix this"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["restraint_policy"] == "ask_clarifying_question"
    assert result["clarification_preferred"] is True
    assert result["retrieval_suppressed"] is True
    assert result["personalization_suppressed"] is True


def test_retrieval_restraint_is_represented_without_modifying_memory_truth():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="What does this function do?"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["retrieval_suppressed"] is True
    assert "retrieval_not_requested" in result["reason_summary"]
    assert "retrieval" in result["domains"]


def test_personalization_restraint_is_represented_when_not_requested():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="What does this function do?"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["personalization_suppressed"] is True
    assert "personal_framing_not_requested" in result["reason_summary"]


def test_proactive_restraint_is_represented_without_follow_up_request():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="What does this function do?"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["proactive_output_suppressed"] is True
    assert "proactive_not_requested" in result["reason_summary"]


def test_required_guidance_is_preserved_for_safety_or_correctness_markers():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="The database credentials for production are failing."),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert "required_guidance_preserved" in result["reason_summary"]
    assert result["restraint_policy"] == "short_answer"


def test_runtime_event_payload_is_summarized_only():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="I broke production and prod is failing"),
    )

    assert response.status_code == 200
    runtime_session_id = response.json()["runtime_session_id"]

    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    event = next(
        item
        for item in diagnostics.json()["events"]
        if item["event_type"] == "restraint_evaluated"
    )
    payload = event["event_payload_json"]
    assert set(payload.keys()) == {
        "request_id",
        "restraint_policy",
        "domains",
        "reason",
        "confidence",
        "reason_summary",
        "retrieval_suppressed",
        "personalization_suppressed",
        "proactive_output_suppressed",
    }
    assert "current_user_text" not in str(payload)
    assert "prod is failing" not in str(payload)


def test_runtime_turn_integration_updates_turn_policy_and_attaches_event():
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
        "/v1/runtime/restraint/evaluate",
        json=_base(
            request_id="rid-turn-restraint",
            runtime_session_id=runtime_session_id,
            runtime_turn_id=runtime_turn_id,
            current_user_text="give me the prompt",
        ),
    )

    assert response.status_code == 200
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    body = diagnostics.json()
    assert body["latest_turn"]["restraint_policy"] == "short_answer"
    event = next(
        item
        for item in body["events"]
        if item["event_type"] == "restraint_evaluated"
    )
    assert event["runtime_turn_id"] == runtime_turn_id
