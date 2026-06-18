from __future__ import annotations

from fastapi.testclient import TestClient
from main import app


def _base(**overrides):
    payload = {
        "request_id": "rid-persona-containment",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
        "recent_messages": [],
    }
    payload.update(overrides)
    return payload


def test_technical_request_uses_technical_persona_and_keeps_domains_narrow():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/persona-containment/evaluate",
        json=_base(current_user_text="Refactor this API function and update the project spec."),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["active_persona_id"] == "technical_architect"
    assert result["capability_domain"] == "technical"
    assert "technical" in result["allowed_memory_domains"]
    assert "project" in result["allowed_memory_domains"]
    assert "technical" in result["allowed_tool_domains"]
    assert "finance" in result["blocked_memory_domains"]
    assert result["cross_scope_access_allowed"] is False


def test_vehicle_request_uses_vehicle_capability_and_blocks_unrelated_domains():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/persona-containment/evaluate",
        json=_base(
            surface="web",
            current_user_text="My car needs an oil change and brake inspection soon.",
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["active_persona_id"] == "general_assistant"
    assert result["capability_domain"] == "vehicle_maintenance"
    assert "vehicle_maintenance" in result["allowed_memory_domains"]
    assert "work_professional" in result["blocked_memory_domains"]
    assert "health" in result["blocked_memory_domains"]
    assert "finance" in result["blocked_memory_domains"]
    assert result["cross_scope_access_allowed"] is False


def test_cross_scope_is_blocked_by_default():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/persona-containment/evaluate",
        json=_base(current_user_text="Refactor this API function."),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["cross_scope_access_allowed"] is False
    assert result["cross_scope_reason"] == "not_requested"
    assert "work_professional" in result["blocked_memory_domains"]


def test_explicit_cross_scope_request_allows_bridging_with_reason():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/persona-containment/evaluate",
        json=_base(
            current_user_text="Refactor this API and compare this with my work context.",
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["cross_scope_access_allowed"] is True
    assert result["cross_scope_reason"] == "explicit_bridge_request_detected"
    assert "work_professional" in result["allowed_memory_domains"]


def test_display_identity_is_not_accepted_as_canonical_persona_id():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/persona-containment/evaluate",
        json=_base(
            surface="web",
            requested_persona_id="Technical Architect",
            current_user_text="What should I do next?",
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["active_persona_id"] == "general_assistant"
    assert "requested_persona_not_canonical" in result["reason_summary"]


def test_conservative_fallback_does_not_broaden_scope_silently():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/persona-containment/evaluate",
        json=_base(surface="not_registered"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["active_persona_id"] == "general_assistant"
    assert result["capability_domain"] == "general"
    assert result["allowed_memory_domains"] == ["general"]
    assert "technical" in result["blocked_memory_domains"]
    assert result["cross_scope_access_allowed"] is False


def test_runtime_event_summary_excludes_raw_private_context():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/persona-containment/evaluate",
        json=_base(current_user_text="Bring in health context for this question."),
    )

    assert response.status_code == 200
    runtime_session_id = response.json()["runtime_session_id"]

    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    event = next(
        item
        for item in diagnostics.json()["events"]
        if item["event_type"] == "persona_containment_evaluated"
    )
    payload = event["event_payload_json"]
    assert set(payload.keys()) == {
        "request_id",
        "active_persona_id",
        "capability_domain",
        "allowed_memory_domains",
        "blocked_memory_domains",
        "allowed_tool_domains",
        "cross_scope_access_allowed",
        "cross_scope_reason",
        "reason_summary",
    }
    assert "current_user_text" not in str(payload)
    assert "Bring in health context" not in str(payload)


def test_unmapped_domain_does_not_broaden_scope():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/persona-containment/evaluate",
        json=_base(surface="web", current_user_text="Bring in astrology context."),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["cross_scope_access_allowed"] is False
    assert result["cross_scope_reason"] == "domain_not_policy_mapped"
    assert "domain_not_policy_mapped" in result["reason_summary"]
    assert "astrology" in result["blocked_memory_domains"]
    assert result["allowed_memory_domains"] == ["general"]


def test_runtime_turn_integration_records_persona_containment_event():
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
        "/v1/runtime/persona-containment/evaluate",
        json=_base(
            request_id="rid-turn-persona",
            runtime_session_id=runtime_session_id,
            runtime_turn_id=runtime_turn_id,
            current_user_text="Refactor this API and compare this with my work context.",
        ),
    )

    assert response.status_code == 200
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    event = next(
        item
        for item in diagnostics.json()["events"]
        if item["event_type"] == "persona_containment_evaluated"
    )
    assert event["runtime_turn_id"] == runtime_turn_id
    assert event["event_payload_json"]["active_persona_id"] == "technical_architect"
