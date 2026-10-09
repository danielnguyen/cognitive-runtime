from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from main import app
from services.runtime_state import clear_states_for_tests, runtime_state_repository


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


def _strict_selection(client, *, surface="web", text="I broke the server and prod is failing"):
    payload = _base(surface=surface)
    started = client.post("/v1/runtime/turns/start", json={
        key: value for key, value in payload.items() if key != "recent_messages"
    })
    assert started.status_code == 200
    payload.update(
        runtime_session_id=started.json()["runtime_session"]["runtime_session_id"],
        runtime_turn_id=started.json()["runtime_turn"]["runtime_turn_id"],
    )
    assert client.post("/v1/runtime/interaction-governance/evaluate", json={
        **payload, "current_user_text": text,
    }).status_code == 200
    identity = client.post("/v1/runtime/identity/resolve", json={
        key: value for key, value in {
            **payload, "persona_selection_mode": "strict",
        }.items() if key != "recent_messages"
    })
    assert identity.status_code == 200
    selection = identity.json()["persona_selection"]
    return {
        **payload, "current_user_text": text, "persona_selection_mode": "strict",
        "persona_selection_ref": selection["selection_ref"],
    }, selection


@pytest.mark.parametrize("surface,persona,domains", [
    ("web", "general_assistant", {"general"}),
    ("vscode", "technical_architect", {"general", "technical", "project", "infrastructure"}),
    ("unregistered", "general_assistant", {"general"}),
])
def test_strict_containment_uses_same_persona_and_never_unions_proposed_scope(
    surface, persona, domains,
):
    client = TestClient(app)
    payload, selection = _strict_selection(client, surface=surface)
    response = client.post("/v1/runtime/persona-containment/evaluate", json={
        **payload, "active_persona_id": persona, "persona_scope_hint": "technical_operator",
    })
    assert response.status_code == 200
    result = response.json()
    assert result["selection_contract"] == "strict_turn"
    assert result["persona_selection"] == selection
    assert result["result"]["active_persona_id"] == persona
    assert selection["contextual_activation"] is False
    for key in (
        "allowed_memory_domains", "allowed_world_state_domains",
        "allowed_relationship_domains", "allowed_tool_domains",
    ):
        assert set(result["result"][key]) <= domains
    assert set(result["result"]["artifact_access_policy"]["allowed_domains"]) <= domains
    assert result["result"]["cross_scope_access_allowed"] is False


@pytest.mark.parametrize("field,value", [
    ("active_persona_id", "personal_companion"),
    ("requested_persona_id", "personal_companion"),
    ("persona_scope_hint", "supportive_listener"),
    ("request_id", "wrong_request"), ("owner_id", "wrong_owner"),
    ("conversation_id", "wrong_conversation"), ("surface", "vscode"),
    ("runtime_session_id", "wrong_session"), ("runtime_turn_id", "wrong_turn"),
    ("persona_selection_ref", "psel_00000000000000000000000000000000"),
    ("expected_thread_revision", 0),
])
def test_strict_containment_rejects_conflicts_and_cross_bound_selections(field, value):
    client = TestClient(app)
    payload, _ = _strict_selection(client)
    response = client.post(
        "/v1/runtime/persona-containment/evaluate", json={**payload, field: value},
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "persona_selection_rejected"}
    events = runtime_state_repository().list_events_for_tests(payload["runtime_session_id"])
    assert not any(event.event_type == "persona_containment_evaluated" for event in events)


def test_strict_containment_supportive_proposal_and_textual_bridge_cannot_open_personal_scope():
    client = TestClient(app)
    payload, selection = _strict_selection(client, text="That was wrong in the report.")
    assert selection["proposed_persona_id"] == "personal_companion"
    response = client.post("/v1/runtime/persona-containment/evaluate", json={
        **payload, "persona_scope_hint": "supportive_listener",
        "current_user_text": "Connect this with my personal context and finance history.",
    })
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["active_persona_id"] == "general_assistant"
    assert set(result["allowed_memory_domains"]) <= {"general"}
    assert result["cross_scope_access_allowed"] is False
    assert "personal" in result["blocked_memory_domains"]
    assert "finance" in result["blocked_memory_domains"]


@pytest.mark.parametrize("corruption", ["raw_json", "persona", "revision", "governance"])
def test_strict_containment_rejects_corrupted_selection_or_changed_governance(corruption):
    import json

    client = TestClient(app)
    payload, selection = _strict_selection(client)
    repo = runtime_state_repository()
    with repo._connect() as conn:
        if corruption == "governance":
            conn.execute(
                "UPDATE conversation_runtime_events SET event_payload_json = ? "
                "WHERE event_type = 'interaction_governance_evaluated'",
                (json.dumps({
                    "request_id": payload["request_id"], "interaction_kind": "question",
                }),),
            )
        else:
            changed = dict(selection)
            if corruption == "persona":
                changed["active_persona_id"] = "personal_companion"
            if corruption == "revision":
                changed["thread_revision"] += 1
            value = (
                "private_invalid_json_sentinel" if corruption == "raw_json" else json.dumps(changed)
            )
            conn.execute(
                "UPDATE conversation_runtime_events SET event_payload_json = ? "
                "WHERE event_type = 'persona_selection_resolved'", (value,),
            )
    response = client.post("/v1/runtime/persona-containment/evaluate", json=payload)
    assert response.status_code == 409
    assert response.json() == {"detail": "persona_selection_rejected"}


def test_strict_selection_survives_repository_replacement_but_not_turn_completion():
    client = TestClient(app)
    payload, selection = _strict_selection(client)
    clear_states_for_tests(db_path=runtime_state_repository().db_path)
    response = client.post("/v1/runtime/persona-containment/evaluate", json=payload)
    assert response.status_code == 200
    assert response.json()["persona_selection"] == selection
    assert client.post("/v1/runtime/turns/complete", json={
        "request_id": payload["request_id"], "runtime_session_id": payload["runtime_session_id"],
        "runtime_turn_id": payload["runtime_turn_id"], "turn_status": "completed",
    }).status_code == 200
    assert client.post("/v1/runtime/persona-containment/evaluate", json=payload).status_code == 409


def test_strict_mode_requires_selection_and_does_not_accept_legacy_identity_as_authority():
    client = TestClient(app)
    assert client.post("/v1/runtime/persona-containment/evaluate", json={
        **_base(), "persona_selection_mode": "strict",
    }).status_code == 422
    legacy = client.post("/v1/runtime/persona-containment/evaluate", json={
        **_base(surface="web"), "persona_scope_hint": "technical_operator",
    })
    assert legacy.status_code == 200
    assert legacy.json()["selection_contract"] == "legacy_unbound"
    assert legacy.json()["persona_selection"] is None
    assert legacy.json()["result"]["active_persona_id"] == "technical_architect"


@pytest.mark.parametrize("field,value", [
    ("contextual_activation", True), ("contextual_activation", 0),
    ("explicit_selection_verified", True), ("explicit_selection_verified", 0),
    ("thread_revision", True), ("proposal_source", "explicit_user"),
])
def test_selection_model_rejects_fabricated_activation_or_consent(field, value):
    from models import PersonaSelectionDecision
    from pydantic import ValidationError

    client = TestClient(app)
    _, selection = _strict_selection(client)
    with pytest.raises(ValidationError):
        PersonaSelectionDecision.model_validate({**selection, field: value})


def test_containment_commit_rechecks_turn_before_publishing_policy(monkeypatch):
    client = TestClient(app)
    payload, _ = _strict_selection(client)
    repo = runtime_state_repository()
    original = repo.persona_selection_events

    def complete_before_publication(**kwargs):
        if kwargs.get("containment_payload") is not None:
            repo.complete_turn(
                request_id=payload["request_id"], runtime_session_id=payload["runtime_session_id"],
                runtime_turn_id=payload["runtime_turn_id"], turn_status="completed",
            )
        return original(**kwargs)

    monkeypatch.setattr(repo, "persona_selection_events", complete_before_publication)
    response = client.post("/v1/runtime/persona-containment/evaluate", json=payload)
    assert response.status_code == 409
    assert not any(event.event_type == "persona_containment_evaluated" for event in (
        repo.list_events_for_tests(payload["runtime_session_id"])
    ))


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
    assert result["artifact_access_policy"] == {
        "enforcement_mode": "mandatory",
        "allowed_content_classes": ["document", "code"],
        "allowed_domains": result["allowed_memory_domains"],
        "maximum_sensitivity": "high",
        "surface_content_capabilities": ["document", "code"],
        "reason_codes": [
            "artifact_policy_applied",
            "restricted_artifact_access_blocked",
            "persona_content_class_limited",
            "surface_content_class_limited",
        ],
    }


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


def test_mixed_vehicle_project_wording_stays_narrow():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/persona-containment/evaluate",
        json=_base(current_user_text="My project car needs a new engine."),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["capability_domain"] == "vehicle_maintenance"
    assert result["allowed_memory_domains"] == ["general", "vehicle_maintenance"]
    assert "project" in result["blocked_memory_domains"]
    assert "technical" in result["blocked_memory_domains"]
    assert "infrastructure" in result["blocked_memory_domains"]
    assert "multi_domain_signal_conservative_scope" in result["reason_summary"]
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


def test_mixed_work_personal_wording_stays_conservative():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/persona-containment/evaluate",
        json=_base(
            surface="web",
            current_user_text="I'm having trouble with my work-life balance.",
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["cross_scope_access_allowed"] is False
    assert result["cross_scope_reason"] == "not_requested"
    assert not {
        "work_professional",
        "personal",
    }.issubset(set(result["allowed_memory_domains"]))
    assert "multi_domain_signal_conservative_scope" in result["reason_summary"]


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
    assert result["artifact_access_policy"]["allowed_domains"] == result["allowed_memory_domains"]
    assert "work_professional" in result["artifact_access_policy"]["allowed_domains"]
    assert "cross_scope_domain_authorized" in result["artifact_access_policy"]["reason_codes"]
    assert result["artifact_access_policy"]["allowed_content_classes"] == ["document", "code"]
    assert result["artifact_access_policy"]["maximum_sensitivity"] == "high"


def test_connect_bridge_phrase_allows_explicit_cross_scope():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/persona-containment/evaluate",
        json=_base(current_user_text="Connect this to my work context."),
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
    assert result["artifact_access_policy"]["allowed_content_classes"] == [
        "document",
        "image",
        "screenshot",
    ]
    assert "Technical Architect" not in str(result["artifact_access_policy"])


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
    assert result["artifact_access_policy"]["allowed_content_classes"] == []
    assert result["artifact_access_policy"]["surface_content_capabilities"] == []
    assert result["artifact_access_policy"]["allowed_domains"] == ["general"]
    assert result["artifact_access_policy"]["maximum_sensitivity"] == "low"
    assert "unknown_surface_no_artifact_access" in result["artifact_access_policy"]["reason_codes"]
    assert "not_registered" not in str(result["artifact_access_policy"])


def test_runtime_event_summary_excludes_raw_private_context():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/persona-containment/evaluate",
        json=_base(
            current_user_text=(
                "Bring in health context for this question with secret.png, "
                "image/png, artifact bytes, and /tmp/runtime.sqlite."
            ),
        ),
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
        "artifact_access_policy",
        "cross_scope_access_allowed",
        "cross_scope_reason",
        "reason_summary",
    }
    policy = payload["artifact_access_policy"]
    assert policy["enforcement_mode"] == "mandatory"
    assert policy["allowed_content_classes"] == ["document", "code"]
    assert policy["allowed_domains"] == payload["allowed_memory_domains"]
    assert policy["maximum_sensitivity"] == "high"
    assert policy["maximum_sensitivity"] != "restricted"
    assert policy["surface_content_capabilities"] == ["document", "code"]
    assert policy["reason_codes"] == [
        "artifact_policy_applied",
        "restricted_artifact_access_blocked",
        "persona_content_class_limited",
        "surface_content_class_limited",
        "cross_scope_domain_authorized",
    ]
    assert "current_user_text" not in str(payload)
    assert "Bring in health context" not in str(payload)
    assert "secret.png" not in str(payload)
    assert "image/png" not in str(payload)
    assert "artifact bytes" not in str(payload)
    assert "/tmp/runtime.sqlite" not in str(payload)


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
    assert result["artifact_access_policy"]["allowed_domains"] == ["general"]
    assert "astrology" not in result["artifact_access_policy"]["allowed_domains"]


def test_web_general_artifact_policy_allows_only_current_image_classes():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/persona-containment/evaluate",
        json=_base(surface="web", current_user_text="Summarize this screenshot and document."),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    policy = result["artifact_access_policy"]
    assert result["active_persona_id"] == "general_assistant"
    assert policy["allowed_content_classes"] == ["document", "image", "screenshot"]
    assert "audio" not in policy["allowed_content_classes"]
    assert "video" not in policy["allowed_content_classes"]
    assert "other" not in policy["allowed_content_classes"]
    assert policy["maximum_sensitivity"] == "high"
    assert policy["maximum_sensitivity"] != "restricted"
    assert policy["allowed_domains"] == result["allowed_memory_domains"]


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
