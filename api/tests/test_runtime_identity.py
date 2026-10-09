import pytest
from fastapi.testclient import TestClient
from main import app
from services.companion_contracts import companion_contracts_repository
from services.runtime_state import clear_states_for_tests, runtime_state_repository


def setup_function():
    clear_states_for_tests()


def _base(surface: str = "vscode"):
    return {
        "request_id": "rid-identity",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": surface,
    }


def test_identity_resolution_uses_surface_binding_default_persona():
    client = TestClient(app)

    response = client.post("/v1/runtime/identity/resolve", json=_base("vscode"))

    assert response.status_code == 200
    body = response.json()
    assert body["surface_binding"]["surface_id"] == "vscode"
    assert body["surface_binding"]["default_persona_id"] == "technical_architect"
    assert body["persona"]["persona_id"] == "technical_architect"
    assert body["trace"]["persona_resolution_reason"] == "surface_binding"
    assert body["trace"]["persona_override_source"] == "none"
    assert body["runtime_identity"]["persona_owns_durable_memory"] is False
    assert "persona=technical_architect" in body["runtime_identity"]["content"]


def test_identity_resolution_ignores_requested_persona_without_override_permission():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/identity/resolve",
        json={**_base("web"), "requested_persona_id": "personal_companion"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["persona"]["persona_id"] == "general_assistant"
    assert body["trace"]["persona_resolution_reason"] == "surface_binding"
    assert body["trace"]["persona_override_source"] == "none"


def test_identity_resolution_supports_internal_test_only_requested_persona_bypass():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/identity/resolve",
        json={
            **_base("web"),
            "requested_persona_id": "personal_companion",
            "allow_requested_persona_bypass": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["persona"]["persona_id"] == "personal_companion"
    assert body["trace"]["persona_resolution_reason"] == "requested_persona_id"
    assert body["trace"]["persona_override_source"] == "internal_test"


def test_identity_resolution_falls_back_to_unknown_surface_binding():
    client = TestClient(app)

    response = client.post("/v1/runtime/identity/resolve", json=_base("not_registered"))

    assert response.status_code == 200
    body = response.json()
    assert body["surface_binding"]["surface_id"] == "unknown"
    assert body["persona"]["persona_id"] == "general_assistant"
    assert body["trace"]["surface_id"] == "unknown"


def test_identity_resolution_records_event_in_runtime_session_diagnostics():
    client = TestClient(app)

    started = client.post(
        "/v1/runtime/turns/start",
        json={**_base("dev"), "input_message_id": "msg-1"},
    )

    assert started.status_code == 200
    runtime_session_id = started.json()["runtime_session"]["runtime_session_id"]
    response = client.post(
        "/v1/runtime/identity/resolve",
        json={**_base("dev"), "runtime_session_id": runtime_session_id},
    )

    assert response.status_code == 200
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}")

    assert diagnostics.status_code == 200
    event_types = [event["event_type"] for event in diagnostics.json()["events"]]
    assert event_types.count("session_resolved") == 1
    assert "identity_resolved" in event_types


def test_identity_resolution_does_not_introduce_world_state_or_relationship_fields():
    client = TestClient(app)

    response = client.post("/v1/runtime/identity/resolve", json=_base("dev"))

    assert response.status_code == 200
    payload_text = str(response.json())
    assert "world_state" not in payload_text
    assert "relationship_edges" not in payload_text


def _strict_turn(client, *, surface="web", text="I broke the server and prod is failing"):
    payload = _base(surface)
    started = client.post("/v1/runtime/turns/start", json=payload)
    assert started.status_code == 200
    payload.update(
        runtime_session_id=started.json()["runtime_session"]["runtime_session_id"],
        runtime_turn_id=started.json()["runtime_turn"]["runtime_turn_id"],
    )
    governance = client.post(
        "/v1/runtime/interaction-governance/evaluate",
        json={**payload, "current_user_text": text},
    )
    assert governance.status_code == 200
    return {**payload, "persona_selection_mode": "strict"}


@pytest.mark.parametrize("surface,persona,source", [
    ("web", "general_assistant", "surface_binding"),
    ("vscode", "technical_architect", "surface_binding"),
    ("not_registered", "general_assistant", "conservative_fallback"),
    ("unknown", "general_assistant", "conservative_fallback"),
])
def test_strict_selection_keeps_surface_authority_with_advisory_technical_proposal(
    surface, persona, source,
):
    client = TestClient(app)
    payload = _strict_turn(client, surface=surface)
    response = client.post("/v1/runtime/identity/resolve", json=payload)
    assert response.status_code == 200
    result = response.json()
    selection = result["persona_selection"]
    assert result["selection_contract"] == "strict_turn"
    assert result["runtime_identity"]["active_persona_id"] == persona
    assert selection["active_persona_id"] == persona
    assert selection["selection_source"] == source
    assert selection["surface"] == surface
    assert selection["proposed_persona_id"] == "technical_architect"
    assert selection["proposal_source"] == "interaction_governance"
    assert selection["proposal_status"] == "advisory"
    assert selection["proposal_reason"] == "contextual_activation_not_enabled"
    assert selection["contextual_activation"] is False
    assert selection["explicit_selection_verified"] is False
    assert selection["thread_revision"] >= 1
    assert all(selection[key] == payload[key] for key in (
        "request_id", "owner_id", "conversation_id", "surface",
        "runtime_session_id", "runtime_turn_id",
    ))


@pytest.mark.parametrize("text,proposed,status", [
    ("That was wrong in the report.", "personal_companion", "advisory"),
    ("What is 2+2?", None, "none"),
])
def test_strict_supportive_and_absent_proposals_do_not_activate(text, proposed, status):
    client = TestClient(app)
    payload = _strict_turn(client, text=text)
    response = client.post("/v1/runtime/identity/resolve", json=payload)
    assert response.status_code == 200
    selection = response.json()["persona_selection"]
    assert selection["active_persona_id"] == "general_assistant"
    assert selection["proposed_persona_id"] == proposed
    assert selection["proposal_status"] == status


def test_strict_unmapped_proposal_is_rejected_without_activation(monkeypatch):
    client = TestClient(app)
    payload = _strict_turn(client)
    monkeypatch.setattr(
        "services.runtime_identity.persona_scope_hint_for_kind", lambda kind: "unmapped",
    )
    response = client.post("/v1/runtime/identity/resolve", json=payload)
    assert response.status_code == 200
    selection = response.json()["persona_selection"]
    assert selection["proposal_status"] == "rejected"
    assert selection["proposal_reason"] == "proposal_unmapped"
    assert selection["proposed_persona_id"] is None
    assert selection["active_persona_id"] == "general_assistant"


@pytest.mark.parametrize("field,value", [
    ("request_id", "other_request"), ("owner_id", "other_owner"),
    ("conversation_id", "other_conversation"), ("surface", "vscode"),
    ("runtime_session_id", "missing_session"), ("runtime_turn_id", "missing_turn"),
    ("expected_thread_revision", 0),
])
def test_strict_identity_rejects_mismatched_scope_without_substitution(field, value):
    client = TestClient(app)
    payload = _strict_turn(client)
    response = client.post("/v1/runtime/identity/resolve", json={**payload, field: value})
    assert response.status_code in {404, 409}
    assert response.json() == {"detail": "persona_selection_rejected"}
    events = runtime_state_repository().list_events_for_tests(payload["runtime_session_id"])
    assert not any(event.event_type == "persona_selection_resolved" for event in events)


@pytest.mark.parametrize("extra", [
    {"persona_scope_hint": "supportive_listener"},
    {"selection_source": "explicit_user"},
    {"active_persona_id": "personal_companion"},
    {"allow_requested_persona_bypass": True},
    {"expected_thread_revision": True},
])
def test_strict_identity_rejects_caller_source_hint_and_test_bypass(extra):
    client = TestClient(app)
    payload = _strict_turn(client)
    assert client.post("/v1/runtime/identity/resolve", json={**payload, **extra}).status_code == 422


def test_strict_requested_selection_is_unverified_even_when_binding_permits_override():
    client = TestClient(app)
    payload = _strict_turn(client)
    repo = companion_contracts_repository()
    with repo._connect() as conn:
        conn.execute(
            "UPDATE surface_bindings SET allow_user_persona_override = 1 WHERE surface_id = 'web'",
        )
    response = client.post(
        "/v1/runtime/identity/resolve",
        json={**payload, "requested_persona_id": "personal_companion"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["persona_selection"]["requested_selection_status"] == "unverified_rejected"
    assert result["persona"]["persona_id"] == "general_assistant"
    assert result["trace"]["persona_override_source"] == "none"
    legacy = client.post(
        "/v1/runtime/identity/resolve",
        json={**_base("web"), "requested_persona_id": "personal_companion"},
    )
    assert legacy.status_code == 200
    assert legacy.json()["persona"]["persona_id"] == "personal_companion"
    assert legacy.json()["selection_contract"] == "legacy_unbound"


def test_strict_selection_requires_governance_for_the_exact_started_request():
    client = TestClient(app)
    started = client.post("/v1/runtime/turns/start", json=_base("web")).json()
    payload = {
        **_base("web"), "persona_selection_mode": "strict",
        "runtime_session_id": started["runtime_session"]["runtime_session_id"],
        "runtime_turn_id": started["runtime_turn"]["runtime_turn_id"],
    }
    assert client.post("/v1/runtime/identity/resolve", json=payload).status_code == 409
    client.post("/v1/runtime/interaction-governance/evaluate", json={
        **payload, "request_id": "wrong", "current_user_text": "That was wrong in the report.",
    })
    assert client.post("/v1/runtime/identity/resolve", json=payload).status_code == 409


@pytest.mark.parametrize("status", ["completed", "abandoned"])
def test_strict_selection_rejects_terminal_turn(status):
    client = TestClient(app)
    payload = _strict_turn(client)
    completed = client.post("/v1/runtime/turns/complete", json={
        "request_id": payload["request_id"], "runtime_session_id": payload["runtime_session_id"],
        "runtime_turn_id": payload["runtime_turn_id"], "turn_status": status,
    })
    assert completed.status_code == 200
    assert client.post("/v1/runtime/identity/resolve", json=payload).status_code == 409


@pytest.mark.parametrize("corruption", ["binding", "missing_binding", "persona", "governance"])
def test_strict_malformed_or_unavailable_authority_is_bounded(corruption):
    client = TestClient(app)
    payload = _strict_turn(client)
    if corruption == "governance":
        repo = runtime_state_repository()
        with repo._connect() as conn:
            conn.execute(
                "UPDATE conversation_runtime_events SET event_payload_json = ? "
                "WHERE event_type = 'interaction_governance_evaluated'",
                ('{"interaction_kind":"private_sentinel"}',),
            )
    else:
        repo = companion_contracts_repository()
        with repo._connect() as conn:
            if corruption == "binding":
                conn.execute(
                    "UPDATE surface_bindings SET default_persona_id = '' WHERE surface_id = 'web'",
                )
            elif corruption == "missing_binding":
                conn.execute("DELETE FROM surface_bindings WHERE surface_id IN ('web', 'unknown')")
            else:
                conn.execute(
                    "UPDATE surface_bindings SET default_persona_id = 'unregistered' "
                    "WHERE surface_id = 'web'",
                )
    response = client.post("/v1/runtime/identity/resolve", json=payload)
    if corruption == "persona":
        assert response.status_code == 200
        assert response.json()["persona_selection"]["selection_source"] == "conservative_fallback"
        assert response.json()["persona"]["persona_id"] == "general_assistant"
    else:
        assert response.status_code == 409
        assert response.json() == {"detail": "persona_selection_rejected"}


def test_strict_selection_is_idempotent_and_events_contain_only_structural_provenance():
    client = TestClient(app)
    text = "I broke the server and prod is failing: private_input_sentinel"
    payload = _strict_turn(client, text=text)
    first = client.post("/v1/runtime/identity/resolve", json=payload)
    second = client.post("/v1/runtime/identity/resolve", json=payload)
    assert first.status_code == second.status_code == 200
    assert first.json()["persona_selection"] == second.json()["persona_selection"]
    events = runtime_state_repository().list_events_for_tests(payload["runtime_session_id"])
    selected = [event for event in events if event.event_type == "persona_selection_resolved"]
    assert len(selected) == 1
    assert "private_input_sentinel" not in str([event.model_dump() for event in events])
    assert "current_user_text" not in selected[0].event_payload_json
    assert "permissions" not in selected[0].event_payload_json


def test_strict_identity_preserves_canonical_companion_profile_provenance():
    client = TestClient(app)
    before = client.post("/v1/companion/policy/compile", json=_base("web"))
    assert before.status_code == 200
    payload = _strict_turn(client, text="That was wrong in the report.")
    assert client.post("/v1/runtime/identity/resolve", json=payload).status_code == 200
    after = client.post("/v1/companion/policy/compile", json=_base("web"))
    assert after.status_code == 200
    for key in ("profile_id", "profile_version"):
        assert before.json()[key] == after.json()[key]


def test_strict_identity_rejects_existing_turn_from_other_session():
    client = TestClient(app)
    payload = _strict_turn(client)
    other = client.post("/v1/runtime/turns/start", json={
        **_base("web"), "owner_id": "other_owner", "conversation_id": "other_conversation",
    }).json()
    response = client.post("/v1/runtime/identity/resolve", json={
        **payload, "runtime_turn_id": other["runtime_turn"]["runtime_turn_id"],
    })
    assert response.status_code == 409


def test_strict_identity_uses_current_revision_and_rejects_noncurrent_pointer():
    client = TestClient(app)
    payload = _strict_turn(client)
    repo = runtime_state_repository()
    with repo._connect() as conn:
        current = conn.execute(
            "SELECT revision FROM conversation_runtime_threads WHERE owner_id = ? "
            "AND conversation_id = ?", (payload["owner_id"], payload["conversation_id"]),
        ).fetchone()["revision"]
    response = client.post("/v1/runtime/identity/resolve", json={
        **payload, "expected_thread_revision": current,
    })
    assert response.status_code == 200
    assert response.json()["persona_selection"]["thread_revision"] == current
    with repo._connect() as conn:
        conn.execute("UPDATE conversation_runtime_threads SET active_runtime_turn_id = NULL")
    assert client.post("/v1/runtime/identity/resolve", json=payload).status_code == 409


def test_strict_identity_conflicting_second_selection_does_not_overwrite_first():
    client = TestClient(app)
    payload = _strict_turn(client)
    first = client.post("/v1/runtime/identity/resolve", json=payload)
    assert first.status_code == 200
    second = client.post("/v1/runtime/identity/resolve", json={
        **payload, "requested_persona_id": "personal_companion",
    })
    assert second.status_code == 409
    selections = [event for event in runtime_state_repository().list_events_for_tests(
        payload["runtime_session_id"],
    ) if event.event_type == "persona_selection_resolved"]
    assert len(selections) == 1
    assert selections[0].event_payload_json == first.json()["persona_selection"]


def test_strict_selection_does_not_default_over_storage_failure(monkeypatch):
    import sqlite3

    from models import RuntimeIdentityResolveRequest
    from services.runtime_identity import resolve_runtime_identity

    client = TestClient(app)
    payload = _strict_turn(client)

    def fail(surface):
        raise sqlite3.OperationalError("private_storage_sentinel")

    monkeypatch.setattr(companion_contracts_repository(), "surface_binding", fail)
    with pytest.raises(sqlite3.OperationalError):
        resolve_runtime_identity(RuntimeIdentityResolveRequest(**payload))


def test_strict_decision_is_single_authority_under_concurrent_identity_calls():
    from concurrent.futures import ThreadPoolExecutor

    from models import RuntimeIdentityResolveRequest
    from services.runtime_identity import resolve_runtime_identity

    client = TestClient(app)
    payload = _strict_turn(client)
    request = RuntimeIdentityResolveRequest(**payload)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: resolve_runtime_identity(request), range(2)))
    assert responses[0].persona_selection == responses[1].persona_selection
    events = runtime_state_repository().list_events_for_tests(payload["runtime_session_id"])
    assert sum(event.event_type == "persona_selection_resolved" for event in events) == 1


def test_selection_commit_rechecks_current_turn_after_resolution(monkeypatch):
    client = TestClient(app)
    payload = _strict_turn(client)
    repo = runtime_state_repository()
    original = repo.persona_selection_events

    def complete_before_commit(**kwargs):
        if kwargs.get("selection") is not None:
            repo.complete_turn(
                request_id=payload["request_id"], runtime_session_id=payload["runtime_session_id"],
                runtime_turn_id=payload["runtime_turn_id"], turn_status="completed",
            )
        return original(**kwargs)

    monkeypatch.setattr(repo, "persona_selection_events", complete_before_commit)
    assert client.post("/v1/runtime/identity/resolve", json=payload).status_code == 409
    assert not any(event.event_type == "persona_selection_resolved" for event in (
        repo.list_events_for_tests(payload["runtime_session_id"])
    ))


def test_strict_selection_cannot_resolve_malformed_registered_persona():
    client = TestClient(app)
    payload = _strict_turn(client)
    with companion_contracts_repository()._connect() as conn:
        conn.execute(
            "UPDATE persona_profiles SET communication_policy_summary_json = ? "
            "WHERE persona_id = 'general_assistant'", ('[7]',),
        )
    response = client.post("/v1/runtime/identity/resolve", json=payload)
    assert response.status_code == 409
    assert not any(event.event_type == "persona_selection_resolved" for event in (
        runtime_state_repository().list_events_for_tests(payload["runtime_session_id"])
    ))
