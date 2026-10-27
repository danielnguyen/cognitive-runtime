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


@pytest.mark.parametrize("endpoint", [
    "/v1/runtime/identity/resolve", "/v2/runtime/identity/resolve",
])
def test_selection_commit_rechecks_current_turn_after_resolution(monkeypatch, endpoint):
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
    assert client.post(endpoint, json=payload).status_code == 409
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


_IDENTITY_V2 = "/v2/runtime/identity/resolve"


def _versioned_identity(client, payload, **extra):
    response = client.post(_IDENTITY_V2, json={**payload, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def _canonical_digest(value):
    import json
    from hashlib import sha256

    return "sha256:" + sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def test_versioned_identity_binds_actual_inputs_result_and_committed_event(monkeypatch):
    import services.runtime_identity as identity

    client = TestClient(app)
    payload = _strict_turn(client, surface="not_registered")
    observed = {}
    original = identity.resolve_runtime_identity

    def capture(body, *, producer_inputs=None):
        result = original(body, producer_inputs=producer_inputs)
        observed.update(producer_inputs)
        return result

    monkeypatch.setattr(identity, "resolve_runtime_identity", capture)
    response = _versioned_identity(client, payload)
    result, provenance = response["result"], response["provenance"]
    selection = result["persona_selection"]
    assert response["schema_version"] == provenance["schema_version"] == "runtime-identity.v2"
    assert provenance["policy_version"] == "strict-persona-selection.v1"
    assert provenance["producer"] == "cognitive-runtime"
    assert provenance["endpoint"] == _IDENTITY_V2
    assert provenance["decision"] == "strict_persona_selection"
    assert provenance["freshness"] == "observed_inputs_only"
    for key in ("request_id", "owner_id", "conversation_id", "surface",
                "runtime_session_id", "runtime_turn_id"):
        assert provenance[key] == selection[key] == payload[key]
    assert provenance["surface"] == "not_registered"
    assert result["surface_binding"]["surface_id"] == "unknown"
    assert provenance["thread_revision"] == selection["thread_revision"]
    assert provenance["selection_ref"] == selection["selection_ref"]
    assert provenance["governance_event_ref"] == selection["governance_event_ref"]
    assert observed.pop("selection_event_ref") == provenance["selection_event_ref"]
    assert observed["selected_persona"] == result["persona"]
    assert observed["surface_binding"] == result["surface_binding"]
    assert observed["runtime_session"] == result["runtime_session"]
    assert observed["request"]["persona_selection_mode"] == "strict"
    assert observed["contextual_hint"] == "technical_operator"
    assert observed["contextual_persona"] == "technical_architect"
    assert observed["proposed_persona_present"] is True
    assert provenance["input_digest"] == _canonical_digest(observed)
    assert provenance["result_digest"] == _canonical_digest(result)
    assert provenance["provenance_ref"] == _canonical_digest({
        key: value for key, value in provenance.items() if key != "provenance_ref"
    })
    events = runtime_state_repository().list_events_for_tests(payload["runtime_session_id"])
    selected = [event for event in events if event.event_type == "persona_selection_resolved"]
    assert len(selected) == 1
    assert selected[0].event_id == provenance["selection_event_ref"]
    assert selected[0].event_payload_json == selection
    assert observed["governance"] == next(event.model_dump() for event in events if (
        event.event_id == provenance["governance_event_ref"]
    ))


def test_versioned_identity_identical_inputs_revalidate_without_cache_or_selection_mutation():
    client = TestClient(app)
    payload = _strict_turn(client)
    legacy_strict = client.post("/v1/runtime/identity/resolve", json=payload).json()
    first = _versioned_identity(client, payload)
    second = _versioned_identity(client, payload, expected_provenance=first["provenance"])
    assert first == second
    assert first["result"] == legacy_strict
    clear_states_for_tests(db_path=runtime_state_repository().db_path)
    assert _versioned_identity(client, payload) == first
    events = runtime_state_repository().list_events_for_tests(payload["runtime_session_id"])
    assert sum(event.event_type == "persona_selection_resolved" for event in events) == 1
    resolved = [event for event in events if event.event_type == "identity_resolved"]
    assert len(resolved) == 4  # Every invocation still emits its existing identity event.
    assert all(event.event_payload_json == resolved[0].event_payload_json for event in resolved)
    assert all("provenance" not in str(event.event_payload_json) for event in events)


@pytest.mark.parametrize("field,value", [
    ("request_id", "other_request"), ("owner_id", "other_owner"),
    ("conversation_id", "other_conversation"), ("surface", "vscode"),
    ("runtime_session_id", "other_session"), ("runtime_turn_id", "other_turn"),
    ("expected_thread_revision", 0),
])
def test_versioned_identity_revalidation_rejects_changed_request_scope(field, value):
    client = TestClient(app)
    payload = _strict_turn(client)
    first = _versioned_identity(client, payload)
    response = client.post(_IDENTITY_V2, json={
        **payload, field: value, "expected_provenance": first["provenance"],
    })
    assert response.status_code == 409
    assert response.json() == {"detail": "persona_selection_rejected"}


@pytest.mark.parametrize("change", [
    "governance_payload", "governance_event", "persona", "binding", "proposed_persona",
    "selected_persona", "revision", "request", "session",
])
def test_versioned_identity_authoritative_changes_invalidate_observed_binding(change):
    import json

    client = TestClient(app)
    payload = _strict_turn(client)
    first = _versioned_identity(client, payload)
    state = runtime_state_repository()
    if change in {"governance_payload", "revision", "session"}:
        with state._connect() as conn:
            if change == "revision":
                conn.execute("UPDATE conversation_runtime_threads SET revision = revision + 1")
            elif change == "session":
                conn.execute("UPDATE conversation_runtime_sessions SET active_mode = 'focused'")
            else:
                event = next(event for event in state.list_events_for_tests(
                    payload["runtime_session_id"],
                ) if event.event_type == "interaction_governance_evaluated")
                conn.execute(
                    "UPDATE conversation_runtime_events SET event_payload_json = ? "
                    "WHERE event_id = ?",
                    (json.dumps({**event.event_payload_json, "reason_codes": ["changed"]}),
                     event.event_id),
                )
    elif change == "governance_event":
        assert client.post("/v1/runtime/interaction-governance/evaluate", json={
            **payload, "current_user_text": "I broke the server and prod is failing",
        }).status_code == 200
    elif change == "request":
        payload = {**payload, "active_mode": "focused"}
    else:
        with companion_contracts_repository()._connect() as conn:
            if change == "persona":
                conn.execute("UPDATE persona_profiles SET description = 'changed' "
                             "WHERE persona_id = 'general_assistant'")
            elif change == "binding":
                conn.execute("UPDATE surface_bindings SET response_length = 'brief' "
                             "WHERE surface_id = 'web'")
            elif change == "proposed_persona":
                conn.execute(
                    "DELETE FROM persona_profiles WHERE persona_id = 'technical_architect'",
                )
            else:
                conn.execute(
                    "UPDATE surface_bindings SET default_persona_id = 'personal_companion' "
                    "WHERE surface_id = 'web'",
                )
    response = client.post(_IDENTITY_V2, json={
        **payload, "expected_provenance": first["provenance"],
    })
    assert response.status_code == 409
    assert response.json() == {"detail": "persona_selection_rejected"}
    if change in {"governance_payload", "persona", "binding", "request", "session"}:
        fresh = _versioned_identity(client, payload)
        assert fresh["provenance"]["input_digest"] != first["provenance"]["input_digest"]
        assert fresh["provenance"]["provenance_ref"] != first["provenance"]["provenance_ref"]
        assert fresh["result"]["persona_selection"] == first["result"]["persona_selection"]
        # This proves the digest covers inputs beyond both HTTP body and selection_ref.
    selections = [event for event in state.list_events_for_tests(payload["runtime_session_id"])
                  if event.event_type == "persona_selection_resolved"]
    assert len(selections) == 1
    assert selections[0].event_payload_json == first["result"]["persona_selection"]


def test_versioned_identity_rejects_every_missing_or_conflicting_provenance_field_at_runtime():
    client = TestClient(app)
    payload = _strict_turn(client)
    provenance = _versioned_identity(client, payload)["provenance"]
    invalid = [None, {}, [], "protected_sentinel", {**provenance, "unknown": "protected_sentinel"}]
    for field in provenance:
        invalid.append({key: value for key, value in provenance.items() if key != field})
        value = provenance[field]
        invalid.append({**provenance, field: value + 1 if type(value) is int else "unknown"})
    # Also exercise well-formed but conflicting references and digests.
    for field in ("input_digest", "result_digest", "provenance_ref"):
        invalid.append({**provenance, field: "sha256:" + "0" * 64})
    invalid.append({**provenance, "selection_ref": "psel_" + "0" * 32})
    for candidate in invalid:
        response = client.post(_IDENTITY_V2, json={**payload, "expected_provenance": candidate})
        assert response.status_code == 409, candidate
        assert response.json() == {"detail": "persona_selection_rejected"}
        assert "protected_sentinel" not in response.text
    assert _versioned_identity(client, payload, expected_provenance=provenance)["provenance"] == (
        provenance
    )


def test_versioned_identity_preserves_v1_default_and_requires_strict_turn():
    client = TestClient(app)
    legacy = client.post("/v1/runtime/identity/resolve", json=_base("web")).json()
    assert set(legacy) == {
        "runtime_session", "surface_binding", "persona", "runtime_identity", "trace",
        "selection_contract", "persona_selection",
    }
    assert legacy["selection_contract"] == "legacy_unbound"
    assert legacy["persona_selection"] is None
    assert client.post(_IDENTITY_V2, json=_base("web")).status_code == 422
    payload = _strict_turn(client)
    for extra in ({"persona_selection_mode": "legacy"}, {"allow_requested_persona_bypass": True},
                  {"active_persona_id": "personal_companion"}):
        assert client.post(_IDENTITY_V2, json={**payload, **extra}).status_code == 422
    strict = client.post("/v1/runtime/identity/resolve", json=payload).json()
    assert set(strict) == set(legacy)
    assert _versioned_identity(client, payload)["result"] == strict


def test_versioned_identity_concurrent_calls_share_one_persisted_authority():
    from concurrent.futures import ThreadPoolExecutor

    from models import RuntimeIdentityResolveRequestV2
    from services.runtime_identity import resolve_runtime_identity_v2

    client = TestClient(app)
    payload = _strict_turn(client)
    request = RuntimeIdentityResolveRequestV2(**payload)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: resolve_runtime_identity_v2(request), range(2)))
    assert responses[0] == responses[1]
    events = runtime_state_repository().list_events_for_tests(payload["runtime_session_id"])
    selections = [event for event in events if event.event_type == "persona_selection_resolved"]
    assert len(selections) == 1
    assert responses[0].provenance.selection_event_ref == selections[0].event_id


def test_versioned_identity_never_reuses_completed_turn_or_transfers_to_next_turn():
    client = TestClient(app)
    payload = _strict_turn(client)
    provenance = _versioned_identity(client, payload)["provenance"]
    assert client.post("/v1/runtime/turns/complete", json={
        "request_id": payload["request_id"], "runtime_session_id": payload["runtime_session_id"],
        "runtime_turn_id": payload["runtime_turn_id"], "turn_status": "completed",
    }).status_code == 200
    assert client.post(_IDENTITY_V2, json={
        **payload, "expected_provenance": provenance,
    }).status_code == 409
    next_turn = _strict_turn(client)
    assert next_turn["runtime_turn_id"] != payload["runtime_turn_id"]
    assert client.post(_IDENTITY_V2, json={
        **next_turn, "expected_provenance": provenance,
    }).status_code == 409
    fresh = _versioned_identity(client, next_turn)
    assert fresh["provenance"]["selection_event_ref"] != provenance["selection_event_ref"]


def test_versioned_identity_provenance_is_bounded_and_does_not_disclose_producer_inputs():
    import json

    client = TestClient(app)
    payload = _strict_turn(client, text="private_user_text_sentinel")
    repo = runtime_state_repository()
    with repo._connect() as conn:
        event = next(event for event in repo.list_events_for_tests(payload["runtime_session_id"])
                     if event.event_type == "interaction_governance_evaluated")
        conn.execute(
            "UPDATE conversation_runtime_events SET event_payload_json = ? WHERE event_id = ?",
            (json.dumps({**event.event_payload_json, "internal": "protected_state_sentinel"}),
             event.event_id),
        )
    result = _versioned_identity(client, payload)
    assert len(json.dumps(result["provenance"])) < 2500
    assert "private_user_text_sentinel" not in json.dumps(result)
    assert "protected_state_sentinel" not in json.dumps(result)
    diagnostics = client.get(f"/v1/runtime/sessions/{payload['runtime_session_id']}").json()
    identity_events = [event for event in diagnostics["events"] if event["event_type"] in {
        "persona_selection_resolved", "identity_resolved",
    }]
    assert "private_user_text_sentinel" not in json.dumps(identity_events)
    assert "protected_state_sentinel" not in json.dumps(identity_events)
    assert "current_user_text" not in json.dumps(result["provenance"])


def test_versioned_identity_retains_containment_and_companion_authority():
    client = TestClient(app)
    before = client.post("/v1/companion/policy/compile", json=_base("web")).json()
    payload = _strict_turn(client)
    response = _versioned_identity(client, payload, requested_persona_id="personal_companion")
    selection = response["result"]["persona_selection"]
    assert selection["active_persona_id"] == "general_assistant"
    assert selection["contextual_activation"] is False
    assert selection["explicit_selection_verified"] is False
    containment = {**payload, "persona_selection_ref": selection["selection_ref"],
                   "current_user_text": "I broke the server and prod is failing"}
    result = client.post("/v1/runtime/persona-containment/evaluate", json=containment)
    assert result.status_code == 200
    assert result.json()["persona_selection"] == selection
    assert result.json()["result"]["allowed_memory_domains"] == ["general"]
    assert result.json()["result"]["cross_scope_access_allowed"] is False
    rejected = client.post("/v1/runtime/persona-containment/evaluate", json={
        **containment, "active_persona_id": "technical_architect",
    })
    assert rejected.status_code == 409
    after = client.post("/v1/companion/policy/compile", json=_base("web")).json()
    assert (before["profile_id"], before["profile_version"]) == (
        after["profile_id"], after["profile_version"],
    )


@pytest.mark.parametrize("fault", ["missing_event", "conflicting_event", "storage_failure"])
def test_versioned_identity_requires_actual_committed_evidence_and_propagates_storage_failure(
    monkeypatch, fault,
):
    import sqlite3

    client = TestClient(app)
    payload = _strict_turn(client)
    repo = runtime_state_repository()
    original = repo.persona_selection_events

    def corrupt_commit(**kwargs):
        if kwargs.get("selection") is not None and fault == "storage_failure":
            raise sqlite3.OperationalError("protected_storage_sentinel")
        session, revision, events = original(**kwargs)
        if kwargs.get("selection") is not None:
            if fault == "missing_event":
                events = [event for event in events if (
                    event.event_type != "persona_selection_resolved"
                )]
            else:
                events = [event.model_copy(update={"event_payload_json": {}}) if (
                    event.event_type == "persona_selection_resolved"
                ) else event for event in events]
        return session, revision, events

    monkeypatch.setattr(repo, "persona_selection_events", corrupt_commit)
    if fault == "storage_failure":
        with pytest.raises(sqlite3.OperationalError):
            client.post(_IDENTITY_V2, json=payload)
    else:
        response = client.post(_IDENTITY_V2, json=payload)
        assert response.status_code == 409
        assert response.json() == {"detail": "persona_selection_rejected"}
    assert not any(event.event_type == "identity_resolved" for event in (
        repo.list_events_for_tests(payload["runtime_session_id"])
    ))
