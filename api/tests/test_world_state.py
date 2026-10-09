from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from main import app
from services.runtime_state import clear_states_for_tests, runtime_state_repository
from services.world_state import (
    TrustedWorldStateVerifier,
    WorldStateRepository,
    clear_trusted_world_state_verifiers_for_tests,
    configure_trusted_world_state_verifiers_for_tests,
    trusted_world_state_verifiers,
)


def _iso(delta_seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=delta_seconds)).isoformat()


def _base() -> dict[str, object]:
    return {
        "request_id": "rid-world-state",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
    }


def _configure_repo_verifier(**overrides) -> None:
    verifier = TrustedWorldStateVerifier(
        verifier_id="repo-status-revalidator",
        verification_source_type="tool_output",
        allowed_source_refs=frozenset({"repo-status-revalidator", "status-check:bounded"}),
        max_authority="verified_tool_output",
        allowed_domains=frozenset({"active_repository"}),
        allowed_attributes=frozenset({"branch_status"}),
        max_confidence=0.99,
        max_freshness_state="fresh",
        max_ttl_seconds=600,
        max_revalidation_interval_seconds=300,
    )
    configure_trusted_world_state_verifiers_for_tests([
        TrustedWorldStateVerifier(**{**verifier.__dict__, **overrides})
    ])


def _registry_yaml(**overrides) -> str:
    values = {
        "verifier_id": "repo-status-revalidator",
        "verification_source_type": "tool_output",
        "allowed_source_refs": ["status-check:bounded"],
        "max_authority": "verified_tool_output",
        "allowed_domains": ["active_repository"],
        "allowed_attributes": ["branch_status"],
        "allowed_entity_ids": [],
        "max_confidence": 0.99,
        "max_ttl_seconds": 600,
        "max_revalidation_interval_seconds": 300,
        "max_freshness_state": "fresh",
    }
    values.update(overrides)
    lines = ["verifiers:", "  - verifier_id: " + str(values["verifier_id"])]
    for key in (
        "verification_source_type",
        "max_authority",
        "max_confidence",
        "max_ttl_seconds",
        "max_revalidation_interval_seconds",
        "max_freshness_state",
    ):
        lines.append(f"    {key}: {values[key]}")
    for key in (
        "allowed_source_refs",
        "allowed_domains",
        "allowed_attributes",
        "allowed_entity_ids",
    ):
        lines.append(f"    {key}:")
        for item in values[key]:
            lines.append(f"      - {item}")
    return "\n".join(lines) + "\n"


def _claim(**overrides) -> dict[str, object]:
    claim = {
        "entity_id": "repo:primary",
        "entity_type": "repository",
        "domain": "active_repository",
        "attribute": "branch_status",
        "value_json": {"branch": "main", "status": "failing"},
        "source_type": "tool_output",
        "source_ref": "pytest",
        "confidence": 0.95,
        "freshness_state": "fresh",
        "state_authority": "verified_tool_output",
        "observed_at": _iso(-60),
        "last_verified_at": _iso(-30),
        "expires_at": _iso(3600),
        "ttl_seconds": 3600,
        "revalidation_interval_seconds": 600,
        "confirmation_policy": "none",
        "sensitivity": "medium",
        "scope_labels": ["technical_context"],
        "supersede_existing_claim_id": None,
    }
    claim.update(overrides)
    return claim


def _strict_world_turn(
    client, *, surface="web", text="I broke the server and prod is failing",
    containment_text=None, selection=True, containment=True,
):
    payload = {**_base(), "surface": surface}
    started = client.post("/v1/runtime/turns/start", json=payload)
    assert started.status_code == 200
    payload.update(
        runtime_session_id=started.json()["runtime_session"]["runtime_session_id"],
        runtime_turn_id=started.json()["runtime_turn"]["runtime_turn_id"],
    )
    assert client.post("/v1/runtime/interaction-governance/evaluate", json={
        **payload, "current_user_text": text,
    }).status_code == 200
    payload["persona_selection_mode"] = "strict"
    decision = None
    if selection:
        resolved = client.post("/v1/runtime/identity/resolve", json=payload)
        assert resolved.status_code == 200
        decision = resolved.json()["persona_selection"]
    payload["persona_selection_ref"] = (
        decision["selection_ref"] if decision else "psel_00000000000000000000000000000000"
    )
    containment_request = {**payload, "current_user_text": containment_text or text}
    if selection and containment:
        assert client.post(
            "/v1/runtime/persona-containment/evaluate", json=containment_request,
        ).status_code == 200
    return payload, decision, containment_request


def _seed_strict_world_claims(client):
    claims = {}
    for domain in (
        "active_task", "active_project", "active_repository", "active_tool_session",
        "active_health_observation", "pending_action", "runtime_surface",
    ):
        response = client.post("/v1/world-state/claims/upsert", json={
            **_base(), "claim": _claim(
                entity_id=f"entity:{domain}", domain=domain, attribute="status",
                value_json={"value": f"private_value_{domain}"},
            ),
        })
        assert response.status_code == 200
        claims[domain] = response.json()["claim"]["world_state_claim_id"]
    return claims


def _assert_strict_world_failure(response, identifiers=()):
    assert response.status_code == 409
    assert response.json() == {"detail": "world_state_authority_rejected"}
    assert not any(key in response.json() for key in (
        "included_claims", "excluded_claim_summaries", "prompt_content", "trace",
    ))
    assert "private_value" not in response.text
    assert "private_sentinel" not in response.text
    assert all(reference not in response.text for reference in identifiers)


@pytest.mark.parametrize("surface,persona,allowed", [
    ("web", "general_assistant", {
        "active_task", "active_project", "pending_action", "runtime_surface",
    }),
    ("dev", "technical_architect", {
        "active_project", "active_repository", "active_artifact", "active_tool_session",
        "active_external_system", "pending_action", "runtime_surface",
    }),
    ("vscode", "technical_architect", {
        "active_project", "active_repository", "active_artifact", "active_tool_session",
        "active_external_system", "pending_action", "runtime_surface",
    }),
    ("unregistered", "general_assistant", {"active_task", "pending_action", "runtime_surface"}),
    ("unknown", "general_assistant", {"active_task", "pending_action", "runtime_surface"}),
])
def test_strict_world_state_uses_authorized_persona_surface_and_bounded_provenance(
    surface, persona, allowed,
):
    client = TestClient(app)
    claims = _seed_strict_world_claims(client)
    payload, decision, _ = _strict_world_turn(client, surface=surface)
    response = client.post("/v1/world-state/resolve", json={
        **payload, "active_persona_id": persona,
        "expected_thread_revision": decision["thread_revision"],
    })
    assert response.status_code == 200
    result = response.json()
    assert result["selection_contract"] == "strict_turn"
    assert result["persona_selection_ref"] == decision["selection_ref"]
    assert result["trace"]["active_persona_id"] == persona
    assert set(result["trace"]["allowed_domains"]) == allowed
    assert {claim["world_state_claim_id"] for claim in result["included_claims"]} == {
        reference for domain, reference in claims.items() if domain in allowed
    }
    assert all(claim["effective_freshness_state"] == "fresh" for claim in result["included_claims"])
    assert decision["contextual_activation"] is False
    assert "private_value_active_health_observation" not in (result["prompt_content"] or "")


@pytest.mark.parametrize("text,proposed", [
    ("I broke the server and prod is failing", "technical_architect"),
    ("That was wrong in the report.", "personal_companion"),
])
def test_strict_world_advisory_proposal_never_opens_additional_domains(text, proposed):
    client = TestClient(app)
    _seed_strict_world_claims(client)
    payload, decision, _ = _strict_world_turn(client, text=text)
    assert decision["proposed_persona_id"] == proposed
    response = client.post("/v1/world-state/resolve", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["trace"]["active_persona_id"] == "general_assistant"
    assert {claim["domain"] for claim in result["included_claims"]} == {
        "active_task", "active_project", "pending_action", "runtime_surface",
    }
    assert "private_value_active_repository" not in result["prompt_content"]
    assert "private_value_active_health_observation" not in result["prompt_content"]


@pytest.mark.parametrize("requested,expected", [
    (["active_project"], {"active_project"}),
    (["active_repository", "active_project"], {"active_project"}),
    (["active_health_observation"], set()),
    (["technical", "personal", "unmapped"], set()),
])
def test_strict_world_requested_domains_only_narrow_without_vocabulary_mapping(requested, expected):
    client = TestClient(app)
    _seed_strict_world_claims(client)
    payload, _, _ = _strict_world_turn(client)
    response = client.post(
        "/v1/world-state/resolve", json={**payload, "requested_domains": requested},
    )
    assert response.status_code == 200
    result = response.json()
    assert set(result["trace"]["allowed_domains"]) == expected
    assert {claim["domain"] for claim in result["included_claims"]} == expected
    if not expected:
        assert result["prompt_content"] is None


@pytest.mark.parametrize("field,value", [
    ("request_id", "other_request"), ("owner_id", "other_owner"),
    ("conversation_id", "other_conversation"), ("surface", "vscode"),
    ("runtime_session_id", "other_session"), ("runtime_turn_id", "other_turn"),
    ("expected_thread_revision", 0), ("active_persona_id", "technical_architect"),
    ("persona_selection_ref", "psel_00000000000000000000000000000000"),
])
def test_strict_world_scope_mismatch_fails_before_protected_read(field, value, monkeypatch):
    import services.world_state as module

    client = TestClient(app)
    claims = _seed_strict_world_claims(client)
    payload, _, _ = _strict_world_turn(client)
    reads = []
    monkeypatch.setattr(module, "get_world_state_diagnostics", lambda **kw: reads.append(kw))
    _assert_strict_world_failure(
        client.post("/v1/world-state/resolve", json={**payload, field: value}), claims.values(),
    )
    assert reads == []
    events = runtime_state_repository().list_events_for_tests(payload["runtime_session_id"])
    assert sum(event.event_type == "persona_selection_resolved" for event in events) == 1
    assert sum(event.event_type == "persona_containment_evaluated" for event in events) == 1


@pytest.mark.parametrize("selection,containment", [(False, False), (True, False)])
def test_strict_world_requires_both_predecessors_without_manufacturing_them(
    selection, containment, monkeypatch,
):
    import services.world_state as module

    client = TestClient(app)
    payload, _, _ = _strict_world_turn(client, selection=selection, containment=containment)
    reads = []
    monkeypatch.setattr(module, "get_world_state_diagnostics", lambda **kw: reads.append(kw))
    _assert_strict_world_failure(client.post("/v1/world-state/resolve", json=payload))
    assert reads == []
    events = runtime_state_repository().list_events_for_tests(payload["runtime_session_id"])
    assert sum(event.event_type == "persona_selection_resolved" for event in events) == (
        int(selection)
    )
    assert not any(event.event_type == "persona_containment_evaluated" for event in events)


@pytest.mark.parametrize("mutation", [
    {"persona_selection_ref": "malformed"}, {"persona_selection_ref": None},
    {"runtime_session_id": None}, {"runtime_turn_id": None}, {"owner_id": ""},
    {"selection_source": "explicit_user"}, {"persona_scope_hint": "technical_operator"},
    {"requested_persona_id": "technical_architect"}, {"expected_thread_revision": True},
    {"persona_selection_mode": "unsupported"},
])
def test_strict_world_rejects_incomplete_or_spoofed_request_contract(mutation):
    client = TestClient(app)
    payload, _, _ = _strict_world_turn(client)
    response = client.post("/v1/world-state/resolve", json={**payload, **mutation})
    assert response.status_code == 422
    assert "private_value" not in response.text


@pytest.mark.parametrize("status", ["completed", "abandoned"])
def test_strict_world_terminal_turn_cannot_publish_state(status, monkeypatch):
    import services.world_state as module

    client = TestClient(app)
    payload, _, _ = _strict_world_turn(client)
    assert client.post("/v1/runtime/turns/complete", json={
        key: payload[key] for key in ("request_id", "runtime_session_id", "runtime_turn_id")
    } | {"turn_status": status}).status_code == 200
    reads = []
    monkeypatch.setattr(module, "get_world_state_diagnostics", lambda **kw: reads.append(kw))
    _assert_strict_world_failure(client.post("/v1/world-state/resolve", json=payload))
    assert reads == []


@pytest.mark.parametrize("corruption", [
    "selection_json", "selection_persona", "governance", "containment_json", "missing_policy",
    "status", "missing_status", "owner", "revision", "persona", "reference", "domains",
    "extra_policy", "contradiction", "legacy", "before_selection",
])
def test_strict_world_corrupted_authority_fails_before_data(corruption, monkeypatch):
    import services.world_state as module

    client = TestClient(app)
    payload, _, _ = _strict_world_turn(client)
    repo = runtime_state_repository()
    with repo._connect() as conn:
        kind = "persona_containment_evaluated"
        if corruption.startswith("selection"):
            kind = "persona_selection_resolved"
        elif corruption == "governance":
            kind = "interaction_governance_evaluated"
        row = conn.execute(
            "SELECT id, event_payload_json FROM conversation_runtime_events WHERE event_type = ?",
            (kind,),
        ).fetchone()
        changed = json.loads(row["event_payload_json"])
        if corruption.endswith("json"):
            value = "private_sentinel invalid_json"
        else:
            if corruption == "selection_persona":
                changed["active_persona_id"] = "personal_companion"
            elif corruption == "governance":
                changed["interaction_kind"] = "question"
            elif corruption in {"missing_policy", "legacy"}:
                del changed["strict_containment"]
                if corruption == "legacy":
                    del changed["persona_selection_ref"]
            elif corruption == "before_selection":
                conn.execute(
                    "UPDATE conversation_runtime_events SET id = 0 WHERE id = ?", (row["id"],),
                )
            else:
                authority = changed["strict_containment"]
                if corruption == "status":
                    authority["status"] = "failed"
                elif corruption == "missing_status":
                    del authority["status"]
                elif corruption == "owner":
                    authority["owner_id"] = "other_owner"
                elif corruption == "revision":
                    authority["thread_revision"] += 1
                elif corruption == "persona":
                    authority["result"]["active_persona_id"] = "personal_companion"
                elif corruption == "reference":
                    authority["persona_selection_ref"] = "psel_00000000000000000000000000000000"
                elif corruption == "domains":
                    authority["result"]["allowed_world_state_domains"] = ["general", "technical"]
                elif corruption == "extra_policy":
                    authority["result"]["current_user_text"] = "private_sentinel"
                elif corruption == "contradiction":
                    authority["result"]["blocked_memory_domains"].append("general")
                    changed["blocked_memory_domains"].append("general")
            value = json.dumps(changed)
        if corruption != "before_selection":
            conn.execute(
                "UPDATE conversation_runtime_events SET event_payload_json = ? WHERE id = ?",
                (value, row["id"]),
            )
    reads = []
    monkeypatch.setattr(module, "get_world_state_diagnostics", lambda **kw: reads.append(kw))
    _assert_strict_world_failure(client.post("/v1/world-state/resolve", json=payload))
    assert reads == []


def test_strict_world_requires_consistent_multiple_containment_publications():
    client = TestClient(app)
    payload, _, containment = _strict_world_turn(client, surface="dev")
    assert client.post(
        "/v1/runtime/persona-containment/evaluate", json=containment,
    ).status_code == 200
    assert client.post("/v1/world-state/resolve", json=payload).status_code == 200
    assert client.post("/v1/runtime/persona-containment/evaluate", json={
        **containment, "current_user_text": "What is 2+2?",
    }).status_code == 200
    _assert_strict_world_failure(client.post("/v1/world-state/resolve", json=payload))


def test_strict_world_unrepresentable_narrowed_containment_never_becomes_broader_state(monkeypatch):
    import services.world_state as module

    client = TestClient(app)
    payload, _, _ = _strict_world_turn(
        client, surface="dev", containment_text="Check tire pressure and vehicle maintenance.",
    )
    reads = []
    monkeypatch.setattr(module, "get_world_state_diagnostics", lambda **kw: reads.append(kw))
    _assert_strict_world_failure(client.post("/v1/world-state/resolve", json=payload))
    assert reads == []


@pytest.mark.parametrize("corruption", [
    "missing_binding", "malformed_binding", "unsupported", "missing_persona", "invalid_persona",
])
def test_strict_world_invalid_authority_never_defaults_to_permission(corruption, monkeypatch):
    import services.world_state as module
    from services.companion_contracts import companion_contracts_repository

    client = TestClient(app)
    payload, _, _ = _strict_world_turn(client)
    with companion_contracts_repository()._connect() as conn:
        if corruption == "missing_binding":
            conn.execute("DELETE FROM surface_bindings WHERE surface_id IN ('web', 'unknown')")
        elif corruption == "malformed_binding":
            conn.execute(
                "UPDATE surface_bindings SET default_persona_id = '' WHERE surface_id = 'web'",
            )
        elif corruption == "unsupported":
            conn.execute(
                "UPDATE surface_bindings SET surface_type = 'unsupported' WHERE surface_id = 'web'",
            )
        elif corruption == "missing_persona":
            conn.execute("DELETE FROM persona_profiles WHERE persona_id = 'general_assistant'")
        else:
            conn.execute(
                "UPDATE persona_profiles SET communication_policy_summary_json = '[7]' "
                "WHERE persona_id = 'general_assistant'",
            )
    reads = []
    monkeypatch.setattr(module, "get_world_state_diagnostics", lambda **kw: reads.append(kw))
    _assert_strict_world_failure(client.post("/v1/world-state/resolve", json=payload))
    assert reads == []


@pytest.mark.parametrize("change", [
    "completion", "revision", "containment", "surface", "governance",
])
def test_strict_world_revalidates_authority_after_protected_read(change, monkeypatch):
    import services.world_state as module

    client = TestClient(app)
    identifiers = _seed_strict_world_claims(client)
    payload, _, containment = _strict_world_turn(client)
    original = module.get_world_state_diagnostics

    def change_after_read(**kwargs):
        result = original(**kwargs)
        repo = runtime_state_repository()
        if change == "completion":
            repo.complete_turn(
                **{key: payload[key] for key in (
                    "request_id", "runtime_session_id", "runtime_turn_id",
                )},
                turn_status="completed",
            )
        elif change == "revision":
            with repo._connect() as conn:
                conn.execute("UPDATE conversation_runtime_threads SET revision = revision + 1")
        elif change == "containment":
            from models import PersonaContainmentEvaluateRequest
            from services.persona_containment import evaluate_persona_containment

            evaluate_persona_containment(PersonaContainmentEvaluateRequest(
                **{**containment, "current_user_text": "What is 2+2?"},
            ))
        elif change == "governance":
            with repo._connect() as conn:
                row = conn.execute(
                    "SELECT id, event_payload_json FROM conversation_runtime_events "
                    "WHERE event_type = 'interaction_governance_evaluated'",
                ).fetchone()
                changed = json.loads(row["event_payload_json"])
                changed["interaction_kind"] = "question"
                conn.execute(
                    "UPDATE conversation_runtime_events SET event_payload_json = ? WHERE id = ?",
                    (json.dumps(changed), row["id"]),
                )
        else:
            from services.companion_contracts import companion_contracts_repository

            with companion_contracts_repository()._connect() as conn:
                conn.execute(
                    "UPDATE surface_bindings SET surface_type = 'ide_extension' "
                    "WHERE surface_id = 'web'",
                )
        return result

    monkeypatch.setattr(module, "get_world_state_diagnostics", change_after_read)
    _assert_strict_world_failure(
        client.post("/v1/world-state/resolve", json=payload), identifiers.values(),
    )


def test_strict_world_selection_and_containment_survive_repository_replacement():
    client = TestClient(app)
    _seed_strict_world_claims(client)
    payload, _, _ = _strict_world_turn(client)
    before = client.post("/v1/world-state/resolve", json=payload)
    assert before.status_code == 200
    clear_states_for_tests(db_path=runtime_state_repository().db_path)
    after = client.post("/v1/world-state/resolve", json=payload)
    assert after.status_code == 200
    assert after.json() == before.json()


def test_legacy_world_request_preserves_independent_persona_and_read_order(monkeypatch):
    import services.world_state as module

    client = TestClient(app)
    _seed_strict_world_claims(client)
    order = []
    read, scope = module.get_world_state_diagnostics, module.resolve_world_state_persona_scope

    def observed_read(**kwargs):
        order.append("read")
        return read(**kwargs)

    def observed_scope(**kwargs):
        order.append("scope")
        return scope(**kwargs)

    monkeypatch.setattr(module, "get_world_state_diagnostics", observed_read)
    monkeypatch.setattr(module, "resolve_world_state_persona_scope", observed_scope)
    response = client.post("/v1/world-state/resolve", json={
        **_base(), "active_persona_id": "technical_architect",
    })
    assert response.status_code == 200
    assert response.json()["selection_contract"] == "legacy_unbound"
    assert response.json()["persona_selection_ref"] is None
    assert "active_repository" in response.json()["trace"]["allowed_domains"]
    assert order == ["read", "scope"]


@pytest.mark.parametrize("overrides,expected,confirmation", [
    ({"observed_at": _iso(-500)}, "aging", False),
    ({"observed_at": _iso(-900), "confirmation_policy": "confirm_before_action"}, "stale", True),
    ({"confirmation_policy": "confirm_before_action"}, "fresh", True),
])
def test_strict_world_preserves_freshness_qualification_and_confirmation(
    overrides, expected, confirmation,
):
    client = TestClient(app)
    assert client.post("/v1/world-state/claims/upsert", json={
        **_base(), "claim": _claim(**overrides),
    }).status_code == 200
    payload, _, _ = _strict_world_turn(client, surface="dev")
    response = client.post("/v1/world-state/resolve", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["included_claims"][0]["effective_freshness_state"] == expected
    assert result["trace"]["confirmation_required"] == confirmation
    if expected != "fresh":
        assert "last_known" in result["prompt_content"]
        assert f"{expected};" in result["prompt_content"]


@pytest.mark.parametrize("sensitivity", ["high", "restricted"])
def test_strict_world_sensitive_values_remain_redacted(sensitivity):
    client = TestClient(app)
    assert client.post("/v1/world-state/claims/upsert", json={
        **_base(), "claim": _claim(
            sensitivity=sensitivity, value_json={"secret": "private_sentinel"},
        ),
    }).status_code == 200
    payload, _, _ = _strict_world_turn(client, surface="dev")
    response = client.post("/v1/world-state/resolve", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["included_claims"][0]["value_json"] is None
    assert result["included_claims"][0]["value_redacted"] is True
    assert "[REDACTED]" in result["prompt_content"]
    assert "private_sentinel" not in response.text


@pytest.mark.parametrize("outcome", ["expired", "conflicted", "superseded"])
def test_strict_world_excludes_expired_conflicted_and_superseded_claims(outcome):
    client = TestClient(app)
    first = client.post("/v1/world-state/claims/upsert", json={
        **_base(), "claim": _claim(
            value_json={"state": "old_private_sentinel"},
            **({"expires_at": _iso(-10)} if outcome == "expired" else {}),
        ),
    })
    assert first.status_code == 200
    old_ref = first.json()["claim"]["world_state_claim_id"]
    if outcome != "expired":
        response = client.post("/v1/world-state/claims/upsert", json={
            **_base(), "claim": _claim(
                value_json={"state": "replacement"},
                supersede_existing_claim_id=old_ref if outcome == "superseded" else None,
            ),
        })
        assert response.status_code == 200
    payload, _, _ = _strict_world_turn(client, surface="dev")
    response = client.post("/v1/world-state/resolve", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert all(claim["world_state_claim_id"] != old_ref for claim in result["included_claims"])
    assert any(
        item["effective_freshness_state"] == outcome for item in result["excluded_claim_summaries"]
    )
    assert "old_private_sentinel" not in (result["prompt_content"] or "")
    if outcome != "superseded":
        assert result["included_claims"] == []
        assert result["prompt_content"] is None


def test_strict_world_barriers_surround_only_one_protected_read(monkeypatch):
    import services.world_state as module

    client = TestClient(app)
    _seed_strict_world_claims(client)
    payload, _, _ = _strict_world_turn(client)
    order = []
    original_authority = module._strict_world_state_authority
    original_read = module.get_world_state_diagnostics

    def authority(body):
        order.append("authority")
        return original_authority(body)

    def read(**kwargs):
        order.append("read")
        return original_read(**kwargs)

    monkeypatch.setattr(module, "_strict_world_state_authority", authority)
    monkeypatch.setattr(module, "get_world_state_diagnostics", read)
    assert client.post("/v1/world-state/resolve", json=payload).status_code == 200
    assert order == ["authority", "read", "authority"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["authority", "protected_read"])
async def test_strict_world_storage_errors_never_become_success_or_disclose_private_errors(
    failure, monkeypatch,
):
    import httpx
    import services.world_state as module
    from models import PersonaContainmentEvaluateRequest, RuntimeIdentityResolveRequest
    from services.persona_containment import evaluate_persona_containment
    from services.runtime_identity import resolve_runtime_identity

    # Set up real repositories without nesting the synchronous TestClient event loop.
    repo = runtime_state_repository()
    session, turn, _ = repo.start_turn(**_base())
    from models import InteractionGovernanceEvaluateRequest
    from services.interaction_governance import evaluate_interaction_governance

    payload = {
        **_base(), "runtime_session_id": session.runtime_session_id,
        "runtime_turn_id": turn.runtime_turn_id,
    }
    evaluate_interaction_governance(InteractionGovernanceEvaluateRequest(
        **payload, current_user_text="I broke the server and prod is failing",
    ))
    payload["persona_selection_mode"] = "strict"
    identity = resolve_runtime_identity(RuntimeIdentityResolveRequest(**payload))
    payload["persona_selection_ref"] = identity.persona_selection.selection_ref
    evaluate_persona_containment(PersonaContainmentEvaluateRequest(
        **payload, current_user_text="I broke the server and prod is failing",
    ))
    reads = []

    def fail(**kwargs):
        raise sqlite3.OperationalError("private_sentinel storage failure")

    if failure == "authority":
        monkeypatch.setattr(repo, "persona_selection_events", fail)
        monkeypatch.setattr(module, "get_world_state_diagnostics", lambda **kw: reads.append(kw))
    else:
        monkeypatch.setattr(module, "get_world_state_diagnostics", fail)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test",
    ) as client:
        response = await client.post("/v1/world-state/resolve", json=payload)
    assert response.status_code == 500
    assert "private_sentinel" not in response.text
    assert "included_claims" not in response.text
    assert reads == []


@pytest.mark.parametrize("state", ["contended", "unavailable", "wrong_pointer"])
def test_strict_world_noncurrent_runtime_authority_blocks_read(state, monkeypatch):
    import services.world_state as module

    client = TestClient(app)
    payload, _, _ = _strict_world_turn(client)
    with runtime_state_repository()._connect() as conn:
        if state == "wrong_pointer":
            conn.execute("UPDATE conversation_runtime_threads SET active_runtime_turn_id = NULL")
        else:
            conn.execute("UPDATE conversation_runtime_threads SET state = ?", (state,))
    reads = []
    monkeypatch.setattr(module, "get_world_state_diagnostics", lambda **kw: reads.append(kw))
    _assert_strict_world_failure(client.post("/v1/world-state/resolve", json=payload))
    assert reads == []


@pytest.mark.parametrize("binding_type", ["ide_extension", "unsupported"])
def test_unknown_world_surface_never_inherits_configured_fallback_type_privileges(binding_type):
    from services.companion_contracts import companion_contracts_repository

    client = TestClient(app)
    _seed_strict_world_claims(client)
    with companion_contracts_repository()._connect() as conn:
        conn.execute(
            "UPDATE surface_bindings SET surface_type = ? WHERE surface_id = 'unknown'",
            (binding_type,),
        )
    payload, _, _ = _strict_world_turn(client, surface="unregistered")
    response = client.post("/v1/world-state/resolve", json=payload)
    if binding_type == "unsupported":
        _assert_strict_world_failure(response)
    else:
        assert response.status_code == 200
        assert response.json()["trace"]["allowed_domains"] == [
            "active_task", "pending_action", "runtime_surface",
        ]
        assert "private_value_active_project" not in response.json()["prompt_content"]


def test_world_state_claim_create_and_metadata_round_trip():
    client = TestClient(app)

    response = client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim()},
    )

    assert response.status_code == 200
    body = response.json()
    claim = body["claim"]
    assert claim["entity_id"] == "repo:primary"
    assert claim["source_ref"] == "pytest"
    assert claim["freshness_state"] == "fresh"
    assert claim["effective_freshness_state"] == "fresh"
    assert body["transitions"][0]["transition_type"] == "created"
    assert claim["value_digest"].startswith("wsvalue_")


def test_world_state_claim_update_preserves_provenance_requirements():
    client = TestClient(app)
    created = client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim()},
    ).json()["claim"]

    response = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                source_ref="pytest-rerun",
                confidence=0.99,
                value_json={"branch": "main", "status": "passing"},
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["claim"]["world_state_claim_id"] != created["world_state_claim_id"]
    assert body["claim"]["source_ref"] == "pytest-rerun"
    assert body["claim"]["source_type"] == "tool_output"


def test_world_state_supersede_records_metadata():
    client = TestClient(app)
    first = client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim(value_json={"state": "open"})},
    ).json()["claim"]

    response = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                value_json={"state": "closed"},
                supersede_existing_claim_id=first["world_state_claim_id"],
            ),
        },
    )
    diagnostics = client.post("/v1/world-state/diagnostics", json=_base()).json()

    assert response.status_code == 200
    superseded = next(
        item
        for item in diagnostics["excluded_claims"]
        if item["world_state_claim_id"] == first["world_state_claim_id"]
    )
    assert superseded["reason"] == "superseded"
    assert superseded["superseded_by_claim_id"] == response.json()["claim"]["world_state_claim_id"]


def test_world_state_conflict_marks_active_claims_without_picking_winner():
    client = TestClient(app)
    client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim(value_json={"state": "open"})},
    )
    client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim(value_json={"state": "blocked"})},
    )

    diagnostics = client.post("/v1/world-state/diagnostics", json=_base())

    assert diagnostics.status_code == 200
    excluded = diagnostics.json()["excluded_claims"]
    assert len(excluded) == 2
    assert {item["effective_freshness_state"] for item in excluded} == {"conflicted"}
    assert all(item["conflict_claim_ids"] for item in excluded)


def test_expired_claim_is_not_treated_as_current():
    client = TestClient(app)
    client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                observed_at=_iso(-7200),
                expires_at=_iso(-3600),
                ttl_seconds=60,
            ),
        },
    )

    diagnostics = client.post("/v1/world-state/diagnostics", json=_base()).json()

    assert diagnostics["claims"] == []
    assert diagnostics["excluded_claims"][0]["effective_freshness_state"] == "expired"


def test_stored_freshness_is_preserved_while_effective_freshness_is_computed():
    client = TestClient(app)
    client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                freshness_state="fresh",
                observed_at=_iso(-500),
                revalidation_interval_seconds=400,
                expires_at=None,
                ttl_seconds=None,
            ),
        },
    )

    diagnostics = client.post("/v1/world-state/diagnostics", json=_base()).json()

    assert diagnostics["claims"][0]["freshness_state"] == "fresh"
    assert diagnostics["claims"][0]["effective_freshness_state"] == "stale"


def test_diagnostics_redact_sensitive_values_by_default():
    client = TestClient(app)
    client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                domain="active_health_observation",
                entity_type="health_observation",
                sensitivity="restricted",
                value_json={"condition": "private"},
            ),
        },
    )

    diagnostics = client.post("/v1/world-state/diagnostics", json=_base()).json()

    assert diagnostics["claims"][0]["value_json"] is None
    assert diagnostics["claims"][0]["value_redacted"] is True


def test_diagnostics_ignore_include_sensitive_values_without_authorization_policy():
    client = TestClient(app)
    client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                sensitivity="restricted",
                value_json={"secret": "still-hidden"},
            ),
        },
    )

    diagnostics = client.post(
        "/v1/world-state/diagnostics",
        json={**_base(), "include_sensitive_values": True},
    ).json()

    assert diagnostics["claims"][0]["value_json"] is None
    assert diagnostics["claims"][0]["value_redacted"] is True


def test_world_state_payload_does_not_introduce_memory_or_relationship_contracts():
    client = TestClient(app)
    response = client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim()},
    )

    assert response.status_code == 200
    payload_text = str(response.json())
    assert "relationship_edges" not in payload_text
    assert "canonical_memory_ref" not in payload_text


def test_world_state_authoritative_verification_persists_transition_and_source():
    client = TestClient(app)
    _configure_repo_verifier()
    started = client.post(
        "/v1/runtime/turns/start",
        json={**_base(), "request_id": "turn-for-verification"},
    ).json()
    created = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                source_type="user_report",
                state_authority="observed_user_report",
                last_verified_at=None,
            ),
        },
    ).json()["claim"]

    response = client.post(
        "/v1/world-state/claims/verify",
        json={
            **_base(),
            "request_id": "verify-claim",
            "runtime_session_id": started["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": started["runtime_turn"]["runtime_turn_id"],
            "world_state_claim_id": created["world_state_claim_id"],
            "expected_value_digest": created["value_digest"],
            "verifier_id": "repo-status-revalidator",
            "verification_source_type": "tool_output",
            "verification_source_ref": "status-check:bounded",
            "observed_at": _iso(-15),
            "verified_at": _iso(-5),
            "resulting_authority": "verified_tool_output",
            "resulting_confidence": 0.97,
            "resulting_freshness_state": "fresh",
            "resulting_ttl_seconds": 600,
            "resulting_revalidation_interval_seconds": 300,
        },
    )
    diagnostics = client.post("/v1/world-state/diagnostics", json=_base()).json()

    assert response.status_code == 200
    body = response.json()
    assert body["claim"]["last_verified_at"] is not None
    assert body["claim"]["verification_verifier_id"] == "repo-status-revalidator"
    assert body["claim"]["verification_source_type"] == "tool_output"
    assert body["claim"]["verification_source_ref"] == "status-check:bounded"
    assert body["claim"]["state_authority"] == "verified_tool_output"
    assert body["transitions"][0]["transition_type"] == "verified"
    assert any(item["transition_type"] == "verified" for item in diagnostics["transitions"])
    payload_text = str(diagnostics)
    assert "status-check:bounded" in payload_text
    assert "failing" not in diagnostics["transitions"][-1]["metadata_json"].values()


def test_world_state_verification_rejects_digest_mismatch_atomically():
    client = TestClient(app)
    _configure_repo_verifier()
    created = client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim()},
    ).json()["claim"]
    before = client.post("/v1/world-state/diagnostics", json=_base()).json()

    response = client.post(
        "/v1/world-state/claims/verify",
        json={
            **_base(),
            "request_id": "verify-mismatch",
            "world_state_claim_id": created["world_state_claim_id"],
            "expected_value_digest": "wsvalue_wrong",
            "verifier_id": "repo-status-revalidator",
            "verification_source_type": "tool_output",
            "verification_source_ref": "status-check:bounded",
            "observed_at": _iso(-15),
            "verified_at": _iso(-5),
            "resulting_authority": "verified_tool_output",
            "resulting_confidence": 0.97,
            "resulting_freshness_state": "fresh",
        },
    )
    after = client.post("/v1/world-state/diagnostics", json=_base()).json()

    assert response.status_code == 409
    assert response.json()["detail"] == "expected_value_mismatch"
    assert after == before


def test_world_state_verification_rejects_untrusted_source():
    client = TestClient(app)
    _configure_repo_verifier()
    started = client.post(
        "/v1/runtime/turns/start",
        json={**_base(), "request_id": "turn-for-rejected-verification"},
    ).json()
    created = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                source_type="model_inference",
                state_authority="model_inferred",
            ),
        },
    ).json()["claim"]

    response = client.post(
        "/v1/world-state/claims/verify",
        json={
            **_base(),
            "request_id": "verify-untrusted",
            "runtime_session_id": started["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": started["runtime_turn"]["runtime_turn_id"],
            "world_state_claim_id": created["world_state_claim_id"],
            "expected_value_digest": created["value_digest"],
            "verifier_id": "repo-status-revalidator",
            "verification_source_type": "model_inference",
            "verification_source_ref": "self-attested",
            "observed_at": _iso(-15),
            "verified_at": _iso(-5),
            "resulting_authority": "verified_tool_output",
            "resulting_confidence": 0.97,
            "resulting_freshness_state": "fresh",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "verification_source_mismatch"
    diagnostics = client.get(
        f"/v1/runtime/sessions/{started['runtime_session']['runtime_session_id']}"
    ).json()
    event = diagnostics["events"][-1]
    assert event["event_type"] == "world_state_verification_evaluated"
    assert event["event_payload_json"]["decision"] == "rejected"
    assert event["event_payload_json"]["reason"] == "verification_source_mismatch"
    assert "passing" not in str(event)


def test_world_state_verification_requires_configured_trusted_verifier():
    client = TestClient(app)
    created = client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim()},
    ).json()["claim"]

    response = client.post(
        "/v1/world-state/claims/verify",
        json={
            **_base(),
            "request_id": "verify-missing-verifier",
            "world_state_claim_id": created["world_state_claim_id"],
            "expected_value_digest": created["value_digest"],
            "verification_source_type": "tool_output",
            "verification_source_ref": "status-check:bounded",
            "observed_at": _iso(-15),
            "verified_at": _iso(-5),
            "resulting_authority": "verified_tool_output",
            "resulting_confidence": 0.97,
            "resulting_freshness_state": "fresh",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "trusted_verifier_required"


def test_world_state_verification_loads_production_registry_from_env(tmp_path, monkeypatch):
    client = TestClient(app)
    clear_trusted_world_state_verifiers_for_tests()
    registry_path = tmp_path / "trusted-verifiers.yaml"
    registry_path.write_text(_registry_yaml(), encoding="utf-8")
    monkeypatch.setenv("TRUSTED_WORLD_STATE_VERIFIERS_PATH", str(registry_path))
    created = client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim()},
    ).json()["claim"]

    response = client.post(
        "/v1/world-state/claims/verify",
        json={
            **_base(),
            "request_id": "verify-env-registry",
            "world_state_claim_id": created["world_state_claim_id"],
            "expected_value_digest": created["value_digest"],
            "verifier_id": "repo-status-revalidator",
            "verification_source_type": "tool_output",
            "verification_source_ref": "status-check:bounded",
            "observed_at": _iso(-15),
            "verified_at": _iso(-5),
            "resulting_authority": "verified_tool_output",
            "resulting_confidence": 0.97,
            "resulting_freshness_state": "fresh",
        },
    )

    assert response.status_code == 200
    assert response.json()["claim"]["verification_verifier_id"] == "repo-status-revalidator"


def test_trusted_verifier_registry_invalid_configs_fail_closed(tmp_path, monkeypatch):
    clear_trusted_world_state_verifiers_for_tests()
    missing_path = tmp_path / "missing.yaml"
    monkeypatch.setenv("TRUSTED_WORLD_STATE_VERIFIERS_PATH", str(missing_path))
    with pytest.raises(RuntimeError, match="trusted_verifier_registry_invalid"):
        trusted_world_state_verifiers()

    duplicate_entries = "verifiers:\n" + _registry_yaml().replace("verifiers:\n", "", 1) * 2
    for name, content in {
        "malformed.yaml": "verifiers: [",
        "duplicate.yaml": duplicate_entries,
        "invalid-policy.yaml": _registry_yaml(allowed_source_refs=[]),
        "unknown-field.yaml": _registry_yaml() + "    surprise: nope\n",
    }.items():
        clear_trusted_world_state_verifiers_for_tests()
        registry_path = tmp_path / name
        registry_path.write_text(content, encoding="utf-8")
        monkeypatch.setenv("TRUSTED_WORLD_STATE_VERIFIERS_PATH", str(registry_path))
        with pytest.raises(RuntimeError, match="trusted_verifier_registry_invalid"):
            trusted_world_state_verifiers()


def test_world_state_verification_uses_configured_temporal_bounds_when_omitted():
    client = TestClient(app)
    _configure_repo_verifier()
    created = client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim(source_type="user_report")},
    ).json()["claim"]

    response = client.post(
        "/v1/world-state/claims/verify",
        json={
            **_base(),
            "request_id": "verify-omitted-bounds",
            "world_state_claim_id": created["world_state_claim_id"],
            "expected_value_digest": created["value_digest"],
            "verifier_id": "repo-status-revalidator",
            "verification_source_type": "tool_output",
            "verification_source_ref": "status-check:bounded",
            "observed_at": _iso(-15),
            "verified_at": _iso(-5),
            "resulting_authority": "verified_tool_output",
            "resulting_confidence": 0.97,
            "resulting_freshness_state": "fresh",
        },
    )

    assert response.status_code == 200
    claim = response.json()["claim"]
    assert claim["ttl_seconds"] == 600
    assert claim["revalidation_interval_seconds"] == 300
    assert claim["expires_at"] is not None


def test_world_state_verification_rejects_unbounded_time_without_mutation():
    client = TestClient(app)
    _configure_repo_verifier()
    created = client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim(source_type="user_report")},
    ).json()["claim"]
    before = client.post("/v1/world-state/diagnostics", json=_base()).json()
    base_verify = {
        **_base(),
        "world_state_claim_id": created["world_state_claim_id"],
        "expected_value_digest": created["value_digest"],
        "verifier_id": "repo-status-revalidator",
        "verification_source_type": "tool_output",
        "verification_source_ref": "status-check:bounded",
        "observed_at": _iso(-15),
        "verified_at": _iso(-5),
        "resulting_authority": "verified_tool_output",
        "resulting_confidence": 0.97,
        "resulting_freshness_state": "fresh",
    }

    oversized_ttl = client.post(
        "/v1/world-state/claims/verify",
        json={**base_verify, "request_id": "verify-ttl-too-large", "resulting_ttl_seconds": 601},
    )
    future_observation = client.post(
        "/v1/world-state/claims/verify",
        json={**base_verify, "request_id": "verify-future-observed", "observed_at": _iso(60)},
    )
    late_expiry = client.post(
        "/v1/world-state/claims/verify",
        json={**base_verify, "request_id": "verify-late-expiry", "resulting_expires_at": _iso(600)},
    )
    after = client.post("/v1/world-state/diagnostics", json=_base()).json()

    assert oversized_ttl.status_code == 403
    assert oversized_ttl.json()["detail"] == "verification_ttl_escalation"
    assert future_observation.status_code == 400
    assert future_observation.json()["detail"] == "verification_timestamp_in_future"
    assert late_expiry.status_code == 403
    assert late_expiry.json()["detail"] == "verification_expiry_escalation"
    assert after == before


def test_world_state_verification_rejects_source_ref_authority_and_domain_escalation():
    client = TestClient(app)
    _configure_repo_verifier(max_authority="derived_from_multiple_sources")
    created = client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim()},
    ).json()["claim"]

    base_verify = {
        **_base(),
        "request_id": "verify-policy",
        "world_state_claim_id": created["world_state_claim_id"],
        "expected_value_digest": created["value_digest"],
        "verifier_id": "repo-status-revalidator",
        "verification_source_type": "tool_output",
        "verification_source_ref": "forged-ref",
        "observed_at": _iso(-15),
        "verified_at": _iso(-5),
        "resulting_authority": "derived_from_multiple_sources",
        "resulting_confidence": 0.97,
        "resulting_freshness_state": "fresh",
    }
    forged = client.post("/v1/world-state/claims/verify", json=base_verify)
    authority = client.post(
        "/v1/world-state/claims/verify",
        json={
            **base_verify,
            "request_id": "verify-authority-escalation",
            "verification_source_ref": "status-check:bounded",
            "resulting_authority": "verified_tool_output",
        },
    )

    _configure_repo_verifier(allowed_domains=frozenset({"active_project"}))
    domain = client.post(
        "/v1/world-state/claims/verify",
        json={
            **base_verify,
            "request_id": "verify-domain-escalation",
            "verification_source_ref": "status-check:bounded",
        },
    )

    assert forged.status_code == 403
    assert forged.json()["detail"] == "verification_source_ref_not_allowed"
    assert authority.status_code == 403
    assert authority.json()["detail"] == "verification_authority_escalation"
    assert domain.status_code == 403
    assert domain.json()["detail"] == "verification_domain_not_allowed"


def test_existing_world_state_database_upgrades_additively(tmp_path):
    db_path = tmp_path / "pre_wave3c.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE runtime_world_state_claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                world_state_claim_id TEXT NOT NULL UNIQUE,
                owner_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                attribute TEXT NOT NULL,
                value_json TEXT NOT NULL,
                material_value_json TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                confidence REAL NOT NULL,
                freshness_state TEXT NOT NULL,
                state_authority TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                last_verified_at TEXT,
                expires_at TEXT,
                ttl_seconds INTEGER,
                revalidation_interval_seconds INTEGER,
                confirmation_policy TEXT NOT NULL,
                sensitivity TEXT NOT NULL,
                scope_labels_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                superseded_by_claim_id TEXT,
                UNIQUE(owner_id, world_state_claim_id)
            );
            CREATE TABLE runtime_world_state_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transition_id TEXT NOT NULL UNIQUE,
                world_state_claim_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                transition_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO runtime_world_state_claims (
                world_state_claim_id, owner_id, entity_id, entity_type, domain, attribute,
                value_json, material_value_json, source_type, source_ref, confidence,
                freshness_state, state_authority, observed_at, last_verified_at, expires_at,
                ttl_seconds, revalidation_interval_seconds, confirmation_policy, sensitivity,
                scope_labels_json, created_at, updated_at, superseded_by_claim_id
            ) VALUES (
                'legacy-claim', 'owner', 'repo:primary', 'repository', 'active_repository',
                'branch_status', '{"status":"passing"}', '{"status":"passing"}',
                'tool_output', 'legacy', 0.9, 'fresh', 'verified_tool_output',
                '2026-01-01T00:00:00+00:00', NULL, NULL, 3600, 600, 'none',
                'medium', '["technical_context"]', '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00', NULL
            );
            """
        )

    repo = WorldStateRepository(db_path=db_path)
    diagnostics = repo.diagnostics(owner_id="owner")
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(runtime_world_state_claims);")}

    assert {
        "verification_verifier_id",
        "verification_source_type",
        "verification_source_ref",
        "last_verified_runtime_session_id",
        "last_verified_runtime_turn_id",
        "last_verification_request_id",
    } <= columns
    preserved = [*diagnostics.claims, *diagnostics.excluded_claims]
    assert preserved[0].world_state_claim_id == "legacy-claim"


def test_world_state_resolve_includes_fresh_eligible_claims():
    client = TestClient(app)
    client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim()},
    )

    response = client.post(
        "/v1/world-state/resolve",
        json={**_base(), "active_persona_id": "technical_architect", "requested_domains": []},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["included_claims"]) == 1
    assert body["trace"]["included_claim_count"] == 1
    assert "active_repository/branch_status" in body["prompt_content"]


def test_world_state_resolve_excludes_claims_outside_scope_and_requested_domains_only_narrow():
    client = TestClient(app)
    client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                domain="active_health_observation",
                entity_type="health_observation",
            ),
        },
    )

    response = client.post(
        "/v1/world-state/resolve",
        json={
            **_base(),
            "active_persona_id": "technical_architect",
            "requested_domains": ["active_health_observation", "active_repository"],
        },
    )

    assert response.status_code == 200
    assert response.json()["included_claims"] == []
    assert (
        response.json()["excluded_claim_summaries"][0]["reason"]
        == "outside_persona_or_surface_scope"
    )


def test_world_state_resolve_qualifies_stale_claims():
    client = TestClient(app)
    client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                observed_at=_iso(-900),
                ttl_seconds=None,
                expires_at=None,
                revalidation_interval_seconds=600,
                confirmation_policy="confirm_before_action",
            ),
        },
    )

    response = client.post(
        "/v1/world-state/resolve",
        json={**_base(), "active_persona_id": "technical_architect"},
    )

    assert response.status_code == 200
    assert response.json()["included_claims"][0]["effective_freshness_state"] == "stale"
    assert "last_known" in response.json()["prompt_content"]
    assert response.json()["trace"]["confirmation_required"] is True


def test_world_state_resolve_excludes_conflicted_claims_without_winner_selection():
    client = TestClient(app)
    client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim(value_json={"state": "open"})},
    )
    client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim(value_json={"state": "closed"})},
    )

    response = client.post(
        "/v1/world-state/resolve",
        json={**_base(), "active_persona_id": "technical_architect"},
    )

    assert response.status_code == 200
    assert response.json()["included_claims"] == []
    assert response.json()["trace"]["conflicted_count"] == 2
    assert all(
        item["effective_freshness_state"] == "conflicted"
        for item in response.json()["excluded_claim_summaries"]
    )


def test_world_state_resolve_redacts_sensitive_prompt_content():
    client = TestClient(app)
    client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                domain="active_repository",
                sensitivity="restricted",
                value_json={"secret": "do-not-show"},
            ),
        },
    )

    response = client.post(
        "/v1/world-state/resolve",
        json={**_base(), "active_persona_id": "technical_architect"},
    )

    assert response.status_code == 200
    assert "do-not-show" not in (response.json()["prompt_content"] or "")
    assert "[REDACTED]" in (response.json()["prompt_content"] or "")
