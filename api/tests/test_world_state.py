from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from main import app
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
