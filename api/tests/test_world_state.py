from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from main import app


def _iso(delta_seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=delta_seconds)).isoformat()


def _base() -> dict[str, object]:
    return {
        "request_id": "rid-world-state",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
    }


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
