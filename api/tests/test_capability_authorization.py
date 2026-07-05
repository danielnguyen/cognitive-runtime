from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from main import app


def _iso(delta_seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=delta_seconds)).isoformat()


def _base() -> dict[str, object]:
    return {
        "request_id": "capability-test",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
    }


def _start_turn(client: TestClient) -> dict[str, object]:
    return client.post(
        "/v1/runtime/turns/start",
        json={**_base(), "request_id": "capability-turn"},
    ).json()


def _claim(**overrides) -> dict[str, object]:
    claim = {
        "entity_id": "repo:primary",
        "entity_type": "repository",
        "domain": "active_repository",
        "attribute": "branch_status",
        "value_json": {"status": "passing"},
        "source_type": "tool_output",
        "source_ref": "pytest",
        "confidence": 0.95,
        "freshness_state": "fresh",
        "state_authority": "verified_tool_output",
        "observed_at": _iso(-30),
        "last_verified_at": _iso(-20),
        "expires_at": _iso(600),
        "ttl_seconds": 600,
        "revalidation_interval_seconds": 300,
        "confirmation_policy": "none",
        "sensitivity": "medium",
        "scope_labels": ["technical_context"],
        "supersede_existing_claim_id": None,
    }
    claim.update(overrides)
    return claim


def _relationship(client: TestClient, **edge_overrides) -> str:
    client.post(
        "/v1/relationships/entities/upsert",
        json={
            **_base(),
            "entity": {
                "entity_id": "persona:technical_architect",
                "entity_type": "persona",
                "canonical_label": "technical architect",
                "domain": "project_context",
                "sensitivity_level": "medium",
                "source_type": "trusted_config",
                "source_ref": "pytest",
                "status": "active",
            },
        },
    )
    client.post(
        "/v1/relationships/entities/upsert",
        json={
            **_base(),
            "entity": {
                "entity_id": "repo:primary",
                "entity_type": "repository",
                "canonical_label": "primary repository",
                "domain": "project_context",
                "sensitivity_level": "medium",
                "source_type": "trusted_config",
                "source_ref": "pytest",
                "status": "active",
            },
        },
    )
    edge = {
        "relationship_id": "rel-repo-project",
        "subject_entity_id": "persona:technical_architect",
        "relationship_type": "works_on",
        "object_entity_id": "repo:primary",
        "relationship_scope": "project_context",
        "source_type": "trusted_config",
        "source_refs_json": ["pytest"],
        "confidence": 0.98,
        "status": "active",
        "sensitivity_level": "medium",
        "mentionability": "use_for_filtering_only",
        "allowed_persona_scopes_json": ["technical_architect"],
        "blocked_persona_scopes_json": [],
    }
    edge.update(edge_overrides)
    response = client.post(
        "/v1/relationships/edges/upsert",
        json={**_base(), "edge": edge, "evidence": []},
    )
    assert response.status_code == 200
    return response.json()["relationship"]["relationship_id"]


def _authorized_request(turn: dict[str, object], **overrides) -> dict[str, object]:
    session = turn["runtime_session"]
    runtime_turn = turn["runtime_turn"]
    request = {
        **_base(),
        "runtime_session_id": session["runtime_session_id"],
        "runtime_turn_id": runtime_turn["runtime_turn_id"],
        "active_persona_id": "technical_architect",
        "authorization_phase": "exposure",
        "capability_id": "runtime.world_state.read",
        "capability_domain": "software_architecture",
        "operation_class": "read",
        "argument_digest": None,
        "supported_surfaces": ["dev"],
        "relationship_requirements": [],
        "selected_relationship_ids": [],
        "world_state_requirements": [],
        "selected_world_state_claim_ids": [],
        "confirmation_challenge_ref": None,
    }
    request.update(overrides)
    return request


def test_exposure_allows_matching_persona_domain_and_surface():
    client = TestClient(app)
    turn = _start_turn(client)

    response = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(turn),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["allowed"] is True
    assert result["phase"] == "exposure"
    assert result["reason_codes"] == ["allowed"]


def test_exposure_denies_blocked_domain_and_unsupported_surface():
    client = TestClient(app)
    turn = _start_turn(client)

    response = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            capability_domain="personal_support",
            supported_surfaces=["web"],
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["allowed"] is False
    assert "capability_domain_denied" in result["reason_codes"]
    assert "surface_unsupported" in result["reason_codes"]


def test_relationship_required_authorization_rejects_revoked_sensitive_and_unselected_edges():
    client = TestClient(app)
    turn = _start_turn(client)
    rel_id = _relationship(client)

    allowed = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            relationship_requirements=[
                {"relationship_scope": "project_context", "relationship_type": "works_on"}
            ],
            selected_relationship_ids=[rel_id],
        ),
    ).json()["result"]
    assert allowed["allowed"] is True
    assert allowed["relationship_ids_used"] == [rel_id]

    sensitive_rel_id = _relationship(
        client,
        relationship_id="rel-sensitive",
        sensitivity_level="restricted",
    )
    denied = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            relationship_requirements=[
                {"relationship_scope": "project_context", "relationship_type": "works_on"}
            ],
            selected_relationship_ids=[sensitive_rel_id],
        ),
    ).json()["result"]
    assert denied["allowed"] is False
    assert "relationship_not_authorized" in denied["reason_codes"]

    revoked = client.post(
        "/v1/relationships/edges/revoke",
        json={**_base(), "relationship_id": rel_id},
    )
    assert revoked.status_code == 200
    denied_after_revoke = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            authorization_phase="dispatch",
            argument_digest="args_read_1",
            relationship_requirements=[
                {"relationship_scope": "project_context", "relationship_type": "works_on"}
            ],
            selected_relationship_ids=[rel_id],
        ),
    ).json()["result"]
    assert denied_after_revoke["allowed"] is False
    assert "relationship_not_authorized" in denied_after_revoke["reason_codes"]


def test_selection_dispatch_world_state_revalidation_then_verification_rerun():
    client = TestClient(app)
    turn = _start_turn(client)
    stale_claim = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                observed_at=_iso(-500),
                expires_at=None,
                ttl_seconds=None,
                revalidation_interval_seconds=300,
            ),
        },
    ).json()["claim"]
    request = _authorized_request(
        turn,
        authorization_phase="selection",
        argument_digest="args_read_1",
        world_state_requirements=[
            {
                "domain": "active_repository",
                "attribute": "branch_status",
                "min_authority": "verified_tool_output",
                "min_confidence": 0.9,
                "max_freshness_state": "fresh",
                "revalidator_id": "repo-status-revalidator",
            }
        ],
        selected_world_state_claim_ids=[stale_claim["world_state_claim_id"]],
    )

    stale = client.post("/v1/capabilities/authorize", json=request).json()["result"]
    assert stale["allowed"] is False
    assert stale["decision_code"] == "revalidation_required"
    assert stale["revalidation_selector"]["world_state_claim_ids"] == [
        stale_claim["world_state_claim_id"]
    ]

    verified = client.post(
        "/v1/world-state/claims/verify",
        json={
            **_base(),
            "request_id": "verify-for-auth",
            "runtime_session_id": turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": turn["runtime_turn"]["runtime_turn_id"],
            "world_state_claim_id": stale_claim["world_state_claim_id"],
            "expected_value_digest": stale_claim["value_digest"],
            "verification_source_type": "tool_output",
            "verification_source_ref": "repo-status-revalidator",
            "observed_at": _iso(-10),
            "verified_at": _iso(-5),
            "resulting_authority": "verified_tool_output",
            "resulting_confidence": 0.98,
            "resulting_freshness_state": "fresh",
            "resulting_expires_at": _iso(600),
            "resulting_ttl_seconds": 600,
            "resulting_revalidation_interval_seconds": 300,
        },
    )
    assert verified.status_code == 200
    rerun = client.post("/v1/capabilities/authorize", json=request).json()["result"]
    assert rerun["allowed"] is True
    assert rerun["world_state_claim_ids_used"] == [stale_claim["world_state_claim_id"]]


def test_inferred_low_confidence_state_does_not_authorize_risky_action():
    client = TestClient(app)
    turn = _start_turn(client)
    claim = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                source_type="model_inference",
                state_authority="model_inferred",
                confidence=0.4,
                confirmation_policy="confirm_before_high_impact_action",
            ),
        },
    ).json()["claim"]

    result = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            authorization_phase="selection",
            capability_id="integration.external_write.test",
            operation_class="external_write",
            argument_digest="args_external_1",
            world_state_requirements=[
                {
                    "domain": "active_repository",
                    "attribute": "branch_status",
                    "min_authority": "verified_tool_output",
                    "min_confidence": 0.9,
                }
            ],
            selected_world_state_claim_ids=[claim["world_state_claim_id"]],
        ),
    ).json()["result"]

    assert result["allowed"] is False
    assert "world_state_not_authorized" in result["reason_codes"]
    assert result["challenge_ref"] is None


def test_risky_confirmation_is_exact_one_use_and_privacy_safe():
    client = TestClient(app)
    turn = _start_turn(client)
    selection_request = _authorized_request(
        turn,
        authorization_phase="selection",
        capability_id="integration.external_write.test",
        operation_class="external_write",
        argument_digest="args_digest_stable",
    )

    selection = client.post("/v1/capabilities/authorize", json=selection_request).json()
    result = selection["result"]
    assert result["allowed"] is False
    assert result["decision_code"] == "confirmation_required"
    challenge_ref = result["challenge_ref"]

    mismatch = client.post(
        "/v1/capabilities/confirm",
        json={
            **_base(),
            "request_id": "confirm-mismatch",
            "runtime_session_id": turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": turn["runtime_turn"]["runtime_turn_id"],
            "confirmation_challenge_ref": challenge_ref,
            "capability_id": "integration.external_write.test",
            "operation_class": "external_write",
            "argument_digest": "args_digest_changed",
            "confirmed": True,
        },
    )
    assert mismatch.status_code == 400
    assert mismatch.json()["detail"] == "confirmation_challenge_mismatch"

    confirmed = client.post(
        "/v1/capabilities/confirm",
        json={
            **_base(),
            "request_id": "confirm-ok",
            "runtime_session_id": turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": turn["runtime_turn"]["runtime_turn_id"],
            "confirmation_challenge_ref": challenge_ref,
            "capability_id": "integration.external_write.test",
            "operation_class": "external_write",
            "argument_digest": "args_digest_stable",
            "confirmed": True,
        },
    )
    assert confirmed.status_code == 200

    dispatch_request = _authorized_request(
        turn,
        authorization_phase="dispatch",
        capability_id="integration.external_write.test",
        operation_class="external_write",
        argument_digest="args_digest_stable",
        confirmation_challenge_ref=challenge_ref,
    )
    dispatch = client.post("/v1/capabilities/authorize", json=dispatch_request).json()["result"]
    assert dispatch["allowed"] is True
    assert dispatch["confirmation_state"] == "accepted"

    replay = client.post("/v1/capabilities/authorize", json=dispatch_request).json()["result"]
    assert replay["allowed"] is False
    assert "challenge_consumed" in replay["reason_codes"]

    diagnostics = client.get(
        f"/v1/runtime/sessions/{turn['runtime_session']['runtime_session_id']}"
    ).json()
    diagnostics_text = str(diagnostics)
    assert "args_digest_stable" not in diagnostics_text
    assert "raw_secret_argument" not in diagnostics_text
    assert "integration.external_write.test" in diagnostics_text


def test_read_and_draft_do_not_require_risky_confirmation():
    client = TestClient(app)
    turn = _start_turn(client)

    read = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            authorization_phase="dispatch",
            operation_class="read",
            argument_digest="args_read_1",
        ),
    ).json()["result"]
    draft = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            authorization_phase="dispatch",
            capability_id="draft.local_message",
            operation_class="draft",
            argument_digest="args_draft_1",
        ),
    ).json()["result"]

    assert read["allowed"] is True
    assert read["confirmation_state"] == "not_required"
    assert draft["allowed"] is True
    assert draft["confirmation_state"] == "not_required"
