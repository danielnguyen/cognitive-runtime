from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from main import app
from services.world_state import (
    TrustedWorldStateVerifier,
    configure_trusted_world_state_verifiers_for_tests,
)


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


def _configure_repo_verifier() -> None:
    configure_trusted_world_state_verifiers_for_tests(
        [
            TrustedWorldStateVerifier(
                verifier_id="repo-status-revalidator",
                verification_source_type="tool_output",
                allowed_source_refs=frozenset({"repo-status-revalidator"}),
                max_authority="verified_tool_output",
                allowed_domains=frozenset({"active_repository"}),
                allowed_attributes=frozenset({"branch_status", "build_status"}),
                max_confidence=0.99,
                max_freshness_state="fresh",
                max_ttl_seconds=600,
                max_revalidation_interval_seconds=300,
            )
        ]
    )


def _configure_two_repo_verifiers() -> None:
    configure_trusted_world_state_verifiers_for_tests(
        [
            TrustedWorldStateVerifier(
                verifier_id="repo-status-revalidator",
                verification_source_type="tool_output",
                allowed_source_refs=frozenset({"repo-status-revalidator"}),
                max_authority="verified_tool_output",
                allowed_domains=frozenset({"active_repository"}),
                allowed_attributes=frozenset({"branch_status"}),
                max_confidence=0.99,
                max_freshness_state="fresh",
                max_ttl_seconds=600,
                max_revalidation_interval_seconds=300,
            ),
            TrustedWorldStateVerifier(
                verifier_id="build-status-revalidator",
                verification_source_type="tool_output",
                allowed_source_refs=frozenset({"build-status-revalidator"}),
                max_authority="verified_tool_output",
                allowed_domains=frozenset({"active_repository"}),
                allowed_attributes=frozenset({"build_status"}),
                max_confidence=0.99,
                max_freshness_state="fresh",
                max_ttl_seconds=600,
                max_revalidation_interval_seconds=300,
            ),
        ]
    )


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


def _consumed_event_count(client: TestClient, runtime_session_id: str) -> int:
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}").json()
    return sum(
        1
        for event in diagnostics["events"]
        if event["event_type"] == "confirmation_challenge_evaluated"
        and event["event_payload_json"].get("confirmation_state") == "consumed"
    )


def _issue_confirmed_challenge(
    client: TestClient,
    *,
    argument_digest: str = "args_dispatch_bound",
    request_overrides: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object], str]:
    origin_turn = _start_turn(client)
    overrides = dict(request_overrides or {})
    capability_id = str(overrides.pop("capability_id", "integration.external_write.test"))
    operation_class = str(overrides.pop("operation_class", "external_write"))
    selection_request = _authorized_request(
        origin_turn,
        authorization_phase="selection",
        capability_id=capability_id,
        operation_class=operation_class,
        argument_digest=argument_digest,
        **overrides,
    )
    selection = client.post("/v1/capabilities/authorize", json=selection_request).json()[
        "result"
    ]
    assert selection["decision_code"] == "confirmation_required"
    confirm_turn = _start_turn(client)
    confirmed = client.post(
        "/v1/capabilities/confirm",
        json={
            **_base(),
            "request_id": "confirm-dispatch-bound",
            "runtime_session_id": origin_turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": confirm_turn["runtime_turn"]["runtime_turn_id"],
            "confirmation_challenge_ref": selection["challenge_ref"],
            "capability_id": capability_id,
            "operation_class": operation_class,
            "argument_digest": argument_digest,
            "confirmed": True,
        },
    )
    assert confirmed.status_code == 200
    return origin_turn, confirm_turn, selection["challenge_ref"]


def _risky_dispatch_request(
    turn: dict[str, object],
    challenge_ref: str,
    *,
    argument_digest: str = "args_dispatch_bound",
    **overrides,
) -> dict[str, object]:
    return _authorized_request(
        turn,
        authorization_phase="dispatch",
        capability_id="integration.external_write.test",
        operation_class="external_write",
        argument_digest=argument_digest,
        confirmation_challenge_ref=challenge_ref,
        **overrides,
    )


def _complete_turn(
    client: TestClient,
    turn: dict[str, object],
    *,
    status: str = "completed",
) -> None:
    response = client.post(
        "/v1/runtime/turns/complete",
        json={
            "request_id": f"complete-{status}",
            "runtime_session_id": turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": turn["runtime_turn"]["runtime_turn_id"],
            "turn_status": status,
        },
    )
    assert response.status_code == 200


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
    _configure_repo_verifier()
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
            "verifier_id": "repo-status-revalidator",
            "verification_source_type": "tool_output",
            "verification_source_ref": "repo-status-revalidator",
            "observed_at": _iso(-10),
            "verified_at": _iso(-5),
            "resulting_authority": "verified_tool_output",
            "resulting_confidence": 0.98,
            "resulting_freshness_state": "fresh",
            "resulting_ttl_seconds": 600,
            "resulting_revalidation_interval_seconds": 300,
        },
    )
    assert verified.status_code == 200
    rerun = client.post("/v1/capabilities/authorize", json=request).json()["result"]
    assert rerun["allowed"] is True
    assert rerun["world_state_claim_ids_used"] == [stale_claim["world_state_claim_id"]]


def test_reverify_before_use_requires_current_turn_verification_then_rerun_allows():
    client = TestClient(app)
    _configure_repo_verifier()
    turn = _start_turn(client)
    claim = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(confirmation_policy="reverify_before_use"),
        },
    ).json()["claim"]
    request = _authorized_request(
        turn,
        authorization_phase="selection",
        argument_digest="args_reverify_policy",
        world_state_requirements=[
            {
                "domain": "active_repository",
                "attribute": "branch_status",
                "revalidator_id": "repo-status-revalidator",
            }
        ],
        selected_world_state_claim_ids=[claim["world_state_claim_id"]],
    )

    first = client.post("/v1/capabilities/authorize", json=request).json()["result"]
    assert first["allowed"] is False
    assert first["decision_code"] == "revalidation_required"
    assert first["revalidation_selector"]["world_state_claim_ids"] == [
        claim["world_state_claim_id"]
    ]

    verified = client.post(
        "/v1/world-state/claims/verify",
        json={
            **_base(),
            "request_id": "verify-reverify-policy",
            "runtime_session_id": turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": turn["runtime_turn"]["runtime_turn_id"],
            "world_state_claim_id": claim["world_state_claim_id"],
            "expected_value_digest": claim["value_digest"],
            "verifier_id": "repo-status-revalidator",
            "verification_source_type": "tool_output",
            "verification_source_ref": "repo-status-revalidator",
            "observed_at": _iso(-10),
            "verified_at": _iso(-5),
            "resulting_authority": "verified_tool_output",
            "resulting_confidence": 0.98,
            "resulting_freshness_state": "fresh",
            "resulting_ttl_seconds": 600,
            "resulting_revalidation_interval_seconds": 300,
        },
    )
    assert verified.status_code == 200

    rerun = client.post("/v1/capabilities/authorize", json=request).json()["result"]
    assert rerun["allowed"] is True
    assert rerun["revalidation_required"] is False


def test_stale_world_state_without_configured_verifier_denies_without_selector():
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

    result = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            authorization_phase="selection",
            argument_digest="args_stale_without_revalidator",
            world_state_requirements=[{"domain": "active_repository"}],
            selected_world_state_claim_ids=[stale_claim["world_state_claim_id"]],
        ),
    ).json()["result"]

    assert result["allowed"] is False
    assert result["decision_code"] == "authorization_denied"
    assert "world_state_revalidator_required" in result["reason_codes"]
    assert result["revalidation_selector"] is None
    assert result["world_state_claim_ids_used"] == []


def test_unknown_or_unauthorized_revalidator_returns_no_selector():
    client = TestClient(app)
    _configure_repo_verifier()
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
    build_claim = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                attribute="other_status",
                value_json={"status": "stale"},
                observed_at=_iso(-500),
                expires_at=None,
                ttl_seconds=None,
                revalidation_interval_seconds=300,
            ),
        },
    ).json()["claim"]

    unknown = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            authorization_phase="selection",
            argument_digest="args_unknown_revalidator",
            world_state_requirements=[
                {"domain": "active_repository", "revalidator_id": "missing-revalidator"}
            ],
            selected_world_state_claim_ids=[stale_claim["world_state_claim_id"]],
        ),
    ).json()["result"]
    unauthorized = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            authorization_phase="selection",
            argument_digest="args_unauthorized_revalidator",
            world_state_requirements=[
                {"domain": "active_repository", "revalidator_id": "repo-status-revalidator"}
            ],
            selected_world_state_claim_ids=[build_claim["world_state_claim_id"]],
        ),
    ).json()["result"]

    assert "world_state_revalidator_not_configured" in unknown["reason_codes"]
    assert unknown["revalidation_selector"] is None
    assert "world_state_revalidator_not_authorized" in unauthorized["reason_codes"]
    assert unauthorized["revalidation_selector"] is None


def test_low_confidence_and_authority_use_configured_revalidation_rule():
    client = TestClient(app)
    _configure_repo_verifier()
    turn = _start_turn(client)
    claim = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(confidence=0.5, state_authority="observed_user_report"),
        },
    ).json()["claim"]

    result = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            authorization_phase="selection",
            argument_digest="args_low_quality_revalidator",
            world_state_requirements=[
                {
                    "domain": "active_repository",
                    "attribute": "branch_status",
                    "min_authority": "verified_tool_output",
                    "min_confidence": 0.9,
                    "revalidator_id": "repo-status-revalidator",
                }
            ],
            selected_world_state_claim_ids=[claim["world_state_claim_id"]],
        ),
    ).json()["result"]

    assert result["decision_code"] == "revalidation_required"
    assert result["revalidation_selector"] == {
        "world_state_claim_ids": [claim["world_state_claim_id"]],
        "revalidator_id": "repo-status-revalidator",
    }
    assert result["world_state_claim_ids_used"] == []


def test_mixed_revalidators_fail_closed_without_mixing_claim_ids():
    client = TestClient(app)
    _configure_two_repo_verifiers()
    turn = _start_turn(client)
    branch_claim = client.post(
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
    build_claim = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                attribute="build_status",
                value_json={"status": "stale"},
                observed_at=_iso(-500),
                expires_at=None,
                ttl_seconds=None,
                revalidation_interval_seconds=300,
            ),
        },
    ).json()["claim"]

    result = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            authorization_phase="selection",
            argument_digest="args_mixed_revalidators",
            world_state_requirements=[
                {
                    "domain": "active_repository",
                    "attribute": "branch_status",
                    "revalidator_id": "repo-status-revalidator",
                },
                {
                    "domain": "active_repository",
                    "attribute": "build_status",
                    "revalidator_id": "build-status-revalidator",
                },
            ],
            selected_world_state_claim_ids=[
                branch_claim["world_state_claim_id"],
                build_claim["world_state_claim_id"],
            ],
        ),
    ).json()["result"]

    assert result["decision_code"] == "authorization_denied"
    assert "world_state_revalidator_conflict" in result["reason_codes"]
    assert result["revalidation_selector"] is None
    assert result["world_state_claim_ids_used"] == []


def test_hard_denial_takes_precedence_over_revalidation_and_challenge():
    client = TestClient(app)
    _configure_repo_verifier()
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

    result = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            active_persona_id="personal_companion",
            authorization_phase="selection",
            operation_class="external_write",
            argument_digest="args_hard_denial_plus_stale",
            world_state_requirements=[
                {"domain": "active_repository", "revalidator_id": "repo-status-revalidator"}
            ],
            selected_world_state_claim_ids=[stale_claim["world_state_claim_id"]],
        ),
    ).json()["result"]

    assert result["decision_code"] == "authorization_denied"
    assert "persona_mismatch" in result["reason_codes"]
    assert result["revalidation_selector"] is None
    assert result["challenge_ref"] is None


def test_revalidation_required_selection_issues_no_confirmation_challenge():
    client = TestClient(app)
    _configure_repo_verifier()
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

    result = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            authorization_phase="selection",
            operation_class="external_write",
            argument_digest="args_revalidation_no_challenge",
            world_state_requirements=[
                {"domain": "active_repository", "revalidator_id": "repo-status-revalidator"}
            ],
            selected_world_state_claim_ids=[stale_claim["world_state_claim_id"]],
        ),
    ).json()["result"]

    assert result["decision_code"] == "revalidation_required"
    assert result["confirmation_state"] == "required"
    assert result["challenge_ref"] is None


def test_current_turn_verification_that_remains_inadequate_does_not_loop():
    client = TestClient(app)
    _configure_repo_verifier()
    turn = _start_turn(client)
    claim = client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim(source_type="user_report")},
    ).json()["claim"]
    verified = client.post(
        "/v1/world-state/claims/verify",
        json={
            **_base(),
            "request_id": "verify-current-turn-low-confidence",
            "runtime_session_id": turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": turn["runtime_turn"]["runtime_turn_id"],
            "world_state_claim_id": claim["world_state_claim_id"],
            "expected_value_digest": claim["value_digest"],
            "verifier_id": "repo-status-revalidator",
            "verification_source_type": "tool_output",
            "verification_source_ref": "repo-status-revalidator",
            "observed_at": _iso(-10),
            "verified_at": _iso(-5),
            "resulting_authority": "verified_tool_output",
            "resulting_confidence": 0.5,
            "resulting_freshness_state": "fresh",
        },
    )
    assert verified.status_code == 200

    result = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            authorization_phase="selection",
            argument_digest="args_current_turn_inadequate",
            world_state_requirements=[
                {
                    "domain": "active_repository",
                    "attribute": "branch_status",
                    "min_confidence": 0.9,
                    "revalidator_id": "repo-status-revalidator",
                }
            ],
            selected_world_state_claim_ids=[claim["world_state_claim_id"]],
        ),
    ).json()["result"]

    assert result["decision_code"] == "authorization_denied"
    assert "world_state_revalidation_inadequate" in result["reason_codes"]
    assert result["revalidation_selector"] is None


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
    assert "world_state_revalidator_required" in result["reason_codes"]
    assert result["challenge_ref"] is None


def test_world_state_authorization_uses_persona_surface_scope_and_sensitivity():
    client = TestClient(app)
    turn = _start_turn(client)
    health_claim = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                entity_id="health:private",
                entity_type="health_observation",
                domain="active_health_observation",
                attribute="condition",
                sensitivity="medium",
            ),
        },
    ).json()["claim"]
    restricted_claim = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                entity_id="repo:secret",
                domain="active_repository",
                sensitivity="restricted",
            ),
        },
    ).json()["claim"]

    outside_scope = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            world_state_requirements=[{"domain": "active_health_observation"}],
            selected_world_state_claim_ids=[health_claim["world_state_claim_id"]],
        ),
    ).json()["result"]
    restricted = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            world_state_requirements=[{"domain": "active_repository"}],
            selected_world_state_claim_ids=[restricted_claim["world_state_claim_id"]],
        ),
    ).json()["result"]

    assert outside_scope["allowed"] is False
    assert "world_state_required" in outside_scope["reason_codes"]
    assert restricted["allowed"] is False
    assert "world_state_not_authorized" in restricted["reason_codes"]


def test_world_state_confirmation_policy_requires_current_confirmation_for_non_read():
    client = TestClient(app)
    turn = _start_turn(client)
    claim = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(confirmation_policy="confirm_before_action"),
        },
    ).json()["claim"]
    request = _authorized_request(
        turn,
        authorization_phase="selection",
        capability_id="draft.local_message",
        operation_class="draft",
        argument_digest="args_draft_claim",
        world_state_requirements=[{"domain": "active_repository"}],
        selected_world_state_claim_ids=[claim["world_state_claim_id"]],
    )

    selection = client.post("/v1/capabilities/authorize", json=request).json()["result"]
    assert selection["allowed"] is False
    assert selection["decision_code"] == "confirmation_required"
    assert selection["challenge_ref"]

    next_turn = _start_turn(client)
    confirmed = client.post(
        "/v1/capabilities/confirm",
        json={
            **_base(),
            "request_id": "confirm-claim-policy",
            "runtime_session_id": turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": next_turn["runtime_turn"]["runtime_turn_id"],
            "confirmation_challenge_ref": selection["challenge_ref"],
            "capability_id": "draft.local_message",
            "operation_class": "draft",
            "argument_digest": "args_draft_claim",
            "confirmed": True,
        },
    )
    assert confirmed.status_code == 200
    dispatch = client.post(
        "/v1/capabilities/authorize",
        json={
            **request,
            "runtime_turn_id": next_turn["runtime_turn"]["runtime_turn_id"],
            "authorization_phase": "dispatch",
            "confirmation_challenge_ref": selection["challenge_ref"],
        },
    ).json()["result"]
    assert dispatch["allowed"] is True


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

    same_turn = client.post(
        "/v1/capabilities/confirm",
        json={
            **_base(),
            "request_id": "confirm-same-turn",
            "runtime_session_id": turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": turn["runtime_turn"]["runtime_turn_id"],
            "confirmation_challenge_ref": challenge_ref,
            "capability_id": "integration.external_write.test",
            "operation_class": "external_write",
            "argument_digest": "args_digest_stable",
            "confirmed": True,
        },
    )
    assert same_turn.status_code == 400
    assert same_turn.json()["detail"] == "confirmation_turn_not_distinct"

    next_turn = _start_turn(client)
    confirmed = client.post(
        "/v1/capabilities/confirm",
        json={
            **_base(),
            "request_id": "confirm-ok",
            "runtime_session_id": turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": next_turn["runtime_turn"]["runtime_turn_id"],
            "confirmation_challenge_ref": challenge_ref,
            "capability_id": "integration.external_write.test",
            "operation_class": "external_write",
            "argument_digest": "args_digest_stable",
            "confirmed": True,
        },
    )
    assert confirmed.status_code == 200

    dispatch_request = _authorized_request(
        next_turn,
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


def test_explicit_confirmation_rejection_cannot_later_dispatch_or_be_accepted():
    client = TestClient(app)
    turn = _start_turn(client)
    selection = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            authorization_phase="selection",
            capability_id="integration.external_write.test",
            operation_class="external_write",
            argument_digest="args_rejected",
        ),
    ).json()["result"]
    next_turn = _start_turn(client)
    rejected = client.post(
        "/v1/capabilities/confirm",
        json={
            **_base(),
            "request_id": "confirm-reject",
            "runtime_session_id": turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": next_turn["runtime_turn"]["runtime_turn_id"],
            "confirmation_challenge_ref": selection["challenge_ref"],
            "capability_id": "integration.external_write.test",
            "operation_class": "external_write",
            "argument_digest": "args_rejected",
            "confirmed": False,
        },
    )
    assert rejected.status_code == 200
    later_accept = client.post(
        "/v1/capabilities/confirm",
        json={
            **_base(),
            "request_id": "confirm-after-reject",
            "runtime_session_id": turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": next_turn["runtime_turn"]["runtime_turn_id"],
            "confirmation_challenge_ref": selection["challenge_ref"],
            "capability_id": "integration.external_write.test",
            "operation_class": "external_write",
            "argument_digest": "args_rejected",
            "confirmed": True,
        },
    )
    assert later_accept.status_code == 409
    dispatch = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            next_turn,
            authorization_phase="dispatch",
            capability_id="integration.external_write.test",
            operation_class="external_write",
            argument_digest="args_rejected",
            confirmation_challenge_ref=selection["challenge_ref"],
        ),
    ).json()["result"]
    assert dispatch["allowed"] is False
    assert "challenge_rejected" in dispatch["reason_codes"]


def test_concurrent_dispatch_consumes_confirmation_once():
    client = TestClient(app)
    turn = _start_turn(client)
    selection = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            authorization_phase="selection",
            capability_id="integration.external_write.test",
            operation_class="external_write",
            argument_digest="args_concurrent",
        ),
    ).json()["result"]
    next_turn = _start_turn(client)
    confirmed = client.post(
        "/v1/capabilities/confirm",
        json={
            **_base(),
            "request_id": "confirm-concurrent",
            "runtime_session_id": turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": next_turn["runtime_turn"]["runtime_turn_id"],
            "confirmation_challenge_ref": selection["challenge_ref"],
            "capability_id": "integration.external_write.test",
            "operation_class": "external_write",
            "argument_digest": "args_concurrent",
            "confirmed": True,
        },
    )
    assert confirmed.status_code == 200
    dispatch_request = _authorized_request(
        next_turn,
        authorization_phase="dispatch",
        capability_id="integration.external_write.test",
        operation_class="external_write",
        argument_digest="args_concurrent",
        confirmation_challenge_ref=selection["challenge_ref"],
    )

    def dispatch_once() -> dict[str, object]:
        return TestClient(app).post(
            "/v1/capabilities/authorize",
            json=dispatch_request,
        ).json()["result"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: dispatch_once(), range(2)))

    allowed = [result for result in results if result["allowed"] is True]
    denied = [result for result in results if result["allowed"] is False]
    assert len(allowed) == 1
    assert len(denied) == 1
    assert "challenge_consumed" in denied[0]["reason_codes"]
    diagnostics = client.get(
        f"/v1/runtime/sessions/{turn['runtime_session']['runtime_session_id']}"
    ).json()
    consumed_events = [
        event
        for event in diagnostics["events"]
        if event["event_type"] == "confirmation_challenge_evaluated"
        and event["event_payload_json"].get("confirmation_state") == "consumed"
    ]
    assert len(consumed_events) == 1


def test_dispatch_requires_confirmed_current_turn_and_missing_turn_does_not_consume():
    client = TestClient(app)
    origin_turn, confirm_turn, challenge_ref = _issue_confirmed_challenge(client)
    runtime_session_id = origin_turn["runtime_session"]["runtime_session_id"]

    missing_turn_request = _risky_dispatch_request(confirm_turn, challenge_ref)
    missing_turn_request["runtime_turn_id"] = None
    missing_turn = client.post(
        "/v1/capabilities/authorize",
        json=missing_turn_request,
    ).json()["result"]
    assert missing_turn["allowed"] is False
    assert "dispatch_turn_required" in missing_turn["reason_codes"]
    assert _consumed_event_count(client, runtime_session_id) == 0

    dispatch = client.post(
        "/v1/capabilities/authorize",
        json=_risky_dispatch_request(confirm_turn, challenge_ref),
    ).json()["result"]
    assert dispatch["allowed"] is True
    assert _consumed_event_count(client, runtime_session_id) == 1


def test_confirmation_on_turn_cannot_dispatch_on_later_turn_without_consuming():
    client = TestClient(app)
    origin_turn, confirm_turn, challenge_ref = _issue_confirmed_challenge(client)
    later_turn = _start_turn(client)
    runtime_session_id = origin_turn["runtime_session"]["runtime_session_id"]

    later_dispatch = client.post(
        "/v1/capabilities/authorize",
        json=_risky_dispatch_request(later_turn, challenge_ref),
    ).json()["result"]
    assert later_dispatch["allowed"] is False
    assert "challenge_turn_mismatch" in later_dispatch["reason_codes"]
    assert _consumed_event_count(client, runtime_session_id) == 0

    older_dispatch = client.post(
        "/v1/capabilities/authorize",
        json=_risky_dispatch_request(confirm_turn, challenge_ref),
    ).json()["result"]
    assert older_dispatch["allowed"] is False
    assert "confirmation_turn_not_current" in older_dispatch["reason_codes"]
    assert _consumed_event_count(client, runtime_session_id) == 0


def test_completed_or_abandoned_confirmation_turn_cannot_dispatch_or_consume():
    for status in ("completed", "abandoned"):
        client = TestClient(app)
        origin_turn, confirm_turn, challenge_ref = _issue_confirmed_challenge(
            client,
            argument_digest=f"args_{status}",
        )
        runtime_session_id = origin_turn["runtime_session"]["runtime_session_id"]
        _complete_turn(client, confirm_turn, status=status)

        dispatch = client.post(
            "/v1/capabilities/authorize",
            json=_risky_dispatch_request(
                confirm_turn,
                challenge_ref,
                argument_digest=f"args_{status}",
            ),
        ).json()["result"]
        assert dispatch["allowed"] is False
        assert "confirmation_turn_not_current" in dispatch["reason_codes"]
        assert _consumed_event_count(client, runtime_session_id) == 0


def test_persona_mismatch_dispatch_denies_without_consuming_then_corrected_dispatch_succeeds():
    client = TestClient(app)
    origin_turn, confirm_turn, challenge_ref = _issue_confirmed_challenge(client)
    runtime_session_id = origin_turn["runtime_session"]["runtime_session_id"]

    denied = client.post(
        "/v1/capabilities/authorize",
        json=_risky_dispatch_request(
            confirm_turn,
            challenge_ref,
            active_persona_id="personal_companion",
        ),
    ).json()["result"]
    assert denied["allowed"] is False
    assert "persona_mismatch" in denied["reason_codes"]
    assert _consumed_event_count(client, runtime_session_id) == 0

    corrected = client.post(
        "/v1/capabilities/authorize",
        json=_risky_dispatch_request(confirm_turn, challenge_ref),
    ).json()["result"]
    assert corrected["allowed"] is True
    assert _consumed_event_count(client, runtime_session_id) == 1


def test_relationship_denial_after_confirmation_does_not_consume():
    client = TestClient(app)
    rel_id = _relationship(client)
    origin_turn, confirm_turn, challenge_ref = _issue_confirmed_challenge(
        client,
        request_overrides={
            "relationship_requirements": [
                {"relationship_scope": "project_context", "relationship_type": "works_on"}
            ],
            "selected_relationship_ids": [rel_id],
        },
    )
    runtime_session_id = origin_turn["runtime_session"]["runtime_session_id"]
    revoked = client.post(
        "/v1/relationships/edges/revoke",
        json={**_base(), "relationship_id": rel_id},
    )
    assert revoked.status_code == 200

    dispatch = client.post(
        "/v1/capabilities/authorize",
        json=_risky_dispatch_request(
            confirm_turn,
            challenge_ref,
            relationship_requirements=[
                {"relationship_scope": "project_context", "relationship_type": "works_on"}
            ],
            selected_relationship_ids=[rel_id],
        ),
    ).json()["result"]
    assert dispatch["allowed"] is False
    assert "relationship_not_authorized" in dispatch["reason_codes"]
    assert _consumed_event_count(client, runtime_session_id) == 0


def test_world_state_denial_and_revalidation_after_confirmation_do_not_consume():
    client = TestClient(app)
    _configure_repo_verifier()
    origin_turn = _start_turn(client)
    claim = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(confirmation_policy="confirm_before_action"),
        },
    ).json()["claim"]
    selection_request = _authorized_request(
        origin_turn,
        authorization_phase="selection",
        capability_id="draft.local_message",
        operation_class="draft",
        argument_digest="args_world_denied",
        world_state_requirements=[{"domain": "active_repository"}],
        selected_world_state_claim_ids=[claim["world_state_claim_id"]],
    )
    selection = client.post("/v1/capabilities/authorize", json=selection_request).json()[
        "result"
    ]
    confirm_turn = _start_turn(client)
    confirmed = client.post(
        "/v1/capabilities/confirm",
        json={
            **_base(),
            "request_id": "confirm-world-denial",
            "runtime_session_id": origin_turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": confirm_turn["runtime_turn"]["runtime_turn_id"],
            "confirmation_challenge_ref": selection["challenge_ref"],
            "capability_id": "draft.local_message",
            "operation_class": "draft",
            "argument_digest": "args_world_denied",
            "confirmed": True,
        },
    )
    assert confirmed.status_code == 200
    client.post(
        "/v1/world-state/claims/upsert",
        json={**_base(), "claim": _claim(value_json={"status": "blocked"})},
    )

    denied = client.post(
        "/v1/capabilities/authorize",
        json={
            **selection_request,
            "runtime_turn_id": confirm_turn["runtime_turn"]["runtime_turn_id"],
            "authorization_phase": "dispatch",
            "confirmation_challenge_ref": selection["challenge_ref"],
        },
    ).json()["result"]
    assert denied["allowed"] is False
    assert "world_state_required" in denied["reason_codes"]
    runtime_session_id = origin_turn["runtime_session"]["runtime_session_id"]
    assert _consumed_event_count(client, runtime_session_id) == 0

    fresh_for_revalidation = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                entity_id="repo:stale",
                attribute="build_status",
                value_json={"status": "stale"},
                confirmation_policy="confirm_before_action",
            ),
        },
    ).json()["claim"]
    stale_origin, stale_turn, stale_challenge = _issue_confirmed_challenge(
        client,
        argument_digest="args_stale_denied",
        request_overrides={
            "capability_id": "draft.local_message",
            "operation_class": "draft",
            "world_state_requirements": [
                {"domain": "active_repository", "revalidator_id": "repo-status-revalidator"}
            ],
            "selected_world_state_claim_ids": [
                fresh_for_revalidation["world_state_claim_id"]
            ],
        },
    )
    stale_claim = client.post(
        "/v1/world-state/claims/upsert",
        json={
            **_base(),
            "claim": _claim(
                entity_id="repo:stale",
                attribute="build_status",
                value_json={"status": "stale"},
                observed_at=_iso(-500),
                expires_at=None,
                ttl_seconds=None,
                revalidation_interval_seconds=300,
                confirmation_policy="confirm_before_action",
            ),
        },
    ).json()["claim"]
    assert stale_claim["world_state_claim_id"] == fresh_for_revalidation["world_state_claim_id"]
    revalidation = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            stale_turn,
            authorization_phase="dispatch",
            capability_id="draft.local_message",
            operation_class="draft",
            argument_digest="args_stale_denied",
            world_state_requirements=[
                {"domain": "active_repository", "revalidator_id": "repo-status-revalidator"}
            ],
            selected_world_state_claim_ids=[stale_claim["world_state_claim_id"]],
            confirmation_challenge_ref=stale_challenge,
        ),
    ).json()["result"]
    assert revalidation["allowed"] is False
    assert revalidation["decision_code"] == "revalidation_required"
    assert _consumed_event_count(
        client,
        stale_origin["runtime_session"]["runtime_session_id"],
    ) == 0


def test_challenge_tuple_mismatch_denies_without_consuming_then_corrected_dispatch_succeeds():
    client = TestClient(app)
    origin_turn, confirm_turn, challenge_ref = _issue_confirmed_challenge(client)
    runtime_session_id = origin_turn["runtime_session"]["runtime_session_id"]

    mismatch = client.post(
        "/v1/capabilities/authorize",
        json=_risky_dispatch_request(
            confirm_turn,
            challenge_ref,
            argument_digest="args_changed",
        ),
    ).json()["result"]
    assert mismatch["allowed"] is False
    assert "challenge_mismatch" in mismatch["reason_codes"]
    assert _consumed_event_count(client, runtime_session_id) == 0

    corrected = client.post(
        "/v1/capabilities/authorize",
        json=_risky_dispatch_request(confirm_turn, challenge_ref),
    ).json()["result"]
    assert corrected["allowed"] is True
    assert _consumed_event_count(client, runtime_session_id) == 1


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
