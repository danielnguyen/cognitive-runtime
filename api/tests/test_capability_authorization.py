from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import services.capability_authorization as capability_authorization_service
from fastapi.testclient import TestClient
from main import app
from services.capability_authorization import (
    capability_authorization_repository,
    configure_capability_registry_for_tests,
)
from services.world_state import (
    TrustedWorldStateVerifier,
    configure_trusted_world_state_verifiers_for_tests,
)


def _iso(delta_seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=delta_seconds)).isoformat()


def _base(*, surface: str = "dev") -> dict[str, object]:
    return {
        "request_id": "capability-test",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": surface,
    }


@pytest.fixture(autouse=True)
def _available_capability_registry():
    configure_capability_registry_for_tests(available=True)
    yield
    configure_capability_registry_for_tests(available=True)


def _start_turn(
    client: TestClient,
    *,
    surface: str = "dev",
    owner_id: str = "owner",
    conversation_id: str = "conv-1",
) -> dict[str, object]:
    return client.post(
        "/v1/runtime/turns/start",
        json={
            **_base(surface=surface),
            "request_id": "capability-turn",
            "owner_id": owner_id,
            "conversation_id": conversation_id,
        },
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


def _relationship_requirement(
    *,
    relationship_scope: str = "project_context",
    relationship_type: str = "works_on",
) -> list[dict[str, str]]:
    return [
        {
            "relationship_scope": relationship_scope,
            "relationship_type": relationship_type,
        }
    ]


def _relationship_authorization_result(
    client: TestClient,
    turn: dict[str, object],
    *,
    relationship_id: str | None = None,
    relationship_requirements: list[dict[str, str]] | None = None,
    **overrides,
) -> dict[str, object]:
    selected_relationship_ids = [relationship_id] if relationship_id else []
    response = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            relationship_requirements=(
                relationship_requirements
                if relationship_requirements is not None
                else _relationship_requirement()
            ),
            selected_relationship_ids=selected_relationship_ids,
            **overrides,
        ),
    )
    assert response.status_code == 200
    return response.json()["result"]


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
        "supported_surfaces": ["dev", "vscode"],
        "relationship_requirements": [],
        "selected_relationship_ids": [],
        "world_state_requirements": [],
        "selected_world_state_claim_ids": [],
        "confirmation_challenge_ref": None,
    }
    request.update(overrides)
    return request


def _authorization_request(
    turn: dict[str, object],
    *,
    authorization_stage: str,
    **overrides,
) -> dict[str, object]:
    request = _authorized_request(turn, **overrides)
    request["authorization_" + "pha" + "se"] = authorization_stage
    return request


def _jellyfin_authorization_request(
    turn: dict[str, object],
    *,
    authorization_stage: str | None = None,
    **overrides,
) -> dict[str, object]:
    request = (
        _authorization_request(turn, authorization_stage=authorization_stage)
        if authorization_stage is not None
        else _authorized_request(turn)
    )
    request.update(
        {
            "capability_id": "jellyfin_restart",
            "capability_domain": "media_operations",
            "operation_class": "high_impact",
            "supported_surfaces": ["desktop", "dev"],
        }
    )
    request.update(overrides)
    return request


def _challenge_row_count() -> int:
    with capability_authorization_repository()._connect() as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM capability_confirmation_challenges;"
            ).fetchone()[0]
        )


def _issued_challenge_event_count() -> int:
    with capability_authorization_repository()._connect() as conn:
        return int(
            conn.execute(
                """
                SELECT COUNT(*) FROM conversation_runtime_events
                WHERE event_type = 'confirmation_challenge_evaluated'
                  AND event_payload_json LIKE '%\"confirmation_state\":\"issued\"%';
                """
            ).fetchone()[0]
        )


def _challenge_record(challenge_ref: str) -> dict[str, object]:
    with capability_authorization_repository()._connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM capability_confirmation_challenges
            WHERE confirmation_challenge_ref = ?;
            """,
            (challenge_ref,),
        ).fetchone()
    assert row is not None
    return dict(row)


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


def _match_request(text: str, *, surface: str = "desktop", persona: str = "home_operator"):
    return {
        **_base(surface=surface),
        "request_id": "capability-match",
        "active_persona_id": persona,
        "current_user_text": text,
    }


def _discovery_request(*, surface: str = "desktop", persona: str = "home_operator"):
    return {
        **_base(surface=surface),
        "request_id": "capability-discovery",
        "active_persona_id": persona,
    }


def _authority_request(
    capability_id: str,
    *,
    surface: str = "desktop",
    persona: str = "home_operator",
    **overrides,
) -> dict[str, object]:
    request = {
        **_base(surface=surface),
        "request_id": "capability-authority",
        "active_persona_id": persona,
        "capability_id": capability_id,
        "target_resolution_state": "resolved",
        "world_state_freshness": "fresh",
        "consequence_flags": {},
        "user_authorization_signal": "explicit",
    }
    request.update(overrides)
    return request


def _flow_request(
    capability_id: str,
    *,
    surface: str = "desktop",
    persona: str = "home_operator",
    **overrides,
) -> dict[str, object]:
    request = {
        **_base(surface=surface),
        "request_id": "capability-flow",
        "active_persona_id": persona,
        "capability_id": capability_id,
        "flow_intent": "execution_requested",
        "target_resolution_state": "resolved",
        "world_state_freshness": "fresh",
        "affects_multiple_systems": False,
        "consequence_flags": {},
        "user_authorization_signal": "explicit",
    }
    request.update(overrides)
    return request


def _action_summary_request(
    client: TestClient,
    *,
    request_id: str = "action-summary",
    owner_id: str = "owner",
    conversation_id: str = "conv-1",
    surface: str = "dev",
    runtime_turn_id: str | None = None,
    **overrides,
) -> dict[str, object]:
    session_response = client.post(
        "/v1/runtime/sessions/resolve",
        json={
            "request_id": f"{request_id}-session",
            "owner_id": owner_id,
            "conversation_id": conversation_id,
            "surface": surface,
        },
    )
    assert session_response.status_code == 200
    runtime_session_id = session_response.json()["runtime_session"][
        "runtime_session_id"
    ]
    request = {
        "request_id": request_id,
        "owner_id": owner_id,
        "conversation_id": conversation_id,
        "surface": surface,
        "runtime_session_id": runtime_session_id,
        "runtime_turn_id": runtime_turn_id,
        "capability_id": "runtime.world_state.read",
        "active_persona_id": "technical_architect",
        "risk_level": "read_only",
        "authority_level": "answer_only",
        "confirmation_status": "not_required",
        "policy_reason_codes": ["registered_capability", "execution_allowed_by_policy"],
        "execution_status": "executed",
        "execution_reason_code": "adapter_completed",
        "verification_status": "passed",
        "verification_reason_code": "result_check_passed",
        "degradation_reason": None,
    }
    request.update(overrides)
    return request


def test_registered_natural_language_request_matches_capability():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/match",
        json=_match_request("Please turn on the office lights."),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["capability_matched"] is True
    assert result["action_taken"] is False
    assert result["reason_codes"] == ["matched"]
    assert result["capability"]["capability_id"] == "office_lights_on"


def test_unregistered_request_returns_no_match_and_no_action_taken():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/match",
        json=_match_request("Archive the old camera footage."),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["capability_matched"] is False
    assert result["action_taken"] is False
    assert result["capability"] is None
    assert result["reason_codes"] == ["no_registered_capability"]


def test_endpoint_name_like_request_does_not_infer_capability():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/match",
        json=_match_request("office_lights_on"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["capability_matched"] is False
    assert result["action_taken"] is False
    assert result["capability"] is None
    assert result["reason_codes"] == ["raw_capability_name_ignored"]


def test_runtime_world_state_request_matches_exact_capability():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/match",
        json=_match_request(
            "Please read runtime world state for this repository.",
            surface="dev",
            persona="technical_architect",
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["capability_matched"] is True
    assert result["action_taken"] is False
    assert result["reason_codes"] == ["matched"]
    capability = result["capability"]
    assert capability["capability_id"] == "runtime.world_state.read"
    assert capability["domain"] == "software_architecture"
    assert capability["operation_kind"] == "read_only"
    assert capability["risk_level"] == "low_read_only"
    assert capability["allowed_surfaces"] == ["dev", "vscode"]
    assert capability["allowed_personas"] == ["technical_architect"]
    assert capability["requires_confirmation"] is False
    assert capability["dry_run_supported"] is True
    assert capability["verification_supported"] is True
    assert capability["audit_required"] is False


def test_raw_runtime_world_state_capability_name_does_not_infer_match():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/match",
        json=_match_request(
            "runtime.world_state.read",
            surface="dev",
            persona="technical_architect",
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["capability_matched"] is False
    assert result["action_taken"] is False
    assert result["capability"] is None
    assert result["reason_codes"] == ["raw_capability_name_ignored"]


def test_discovery_summary_includes_allowed_and_blocked_examples():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/discover",
        json=_discovery_request(),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["registry_available"] is True
    assert result["action_taken"] is False
    allowed_ids = {item["capability_id"] for item in result["allowed_examples"]}
    blocked_ids = {item["capability_id"] for item in result["blocked_examples"]}
    assert "office_lights_on" in allowed_ids
    assert "external_purchase" in blocked_ids


def test_runtime_world_state_discovery_is_allowed_for_matching_context():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/discover",
        json=_discovery_request(surface="dev", persona="technical_architect"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["action_taken"] is False
    allowed = {
        item["capability_id"]: item for item in result["allowed_examples"]
    }
    assert "runtime.world_state.read" in allowed
    assert allowed["runtime.world_state.read"]["operation_kind"] == "read_only"
    assert allowed["runtime.world_state.read"]["risk_level"] == "low_read_only"
    assert allowed["runtime.world_state.read"]["reason_codes"] == ["matched"]


def test_surface_filtering_changes_capability_eligibility():
    client = TestClient(app)

    allowed = client.post(
        "/v1/capabilities/match",
        json=_match_request("restart Jellyfin", surface="desktop"),
    ).json()["result"]
    blocked = client.post(
        "/v1/capabilities/match",
        json=_match_request("restart Jellyfin", surface="voice_private"),
    ).json()["result"]

    assert allowed["capability_matched"] is True
    assert blocked["capability_matched"] is False
    assert blocked["reason_codes"] == ["surface_not_allowed"]
    assert blocked["capability"]["capability_id"] == "jellyfin_restart"


def test_persona_filtering_changes_capability_eligibility():
    client = TestClient(app)

    allowed = client.post(
        "/v1/capabilities/match",
        json=_match_request("restart Jellyfin", persona="technical_architect"),
    ).json()["result"]
    blocked = client.post(
        "/v1/capabilities/match",
        json=_match_request("restart Jellyfin", persona="default_assistant"),
    ).json()["result"]

    assert allowed["capability_matched"] is True
    assert blocked["capability_matched"] is False
    assert blocked["reason_codes"] == ["persona_not_allowed"]
    assert blocked["capability"]["capability_id"] == "jellyfin_restart"


@pytest.mark.parametrize(
    ("surface", "persona", "reason_code"),
    [
        ("desktop", "technical_architect", "surface_not_allowed"),
        ("dev", "home_operator", "persona_not_allowed"),
    ],
)
def test_runtime_world_state_context_filtering_applies(
    surface: str,
    persona: str,
    reason_code: str,
):
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/match",
        json=_match_request(
            "show runtime world state",
            surface=surface,
            persona=persona,
        ),
    ).json()["result"]

    assert result["capability_matched"] is False
    assert result["action_taken"] is False
    assert result["reason_codes"] == [reason_code]
    assert result["capability"]["capability_id"] == "runtime.world_state.read"


def test_match_returns_metadata_for_later_authority_decisions():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/match",
        json=_match_request("Run a service health check.", persona="technical_architect"),
    )

    capability = response.json()["result"]["capability"]
    assert set(capability) >= {
        "capability_id",
        "display_name",
        "domain",
        "description",
        "operation_kind",
        "risk_level",
        "requires_confirmation",
        "allowed_surfaces",
        "allowed_personas",
        "reversible",
        "dry_run_supported",
        "verification_supported",
        "audit_required",
    }
    assert capability["operation_kind"] == "read_only"


def test_registry_presence_does_not_authorize_or_execute():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/match",
        json=_match_request("Turn on the office lights."),
    ).json()["result"]

    assert result["capability_matched"] is True
    assert result["action_taken"] is False
    assert "allowed" not in result
    assert "decision_code" not in result


def test_registry_unavailable_fallback_reports_no_action_taken_or_match():
    client = TestClient(app)
    configure_capability_registry_for_tests(available=False)

    response = client.post(
        "/v1/capabilities/match",
        json=_match_request("Turn on the office lights."),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["capability_matched"] is False
    assert result["action_taken"] is False
    assert result["capability"] is None
    assert result["reason_codes"] == ["registry_unavailable"]


def test_read_only_authority_does_not_require_confirmation():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/authority",
        json=_authority_request(
            "service_health_check",
            surface="dev",
            persona="technical_architect",
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["capability_id"] == "service_health_check"
    assert result["risk_level"] == "read_only"
    assert result["authority_level"] == "answer_only"
    assert result["requires_confirmation"] is False
    assert result["allowed"] is True
    assert result["action_taken"] is False


def test_runtime_world_state_authority_is_answer_only_without_confirmation():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/authority",
        json=_authority_request(
            "runtime.world_state.read",
            surface="dev",
            persona="technical_architect",
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["capability_id"] == "runtime.world_state.read"
    assert result["risk_level"] == "read_only"
    assert result["authority_level"] == "answer_only"
    assert result["requires_confirmation"] is False
    assert result["allowed"] is True
    assert result["action_taken"] is False


def test_low_reversible_authority_can_execute_when_context_permits():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/authority",
        json=_authority_request("office_lights_on"),
    ).json()["result"]

    assert result["risk_level"] == "low_reversible"
    assert result["authority_level"] == "execute_low_risk"
    assert result["requires_confirmation"] is False
    assert result["allowed"] is True
    assert result["action_taken"] is False


def test_prepare_authority_returns_prepare_only():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/authority",
        json=_authority_request("draft_notification"),
    ).json()["result"]

    assert result["authority_level"] == "prepare_only"
    assert result["requires_confirmation"] is False
    assert result["allowed"] is True
    assert result["action_taken"] is False


def test_medium_authority_requires_confirmation_without_direct_execution():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/authority",
        json=_authority_request(
            "jellyfin_restart",
            surface="dev",
            persona="technical_architect",
        ),
    ).json()["result"]

    assert result["risk_level"] == "medium_requires_confirmation"
    assert result["authority_level"] == "execute_after_confirmation"
    assert result["requires_confirmation"] is True
    assert result["allowed"] is False
    assert result["action_taken"] is False


def test_high_consequence_input_requires_confirmation():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/authority",
        json=_authority_request(
            "office_lights_on",
            consequence_flags={"security": True},
        ),
    ).json()["result"]

    assert result["risk_level"] == "high_requires_confirmation"
    assert result["authority_level"] == "execute_after_confirmation"
    assert result["requires_confirmation"] is True
    assert result["allowed"] is False
    assert result["action_taken"] is False


def test_blocked_external_authority_stays_blocked():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/authority",
        json=_authority_request("external_purchase"),
    ).json()["result"]

    assert result["risk_level"] == "blocked"
    assert result["authority_level"] == "blocked"
    assert result["requires_confirmation"] is True
    assert result["allowed"] is False
    assert result["action_taken"] is False


@pytest.mark.parametrize("target_state", ["ambiguous", "missing"])
def test_unresolved_target_prevents_execution(target_state: str):
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/authority",
        json=_authority_request(
            "office_lights_on",
            target_resolution_state=target_state,
        ),
    ).json()["result"]

    assert result["risk_level"] == "blocked"
    assert result["authority_level"] == "blocked"
    assert result["allowed"] is False
    assert result["action_taken"] is False


@pytest.mark.parametrize("freshness", ["stale", "unknown"])
def test_uncertain_world_state_increases_conservatism(freshness: str):
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/authority",
        json=_authority_request(
            "office_lights_on",
            world_state_freshness=freshness,
        ),
    ).json()["result"]

    assert result["risk_level"] == "medium_requires_confirmation"
    assert result["authority_level"] == "execute_after_confirmation"
    assert result["requires_confirmation"] is True
    assert result["allowed"] is False
    assert result["action_taken"] is False


@pytest.mark.parametrize(
    ("surface", "persona"),
    [("voice_private", "technical_architect"), ("dev", "default_assistant")],
)
def test_context_mismatch_denies_execution_authority(surface: str, persona: str):
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/authority",
        json=_authority_request(
            "jellyfin_restart",
            surface=surface,
            persona=persona,
        ),
    ).json()["result"]

    assert result["authority_level"] == "blocked"
    assert result["allowed"] is False
    assert result["action_taken"] is False


def test_context_permission_does_not_bypass_elevated_risk():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/authority",
        json=_authority_request(
            "jellyfin_restart",
            surface="dev",
            persona="technical_architect",
        ),
    ).json()["result"]

    assert result["risk_level"] == "medium_requires_confirmation"
    assert result["authority_level"] == "execute_after_confirmation"
    assert result["requires_confirmation"] is True
    assert result["allowed"] is False


def test_vague_authorization_does_not_permit_execution():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/authority",
        json=_authority_request(
            "office_lights_on",
            user_authorization_signal="vague",
        ),
    ).json()["result"]

    assert result["authority_level"] == "suggest_only"
    assert result["allowed"] is False
    assert result["action_taken"] is False
    assert "explicit_authorization_absent" in result["reason_summary"]


def test_vent_like_interaction_suppresses_execution():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/authority",
        json=_authority_request(
            "office_lights_on",
            interaction_governance_kind="vent_or_expression",
        ),
    ).json()["result"]

    assert result["authority_level"] == "suggest_only"
    assert result["allowed"] is False
    assert result["action_taken"] is False
    assert "interaction_suppresses_execution" in result["reason_summary"]


def test_authority_endpoint_response_matches_contract_without_execution():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/authority",
        json=_authority_request("office_lights_on"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert set(result) == {
        "capability_id",
        "risk_level",
        "authority_level",
        "requires_confirmation",
        "allowed",
        "reason_summary",
        "action_taken",
    }
    assert result["action_taken"] is False


def test_preview_flow_returns_dry_run_effects_without_action():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(
            "office_lights_on",
            flow_intent="preview_requested",
            target_label="office lights",
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["dry_run_required"] is True
    assert result["dry_run_supported"] is True
    assert result["dry_run_effects"][0]["display_name"] == "Turn on office lights"
    assert result["dry_run_effects"][0]["target_label"] == "office lights"
    assert "Would evaluate Turn on office lights" in result["dry_run_effects"][0][
        "intended_effect"
    ]
    assert result["execution_allowed"] is False
    assert result["action_taken"] is False


def test_runtime_world_state_preview_flow_returns_dry_run_without_action():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(
            "runtime.world_state.read",
            surface="dev",
            persona="technical_architect",
            flow_intent="preview_requested",
            target_label="repository context",
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["capability_id"] == "runtime.world_state.read"
    assert result["dry_run_required"] is True
    assert result["dry_run_supported"] is True
    assert result["dry_run_effects"][0]["display_name"] == "Read runtime world state"
    assert result["dry_run_effects"][0]["target_label"] == "repository context"
    assert result["execution_allowed"] is False
    assert result["action_taken"] is False


def test_runtime_world_state_allowed_flow_signals_verification_without_action():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(
            "runtime.world_state.read",
            surface="dev",
            persona="technical_architect",
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["capability_id"] == "runtime.world_state.read"
    assert result["dry_run_supported"] is True
    assert result["execution_allowed"] is True
    assert result["verification_supported"] is True
    assert result["verification_required"] is True
    assert result["verification_method"] == "capability_verification"
    assert result["action_taken"] is False


def test_preview_intent_requires_dry_run():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/flow",
        json=_flow_request("office_lights_on", flow_intent="preview_requested"),
    ).json()["result"]

    assert result["dry_run_required"] is True
    assert "preview_requested" in result["reason_summary"]
    assert "dry_run_required" in result["reason_summary"]
    assert result["action_taken"] is False


@pytest.mark.parametrize("target_state", ["ambiguous", "missing"])
def test_unresolved_target_flow_requires_dry_run_and_blocks_execution(target_state: str):
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(
            "office_lights_on",
            target_resolution_state=target_state,
        ),
    ).json()["result"]

    assert result["dry_run_required"] is True
    assert result["confirmation_required"] is False
    assert result["execution_allowed"] is False
    assert result["action_taken"] is False
    assert "target_not_resolved" in result["reason_summary"]


@pytest.mark.parametrize(
    ("capability_id", "request_overrides"),
    [
        (
            "jellyfin_restart",
            {"surface": "dev", "persona": "technical_architect"},
        ),
        ("office_lights_on", {"consequence_flags": {"security": True}}),
    ],
)
def test_elevated_risk_flow_requires_dry_run(
    capability_id: str,
    request_overrides: dict[str, object],
):
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(capability_id, **request_overrides),
    ).json()["result"]

    assert result["dry_run_required"] is True
    assert result["confirmation_required"] is True
    assert result["execution_allowed"] is False
    assert result["action_taken"] is False
    assert "medium_or_high_risk" in result["reason_summary"]


def test_multiple_system_flow_requires_dry_run_before_execution():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/flow",
        json=_flow_request("office_lights_on", affects_multiple_systems=True),
    ).json()["result"]

    assert result["dry_run_required"] is True
    assert result["execution_allowed"] is False
    assert "multiple_systems" in result["reason_summary"]
    assert "dry_run_pending" in result["reason_summary"]
    assert result["action_taken"] is False


@pytest.mark.parametrize(
    ("capability_id", "request_overrides"),
    [
        ("office_lights_on", {"consequence_flags": {"destructive": True}}),
        (
            "jellyfin_restart",
            {"surface": "dev", "persona": "technical_architect"},
        ),
    ],
)
def test_destructive_or_difficult_to_reverse_flow_requires_dry_run(
    capability_id: str,
    request_overrides: dict[str, object],
):
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(capability_id, **request_overrides),
    ).json()["result"]

    assert result["dry_run_required"] is True
    assert "difficult_to_reverse" in result["reason_summary"]
    assert result["execution_allowed"] is False
    assert result["action_taken"] is False


def test_scoped_confirmation_text_names_capability_target_and_reason():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(
            "jellyfin_restart",
            surface="dev",
            persona="technical_architect",
            target_label="media server",
        ),
    ).json()["result"]

    assert result["confirmation_required"] is True
    assert result["confirmation_text"] is not None
    assert "Restart Jellyfin" in result["confirmation_text"]
    assert "media server" in result["confirmation_text"]
    assert "difficult to reverse" in result["confirmation_text"]
    assert result["confirmation_text"] != "Are you sure?"
    assert result["action_taken"] is False


def test_elevated_risk_flow_requires_scoped_confirmation():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(
            "jellyfin_restart",
            surface="dev",
            persona="technical_architect",
            target_label="media server",
        ),
    ).json()["result"]

    assert result["confirmation_required"] is True
    assert result["confirmation_text"].startswith("Confirm Restart Jellyfin")
    assert result["execution_allowed"] is False
    assert result["action_taken"] is False


def test_required_confirmation_blocks_execution_until_received():
    client = TestClient(app)

    pending = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(
            "jellyfin_restart",
            surface="dev",
            persona="technical_architect",
        ),
    ).json()["result"]
    confirmed = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(
            "jellyfin_restart",
            surface="dev",
            persona="technical_architect",
            flow_intent="confirmation_received",
        ),
    ).json()["result"]

    assert pending["confirmation_required"] is True
    assert pending["execution_allowed"] is False
    assert confirmed["confirmation_required"] is True
    assert confirmed["execution_allowed"] is True
    assert pending["action_taken"] is False
    assert confirmed["action_taken"] is False


def test_confirmation_cancellation_keeps_execution_blocked():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(
            "jellyfin_restart",
            surface="dev",
            persona="technical_architect",
            flow_intent="confirmation_cancelled",
        ),
    ).json()["result"]

    assert result["execution_allowed"] is False
    assert result["action_taken"] is False
    assert "confirmation_cancelled" in result["reason_summary"]


def test_confirmation_expiry_keeps_execution_blocked():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(
            "jellyfin_restart",
            surface="dev",
            persona="technical_architect",
            flow_intent="confirmation_expired",
        ),
    ).json()["result"]

    assert result["execution_allowed"] is False
    assert result["action_taken"] is False
    assert "confirmation_expired" in result["reason_summary"]


def test_ambiguous_target_flow_blocks_execution_pending_clarification():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(
            "jellyfin_restart",
            surface="dev",
            persona="technical_architect",
            target_resolution_state="ambiguous",
            target_label="media service",
        ),
    ).json()["result"]

    assert result["dry_run_required"] is True
    assert result["confirmation_required"] is False
    assert result["execution_allowed"] is False
    assert result["action_taken"] is False
    assert "target_not_resolved" in result["reason_summary"]


def test_dry_run_flow_is_not_execution():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/flow",
        json=_flow_request("office_lights_on", flow_intent="preview_requested"),
    ).json()["result"]

    assert result["dry_run_required"] is True
    assert result["dry_run_effects"]
    assert result["execution_allowed"] is False
    assert result["action_taken"] is False
    assert "execution_allowed_by_policy" not in result["reason_summary"]
    assert not any("executed" in reason for reason in result["reason_summary"])


def test_verification_requirement_returned_for_verifiable_allowed_flow():
    client = TestClient(app)

    result = client.post(
        "/v1/capabilities/flow",
        json=_flow_request(
            "jellyfin_restart",
            surface="dev",
            persona="technical_architect",
            flow_intent="confirmation_received",
        ),
    ).json()["result"]

    assert result["execution_allowed"] is True
    assert result["verification_supported"] is True
    assert result["verification_required"] is True
    assert result["verification_method"] == "capability_verification"
    assert result["action_taken"] is False


def test_flow_endpoint_response_matches_contract_without_execution():
    client = TestClient(app)

    response = client.post(
        "/v1/capabilities/flow",
        json=_flow_request("office_lights_on"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert set(result) == {
        "capability_id",
        "dry_run_required",
        "dry_run_supported",
        "dry_run_effects",
        "confirmation_required",
        "confirmation_text",
        "execution_allowed",
        "verification_required",
        "verification_supported",
        "verification_method",
        "reason_summary",
        "action_taken",
    }
    assert result["capability_id"] == "office_lights_on"
    assert result["action_taken"] is False


def test_action_summary_returns_complete_contract_and_matching_runtime_event():
    client = TestClient(app)
    request = _action_summary_request(client)

    response = client.post("/v1/capabilities/action-summary", json=request)

    assert response.status_code == 200
    payload = response.json()
    result = payload["result"]
    assert set(result) == {
        "action_id",
        "capability_id",
        "requested_by",
        "surface_type",
        "active_persona_id",
        "risk_level",
        "authority_level",
        "confirmation_status",
        "execution_status",
        "verification_status",
        "degradation_reason",
        "policy_reason_codes",
        "execution_reason_code",
        "verification_reason_code",
        "user_visible_summary",
    }
    assert result == {
        "action_id": result["action_id"],
        "capability_id": "runtime.world_state.read",
        "requested_by": "conversation_participant",
        "surface_type": "dev",
        "active_persona_id": "technical_architect",
        "risk_level": "read_only",
        "authority_level": "answer_only",
        "confirmation_status": "not_required",
        "execution_status": "executed",
        "verification_status": "passed",
        "degradation_reason": None,
        "policy_reason_codes": [
            "registered_capability",
            "execution_allowed_by_policy",
        ],
        "execution_reason_code": "adapter_completed",
        "verification_reason_code": "result_check_passed",
        "user_visible_summary": (
            "Action runtime.world_state.read was executed and verification passed."
        ),
    }
    assert result["action_id"].startswith("act_")
    assert request["request_id"] not in result["action_id"]

    diagnostics = client.get(
        f"/v1/runtime/sessions/{request['runtime_session_id']}"
    ).json()
    event = next(
        event
        for event in diagnostics["events"]
        if event["event_type"] == "action_summary_recorded"
    )
    event_payload = event["event_payload_json"]
    for field, value in result.items():
        assert event_payload[field] == value
    assert event_payload["request_id"] == request["request_id"]
    assert event_payload["runtime_session_id"] == request["runtime_session_id"]


def test_action_summary_identity_is_stable_and_changes_with_action_identity():
    client = TestClient(app)
    original = _action_summary_request(client)

    first = client.post("/v1/capabilities/action-summary", json=original).json()["result"]
    repeated = client.post("/v1/capabilities/action-summary", json=original).json()["result"]
    assert first["action_id"] == repeated["action_id"]

    changed_requests = [
        {**original, "request_id": "action-summary-other"},
        {**original, "capability_id": "service_health_check"},
        _action_summary_request(client, owner_id="owner-other"),
        _action_summary_request(client, conversation_id="conv-other"),
    ]
    turn = client.post(
        "/v1/runtime/turns/start",
        json={
            **_base(),
            "request_id": "action-summary-turn",
        },
    ).json()["runtime_turn"]
    changed_requests.append({**original, "runtime_turn_id": turn["runtime_turn_id"]})

    changed_ids = {
        client.post("/v1/capabilities/action-summary", json=request).json()["result"][
            "action_id"
        ]
        for request in changed_requests
    }
    assert first["action_id"] not in changed_ids
    assert len(changed_ids) == len(changed_requests)


def test_blocked_action_summary_reports_no_action_and_rejects_passed_verification():
    client = TestClient(app)
    request = _action_summary_request(
        client,
        risk_level="blocked",
        authority_level="blocked",
        confirmation_status="required_pending",
        policy_reason_codes=["surface_not_allowed"],
        execution_status="blocked_by_policy",
        execution_reason_code=None,
        verification_status="unknown",
        verification_reason_code=None,
    )

    response = client.post("/v1/capabilities/action-summary", json=request)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["execution_status"] == "blocked_by_policy"
    assert result["degradation_reason"] == "surface_not_allowed"
    assert "No action was taken" in result["user_visible_summary"]
    assert "success" not in result["user_visible_summary"].lower()

    impossible = client.post(
        "/v1/capabilities/action-summary",
        json={**request, "verification_status": "passed"},
    )
    assert impossible.status_code == 422


def test_not_attempted_action_summary_reports_no_action():
    client = TestClient(app)
    request = _action_summary_request(
        client,
        execution_status="not_attempted",
        execution_reason_code=None,
        verification_status="not_supported",
        verification_reason_code=None,
    )
    result = client.post(
        "/v1/capabilities/action-summary",
        json=request,
    ).json()["result"]

    assert result["execution_status"] == "not_attempted"
    assert result["verification_status"] == "not_supported"
    assert "No action was taken" in result["user_visible_summary"]
    impossible = client.post(
        "/v1/capabilities/action-summary",
        json={**request, "verification_status": "failed"},
    )
    assert impossible.status_code == 422


def test_action_summary_rejects_execution_that_policy_does_not_permit():
    client = TestClient(app)
    response = client.post(
        "/v1/capabilities/action-summary",
        json=_action_summary_request(
            client,
            risk_level="blocked",
            authority_level="blocked",
            confirmation_status="accepted",
            execution_status="executed",
        ),
    )

    assert response.status_code == 422


def test_failed_action_summary_is_degraded_without_success_claim():
    client = TestClient(app)
    response = client.post(
        "/v1/capabilities/action-summary",
        json=_action_summary_request(
            client,
            execution_status="failed",
            execution_reason_code="integration_unavailable",
            verification_status="unknown",
            verification_reason_code=None,
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["execution_status"] == "failed"
    assert result["degradation_reason"] == "integration_unavailable"
    assert "failed" in result["user_visible_summary"]
    assert "success" not in result["user_visible_summary"].lower()


def test_partial_action_summary_remains_explicitly_degraded():
    client = TestClient(app)
    result = client.post(
        "/v1/capabilities/action-summary",
        json=_action_summary_request(
            client,
            execution_status="partially_executed",
            execution_reason_code=None,
            verification_status="passed",
        ),
    ).json()["result"]

    assert result["execution_status"] == "partially_executed"
    assert result["degradation_reason"] == "partial_execution"
    assert "partially completed" in result["user_visible_summary"]
    assert "degraded" in result["user_visible_summary"]


def test_unknown_action_summary_preserves_uncertainty_without_retry_or_success():
    client = TestClient(app)
    result = client.post(
        "/v1/capabilities/action-summary",
        json=_action_summary_request(
            client,
            execution_status="unknown",
            execution_reason_code=None,
            verification_status="unknown",
            verification_reason_code=None,
        ),
    ).json()["result"]

    assert result["execution_status"] == "unknown"
    assert result["degradation_reason"] == "execution_state_unknown"
    assert "could not be confirmed" in result["user_visible_summary"]
    assert "No success is claimed" in result["user_visible_summary"]
    assert "retry" not in result["user_visible_summary"].lower()


@pytest.mark.parametrize(
    ("verification_status", "expected_text", "claims_verified_success"),
    [
        ("passed", "verification passed", True),
        ("failed", "verification failed", False),
        ("unknown", "verification could not be confirmed", False),
        ("not_required", "Verification was not required", False),
        ("not_supported", "Verification is not supported", False),
    ],
)
def test_executed_action_summary_reports_verification_truthfully(
    verification_status: str,
    expected_text: str,
    claims_verified_success: bool,
):
    client = TestClient(app)
    result = client.post(
        "/v1/capabilities/action-summary",
        json=_action_summary_request(
            client,
            verification_status=verification_status,
            verification_reason_code=(
                "result_check_failed" if verification_status == "failed" else None
            ),
        ),
    ).json()["result"]

    assert expected_text in result["user_visible_summary"]
    assert ("verification passed" in result["user_visible_summary"]) is (
        claims_verified_success
    )
    if verification_status == "failed":
        assert result["degradation_reason"] == "result_check_failed"
    if verification_status == "unknown":
        assert result["degradation_reason"] == "verification_unknown"


def test_cancelled_action_summary_is_distinct_from_policy_blocking():
    client = TestClient(app)
    result = client.post(
        "/v1/capabilities/action-summary",
        json=_action_summary_request(
            client,
            confirmation_status="cancelled",
            execution_status="cancelled_by_user",
            execution_reason_code=None,
            verification_status="not_required",
            verification_reason_code=None,
        ),
    ).json()["result"]

    assert result["execution_status"] == "cancelled_by_user"
    assert result["execution_status"] != "blocked_by_policy"
    assert result["degradation_reason"] is None
    assert "cancelled" in result["user_visible_summary"]
    assert "No action was taken" in result["user_visible_summary"]


@pytest.mark.parametrize(
    "private_field",
    [
        "raw_prompt",
        "raw_output",
        "metadata",
        "credentials",
        "endpoint_url",
        "exception_details",
        "tool_arguments",
    ],
)
def test_action_summary_request_rejects_private_or_arbitrary_fields(private_field: str):
    client = TestClient(app)
    request = _action_summary_request(client)
    request[private_field] = {"secret": "private-value"}

    response = client.post("/v1/capabilities/action-summary", json=request)

    assert response.status_code == 422


def test_action_summary_request_rejects_url_shaped_identifiers():
    client = TestClient(app)
    request = _action_summary_request(client)

    response = client.post(
        "/v1/capabilities/action-summary",
        json={**request, "capability_id": "https://private.example/action"},
    )

    assert response.status_code == 422


def test_action_summary_event_contains_only_bounded_normalized_fields():
    client = TestClient(app)
    request = _action_summary_request(
        client,
        execution_status="executed",
        verification_status="failed",
        verification_reason_code="result_check_failed",
    )
    response = client.post("/v1/capabilities/action-summary", json=request)
    result = response.json()["result"]

    diagnostics = client.get(
        f"/v1/runtime/sessions/{request['runtime_session_id']}"
    ).json()
    event_payload = next(
        event["event_payload_json"]
        for event in diagnostics["events"]
        if event["event_type"] == "action_summary_recorded"
    )

    assert set(event_payload) == {
        "request_id",
        "owner_id",
        "conversation_id",
        "runtime_session_id",
        "runtime_turn_id",
        *result.keys(),
    }
    assert all(event_payload[field] == value for field, value in result.items())
    assert "private-value" not in str(event_payload)


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
            capability_id="integration.generic.read.test",
            capability_domain="personal_support",
            supported_surfaces=["web"],
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["allowed"] is False
    assert "capability_domain_denied" in result["reason_codes"]
    assert "surface_unsupported" in result["reason_codes"]


def test_registered_world_state_authorization_uses_canonical_registry_metadata():
    client = TestClient(app)
    turn = _start_turn(client)

    response = client.post(
        "/v1/capabilities/authorize",
        json=_authorization_request(
            turn,
            authorization_stage="selection",
            argument_digest="args_registered_world_state",
            supported_surfaces=["vscode", "dev"],
            request_id="registered-world-state-authorization",
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["allowed"] is True
    assert result["decision_code"] == "allowed"
    assert result["confirmation_state"] == "not_required"
    assert result["challenge_ref"] is None
    assert result["challenge_expires_at"] is None

    runtime_session_id = turn["runtime_session"]["runtime_session_id"]
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}").json()
    event_payload = next(
        event["event_payload_json"]
        for event in diagnostics["events"]
        if event["event_type"] == "capability_authorization_evaluated"
        and event["event_payload_json"]["request_id"]
        == "registered-world-state-authorization"
    )
    assert event_payload["operation_class"] == "read"
    assert event_payload["challenge_expires_at"] is None


@pytest.mark.parametrize(
    (
        "request_overrides",
        "surface",
        "resolved_persona",
        "expected_reason",
    ),
    [
        (
            {"capability_domain": "software_architecture"},
            "dev",
            "technical_architect",
            "registered_capability_domain_mismatch",
        ),
        (
            {"operation_class": "read"},
            "dev",
            "technical_architect",
            "registered_operation_class_mismatch",
        ),
        (
            {"supported_surfaces": ["dev"]},
            "dev",
            "technical_architect",
            "registered_supported_surfaces_mismatch",
        ),
        (
            {"supported_surfaces": ["desktop", "dev", "web"]},
            "dev",
            "technical_architect",
            "registered_supported_surfaces_mismatch",
        ),
        (
            {"supported_surfaces": ["desktop", "dev", "dev"]},
            "dev",
            "technical_architect",
            "registered_supported_surfaces_mismatch",
        ),
        (
            {"surface": "web"},
            "web",
            "technical_architect",
            "registered_surface_not_allowed",
        ),
        (
            {"active_persona_id": "general_assistant"},
            "dev",
            "general_assistant",
            "registered_persona_not_allowed",
        ),
    ],
)
def test_registered_metadata_and_eligibility_mismatch_fail_before_context_selection(
    monkeypatch,
    request_overrides: dict[str, object],
    surface: str,
    resolved_persona: str,
    expected_reason: str,
):
    client = TestClient(app)
    turn = _start_turn(client, surface=surface)
    monkeypatch.setattr(
        capability_authorization_service,
        "resolve_runtime_identity",
        lambda _: SimpleNamespace(
            runtime_identity=SimpleNamespace(
                active_persona_id=resolved_persona,
                capability_domain="software_architecture",
                advisory_tool_permission_summary=["inspect_repository"],
            )
        ),
    )

    def unexpected_context_selection(*args, **kwargs):
        raise AssertionError("registered mismatch reached context selection")

    monkeypatch.setattr(
        capability_authorization_service,
        "select_relationships",
        unexpected_context_selection,
    )
    monkeypatch.setattr(
        capability_authorization_service,
        "resolve_world_state",
        unexpected_context_selection,
    )
    request = _jellyfin_authorization_request(
        turn,
        request_id=f"registered-mismatch-{expected_reason}",
        authorization_stage="selection",
        argument_digest="args_registered_mismatch",
        relationship_requirements=[{"relationship_scope": "project_context"}],
        selected_relationship_ids=["rel-untrusted"],
        world_state_requirements=[{"domain": "active_repository"}],
        selected_world_state_claim_ids=["wsc-untrusted"],
        **request_overrides,
    )

    response = client.post("/v1/capabilities/authorize", json=request)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["allowed"] is False
    assert result["decision_code"] == "authorization_denied"
    assert result["reason_codes"] == [expected_reason]
    assert result["challenge_ref"] is None
    assert result["challenge_expires_at"] is None
    assert result["revalidation_required"] is False
    assert result["revalidation_selector"] is None
    assert result["relationship_ids_used"] == []
    assert result["world_state_claim_ids_used"] == []
    assert _challenge_row_count() == 0
    assert _issued_challenge_event_count() == 0

    runtime_session_id = turn["runtime_session"]["runtime_session_id"]
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}").json()
    event_payload = next(
        event["event_payload_json"]
        for event in diagnostics["events"]
        if event["event_type"] == "capability_authorization_evaluated"
        and event["event_payload_json"]["request_id"] == request["request_id"]
    )
    assert event_payload["operation_class"] == "high_impact"
    assert event_payload["challenge_expires_at"] is None


def test_canonical_jellyfin_selection_issues_one_bounded_challenge():
    client = TestClient(app)
    registry_record = client.post(
        "/v1/capabilities/match",
        json=_match_request(
            "restart Jellyfin",
            surface="dev",
            persona="technical_architect",
        ),
    ).json()["result"]["capability"]
    assert registry_record["capability_id"] == "jellyfin_restart"
    assert registry_record["domain"] == "media_operations"
    assert registry_record["operation_kind"] == "restart"
    assert registry_record["allowed_surfaces"] == ["desktop", "dev"]
    assert registry_record["allowed_personas"] == [
        "home_operator",
        "technical_architect",
    ]
    assert registry_record["requires_confirmation"] is True

    turn = _start_turn(client)
    request = _jellyfin_authorization_request(
        turn,
        request_id="jellyfin-selection",
        authorization_stage="selection",
        argument_digest="args_jellyfin_restart",
    )

    response = client.post("/v1/capabilities/authorize", json=request)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["allowed"] is False
    assert result["decision_code"] == "confirmation_required"
    assert result["reason_codes"] == ["confirmation_required"]
    assert result["confirmation_state"] == "issued"
    assert result["challenge_ref"]
    assert result["challenge_expires_at"]
    assert _challenge_row_count() == 1
    assert _issued_challenge_event_count() == 1

    row = _challenge_record(result["challenge_ref"])
    assert row["capability_id"] == "jellyfin_restart"
    assert row["operation_class"] == "high_impact"
    assert row["argument_digest"] == "args_jellyfin_restart"
    assert row["expires_at"] == result["challenge_expires_at"]
    assert row["consumed_at"] is None

    runtime_session_id = turn["runtime_session"]["runtime_session_id"]
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}").json()
    authorization_event = next(
        event["event_payload_json"]
        for event in diagnostics["events"]
        if event["event_type"] == "capability_authorization_evaluated"
        and event["event_payload_json"]["request_id"] == "jellyfin-selection"
    )
    assert authorization_event["operation_class"] == "high_impact"
    assert authorization_event["challenge_ref"] == result["challenge_ref"]
    assert authorization_event["challenge_expires_at"] == row["expires_at"]
    assert not any(
        event["event_type"] == "action_summary_recorded"
        for event in diagnostics["events"]
    )


def test_distinct_current_turn_resumes_exact_jellyfin_challenge_without_replacement():
    client = TestClient(app)
    origin_turn = _start_turn(client)
    first = client.post(
        "/v1/capabilities/authorize",
        json=_jellyfin_authorization_request(
            origin_turn,
            request_id="jellyfin-first-selection",
            authorization_stage="selection",
            argument_digest="args_jellyfin_continuation",
        ),
    ).json()["result"]
    stored = _challenge_record(first["challenge_ref"])
    continuation_turn = _start_turn(client)

    continuation = client.post(
        "/v1/capabilities/authorize",
        json=_jellyfin_authorization_request(
            continuation_turn,
            request_id="jellyfin-continuation-selection",
            authorization_stage="selection",
            argument_digest="args_jellyfin_continuation",
            confirmation_challenge_ref=first["challenge_ref"],
        ),
    ).json()["result"]

    assert continuation["allowed"] is False
    assert continuation["decision_code"] == "confirmation_required"
    assert continuation["reason_codes"] == ["confirmation_required"]
    assert continuation["confirmation_state"] == "issued"
    assert continuation["challenge_ref"] == first["challenge_ref"]
    assert continuation["challenge_expires_at"] == first["challenge_expires_at"]
    assert continuation["challenge_expires_at"] == stored["expires_at"]
    assert _challenge_row_count() == 1
    assert _issued_challenge_event_count() == 1
    assert _challenge_record(first["challenge_ref"])["status"] == "issued"

    runtime_session_id = origin_turn["runtime_session"]["runtime_session_id"]
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}").json()
    challenge_events = [
        event["event_payload_json"]
        for event in diagnostics["events"]
        if event["event_type"] == "confirmation_challenge_evaluated"
    ]
    assert [event["confirmation_state"] for event in challenge_events] == ["issued"]
    continuation_event = next(
        event["event_payload_json"]
        for event in diagnostics["events"]
        if event["event_type"] == "capability_authorization_evaluated"
        and event["event_payload_json"]["request_id"]
        == "jellyfin-continuation-selection"
    )
    assert continuation_event["challenge_ref"] == first["challenge_ref"]
    assert continuation_event["challenge_expires_at"] == stored["expires_at"]
    assert not any(
        event["event_type"] == "action_summary_recorded"
        for event in diagnostics["events"]
    )


def test_relationship_required_exposure_allows_active_eligible_relationship():
    client = TestClient(app)
    turn = _start_turn(client)
    rel_id = _relationship(client)

    result = _relationship_authorization_result(client, turn, relationship_id=rel_id)

    assert result["allowed"] is True
    assert result["phase"] == "exposure"
    assert result["relationship_ids_used"] == [rel_id]


def test_relationship_required_selection_allows_active_eligible_relationship():
    client = TestClient(app)
    turn = _start_turn(client)
    rel_id = _relationship(client)

    result = _relationship_authorization_result(
        client,
        turn,
        relationship_id=rel_id,
        authorization_phase="selection",
        argument_digest="args_relationship_selection",
    )

    assert result["allowed"] is True
    assert result["phase"] == "selection"
    assert result["relationship_ids_used"] == [rel_id]


def test_relationship_required_dispatch_allows_active_eligible_relationship():
    client = TestClient(app)
    turn = _start_turn(client)
    rel_id = _relationship(client)

    result = _relationship_authorization_result(
        client,
        turn,
        relationship_id=rel_id,
        authorization_phase="dispatch",
        argument_digest="args_relationship_dispatch",
    )

    assert result["allowed"] is True
    assert result["phase"] == "dispatch"
    assert result["relationship_ids_used"] == [rel_id]


def test_relationship_required_missing_context_denies_with_bounded_reason():
    client = TestClient(app)
    turn = _start_turn(client)

    result = _relationship_authorization_result(client, turn)

    assert result["allowed"] is False
    assert result["decision_code"] == "authorization_denied"
    assert result["reason_codes"] == ["relationship_required"]
    assert result["relationship_ids_used"] == []


def test_relationship_required_uses_eligible_selection_when_no_ids_are_provided():
    client = TestClient(app)
    turn = _start_turn(client)
    rel_id = _relationship(client)

    result = _relationship_authorization_result(client, turn)

    assert result["allowed"] is True
    assert result["relationship_ids_used"] == [rel_id]


def test_selected_unrelated_relationship_denies_with_not_authorized_reason():
    client = TestClient(app)
    turn = _start_turn(client)
    rel_id = _relationship(
        client,
        relationship_id="rel-unrelated-type",
        relationship_type="documents",
    )

    result = _relationship_authorization_result(client, turn, relationship_id=rel_id)

    assert result["allowed"] is False
    assert result["decision_code"] == "authorization_denied"
    assert result["reason_codes"] == ["relationship_not_authorized"]
    assert result["relationship_ids_used"] == []


@pytest.mark.parametrize(
    ("case_id", "edge_overrides", "requirement"),
    [
        ("revoked", {"status": "revoked"}, None),
        ("restricted", {"sensitivity_level": "restricted"}, None),
        ("expired", {"valid_until": _iso(-60)}, None),
        ("provisional", {"status": "provisional", "source_type": "tool_output"}, None),
        (
            "low-confidence",
            {"confidence": 0.4, "source_type": "tool_output"},
            None,
        ),
        ("wrong-type", {"relationship_type": "documents"}, None),
        (
            "wrong-scope",
            {"relationship_scope": "operations_context"},
            None,
        ),
        (
            "outside-active-persona-scope",
            {"allowed_persona_scopes_json": ["personal_companion"]},
            None,
        ),
    ],
)
def test_selected_ineligible_relationship_states_do_not_authorize(
    case_id: str,
    edge_overrides: dict[str, object],
    requirement: list[dict[str, str]] | None,
):
    client = TestClient(app)
    turn = _start_turn(client)
    rel_id = _relationship(
        client,
        relationship_id=f"rel-{case_id}",
        **edge_overrides,
    )

    result = _relationship_authorization_result(
        client,
        turn,
        relationship_id=rel_id,
        relationship_requirements=requirement,
    )

    assert result["allowed"] is False
    assert result["decision_code"] == "authorization_denied"
    assert result["reason_codes"] == ["relationship_not_authorized"]
    assert result["relationship_ids_used"] == []


def test_selected_conflicted_relationship_does_not_authorize():
    client = TestClient(app)
    turn = _start_turn(client)
    first_rel_id = _relationship(
        client,
        relationship_id="rel-conflicted-one",
        relationship_type="defaults_to",
    )
    client.post(
        "/v1/relationships/entities/upsert",
        json={
            **_base(),
            "entity": {
                "entity_id": "repo:secondary",
                "entity_type": "repository",
                "canonical_label": "secondary repository",
                "domain": "project_context",
                "sensitivity_level": "medium",
                "source_type": "trusted_config",
                "source_ref": "pytest",
                "status": "active",
            },
        },
    )
    _relationship(
        client,
        relationship_id="rel-conflicted-two",
        object_entity_id="repo:secondary",
        relationship_type="defaults_to",
    )

    result = _relationship_authorization_result(
        client,
        turn,
        relationship_id=first_rel_id,
        relationship_requirements=_relationship_requirement(
            relationship_type="defaults_to"
        ),
    )

    assert result["allowed"] is False
    assert result["reason_codes"] == ["relationship_not_authorized"]
    assert result["relationship_ids_used"] == []


def test_relationship_outside_surface_scope_does_not_authorize():
    client = TestClient(app)
    turn = _start_turn(client, surface="unknown")
    rel_id = _relationship(
        client,
        relationship_id="rel-surface-scope",
        relationship_scope="creative_context",
    )

    result = _relationship_authorization_result(
        client,
        turn,
        relationship_id=rel_id,
        relationship_requirements=_relationship_requirement(
            relationship_scope="creative_context"
        ),
        capability_id="integration.generic.read.test",
        surface="unknown",
        active_persona_id="general_assistant",
        capability_domain="general_assistance",
        supported_surfaces=["unknown"],
    )

    assert result["allowed"] is False
    assert result["reason_codes"] == ["relationship_not_authorized"]
    assert result["relationship_ids_used"] == []


def test_relationship_ids_used_are_bounded_to_eligible_selected_ids():
    client = TestClient(app)
    turn = _start_turn(client)
    eligible_rel_id = _relationship(client, relationship_id="rel-eligible")
    wrong_type_rel_id = _relationship(
        client,
        relationship_id="rel-wrong-type",
        relationship_type="documents",
    )

    result = client.post(
        "/v1/capabilities/authorize",
        json=_authorized_request(
            turn,
            relationship_requirements=_relationship_requirement(),
            selected_relationship_ids=[eligible_rel_id, wrong_type_rel_id],
        ),
    ).json()["result"]

    assert result["allowed"] is True
    assert result["relationship_ids_used"] == [eligible_rel_id]
    assert wrong_type_rel_id not in result["relationship_ids_used"]
    assert "source_refs_json" not in str(result)
    assert "pytest" not in str(result)


def test_relationship_authorization_event_records_bounded_ids_and_reasons():
    client = TestClient(app)
    turn = _start_turn(client)
    rel_id = _relationship(client, relationship_id="rel-event-bounded")

    result = _relationship_authorization_result(
        client,
        turn,
        relationship_id=rel_id,
        request_id="relationship-event-allow",
    )
    assert result["allowed"] is True

    runtime_session_id = turn["runtime_session"]["runtime_session_id"]
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}").json()
    event_payload = [
        event["event_payload_json"]
        for event in diagnostics["events"]
        if event["event_type"] == "capability_authorization_evaluated"
        and event["event_payload_json"]["request_id"] == "relationship-event-allow"
    ][-1]

    assert event_payload["relationship_ids_used"] == [rel_id]
    assert event_payload["reason_codes"] == ["allowed"]
    assert "source_refs_json" not in str(event_payload)
    assert "relationship_evidence" not in str(event_payload)
    assert "pytest" not in str(event_payload)

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
            capability_id="integration.external_write.test",
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
            capability_id="integration.external_write.test",
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


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("missing", "challenge_missing"),
        ("same_turn", "challenge_turn_mismatch"),
        ("not_current", "confirmation_turn_not_current"),
        ("created_before_issue", "confirmation_turn_not_current"),
        ("wrong_owner", "challenge_missing"),
        ("wrong_conversation_session", "challenge_mismatch"),
        ("wrong_capability", "challenge_mismatch"),
        ("wrong_operation_class", "challenge_mismatch"),
        ("wrong_argument_digest", "challenge_mismatch"),
    ],
)
def test_invalid_continuation_identity_does_not_issue_replacement_or_allow_selection(
    case: str,
    expected_reason: str,
):
    client = TestClient(app)
    origin_turn = _start_turn(client)
    first = client.post(
        "/v1/capabilities/authorize",
        json=_authorization_request(
            origin_turn,
            authorization_stage="selection",
            request_id=f"continuation-origin-{case}",
            capability_id="integration.external_write.test",
            operation_class="external_write",
            argument_digest="args_continuation_identity",
        ),
    ).json()["result"]
    continuation_turn = _start_turn(client)
    capability_id = "integration.external_write.test"
    operation_class = "external_write"
    argument_digest = "args_continuation_identity"
    challenge_ref = first["challenge_ref"]
    request_overrides: dict[str, object] = {}
    attempt_turn = continuation_turn

    if case == "missing":
        challenge_ref = "capconfirm_missing"
    elif case == "same_turn":
        attempt_turn = origin_turn
    elif case == "not_current":
        _start_turn(client)
    elif case == "created_before_issue":
        issued_at = str(_challenge_record(first["challenge_ref"])["issued_at"])
        with capability_authorization_repository()._connect() as conn:
            conn.execute(
                """
                UPDATE conversation_runtime_turns
                SET created_at = ?
                WHERE runtime_turn_id = ?;
                """,
                (
                    (datetime.fromisoformat(issued_at) - timedelta(seconds=1)).isoformat(),
                    continuation_turn["runtime_turn"]["runtime_turn_id"],
                ),
            )
    elif case == "wrong_owner":
        attempt_turn = _start_turn(client, owner_id="owner-2")
        request_overrides["owner_id"] = "owner-2"
    elif case == "wrong_conversation_session":
        attempt_turn = _start_turn(client, conversation_id="conv-2")
        request_overrides["conversation_id"] = "conv-2"
    elif case == "wrong_capability":
        capability_id = "integration.external_write.other"
    elif case == "wrong_operation_class":
        operation_class = "high_impact"
    elif case == "wrong_argument_digest":
        argument_digest = "args_continuation_changed"

    rows_before = _challenge_row_count()
    issued_events_before = _issued_challenge_event_count()
    result = client.post(
        "/v1/capabilities/authorize",
        json=_authorization_request(
            attempt_turn,
            authorization_stage="selection",
            request_id=f"continuation-invalid-{case}",
            capability_id=capability_id,
            operation_class=operation_class,
            argument_digest=argument_digest,
            confirmation_challenge_ref=challenge_ref,
            **request_overrides,
        ),
    ).json()["result"]

    assert result["allowed"] is False
    expected_decision = (
        "authorization_denied"
        if expected_reason == "confirmation_turn_not_current"
        else "confirmation_rejected"
    )
    assert result["decision_code"] == expected_decision
    assert result["reason_codes"] == [expected_reason]
    assert result["challenge_ref"] is None
    assert result["challenge_expires_at"] is None
    assert _challenge_row_count() == rows_before == 1
    assert _issued_challenge_event_count() == issued_events_before == 1
    assert _challenge_record(first["challenge_ref"])["status"] == "issued"
    assert _challenge_record(first["challenge_ref"])["consumed_at"] is None


@pytest.mark.parametrize(
    ("challenge_state", "expected_reason"),
    [
        ("expired", "challenge_expired"),
        ("rejected", "challenge_rejected"),
        ("consumed", "challenge_consumed"),
    ],
)
def test_invalid_continuation_state_does_not_issue_replacement_or_dispatch_again(
    challenge_state: str,
    expected_reason: str,
):
    client = TestClient(app)
    origin_turn = _start_turn(client)
    first = client.post(
        "/v1/capabilities/authorize",
        json=_authorization_request(
            origin_turn,
            authorization_stage="selection",
            request_id=f"continuation-state-origin-{challenge_state}",
            capability_id="integration.external_write.test",
            operation_class="external_write",
            argument_digest="args_continuation_state",
        ),
    ).json()["result"]
    continuation_turn = _start_turn(client)
    challenge_ref = first["challenge_ref"]

    if challenge_state == "expired":
        with capability_authorization_repository()._connect() as conn:
            conn.execute(
                """
                UPDATE capability_confirmation_challenges
                SET expires_at = ?, updated_at = ?
                WHERE confirmation_challenge_ref = ?;
                """,
                (_iso(-60), _iso(-60), challenge_ref),
            )
    else:
        confirmation = client.post(
            "/v1/capabilities/confirm",
            json={
                **_base(),
                "request_id": f"continuation-state-confirm-{challenge_state}",
                "runtime_session_id": origin_turn["runtime_session"][
                    "runtime_session_id"
                ],
                "runtime_turn_id": continuation_turn["runtime_turn"][
                    "runtime_turn_id"
                ],
                "confirmation_challenge_ref": challenge_ref,
                "capability_id": "integration.external_write.test",
                "operation_class": "external_write",
                "argument_digest": "args_continuation_state",
                "confirmed": challenge_state == "consumed",
            },
        )
        assert confirmation.status_code == 200
        if challenge_state == "consumed":
            dispatch = client.post(
                "/v1/capabilities/authorize",
                json=_authorization_request(
                    continuation_turn,
                    authorization_stage="dispatch",
                    request_id="continuation-state-initial-dispatch",
                    capability_id="integration.external_write.test",
                    operation_class="external_write",
                    argument_digest="args_continuation_state",
                    confirmation_challenge_ref=challenge_ref,
                ),
            ).json()["result"]
            assert dispatch["allowed"] is True

    rows_before = _challenge_row_count()
    issued_events_before = _issued_challenge_event_count()
    continuation = client.post(
        "/v1/capabilities/authorize",
        json=_authorization_request(
            continuation_turn,
            authorization_stage="selection",
            request_id=f"continuation-state-invalid-{challenge_state}",
            capability_id="integration.external_write.test",
            operation_class="external_write",
            argument_digest="args_continuation_state",
            confirmation_challenge_ref=challenge_ref,
        ),
    ).json()["result"]
    replay_dispatch = client.post(
        "/v1/capabilities/authorize",
        json=_authorization_request(
            continuation_turn,
            authorization_stage="dispatch",
            request_id=f"continuation-state-dispatch-{challenge_state}",
            capability_id="integration.external_write.test",
            operation_class="external_write",
            argument_digest="args_continuation_state",
            confirmation_challenge_ref=challenge_ref,
        ),
    ).json()["result"]

    assert continuation["allowed"] is False
    assert continuation["decision_code"] == "confirmation_rejected"
    assert continuation["reason_codes"] == [expected_reason]
    assert continuation["challenge_ref"] is None
    assert continuation["challenge_expires_at"] is None
    assert replay_dispatch["allowed"] is False
    assert expected_reason in replay_dispatch["reason_codes"]
    assert _challenge_row_count() == rows_before == 1
    assert _issued_challenge_event_count() == issued_events_before == 1


def test_unregistered_generic_confirmation_continuation_and_atomic_consumption_remain_compatible():
    client = TestClient(app)
    origin_turn = _start_turn(client)
    first = client.post(
        "/v1/capabilities/authorize",
        json=_authorization_request(
            origin_turn,
            authorization_stage="selection",
            request_id="generic-compatibility-origin",
            capability_id="integration.external_write.test",
            operation_class="external_write",
            argument_digest="args_generic_compatibility",
        ),
    ).json()["result"]
    confirmation_turn = _start_turn(client)
    continuation = client.post(
        "/v1/capabilities/authorize",
        json=_authorization_request(
            confirmation_turn,
            authorization_stage="selection",
            request_id="generic-compatibility-continuation",
            capability_id="integration.external_write.test",
            operation_class="external_write",
            argument_digest="args_generic_compatibility",
            confirmation_challenge_ref=first["challenge_ref"],
        ),
    ).json()["result"]
    confirmed = client.post(
        "/v1/capabilities/confirm",
        json={
            **_base(),
            "request_id": "confirm-generic-compatibility",
            "runtime_session_id": origin_turn["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": confirmation_turn["runtime_turn"]["runtime_turn_id"],
            "confirmation_challenge_ref": first["challenge_ref"],
            "capability_id": "integration.external_write.test",
            "operation_class": "external_write",
            "argument_digest": "args_generic_compatibility",
            "confirmed": True,
        },
    )
    assert confirmed.status_code == 200
    dispatch_request = _authorization_request(
        confirmation_turn,
        authorization_stage="dispatch",
        request_id="generic-compatibility-dispatch",
        capability_id="integration.external_write.test",
        operation_class="external_write",
        argument_digest="args_generic_compatibility",
        confirmation_challenge_ref=first["challenge_ref"],
    )
    first_dispatch = client.post(
        "/v1/capabilities/authorize",
        json=dispatch_request,
    ).json()["result"]
    runtime_session_id = origin_turn["runtime_session"]["runtime_session_id"]
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}").json()
    dispatch_event = next(
        event["event_payload_json"]
        for event in diagnostics["events"]
        if event["event_type"] == "capability_authorization_evaluated"
        and event["event_payload_json"]["request_id"]
        == "generic-compatibility-dispatch"
    )
    event_stage_key = "pha" + "se"
    assert first_dispatch["allowed"] is True
    assert first_dispatch["confirmation_state"] == "accepted"
    assert first_dispatch["challenge_ref"] == first["challenge_ref"]
    assert first_dispatch["challenge_expires_at"] is None
    assert dispatch_event["challenge_ref"] == first["challenge_ref"]
    assert dispatch_event["operation_class"] == "external_write"
    assert dispatch_event[event_stage_key] == "dispatch"
    assert set(dispatch_event) == {
        "request_id",
        event_stage_key,
        "capability_id",
        "operation_class",
        "decision_code",
        "reason_codes",
        "relationship_ids_used",
        "world_state_claim_ids_used",
        "confirmation_state",
        "challenge_ref",
        "challenge_expires_at",
        "revalidation_required",
    }
    consumed_events_before_replay = _consumed_event_count(client, runtime_session_id)
    rows_before_replay = _challenge_row_count()
    issued_events_before_replay = _issued_challenge_event_count()
    replay = client.post(
        "/v1/capabilities/authorize",
        json=dispatch_request,
    ).json()["result"]

    assert continuation["decision_code"] == "confirmation_required"
    assert continuation["challenge_ref"] == first["challenge_ref"]
    assert continuation["challenge_expires_at"] == first["challenge_expires_at"]
    assert _challenge_row_count() == 1
    assert _issued_challenge_event_count() == 1
    assert replay["allowed"] is False
    assert replay["reason_codes"] == ["challenge_consumed"]
    assert _challenge_row_count() == rows_before_replay == 1
    assert _issued_challenge_event_count() == issued_events_before_replay == 1
    assert _consumed_event_count(client, runtime_session_id) == consumed_events_before_replay == 1


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
