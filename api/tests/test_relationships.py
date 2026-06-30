from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from main import _relationship_domain_http_error, app
from services.relationships import RelationshipRepository


def _iso(delta_seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=delta_seconds)).isoformat()


def _base(owner_id: str = "owner") -> dict[str, object]:
    return {
        "request_id": "rid-relationship",
        "owner_id": owner_id,
        "conversation_id": "conv-1",
        "surface": "dev",
    }


def _entity(
    entity_id: str,
    *,
    label: str,
    entity_type: str = "project",
    domain: str = "project_context",
) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "canonical_label": label,
        "display_label": label.title(),
        "domain": domain,
        "sensitivity_level": "medium",
        "source_type": "trusted_config",
        "source_ref": "config:test",
        "canonical_memory_ref": None,
        "artifact_ref": None,
        "status": "active",
        "archived_at": None,
    }


def _edge(**overrides) -> dict[str, object]:
    payload = {
        "relationship_id": None,
        "subject_entity_id": "project:alpha",
        "relationship_type": "works_on",
        "object_entity_id": "repo:alpha",
        "relationship_scope": "project_context",
        "source_type": "trusted_config",
        "source_refs_json": ["config:project-alpha"],
        "confidence": 0.8,
        "status": "active",
        "sensitivity_level": "medium",
        "mentionability": "mentionable",
        "allowed_persona_scopes_json": [],
        "blocked_persona_scopes_json": [],
        "valid_from": _iso(-3600),
        "valid_until": None,
        "supersede_existing_relationship_id": None,
        "superseded_by_relationship_id": None,
        "revoked_at": None,
    }
    payload.update(overrides)
    return payload


def _seed_entities(client: TestClient, *, owner_id: str = "owner") -> None:
    base = _base(owner_id)
    client.post(
        "/v1/relationships/entities/upsert",
        json={**base, "entity": _entity("project:alpha", label="project alpha")},
    )
    client.post(
        "/v1/relationships/entities/upsert",
        json={
            **base,
            "entity": _entity("repo:alpha", label="repo alpha", entity_type="repository"),
        },
    )
    client.post(
        "/v1/relationships/entities/upsert",
        json={**base, "entity": _entity("repo:beta", label="repo beta", entity_type="repository")},
    )
    client.post(
        "/v1/relationships/entities/upsert",
        json={
            **base,
            "entity": _entity(
                "person:alex",
                label="alex",
                entity_type="person",
                domain="professional_context",
            ),
        },
    )


def _diagnostics(client: TestClient, *, owner_id: str = "owner") -> dict[str, object]:
    return client.post("/v1/relationships/diagnostics", json=_base(owner_id)).json()


def _evidence(summary: str = "Configured project-repo binding.") -> dict[str, object]:
    return {
        "evidence_type": "config_reference",
        "source_ref": "config:project-alpha",
        "summary": summary,
        "confidence_delta": 0.2,
    }


def test_entity_create_and_upsert_round_trip_preserves_provenance():
    client = TestClient(app)

    first = client.post(
        "/v1/relationships/entities/upsert",
        json={**_base(), "entity": _entity("project:alpha", label="project alpha")},
    )
    second = client.post(
        "/v1/relationships/entities/upsert",
        json={
            **_base(),
            "entity": {
                **_entity("project:alpha", label="project alpha"),
                "source_ref": "config:test:v2",
                "display_label": "Project Alpha Updated",
            },
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    body = second.json()["entity"]
    assert body["entity_id"] == "project:alpha"
    assert body["source_ref"] == "config:test:v2"
    assert body["status"] == "active"


def test_relationship_edge_create_with_evidence_round_trip():
    client = TestClient(app)
    _seed_entities(client)

    response = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(),
            "evidence": [_evidence()],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["relationship"]["status"] == "active"
    assert body["evidence"][0]["evidence_type"] == "config_reference"


def test_trusted_active_social_relationships_succeed_through_endpoint():
    client = TestClient(app)
    _seed_entities(client)
    trusted_cases = [
        (
            "explicit_user_confirmation",
            "user_confirmation",
            "chat:user-confirmed",
            "User explicitly confirmed the collaboration.",
        ),
        (
            "trusted_config",
            "config_reference",
            "config:trusted-collaboration",
            "Trusted configuration defined the collaboration.",
        ),
        (
            "trusted_integration_metadata",
            "integration_metadata",
            "integration:directory",
            "Trusted integration metadata supplied the collaboration.",
        ),
    ]

    for source_type, evidence_type, source_ref, summary in trusted_cases:
        response = client.post(
            "/v1/relationships/edges/upsert",
            json={
                **_base(),
                "edge": _edge(
                    relationship_type="collaborates_with",
                    subject_entity_id="person:alex",
                    object_entity_id="project:alpha",
                    relationship_scope="professional_context",
                    source_type=source_type,
                    source_refs_json=[source_ref],
                    status="active",
                ),
                "evidence": [
                    {
                        "evidence_type": evidence_type,
                        "source_ref": source_ref,
                        "summary": summary,
                        "confidence_delta": 0.1,
                    }
                ],
            },
        )

        assert response.status_code == 200
        body = response.json()
        relationship = body["relationship"]
        assert relationship["status"] == "active"
        assert relationship["source_type"] == source_type
        assert relationship["source_refs_json"] == [source_ref]
        assert body["evidence"][0]["summary"] == summary
        diagnostics = _diagnostics(client)
        stored = next(
            item
            for item in diagnostics["relationships"]
            if item["relationship_id"] == relationship["relationship_id"]
        )
        assert stored["relationship_id"] == relationship["relationship_id"]
        assert stored["status"] == "active"
        assert any(item["source_ref"] == source_ref for item in diagnostics["evidence"])


def test_confirm_provisional_relationship():
    client = TestClient(app)
    _seed_entities(client)
    created = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(status="provisional", source_type="tool_output"),
            "evidence": [],
        },
    ).json()["relationship"]

    response = client.post(
        "/v1/relationships/edges/confirm",
        json={
            **_base(),
            "relationship_id": created["relationship_id"],
            "evidence": {
                "evidence_type": "user_confirmation",
                "source_ref": "chat:user-confirmed",
                "summary": "User confirmed this relationship.",
                "confidence_delta": 0.1,
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["relationship"]["status"] == "active"


def test_confirm_requires_evidence_and_confirmable_status():
    client = TestClient(app, raise_server_exceptions=False)
    _seed_entities(client)
    created = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(status="provisional", source_type="tool_output"),
            "evidence": [],
        },
    ).json()["relationship"]

    no_evidence = client.post(
        "/v1/relationships/edges/confirm",
        json={**_base(), "relationship_id": created["relationship_id"], "evidence": None},
    )

    active_edge = client.post(
        "/v1/relationships/edges/upsert",
        json={**_base(), "edge": _edge(), "evidence": []},
    ).json()["relationship"]
    wrong_status = client.post(
        "/v1/relationships/edges/confirm",
        json={
            **_base(),
            "relationship_id": active_edge["relationship_id"],
            "evidence": {
                "evidence_type": "user_confirmation",
                "source_ref": "chat:user-confirmed",
                "summary": "Explicit confirmation.",
                "confidence_delta": 0.1,
            },
        },
    )

    assert no_evidence.status_code == 400
    assert no_evidence.json() == {"detail": "relationship_confirmation_evidence_required"}
    assert wrong_status.status_code == 409
    assert wrong_status.json() == {"detail": "relationship_edge_status_not_confirmable"}
    diagnostics = _diagnostics(client)
    created_after = next(
        item
        for item in diagnostics["relationships"]
        if item["relationship_id"] == created["relationship_id"]
    )
    active_after = next(
        item
        for item in diagnostics["relationships"]
        if item["relationship_id"] == active_edge["relationship_id"]
    )
    assert created_after["status"] == "provisional"
    assert active_after["status"] == "active"
    assert diagnostics["evidence"] == []


def test_confirm_rejects_revoked_and_expired_edges_without_mutation():
    client = TestClient(app, raise_server_exceptions=False)
    _seed_entities(client)
    revoked = client.post(
        "/v1/relationships/edges/upsert",
        json={**_base(), "edge": _edge(status="revoked", revoked_at=_iso(-120)), "evidence": []},
    ).json()["relationship"]
    expired = client.post(
        "/v1/relationships/edges/upsert",
        json={**_base(), "edge": _edge(status="expired", valid_until=_iso(-60)), "evidence": []},
    ).json()["relationship"]

    for relationship in (revoked, expired):
        response = client.post(
            "/v1/relationships/edges/confirm",
            json={
                **_base(),
                "relationship_id": relationship["relationship_id"],
                "evidence": {
                    "evidence_type": "user_confirmation",
                    "source_ref": "chat:user-confirmed",
                    "summary": "Attempted confirmation.",
                    "confidence_delta": 0.1,
                },
            },
        )

        assert response.status_code == 409
        assert response.json() == {"detail": "relationship_edge_status_not_confirmable"}

    diagnostics = _diagnostics(client)
    by_id = {item["relationship_id"]: item for item in diagnostics["relationships"]}
    assert by_id[revoked["relationship_id"]]["status"] == "revoked"
    assert by_id[revoked["relationship_id"]]["revoked_at"] == revoked["revoked_at"]
    assert by_id[expired["relationship_id"]]["status"] == "expired"
    assert diagnostics["evidence"] == []


def test_revoke_relationship():
    client = TestClient(app)
    _seed_entities(client)
    created = client.post(
        "/v1/relationships/edges/upsert",
        json={**_base(), "edge": _edge(), "evidence": []},
    ).json()["relationship"]

    response = client.post(
        "/v1/relationships/edges/revoke",
        json={**_base(), "relationship_id": created["relationship_id"], "evidence": None},
    )

    assert response.status_code == 200
    assert response.json()["relationship"]["status"] == "revoked"
    assert response.json()["relationship"]["revoked_at"] is not None


def test_supersede_relationship_marks_previous_edge_superseded():
    client = TestClient(app)
    _seed_entities(client)
    first = client.post(
        "/v1/relationships/edges/upsert",
        json={**_base(), "edge": _edge(object_entity_id="repo:alpha"), "evidence": []},
    ).json()["relationship"]

    response = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:beta",
                supersede_existing_relationship_id=first["relationship_id"],
            ),
            "evidence": [],
        },
    )
    diagnostics = client.post("/v1/relationships/diagnostics", json=_base()).json()

    assert response.status_code == 200
    superseded = next(
        item
        for item in diagnostics["relationships"]
        if item["relationship_id"] == first["relationship_id"]
    )
    assert superseded["status"] == "superseded"
    assert (
        superseded["superseded_by_relationship_id"]
        == response.json()["relationship"]["relationship_id"]
    )


def test_model_inferred_relationship_is_not_active_by_default():
    client = TestClient(app)
    _seed_entities(client)

    response = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(source_type="model_inference", status="inferred"),
            "evidence": [],
        },
    )

    assert response.status_code == 200
    assert response.json()["relationship"]["status"] == "inferred"


def test_model_inference_cannot_create_active_edge_through_endpoint():
    client = TestClient(app, raise_server_exceptions=False)
    _seed_entities(client)

    rejected = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(source_type="model_inference", status="active"),
            "evidence": [
                {
                    "evidence_type": "model_rationale",
                    "source_ref": "model:turn",
                    "summary": "Model attempted to activate the edge.",
                    "confidence_delta": 0.1,
                }
            ],
        },
    )
    inferred = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(source_type="model_inference", status="inferred"),
            "evidence": [],
        },
    )
    needs_confirmation = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                relationship_type="collaborates_with",
                subject_entity_id="person:alex",
                object_entity_id="project:alpha",
                relationship_scope="professional_context",
                source_type="model_inference",
                status="needs_confirmation",
            ),
            "evidence": [],
        },
    )

    assert rejected.status_code == 400
    assert rejected.json() == {"detail": "model_inference_cannot_create_active_relationship"}
    assert inferred.status_code == 200
    assert inferred.json()["relationship"]["status"] == "inferred"
    assert needs_confirmation.status_code == 200
    assert needs_confirmation.json()["relationship"]["status"] == "needs_confirmation"
    diagnostics = _diagnostics(client)
    assert len(diagnostics["relationships"]) == 2
    assert diagnostics["evidence"] == []


def test_sensitive_or_social_model_inference_requires_confirmation():
    client = TestClient(app)
    _seed_entities(client)

    response = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                relationship_type="collaborates_with",
                subject_entity_id="person:alex",
                object_entity_id="project:alpha",
                relationship_scope="professional_context",
                source_type="model_inference",
                status="needs_confirmation",
                sensitivity_level="medium",
            ),
            "evidence": [],
        },
    )

    assert response.status_code == 200
    assert response.json()["relationship"]["status"] == "needs_confirmation"


def test_active_socialish_relationship_requires_trusted_provenance():
    client = TestClient(app, raise_server_exceptions=False)
    _seed_entities(client)

    response = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                relationship_type="colleague_of",
                subject_entity_id="person:alex",
                object_entity_id="project:alpha",
                relationship_scope="professional_context",
                source_type="tool_output",
                status="active",
            ),
            "evidence": [],
        },
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": "trusted_provenance_required_for_active_socialish_relationship"
    }
    diagnostics = _diagnostics(client)
    assert diagnostics["relationships"] == []
    assert diagnostics["evidence"] == []


def test_diagnostics_redact_restricted_details_without_hidden_scores():
    client = TestClient(app)
    _seed_entities(client)
    client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(sensitivity_level="restricted", source_refs_json=["secret:ref"]),
            "evidence": [
                {
                    "evidence_type": "config_reference",
                    "source_ref": "secret:ref",
                    "summary": "Restricted supporting context.",
                    "confidence_delta": 0.2,
                }
            ],
        },
    )

    diagnostics = client.post("/v1/relationships/diagnostics", json=_base()).json()

    assert diagnostics["relationships"][0]["source_refs_json"] == []
    assert diagnostics["relationships"][0]["source_refs_redacted"] is True
    assert diagnostics["evidence"][0]["summary"] is None
    payload_text = str(diagnostics)
    assert "score" not in payload_text
    assert "world_state_claim" not in payload_text


def test_diagnostics_ignore_include_restricted_details_without_authorization():
    client = TestClient(app)
    _seed_entities(client)
    client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(sensitivity_level="restricted", source_refs_json=["secret:ref"]),
            "evidence": [
                {
                    "evidence_type": "config_reference",
                    "source_ref": "secret:ref",
                    "summary": "Restricted supporting context.",
                    "confidence_delta": 0.2,
                }
            ],
        },
    )

    diagnostics = client.post(
        "/v1/relationships/diagnostics",
        json={**_base(), "include_restricted_details": True},
    ).json()

    assert diagnostics["relationships"][0]["source_refs_json"] == []
    assert diagnostics["relationships"][0]["source_refs_redacted"] is True
    assert diagnostics["evidence"][0]["summary"] is None
    assert diagnostics["evidence"][0]["summary_redacted"] is True


def test_relationship_select_enforces_scope_confidence_and_mentionability_rules():
    client = TestClient(app)
    _seed_entities(client)
    active = client.post(
        "/v1/relationships/edges/upsert",
        json={**_base(), "edge": _edge(), "evidence": []},
    ).json()["relationship"]
    low_conf = client.post(
        "/v1/relationships/edges/upsert",
        json={**_base(), "edge": _edge(confidence=0.4, source_type="tool_output"), "evidence": []},
    ).json()["relationship"]
    trusted_low = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(confidence=0.4, source_type="trusted_config"),
            "evidence": [],
        },
    ).json()["relationship"]
    routing_only = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(mentionability="use_for_routing_only", relationship_type="depends_on"),
            "evidence": [],
        },
    ).json()["relationship"]
    restricted = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(mentionability="restricted", relationship_type="references"),
            "evidence": [],
        },
    ).json()["relationship"]

    response = client.post(
        "/v1/relationships/select",
        json={
            **_base(),
            "active_persona_id": "technical_architect",
            "requested_scopes": ["project_context"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    ids = {item["relationship_id"] for item in body["selected_relationships"]}
    assert active["relationship_id"] in ids
    assert trusted_low["relationship_id"] in ids
    assert low_conf["relationship_id"] not in ids
    assert routing_only["relationship_id"] in ids
    assert restricted["relationship_id"] not in ids
    assert "use_for_routing_only" not in (body["prompt_content"] or "")
    projection = body["retrieval_scope_projection"]
    assert projection == {
        "applied": True,
        "relationship_ids": [
            active["relationship_id"],
            trusted_low["relationship_id"],
            routing_only["relationship_id"],
        ],
        "entity_ids": ["project:alpha", "repo:alpha"],
        "relationship_scopes": ["project_context"],
        "reason_codes": ["eligible_relationship_scope_selected"],
    }
    assert "works_on" not in str(projection)
    assert "depends_on" not in str(projection)
    assert "config:project-alpha" not in str(projection)
    assert "Project Alpha" not in str(projection)
    assert low_conf["relationship_id"] in body["trace"]["relationship_edges_excluded"]
    assert (
        body["trace"]["relationship_exclusion_reasons"][low_conf["relationship_id"]]
        == "below_confidence_threshold"
    )
    assert (
        body["trace"]["relationship_exclusion_reasons"][restricted["relationship_id"]]
        == "authorization_required"
    )


def test_relationship_select_excludes_status_scope_confidence_persona_and_expiry_cases():
    client = TestClient(app)
    _seed_entities(client)
    for entity_id, label in (
        ("repo:revoked", "revoked repo marker"),
        ("repo:superseded", "superseded repo marker"),
        ("repo:expired", "expired repo marker"),
        ("repo:restricted", "restricted repo marker"),
        ("repo:low-confidence", "low confidence repo marker"),
        ("repo:blocked-persona", "blocked persona repo marker"),
        ("repo:outside-scope", "outside scope repo marker"),
        ("repo:needs-confirmation", "needs confirmation repo marker"),
        ("repo:restricted-sensitivity", "restricted sensitivity repo marker"),
    ):
        client.post(
            "/v1/relationships/entities/upsert",
            json={
                **_base(),
                "entity": _entity(entity_id, label=label, entity_type="repository"),
            },
        )
    revoked = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:revoked",
                status="revoked",
                relationship_type="works_on",
            ),
            "evidence": [],
        },
    ).json()["relationship"]
    active_then_superseded = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:superseded",
                relationship_type="contains",
            ),
            "evidence": [],
        },
    ).json()["relationship"]
    superseding = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:beta",
                relationship_type="contains",
                supersede_existing_relationship_id=active_then_superseded["relationship_id"],
            ),
            "evidence": [],
        },
    ).json()["relationship"]
    expired = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:expired",
                relationship_type="documents",
                valid_until=_iso(-60),
            ),
            "evidence": [],
        },
    ).json()["relationship"]
    restricted = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:restricted",
                relationship_type="references",
                mentionability="restricted",
            ),
            "evidence": [],
        },
    ).json()["relationship"]
    low_confidence = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:low-confidence",
                relationship_type="depends_on",
                confidence=0.4,
                source_type="tool_output",
            ),
            "evidence": [],
        },
    ).json()["relationship"]
    blocked_persona = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:blocked-persona",
                relationship_type="responsible_for",
                blocked_persona_scopes_json=["technical_architect"],
            ),
            "evidence": [],
        },
    ).json()["relationship"]
    outside_scope = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:outside-scope",
                relationship_type="related_to",
                relationship_scope="personal_context",
            ),
            "evidence": [],
        },
    ).json()["relationship"]
    needs_confirmation = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:needs-confirmation",
                relationship_type="manages",
                status="needs_confirmation",
                source_type="model_inference",
            ),
            "evidence": [],
        },
    ).json()["relationship"]
    restricted_sensitivity = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:restricted-sensitivity",
                relationship_type="references",
                sensitivity_level="restricted",
            ),
            "evidence": [],
        },
    ).json()["relationship"]

    response = client.post(
        "/v1/relationships/select",
        json={
            **_base(),
            "active_persona_id": "technical_architect",
            "requested_scopes": ["project_context"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    reasons = body["trace"]["relationship_exclusion_reasons"]
    assert reasons[revoked["relationship_id"]] == "status_revoked"
    assert reasons[active_then_superseded["relationship_id"]] == "status_superseded"
    assert reasons[expired["relationship_id"]] == "expired"
    assert reasons[restricted["relationship_id"]] == "authorization_required"
    assert reasons[low_confidence["relationship_id"]] == "below_confidence_threshold"
    assert reasons[blocked_persona["relationship_id"]] == "blocked_persona_scope"
    assert reasons[outside_scope["relationship_id"]] == "outside_persona_or_surface_scope"
    assert reasons[needs_confirmation["relationship_id"]] == "status_needs_confirmation"
    assert reasons[restricted_sensitivity["relationship_id"]] == "authorization_required"
    assert body["trace"]["relationship_confirmation_required"] is True
    selected_ids = {item["relationship_id"] for item in body["selected_relationships"]}
    assert selected_ids == {superseding["relationship_id"]}
    projection = body["retrieval_scope_projection"]
    assert projection == {
        "applied": True,
        "relationship_ids": [superseding["relationship_id"]],
        "entity_ids": ["project:alpha", "repo:beta"],
        "relationship_scopes": ["project_context"],
        "reason_codes": ["eligible_relationship_scope_selected"],
    }
    excluded_object_ids = {
        "repo:revoked",
        "repo:superseded",
        "repo:expired",
        "repo:restricted",
        "repo:low-confidence",
        "repo:blocked-persona",
        "repo:outside-scope",
        "repo:needs-confirmation",
        "repo:restricted-sensitivity",
    }
    assert excluded_object_ids.isdisjoint(projection["entity_ids"])
    prompt = body["prompt_content"] or ""
    assert "Project Alpha contains Repo Beta" in prompt
    assert "scope=project_context" in prompt
    assert "confidence=0.80" in prompt
    for excluded_prompt_value in (
        "Revoked Repo Marker",
        "Superseded Repo Marker",
        "Expired Repo Marker",
        "Restricted Repo Marker",
        "Low Confidence Repo Marker",
        "Blocked Persona Repo Marker",
        "Outside Scope Repo Marker",
        "Needs Confirmation Repo Marker",
        "Restricted Sensitivity Repo Marker",
        "works_on",
        "documents",
        "references",
        "depends_on",
        "responsible_for",
        "related_to",
    ):
        assert excluded_prompt_value not in prompt


def test_relationship_select_excludes_conflicted_relationships_without_winner_selection():
    client = TestClient(app)
    _seed_entities(client)
    first = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:alpha",
                relationship_type="defaults_to",
                source_type="trusted_config",
            ),
            "evidence": [],
        },
    ).json()["relationship"]
    second = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:beta",
                relationship_type="defaults_to",
                source_type="trusted_config",
            ),
            "evidence": [],
        },
    ).json()["relationship"]

    response = client.post(
        "/v1/relationships/select",
        json={**_base(), "active_persona_id": "technical_architect"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selected_relationships"] == []
    excluded_ids = {
        item["relationship_id"]
        for item in body["excluded_relationship_summaries"]
    }
    assert {first["relationship_id"], second["relationship_id"]}.issubset(excluded_ids)
    assert (
        body["trace"]["relationship_exclusion_reasons"][first["relationship_id"]]
        == "conflicted"
    )
    assert (
        body["trace"]["relationship_exclusion_reasons"][second["relationship_id"]]
        == "conflicted"
    )
    assert body["trace"]["relationship_conflicts"]
    assert body["trace"]["relationship_confirmation_required"] is True
    assert body["prompt_content"] is None
    assert body["retrieval_scope_projection"] == {
        "applied": False,
        "relationship_ids": [],
        "entity_ids": [],
        "relationship_scopes": [],
        "reason_codes": ["no_eligible_relationship_scope"],
    }


def test_relationship_select_allows_multiple_contains_edges_without_conflict():
    client = TestClient(app)
    _seed_entities(client)
    first = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:alpha",
                relationship_type="contains",
                source_type="trusted_config",
            ),
            "evidence": [],
        },
    ).json()["relationship"]
    second = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                object_entity_id="repo:beta",
                relationship_type="contains",
                source_type="trusted_config",
            ),
            "evidence": [],
        },
    ).json()["relationship"]

    response = client.post(
        "/v1/relationships/select",
        json={
            **_base(),
            "active_persona_id": "technical_architect",
            "requested_scopes": ["project_context"],
            "relationship_types": ["contains"],
            "entity_ids": ["project:alpha"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    selected_ids = {item["relationship_id"] for item in body["selected_relationships"]}
    assert selected_ids == {first["relationship_id"], second["relationship_id"]}
    assert body["trace"]["relationship_conflicts"] == []
    assert body["trace"]["selected_relationship_count"] == 2
    assert {item["object_entity_id"] for item in body["selected_relationships"]} == {
        "repo:alpha",
        "repo:beta",
    }
    assert body["retrieval_scope_projection"] == {
        "applied": True,
        "relationship_ids": [first["relationship_id"], second["relationship_id"]],
        "entity_ids": ["project:alpha", "repo:alpha", "repo:beta"],
        "relationship_scopes": ["project_context"],
        "reason_codes": ["eligible_relationship_scope_selected"],
    }


def test_filtering_only_relationship_projects_to_retrieval_without_prompt_mention():
    client = TestClient(app)
    _seed_entities(client)
    filtering_only = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                relationship_id="rel_filtering_only",
                mentionability="use_for_filtering_only",
                relationship_type="documents",
            ),
            "evidence": [_evidence("Filtering-only relationship evidence.")],
        },
    ).json()["relationship"]

    response = client.post(
        "/v1/relationships/select",
        json={
            **_base(),
            "active_persona_id": "technical_architect",
            "requested_scopes": ["project_context"],
            "relationship_types": ["documents"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selected_relationships"][0]["relationship_id"] == filtering_only["relationship_id"]
    assert body["prompt_content"] is None
    assert body["trace"]["relationship_edges_used"] == [filtering_only["relationship_id"]]
    assert (
        body["trace"]["relationship_exclusion_reasons"][filtering_only["relationship_id"]]
        == "use_for_filtering_only"
    )
    projection = body["retrieval_scope_projection"]
    assert projection == {
        "applied": True,
        "relationship_ids": [filtering_only["relationship_id"]],
        "entity_ids": ["project:alpha", "repo:alpha"],
        "relationship_scopes": ["project_context"],
        "reason_codes": ["eligible_relationship_scope_selected"],
    }
    for forbidden in (
        "documents",
        "active",
        "0.8",
        "medium",
        "use_for_filtering_only",
        "config:project-alpha",
        "Filtering-only relationship evidence",
        "Project Alpha",
        "Repo Alpha",
    ):
        assert forbidden not in str(projection)


def test_relationship_retrieval_projection_is_owner_isolated():
    client = TestClient(app)
    _seed_entities(client, owner_id="real-owner")
    client.post(
        "/v1/relationships/entities/upsert",
        json={
            **_base("other-owner"),
            "entity": _entity(
                "other:project:alpha",
                label="other project alpha",
                entity_type="project",
            ),
        },
    )
    client.post(
        "/v1/relationships/entities/upsert",
        json={
            **_base("other-owner"),
            "entity": _entity(
                "other:repo:beta",
                label="other repo beta",
                entity_type="repository",
            ),
        },
    )
    real = client.post(
        "/v1/relationships/edges/upsert",
        json={**_base("real-owner"), "edge": _edge(), "evidence": []},
    ).json()["relationship"]
    other = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base("other-owner"),
            "edge": _edge(
                relationship_id="rel_other_owner",
                subject_entity_id="other:project:alpha",
                object_entity_id="other:repo:beta",
            ),
            "evidence": [],
        },
    ).json()["relationship"]

    response = client.post(
        "/v1/relationships/select",
        json={
            **_base("real-owner"),
            "active_persona_id": "technical_architect",
            "requested_scopes": ["project_context"],
        },
    )

    assert response.status_code == 200
    projection = response.json()["retrieval_scope_projection"]
    assert projection == {
        "applied": True,
        "relationship_ids": [real["relationship_id"]],
        "entity_ids": ["project:alpha", "repo:alpha"],
        "relationship_scopes": ["project_context"],
        "reason_codes": ["eligible_relationship_scope_selected"],
    }
    assert other["relationship_id"] not in projection["relationship_ids"]
    assert "other:repo:beta" not in projection["entity_ids"]


def test_cross_owner_confirmation_and_revocation_return_owner_scoped_404_without_mutation():
    client = TestClient(app, raise_server_exceptions=False)
    _seed_entities(client, owner_id="real-owner")
    created = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base("real-owner"),
            "edge": _edge(status="provisional", source_type="tool_output"),
            "evidence": [],
        },
    ).json()["relationship"]

    confirm = client.post(
        "/v1/relationships/edges/confirm",
        json={
            **_base("other-owner"),
            "relationship_id": created["relationship_id"],
            "evidence": {
                "evidence_type": "user_confirmation",
                "source_ref": "chat:other-owner",
                "summary": "Wrong owner confirmation attempt.",
                "confidence_delta": 0.1,
            },
        },
    )
    revoke = client.post(
        "/v1/relationships/edges/revoke",
        json={
            **_base("other-owner"),
            "relationship_id": created["relationship_id"],
            "evidence": None,
        },
    )

    assert confirm.status_code == 404
    assert confirm.json() == {"detail": "relationship_edge_not_found"}
    assert revoke.status_code == 404
    assert revoke.json() == {"detail": "relationship_edge_not_found"}
    diagnostics = _diagnostics(client, owner_id="real-owner")
    assert diagnostics["relationships"][0]["status"] == "provisional"
    assert diagnostics["evidence"] == []


def test_superseding_missing_or_cross_owner_edge_rolls_back_replacement_creation():
    client = TestClient(app, raise_server_exceptions=False)
    _seed_entities(client, owner_id="real-owner")
    client.post(
        "/v1/relationships/entities/upsert",
        json={
            **_base("other-owner"),
            "entity": _entity(
                "other:project:alpha",
                label="other project alpha",
                entity_type="project",
                domain="project_context",
            ),
        },
    )
    client.post(
        "/v1/relationships/entities/upsert",
        json={
            **_base("other-owner"),
            "entity": _entity(
                "other:repo:beta",
                label="other repo beta",
                entity_type="repository",
                domain="project_context",
            ),
        },
    )
    real = client.post(
        "/v1/relationships/edges/upsert",
        json={**_base("real-owner"), "edge": _edge(), "evidence": []},
    ).json()["relationship"]

    missing = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base("other-owner"),
            "edge": _edge(
                relationship_id="rel_replacement_missing",
                subject_entity_id="other:project:alpha",
                object_entity_id="other:repo:beta",
                supersede_existing_relationship_id="rel_missing",
            ),
            "evidence": [_evidence()],
        },
    )
    cross_owner = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base("other-owner"),
            "edge": _edge(
                relationship_id="rel_replacement_cross_owner",
                subject_entity_id="other:project:alpha",
                object_entity_id="other:repo:beta",
                supersede_existing_relationship_id=real["relationship_id"],
            ),
            "evidence": [_evidence()],
        },
    )

    assert missing.status_code == 404
    assert missing.json() == {"detail": "relationship_edge_not_found"}
    assert cross_owner.status_code == 404
    assert cross_owner.json() == {"detail": "relationship_edge_not_found"}
    assert _diagnostics(client, owner_id="other-owner")["relationships"] == []
    real_diagnostics = _diagnostics(client, owner_id="real-owner")
    assert real_diagnostics["relationships"][0]["status"] == "active"
    assert real_diagnostics["relationships"][0]["superseded_by_relationship_id"] is None


def test_confirmation_rolls_back_status_update_when_evidence_insert_fails(monkeypatch):
    client = TestClient(app)
    _seed_entities(client)
    created = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(status="provisional", source_type="tool_output"),
            "evidence": [],
        },
    ).json()["relationship"]

    def fail_evidence_insert(self, *args, **kwargs):
        raise RuntimeError("synthetic_evidence_failure")

    monkeypatch.setattr(RelationshipRepository, "_insert_evidence", fail_evidence_insert)
    with pytest.raises(RuntimeError, match="synthetic_evidence_failure"):
        client.post(
            "/v1/relationships/edges/confirm",
            json={
                **_base(),
                "relationship_id": created["relationship_id"],
                "evidence": {
                    "evidence_type": "user_confirmation",
                    "source_ref": "chat:user-confirmed",
                    "summary": "Should not be committed.",
                    "confidence_delta": 0.1,
                },
            },
        )

    diagnostics = _diagnostics(client)
    stored = diagnostics["relationships"][0]
    assert stored["relationship_id"] == created["relationship_id"]
    assert stored["status"] == "provisional"
    assert diagnostics["evidence"] == []


def test_relationship_known_errors_are_privacy_safe_and_unknown_errors_are_not_translated():
    client = TestClient(app, raise_server_exceptions=False)
    _seed_entities(client)

    response = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                relationship_type="colleague_of",
                subject_entity_id="person:alex",
                object_entity_id="project:alpha",
                relationship_scope="professional_context",
                source_type="tool_output",
                source_refs_json=["artifact:private-source"],
                status="active",
            ),
            "evidence": [
                {
                    "evidence_type": "artifact_reference",
                    "source_ref": "artifact:private-source",
                    "summary": "Private evidence summary should not leak.",
                    "confidence_delta": 0.1,
                }
            ],
        },
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": "trusted_provenance_required_for_active_socialish_relationship"
    }
    payload = response.text
    for forbidden in (
        "Private evidence summary",
        "artifact:private-source",
        "person:alex",
        "project:alpha",
        "sqlite",
        "Traceback",
        "rel_",
    ):
        assert forbidden not in payload
    assert _relationship_domain_http_error(RuntimeError("synthetic_unknown_failure")) is None
