from __future__ import annotations

from fastapi.testclient import TestClient
from main import app


def _base(owner_id: str = "owner") -> dict[str, object]:
    return {
        "request_id": "rid-social-context",
        "owner_id": owner_id,
        "conversation_id": "conv-1",
        "surface": "dev",
    }


def _entity(entity_id: str, *, label: str, entity_type: str, domain: str) -> dict[str, object]:
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


def _seed_relationship(client: TestClient, *, owner_id: str = "owner") -> str:
    base = _base(owner_id)
    client.post(
        "/v1/relationships/entities/upsert",
        json={
            **base,
            "entity": _entity(
                "project:alpha",
                label="project alpha",
                entity_type="project",
                domain="project_context",
            ),
        },
    )
    client.post(
        "/v1/relationships/entities/upsert",
        json={
            **base,
            "entity": _entity(
                "repo:alpha",
                label="repo alpha",
                entity_type="repository",
                domain="project_context",
            ),
        },
    )
    response = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **base,
            "edge": {
                "relationship_id": None,
                "subject_entity_id": "project:alpha",
                "relationship_type": "works_on",
                "object_entity_id": "repo:alpha",
                "relationship_scope": "project_context",
                "source_type": "trusted_config",
                "source_refs_json": ["config:project-alpha"],
                "confidence": 0.9,
                "status": "active",
                "sensitivity_level": "medium",
                "mentionability": "mentionable",
                "allowed_persona_scopes_json": [],
                "blocked_persona_scopes_json": [],
                "valid_from": None,
                "valid_until": None,
                "supersede_existing_relationship_id": None,
                "superseded_by_relationship_id": None,
                "revoked_at": None,
            },
            "evidence": [],
        },
    )
    return response.json()["relationship"]["relationship_id"]


def _diagnostics(client: TestClient, *, owner_id: str = "owner") -> dict[str, object]:
    return client.post("/v1/social-context/diagnostics", json=_base(owner_id)).json()


def test_social_context_item_requires_source_refs():
    client = TestClient(app, raise_server_exceptions=False)
    relationship_id = _seed_relationship(client)

    response = client.post(
        "/v1/social-context/items/upsert",
        json={
            **_base(),
            "item": {
                "social_context_id": None,
                "context_type": "relationship_reference",
                "summary": "Professional repo context.",
                "source_refs_json": [],
                "relationship_edge_refs_json": [relationship_id],
                "confidence": 0.8,
                "freshness": "fresh",
                "mentionability": "mentionable",
            },
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "social_context_source_refs_required"}
    assert _diagnostics(client)["items"] == []


def test_social_context_relationship_refs_must_point_to_approved_edges():
    client = TestClient(app, raise_server_exceptions=False)
    _seed_relationship(client)

    response = client.post(
        "/v1/social-context/items/upsert",
        json={
            **_base(),
            "item": {
                "social_context_id": None,
                "context_type": "relationship_reference",
                "summary": "Professional repo context.",
                "source_refs_json": ["artifact:note"],
                "relationship_edge_refs_json": ["rel_missing"],
                "confidence": 0.8,
                "freshness": "fresh",
                "mentionability": "mentionable",
            },
        },
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "social_context_requires_approved_relationship_edge"}
    assert _diagnostics(client)["items"] == []


def test_social_context_usage_for_missing_item_returns_owner_scoped_404_without_event():
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/v1/social-context/usage-events/record",
        json={
            **_base(),
            "event": {
                "event_id": None,
                "social_context_id": "socctx_missing",
                "relationship_edge_refs_json": ["rel_private_other_owner"],
                "runtime_turn_id": "rtturn_1",
                "usage_type": "diagnostic_review",
                "policy_decision": "suppressed",
            },
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "social_context_item_not_found"}
    diagnostics = _diagnostics(client)
    assert diagnostics["items"] == []
    assert diagnostics["usage_events"] == []


def test_social_context_known_errors_are_privacy_safe():
    client = TestClient(app, raise_server_exceptions=False)
    _seed_relationship(client)

    response = client.post(
        "/v1/social-context/items/upsert",
        json={
            **_base(),
            "item": {
                "social_context_id": None,
                "context_type": "relationship_reference",
                "summary": "Private social-context summary should not leak.",
                "source_refs_json": ["artifact:private-social-note"],
                "relationship_edge_refs_json": ["rel_private_other_owner"],
                "confidence": 0.8,
                "freshness": "fresh",
                "mentionability": "mentionable",
            },
        },
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "social_context_requires_approved_relationship_edge"}
    payload = response.text
    for forbidden in (
        "Private social-context summary",
        "artifact:private-social-note",
        "rel_private_other_owner",
        "sqlite",
        "Traceback",
    ):
        assert forbidden not in payload


def test_social_context_diagnostics_expose_suppression_and_references_without_hidden_scores():
    client = TestClient(app)
    relationship_id = _seed_relationship(client)
    created = client.post(
        "/v1/social-context/items/upsert",
        json={
            **_base(),
            "item": {
                "social_context_id": None,
                "context_type": "known_boundary",
                "summary": "Do not surface repo relationship casually.",
                "source_refs_json": ["artifact:boundary-note"],
                "relationship_edge_refs_json": [relationship_id],
                "confidence": 0.4,
                "freshness": "fresh",
                "mentionability": "suppress_by_default",
            },
        },
    ).json()["item"]

    usage = client.post(
        "/v1/social-context/usage-events/record",
        json={
            **_base(),
            "event": {
                "event_id": None,
                "social_context_id": created["social_context_id"],
                "relationship_edge_refs_json": [relationship_id],
                "runtime_turn_id": "rtturn_1",
                "usage_type": "diagnostic_review",
                "policy_decision": "suppressed",
            },
        },
    )
    diagnostics = client.post("/v1/social-context/diagnostics", json=_base()).json()

    assert usage.status_code == 200
    assert diagnostics["items"][0]["suppressed_by_default"] is True
    assert set(diagnostics["items"][0]["suppression_reasons"]) == {
        "low_confidence",
        "mentionability_restricted",
    }
    assert diagnostics["items"][0]["relationship_edge_refs_json"] == [relationship_id]
    assert diagnostics["usage_events"][0]["policy_decision"] == "suppressed"
    payload_text = str(diagnostics)
    assert "emotion" not in payload_text
    assert "dependency" not in payload_text
    assert "score" not in payload_text
