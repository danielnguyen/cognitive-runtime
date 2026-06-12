from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from main import app


def _iso(delta_seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=delta_seconds)).isoformat()


def _base() -> dict[str, object]:
    return {
        "request_id": "rid-relationship",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
    }


def _entity(entity_id: str, *, label: str, entity_type: str = "project", domain: str = "project_context") -> dict[str, object]:
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


def _seed_entities(client: TestClient) -> None:
    client.post("/v1/relationships/entities/upsert", json={**_base(), "entity": _entity("project:alpha", label="project alpha")})
    client.post(
        "/v1/relationships/entities/upsert",
        json={**_base(), "entity": _entity("repo:alpha", label="repo alpha", entity_type="repository")},
    )
    client.post(
        "/v1/relationships/entities/upsert",
        json={**_base(), "entity": _entity("repo:beta", label="repo beta", entity_type="repository")},
    )
    client.post(
        "/v1/relationships/entities/upsert",
        json={**_base(), "entity": _entity("person:alex", label="alex", entity_type="person", domain="professional_context")},
    )


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
            "evidence": [
                {
                    "evidence_type": "config_reference",
                    "source_ref": "config:project-alpha",
                    "summary": "Configured project-repo binding.",
                    "confidence_delta": 0.2,
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["relationship"]["status"] == "active"
    assert body["evidence"][0]["evidence_type"] == "config_reference"


def test_confirm_provisional_relationship():
    client = TestClient(app)
    _seed_entities(client)
    created = client.post(
        "/v1/relationships/edges/upsert",
        json={**_base(), "edge": _edge(status="provisional", source_type="tool_output"), "evidence": []},
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
        json={**_base(), "edge": _edge(status="provisional", source_type="tool_output"), "evidence": []},
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

    assert no_evidence.status_code == 500
    assert wrong_status.status_code == 500


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
        item for item in diagnostics["relationships"] if item["relationship_id"] == first["relationship_id"]
    )
    assert superseded["status"] == "superseded"
    assert superseded["superseded_by_relationship_id"] == response.json()["relationship"]["relationship_id"]


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

    assert response.status_code == 500


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
        json={**_base(), "edge": _edge(confidence=0.4, source_type="trusted_config"), "evidence": []},
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
        json={**_base(), "active_persona_id": "technical_architect", "requested_scopes": ["project_context"]},
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
    assert low_conf["relationship_id"] in body["trace"]["relationship_edges_excluded"]
    assert body["trace"]["relationship_exclusion_reasons"][low_conf["relationship_id"]] == "below_confidence_threshold"
    assert body["trace"]["relationship_exclusion_reasons"][restricted["relationship_id"]] == "authorization_required"


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
    assert response.json()["selected_relationships"] == []
    excluded_ids = {item["relationship_id"] for item in response.json()["excluded_relationship_summaries"]}
    assert {first["relationship_id"], second["relationship_id"]}.issubset(excluded_ids)
    assert response.json()["trace"]["relationship_conflicts"]


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
