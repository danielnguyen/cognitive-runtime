from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from main import _relationship_domain_http_error, app
from services.relationships import RelationshipRepository, relationship_repository
from services.runtime_state import clear_states_for_tests, runtime_state_repository


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


def _strict_relationship_turn(
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
    request = {**payload, "persona_selection_mode": "strict"}
    decision = None
    if selection:
        identity = client.post("/v1/runtime/identity/resolve", json=request)
        assert identity.status_code == 200
        decision = identity.json()["persona_selection"]
    request["persona_selection_ref"] = (
        decision["selection_ref"] if decision else "psel_00000000000000000000000000000000"
    )
    containment_request = {**request, "current_user_text": containment_text or text}
    if selection and containment:
        result = client.post("/v1/runtime/persona-containment/evaluate", json=containment_request)
        assert result.status_code == 200
    return request, decision, containment_request


def _seed_strict_relationships(client):
    _seed_entities(client)
    cases = {
        "rel-project": {},
        "rel-professional": {"relationship_scope": "professional_context"},
        "rel-system": {
            "relationship_scope": "system_configuration", "mentionability": "use_for_routing_only",
        },
        "rel-private-hidden": {
            "relationship_scope": "personal_context",
            "allowed_persona_scopes_json": ["personal_companion"],
        },
        "rel-technical-only": {"allowed_persona_scopes_json": ["technical_architect"]},
    }
    for reference, overrides in cases.items():
        response = client.post("/v1/relationships/edges/upsert", json={
            **_base(), "edge": _edge(relationship_id=reference, **overrides),
            "evidence": [_evidence()],
        })
        assert response.status_code == 200


def _assert_relationship_authority_failure(response):
    assert response.status_code == 409
    assert response.json() == {"detail": "relationship_authority_rejected"}
    assert not any(key in response.json() for key in (
        "selected_relationships", "prompt_content", "retrieval_scope_projection", "trace",
    ))
    assert "rel-private-hidden" not in response.text
    assert "private_sentinel" not in response.text


@pytest.mark.parametrize("surface,persona,selected", [
    ("web", "general_assistant", {"rel-project", "rel-professional"}),
    ("dev", "technical_architect", {
        "rel-project", "rel-professional", "rel-system", "rel-technical-only",
    }),
    ("vscode", "technical_architect", {
        "rel-project", "rel-professional", "rel-system", "rel-technical-only",
    }),
    ("unregistered", "general_assistant", {
        "rel-project", "rel-professional", "rel-system",
    }),
])
def test_strict_relationships_use_selected_persona_and_existing_surface_ceiling(
    surface, persona, selected,
):
    client = TestClient(app)
    _seed_strict_relationships(client)
    payload, decision, _ = _strict_relationship_turn(client, surface=surface)
    response = client.post("/v1/relationships/select", json={
        **payload, "active_persona_id": persona,
        "expected_thread_revision": decision["thread_revision"],
    })
    assert response.status_code == 200
    result = response.json()
    assert result["selection_contract"] == "strict_turn"
    assert result["persona_selection_ref"] == decision["selection_ref"]
    assert result["trace"]["active_persona_id"] == persona
    assert decision["contextual_activation"] is False
    assert {edge["relationship_id"] for edge in result["selected_relationships"]} == selected
    projection = result["retrieval_scope_projection"]
    assert set(projection["relationship_ids"]) == selected
    assert projection["entity_ids"] == ["project:alpha", "repo:alpha"]
    assert "personal_context" not in projection["relationship_scopes"]
    assert "system_configuration" not in (result["prompt_content"] or "")


@pytest.mark.parametrize("text,proposed", [
    ("I broke the server and prod is failing", "technical_architect"),
    ("That was wrong in the report.", "personal_companion"),
])
def test_strict_advisory_proposal_cannot_expand_relationships_or_retrieval(text, proposed):
    client = TestClient(app)
    _seed_strict_relationships(client)
    payload, decision, _ = _strict_relationship_turn(client, text=text)
    assert decision["proposed_persona_id"] == proposed
    response = client.post("/v1/relationships/select", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["trace"]["active_persona_id"] == "general_assistant"
    assert set(result["retrieval_scope_projection"]["relationship_ids"]) == {
        "rel-project", "rel-professional",
    }
    assert "rel-private-hidden" not in result["retrieval_scope_projection"]["relationship_ids"]
    assert "rel-technical-only" not in result["retrieval_scope_projection"]["relationship_ids"]


@pytest.mark.parametrize("scopes,expected", [
    (["project_context"], {"rel-project"}),
    (["personal_context"], set()),
    (["project_context", "personal_context"], {"rel-project"}),
    (["unmapped_context"], set()),
])
def test_strict_requested_relationship_scopes_only_narrow(scopes, expected):
    client = TestClient(app)
    _seed_strict_relationships(client)
    payload, _, _ = _strict_relationship_turn(client)
    response = client.post("/v1/relationships/select", json={**payload, "requested_scopes": scopes})
    assert response.status_code == 200
    result = response.json()
    assert {edge["relationship_id"] for edge in result["selected_relationships"]} == expected
    projection = result["retrieval_scope_projection"]
    if expected:
        assert projection == {
            "applied": True, "relationship_ids": ["rel-project"],
            "entity_ids": ["project:alpha", "repo:alpha"],
            "relationship_scopes": ["project_context"],
            "reason_codes": ["eligible_relationship_scope_selected"],
        }
    else:
        assert projection["applied"] is False
        assert projection["relationship_ids"] == projection["entity_ids"] == []


@pytest.mark.parametrize("field,value", [
    ("request_id", "wrong_request"), ("owner_id", "wrong_owner"),
    ("conversation_id", "wrong_conversation"), ("surface", "vscode"),
    ("runtime_session_id", "wrong_session"), ("runtime_turn_id", "wrong_turn"),
    ("expected_thread_revision", 0), ("active_persona_id", "personal_companion"),
    ("persona_selection_ref", "psel_00000000000000000000000000000000"),
])
def test_strict_relationship_mismatch_fails_before_protected_read(field, value, monkeypatch):
    client = TestClient(app)
    _seed_strict_relationships(client)
    payload, _, _ = _strict_relationship_turn(client)
    reads = []
    monkeypatch.setattr(relationship_repository(), "diagnostics", lambda **kw: reads.append(kw))
    response = client.post("/v1/relationships/select", json={**payload, field: value})
    _assert_relationship_authority_failure(response)
    assert reads == []
    events = runtime_state_repository().list_events_for_tests(payload["runtime_session_id"])
    assert sum(event.event_type == "persona_selection_resolved" for event in events) == 1


@pytest.mark.parametrize("selection,containment", [(False, False), (True, False)])
def test_strict_relationship_requires_both_predecessors_without_fabrication(
    selection, containment, monkeypatch,
):
    client = TestClient(app)
    payload, _, _ = _strict_relationship_turn(client, selection=selection, containment=containment)
    reads = []
    monkeypatch.setattr(relationship_repository(), "diagnostics", lambda **kw: reads.append(kw))
    response = client.post("/v1/relationships/select", json=payload)
    _assert_relationship_authority_failure(response)
    assert reads == []
    events = runtime_state_repository().list_events_for_tests(payload["runtime_session_id"])
    assert sum(event.event_type == "persona_selection_resolved" for event in events) == (
        int(selection)
    )
    assert not any(event.event_type == "persona_containment_evaluated" for event in events)


@pytest.mark.parametrize("mutation", [
    {"persona_selection_ref": "malformed"}, {"selection_source": "explicit_user"},
    {"persona_scope_hint": "supportive_listener"}, {"requested_persona_id": "personal_companion"},
    {"runtime_turn_id": None}, {"runtime_session_id": None}, {"persona_selection_ref": None},
    {"expected_thread_revision": True},
])
def test_strict_relationship_request_rejects_incomplete_and_spoofed_fields(mutation):
    client = TestClient(app)
    payload, _, _ = _strict_relationship_turn(client)
    response = client.post("/v1/relationships/select", json={**payload, **mutation})
    assert response.status_code == 422
    assert "rel-private-hidden" not in response.text


@pytest.mark.parametrize("status", ["completed", "abandoned"])
def test_strict_relationship_rejects_terminal_turn(status, monkeypatch):
    client = TestClient(app)
    payload, _, _ = _strict_relationship_turn(client)
    assert client.post("/v1/runtime/turns/complete", json={
        key: payload[key] for key in ("request_id", "runtime_session_id", "runtime_turn_id")
    } | {"turn_status": status}).status_code == 200
    reads = []
    monkeypatch.setattr(relationship_repository(), "diagnostics", lambda **kw: reads.append(kw))
    _assert_relationship_authority_failure(client.post("/v1/relationships/select", json=payload))
    assert reads == []


@pytest.mark.parametrize("corruption", [
    "selection_json", "selection_persona", "governance", "containment_json", "status",
    "missing_status", "owner", "revision", "persona", "reference", "missing_policy",
    "domains_string", "confidence_bool", "extra_policy", "session", "contradictory_domains",
])
def test_strict_relationship_corrupt_authority_fails_before_read(corruption, monkeypatch):
    client = TestClient(app)
    payload, _, _ = _strict_relationship_turn(client)
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
            elif corruption == "missing_policy":
                del changed["strict_containment"]
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
                elif corruption == "domains_string":
                    authority["result"]["allowed_relationship_domains"] = "general"
                elif corruption == "confidence_bool":
                    authority["result"]["confidence"] = True
                elif corruption == "extra_policy":
                    authority["result"]["current_user_text"] = "private_sentinel"
                elif corruption == "session":
                    authority["runtime_session_id"] = "wrong_session"
                elif corruption == "contradictory_domains":
                    authority["result"]["blocked_memory_domains"].append("general")
                    changed["blocked_memory_domains"].append("general")
            value = json.dumps(changed)
        conn.execute(
            "UPDATE conversation_runtime_events SET event_payload_json = ? WHERE id = ?",
            (value, row["id"]),
        )
    reads = []
    monkeypatch.setattr(relationship_repository(), "diagnostics", lambda **kw: reads.append(kw))
    _assert_relationship_authority_failure(client.post("/v1/relationships/select", json=payload))
    assert reads == []


def test_strict_relationship_accepts_only_consistent_multiple_containment_events():
    client = TestClient(app)
    _seed_strict_relationships(client)
    payload, _, containment = _strict_relationship_turn(client, surface="dev")
    assert client.post(
        "/v1/runtime/persona-containment/evaluate", json=containment,
    ).status_code == 200
    assert client.post("/v1/relationships/select", json=payload).status_code == 200
    assert client.post("/v1/runtime/persona-containment/evaluate", json={
        **containment, "current_user_text": "What is 2+2?",
    }).status_code == 200
    # Same persona/domain envelope, but conflicting independently published policy.
    _assert_relationship_authority_failure(client.post("/v1/relationships/select", json=payload))


def test_strict_narrowed_containment_fails_without_domain_scope_translation(monkeypatch):
    client = TestClient(app)
    payload, _, _ = _strict_relationship_turn(
        client, surface="dev", containment_text="Check vehicle maintenance and tire pressure.",
    )
    reads = []
    monkeypatch.setattr(relationship_repository(), "diagnostics", lambda **kw: reads.append(kw))
    _assert_relationship_authority_failure(client.post("/v1/relationships/select", json=payload))
    assert reads == []


def test_strict_relationship_selection_survives_runtime_repository_replacement():
    client = TestClient(app)
    _seed_strict_relationships(client)
    payload, _, _ = _strict_relationship_turn(client)
    before = client.post("/v1/relationships/select", json=payload)
    assert before.status_code == 200
    clear_states_for_tests(db_path=runtime_state_repository().db_path)
    after = client.post("/v1/relationships/select", json=payload)
    assert after.status_code == 200
    assert after.json() == before.json()


@pytest.mark.parametrize("change", ["completion", "revision", "containment", "surface"])
def test_strict_relationship_publication_revalidates_after_protected_read(change, monkeypatch):
    client = TestClient(app)
    _seed_strict_relationships(client)
    payload, _, containment = _strict_relationship_turn(client)
    repo = relationship_repository()
    original = repo.diagnostics

    def changed_after_read(**kwargs):
        result = original(**kwargs)
        state = runtime_state_repository()
        if change == "completion":
            state.complete_turn(
                **{key: payload[key] for key in (
                    "request_id", "runtime_session_id", "runtime_turn_id",
                )}, turn_status="completed",
            )
        elif change == "revision":
            with state._connect() as conn:
                conn.execute("UPDATE conversation_runtime_threads SET revision = revision + 1")
        elif change == "containment":
            from models import PersonaContainmentEvaluateRequest
            from services.persona_containment import evaluate_persona_containment

            evaluate_persona_containment(PersonaContainmentEvaluateRequest(
                **{**containment, "current_user_text": "What is 2+2?"},
            ))
        else:
            from services.companion_contracts import companion_contracts_repository

            with companion_contracts_repository()._connect() as conn:
                conn.execute(
                    "UPDATE surface_bindings SET surface_type = 'ide_extension' "
                    "WHERE surface_id = 'web'",
                )
        return result

    monkeypatch.setattr(repo, "diagnostics", changed_after_read)
    _assert_relationship_authority_failure(client.post("/v1/relationships/select", json=payload))


def test_legacy_relationship_calls_retain_existing_independent_persona_behavior():
    client = TestClient(app)
    _seed_strict_relationships(client)
    response = client.post("/v1/relationships/select", json={
        **_base(), "surface": "web", "active_persona_id": "personal_companion",
        "requested_scopes": ["personal_context"],
    })
    assert response.status_code == 200
    result = response.json()
    assert result["selection_contract"] == "legacy_unbound"
    assert result["persona_selection_ref"] is None
    assert result["retrieval_scope_projection"]["relationship_ids"] == ["rel-private-hidden"]


@pytest.mark.parametrize("edge,reason,confirmation", [
    ({"sensitivity_level": "restricted"}, "authorization_required", True),
    ({"mentionability": "suppress_by_default"}, "suppressed_by_default", False),
    ({"mentionability": "confirm_before_mentioning"}, "authorization_required", True),
    ({"status": "revoked"}, "status_revoked", False),
    ({"confidence": 0.5, "source_type": "tool_output"}, "below_confidence_threshold", False),
    ({"blocked_persona_scopes_json": ["general_assistant"]}, "blocked_persona_scope", False),
    ({"valid_until": _iso(-100)}, "expired", False),
])
def test_strict_relationships_preserve_independent_safety_checks(edge, reason, confirmation):
    client = TestClient(app)
    _seed_entities(client)
    inserted = client.post("/v1/relationships/edges/upsert", json={
        **_base(), "edge": _edge(relationship_id="rel-safety", **edge),
    })
    assert inserted.status_code == 200
    payload, _, _ = _strict_relationship_turn(client)
    response = client.post("/v1/relationships/select", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["selected_relationships"] == []
    assert result["prompt_content"] is None
    assert result["retrieval_scope_projection"]["relationship_ids"] == []
    assert result["trace"]["relationship_exclusion_reasons"]["rel-safety"] == reason
    assert result["trace"]["relationship_confirmation_required"] == confirmation


def test_strict_relationship_conflicts_never_choose_a_winner():
    client = TestClient(app)
    _seed_entities(client)
    for reference, target in (("rel-conflict-a", "repo:alpha"), ("rel-conflict-b", "repo:beta")):
        assert client.post("/v1/relationships/edges/upsert", json={
            **_base(), "edge": _edge(
                relationship_id=reference, relationship_type="bound_to", object_entity_id=target,
            ),
        }).status_code == 200
    payload, _, _ = _strict_relationship_turn(client)
    response = client.post("/v1/relationships/select", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["selected_relationships"] == []
    assert result["prompt_content"] is None
    assert result["trace"]["relationship_confirmation_required"] is True
    assert set(result["trace"]["relationship_conflicts"]) == {"rel-conflict-a", "rel-conflict-b"}
    assert result["retrieval_scope_projection"]["applied"] is False


@pytest.mark.parametrize("corruption", ["missing", "empty", "unsupported"])
def test_strict_relationship_surface_authority_fails_before_data(corruption, monkeypatch):
    from services.companion_contracts import companion_contracts_repository

    client = TestClient(app)
    payload, _, _ = _strict_relationship_turn(client)
    with companion_contracts_repository()._connect() as conn:
        if corruption == "missing":
            conn.execute("DELETE FROM surface_bindings WHERE surface_id IN ('web', 'unknown')")
        elif corruption == "empty":
            conn.execute("UPDATE surface_bindings SET surface_type = '' WHERE surface_id = 'web'")
        else:
            conn.execute(
                "UPDATE surface_bindings SET surface_type = 'unsupported_surface' "
                "WHERE surface_id = 'web'",
            )
    reads = []
    monkeypatch.setattr(relationship_repository(), "diagnostics", lambda **kw: reads.append(kw))
    _assert_relationship_authority_failure(client.post("/v1/relationships/select", json=payload))
    assert reads == []


def test_strict_containment_must_precede_relationship_read_and_projection(monkeypatch):
    import services.relationships as module

    client = TestClient(app)
    _seed_strict_relationships(client)
    payload, _, _ = _strict_relationship_turn(client)
    events = runtime_state_repository().list_events_for_tests(payload["runtime_session_id"])
    types = [event.event_type for event in events]
    assert types.index("persona_selection_resolved") < types.index("persona_containment_evaluated")
    order = []
    original_authority = module._strict_relationship_authority
    original_read = relationship_repository().diagnostics
    original_projection = module._retrieval_scope_projection

    def authority(body):
        order.append("authority")
        return original_authority(body)

    def read(**kwargs):
        order.append("protected_read")
        return original_read(**kwargs)

    def projection(selected):
        order.append("projection")
        return original_projection(selected)

    monkeypatch.setattr(module, "_strict_relationship_authority", authority)
    monkeypatch.setattr(relationship_repository(), "diagnostics", read)
    monkeypatch.setattr(module, "_retrieval_scope_projection", projection)
    assert client.post("/v1/relationships/select", json=payload).status_code == 200
    assert order == ["authority", "protected_read", "projection", "authority"]


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
    for entity_id, label, entity_type in (
        ("project:suppressed", "suppressed project marker", "project"),
        ("repo:suppressed", "suppressed repo marker", "repository"),
    ):
        client.post(
            "/v1/relationships/entities/upsert",
            json={
                **_base(),
                "entity": _entity(
                    entity_id,
                    label=label,
                    entity_type=entity_type,
                    domain="operations_context",
                ),
            },
        )
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
    suppressed = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                relationship_id="rel_suppressed_mixed",
                subject_entity_id="project:suppressed",
                object_entity_id="repo:suppressed",
                relationship_type="maintains",
                relationship_scope="operations_context",
                mentionability="suppress_by_default",
            ),
            "evidence": [],
        },
    ).json()["relationship"]

    response = client.post(
        "/v1/relationships/select",
        json={
            **_base(),
            "active_persona_id": "technical_architect",
            "requested_scopes": ["project_context", "operations_context"],
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
    assert suppressed["relationship_id"] not in ids
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
    assert suppressed["relationship_id"] not in projection["relationship_ids"]
    assert "project:suppressed" not in projection["entity_ids"]
    assert "repo:suppressed" not in projection["entity_ids"]
    assert "operations_context" not in projection["relationship_scopes"]
    assert low_conf["relationship_id"] in body["trace"]["relationship_edges_excluded"]
    assert (
        body["trace"]["relationship_exclusion_reasons"][low_conf["relationship_id"]]
        == "below_confidence_threshold"
    )
    assert (
        body["trace"]["relationship_exclusion_reasons"][restricted["relationship_id"]]
        == "authorization_required"
    )
    assert suppressed["relationship_id"] in body["trace"]["relationship_edges_excluded"]
    assert (
        body["trace"]["relationship_exclusion_reasons"][suppressed["relationship_id"]]
        == "suppressed_by_default"
    )
    assert suppressed["relationship_id"] not in body["trace"]["relationship_edges_used"]


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


def test_suppress_by_default_relationship_is_excluded_from_retrieval_projection():
    client = TestClient(app)
    for entity_id, label, entity_type in (
        ("project:suppress-only", "suppress only project marker", "project"),
        ("repo:suppress-only", "suppress only repo marker", "repository"),
    ):
        client.post(
            "/v1/relationships/entities/upsert",
            json={
                **_base(),
                "entity": _entity(
                    entity_id,
                    label=label,
                    entity_type=entity_type,
                    domain="operations_context",
                ),
            },
        )
    suppressed = client.post(
        "/v1/relationships/edges/upsert",
        json={
            **_base(),
            "edge": _edge(
                relationship_id="rel_suppress_only",
                subject_entity_id="project:suppress-only",
                object_entity_id="repo:suppress-only",
                relationship_type="maintains",
                relationship_scope="operations_context",
                mentionability="suppress_by_default",
            ),
            "evidence": [],
        },
    ).json()["relationship"]

    response = client.post(
        "/v1/relationships/select",
        json={
            **_base(),
            "active_persona_id": "technical_architect",
            "requested_scopes": ["operations_context"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    relationship_id = suppressed["relationship_id"]
    assert body["selected_relationships"] == []
    assert body["prompt_content"] is None
    assert body["retrieval_scope_projection"] == {
        "applied": False,
        "relationship_ids": [],
        "entity_ids": [],
        "relationship_scopes": [],
        "reason_codes": ["no_eligible_relationship_scope"],
    }
    excluded_ids = {
        item["relationship_id"]
        for item in body["excluded_relationship_summaries"]
    }
    assert relationship_id in excluded_ids
    assert relationship_id in body["trace"]["relationship_edges_excluded"]
    assert (
        body["trace"]["relationship_exclusion_reasons"][relationship_id]
        == "suppressed_by_default"
    )
    assert relationship_id not in body["trace"]["relationship_edges_used"]


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
