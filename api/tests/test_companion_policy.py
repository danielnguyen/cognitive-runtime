from fastapi.testclient import TestClient

from main import app
from services.runtime_state import clear_states_for_tests


def setup_function():
    clear_states_for_tests()


def _base():
    return {
        "request_id": "rid-companion",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
    }


def test_compile_returns_bounded_ordered_policy_overlays():
    client = TestClient(app)

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    body = response.json()
    assert body["profile_id"] == "companion_profile_r17_mvp"
    assert body["profile_version"] == 1
    assert body["contract_id"] == "interaction_contract_r19_mvp"
    assert body["contract_version"] == 1
    assert body["scene_id"] == "general"
    assert body["scene_source"] == "general"
    assert body["warnings"] == []
    assert [overlay["overlay_type"] for overlay in body["overlays"]] == [
        "interaction_contract",
        "companion_profile",
        "scene_policy",
    ]
    for overlay in body["overlays"]:
        assert overlay["role"] == "system"
        assert 0 < len(overlay["content"]) <= 1200


def test_explicit_scene_resolves_scene_policy():
    client = TestClient(app)

    response = client.post(
        "/v1/companion/policy/compile",
        json={**_base(), "requested_scene": "planning"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "planning"
    assert body["scene_confidence"] == 1.0
    assert body["scene_source"] == "requested_scene"
    assert body["warnings"] == []
    assert "Scene policy: planning" in body["overlays"][2]["content"]


def test_requested_scene_aliases_return_canonical_scene_ids():
    client = TestClient(app)

    aliases = {
        "coding": "coding_build",
        "coding_build_mode": "coding_build",
        "reflective_conversation": "reflective",
        "notifications_briefings": "briefing",
    }
    for alias, canonical in aliases.items():
        response = client.post(
            "/v1/companion/policy/compile",
            json={**_base(), "requested_scene": alias},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["scene_id"] == canonical
        assert body["scene_source"] == "requested_scene"
        assert body["warnings"] == []


def test_runtime_scene_alias_returns_canonical_scene_id():
    client = TestClient(app)
    client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"active_scene": "coding_build_mode"}},
    )

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "coding_build"
    assert body["scene_source"] == "runtime_state"
    assert body["warnings"] == []


def test_unknown_requested_scene_falls_back_without_using_runtime_scene():
    client = TestClient(app)
    client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"active_scene": "planning"}},
    )

    response = client.post(
        "/v1/companion/policy/compile",
        json={**_base(), "requested_scene": "not_a_scene"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "general"
    assert body["scene_source"] == "fallback_general"
    assert body["warnings"] == ["unknown_requested_scene"]


def test_runtime_scene_used_only_when_requested_scene_absent():
    client = TestClient(app)
    client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"active_scene": "planning"}},
    )

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "planning"
    assert body["scene_source"] == "runtime_state"
    assert body["warnings"] == []


def test_unknown_runtime_scene_falls_back_with_warning():
    client = TestClient(app)
    client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"active_scene": "not_a_scene"}},
    )

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "general"
    assert body["scene_source"] == "fallback_general"
    assert body["warnings"] == ["unknown_runtime_scene"]


def test_compile_output_is_deterministic():
    client = TestClient(app)
    payload = {**_base(), "requested_scene": "work_triage"}

    first = client.post("/v1/companion/policy/compile", json=payload)
    second = client.post("/v1/companion/policy/compile", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["overlays"] == second.json()["overlays"]
    assert first.json()["scene_id"] == second.json()["scene_id"]


def test_structurally_invalid_requested_scene_is_rejected():
    client = TestClient(app)

    for value in ({"scene": "planning"}, ["planning"], 123):
        response = client.post(
            "/v1/companion/policy/compile",
            json={**_base(), "requested_scene": value},
        )
        assert response.status_code == 422


def test_oversized_requested_scene_is_rejected():
    client = TestClient(app)

    response = client.post(
        "/v1/companion/policy/compile",
        json={**_base(), "requested_scene": "x" * 65},
    )

    assert response.status_code == 422
