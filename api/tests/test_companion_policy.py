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
    assert body["profile_id"] == "default_companion_profile"
    assert body["profile_version"] == 1
    assert body["contract_id"] == "default_interaction_contract"
    assert body["contract_version"] == 2
    assert body["scene_id"] == "general"
    assert body["scene_source"] == "general"
    assert body["warnings"] == ["default_contract_applied"]
    assert [overlay["overlay_type"] for overlay in body["overlays"]] == [
        "interaction_contract",
        "companion_profile",
        "scene_policy",
    ]
    for overlay in body["overlays"]:
        assert overlay["role"] == "system"
        assert 0 < len(overlay["content"]) <= 1200


def test_compile_returns_structured_interaction_contract():
    client = TestClient(app)

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    contract = response.json()["interaction_contract"]
    assert contract["contract_id"] == "default_interaction_contract"
    assert contract["contract_version"] == 2
    assert contract["owner_id"] == "owner"
    assert contract["scope"] == "global_default"
    assert contract["source"] == "default_compiled"
    for field in (
        "trust_rules",
        "interaction_boundaries",
        "repair_rules",
        "memory_or_recall_boundaries",
        "autonomy_rules",
        "tone_constraints",
        "allowed_intervention_styles",
        "disallowed_intervention_styles",
        "defer_conditions",
    ):
        assert contract[field]
    assert "perform_closeness" not in contract["allowed_intervention_styles"]
    assert "performative_memory" in contract["disallowed_intervention_styles"]


def test_compile_returns_inspectable_contract_trace():
    client = TestClient(app)

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    trace = response.json()["contract_trace"]
    assert trace["contract_id"] == "default_interaction_contract"
    assert trace["contract_version"] == 2
    assert trace["source"] == "default_compiled"
    assert trace["scope"] == "global_default"
    assert trace["warnings"] == ["default_contract_applied"]
    assert trace["selected_rule_groups"] == [
        "trust_rules",
        "interaction_boundaries",
        "repair_rules",
        "memory_or_recall_boundaries",
        "autonomy_rules",
        "tone_constraints",
        "allowed_intervention_styles",
        "disallowed_intervention_styles",
        "defer_conditions",
    ]
    assert trace["selected_boundary_rules"]
    assert trace["selected_repair_rules"]


def test_interaction_contract_overlay_is_concise_and_not_interrupt_execution():
    client = TestClient(app)

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    overlay = response.json()["overlays"][0]
    assert overlay["overlay_type"] == "interaction_contract"
    content = overlay["content"]
    assert "Interaction contract:" in content
    assert "Memory:" in content
    assert "Repair:" in content
    blocked_phrases = (
        "detect loops",
        "interrupt when",
        "trigger class",
        "grounding trigger",
        "speculation-loop",
    )
    assert all(phrase not in content.lower() for phrase in blocked_phrases)


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
    assert body["warnings"] == ["default_contract_applied"]
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
        assert body["warnings"] == ["default_contract_applied"]


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
    assert body["warnings"] == ["default_contract_applied"]


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
    assert body["warnings"] == ["unknown_requested_scene", "default_contract_applied"]
    assert body["contract_trace"]["warnings"] == [
        "default_contract_applied",
        "unknown_requested_scene",
    ]


def test_unknown_surface_falls_back_to_default_contract_with_warning():
    client = TestClient(app)

    response = client.post(
        "/v1/companion/policy/compile",
        json={**_base(), "surface": "new_surface"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["interaction_contract"]["source"] == "default_compiled"
    assert body["contract_trace"]["warnings"] == [
        "default_contract_applied",
        "unknown_surface_default_contract",
    ]
    assert body["warnings"] == [
        "default_contract_applied",
        "unknown_surface_default_contract",
    ]


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
    assert body["warnings"] == ["default_contract_applied"]


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
    assert body["warnings"] == ["unknown_runtime_scene", "default_contract_applied"]
    assert body["contract_trace"]["warnings"] == [
        "default_contract_applied",
        "unknown_runtime_scene",
    ]


def test_compile_output_is_deterministic():
    client = TestClient(app)
    payload = {**_base(), "requested_scene": "work_triage"}

    first = client.post("/v1/companion/policy/compile", json=payload)
    second = client.post("/v1/companion/policy/compile", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["overlays"] == second.json()["overlays"]
    assert first.json()["interaction_contract"] == second.json()["interaction_contract"]
    assert first.json()["contract_trace"] == second.json()["contract_trace"]
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


def test_active_profile_endpoint_returns_seeded_default_record():
    client = TestClient(app)

    response = client.get("/v1/companion/profile/active")

    assert response.status_code == 200
    body = response.json()
    assert body["profile_id"] == "default_companion_profile"
    assert body["profile_version"] == 1
    assert body["scope"] == "global_default"
    assert body["source"] == "seeded_default"
    assert body["status"] == "active"
    assert body["role_label"] == "personal_intelligence_companion"
    assert body["core_traits_json"]["directness"] == "high"


def test_profile_compile_endpoint_matches_policy_compile_alias():
    client = TestClient(app)
    payload = {**_base(), "requested_scene": "planning"}

    profile_response = client.post("/v1/companion/profile/compile", json=payload)
    policy_response = client.post("/v1/companion/policy/compile", json=payload)

    assert profile_response.status_code == 200
    assert policy_response.status_code == 200
    assert profile_response.json() == policy_response.json()


def test_compile_uses_seeded_general_scene_for_unknown_scene():
    client = TestClient(app)

    response = client.post(
        "/v1/companion/profile/compile",
        json={**_base(), "requested_scene": "unknown_scene"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "general"
    assert body["scene_source"] == "fallback_general"
    assert "Scene policy: use the general operating mode" in body["overlays"][2]["content"]
    assert body["profile_version"] == 1
    assert body["contract_version"] == 2
