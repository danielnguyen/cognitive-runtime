from fastapi.testclient import TestClient
from main import app
from services.companion_contracts import companion_contracts_repository


def _base():
    return {
        "request_id": "rid-diagnostic",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
    }


def test_scene_resolve_prefers_requested_scene_and_records_event():
    client = TestClient(app)

    response = client.post(
        "/v1/companion/scene/resolve",
        json={**_base(), "requested_scene": "planning"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "planning"
    assert body["scene_source"] == "requested_scene"
    assert body["scene_version"] == 1

    events = companion_contracts_repository().list_scene_resolution_events_for_tests()
    assert len(events) == 1
    assert events[0]["resolved_scene_id"] == "planning"
    assert events[0]["source"] == "requested_scene"


def test_scene_resolve_aliases_to_canonical_scene():
    client = TestClient(app)

    response = client.post(
        "/v1/companion/scene/resolve",
        json={**_base(), "requested_scene": "coding"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "coding_build"
    assert body["signals_json"]["alias_used"] is True


def test_scene_resolve_uses_runtime_scene_only_when_requested_scene_absent():
    client = TestClient(app)
    client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"active_scene": "planning"}},
    )

    response = client.post("/v1/companion/scene/resolve", json=_base())

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "planning"
    assert body["scene_source"] == "runtime_state"


def test_scene_resolve_unknown_requested_and_runtime_scene_fall_back_to_general():
    client = TestClient(app)
    client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"active_scene": "not_a_scene"}},
    )

    requested = client.post(
        "/v1/companion/scene/resolve",
        json={**_base(), "requested_scene": "unknown_scene"},
    )
    runtime_only = client.post("/v1/companion/scene/resolve", json=_base())

    assert requested.status_code == 200
    assert runtime_only.status_code == 200
    assert requested.json()["scene_id"] == "general"
    assert requested.json()["scene_source"] == "fallback_general"
    assert runtime_only.json()["scene_id"] == "general"
    assert runtime_only.json()["scene_source"] == "fallback_general"


def test_scene_detail_returns_canonical_policy_for_alias_and_unknown_is_404():
    client = TestClient(app)

    alias_response = client.get("/v1/companion/scene/coding")
    missing_response = client.get("/v1/companion/scene/not_a_scene")

    assert alias_response.status_code == 200
    assert alias_response.json()["scene_id"] == "coding_build"
    assert missing_response.status_code == 404


def test_interaction_validation_flags_pressure_and_exclusivity():
    client = TestClient(app)

    response = client.post(
        "/v1/companion/interaction-contract/validate",
        json={
            **_base(),
            "text": "After all I've done, don't leave me. Only I understand you.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["diagnostic_only"] is True
    assert body["result"] == "fail"
    assert "guilt_pressure" in body["warnings"]
    assert "exclusivity_framing" in body["warnings"]


def test_interaction_validation_flags_unsupported_memory_claim_and_poor_repair():
    client = TestClient(app)

    response = client.post(
        "/v1/companion/interaction-contract/validate",
        json={
            **_base(),
            "text": "I remember from our past chats, and sorry, sorry about that.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result"] == "warn"
    assert "unsupported_memory_claim" in body["warnings"]
    assert "apology_loop" in body["warnings"]


def test_interaction_validation_permits_useful_disagreement_and_task_relevant_memory():
    client = TestClient(app)
    text = (
        "I disagree with that approach. Based on what you mentioned earlier in this "
        "conversation, the safer next step is to revert the config change."
    )

    response = client.post(
        "/v1/companion/interaction-contract/validate",
        json={**_base(), "text": text},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["diagnostic_only"] is True
    assert body["result"] == "pass"
    assert body["warnings"] == []


def test_repair_simulation_is_correction_first_and_records_event():
    client = TestClient(app)

    response = client.post(
        "/v1/companion/repair/simulate",
        json={
            **_base(),
            "miss_description": "the dependency order",
            "corrected_substance": "The correct dependency order is API first, then worker.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["diagnostic_only"] is True
    assert body["repair_text"].startswith(
        "The correct dependency order is API first, then worker."
    )
    assert "sorry" not in body["repair_text"].lower()

    events = companion_contracts_repository().list_interaction_boundary_events_for_tests()
    assert len(events) == 1
    assert events[0]["check_type"] == "repair_simulation"


def test_validation_event_uses_bounded_input_summary_not_full_text():
    client = TestClient(app)
    long_text = "Only I understand you. " + ("x" * 400)

    response = client.post(
        "/v1/companion/interaction-contract/validate",
        json={**_base(), "text": long_text},
    )

    assert response.status_code == 200
    events = companion_contracts_repository().list_interaction_boundary_events_for_tests()
    assert len(events) == 1
    assert len(events[0]["input_summary"]) <= 240
    assert events[0]["input_summary"] != long_text


def test_compile_and_interrupt_do_not_record_interaction_boundary_events():
    client = TestClient(app)

    compile_response = client.post("/v1/companion/policy/compile", json=_base())
    interrupt_response = client.post(
        "/v1/interrupt/evaluate",
        json={
            **_base(),
            "current_user_text": "Should I split the module or keep it as is?",
        },
    )

    assert compile_response.status_code == 200
    assert interrupt_response.status_code == 200
    assert companion_contracts_repository().list_interaction_boundary_events_for_tests() == []
