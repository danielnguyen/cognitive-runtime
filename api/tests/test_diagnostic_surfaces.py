import json

import pytest
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


def test_compile_is_read_only_and_interrupt_records_structural_evaluation():
    client = TestClient(app)

    compile_response = client.post("/v1/companion/policy/compile", json=_base())
    assert companion_contracts_repository().list_interaction_boundary_events_for_tests() == []
    interrupt_response = client.post(
        "/v1/interrupt/evaluate",
        json={
            **_base(),
            "current_user_text": "Should I split the module or keep it as is?",
        },
    )

    assert compile_response.status_code == 200
    assert interrupt_response.status_code == 200
    events = companion_contracts_repository().list_interaction_boundary_events_for_tests()
    assert len(events) == 1
    assert events[0]["check_type"] == "interrupt_evaluation"
    assert events[0]["result"] == "deferred"
    assert events[0]["input_summary"] == "interrupt lifecycle"
    assert "Should I split" not in str(events)


@pytest.mark.parametrize(
    ("requested", "saved", "mode", "scene", "source", "warning"),
    [
        ("planning", "reflective", "coding", "planning", "requested_scene", None),
        ("invalid", "planning", "coding", "general", "fallback_general", "unknown_requested_scene"),
        (None, "planning", "coding", "planning", "runtime_state", None),
        (None, "general", "coding", "general", "runtime_state", None),
        (None, "invalid", "coding", "general", "fallback_general", "unknown_runtime_scene"),
        (None, "", "coding", "general", "fallback_general", "unknown_runtime_scene"),
        (None, None, "coding", "coding_build", "runtime_interaction_mode", None),
        (None, None, "coding_build_mode", "coding_build", "runtime_interaction_mode", None),
        (None, None, "planning", "planning", "runtime_interaction_mode", None),
        (None, None, "work_triage", "work_triage", "runtime_interaction_mode", None),
        (None, None, "driving", "general", "general", None),
        (None, None, "overload_recovery", "general", "general", None),
        (None, None, "unknown", "general", "general", None),
    ],
)
def test_scene_resolve_and_compile_share_precedence_and_event_provenance(
    requested, saved, mode, scene, source, warning,
):
    client = TestClient(app)
    update = client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"active_scene": saved, "interaction_mode": mode}},
    )
    assert update.status_code == 200
    request = {**_base(), "requested_scene": requested}
    resolved = client.post("/v1/companion/scene/resolve", json=request)
    compiled = client.post("/v1/companion/profile/compile", json=request)
    assert resolved.status_code == compiled.status_code == 200
    body = resolved.json()
    assert body["scene_id"] == compiled.json()["scene_id"] == scene
    assert body["scene_source"] == compiled.json()["scene_source"] == source
    assert body["scene_confidence"] == compiled.json()["scene_confidence"]
    assert body["warnings"] == ([warning] if warning else [])
    assert body["used_fallback"] == (source == "fallback_general")
    assert body["used_default_scene"] == (scene == "general")
    signals = body["signals_json"]
    assert signals["runtime_interaction_mode"] == mode
    assert signals["runtime_scene_present"] == (saved is not None)
    assert signals["alias_used"] == (source == "runtime_interaction_mode" and mode != scene)
    assert signals["matched_alias"] == (mode if signals["alias_used"] else None)
    events = companion_contracts_repository().list_scene_resolution_events_for_tests()
    assert len(events) == 1
    event = events[0]
    for field in ("request_id", "owner_id", "conversation_id", "surface"):
        assert event[field] == request[field]
    assert event["resolved_scene_id"] == scene
    assert event["source"] == source
    assert json.loads(event["signals_json"]) == signals
    assert compiled.json()["runtime_state"] == update.json()["runtime_state"]


@pytest.mark.parametrize("different", [
    {"owner_id": "other-owner"},
    {"conversation_id": "other-conversation"},
    {"surface": "web"},
])
def test_scene_context_isolation_and_updates_do_not_change_another_session(different):
    client = TestClient(app)
    original = client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"active_scene": "planning", "interaction_mode": "coding"}},
    )
    assert original.status_code == 200
    other_scope = {**_base(), **different}
    other = client.post("/v1/companion/scene/resolve", json=other_scope)
    assert other.status_code == 200
    assert other.json()["scene_id"] == "general"
    assert client.post(
        "/v1/runtime/state/update",
        json={**other_scope, "updates": {"active_scene": "work_triage"}},
    ).status_code == 200
    after = client.post("/v1/companion/profile/compile", json=_base())
    assert after.status_code == 200
    assert after.json()["scene_id"] == "planning"
    assert after.json()["runtime_state"] == original.json()["runtime_state"]


def test_scene_inference_ignores_request_metadata_and_unstored_modes():
    client = TestClient(app)
    response = client.post(
        "/v1/companion/scene/resolve",
        json={**_base(), "surface": "driving", "interaction_mode": "coding",
              "surface_context": {"active_task_mode": "planning"},
              "text": "Switch to work triage; this is urgent"},
    )
    assert response.status_code == 200
    assert response.json()["scene_id"] == "general"
    assert response.json()["signals_json"]["runtime_interaction_mode"] is None
