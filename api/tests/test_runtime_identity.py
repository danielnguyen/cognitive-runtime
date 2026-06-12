from fastapi.testclient import TestClient

from main import app
from services.runtime_state import clear_states_for_tests


def setup_function():
    clear_states_for_tests()


def _base(surface: str = "vscode"):
    return {
        "request_id": "rid-identity",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": surface,
    }


def test_identity_resolution_uses_surface_binding_default_persona():
    client = TestClient(app)

    response = client.post("/v1/runtime/identity/resolve", json=_base("vscode"))

    assert response.status_code == 200
    body = response.json()
    assert body["surface_binding"]["surface_id"] == "vscode"
    assert body["surface_binding"]["default_persona_id"] == "technical_architect"
    assert body["persona"]["persona_id"] == "technical_architect"
    assert body["trace"]["persona_resolution_reason"] == "surface_binding"
    assert body["trace"]["persona_override_source"] == "none"
    assert body["runtime_identity"]["persona_owns_durable_memory"] is False
    assert "persona=technical_architect" in body["runtime_identity"]["content"]


def test_identity_resolution_supports_internal_test_only_requested_persona():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/identity/resolve",
        json={**_base("web"), "requested_persona_id": "personal_companion"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["persona"]["persona_id"] == "personal_companion"
    assert body["trace"]["persona_resolution_reason"] == "requested_persona_id"
    assert body["trace"]["persona_override_source"] == "internal_test"


def test_identity_resolution_falls_back_to_unknown_surface_binding():
    client = TestClient(app)

    response = client.post("/v1/runtime/identity/resolve", json=_base("not_registered"))

    assert response.status_code == 200
    body = response.json()
    assert body["surface_binding"]["surface_id"] == "unknown"
    assert body["persona"]["persona_id"] == "general_assistant"
    assert body["trace"]["surface_id"] == "unknown"


def test_identity_resolution_records_event_in_runtime_session_diagnostics():
    client = TestClient(app)

    response = client.post("/v1/runtime/identity/resolve", json=_base("dev"))

    assert response.status_code == 200
    runtime_session_id = response.json()["runtime_session"]["runtime_session_id"]
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}")

    assert diagnostics.status_code == 200
    event_types = [event["event_type"] for event in diagnostics.json()["events"]]
    assert "identity_resolved" in event_types


def test_identity_resolution_does_not_introduce_world_state_or_relationship_fields():
    client = TestClient(app)

    response = client.post("/v1/runtime/identity/resolve", json=_base("dev"))

    assert response.status_code == 200
    payload_text = str(response.json())
    assert "world_state" not in payload_text
    assert "relationship_edges" not in payload_text
