from fastapi.testclient import TestClient
from main import app
from services.runtime_state import clear_states_for_tests


def setup_function():
    clear_states_for_tests()


def test_resolve_creates_and_reuses_runtime_state():
    client = TestClient(app)
    payload = {
        "request_id": "rid-1",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
    }

    first = client.post("/v1/runtime/state/resolve", json=payload)
    second = client.post("/v1/runtime/state/resolve", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    first_state = first.json()["runtime_state"]
    second_state = second.json()["runtime_state"]
    assert first_state["runtime_state_id"] == second_state["runtime_state_id"]
    assert first_state["temporary_constraints"] == []


def test_update_and_overlay_are_bounded_and_mechanical():
    client = TestClient(app)
    base = {
        "request_id": "rid-2",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
    }

    update = client.post(
        "/v1/runtime/state/update",
        json={
            **base,
            "updates": {
                "active_scene": "planning",
                "interaction_mode": "actionable",
                "temporary_constraints": ["preserve_flow"],
                "reset_after_turn": True,
            },
        },
    )
    overlay = client.post("/v1/runtime/overlay", json=base)

    assert update.status_code == 200
    assert overlay.status_code == 200
    body = overlay.json()
    assert body["omitted"] is False
    assert body["overlay"]["content"] == (
        "Runtime context: scene=planning; interaction_mode=actionable; "
        "constraints=preserve_flow."
    )
    assert "Prefer" not in body["overlay"]["content"]
    assert "preserve flow" not in body["overlay"]["content"]
    assert body["runtime_state"]["reset_after_turn"] is True


def test_overlay_is_omitted_for_empty_state():
    client = TestClient(app)
    response = client.post(
        "/v1/runtime/overlay",
        json={
            "request_id": "rid-3",
            "owner_id": "owner",
            "conversation_id": "conv-1",
            "surface": "chat",
        },
    )

    assert response.status_code == 200
    assert response.json()["overlay"] is None
    assert response.json()["omission_reason"] == "empty_runtime_state"


def test_reset_clears_runtime_fields():
    client = TestClient(app)
    base = {
        "request_id": "rid-4",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
    }
    client.post(
        "/v1/runtime/state/update",
        json={
            **base,
            "updates": {
                "active_scene": "planning",
                "interaction_mode": "actionable",
                "temporary_constraints": ["preserve_flow"],
                "reset_after_turn": True,
                "trace_refs": ["rid-4"],
            },
        },
    )

    reset = client.post("/v1/runtime/state/reset", json={**base, "reason": "test"})

    assert reset.status_code == 200
    state = reset.json()["runtime_state"]
    assert state["active_scene"] is None
    assert state["interaction_mode"] is None
    assert state["temporary_constraints"] == []
    assert state["reset_after_turn"] is False
    assert state["trace_refs"] == []


def test_explicit_null_update_fields_are_ignored():
    client = TestClient(app)
    base = {
        "request_id": "rid-5",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
    }
    client.post(
        "/v1/runtime/state/update",
        json={
            **base,
            "updates": {
                "temporary_constraints": ["preserve_flow"],
                "trace_refs": ["rid-5"],
                "reset_after_turn": True,
            },
        },
    )

    update = client.post(
        "/v1/runtime/state/update",
        json={
            **base,
            "updates": {
                "temporary_constraints": None,
                "trace_refs": None,
                "reset_after_turn": None,
            },
        },
    )

    assert update.status_code == 200
    state = update.json()["runtime_state"]
    assert state["temporary_constraints"] == ["preserve_flow"]
    assert state["trace_refs"] == ["rid-5"]
    assert state["reset_after_turn"] is True


def test_oversized_constraint_labels_are_rejected():
    client = TestClient(app)
    response = client.post(
        "/v1/runtime/state/update",
        json={
            "request_id": "rid-6",
            "owner_id": "owner",
            "conversation_id": "conv-1",
            "surface": "dev",
            "updates": {"temporary_constraints": ["x" * 65]},
        },
    )

    assert response.status_code == 422


def test_oversized_trace_refs_are_rejected():
    client = TestClient(app)
    response = client.post(
        "/v1/runtime/state/update",
        json={
            "request_id": "rid-7",
            "owner_id": "owner",
            "conversation_id": "conv-1",
            "surface": "dev",
            "updates": {"trace_refs": ["x" * 121]},
        },
    )

    assert response.status_code == 422
