from pathlib import Path

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


def test_runtime_session_resolution_is_stable_and_records_event():
    client = TestClient(app)
    payload = {
        "request_id": "rid-session",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
        "active_mode": "actionable",
    }

    first = client.post("/v1/runtime/sessions/resolve", json=payload)
    second = client.post("/v1/runtime/sessions/resolve", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    runtime_session_id = first.json()["runtime_session"]["runtime_session_id"]
    assert runtime_session_id == second.json()["runtime_session"]["runtime_session_id"]

    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    body = diagnostics.json()
    assert body["runtime_session"]["status"] == "active"
    assert body["events"][0]["event_type"] == "session_resolved"


def test_turn_lifecycle_is_durable_and_inspectable(tmp_path: Path):
    runtime_db_path = tmp_path / "runtime" / "runtime_state.sqlite3"
    clear_states_for_tests(db_path=runtime_db_path)
    client = TestClient(app)

    started = client.post(
        "/v1/runtime/turns/start",
        json={
            "request_id": "rid-turn-1",
            "owner_id": "owner",
            "conversation_id": "conv-1",
            "surface": "dev",
            "input_message_id": "msg-1",
            "intent_class": "task",
        },
    )
    assert started.status_code == 200
    started_body = started.json()
    runtime_session_id = started_body["runtime_session"]["runtime_session_id"]
    runtime_turn_id = started_body["runtime_turn"]["runtime_turn_id"]
    assert started_body["event"]["event_type"] == "turn_started"

    updated = client.post(
        "/v1/runtime/turns/update",
        json={
            "request_id": "rid-turn-2",
            "runtime_session_id": runtime_session_id,
            "runtime_turn_id": runtime_turn_id,
            "turn_status": "retrieving",
            "timing_policy": "normal",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["runtime_turn"]["turn_status"] == "retrieving"

    completed = client.post(
        "/v1/runtime/turns/complete",
        json={
            "request_id": "rid-turn-3",
            "runtime_session_id": runtime_session_id,
            "runtime_turn_id": runtime_turn_id,
            "turn_status": "completed",
            "continuation_state": "none",
        },
    )
    assert completed.status_code == 200
    assert completed.json()["runtime_turn"]["turn_status"] == "completed"

    clear_states_for_tests(db_path=runtime_db_path)
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    body = diagnostics.json()
    assert body["latest_turn"]["runtime_turn_id"] == runtime_turn_id
    assert body["latest_turn"]["turn_status"] == "completed"
    assert [event["event_type"] for event in body["events"]] == [
        "session_resolved",
        "turn_started",
        "turn_updated",
        "turn_completed",
    ]


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
