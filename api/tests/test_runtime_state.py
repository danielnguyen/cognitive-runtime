import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import main as main_module
import pytest
from fastapi.testclient import TestClient
from main import app
from models import (
    ContinuationSelectionRequest,
    ContinuationSelectionResponse,
    ContinuationSelectionResult,
)
from pydantic import ValidationError
from services.runtime_state import RuntimeStateRepository, clear_states_for_tests


def setup_function():
    clear_states_for_tests()


def _use_database(tmp_path: Path, name: str = "runtime.sqlite3") -> Path:
    db_path = tmp_path / name
    clear_states_for_tests(db_path=db_path)
    return db_path


def _thread(client: TestClient, owner_id: str, conversation_id: str):
    return client.post(
        "/v1/runtime/threads/resolve",
        json={
            "request_id": "thread-read",
            "owner_id": owner_id,
            "conversation_id": conversation_id,
        },
    )


def _start(
    client: TestClient,
    *,
    request_id: str,
    owner_id: str,
    conversation_id: str,
    surface: str,
    expected_thread_revision: int | None = None,
):
    payload = {
        "request_id": request_id,
        "owner_id": owner_id,
        "conversation_id": conversation_id,
        "surface": surface,
    }
    if expected_thread_revision is not None:
        payload["expected_thread_revision"] = expected_thread_revision
    return client.post("/v1/runtime/turns/start", json=payload)


def _complete(
    client: TestClient,
    *,
    request_id: str,
    runtime_session_id: str,
    runtime_turn_id: str,
    turn_status: str = "completed",
):
    return client.post(
        "/v1/runtime/turns/complete",
        json={
            "request_id": request_id,
            "runtime_session_id": runtime_session_id,
            "runtime_turn_id": runtime_turn_id,
            "turn_status": turn_status,
        },
    )


def _runtime_rows(db_path: Path) -> tuple[tuple[tuple[object, ...], ...], ...]:
    with sqlite3.connect(db_path) as conn:
        return tuple(
            tuple(tuple(row) for row in conn.execute(query).fetchall())
            for query in (
                "SELECT * FROM conversation_runtime_sessions ORDER BY id;",
                "SELECT * FROM conversation_runtime_threads ORDER BY id;",
                "SELECT * FROM conversation_runtime_turns ORDER BY id;",
                "SELECT * FROM conversation_runtime_events ORDER BY id;",
            )
        )


def _continuation_candidate(
    conversation_id: str,
    *,
    lifecycle_state: str = "open",
    durable_updated_at: datetime | None = None,
) -> dict[str, object]:
    return {
        "conversation_id": conversation_id,
        "lifecycle_state": lifecycle_state,
        "durable_updated_at": (
            durable_updated_at or datetime.now(UTC)
        ).isoformat(),
    }


def _select_continuation(
    client: TestClient,
    *,
    candidates: list[dict[str, object]],
    owner_id: str = "owner-selection",
    surface: str = "surface-current",
    candidate_set_complete: bool = True,
    stale_after_seconds: int = 3600,
):
    return client.post(
        "/v1/runtime/continuations/select",
        json={
            "request_id": "continuation-selection-request",
            "owner_id": owner_id,
            "surface": surface,
            "candidate_set_complete": candidate_set_complete,
            "stale_after_seconds": stale_after_seconds,
            "candidates": candidates,
        },
    )


def _resolve_selection_session(
    client: TestClient,
    *,
    owner_id: str,
    conversation_id: str,
    surface: str = "surface-existing",
) -> dict[str, object]:
    response = client.post(
        "/v1/runtime/sessions/resolve",
        json={
            "request_id": f"resolve-{conversation_id}-{surface}",
            "owner_id": owner_id,
            "conversation_id": conversation_id,
            "surface": surface,
        },
    )
    assert response.status_code == 200
    return response.json()["runtime_session"]


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


def test_one_thread_projection_is_shared_by_distinct_surface_sessions(tmp_path: Path):
    db_path = _use_database(tmp_path)
    client = TestClient(app)
    owner_id = "owner-shared"
    conversation_id = "conversation-shared"

    first_session = client.post(
        "/v1/runtime/sessions/resolve",
        json={
            "request_id": "session-first",
            "owner_id": owner_id,
            "conversation_id": conversation_id,
            "surface": "surface-z",
            "surface_session_id": "transport-z",
        },
    )
    second_session = client.post(
        "/v1/runtime/sessions/resolve",
        json={
            "request_id": "session-second",
            "owner_id": owner_id,
            "conversation_id": conversation_id,
            "surface": "surface-a",
            "surface_session_id": "transport-a",
        },
    )
    assert first_session.status_code == second_session.status_code == 200
    assert (
        first_session.json()["runtime_session"]["runtime_session_id"]
        != second_session.json()["runtime_session"]["runtime_session_id"]
    )

    first_projection = _thread(client, owner_id, conversation_id)
    repeated_projection = _thread(client, owner_id, conversation_id)
    assert first_projection.status_code == repeated_projection.status_code == 200
    assert first_projection.json() == repeated_projection.json()
    assert first_projection.json()["participating_surfaces"] == ["surface-a", "surface-z"]
    assert first_projection.json()["participating_session_count"] == 2

    assert _thread(client, owner_id, "conversation-other").status_code == 200
    assert _thread(client, "owner-other", conversation_id).status_code == 200
    with sqlite3.connect(db_path) as conn:
        shared_count = conn.execute(
            """
            SELECT COUNT(*) FROM conversation_runtime_threads
            WHERE owner_id = ? AND conversation_id = ?;
            """,
            (owner_id, conversation_id),
        ).fetchone()[0]
        total_count = conn.execute(
            "SELECT COUNT(*) FROM conversation_runtime_threads;"
        ).fetchone()[0]
    assert shared_count == 1
    assert total_count == 3


@pytest.mark.parametrize(
    "path",
    [
        "/v1/runtime/sessions/resolve",
        "/v1/runtime/turns/start",
        "/v1/runtime/state/resolve",
    ],
)
def test_public_runtime_requests_reject_empty_surface_without_persistence(
    tmp_path: Path,
    path: str,
):
    db_path = _use_database(tmp_path)
    client = TestClient(app)

    response = client.post(
        path,
        json={
            "request_id": "empty-surface-request",
            "owner_id": "owner-empty-surface",
            "conversation_id": "conversation-empty-surface",
            "surface": "",
        },
    )

    assert response.status_code == 422
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_runtime_sessions;"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_runtime_threads;"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_runtime_turns;"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_runtime_events;"
        ).fetchone()[0] == 0


def test_legacy_empty_surface_remains_visible_in_thread_projection(tmp_path: Path):
    db_path = _use_database(tmp_path, "legacy-empty-surface.sqlite3")
    timestamp = "2026-01-01T00:00:00+00:00"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO conversation_runtime_sessions (
                runtime_session_id, runtime_state_id, owner_id, conversation_id,
                surface, surface_session_id, status, active_mode, attention_state,
                active_scene, interaction_mode, attention_focus_json,
                temporary_constraints_json, reset_after_turn, trace_refs_json,
                started_at, last_activity_at, closed_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, '', NULL, 'active', NULL, NULL, NULL, NULL,
                      'null', '[]', 0, '[]', ?, ?, NULL, ?, ?);
            """,
            (
                "legacy-empty-surface-session",
                "legacy-empty-surface-state",
                "owner-legacy-empty-surface",
                "conversation-legacy-empty-surface",
                timestamp,
                timestamp,
                timestamp,
                timestamp,
            ),
        )
        stored_before = conn.execute(
            """
            SELECT * FROM conversation_runtime_sessions
            WHERE runtime_session_id = 'legacy-empty-surface-session';
            """
        ).fetchone()

    response = _thread(
        TestClient(app),
        "owner-legacy-empty-surface",
        "conversation-legacy-empty-surface",
    )

    assert response.status_code == 200
    projection = response.json()
    assert projection["state"] == "idle"
    assert projection["participating_surfaces"] == [""]
    assert projection["participating_session_count"] == 1
    with sqlite3.connect(db_path) as conn:
        stored_after = conn.execute(
            """
            SELECT * FROM conversation_runtime_sessions
            WHERE runtime_session_id = 'legacy-empty-surface-session';
            """
        ).fetchone()
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_runtime_sessions;"
        ).fetchone()[0] == 1
    assert stored_after == stored_before


def test_cross_surface_admission_contends_without_mutation(tmp_path: Path):
    db_path = _use_database(tmp_path)
    client = TestClient(app)
    owner_id = "owner-contention"
    conversation_id = "conversation-contention"

    admitted = _start(
        client,
        request_id="admission-first",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-one",
    )
    assert admitted.status_code == 200
    before = _thread(client, owner_id, conversation_id).json()

    second_session = client.post(
        "/v1/runtime/sessions/resolve",
        json={
            "request_id": "session-while-active",
            "owner_id": owner_id,
            "conversation_id": conversation_id,
            "surface": "surface-two",
        },
    )
    assert second_session.status_code == 200
    after_session_resolution = _thread(client, owner_id, conversation_id).json()
    assert after_session_resolution["active_runtime_turn_id"] == before["active_runtime_turn_id"]
    assert after_session_resolution["revision"] == before["revision"]

    rejected = _start(
        client,
        request_id="admission-second",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-two",
    )
    assert (rejected.status_code, rejected.json()) == (
        409,
        {"detail": "runtime_thread_contended"},
    )
    after = _thread(client, owner_id, conversation_id).json()
    assert after["active_runtime_turn_id"] == before["active_runtime_turn_id"]
    assert after["active_surface"] == "surface-one"
    assert after["revision"] == before["revision"]
    with sqlite3.connect(db_path) as conn:
        turn_count = conn.execute(
            """
            SELECT COUNT(*) FROM conversation_runtime_turns AS turns
            JOIN conversation_runtime_sessions AS sessions
              ON sessions.runtime_session_id = turns.runtime_session_id
            WHERE sessions.owner_id = ? AND sessions.conversation_id = ?
              AND turns.turn_status NOT IN ('completed', 'abandoned');
            """,
            (owner_id, conversation_id),
        ).fetchone()[0]
    assert turn_count == 1


def test_concurrent_repository_admission_allows_exactly_one_writer(tmp_path: Path):
    db_path = tmp_path / "concurrent.sqlite3"
    repositories = [RuntimeStateRepository(db_path), RuntimeStateRepository(db_path)]
    barrier = threading.Barrier(2)

    def attempt(index: int):
        barrier.wait(timeout=5)
        try:
            result = repositories[index].start_turn(
                request_id=f"overlap-{index}",
                owner_id="owner-overlap",
                conversation_id="conversation-overlap",
                surface=f"surface-{index}",
            )
            return "admitted", result[1].runtime_turn_id
        except RuntimeError as exc:
            return str(exc), None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, (0, 1)))

    assert sorted(outcome[0] for outcome in outcomes) == [
        "admitted",
        "runtime_thread_contended",
    ]
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversation_runtime_turns;").fetchone()[0] == 1


def test_revision_compare_and_set_progression_and_terminal_release(tmp_path: Path):
    _use_database(tmp_path)
    client = TestClient(app)
    owner_id = "owner-revision"
    conversation_id = "conversation-revision"
    initial = _thread(client, owner_id, conversation_id).json()
    assert initial["state"] == "idle"
    assert initial["revision"] == 0

    stale = _start(
        client,
        request_id="revision-stale",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-main",
        expected_thread_revision=1,
    )
    assert (stale.status_code, stale.json()) == (
        409,
        {"detail": "runtime_thread_revision_conflict"},
    )
    assert _thread(client, owner_id, conversation_id).json() == initial

    admitted = _start(
        client,
        request_id="revision-match",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-main",
        expected_thread_revision=0,
    )
    assert admitted.status_code == 200
    admitted_body = admitted.json()
    assert _thread(client, owner_id, conversation_id).json()["revision"] == 1

    updated = client.post(
        "/v1/runtime/turns/update",
        json={
            "request_id": "revision-update",
            "runtime_session_id": admitted_body["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": admitted_body["runtime_turn"]["runtime_turn_id"],
            "turn_status": "responding",
        },
    )
    assert updated.status_code == 200
    assert _thread(client, owner_id, conversation_id).json()["revision"] == 1

    completed = _complete(
        client,
        request_id="revision-complete",
        runtime_session_id=admitted_body["runtime_session"]["runtime_session_id"],
        runtime_turn_id=admitted_body["runtime_turn"]["runtime_turn_id"],
    )
    assert completed.status_code == 200
    released = _thread(client, owner_id, conversation_id).json()
    assert released["state"] == "idle"
    assert released["active_runtime_turn_id"] is None
    assert released["revision"] == 2

    abandoned = _start(
        client,
        request_id="revision-abandon-start",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-main",
        expected_thread_revision=2,
    ).json()
    abandonment = _complete(
        client,
        request_id="revision-abandon-complete",
        runtime_session_id=abandoned["runtime_session"]["runtime_session_id"],
        runtime_turn_id=abandoned["runtime_turn"]["runtime_turn_id"],
        turn_status="abandoned",
    )
    assert abandonment.status_code == 200
    assert _thread(client, owner_id, conversation_id).json()["revision"] == 4


def test_admission_and_completion_retries_are_idempotent(tmp_path: Path):
    db_path = _use_database(tmp_path)
    client = TestClient(app)
    owner_id = "owner-retry"
    conversation_id = "conversation-retry"

    first = _start(
        client,
        request_id="retry-start",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-one",
    )
    retry = _start(
        client,
        request_id="retry-start",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-one",
        expected_thread_revision=0,
    )
    assert first.status_code == retry.status_code == 200
    assert first.json() == retry.json()
    assert _thread(client, owner_id, conversation_id).json()["revision"] == 1

    other_surface = _start(
        client,
        request_id="retry-start",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-two",
    )
    assert other_surface.status_code == 409
    assert other_surface.json() == {"detail": "runtime_thread_contended"}
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversation_runtime_turns;").fetchone()[0] == 1
        start_events = conn.execute(
            "SELECT COUNT(*) FROM conversation_runtime_events WHERE event_type = 'turn_started';"
        ).fetchone()[0]
    assert start_events == 1

    body = first.json()
    complete = _complete(
        client,
        request_id="retry-complete",
        runtime_session_id=body["runtime_session"]["runtime_session_id"],
        runtime_turn_id=body["runtime_turn"]["runtime_turn_id"],
    )
    complete_retry = _complete(
        client,
        request_id="retry-complete",
        runtime_session_id=body["runtime_session"]["runtime_session_id"],
        runtime_turn_id=body["runtime_turn"]["runtime_turn_id"],
    )
    assert complete.status_code == complete_retry.status_code == 200
    assert complete.json() == complete_retry.json()
    assert _thread(client, owner_id, conversation_id).json()["revision"] == 2
    with sqlite3.connect(db_path) as conn:
        terminal_events = conn.execute(
            "SELECT COUNT(*) FROM conversation_runtime_events WHERE event_type = 'turn_completed';"
        ).fetchone()[0]
    assert terminal_events == 1


def test_update_and_completion_require_the_current_owning_session(tmp_path: Path):
    _use_database(tmp_path)
    client = TestClient(app)
    owner_id = "owner-association"
    conversation_id = "conversation-association"
    other_session = client.post(
        "/v1/runtime/sessions/resolve",
        json={
            "request_id": "association-session",
            "owner_id": owner_id,
            "conversation_id": conversation_id,
            "surface": "surface-other",
        },
    ).json()["runtime_session"]
    admitted = _start(
        client,
        request_id="association-start",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-main",
    ).json()
    turn_id = admitted["runtime_turn"]["runtime_turn_id"]
    before = _thread(client, owner_id, conversation_id).json()

    wrong_update = client.post(
        "/v1/runtime/turns/update",
        json={
            "request_id": "association-wrong-update",
            "runtime_session_id": other_session["runtime_session_id"],
            "runtime_turn_id": turn_id,
            "turn_status": "retrieving",
        },
    )
    wrong_complete = _complete(
        client,
        request_id="association-wrong-complete",
        runtime_session_id=other_session["runtime_session_id"],
        runtime_turn_id=turn_id,
    )
    assert wrong_update.status_code == wrong_complete.status_code == 400
    assert wrong_update.json() == wrong_complete.json() == {
        "detail": "runtime_turn_session_mismatch"
    }
    assert _thread(client, owner_id, conversation_id).json()["revision"] == before["revision"]

    missing_turn = client.post(
        "/v1/runtime/turns/update",
        json={
            "request_id": "association-missing-turn",
            "runtime_session_id": admitted["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": "turn-missing",
            "turn_status": "retrieving",
        },
    )
    assert (missing_turn.status_code, missing_turn.json()) == (
        404,
        {"detail": "runtime_turn_not_found"},
    )

    correct = client.post(
        "/v1/runtime/turns/update",
        json={
            "request_id": "association-correct-update",
            "runtime_session_id": admitted["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": turn_id,
            "turn_status": "retrieving",
        },
    )
    assert correct.status_code == 200


def test_stale_completion_cannot_release_a_later_turn(tmp_path: Path):
    _use_database(tmp_path)
    client = TestClient(app)
    owner_id = "owner-stale-terminal"
    conversation_id = "conversation-stale-terminal"
    first = _start(
        client,
        request_id="stale-first-start",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-main",
    ).json()
    first_session_id = first["runtime_session"]["runtime_session_id"]
    first_turn_id = first["runtime_turn"]["runtime_turn_id"]
    assert _complete(
        client,
        request_id="stale-first-complete",
        runtime_session_id=first_session_id,
        runtime_turn_id=first_turn_id,
    ).status_code == 200

    second = _start(
        client,
        request_id="stale-second-start",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-main",
    ).json()
    before = _thread(client, owner_id, conversation_id).json()

    stale = _complete(
        client,
        request_id="stale-late-attempt",
        runtime_session_id=first_session_id,
        runtime_turn_id=first_turn_id,
    )
    assert (stale.status_code, stale.json()) == (
        409,
        {"detail": "runtime_turn_not_current"},
    )
    exact_retry = _complete(
        client,
        request_id="stale-first-complete",
        runtime_session_id=first_session_id,
        runtime_turn_id=first_turn_id,
    )
    assert exact_retry.status_code == 200
    after = _thread(client, owner_id, conversation_id).json()
    assert after["active_runtime_turn_id"] == second["runtime_turn"]["runtime_turn_id"]
    assert after["revision"] == before["revision"]


def test_active_and_idle_thread_state_survive_repository_recreation(tmp_path: Path):
    db_path = tmp_path / "restart.sqlite3"
    first_repository = RuntimeStateRepository(db_path)
    session, turn, _ = first_repository.start_turn(
        request_id="restart-start",
        owner_id="owner-restart",
        conversation_id="conversation-restart",
        surface="surface-main",
    )

    reopened = RuntimeStateRepository(db_path)
    active = reopened.resolve_thread(
        owner_id="owner-restart",
        conversation_id="conversation-restart",
    )
    assert active.state == "active"
    assert active.active_runtime_turn_id == turn.runtime_turn_id
    try:
        reopened.start_turn(
            request_id="restart-competing",
            owner_id="owner-restart",
            conversation_id="conversation-restart",
            surface="surface-other",
        )
    except RuntimeError as exc:
        assert str(exc) == "runtime_thread_contended"
    else:
        raise AssertionError("competing turn was admitted after repository recreation")

    reopened.complete_turn(
        request_id="restart-complete",
        runtime_session_id=session.runtime_session_id,
        runtime_turn_id=turn.runtime_turn_id,
        turn_status="completed",
    )
    final_repository = RuntimeStateRepository(db_path)
    idle = final_repository.resolve_thread(
        owner_id="owner-restart",
        conversation_id="conversation-restart",
    )
    assert idle.state == "idle"
    assert idle.revision == 2


def test_missing_projection_with_no_active_turn_reconstructs_idle(tmp_path: Path):
    db_path = tmp_path / "reconstruct-idle.sqlite3"
    repository = RuntimeStateRepository(db_path)
    repository.resolve_session(
        request_id="reconstruct-session",
        owner_id="owner-reconstruct-idle",
        conversation_id="conversation-reconstruct-idle",
        surface="surface-main",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM conversation_runtime_threads;")

    projection = RuntimeStateRepository(db_path).resolve_thread(
        owner_id="owner-reconstruct-idle",
        conversation_id="conversation-reconstruct-idle",
    )
    assert projection.state == "idle"
    assert projection.revision == 0


def test_missing_projection_with_one_active_turn_reconstructs_active(tmp_path: Path):
    db_path = tmp_path / "reconstruct-active.sqlite3"
    repository = RuntimeStateRepository(db_path)
    session = repository.resolve_session(
        request_id="reconstruct-session",
        owner_id="owner-reconstruct-active",
        conversation_id="conversation-reconstruct-active",
        surface="surface-main",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO conversation_runtime_turns (
                runtime_turn_id, runtime_session_id, input_message_id, turn_status,
                intent_class, timing_policy, restraint_policy, continuation_state,
                created_at, updated_at, completed_at
            ) VALUES (?, ?, NULL, 'received', NULL, NULL, NULL, NULL, ?, ?, NULL);
            """,
            ("turn-reconstruct-active", session.runtime_session_id, "time-a", "time-a"),
        )
        conn.execute("DELETE FROM conversation_runtime_threads;")

    projection = RuntimeStateRepository(db_path).resolve_thread(
        owner_id="owner-reconstruct-active",
        conversation_id="conversation-reconstruct-active",
    )
    assert projection.state == "active"
    assert projection.active_runtime_session_id == session.runtime_session_id
    assert projection.active_runtime_turn_id == "turn-reconstruct-active"
    assert projection.active_surface == "surface-main"


def test_missing_projection_with_multiple_active_turns_reconstructs_contended(
    tmp_path: Path,
):
    db_path = tmp_path / "reconstruct-contended.sqlite3"
    repository = RuntimeStateRepository(db_path)
    sessions = [
        repository.resolve_session(
            request_id=f"reconstruct-session-{index}",
            owner_id="owner-reconstruct-contended",
            conversation_id="conversation-reconstruct-contended",
            surface=f"surface-{index}",
        )
        for index in range(2)
    ]
    with sqlite3.connect(db_path) as conn:
        for index, session in enumerate(sessions):
            conn.execute(
                """
                INSERT INTO conversation_runtime_turns (
                    runtime_turn_id, runtime_session_id, input_message_id, turn_status,
                    intent_class, timing_policy, restraint_policy, continuation_state,
                    created_at, updated_at, completed_at
                ) VALUES (?, ?, NULL, 'received', NULL, NULL, NULL, NULL, ?, ?, NULL);
                """,
                (
                    f"turn-reconstruct-{index}",
                    session.runtime_session_id,
                    f"time-{index}",
                    f"time-{index}",
                ),
            )
        conn.execute("DELETE FROM conversation_runtime_threads;")

    reopened = RuntimeStateRepository(db_path)
    projection = reopened.resolve_thread(
        owner_id="owner-reconstruct-contended",
        conversation_id="conversation-reconstruct-contended",
    )
    assert projection.state == "contended"
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversation_runtime_turns;").fetchone()[0] == 2
    try:
        reopened.start_turn(
            request_id="reconstruct-third",
            owner_id="owner-reconstruct-contended",
            conversation_id="conversation-reconstruct-contended",
            surface="surface-third",
        )
    except RuntimeError as exc:
        assert str(exc) == "runtime_thread_contended"
    else:
        raise AssertionError("contended reconstruction admitted another turn")


def test_inconsistent_active_reference_becomes_unavailable(tmp_path: Path):
    db_path = tmp_path / "inconsistent.sqlite3"
    repository = RuntimeStateRepository(db_path)
    repository.start_turn(
        request_id="inconsistent-start",
        owner_id="owner-inconsistent",
        conversation_id="conversation-inconsistent",
        surface="surface-main",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE conversation_runtime_threads
            SET active_runtime_turn_id = 'turn-missing'
            WHERE owner_id = 'owner-inconsistent';
            """
        )

    reopened = RuntimeStateRepository(db_path)
    projection = reopened.resolve_thread(
        owner_id="owner-inconsistent",
        conversation_id="conversation-inconsistent",
    )
    assert projection.state == "unavailable"
    assert projection.active_runtime_turn_id == "turn-missing"
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversation_runtime_turns;").fetchone()[0] == 1
    try:
        reopened.start_turn(
            request_id="inconsistent-next",
            owner_id="owner-inconsistent",
            conversation_id="conversation-inconsistent",
            surface="surface-main",
        )
    except RuntimeError as exc:
        assert str(exc) == "runtime_thread_unavailable"
    else:
        raise AssertionError("unavailable thread admitted another turn")


def test_thread_api_is_bounded_and_persistence_failures_do_not_leak(
    tmp_path: Path,
    monkeypatch,
):
    _use_database(tmp_path)
    client = TestClient(app)
    projection = _thread(client, "owner-bounded", "conversation-bounded")
    assert projection.status_code == 200
    assert set(projection.json()) == {
        "owner_id",
        "conversation_id",
        "state",
        "revision",
        "active_runtime_session_id",
        "active_runtime_turn_id",
        "active_surface",
        "participating_surfaces",
        "participating_session_count",
        "last_activity_at",
        "created_at",
        "updated_at",
    }
    serialized = str(projection.json()).lower()
    for excluded in ("content", "eligib", "lifecycle", "prompt", "provider"):
        assert excluded not in serialized

    malformed = client.post(
        "/v1/runtime/turns/start",
        json={
            "request_id": "bounded-malformed",
            "owner_id": "owner-bounded",
            "conversation_id": "conversation-bounded",
            "surface": "surface-main",
            "expected_thread_revision": -1,
        },
    )
    assert malformed.status_code == 422

    def unavailable(**_kwargs):
        raise sqlite3.OperationalError("storage path /private/runtime.sqlite3 failed")

    monkeypatch.setattr(main_module, "resolve_runtime_thread", unavailable)
    failed = _thread(client, "owner-bounded", "conversation-bounded")
    assert (failed.status_code, failed.json()) == (
        503,
        {"detail": "runtime_state_persistence_unavailable"},
    )
    assert "/private" not in str(failed.json())


def test_missing_session_errors_are_bounded(tmp_path: Path):
    _use_database(tmp_path)
    client = TestClient(app)
    update = client.post(
        "/v1/runtime/turns/update",
        json={
            "request_id": "missing-session-update",
            "runtime_session_id": "session-missing",
            "runtime_turn_id": "turn-missing",
            "turn_status": "retrieving",
        },
    )
    diagnostics = client.get("/v1/runtime/sessions/session-missing")
    assert (update.status_code, update.json()) == (
        404,
        {"detail": "runtime_session_not_found"},
    )
    assert (diagnostics.status_code, diagnostics.json()) == (
        404,
        {"detail": "runtime_session_not_found"},
    )


def test_cross_conversation_session_cannot_mutate_another_conversation_turn(
    tmp_path: Path,
):
    db_path = _use_database(tmp_path)
    client = TestClient(app)
    admitted = _start(
        client,
        request_id="cross-conversation-start",
        owner_id="owner-cross-conversation",
        conversation_id="conversation-primary",
        surface="surface-main",
    ).json()
    other_session = client.post(
        "/v1/runtime/sessions/resolve",
        json={
            "request_id": "cross-conversation-session",
            "owner_id": "owner-cross-conversation",
            "conversation_id": "conversation-other",
            "surface": "surface-main",
        },
    ).json()["runtime_session"]
    before_primary = _thread(
        client,
        "owner-cross-conversation",
        "conversation-primary",
    ).json()
    before_other = _thread(
        client,
        "owner-cross-conversation",
        "conversation-other",
    ).json()
    before_rows = _runtime_rows(db_path)

    update = client.post(
        "/v1/runtime/turns/update",
        json={
            "request_id": "cross-conversation-update",
            "runtime_session_id": other_session["runtime_session_id"],
            "runtime_turn_id": admitted["runtime_turn"]["runtime_turn_id"],
            "turn_status": "retrieving",
        },
    )
    completion = _complete(
        client,
        request_id="cross-conversation-complete",
        runtime_session_id=other_session["runtime_session_id"],
        runtime_turn_id=admitted["runtime_turn"]["runtime_turn_id"],
    )

    assert (update.status_code, update.json()) == (
        400,
        {"detail": "runtime_turn_session_mismatch"},
    )
    assert (completion.status_code, completion.json()) == (
        400,
        {"detail": "runtime_turn_session_mismatch"},
    )
    assert _runtime_rows(db_path) == before_rows
    assert _thread(
        client,
        "owner-cross-conversation",
        "conversation-primary",
    ).json() == before_primary
    assert _thread(
        client,
        "owner-cross-conversation",
        "conversation-other",
    ).json() == before_other


def test_contended_thread_rejects_update_and_completion_without_mutation(tmp_path: Path):
    db_path = _use_database(tmp_path, "contended-mutation.sqlite3")
    repository = RuntimeStateRepository(db_path)
    sessions = [
        repository.resolve_session(
            request_id=f"contended-session-{index}",
            owner_id="owner-contended-mutation",
            conversation_id="conversation-contended-mutation",
            surface=f"surface-{index}",
        )
        for index in range(2)
    ]
    with sqlite3.connect(db_path) as conn:
        for index, session in enumerate(sessions):
            conn.execute(
                """
                INSERT INTO conversation_runtime_turns (
                    runtime_turn_id, runtime_session_id, input_message_id, turn_status,
                    intent_class, timing_policy, restraint_policy, continuation_state,
                    created_at, updated_at, completed_at
                ) VALUES (?, ?, NULL, 'received', NULL, NULL, NULL, NULL, ?, ?, NULL);
                """,
                (
                    f"turn-contended-mutation-{index}",
                    session.runtime_session_id,
                    f"time-{index}",
                    f"time-{index}",
                ),
            )
        conn.execute("DELETE FROM conversation_runtime_threads;")

    client = TestClient(app)
    projection = _thread(
        client,
        "owner-contended-mutation",
        "conversation-contended-mutation",
    )
    assert projection.status_code == 200
    assert projection.json()["state"] == "contended"
    before_rows = _runtime_rows(db_path)

    update = client.post(
        "/v1/runtime/turns/update",
        json={
            "request_id": "contended-update",
            "runtime_session_id": sessions[0].runtime_session_id,
            "runtime_turn_id": "turn-contended-mutation-0",
            "turn_status": "retrieving",
        },
    )
    completion = _complete(
        client,
        request_id="contended-complete",
        runtime_session_id=sessions[0].runtime_session_id,
        runtime_turn_id="turn-contended-mutation-0",
    )

    assert (update.status_code, update.json()) == (
        409,
        {"detail": "runtime_thread_contended"},
    )
    assert (completion.status_code, completion.json()) == (
        409,
        {"detail": "runtime_thread_contended"},
    )
    assert _runtime_rows(db_path) == before_rows


def test_unavailable_thread_rejects_update_and_completion_without_mutation(tmp_path: Path):
    db_path = _use_database(tmp_path, "unavailable-mutation.sqlite3")
    client = TestClient(app)
    admitted = _start(
        client,
        request_id="unavailable-mutation-start",
        owner_id="owner-unavailable-mutation",
        conversation_id="conversation-unavailable-mutation",
        surface="surface-main",
    ).json()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE conversation_runtime_threads
            SET active_runtime_turn_id = 'turn-missing'
            WHERE owner_id = ? AND conversation_id = ?;
            """,
            ("owner-unavailable-mutation", "conversation-unavailable-mutation"),
        )

    projection = _thread(
        client,
        "owner-unavailable-mutation",
        "conversation-unavailable-mutation",
    )
    assert projection.status_code == 200
    assert projection.json()["state"] == "unavailable"
    before_rows = _runtime_rows(db_path)
    session_id = admitted["runtime_session"]["runtime_session_id"]
    turn_id = admitted["runtime_turn"]["runtime_turn_id"]

    update = client.post(
        "/v1/runtime/turns/update",
        json={
            "request_id": "unavailable-update",
            "runtime_session_id": session_id,
            "runtime_turn_id": turn_id,
            "turn_status": "retrieving",
        },
    )
    completion = _complete(
        client,
        request_id="unavailable-complete",
        runtime_session_id=session_id,
        runtime_turn_id=turn_id,
    )

    assert (update.status_code, update.json()) == (
        503,
        {"detail": "runtime_thread_unavailable"},
    )
    assert (completion.status_code, completion.json()) == (
        503,
        {"detail": "runtime_thread_unavailable"},
    )
    assert _runtime_rows(db_path) == before_rows


@pytest.mark.parametrize(
    ("function_name", "method", "path", "payload"),
    [
        (
            "resolve_runtime_session",
            "POST",
            "/v1/runtime/sessions/resolve",
            {
                "request_id": "persistence-session",
                "owner_id": "owner-persistence",
                "conversation_id": "conversation-persistence",
                "surface": "surface-main",
            },
        ),
        (
            "start_turn",
            "POST",
            "/v1/runtime/turns/start",
            {
                "request_id": "persistence-start",
                "owner_id": "owner-persistence",
                "conversation_id": "conversation-persistence",
                "surface": "surface-main",
            },
        ),
        (
            "update_turn",
            "POST",
            "/v1/runtime/turns/update",
            {
                "request_id": "persistence-update",
                "runtime_session_id": "session-persistence",
                "runtime_turn_id": "turn-persistence",
                "turn_status": "retrieving",
            },
        ),
        (
            "complete_turn",
            "POST",
            "/v1/runtime/turns/complete",
            {
                "request_id": "persistence-complete",
                "runtime_session_id": "session-persistence",
                "runtime_turn_id": "turn-persistence",
                "turn_status": "completed",
            },
        ),
        (
            "get_runtime_session",
            "GET",
            "/v1/runtime/sessions/session-persistence",
            None,
        ),
    ],
)
def test_runtime_endpoints_bound_persistence_failures(
    tmp_path: Path,
    monkeypatch,
    function_name: str,
    method: str,
    path: str,
    payload: dict[str, object] | None,
):
    _use_database(tmp_path)
    client = TestClient(app)

    def unavailable(*_args, **_kwargs):
        raise sqlite3.OperationalError("storage path /private/runtime.sqlite3 SELECT failed")

    monkeypatch.setattr(main_module, function_name, unavailable)
    response = client.request(method, path, json=payload)

    assert (response.status_code, response.json()) == (
        503,
        {"detail": "runtime_state_persistence_unavailable"},
    )
    serialized = str(response.json()).lower()
    assert "/private" not in serialized
    assert "select" not in serialized


def test_admission_failure_after_turn_insert_rolls_back_all_state(
    tmp_path: Path,
    monkeypatch,
):
    db_path = tmp_path / "admission-rollback.sqlite3"
    repository = RuntimeStateRepository(db_path)
    repository.resolve_session(
        request_id="rollback-session",
        owner_id="owner-rollback",
        conversation_id="conversation-rollback",
        surface="surface-main",
    )
    before_rows = _runtime_rows(db_path)
    original_record_event = repository._record_event

    def fail_before_admission_commit(conn, **kwargs):
        if kwargs["event_type"] == "turn_started":
            raise sqlite3.OperationalError("injected admission failure")
        return original_record_event(conn, **kwargs)

    monkeypatch.setattr(repository, "_record_event", fail_before_admission_commit)
    with pytest.raises(sqlite3.OperationalError, match="injected admission failure"):
        repository.start_turn(
            request_id="rollback-start",
            owner_id="owner-rollback",
            conversation_id="conversation-rollback",
            surface="surface-main",
        )

    assert _runtime_rows(db_path) == before_rows
    projection = RuntimeStateRepository(db_path).resolve_thread(
        owner_id="owner-rollback",
        conversation_id="conversation-rollback",
    )
    assert projection.state == "idle"
    assert projection.revision == 0
    assert projection.active_runtime_session_id is None
    assert projection.active_runtime_turn_id is None


def test_concurrent_exact_admission_retry_returns_one_turn(tmp_path: Path):
    db_path = tmp_path / "concurrent-retry.sqlite3"
    repositories = [RuntimeStateRepository(db_path), RuntimeStateRepository(db_path)]
    barrier = threading.Barrier(2)

    def attempt(index: int):
        barrier.wait(timeout=5)
        return repositories[index].start_turn(
            request_id="concurrent-retry-start",
            owner_id="owner-concurrent-retry",
            conversation_id="conversation-concurrent-retry",
            surface="surface-main",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, (0, 1)))

    assert outcomes[0][0].runtime_session_id == outcomes[1][0].runtime_session_id
    assert outcomes[0][1].runtime_turn_id == outcomes[1][1].runtime_turn_id
    assert outcomes[0][2].event_id == outcomes[1][2].event_id
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversation_runtime_turns;").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_runtime_events WHERE event_type = 'turn_started';"
        ).fetchone()[0] == 1
        revision = conn.execute(
            "SELECT revision FROM conversation_runtime_threads;"
        ).fetchone()[0]
    assert revision == 1


@pytest.mark.parametrize(
    ("active_turn_count", "expected_state"),
    [(0, "idle"), (1, "active"), (2, "contended")],
)
def test_legacy_database_initialization_reconstructs_conservatively(
    tmp_path: Path,
    active_turn_count: int,
    expected_state: str,
):
    db_path = tmp_path / f"legacy-{active_turn_count}.sqlite3"
    owner_id = f"owner-legacy-{active_turn_count}"
    conversation_id = f"conversation-legacy-{active_turn_count}"
    timestamp = "2026-01-01T00:00:00+00:00"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE conversation_runtime_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                runtime_session_id TEXT NOT NULL UNIQUE,
                runtime_state_id TEXT NOT NULL UNIQUE,
                owner_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                surface TEXT NOT NULL,
                surface_session_id TEXT,
                status TEXT NOT NULL,
                active_mode TEXT,
                attention_state TEXT,
                active_scene TEXT,
                interaction_mode TEXT,
                attention_focus_json TEXT NOT NULL,
                temporary_constraints_json TEXT NOT NULL,
                reset_after_turn INTEGER NOT NULL,
                trace_refs_json TEXT NOT NULL,
                started_at TEXT NOT NULL,
                last_activity_at TEXT NOT NULL,
                closed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(owner_id, conversation_id, surface)
            );
            CREATE TABLE conversation_runtime_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                runtime_turn_id TEXT NOT NULL UNIQUE,
                runtime_session_id TEXT NOT NULL,
                input_message_id TEXT,
                turn_status TEXT NOT NULL,
                intent_class TEXT,
                timing_policy TEXT,
                restraint_policy TEXT,
                continuation_state TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY(runtime_session_id)
                    REFERENCES conversation_runtime_sessions(runtime_session_id)
            );
            """
        )
        session_count = max(1, active_turn_count)
        for index in range(session_count):
            session_id = f"legacy-session-{active_turn_count}-{index}"
            conn.execute(
                """
                INSERT INTO conversation_runtime_sessions (
                    runtime_session_id, runtime_state_id, owner_id, conversation_id,
                    surface, surface_session_id, status, active_mode, attention_state,
                    active_scene, interaction_mode, attention_focus_json,
                    temporary_constraints_json, reset_after_turn, trace_refs_json,
                    started_at, last_activity_at, closed_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, NULL, 'active', NULL, NULL, NULL, NULL,
                          'null', '[]', 0, '[]', ?, ?, NULL, ?, ?);
                """,
                (
                    session_id,
                    f"legacy-state-{active_turn_count}-{index}",
                    owner_id,
                    conversation_id,
                    f"surface-{index}",
                    timestamp,
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            )
            if index < active_turn_count:
                conn.execute(
                    """
                    INSERT INTO conversation_runtime_turns (
                        runtime_turn_id, runtime_session_id, input_message_id,
                        turn_status, intent_class, timing_policy, restraint_policy,
                        continuation_state, created_at, updated_at, completed_at
                    ) VALUES (?, ?, NULL, 'received', NULL, NULL, NULL, NULL, ?, ?, NULL);
                    """,
                    (
                        f"legacy-turn-{active_turn_count}-{index}",
                        session_id,
                        timestamp,
                        timestamp,
                    ),
                )
        assert conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'conversation_runtime_threads';
            """
        ).fetchone() is None

    repository = RuntimeStateRepository(db_path)
    projection = repository.resolve_thread(
        owner_id=owner_id,
        conversation_id=conversation_id,
    )

    assert projection.state == expected_state
    assert projection.revision == 0
    if active_turn_count == 1:
        assert projection.active_runtime_session_id == "legacy-session-1-0"
        assert projection.active_runtime_turn_id == "legacy-turn-1-0"
        assert projection.active_surface == "surface-0"
    else:
        assert projection.active_runtime_session_id is None
        assert projection.active_runtime_turn_id is None
        assert projection.active_surface is None
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_runtime_sessions;"
        ).fetchone()[0] == max(1, active_turn_count)
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_runtime_turns;"
        ).fetchone()[0] == active_turn_count


def test_continuation_selection_request_validation_is_strict_and_bounded(
    tmp_path: Path,
):
    db_path = _use_database(tmp_path, "selection-validation.sqlite3")
    client = TestClient(app)
    valid = {
        "request_id": "selection-validation",
        "owner_id": "owner-selection-validation",
        "surface": "surface-current",
        "candidate_set_complete": True,
        "stale_after_seconds": 3600,
        "candidates": [_continuation_candidate("conversation-validation")],
    }
    invalid_payloads = [
        {**valid, "candidates": [
            _continuation_candidate(f"conversation-{index}") for index in range(9)
        ]},
        {**valid, "request_id": ""},
        {**valid, "request_id": "r" * 121},
        {**valid, "owner_id": ""},
        {**valid, "owner_id": "o" * 121},
        {**valid, "surface": ""},
        {**valid, "surface": "s" * 65},
        {**valid, "candidate_set_complete": 1},
        {**valid, "stale_after_seconds": True},
        {**valid, "stale_after_seconds": 59},
        {**valid, "stale_after_seconds": 86401},
        {**valid, "extra": "forbidden"},
        {
            **valid,
            "candidates": [
                {**_continuation_candidate("conversation-validation"), "extra": "forbidden"}
            ],
        },
        {
            **valid,
            "candidates": [
                _continuation_candidate(
                    "conversation-validation",
                    lifecycle_state="unknown",
                )
            ],
        },
        {
            **valid,
            "candidates": [
                _continuation_candidate("conversation-validation")
                | {"durable_updated_at": "2026-01-01T00:00:00"}
            ],
        },
        {
            **valid,
            "candidates": [
                _continuation_candidate("conversation-validation")
                | {"durable_updated_at": "not-a-timestamp"}
            ],
        },
        {
            **valid,
            "candidates": [
                _continuation_candidate("conversation-validation")
                | {"durable_updated_at": 1_700_000_000}
            ],
        },
        {
            **valid,
            "candidates": [
                _continuation_candidate(""),
            ],
        },
        {
            **valid,
            "candidates": [
                _continuation_candidate("c" * 121),
            ],
        },
    ]

    for payload in invalid_payloads:
        response = client.post("/v1/runtime/continuations/select", json=payload)
        assert response.status_code == 422

    assert _runtime_rows(db_path) == ((), (), (), ())


def test_duplicate_continuation_candidates_fail_before_repository_access(
    tmp_path: Path,
    monkeypatch,
):
    db_path = _use_database(tmp_path, "selection-duplicates.sqlite3")

    def unexpected_repository_access(_request):
        raise AssertionError("repository must not be called")

    monkeypatch.setattr(
        main_module,
        "select_runtime_continuation",
        unexpected_repository_access,
    )
    candidate = _continuation_candidate("conversation-duplicate")
    response = _select_continuation(
        TestClient(app),
        candidates=[candidate, candidate],
    )

    assert response.status_code == 422
    assert _runtime_rows(db_path) == ((), (), (), ())


def test_continuation_selection_response_models_reject_extras_and_incoherence():
    valid_result = {
        "outcome": "create_new",
        "timing_policy": "answer_now",
        "selected_conversation_id": None,
        "selected_thread_revision": None,
        "candidate_count": 0,
        "eligible_candidate_count": 0,
        "reason_codes": ["no_candidates"],
        "policy_version": "continuation-selection.v1",
    }
    valid_response = {
        "schema_version": "runtime-continuation-selection.v1",
        "request_id": "selection-response",
        "owner_id": "owner-selection-response",
        "surface": "surface-current",
        "result": valid_result,
    }

    with pytest.raises(ValidationError):
        ContinuationSelectionResult.model_validate({**valid_result, "extra": "forbidden"})
    with pytest.raises(ValidationError):
        ContinuationSelectionResponse.model_validate(
            {**valid_response, "extra": "forbidden"}
        )
    with pytest.raises(ValidationError):
        ContinuationSelectionResult.model_validate(
            {
                **valid_result,
                "selected_conversation_id": "conversation-forbidden",
                "selected_thread_revision": 0,
            }
        )
    with pytest.raises(ValidationError):
        ContinuationSelectionResult.model_validate(
            {**valid_result, "timing_policy": "close_turn"}
        )
    with pytest.raises(ValidationError):
        ContinuationSelectionResult.model_validate(
            {**valid_result, "reason_codes": ["no_candidates", "no_candidates"]}
        )


def test_complete_empty_candidate_set_authorizes_create_new_without_state(
    tmp_path: Path,
):
    db_path = _use_database(tmp_path, "selection-empty.sqlite3")
    before = _runtime_rows(db_path)

    response = _select_continuation(TestClient(app), candidates=[])

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "runtime-continuation-selection.v1",
        "request_id": "continuation-selection-request",
        "owner_id": "owner-selection",
        "surface": "surface-current",
        "result": {
            "outcome": "create_new",
            "timing_policy": "answer_now",
            "selected_conversation_id": None,
            "selected_thread_revision": None,
            "candidate_count": 0,
            "eligible_candidate_count": 0,
            "reason_codes": ["no_candidates"],
            "policy_version": "continuation-selection.v1",
        },
    }
    assert _runtime_rows(db_path) == before


def test_missing_other_owner_and_sessionless_runtime_state_cannot_resume(
    tmp_path: Path,
):
    db_path = _use_database(tmp_path, "selection-missing.sqlite3")
    client = TestClient(app)
    candidate = _continuation_candidate("conversation-missing")

    missing = _select_continuation(client, candidates=[candidate])
    assert missing.status_code == 200
    assert missing.json()["result"]["outcome"] == "create_new"
    assert missing.json()["result"]["reason_codes"] == [
        "no_eligible_candidates",
        "runtime_state_missing",
    ]
    assert _runtime_rows(db_path) == ((), (), (), ())

    _resolve_selection_session(
        client,
        owner_id="owner-other",
        conversation_id="conversation-missing",
    )
    other_owner_before = _runtime_rows(db_path)
    other_owner = _select_continuation(client, candidates=[candidate])
    assert other_owner.json()["result"] == missing.json()["result"]
    assert _runtime_rows(db_path) == other_owner_before

    _thread(client, "owner-selection", "conversation-sessionless")
    sessionless_before = _runtime_rows(db_path)
    sessionless = _select_continuation(
        client,
        candidates=[_continuation_candidate("conversation-sessionless")],
    )
    assert sessionless.status_code == 200
    assert sessionless.json()["result"]["outcome"] == "create_new"
    assert sessionless.json()["result"]["reason_codes"] == [
        "no_eligible_candidates",
        "runtime_session_missing",
    ]
    assert _runtime_rows(db_path) == sessionless_before


def test_one_fresh_idle_candidate_resumes_without_mutation_and_survives_reopen(
    tmp_path: Path,
):
    db_path = _use_database(tmp_path, "selection-resume.sqlite3")
    client = TestClient(app)
    owner_id = "owner-selection"
    conversation_id = "conversation-resume"
    _resolve_selection_session(
        client,
        owner_id=owner_id,
        conversation_id=conversation_id,
    )
    candidates = [_continuation_candidate(conversation_id)]
    before = _runtime_rows(db_path)

    first = _select_continuation(client, candidates=candidates, surface="surface-a")
    second = _select_continuation(client, candidates=candidates, surface="surface-b")

    assert first.status_code == second.status_code == 200
    assert first.json()["result"] == second.json()["result"]
    assert first.json()["result"] == {
        "outcome": "resume",
        "timing_policy": "resume_previous_thread",
        "selected_conversation_id": conversation_id,
        "selected_thread_revision": 0,
        "candidate_count": 1,
        "eligible_candidate_count": 1,
        "reason_codes": ["one_eligible_candidate"],
        "policy_version": "continuation-selection.v1",
    }
    assert _runtime_rows(db_path) == before

    request = ContinuationSelectionRequest.model_validate(
        {
            "request_id": "selection-after-reopen",
            "owner_id": owner_id,
            "surface": "surface-after-reopen",
            "candidate_set_complete": True,
            "stale_after_seconds": 3600,
            "candidates": [
                {
                    **candidates[0],
                    "durable_updated_at": datetime.fromisoformat(
                        str(candidates[0]["durable_updated_at"])
                    ),
                }
            ],
        }
    )
    reopened_result = RuntimeStateRepository(db_path).select_continuation(request)
    assert reopened_result.result.selected_conversation_id == conversation_id
    assert reopened_result.result.selected_thread_revision == 0
    assert _runtime_rows(db_path) == before


@pytest.mark.parametrize(
    ("lifecycle_state", "durable_age", "runtime_age", "expected_reasons"),
    [
        ("open", 7200, 0, ["no_eligible_candidates", "candidate_stale"]),
        ("open", 0, 7200, ["no_eligible_candidates", "candidate_stale"]),
        ("open", 7200, 7200, ["no_eligible_candidates", "candidate_stale"]),
        ("closed", 0, 0, ["no_eligible_candidates", "candidate_not_open"]),
        ("superseded", 0, 0, ["no_eligible_candidates", "candidate_not_open"]),
    ],
)
def test_stale_and_nonopen_candidates_cannot_resume(
    tmp_path: Path,
    lifecycle_state: str,
    durable_age: int,
    runtime_age: int,
    expected_reasons: list[str],
):
    db_path = _use_database(
        tmp_path,
        f"selection-{lifecycle_state}-{durable_age}-{runtime_age}.sqlite3",
    )
    client = TestClient(app)
    conversation_id = "conversation-ineligible"
    _resolve_selection_session(
        client,
        owner_id="owner-selection",
        conversation_id=conversation_id,
    )
    if runtime_age:
        stale_runtime = (datetime.now(UTC) - timedelta(seconds=runtime_age)).isoformat()
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                UPDATE conversation_runtime_threads
                SET last_activity_at = ?
                WHERE owner_id = ? AND conversation_id = ?;
                """,
                (stale_runtime, "owner-selection", conversation_id),
            )
    before = _runtime_rows(db_path)
    response = _select_continuation(
        client,
        candidates=[
            _continuation_candidate(
                conversation_id,
                lifecycle_state=lifecycle_state,
                durable_updated_at=datetime.now(UTC) - timedelta(seconds=durable_age),
            )
        ],
    )

    assert response.status_code == 200
    assert response.json()["result"]["outcome"] == "create_new"
    assert response.json()["result"]["reason_codes"] == expected_reasons
    assert _runtime_rows(db_path) == before


@pytest.mark.parametrize("future_source", ["durable", "runtime"])
def test_future_timestamp_declines_without_mutation(
    tmp_path: Path,
    future_source: str,
):
    db_path = _use_database(tmp_path, f"selection-future-{future_source}.sqlite3")
    client = TestClient(app)
    conversation_id = "conversation-future"
    _resolve_selection_session(
        client,
        owner_id="owner-selection",
        conversation_id=conversation_id,
    )
    future_time = datetime.now(UTC) + timedelta(seconds=301)
    if future_source == "runtime":
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                UPDATE conversation_runtime_threads
                SET last_activity_at = ?
                WHERE owner_id = ? AND conversation_id = ?;
                """,
                (future_time.isoformat(), "owner-selection", conversation_id),
            )
    before = _runtime_rows(db_path)

    response = _select_continuation(
        client,
        candidates=[
            _continuation_candidate(
                conversation_id,
                durable_updated_at=(
                    future_time if future_source == "durable" else datetime.now(UTC)
                ),
            )
        ],
    )

    assert response.status_code == 200
    assert response.json()["result"]["outcome"] == "decline"
    assert response.json()["result"]["reason_codes"] == [
        "runtime_state_inconsistent"
    ]
    assert _runtime_rows(db_path) == before


def test_one_eligible_candidate_is_selected_with_nonblocking_candidates(
    tmp_path: Path,
):
    db_path = _use_database(tmp_path, "selection-one-with-others.sqlite3")
    client = TestClient(app)
    eligible_id = "conversation-eligible"
    _resolve_selection_session(
        client,
        owner_id="owner-selection",
        conversation_id=eligible_id,
    )
    before = _runtime_rows(db_path)

    response = _select_continuation(
        client,
        candidates=[
            _continuation_candidate("conversation-missing"),
            _continuation_candidate(
                "conversation-closed",
                lifecycle_state="closed",
            ),
            _continuation_candidate(
                "conversation-stale",
                durable_updated_at=datetime.now(UTC) - timedelta(hours=2),
            ),
            _continuation_candidate(eligible_id),
        ],
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["outcome"] == "resume"
    assert result["selected_conversation_id"] == eligible_id
    assert result["eligible_candidate_count"] == 1
    assert _runtime_rows(db_path) == before


def test_no_eligible_candidate_reasons_are_deterministic_and_bounded(tmp_path: Path):
    db_path = _use_database(tmp_path, "selection-reason-order.sqlite3")
    client = TestClient(app)
    before = _runtime_rows(db_path)
    candidates = [
        _continuation_candidate(
            "conversation-stale",
            durable_updated_at=datetime.now(UTC) - timedelta(hours=2),
        ),
        _continuation_candidate(
            "conversation-closed",
            lifecycle_state="closed",
        ),
        _continuation_candidate("conversation-missing"),
    ]

    forward = _select_continuation(client, candidates=candidates)
    reverse = _select_continuation(client, candidates=list(reversed(candidates)))

    assert forward.status_code == reverse.status_code == 200
    assert forward.json()["result"] == reverse.json()["result"]
    assert forward.json()["result"]["reason_codes"] == [
        "no_eligible_candidates",
        "candidate_not_open",
        "runtime_state_missing",
        "candidate_stale",
    ]
    assert _runtime_rows(db_path) == before


def test_multiple_eligible_candidates_clarify_without_recency_tiebreaker(
    tmp_path: Path,
):
    db_path = _use_database(tmp_path, "selection-multiple.sqlite3")
    client = TestClient(app)
    candidate_ids = ["conversation-first", "conversation-second"]
    for conversation_id in candidate_ids:
        _resolve_selection_session(
            client,
            owner_id="owner-selection",
            conversation_id=conversation_id,
        )
    candidates = [
        _continuation_candidate(
            candidate_ids[0],
            durable_updated_at=datetime.now(UTC) - timedelta(minutes=5),
        ),
        _continuation_candidate(candidate_ids[1]),
    ]
    before = _runtime_rows(db_path)

    forward = _select_continuation(client, candidates=candidates)
    reverse = _select_continuation(client, candidates=list(reversed(candidates)))

    assert forward.status_code == reverse.status_code == 200
    assert forward.json()["result"] == reverse.json()["result"]
    assert forward.json()["result"] == {
        "outcome": "clarify",
        "timing_policy": "ask_clarifying_question",
        "selected_conversation_id": None,
        "selected_thread_revision": None,
        "candidate_count": 2,
        "eligible_candidate_count": 2,
        "reason_codes": ["multiple_eligible_candidates"],
        "policy_version": "continuation-selection.v1",
    }
    assert _runtime_rows(db_path) == before


def test_incomplete_candidate_set_clarifies_before_inspection(tmp_path: Path):
    db_path = _use_database(tmp_path, "selection-incomplete.sqlite3")
    client = TestClient(app)
    conversation_id = "conversation-incomplete"
    _resolve_selection_session(
        client,
        owner_id="owner-selection",
        conversation_id=conversation_id,
    )
    before = _runtime_rows(db_path)

    response = _select_continuation(
        client,
        candidates=[_continuation_candidate(conversation_id)],
        candidate_set_complete=False,
    )

    assert response.status_code == 200
    assert response.json()["result"] == {
        "outcome": "clarify",
        "timing_policy": "ask_clarifying_question",
        "selected_conversation_id": None,
        "selected_thread_revision": None,
        "candidate_count": 1,
        "eligible_candidate_count": 0,
        "reason_codes": ["candidate_set_incomplete"],
        "policy_version": "continuation-selection.v1",
    }
    assert _runtime_rows(db_path) == before


def test_active_candidate_waits_and_blocks_an_idle_candidate(tmp_path: Path):
    db_path = _use_database(tmp_path, "selection-active.sqlite3")
    client = TestClient(app)
    active_id = "conversation-active"
    idle_id = "conversation-idle"
    started = _start(
        client,
        request_id="selection-active-turn",
        owner_id="owner-selection",
        conversation_id=active_id,
        surface="surface-active",
    )
    assert started.status_code == 200
    _resolve_selection_session(
        client,
        owner_id="owner-selection",
        conversation_id=idle_id,
    )
    before = _runtime_rows(db_path)

    response = _select_continuation(
        client,
        candidates=[
            _continuation_candidate(idle_id),
            _continuation_candidate(active_id),
        ],
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["outcome"] == "wait"
    assert result["timing_policy"] == "pause_or_wait"
    assert result["eligible_candidate_count"] == 1
    assert result["selected_conversation_id"] is None
    assert result["selected_thread_revision"] is None
    assert result["reason_codes"] == ["active_thread_present"]
    assert _runtime_rows(db_path) == before


@pytest.mark.parametrize(
    ("stored_state", "expected_reason"),
    [
        ("contended", "contended_thread_present"),
        ("unavailable", "unavailable_thread_present"),
        ("inconsistent", "runtime_state_inconsistent"),
    ],
)
def test_blocking_thread_state_declines_and_preserves_all_runtime_rows(
    tmp_path: Path,
    stored_state: str,
    expected_reason: str,
):
    db_path = _use_database(tmp_path, f"selection-{stored_state}.sqlite3")
    client = TestClient(app)
    blocking_id = f"conversation-{stored_state}"
    eligible_id = "conversation-safe-idle"
    if stored_state == "inconsistent":
        started = _start(
            client,
            request_id="selection-inconsistent-turn",
            owner_id="owner-selection",
            conversation_id=blocking_id,
            surface="surface-blocking",
        )
        assert started.status_code == 200
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                UPDATE conversation_runtime_threads
                SET active_runtime_turn_id = 'missing-active-turn'
                WHERE owner_id = ? AND conversation_id = ?;
                """,
                ("owner-selection", blocking_id),
            )
    else:
        _resolve_selection_session(
            client,
            owner_id="owner-selection",
            conversation_id=blocking_id,
        )
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                UPDATE conversation_runtime_threads
                SET state = ?
                WHERE owner_id = ? AND conversation_id = ?;
                """,
                (stored_state, "owner-selection", blocking_id),
            )
    _resolve_selection_session(
        client,
        owner_id="owner-selection",
        conversation_id=eligible_id,
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE conversation_runtime_events
            SET event_payload_json = '{"private":"event-sentinel"}'
            WHERE id = (SELECT MIN(id) FROM conversation_runtime_events);
            """
        )
    before = _runtime_rows(db_path)

    response = _select_continuation(
        client,
        candidates=[
            _continuation_candidate(eligible_id),
            _continuation_candidate(blocking_id),
        ],
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["outcome"] == "decline"
    assert result["timing_policy"] == "close_turn"
    assert result["selected_conversation_id"] is None
    assert result["selected_thread_revision"] is None
    assert result["reason_codes"] == [expected_reason]
    assert "event-sentinel" not in response.text
    assert _runtime_rows(db_path) == before


def test_selection_returns_revision_without_reserving_or_consuming_it(tmp_path: Path):
    db_path = _use_database(tmp_path, "selection-revision.sqlite3")
    client = TestClient(app)
    owner_id = "owner-selection"
    conversation_id = "conversation-revision-selection"
    admitted = _start(
        client,
        request_id="selection-revision-initial",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-existing",
    )
    assert admitted.status_code == 200
    completed = _complete(
        client,
        request_id="selection-revision-complete",
        runtime_session_id=admitted.json()["runtime_session"]["runtime_session_id"],
        runtime_turn_id=admitted.json()["runtime_turn"]["runtime_turn_id"],
    )
    assert completed.status_code == 200
    before = _runtime_rows(db_path)

    selection = _select_continuation(
        client,
        candidates=[_continuation_candidate(conversation_id)],
        owner_id=owner_id,
    )

    assert selection.status_code == 200
    assert selection.json()["result"]["selected_thread_revision"] == 2
    assert _runtime_rows(db_path) == before

    next_turn = _start(
        client,
        request_id="selection-revision-next",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-current",
        expected_thread_revision=2,
    )
    assert next_turn.status_code == 200
    released = _complete(
        client,
        request_id="selection-revision-release",
        runtime_session_id=next_turn.json()["runtime_session"]["runtime_session_id"],
        runtime_turn_id=next_turn.json()["runtime_turn"]["runtime_turn_id"],
    )
    assert released.status_code == 200
    stale = _start(
        client,
        request_id="selection-revision-stale",
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface="surface-current",
        expected_thread_revision=2,
    )
    assert stale.status_code == 409
    assert stale.json() == {"detail": "runtime_thread_revision_conflict"}


def test_selection_persistence_failure_is_bounded(monkeypatch):
    def fail_selection(_request):
        raise sqlite3.OperationalError("private storage detail")

    monkeypatch.setattr(main_module, "select_runtime_continuation", fail_selection)
    response = _select_continuation(
        TestClient(app),
        candidates=[_continuation_candidate("conversation-persistence")],
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "runtime_state_persistence_unavailable"}
    assert "private storage detail" not in response.text
