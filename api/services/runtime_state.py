from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from models import (
    AttentionFocus,
    ContinuationCandidate,
    ContinuationSelectionRequest,
    ContinuationSelectionResponse,
    ContinuationSelectionResult,
    RetirementReservationCancelRequest,
    RetirementReservationCancelResponse,
    RetirementReservationFinalizeRequest,
    RetirementReservationFinalizeResponse,
    RetirementReservationRequest,
    RetirementReservationResponse,
    RetirementReservationResult,
    RuntimeEvent,
    RuntimeOverlay,
    RuntimeSession,
    RuntimeSessionDiagnosticsResponse,
    RuntimeState,
    RuntimeStateUpdate,
    RuntimeThreadProjection,
    RuntimeTurn,
)

DEFAULT_RUNTIME_DB_PATH = "./data/runtime_state.sqlite3"
_TERMINAL_TURN_STATUSES = {"completed", "abandoned"}
_CONTINUATION_CLOCK_SKEW_SECONDS = 300
_CONTINUATION_REASON_ORDER = (
    "candidate_set_incomplete",
    "contended_thread_present",
    "unavailable_thread_present",
    "runtime_state_inconsistent",
    "active_thread_present",
    "multiple_eligible_candidates",
    "one_eligible_candidate",
    "no_candidates",
    "no_eligible_candidates",
    "candidate_not_open",
    "runtime_state_missing",
    "runtime_session_missing",
    "candidate_stale",
)
_RUNTIME_REPOSITORY: RuntimeStateRepository | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _digest(prefix: str, material: str) -> str:
    digest = sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _state_id(owner_id: str, conversation_id: str, surface: str) -> str:
    return _digest("rtstate", f"{owner_id}:{conversation_id}:{surface}")


def _session_id(owner_id: str, conversation_id: str, surface: str) -> str:
    return _digest("rtsession", f"{owner_id}:{conversation_id}:{surface}")


def _turn_id(runtime_session_id: str, request_id: str, created_at: str) -> str:
    return _digest("rtturn", f"{runtime_session_id}:{request_id}:{created_at}")


def _event_id(runtime_session_id: str, event_type: str, created_at: str, ordinal: int) -> str:
    return _digest("rtevent", f"{runtime_session_id}:{event_type}:{created_at}:{ordinal}")


def _retirement_reservation_id() -> str:
    return _digest("rtreservation", uuid4().hex)


def runtime_state_db_path() -> Path:
    return Path(os.environ.get("COGNITIVE_RUNTIME_DB_PATH") or DEFAULT_RUNTIME_DB_PATH)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


class RuntimeStateRepository:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or runtime_state_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_runtime_sessions (
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

                CREATE TABLE IF NOT EXISTS conversation_runtime_threads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    state TEXT NOT NULL
                        CHECK(state IN ('idle', 'active', 'contended', 'unavailable')),
                    revision INTEGER NOT NULL CHECK(revision >= 0),
                    active_runtime_session_id TEXT,
                    active_runtime_turn_id TEXT,
                    active_surface TEXT,
                    active_request_id TEXT,
                    last_activity_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(owner_id, conversation_id)
                );

                CREATE TABLE IF NOT EXISTS conversation_runtime_turns (
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

                CREATE TABLE IF NOT EXISTS conversation_runtime_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    runtime_session_id TEXT NOT NULL,
                    runtime_turn_id TEXT,
                    event_type TEXT NOT NULL,
                    event_payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(runtime_session_id)
                        REFERENCES conversation_runtime_sessions(runtime_session_id)
                );

                CREATE TABLE IF NOT EXISTS conversation_runtime_retirement_reservations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reservation_id TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    thread_revision INTEGER NOT NULL CHECK(thread_revision >= 0),
                    durable_updated_at TEXT NOT NULL,
                    retirement_before TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(owner_id, conversation_id),
                    FOREIGN KEY(owner_id, conversation_id)
                        REFERENCES conversation_runtime_threads(owner_id, conversation_id)
                );
                """
            )

    def resolve_thread(
        self,
        *,
        owner_id: str,
        conversation_id: str,
    ) -> RuntimeThreadProjection:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            row = self._ensure_thread(
                conn,
                owner_id=owner_id,
                conversation_id=conversation_id,
            )
            return self._thread_from_row(conn, row)

    def select_continuation(
        self,
        request: ContinuationSelectionRequest,
        *,
        evaluated_at: datetime | None = None,
    ) -> ContinuationSelectionResponse:
        if not request.candidate_set_complete:
            return self._continuation_response(
                request,
                outcome="clarify",
                eligible_candidate_count=0,
                reason_codes={"candidate_set_incomplete"},
            )

        now = evaluated_at or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("continuation_evaluation_timezone_required")
        now = now.astimezone(UTC)

        evaluations: list[dict[str, Any]] = []
        with self._connect() as conn:
            conn.execute("BEGIN;")
            for candidate in request.candidates:
                evaluations.append(
                    self._evaluate_continuation_candidate(
                        conn,
                        owner_id=request.owner_id,
                        candidate=candidate,
                        stale_after_seconds=request.stale_after_seconds,
                        evaluated_at=now,
                    )
                )

        eligible = [item for item in evaluations if item["eligible"]]
        blocking_reasons = {
            reason
            for item in evaluations
            for reason in item["reason_codes"]
            if reason
            in {
                "contended_thread_present",
                "unavailable_thread_present",
                "runtime_state_inconsistent",
            }
        }
        if blocking_reasons:
            return self._continuation_response(
                request,
                outcome="decline",
                eligible_candidate_count=len(eligible),
                reason_codes=blocking_reasons,
            )
        if any("active_thread_present" in item["reason_codes"] for item in evaluations):
            return self._continuation_response(
                request,
                outcome="wait",
                eligible_candidate_count=len(eligible),
                reason_codes={"active_thread_present"},
            )
        if len(eligible) > 1:
            return self._continuation_response(
                request,
                outcome="clarify",
                eligible_candidate_count=len(eligible),
                reason_codes={"multiple_eligible_candidates"},
            )
        if len(eligible) == 1:
            selected = eligible[0]
            return self._continuation_response(
                request,
                outcome="resume",
                eligible_candidate_count=1,
                reason_codes={"one_eligible_candidate"},
                selected_conversation_id=selected["conversation_id"],
                selected_thread_revision=selected["revision"],
            )
        if not evaluations:
            reasons = {"no_candidates"}
        else:
            reasons = {
                "no_eligible_candidates",
                *(
                    reason
                    for item in evaluations
                    for reason in item["reason_codes"]
                    if reason
                    in {
                        "candidate_not_open",
                        "runtime_state_missing",
                        "runtime_session_missing",
                        "candidate_stale",
                    }
                ),
            }
        return self._continuation_response(
            request,
            outcome="create_new",
            eligible_candidate_count=0,
            reason_codes=reasons,
        )

    def reserve_retirement(
        self,
        request: RetirementReservationRequest,
    ) -> RetirementReservationResponse:
        durable_updated_at = request.durable_updated_at.astimezone(UTC)
        retirement_before = request.retirement_before.astimezone(UTC)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            existing = self._retirement_reservation_by_key(
                conn,
                owner_id=request.owner_id,
                conversation_id=request.conversation_id,
            )
            if existing is not None:
                if not self._retirement_reservation_invariant_holds(conn, existing):
                    return self._retirement_reservation_response(
                        request,
                        outcome="decline",
                        reason="runtime_state_inconsistent",
                    )
                try:
                    reserved_durable_updated_at = datetime.fromisoformat(
                        existing["durable_updated_at"]
                    )
                    if (
                        reserved_durable_updated_at.tzinfo is None
                        or reserved_durable_updated_at.utcoffset() is None
                    ):
                        raise ValueError("retirement_reservation_timestamp_invalid")
                except (TypeError, ValueError):
                    return self._retirement_reservation_response(
                        request,
                        outcome="decline",
                        reason="runtime_state_inconsistent",
                    )
                return self._retirement_reservation_response(
                    request,
                    outcome="reserved",
                    reason="existing_retirement_reservation",
                    reservation_id=existing["reservation_id"],
                    reserved_thread_revision=existing["thread_revision"],
                    reserved_durable_updated_at=reserved_durable_updated_at.astimezone(UTC),
                )

            if request.lifecycle_state != "open":
                return self._retirement_reservation_response(
                    request,
                    outcome="decline",
                    reason="candidate_not_open",
                )
            if not durable_updated_at < retirement_before:
                return self._retirement_reservation_response(
                    request,
                    outcome="decline",
                    reason="durable_activity_not_over_horizon",
                )

            inspection = self._inspect_continuation_thread(
                conn,
                owner_id=request.owner_id,
                conversation_id=request.conversation_id,
            )
            state = inspection["state"]
            if state == "missing":
                return self._retirement_reservation_response(
                    request,
                    outcome="decline",
                    reason="runtime_state_missing",
                )
            if state == "inconsistent":
                return self._retirement_reservation_response(
                    request,
                    outcome="decline",
                    reason="runtime_state_inconsistent",
                )
            if state == "active":
                return self._retirement_reservation_response(
                    request,
                    outcome="wait",
                    reason="runtime_thread_active",
                )
            if state == "contended":
                return self._retirement_reservation_response(
                    request,
                    outcome="decline",
                    reason="runtime_thread_contended",
                )
            if state == "unavailable":
                return self._retirement_reservation_response(
                    request,
                    outcome="decline",
                    reason="runtime_thread_unavailable",
                )

            runtime_activity = inspection["last_activity_at"]
            if runtime_activity is None or not runtime_activity < retirement_before:
                return self._retirement_reservation_response(
                    request,
                    outcome="decline",
                    reason="runtime_activity_not_over_horizon",
                )

            thread_revision = inspection["revision"]
            assert isinstance(thread_revision, int) and thread_revision >= 0
            created_at = _now()
            durable_value = durable_updated_at.isoformat()
            retirement_value = retirement_before.isoformat()
            reservation_id = _retirement_reservation_id()
            conn.execute(
                """
                INSERT INTO conversation_runtime_retirement_reservations (
                    reservation_id, owner_id, conversation_id, thread_revision,
                    durable_updated_at, retirement_before, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    reservation_id,
                    request.owner_id,
                    request.conversation_id,
                    thread_revision,
                    durable_value,
                    retirement_value,
                    created_at,
                ),
            )
            return self._retirement_reservation_response(
                request,
                outcome="reserved",
                reason="safe_idle_retirement_reserved",
                reservation_id=reservation_id,
                reserved_thread_revision=thread_revision,
                reserved_durable_updated_at=durable_updated_at,
            )

    def cancel_retirement_reservation(
        self,
        request: RetirementReservationCancelRequest,
    ) -> RetirementReservationCancelResponse:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            reservation = self._required_retirement_reservation(
                conn,
                owner_id=request.owner_id,
                conversation_id=request.conversation_id,
            )
            self._validate_retirement_reservation_identity(
                reservation,
                reservation_id=request.reservation_id,
                reserved_thread_revision=request.reserved_thread_revision,
            )
            if not self._retirement_reservation_invariant_holds(conn, reservation):
                raise RuntimeError("runtime_retirement_reservation_invariant_conflict")
            deleted = conn.execute(
                """
                DELETE FROM conversation_runtime_retirement_reservations
                WHERE owner_id = ? AND conversation_id = ?
                  AND reservation_id = ? AND thread_revision = ?;
                """,
                (
                    request.owner_id,
                    request.conversation_id,
                    request.reservation_id,
                    request.reserved_thread_revision,
                ),
            )
            if deleted.rowcount != 1:
                raise RuntimeError("runtime_retirement_reservation_invariant_conflict")
            return RetirementReservationCancelResponse(
                request_id=request.request_id,
                owner_id=request.owner_id,
                conversation_id=request.conversation_id,
                reservation_id=request.reservation_id,
                thread_revision=request.reserved_thread_revision,
            )

    def finalize_retirement_reservation(
        self,
        request: RetirementReservationFinalizeRequest,
    ) -> RetirementReservationFinalizeResponse:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            reservation = self._required_retirement_reservation(
                conn,
                owner_id=request.owner_id,
                conversation_id=request.conversation_id,
            )
            self._validate_retirement_reservation_identity(
                reservation,
                reservation_id=request.reservation_id,
                reserved_thread_revision=request.reserved_thread_revision,
            )
            if not self._retirement_reservation_invariant_holds(conn, reservation):
                raise RuntimeError("runtime_retirement_reservation_invariant_conflict")

            fenced_revision = request.reserved_thread_revision + 1
            updated = conn.execute(
                """
                UPDATE conversation_runtime_threads
                SET revision = ?, updated_at = ?
                WHERE owner_id = ? AND conversation_id = ?
                  AND state = 'idle' AND revision = ?;
                """,
                (
                    fenced_revision,
                    _now(),
                    request.owner_id,
                    request.conversation_id,
                    request.reserved_thread_revision,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("runtime_retirement_reservation_invariant_conflict")
            deleted = conn.execute(
                """
                DELETE FROM conversation_runtime_retirement_reservations
                WHERE owner_id = ? AND conversation_id = ?
                  AND reservation_id = ? AND thread_revision = ?;
                """,
                (
                    request.owner_id,
                    request.conversation_id,
                    request.reservation_id,
                    request.reserved_thread_revision,
                ),
            )
            if deleted.rowcount != 1:
                raise RuntimeError("runtime_retirement_reservation_invariant_conflict")
            return RetirementReservationFinalizeResponse(
                request_id=request.request_id,
                owner_id=request.owner_id,
                conversation_id=request.conversation_id,
                reservation_id=request.reservation_id,
                previous_thread_revision=request.reserved_thread_revision,
                fenced_thread_revision=fenced_revision,
            )

    @staticmethod
    def _retirement_reservation_response(
        request: RetirementReservationRequest,
        *,
        outcome: str,
        reason: str,
        reservation_id: str | None = None,
        reserved_thread_revision: int | None = None,
        reserved_durable_updated_at: datetime | None = None,
    ) -> RetirementReservationResponse:
        return RetirementReservationResponse(
            request_id=request.request_id,
            owner_id=request.owner_id,
            conversation_id=request.conversation_id,
            result=RetirementReservationResult(
                outcome=outcome,
                reservation_id=reservation_id,
                reserved_thread_revision=reserved_thread_revision,
                reserved_durable_updated_at=reserved_durable_updated_at,
                reason_codes=[reason],
            ),
        )

    def _evaluate_continuation_candidate(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        candidate: ContinuationCandidate,
        stale_after_seconds: int,
        evaluated_at: datetime,
    ) -> dict[str, Any]:
        reasons: set[str] = set()
        durable_time = candidate.durable_updated_at.astimezone(UTC)
        if self._continuation_time_is_future(durable_time, evaluated_at):
            reasons.add("runtime_state_inconsistent")
        elif self._continuation_time_is_stale(
            durable_time,
            evaluated_at,
            stale_after_seconds,
        ):
            reasons.add("candidate_stale")
        if candidate.lifecycle_state != "open":
            reasons.add("candidate_not_open")

        inspection = self._inspect_continuation_thread(
            conn,
            owner_id=owner_id,
            conversation_id=candidate.conversation_id,
        )
        state = inspection["state"]
        if state == "missing":
            reasons.add("runtime_state_missing")
        elif state == "inconsistent":
            reasons.add("runtime_state_inconsistent")
        elif state == "contended":
            reasons.add("contended_thread_present")
        elif state == "unavailable":
            reasons.add("unavailable_thread_present")
        elif state == "active":
            reasons.add("active_thread_present")
        elif inspection["participating_session_count"] == 0:
            reasons.add("runtime_session_missing")

        runtime_time = inspection.get("last_activity_at")
        if runtime_time is not None:
            if self._continuation_time_is_future(runtime_time, evaluated_at):
                reasons.add("runtime_state_inconsistent")
            elif self._continuation_time_is_stale(
                runtime_time,
                evaluated_at,
                stale_after_seconds,
            ):
                reasons.add("candidate_stale")

        eligible = not reasons and state == "idle"
        return {
            "conversation_id": candidate.conversation_id,
            "eligible": eligible,
            "reason_codes": reasons,
            "revision": inspection.get("revision"),
        }

    def _inspect_continuation_thread(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        row = self._thread_by_key(
            conn,
            owner_id=owner_id,
            conversation_id=conversation_id,
        )
        if row is None:
            return {
                "state": "missing",
                "revision": None,
                "participating_session_count": 0,
                "participating_surfaces": (),
                "last_activity_at": None,
            }

        session_rows = conn.execute(
            """
            SELECT surface FROM conversation_runtime_sessions
            WHERE owner_id = ? AND conversation_id = ?
            ORDER BY surface ASC;
            """,
            (owner_id, conversation_id),
        ).fetchall()
        active_rows = self._non_terminal_turn_rows(
            conn,
            owner_id=owner_id,
            conversation_id=conversation_id,
        )
        state = row["state"]
        revision = row["revision"]
        revision_is_valid = (
            isinstance(revision, int) and not isinstance(revision, bool) and revision >= 0
        )
        if state == "idle":
            consistent = (
                revision_is_valid
                and not active_rows
                and all(
                    row[field] is None
                    for field in (
                        "active_runtime_session_id",
                        "active_runtime_turn_id",
                        "active_surface",
                        "active_request_id",
                    )
                )
            )
        elif state == "active" and len(active_rows) == 1:
            active_row = active_rows[0]
            consistent = revision_is_valid and all(
                (
                    row["active_runtime_session_id"] == active_row["runtime_session_id"],
                    row["active_runtime_turn_id"] == active_row["runtime_turn_id"],
                    row["active_surface"] == active_row["surface"],
                )
            )
        elif state in {"contended", "unavailable"}:
            consistent = revision_is_valid
        else:
            consistent = False

        try:
            last_activity_at = datetime.fromisoformat(row["last_activity_at"])
            if last_activity_at.tzinfo is None or last_activity_at.utcoffset() is None:
                consistent = False
                last_activity_at = None
            else:
                last_activity_at = last_activity_at.astimezone(UTC)
        except (TypeError, ValueError):
            consistent = False
            last_activity_at = None

        return {
            "state": state if consistent else "inconsistent",
            "revision": revision,
            "participating_session_count": len(session_rows),
            "participating_surfaces": tuple(
                sorted({session_row["surface"] for session_row in session_rows})
            ),
            "last_activity_at": last_activity_at,
        }

    @staticmethod
    def _continuation_time_is_future(value: datetime, evaluated_at: datetime) -> bool:
        return (value - evaluated_at).total_seconds() > _CONTINUATION_CLOCK_SKEW_SECONDS

    @staticmethod
    def _continuation_time_is_stale(
        value: datetime,
        evaluated_at: datetime,
        stale_after_seconds: int,
    ) -> bool:
        return (evaluated_at - value).total_seconds() > stale_after_seconds

    @staticmethod
    def _continuation_response(
        request: ContinuationSelectionRequest,
        *,
        outcome: str,
        eligible_candidate_count: int,
        reason_codes: set[str],
        selected_conversation_id: str | None = None,
        selected_thread_revision: int | None = None,
    ) -> ContinuationSelectionResponse:
        timing_policy = {
            "resume": "resume_previous_thread",
            "create_new": "answer_now",
            "clarify": "ask_clarifying_question",
            "wait": "pause_or_wait",
            "decline": "close_turn",
        }[outcome]
        ordered_reasons = [
            reason for reason in _CONTINUATION_REASON_ORDER if reason in reason_codes
        ]
        return ContinuationSelectionResponse(
            request_id=request.request_id,
            owner_id=request.owner_id,
            surface=request.surface,
            result=ContinuationSelectionResult(
                outcome=outcome,
                timing_policy=timing_policy,
                selected_conversation_id=selected_conversation_id,
                selected_thread_revision=selected_thread_revision,
                candidate_count=len(request.candidates),
                eligible_candidate_count=eligible_candidate_count,
                reason_codes=ordered_reasons,
            ),
        )

    def _ensure_thread(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        conversation_id: str,
    ) -> sqlite3.Row:
        row = self._thread_by_key(
            conn,
            owner_id=owner_id,
            conversation_id=conversation_id,
        )
        if row is None:
            active_rows = self._non_terminal_turn_rows(
                conn,
                owner_id=owner_id,
                conversation_id=conversation_id,
            )
            now = _now()
            state = "idle"
            active_runtime_session_id = None
            active_runtime_turn_id = None
            active_surface = None
            if len(active_rows) == 1:
                state = "active"
                active_runtime_session_id = active_rows[0]["runtime_session_id"]
                active_runtime_turn_id = active_rows[0]["runtime_turn_id"]
                active_surface = active_rows[0]["surface"]
            elif len(active_rows) > 1:
                state = "contended"
            conn.execute(
                """
                INSERT INTO conversation_runtime_threads (
                    owner_id, conversation_id, state, revision,
                    active_runtime_session_id, active_runtime_turn_id,
                    active_surface, active_request_id, last_activity_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    owner_id,
                    conversation_id,
                    state,
                    0,
                    active_runtime_session_id,
                    active_runtime_turn_id,
                    active_surface,
                    None,
                    now,
                    now,
                    now,
                ),
            )
            row = self._thread_by_key(
                conn,
                owner_id=owner_id,
                conversation_id=conversation_id,
            )
            assert row is not None
        return self._validated_thread_row(conn, row)

    def _validated_thread_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> sqlite3.Row:
        if row["state"] in {"contended", "unavailable"}:
            return row

        active_rows = self._non_terminal_turn_rows(
            conn,
            owner_id=row["owner_id"],
            conversation_id=row["conversation_id"],
        )
        consistent = not active_rows if row["state"] == "idle" else False
        if row["state"] == "active" and len(active_rows) == 1:
            active_row = active_rows[0]
            consistent = all(
                (
                    row["active_runtime_session_id"] == active_row["runtime_session_id"],
                    row["active_runtime_turn_id"] == active_row["runtime_turn_id"],
                    row["active_surface"] == active_row["surface"],
                )
            )

        if consistent:
            return row

        now = _now()
        conn.execute(
            """
            UPDATE conversation_runtime_threads
            SET state = 'unavailable', updated_at = ?
            WHERE owner_id = ? AND conversation_id = ?;
            """,
            (now, row["owner_id"], row["conversation_id"]),
        )
        unavailable = self._thread_by_key(
            conn,
            owner_id=row["owner_id"],
            conversation_id=row["conversation_id"],
        )
        assert unavailable is not None
        return unavailable

    def _resolve_session_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        request_id: str,
        owner_id: str,
        conversation_id: str,
        surface: str,
        surface_session_id: str | None,
        active_mode: str | None,
        record_event: bool = True,
    ) -> RuntimeSession:
        now = _now()
        row = conn.execute(
            """
            SELECT * FROM conversation_runtime_sessions
            WHERE owner_id = ? AND conversation_id = ? AND surface = ?
            LIMIT 1;
            """,
            (owner_id, conversation_id, surface),
        ).fetchone()
        if row is None:
            runtime_session_id = _session_id(owner_id, conversation_id, surface)
            runtime_state_id = _state_id(owner_id, conversation_id, surface)
            conn.execute(
                """
                INSERT INTO conversation_runtime_sessions (
                    runtime_session_id, runtime_state_id, owner_id, conversation_id,
                    surface, surface_session_id, status, active_mode, attention_state,
                    active_scene, interaction_mode, attention_focus_json,
                    temporary_constraints_json, reset_after_turn, trace_refs_json,
                    started_at, last_activity_at, closed_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    runtime_session_id,
                    runtime_state_id,
                    owner_id,
                    conversation_id,
                    surface,
                    surface_session_id,
                    "active",
                    active_mode,
                    None,
                    None,
                    None,
                    _json(None),
                    _json([]),
                    0,
                    _json([]),
                    now,
                    now,
                    None,
                    now,
                    now,
                ),
            )
        else:
            updates: dict[str, Any] = {
                "last_activity_at": now,
                "updated_at": now,
            }
            if surface_session_id is not None:
                updates["surface_session_id"] = surface_session_id
            if active_mode is not None:
                updates["active_mode"] = active_mode
            self._update_session_row(conn, row["runtime_session_id"], updates)

        session = self._session_by_key(
            conn,
            owner_id=owner_id,
            conversation_id=conversation_id,
            surface=surface,
        )
        assert session is not None
        if record_event:
            self._record_event(
                conn,
                runtime_session_id=session.runtime_session_id,
                runtime_turn_id=None,
                event_type="session_resolved",
                event_payload_json={
                    "request_id": request_id,
                    "surface": surface,
                    "conversation_id": conversation_id,
                },
            )
        return session

    def resolve_session(
        self,
        *,
        request_id: str,
        owner_id: str,
        conversation_id: str,
        surface: str,
        surface_session_id: str | None = None,
        active_mode: str | None = None,
    ) -> RuntimeSession:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            self._raise_if_retirement_reserved(
                conn,
                owner_id=owner_id,
                conversation_id=conversation_id,
            )
            self._ensure_thread(
                conn,
                owner_id=owner_id,
                conversation_id=conversation_id,
            )
            session = self._resolve_session_in_transaction(
                conn,
                request_id=request_id,
                owner_id=owner_id,
                conversation_id=conversation_id,
                surface=surface,
                surface_session_id=surface_session_id,
                active_mode=active_mode,
            )
            now = _now()
            conn.execute(
                """
                UPDATE conversation_runtime_threads
                SET last_activity_at = ?, updated_at = ?
                WHERE owner_id = ? AND conversation_id = ?;
                """,
                (now, now, owner_id, conversation_id),
            )
            return session

    def session_by_id(self, runtime_session_id: str) -> RuntimeSession | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversation_runtime_sessions WHERE runtime_session_id = ? LIMIT 1;",
                (runtime_session_id,),
            ).fetchone()
        if row is None:
            return None
        return self._session_from_row(row)

    def turn_by_id(self, runtime_turn_id: str) -> RuntimeTurn | None:
        with self._connect() as conn:
            return self._turn_by_id(conn, runtime_turn_id)

    def resolve_state(self, *, owner_id: str, conversation_id: str, surface: str) -> RuntimeState:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM conversation_runtime_sessions
                WHERE owner_id = ? AND conversation_id = ? AND surface = ?
                LIMIT 1;
                """,
                (owner_id, conversation_id, surface),
            ).fetchone()
        if row is None:
            session = self.resolve_session(
                request_id="runtime-state-bootstrap",
                owner_id=owner_id,
                conversation_id=conversation_id,
                surface=surface,
            )
            return self.runtime_state_by_session_id(session.runtime_session_id)
        return self._runtime_state_from_row(row)

    def runtime_state_by_session_id(self, runtime_session_id: str) -> RuntimeState:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversation_runtime_sessions WHERE runtime_session_id = ? LIMIT 1;",
                (runtime_session_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("runtime_session_not_found")
        return self._runtime_state_from_row(row)

    def update_state(
        self,
        *,
        owner_id: str,
        conversation_id: str,
        surface: str,
        updates: RuntimeStateUpdate,
    ) -> RuntimeState:
        state = self.resolve_state(
            owner_id=owner_id,
            conversation_id=conversation_id,
            surface=surface,
        )
        payload = updates.model_dump(exclude_unset=True, exclude_none=True)
        db_updates: dict[str, Any] = {"updated_at": _now(), "last_activity_at": _now()}
        if "active_scene" in payload:
            db_updates["active_scene"] = payload["active_scene"]
        if "interaction_mode" in payload:
            db_updates["interaction_mode"] = payload["interaction_mode"]
        if "attention_focus" in payload:
            db_updates["attention_focus_json"] = _json(payload["attention_focus"])
            db_updates["attention_state"] = payload["attention_focus"].get("status")
        if "temporary_constraints" in payload:
            db_updates["temporary_constraints_json"] = _json(payload["temporary_constraints"])
        if "reset_after_turn" in payload:
            db_updates["reset_after_turn"] = int(payload["reset_after_turn"])
        if "trace_refs" in payload:
            db_updates["trace_refs_json"] = _json(payload["trace_refs"])

        with self._connect() as conn:
            self._update_session_row(
                conn,
                state.runtime_state_id.replace("rtstate", "rtsession", 1),
                db_updates,
            )
        return self.resolve_state(
            owner_id=owner_id,
            conversation_id=conversation_id,
            surface=surface,
        )

    def reset_state(self, *, owner_id: str, conversation_id: str, surface: str) -> RuntimeState:
        state = self.resolve_state(
            owner_id=owner_id,
            conversation_id=conversation_id,
            surface=surface,
        )
        with self._connect() as conn:
            self._update_session_row(
                conn,
                state.runtime_state_id.replace("rtstate", "rtsession", 1),
                {
                    "active_scene": None,
                    "interaction_mode": None,
                    "attention_state": None,
                    "attention_focus_json": _json(None),
                    "temporary_constraints_json": _json([]),
                    "reset_after_turn": 0,
                    "trace_refs_json": _json([]),
                    "updated_at": _now(),
                    "last_activity_at": _now(),
                },
            )
        return self.resolve_state(
            owner_id=owner_id,
            conversation_id=conversation_id,
            surface=surface,
        )

    def start_turn(
        self,
        *,
        request_id: str,
        owner_id: str,
        conversation_id: str,
        surface: str,
        surface_session_id: str | None = None,
        active_mode: str | None = None,
        input_message_id: str | None = None,
        intent_class: str | None = None,
        timing_policy: str | None = None,
        restraint_policy: str | None = None,
        continuation_state: str | None = None,
        expected_thread_revision: int | None = None,
    ) -> tuple[RuntimeSession, RuntimeTurn, RuntimeEvent]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            self._raise_if_retirement_reserved(
                conn,
                owner_id=owner_id,
                conversation_id=conversation_id,
            )
            thread = self._ensure_thread(
                conn,
                owner_id=owner_id,
                conversation_id=conversation_id,
            )
            if thread["state"] == "active":
                if (
                    thread["active_request_id"] == request_id
                    and thread["active_surface"] == surface
                ):
                    session = self._session_by_id(conn, thread["active_runtime_session_id"])
                    turn = self._turn_by_id(conn, thread["active_runtime_turn_id"])
                    event = self._turn_event(
                        conn,
                        runtime_turn_id=thread["active_runtime_turn_id"],
                        event_type="turn_started",
                        request_id=request_id,
                    )
                    if session is None or turn is None or event is None:
                        raise RuntimeError("runtime_thread_unavailable")
                    return session, turn, event
                raise RuntimeError("runtime_thread_contended")
            if thread["state"] == "contended":
                raise RuntimeError("runtime_thread_contended")
            if thread["state"] == "unavailable":
                raise RuntimeError("runtime_thread_unavailable")
            if expected_thread_revision is not None and (
                expected_thread_revision != thread["revision"]
            ):
                raise RuntimeError("runtime_thread_revision_conflict")

            session = self._resolve_session_in_transaction(
                conn,
                request_id=request_id,
                owner_id=owner_id,
                conversation_id=conversation_id,
                surface=surface,
                surface_session_id=surface_session_id,
                active_mode=active_mode,
            )
            created_at = _now()
            runtime_turn_id = _turn_id(session.runtime_session_id, request_id, created_at)
            conn.execute(
                """
                INSERT INTO conversation_runtime_turns (
                    runtime_turn_id, runtime_session_id, input_message_id, turn_status,
                    intent_class, timing_policy, restraint_policy, continuation_state,
                    created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    runtime_turn_id,
                    session.runtime_session_id,
                    input_message_id,
                    "received",
                    intent_class,
                    timing_policy,
                    restraint_policy,
                    continuation_state,
                    created_at,
                    created_at,
                    None,
                ),
            )
            self._update_session_row(
                conn,
                session.runtime_session_id,
                {
                    "status": "active",
                    "last_activity_at": created_at,
                    "updated_at": created_at,
                    "surface_session_id": surface_session_id,
                },
            )
            event = self._record_event(
                conn,
                runtime_session_id=session.runtime_session_id,
                runtime_turn_id=runtime_turn_id,
                event_type="turn_started",
                event_payload_json={
                    "request_id": request_id,
                    "turn_status": "received",
                    "input_message_id": input_message_id,
                },
            )
            thread_update = conn.execute(
                """
                UPDATE conversation_runtime_threads
                SET state = 'active', revision = revision + 1,
                    active_runtime_session_id = ?, active_runtime_turn_id = ?,
                    active_surface = ?, active_request_id = ?,
                    last_activity_at = ?, updated_at = ?
                WHERE owner_id = ? AND conversation_id = ?;
                """,
                (
                    session.runtime_session_id,
                    runtime_turn_id,
                    surface,
                    request_id,
                    created_at,
                    created_at,
                    owner_id,
                    conversation_id,
                ),
            )
            if thread_update.rowcount != 1:
                raise RuntimeError("runtime_thread_unavailable")
            turn = self._turn_by_id(conn, runtime_turn_id)
            updated_session = self._session_by_id(conn, session.runtime_session_id)
        assert turn is not None and updated_session is not None
        return updated_session, turn, event

    def update_turn(
        self,
        *,
        request_id: str,
        runtime_session_id: str,
        runtime_turn_id: str,
        turn_status: str,
        timing_policy: str | None = None,
        restraint_policy: str | None = None,
        continuation_state: str | None = None,
    ) -> tuple[RuntimeSession, RuntimeTurn, RuntimeEvent]:
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            session = self._session_by_id(conn, runtime_session_id)
            if session is None:
                raise RuntimeError("runtime_session_not_found")
            turn = self._turn_by_id(conn, runtime_turn_id)
            if turn is None:
                raise RuntimeError("runtime_turn_not_found")
            if turn.runtime_session_id != runtime_session_id:
                raise RuntimeError("runtime_turn_session_mismatch")
            if turn.turn_status in _TERMINAL_TURN_STATUSES:
                raise RuntimeError("runtime_turn_not_current")
            if turn_status in _TERMINAL_TURN_STATUSES:
                raise RuntimeError("runtime_turn_not_current")
            self._validate_current_turn(conn, session=session, turn=turn)
            updates: dict[str, Any] = {
                "turn_status": turn_status,
                "updated_at": now,
            }
            if timing_policy is not None:
                updates["timing_policy"] = timing_policy
            if restraint_policy is not None:
                updates["restraint_policy"] = restraint_policy
            if continuation_state is not None:
                updates["continuation_state"] = continuation_state
            if turn_status in _TERMINAL_TURN_STATUSES:
                updates["completed_at"] = now
            self._update_turn_row(conn, runtime_turn_id, updates)
            self._update_session_row(
                conn,
                runtime_session_id,
                {
                    "status": "active",
                    "last_activity_at": now,
                    "updated_at": now,
                },
            )
            event = self._record_event(
                conn,
                runtime_session_id=runtime_session_id,
                runtime_turn_id=runtime_turn_id,
                event_type="turn_updated",
                event_payload_json={
                    "request_id": request_id,
                    "turn_status": turn_status,
                },
            )
            thread_update = conn.execute(
                """
                UPDATE conversation_runtime_threads
                SET last_activity_at = ?, updated_at = ?
                WHERE owner_id = ? AND conversation_id = ?;
                """,
                (now, now, session.owner_id, session.conversation_id),
            )
            if thread_update.rowcount != 1:
                raise RuntimeError("runtime_thread_unavailable")
            updated_turn = self._turn_by_id(conn, runtime_turn_id)
            updated_session = self._session_by_id(conn, runtime_session_id)
        assert updated_turn is not None and updated_session is not None
        return updated_session, updated_turn, event

    def complete_turn(
        self,
        *,
        request_id: str,
        runtime_session_id: str,
        runtime_turn_id: str,
        turn_status: str,
        continuation_state: str | None = None,
    ) -> tuple[RuntimeSession, RuntimeTurn, RuntimeEvent]:
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            session = self._session_by_id(conn, runtime_session_id)
            if session is None:
                raise RuntimeError("runtime_session_not_found")
            turn = self._turn_by_id(conn, runtime_turn_id)
            if turn is None:
                raise RuntimeError("runtime_turn_not_found")
            if turn.runtime_session_id != runtime_session_id:
                raise RuntimeError("runtime_turn_session_mismatch")
            if turn.turn_status in _TERMINAL_TURN_STATUSES:
                event = self._turn_event(
                    conn,
                    runtime_turn_id=runtime_turn_id,
                    event_type="turn_completed",
                    request_id=request_id,
                    turn_status=turn_status,
                )
                if (
                    turn.turn_status == turn_status
                    and event is not None
                    and event.event_payload_json.get("continuation_state")
                    == continuation_state
                ):
                    return session, turn, event
                raise RuntimeError("runtime_turn_not_current")

            thread = self._validate_current_turn(conn, session=session, turn=turn)
            self._update_turn_row(
                conn,
                runtime_turn_id,
                {
                    "turn_status": turn_status,
                    "continuation_state": continuation_state,
                    "updated_at": now,
                    "completed_at": now,
                },
            )
            self._update_session_row(
                conn,
                runtime_session_id,
                {
                    "status": "active",
                    "last_activity_at": now,
                    "updated_at": now,
                },
            )
            event = self._record_event(
                conn,
                runtime_session_id=runtime_session_id,
                runtime_turn_id=runtime_turn_id,
                event_type="turn_completed",
                event_payload_json={
                    "request_id": request_id,
                    "turn_status": turn_status,
                    "continuation_state": continuation_state,
                },
            )
            thread_update = conn.execute(
                """
                UPDATE conversation_runtime_threads
                SET state = 'idle', revision = revision + 1,
                    active_runtime_session_id = NULL,
                    active_runtime_turn_id = NULL,
                    active_surface = NULL, active_request_id = NULL,
                    last_activity_at = ?, updated_at = ?
                WHERE owner_id = ? AND conversation_id = ? AND revision = ?;
                """,
                (
                    now,
                    now,
                    session.owner_id,
                    session.conversation_id,
                    thread["revision"],
                ),
            )
            if thread_update.rowcount != 1:
                raise RuntimeError("runtime_thread_unavailable")
            updated_turn = self._turn_by_id(conn, runtime_turn_id)
            updated_session = self._session_by_id(conn, runtime_session_id)
        assert updated_turn is not None and updated_session is not None
        return updated_session, updated_turn, event

    def _validate_current_turn(
        self,
        conn: sqlite3.Connection,
        *,
        session: RuntimeSession,
        turn: RuntimeTurn,
    ) -> sqlite3.Row:
        thread = self._ensure_thread(
            conn,
            owner_id=session.owner_id,
            conversation_id=session.conversation_id,
        )
        if thread["state"] == "contended":
            raise RuntimeError("runtime_thread_contended")
        if thread["state"] == "unavailable":
            raise RuntimeError("runtime_thread_unavailable")
        if (
            thread["state"] != "active"
            or thread["active_runtime_session_id"] != session.runtime_session_id
            or thread["active_runtime_turn_id"] != turn.runtime_turn_id
            or thread["active_surface"] != session.surface
        ):
            raise RuntimeError("runtime_turn_not_current")
        return thread

    def _turn_event(
        self,
        conn: sqlite3.Connection,
        *,
        runtime_turn_id: str,
        event_type: str,
        request_id: str,
        turn_status: str | None = None,
    ) -> RuntimeEvent | None:
        rows = conn.execute(
            """
            SELECT * FROM conversation_runtime_events
            WHERE runtime_turn_id = ? AND event_type = ?
            ORDER BY id ASC;
            """,
            (runtime_turn_id, event_type),
        ).fetchall()
        for row in rows:
            event = self._event_from_row(row)
            if event.event_payload_json.get("request_id") != request_id:
                continue
            if turn_status is not None and (
                event.event_payload_json.get("turn_status") != turn_status
            ):
                continue
            return event
        return None

    def record_session_event(
        self,
        *,
        runtime_session_id: str,
        runtime_turn_id: str | None,
        event_type: str,
        event_payload_json: dict[str, Any],
    ) -> RuntimeEvent:
        with self._connect() as conn:
            return self._record_event(
                conn,
                runtime_session_id=runtime_session_id,
                runtime_turn_id=runtime_turn_id,
                event_type=event_type,
                event_payload_json=event_payload_json,
            )

    def update_turn_intent_class(
        self,
        *,
        runtime_session_id: str,
        runtime_turn_id: str,
        intent_class: str,
    ) -> RuntimeTurn:
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            session = self._session_by_id(conn, runtime_session_id)
            if session is None:
                raise RuntimeError("runtime_session_not_found")
            turn = self._turn_by_id(conn, runtime_turn_id)
            if turn is None:
                raise RuntimeError("runtime_turn_not_found")
            if turn.runtime_session_id != runtime_session_id:
                raise RuntimeError("runtime_turn_session_mismatch")
            if turn.turn_status in _TERMINAL_TURN_STATUSES:
                raise RuntimeError("runtime_turn_not_current")
            self._validate_current_turn(conn, session=session, turn=turn)
            self._update_turn_row(
                conn,
                runtime_turn_id,
                {
                    "intent_class": intent_class,
                    "updated_at": now,
                },
            )
            updated_turn = self._turn_by_id(conn, runtime_turn_id)
        assert updated_turn is not None
        return updated_turn

    def update_turn_restraint_policy(
        self,
        *,
        runtime_session_id: str,
        runtime_turn_id: str,
        restraint_policy: str,
    ) -> RuntimeTurn:
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            session = self._session_by_id(conn, runtime_session_id)
            if session is None:
                raise RuntimeError("runtime_session_not_found")
            turn = self._turn_by_id(conn, runtime_turn_id)
            if turn is None:
                raise RuntimeError("runtime_turn_not_found")
            if turn.runtime_session_id != runtime_session_id:
                raise RuntimeError("runtime_turn_session_mismatch")
            if turn.turn_status in _TERMINAL_TURN_STATUSES:
                raise RuntimeError("runtime_turn_not_current")
            self._validate_current_turn(conn, session=session, turn=turn)
            self._update_turn_row(
                conn,
                runtime_turn_id,
                {
                    "restraint_policy": restraint_policy,
                    "updated_at": now,
                },
            )
            updated_turn = self._turn_by_id(conn, runtime_turn_id)
        assert updated_turn is not None
        return updated_turn

    def diagnostics(self, runtime_session_id: str) -> RuntimeSessionDiagnosticsResponse:
        with self._connect() as conn:
            session = self._session_by_id(conn, runtime_session_id)
            if session is None:
                raise RuntimeError("runtime_session_not_found")
            active_turn = conn.execute(
                """
                SELECT * FROM conversation_runtime_turns
                WHERE runtime_session_id = ? AND turn_status NOT IN ('completed', 'abandoned')
                ORDER BY id DESC
                LIMIT 1;
                """,
                (runtime_session_id,),
            ).fetchone()
            latest_turn = conn.execute(
                """
                SELECT * FROM conversation_runtime_turns
                WHERE runtime_session_id = ?
                ORDER BY id DESC
                LIMIT 1;
                """,
                (runtime_session_id,),
            ).fetchone()
            event_rows = conn.execute(
                """
                SELECT * FROM conversation_runtime_events
                WHERE runtime_session_id = ?
                ORDER BY id ASC;
                """,
                (runtime_session_id,),
            ).fetchall()
        return RuntimeSessionDiagnosticsResponse(
            runtime_session=session,
            active_turn=self._turn_from_row(active_turn) if active_turn is not None else None,
            latest_turn=self._turn_from_row(latest_turn) if latest_turn is not None else None,
            events=[self._event_from_row(row) for row in event_rows],
        )

    def list_events_for_tests(self, runtime_session_id: str) -> list[RuntimeEvent]:
        return self.diagnostics(runtime_session_id).events

    def _retirement_reservation_by_key(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        conversation_id: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT * FROM conversation_runtime_retirement_reservations
            WHERE owner_id = ? AND conversation_id = ?
            LIMIT 1;
            """,
            (owner_id, conversation_id),
        ).fetchone()

    def _required_retirement_reservation(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        conversation_id: str,
    ) -> sqlite3.Row:
        reservation = self._retirement_reservation_by_key(
            conn,
            owner_id=owner_id,
            conversation_id=conversation_id,
        )
        if reservation is None:
            raise RuntimeError("runtime_retirement_reservation_not_found")
        return reservation

    @staticmethod
    def _validate_retirement_reservation_identity(
        reservation: sqlite3.Row,
        *,
        reservation_id: str,
        reserved_thread_revision: int,
    ) -> None:
        if reservation["reservation_id"] != reservation_id:
            raise RuntimeError("runtime_retirement_reservation_conflict")
        if reservation["thread_revision"] != reserved_thread_revision:
            raise RuntimeError("runtime_retirement_reservation_revision_conflict")

    def _retirement_reservation_invariant_holds(
        self,
        conn: sqlite3.Connection,
        reservation: sqlite3.Row,
    ) -> bool:
        inspection = self._inspect_continuation_thread(
            conn,
            owner_id=reservation["owner_id"],
            conversation_id=reservation["conversation_id"],
        )
        return (
            inspection["state"] == "idle"
            and inspection["revision"] == reservation["thread_revision"]
        )

    def _raise_if_retirement_reserved(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        conversation_id: str,
    ) -> None:
        if (
            self._retirement_reservation_by_key(
                conn,
                owner_id=owner_id,
                conversation_id=conversation_id,
            )
            is not None
        ):
            raise RuntimeError("runtime_thread_retirement_reserved")

    def _thread_by_key(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        conversation_id: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT * FROM conversation_runtime_threads
            WHERE owner_id = ? AND conversation_id = ?
            LIMIT 1;
            """,
            (owner_id, conversation_id),
        ).fetchone()

    def _non_terminal_turn_rows(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        conversation_id: str,
    ) -> list[sqlite3.Row]:
        return conn.execute(
            """
            SELECT turns.*, sessions.owner_id, sessions.conversation_id, sessions.surface
            FROM conversation_runtime_turns AS turns
            JOIN conversation_runtime_sessions AS sessions
              ON sessions.runtime_session_id = turns.runtime_session_id
            WHERE sessions.owner_id = ? AND sessions.conversation_id = ?
              AND turns.turn_status NOT IN ('completed', 'abandoned')
            ORDER BY turns.id ASC;
            """,
            (owner_id, conversation_id),
        ).fetchall()

    def _record_event(
        self,
        conn: sqlite3.Connection,
        *,
        runtime_session_id: str,
        runtime_turn_id: str | None,
        event_type: str,
        event_payload_json: dict[str, Any],
    ) -> RuntimeEvent:
        created_at = _now()
        ordinal = int(
            conn.execute(
                "SELECT COUNT(*) FROM conversation_runtime_events WHERE runtime_session_id = ?;",
                (runtime_session_id,),
            ).fetchone()[0]
        )
        event_id = _event_id(runtime_session_id, event_type, created_at, ordinal)
        conn.execute(
            """
            INSERT INTO conversation_runtime_events (
                event_id, runtime_session_id, runtime_turn_id, event_type,
                event_payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                event_id,
                runtime_session_id,
                runtime_turn_id,
                event_type,
                _json(event_payload_json),
                created_at,
            ),
        )
        row = conn.execute(
            "SELECT * FROM conversation_runtime_events WHERE event_id = ? LIMIT 1;",
            (event_id,),
        ).fetchone()
        assert row is not None
        return self._event_from_row(row)

    def _session_by_key(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        conversation_id: str,
        surface: str,
    ) -> RuntimeSession | None:
        row = conn.execute(
            """
            SELECT * FROM conversation_runtime_sessions
            WHERE owner_id = ? AND conversation_id = ? AND surface = ?
            LIMIT 1;
            """,
            (owner_id, conversation_id, surface),
        ).fetchone()
        if row is None:
            return None
        return self._session_from_row(row)

    def _session_by_id(
        self,
        conn: sqlite3.Connection,
        runtime_session_id: str,
    ) -> RuntimeSession | None:
        row = conn.execute(
            "SELECT * FROM conversation_runtime_sessions WHERE runtime_session_id = ? LIMIT 1;",
            (runtime_session_id,),
        ).fetchone()
        if row is None:
            return None
        return self._session_from_row(row)

    def _turn_by_id(self, conn: sqlite3.Connection, runtime_turn_id: str) -> RuntimeTurn | None:
        row = conn.execute(
            "SELECT * FROM conversation_runtime_turns WHERE runtime_turn_id = ? LIMIT 1;",
            (runtime_turn_id,),
        ).fetchone()
        if row is None:
            return None
        return self._turn_from_row(row)

    def _update_session_row(
        self,
        conn: sqlite3.Connection,
        runtime_session_id: str,
        updates: dict[str, Any],
    ) -> None:
        cleaned = {key: value for key, value in updates.items() if value is not ...}
        if not cleaned:
            return
        columns = ", ".join(f"{key} = ?" for key in cleaned)
        conn.execute(
            f"UPDATE conversation_runtime_sessions SET {columns} WHERE runtime_session_id = ?;",
            (*cleaned.values(), runtime_session_id),
        )

    def _update_turn_row(
        self,
        conn: sqlite3.Connection,
        runtime_turn_id: str,
        updates: dict[str, Any],
    ) -> None:
        columns = ", ".join(f"{key} = ?" for key in updates)
        conn.execute(
            f"UPDATE conversation_runtime_turns SET {columns} WHERE runtime_turn_id = ?;",
            (*updates.values(), runtime_turn_id),
        )

    def _session_from_row(self, row: sqlite3.Row) -> RuntimeSession:
        return RuntimeSession(
            runtime_session_id=row["runtime_session_id"],
            owner_id=row["owner_id"],
            conversation_id=row["conversation_id"],
            surface=row["surface"],
            surface_session_id=row["surface_session_id"],
            status=row["status"],
            active_mode=row["active_mode"],
            attention_state=row["attention_state"],
            started_at=row["started_at"],
            last_activity_at=row["last_activity_at"],
            closed_at=row["closed_at"],
        )

    def _runtime_state_from_row(self, row: sqlite3.Row) -> RuntimeState:
        attention_focus = _load_json(row["attention_focus_json"], None)
        return RuntimeState(
            runtime_state_id=row["runtime_state_id"],
            owner_id=row["owner_id"],
            conversation_id=row["conversation_id"],
            surface=row["surface"],
            active_scene=row["active_scene"],
            interaction_mode=row["interaction_mode"],
            attention_focus=AttentionFocus.model_validate(attention_focus)
            if attention_focus is not None
            else None,
            temporary_constraints=_load_json(row["temporary_constraints_json"], []),
            reset_after_turn=bool(row["reset_after_turn"]),
            trace_refs=_load_json(row["trace_refs_json"], []),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _thread_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> RuntimeThreadProjection:
        session_rows = conn.execute(
            """
            SELECT surface FROM conversation_runtime_sessions
            WHERE owner_id = ? AND conversation_id = ?
            ORDER BY surface ASC;
            """,
            (row["owner_id"], row["conversation_id"]),
        ).fetchall()
        participating_surfaces = sorted({item["surface"] for item in session_rows})[:32]
        return RuntimeThreadProjection(
            owner_id=row["owner_id"],
            conversation_id=row["conversation_id"],
            state=row["state"],
            revision=row["revision"],
            active_runtime_session_id=row["active_runtime_session_id"],
            active_runtime_turn_id=row["active_runtime_turn_id"],
            active_surface=row["active_surface"],
            participating_surfaces=participating_surfaces,
            participating_session_count=len(session_rows),
            last_activity_at=row["last_activity_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _turn_from_row(self, row: sqlite3.Row) -> RuntimeTurn:
        return RuntimeTurn(
            runtime_turn_id=row["runtime_turn_id"],
            runtime_session_id=row["runtime_session_id"],
            input_message_id=row["input_message_id"],
            turn_status=row["turn_status"],
            intent_class=row["intent_class"],
            timing_policy=row["timing_policy"],
            restraint_policy=row["restraint_policy"],
            continuation_state=row["continuation_state"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )

    def _event_from_row(self, row: sqlite3.Row) -> RuntimeEvent:
        return RuntimeEvent(
            event_id=row["event_id"],
            runtime_session_id=row["runtime_session_id"],
            runtime_turn_id=row["runtime_turn_id"],
            event_type=row["event_type"],
            event_payload_json=_load_json(row["event_payload_json"], {}),
            created_at=row["created_at"],
        )


def runtime_state_repository() -> RuntimeStateRepository:
    global _RUNTIME_REPOSITORY
    if _RUNTIME_REPOSITORY is None:
        _RUNTIME_REPOSITORY = RuntimeStateRepository()
    return _RUNTIME_REPOSITORY


def resolve_runtime_session(
    *,
    request_id: str,
    owner_id: str,
    conversation_id: str,
    surface: str,
    surface_session_id: str | None = None,
    active_mode: str | None = None,
) -> RuntimeSession:
    return runtime_state_repository().resolve_session(
        request_id=request_id,
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface=surface,
        surface_session_id=surface_session_id,
        active_mode=active_mode,
    )


def get_runtime_session(runtime_session_id: str) -> RuntimeSessionDiagnosticsResponse:
    return runtime_state_repository().diagnostics(runtime_session_id)


def runtime_session_by_id(runtime_session_id: str) -> RuntimeSession | None:
    return runtime_state_repository().session_by_id(runtime_session_id)


def resolve_runtime_thread(
    *,
    owner_id: str,
    conversation_id: str,
) -> RuntimeThreadProjection:
    return runtime_state_repository().resolve_thread(
        owner_id=owner_id,
        conversation_id=conversation_id,
    )


def select_runtime_continuation(
    request: ContinuationSelectionRequest,
) -> ContinuationSelectionResponse:
    return runtime_state_repository().select_continuation(request)


def reserve_runtime_retirement(
    request: RetirementReservationRequest,
) -> RetirementReservationResponse:
    return runtime_state_repository().reserve_retirement(request)


def cancel_runtime_retirement(
    request: RetirementReservationCancelRequest,
) -> RetirementReservationCancelResponse:
    return runtime_state_repository().cancel_retirement_reservation(request)


def finalize_runtime_retirement(
    request: RetirementReservationFinalizeRequest,
) -> RetirementReservationFinalizeResponse:
    return runtime_state_repository().finalize_retirement_reservation(request)


def resolve_state(*, owner_id: str, conversation_id: str, surface: str) -> RuntimeState:
    return runtime_state_repository().resolve_state(
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface=surface,
    )


def update_state(
    *,
    owner_id: str,
    conversation_id: str,
    surface: str,
    updates: RuntimeStateUpdate,
) -> RuntimeState:
    return runtime_state_repository().update_state(
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface=surface,
        updates=updates,
    )


def reset_state(*, owner_id: str, conversation_id: str, surface: str) -> RuntimeState:
    return runtime_state_repository().reset_state(
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface=surface,
    )


def start_turn(
    *,
    request_id: str,
    owner_id: str,
    conversation_id: str,
    surface: str,
    surface_session_id: str | None = None,
    active_mode: str | None = None,
    input_message_id: str | None = None,
    intent_class: str | None = None,
    timing_policy: str | None = None,
    restraint_policy: str | None = None,
    continuation_state: str | None = None,
    expected_thread_revision: int | None = None,
) -> tuple[RuntimeSession, RuntimeTurn, RuntimeEvent]:
    return runtime_state_repository().start_turn(
        request_id=request_id,
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface=surface,
        surface_session_id=surface_session_id,
        active_mode=active_mode,
        input_message_id=input_message_id,
        intent_class=intent_class,
        timing_policy=timing_policy,
        restraint_policy=restraint_policy,
        continuation_state=continuation_state,
        expected_thread_revision=expected_thread_revision,
    )


def update_turn(
    *,
    request_id: str,
    runtime_session_id: str,
    runtime_turn_id: str,
    turn_status: str,
    timing_policy: str | None = None,
    restraint_policy: str | None = None,
    continuation_state: str | None = None,
) -> tuple[RuntimeSession, RuntimeTurn, RuntimeEvent]:
    return runtime_state_repository().update_turn(
        request_id=request_id,
        runtime_session_id=runtime_session_id,
        runtime_turn_id=runtime_turn_id,
        turn_status=turn_status,
        timing_policy=timing_policy,
        restraint_policy=restraint_policy,
        continuation_state=continuation_state,
    )


def complete_turn(
    *,
    request_id: str,
    runtime_session_id: str,
    runtime_turn_id: str,
    turn_status: str,
    continuation_state: str | None = None,
) -> tuple[RuntimeSession, RuntimeTurn, RuntimeEvent]:
    return runtime_state_repository().complete_turn(
        request_id=request_id,
        runtime_session_id=runtime_session_id,
        runtime_turn_id=runtime_turn_id,
        turn_status=turn_status,
        continuation_state=continuation_state,
    )


def record_runtime_event(
    *,
    runtime_session_id: str,
    runtime_turn_id: str | None,
    event_type: str,
    event_payload_json: dict[str, Any],
) -> RuntimeEvent:
    return runtime_state_repository().record_session_event(
        runtime_session_id=runtime_session_id,
        runtime_turn_id=runtime_turn_id,
        event_type=event_type,
        event_payload_json=event_payload_json,
    )


def update_runtime_turn_intent_class(
    *,
    runtime_session_id: str,
    runtime_turn_id: str,
    intent_class: str,
) -> RuntimeTurn:
    return runtime_state_repository().update_turn_intent_class(
        runtime_session_id=runtime_session_id,
        runtime_turn_id=runtime_turn_id,
        intent_class=intent_class,
    )


def update_runtime_turn_restraint_policy(
    *,
    runtime_session_id: str,
    runtime_turn_id: str,
    restraint_policy: str,
) -> RuntimeTurn:
    return runtime_state_repository().update_turn_restraint_policy(
        runtime_session_id=runtime_session_id,
        runtime_turn_id=runtime_turn_id,
        restraint_policy=restraint_policy,
    )


def validate_runtime_turn_session(
    *,
    runtime_session_id: str,
    runtime_turn_id: str,
) -> RuntimeTurn:
    turn = runtime_state_repository().turn_by_id(runtime_turn_id)
    if turn is None:
        raise RuntimeError("runtime_turn_not_found")
    if turn.runtime_session_id != runtime_session_id:
        raise RuntimeError("runtime_turn_session_mismatch")
    return turn


def build_overlay(state: RuntimeState) -> tuple[RuntimeOverlay | None, str | None]:
    source_fields: list[str] = []
    parts: list[str] = []
    if state.active_scene:
        parts.append(f"scene={state.active_scene}")
        source_fields.append("active_scene")
    if state.interaction_mode:
        parts.append(f"interaction_mode={state.interaction_mode}")
        source_fields.append("interaction_mode")
    if state.temporary_constraints:
        parts.append(f"constraints={','.join(state.temporary_constraints)}")
        source_fields.append("temporary_constraints")

    if not parts:
        return None, "empty_runtime_state"

    material = "|".join(
        [
            state.runtime_state_id,
            state.active_scene or "",
            state.interaction_mode or "",
            ",".join(state.temporary_constraints),
            ",".join(source_fields),
        ]
    )
    overlay_id = _digest("rtoverlay", material)
    return (
        RuntimeOverlay(
            overlay_id=overlay_id,
            runtime_state_id=state.runtime_state_id,
            content=f"Runtime context: {'; '.join(parts)}.",
            source_fields=source_fields,
        ),
        None,
    )


def clear_states_for_tests(db_path: Path | None = None) -> None:
    global _RUNTIME_REPOSITORY
    _RUNTIME_REPOSITORY = RuntimeStateRepository(db_path=db_path)
