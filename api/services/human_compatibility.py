from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from models import (
    HumanCompatibilityDiagnosticsRequest,
    HumanCompatibilityDiagnosticsResponse,
    HumanCompatibilityReviewRequest,
    HumanCompatibilityReviewResponse,
    HumanCompatibilityReviewResult,
    HumanCompatibilityReviewView,
    HumanCompatibilityRiskFlagInput,
    HumanCompatibilityRiskFlagView,
)
from services.runtime_state import runtime_state_db_path

_HUMAN_COMPATIBILITY_REPOSITORY: HumanCompatibilityRepository | None = None
_PRINCIPLES_CHECKED = [
    "usefulness_over_attachment",
    "continuity_without_possession",
    "warmth_without_reciprocity_claims",
    "transparency_over_hidden_influence",
    "restraint_over_intensity",
    "user_agency_first",
]
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
_REJECTING_FLAG_TYPES = {"hidden_influence", "agency_erosion"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _digest(prefix: str, material: str) -> str:
    return f"{prefix}_{sha256(material.encode('utf-8')).hexdigest()[:16]}"


class HumanCompatibilityRepository:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or runtime_state_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS human_compatibility_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT NOT NULL,
                    review_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    feature_ref TEXT NOT NULL,
                    spec_ref TEXT NOT NULL,
                    review_surfaces_json TEXT NOT NULL,
                    proposed_behavior_summary TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    review_result TEXT NOT NULL,
                    review_notes TEXT,
                    mitigations_json TEXT NOT NULL,
                    runtime_turn_id TEXT,
                    principles_checked_json TEXT NOT NULL,
                    mitigations_required INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(owner_id, review_id)
                );

                CREATE TABLE IF NOT EXISTS interaction_risk_flags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT NOT NULL,
                    flag_id TEXT NOT NULL,
                    runtime_turn_id TEXT,
                    risk_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    triggering_policy TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(owner_id, flag_id)
                );
                """
            )

    def submit_review(self, body: HumanCompatibilityReviewRequest) -> HumanCompatibilityReviewResponse:
        now = _now()
        review_result = self._review_result(body)
        review_id = body.review_id or _digest(
            "hcrev",
            f"{body.owner_id}:{body.feature_ref}:{body.spec_ref}:{body.request_id}:{now}",
        )
        mitigations_required = body.risk_level == "high" and review_result == "mitigations_required"
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM human_compatibility_reviews WHERE owner_id = ? AND review_id = ? LIMIT 1;",
                (body.owner_id, review_id),
            ).fetchone()
            created_at = existing["created_at"] if existing is not None else now
            conn.execute(
                """
                INSERT INTO human_compatibility_reviews (
                    owner_id, review_id, request_id, feature_ref, spec_ref,
                    review_surfaces_json, proposed_behavior_summary, risk_level,
                    review_result, review_notes, mitigations_json, runtime_turn_id,
                    principles_checked_json, mitigations_required, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, review_id) DO UPDATE SET
                    request_id = excluded.request_id,
                    feature_ref = excluded.feature_ref,
                    spec_ref = excluded.spec_ref,
                    review_surfaces_json = excluded.review_surfaces_json,
                    proposed_behavior_summary = excluded.proposed_behavior_summary,
                    risk_level = excluded.risk_level,
                    review_result = excluded.review_result,
                    review_notes = excluded.review_notes,
                    mitigations_json = excluded.mitigations_json,
                    runtime_turn_id = excluded.runtime_turn_id,
                    principles_checked_json = excluded.principles_checked_json,
                    mitigations_required = excluded.mitigations_required,
                    created_at = excluded.created_at;
                """,
                (
                    body.owner_id,
                    review_id,
                    body.request_id,
                    body.feature_ref,
                    body.spec_ref,
                    _json(body.review_surfaces),
                    body.proposed_behavior_summary,
                    body.risk_level,
                    review_result,
                    body.review_notes,
                    _json(body.mitigations_json),
                    body.runtime_turn_id,
                    _json(_PRINCIPLES_CHECKED),
                    int(mitigations_required),
                    created_at,
                ),
            )

            stored_flag_ids: list[str] = []
            flags_recorded = 0
            for index, flag in enumerate(body.interaction_risk_flags):
                flag_id, was_recorded = self._record_flag(
                    conn,
                    owner_id=body.owner_id,
                    runtime_turn_id=body.runtime_turn_id,
                    flag=flag,
                    request_id=body.request_id,
                    ordinal=index,
                )
                stored_flag_ids.append(flag_id)
                if was_recorded:
                    flags_recorded += 1

        return HumanCompatibilityReviewResponse(
            review_id=review_id,
            review_result=review_result,
            principles_checked=list(_PRINCIPLES_CHECKED),
            mitigations_required=mitigations_required,
            flags_recorded=flags_recorded,
            diagnostics={
                "request_id": body.request_id,
                "owner_id": body.owner_id,
                "runtime_turn_id": body.runtime_turn_id,
                "spec_ref": body.spec_ref,
                "review_surface_count": len(body.review_surfaces),
                "high_risk": body.risk_level == "high",
                "stored_flag_ids": stored_flag_ids,
                "validation_version": "r49_v1",
            },
        )

    def diagnostics(
        self,
        body: HumanCompatibilityDiagnosticsRequest,
    ) -> HumanCompatibilityDiagnosticsResponse:
        filters = ["owner_id = ?"]
        parameters: list[Any] = [body.owner_id]
        if body.feature_ref:
            filters.append("feature_ref = ?")
            parameters.append(body.feature_ref)
        if body.runtime_turn_id:
            filters.append("runtime_turn_id = ?")
            parameters.append(body.runtime_turn_id)
        where_clause = " AND ".join(filters)
        with self._connect() as conn:
            review_rows = conn.execute(
                f"""
                SELECT * FROM human_compatibility_reviews
                WHERE {where_clause}
                ORDER BY created_at, review_id;
                """,
                parameters,
            ).fetchall()

            flag_filters = ["owner_id = ?"]
            flag_params: list[Any] = [body.owner_id]
            if body.runtime_turn_id:
                flag_filters.append("runtime_turn_id = ?")
                flag_params.append(body.runtime_turn_id)
            flag_where_clause = " AND ".join(flag_filters)
            flag_rows = conn.execute(
                f"""
                SELECT * FROM interaction_risk_flags
                WHERE {flag_where_clause}
                ORDER BY created_at, flag_id;
                """,
                flag_params,
            ).fetchall()

        reviews = [self._review_view_from_row(row) for row in review_rows]
        if body.feature_ref:
            allowed_runtime_turn_ids = {
                review.runtime_turn_id for review in reviews if review.runtime_turn_id is not None
            }
            flags = [
                self._flag_view_from_row(row)
                for row in flag_rows
                if row["runtime_turn_id"] is None or row["runtime_turn_id"] in allowed_runtime_turn_ids
            ]
        else:
            flags = [self._flag_view_from_row(row) for row in flag_rows]
        return HumanCompatibilityDiagnosticsResponse(
            reviews=reviews,
            interaction_risk_flags=flags,
        )

    def _record_flag(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        runtime_turn_id: str | None,
        flag: HumanCompatibilityRiskFlagInput,
        request_id: str,
        ordinal: int,
    ) -> tuple[str, bool]:
        now = _now()
        flag_id = flag.flag_id or _digest(
            "hcflag",
            f"{owner_id}:{request_id}:{flag.risk_type}:{flag.severity}:{flag.triggering_policy}:{ordinal}:{now}",
        )
        before_changes = conn.total_changes
        conn.execute(
            """
            INSERT OR IGNORE INTO interaction_risk_flags (
                owner_id, flag_id, runtime_turn_id, risk_type, severity, triggering_policy, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                owner_id,
                flag_id,
                runtime_turn_id,
                flag.risk_type,
                flag.severity,
                flag.triggering_policy,
                now,
            ),
        )
        return flag_id, conn.total_changes > before_changes

    def _review_result(self, body: HumanCompatibilityReviewRequest) -> HumanCompatibilityReviewResult:
        max_flag_severity = max(
            (_RISK_ORDER[flag.severity] for flag in body.interaction_risk_flags),
            default=-1,
        )
        if max_flag_severity > _RISK_ORDER[body.risk_level]:
            return "rejected"
        if body.risk_level == "high":
            if not body.interaction_risk_flags:
                return "requires_human_review"
            flagged_types = {flag.risk_type for flag in body.interaction_risk_flags}
            if _REJECTING_FLAG_TYPES.issubset(flagged_types):
                return "rejected"
            return "requires_human_review"
        return "approved"

    def _review_view_from_row(self, row: sqlite3.Row) -> HumanCompatibilityReviewView:
        return HumanCompatibilityReviewView(
            review_id=row["review_id"],
            owner_id=row["owner_id"],
            request_id=row["request_id"],
            feature_ref=row["feature_ref"],
            spec_ref=row["spec_ref"],
            review_surfaces=_load_json(row["review_surfaces_json"], []),
            proposed_behavior_summary=row["proposed_behavior_summary"],
            risk_level=row["risk_level"],
            review_result=row["review_result"],
            review_notes=row["review_notes"],
            mitigations_json=_load_json(row["mitigations_json"], None),
            runtime_turn_id=row["runtime_turn_id"],
            principles_checked=_load_json(row["principles_checked_json"], list(_PRINCIPLES_CHECKED)),
            mitigations_required=bool(row["mitigations_required"]),
            created_at=row["created_at"],
        )

    def _flag_view_from_row(self, row: sqlite3.Row) -> HumanCompatibilityRiskFlagView:
        return HumanCompatibilityRiskFlagView(
            flag_id=row["flag_id"],
            owner_id=row["owner_id"],
            runtime_turn_id=row["runtime_turn_id"],
            risk_type=row["risk_type"],
            severity=row["severity"],
            triggering_policy=row["triggering_policy"],
            created_at=row["created_at"],
        )


def human_compatibility_repository() -> HumanCompatibilityRepository:
    global _HUMAN_COMPATIBILITY_REPOSITORY
    if _HUMAN_COMPATIBILITY_REPOSITORY is None:
        _HUMAN_COMPATIBILITY_REPOSITORY = HumanCompatibilityRepository()
    return _HUMAN_COMPATIBILITY_REPOSITORY


def clear_human_compatibility_for_tests(db_path: Path | None = None) -> None:
    global _HUMAN_COMPATIBILITY_REPOSITORY
    _HUMAN_COMPATIBILITY_REPOSITORY = HumanCompatibilityRepository(db_path=db_path)


def submit_human_compatibility_review(
    body: HumanCompatibilityReviewRequest,
) -> HumanCompatibilityReviewResponse:
    return human_compatibility_repository().submit_review(body)


def get_human_compatibility_diagnostics(
    body: HumanCompatibilityDiagnosticsRequest,
) -> HumanCompatibilityDiagnosticsResponse:
    return human_compatibility_repository().diagnostics(body)
