from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from models import (
    SocialContextDiagnosticsRequest,
    SocialContextDiagnosticsResponse,
    SocialContextItemInput,
    SocialContextItemResponse,
    SocialContextItemView,
    SocialContextUsageEventInput,
    SocialContextUsageEventResponse,
    SocialContextUsageEventView,
)
from services.relationships import relationship_repository
from services.runtime_state import runtime_state_db_path

_SOCIAL_CONTEXT_REPOSITORY: SocialContextRepository | None = None
_RESTRICTED_MENTIONABILITY = {
    "confirm_before_mentioning",
    "suppress_by_default",
    "restricted",
}
_RESTRICTED_SENSITIVITY_LEVELS = {"high", "restricted"}


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


class SocialContextRepository:
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
                CREATE TABLE IF NOT EXISTS social_context_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    social_context_id TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL,
                    context_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source_refs_json TEXT NOT NULL,
                    relationship_edge_refs_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    freshness TEXT NOT NULL,
                    mentionability TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS social_context_usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    social_context_id TEXT NOT NULL,
                    relationship_edge_refs_json TEXT NOT NULL,
                    runtime_turn_id TEXT,
                    usage_type TEXT NOT NULL,
                    policy_decision TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(social_context_id) REFERENCES social_context_items(social_context_id)
                );
                """
            )

    def upsert_item(self, *, owner_id: str, item: SocialContextItemInput) -> SocialContextItemView:
        self._validate_item(owner_id=owner_id, item=item)
        now = _now()
        social_context_id = item.social_context_id or _digest(
            "socctx",
            f"{owner_id}:{item.context_type}:{item.summary}:{now}",
        )
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM social_context_items WHERE owner_id = ? AND social_context_id = ? LIMIT 1;",
                (owner_id, social_context_id),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO social_context_items (
                        social_context_id, owner_id, context_type, summary,
                        source_refs_json, relationship_edge_refs_json, confidence,
                        freshness, mentionability, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        social_context_id,
                        owner_id,
                        item.context_type,
                        item.summary,
                        _json(item.source_refs_json),
                        _json(item.relationship_edge_refs_json),
                        item.confidence,
                        item.freshness,
                        item.mentionability,
                        now,
                        now,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE social_context_items
                    SET context_type = ?, summary = ?, source_refs_json = ?,
                        relationship_edge_refs_json = ?, confidence = ?, freshness = ?,
                        mentionability = ?, updated_at = ?
                    WHERE owner_id = ? AND social_context_id = ?;
                    """,
                    (
                        item.context_type,
                        item.summary,
                        _json(item.source_refs_json),
                        _json(item.relationship_edge_refs_json),
                        item.confidence,
                        item.freshness,
                        item.mentionability,
                        now,
                        owner_id,
                        social_context_id,
                    ),
                )
            row = conn.execute(
                "SELECT * FROM social_context_items WHERE owner_id = ? AND social_context_id = ? LIMIT 1;",
                (owner_id, social_context_id),
            ).fetchone()
        assert row is not None
        return self._item_view_from_row(row)

    def record_usage_event(
        self,
        *,
        owner_id: str,
        event: SocialContextUsageEventInput,
    ) -> SocialContextUsageEventView:
        now = _now()
        with self._connect() as conn:
            item = conn.execute(
                "SELECT * FROM social_context_items WHERE owner_id = ? AND social_context_id = ? LIMIT 1;",
                (owner_id, event.social_context_id),
            ).fetchone()
            if item is None:
                raise RuntimeError("social_context_item_not_found")
            event_id = event.event_id or _digest(
                "socuse",
                f"{event.social_context_id}:{event.usage_type}:{event.policy_decision}:{now}",
            )
            conn.execute(
                """
                INSERT INTO social_context_usage_events (
                    event_id, social_context_id, relationship_edge_refs_json,
                    runtime_turn_id, usage_type, policy_decision, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    event_id,
                    event.social_context_id,
                    _json(event.relationship_edge_refs_json),
                    event.runtime_turn_id,
                    event.usage_type,
                    event.policy_decision,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM social_context_usage_events WHERE event_id = ? LIMIT 1;",
                (event_id,),
            ).fetchone()
        assert row is not None
        return self._event_view_from_row(row)

    def diagnostics(self, *, owner_id: str) -> SocialContextDiagnosticsResponse:
        with self._connect() as conn:
            item_rows = conn.execute(
                "SELECT * FROM social_context_items WHERE owner_id = ? ORDER BY created_at, social_context_id;",
                (owner_id,),
            ).fetchall()
            usage_rows = conn.execute(
                """
                SELECT social_context_usage_events.*
                FROM social_context_usage_events
                JOIN social_context_items ON social_context_items.social_context_id = social_context_usage_events.social_context_id
                WHERE social_context_items.owner_id = ?
                ORDER BY social_context_usage_events.created_at, social_context_usage_events.event_id;
                """,
                (owner_id,),
            ).fetchall()
        return SocialContextDiagnosticsResponse(
            items=[self._item_view_from_row(row) for row in item_rows],
            usage_events=[self._event_view_from_row(row) for row in usage_rows],
        )

    def _validate_item(self, *, owner_id: str, item: SocialContextItemInput) -> None:
        if not item.source_refs_json:
            raise RuntimeError("social_context_source_refs_required")
        approved_relationship_ids = set()
        for edge in relationship_repository().diagnostics(owner_id=owner_id).relationships:
            if edge.status == "active" and edge.sensitivity_level not in _RESTRICTED_SENSITIVITY_LEVELS:
                approved_relationship_ids.add(edge.relationship_id)
        for relationship_id in item.relationship_edge_refs_json:
            if relationship_id not in approved_relationship_ids:
                raise RuntimeError("social_context_requires_approved_relationship_edge")

    def _item_view_from_row(self, row: sqlite3.Row) -> SocialContextItemView:
        source_refs = _load_json(row["source_refs_json"], [])
        relationship_refs = _load_json(row["relationship_edge_refs_json"], [])
        suppressed = (
            row["confidence"] < 0.75 or row["mentionability"] in _RESTRICTED_MENTIONABILITY
        )
        suppression_reasons: list[str] = []
        if row["confidence"] < 0.75:
            suppression_reasons.append("low_confidence")
        if row["mentionability"] in _RESTRICTED_MENTIONABILITY:
            suppression_reasons.append("mentionability_restricted")
        return SocialContextItemView(
            social_context_id=row["social_context_id"],
            owner_id=row["owner_id"],
            context_type=row["context_type"],
            summary=row["summary"],
            source_refs_json=source_refs,
            relationship_edge_refs_json=relationship_refs,
            confidence=row["confidence"],
            freshness=row["freshness"],
            mentionability=row["mentionability"],
            suppressed_by_default=suppressed,
            suppression_reasons=suppression_reasons,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _event_view_from_row(self, row: sqlite3.Row) -> SocialContextUsageEventView:
        return SocialContextUsageEventView(
            event_id=row["event_id"],
            social_context_id=row["social_context_id"],
            relationship_edge_refs_json=_load_json(row["relationship_edge_refs_json"], []),
            runtime_turn_id=row["runtime_turn_id"],
            usage_type=row["usage_type"],
            policy_decision=row["policy_decision"],
            created_at=row["created_at"],
        )


def social_context_repository() -> SocialContextRepository:
    global _SOCIAL_CONTEXT_REPOSITORY
    if _SOCIAL_CONTEXT_REPOSITORY is None:
        _SOCIAL_CONTEXT_REPOSITORY = SocialContextRepository()
    return _SOCIAL_CONTEXT_REPOSITORY


def clear_social_context_for_tests(db_path: Path | None = None) -> None:
    global _SOCIAL_CONTEXT_REPOSITORY
    _SOCIAL_CONTEXT_REPOSITORY = SocialContextRepository(db_path=db_path)


def upsert_social_context_item(*, owner_id: str, item: SocialContextItemInput) -> SocialContextItemView:
    return social_context_repository().upsert_item(owner_id=owner_id, item=item)


def record_social_context_usage_event(
    *,
    owner_id: str,
    event: SocialContextUsageEventInput,
) -> SocialContextUsageEventView:
    return social_context_repository().record_usage_event(owner_id=owner_id, event=event)


def get_social_context_diagnostics(
    body: SocialContextDiagnosticsRequest,
) -> SocialContextDiagnosticsResponse:
    return social_context_repository().diagnostics(owner_id=body.owner_id)
