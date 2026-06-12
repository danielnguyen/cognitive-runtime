from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from models import (
    RelationshipDiagnosticsResponse,
    RelationshipEdgeConfirmRequest,
    RelationshipEdgeEvidenceInput,
    RelationshipEdgeEvidenceView,
    RelationshipEdgeInput,
    RelationshipEdgeResponse,
    RelationshipEdgeRevokeRequest,
    RelationshipEdgeView,
    RelationshipEntityInput,
    RelationshipEntityResponse,
    RelationshipEntityView,
    RelationshipExcludedSummary,
    RelationshipGraphDiagnosticsRequest,
    RelationshipStatus,
    RelationshipSelectRequest,
    RelationshipSelectResponse,
    RelationshipSelectTrace,
    RuntimeIdentityResolveRequest,
)
from services.companion_contracts import companion_contracts_repository
from services.runtime_identity import resolve_runtime_identity
from services.runtime_state import runtime_state_db_path

_RELATIONSHIP_REPOSITORY: RelationshipRepository | None = None
_REDACTED = "[REDACTED]"
_TRUSTED_ACTIVE_EDGE_SOURCES = {
    "explicit_user_confirmation",
    "trusted_config",
    "trusted_integration_metadata",
    "trusted_import_metadata",
    "system_default",
}
_THRESHOLD_EXEMPT_SOURCES = {"trusted_config", "system_default"}
_MODEL_INFERENCE_ALLOWED_STATUSES = {"inferred", "needs_confirmation"}
_SOCIALISH_RELATIONSHIP_TYPES = {"colleague_of", "collaborates_with"}
_TERMINAL_EDGE_STATUSES = {"revoked", "superseded", "expired"}
_RESTRICTED_SENSITIVITY_LEVELS = {"high", "restricted"}
_RELATIONSHIP_SCOPE_ALLOWLISTS: dict[str, set[str]] = {
    "general_assistant": {
        "professional_context",
        "project_context",
        "system_configuration",
        "operations_context",
        "creative_context",
    },
    "technical_architect": {
        "professional_context",
        "project_context",
        "system_configuration",
        "operations_context",
        "creative_context",
    },
    "operations_assistant": {
        "professional_context",
        "project_context",
        "system_configuration",
        "operations_context",
    },
    "personal_companion": {
        "personal_context",
        "professional_context",
        "creative_context",
    },
}
_SURFACE_SCOPE_RESTRICTIONS: dict[str, set[str]] = {
    "developer_surface": {
        "professional_context",
        "project_context",
        "system_configuration",
        "operations_context",
        "creative_context",
    },
    "ide_extension": {
        "professional_context",
        "project_context",
        "system_configuration",
        "operations_context",
        "creative_context",
    },
    "web_app": {
        "professional_context",
        "project_context",
        "personal_context",
        "creative_context",
    },
    "unknown_surface": {
        "professional_context",
        "project_context",
        "system_configuration",
    },
}


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


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _is_expired(valid_until: str | None, *, now: datetime) -> bool:
    parsed = _parse_ts(valid_until)
    return parsed is not None and parsed <= now


def _is_model_inference(edge: RelationshipEdgeInput) -> bool:
    return edge.source_type == "model_inference"


class RelationshipRepository:
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
                CREATE TABLE IF NOT EXISTS relationship_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    canonical_label TEXT NOT NULL,
                    display_label TEXT,
                    domain TEXT NOT NULL,
                    sensitivity_level TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    canonical_memory_ref TEXT,
                    artifact_ref TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    archived_at TEXT,
                    UNIQUE(owner_id, entity_id)
                );

                CREATE TABLE IF NOT EXISTS relationship_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    relationship_id TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL,
                    subject_entity_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    object_entity_id TEXT NOT NULL,
                    relationship_scope TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_refs_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    sensitivity_level TEXT NOT NULL,
                    mentionability TEXT NOT NULL,
                    allowed_persona_scopes_json TEXT NOT NULL,
                    blocked_persona_scopes_json TEXT NOT NULL,
                    valid_from TEXT,
                    valid_until TEXT,
                    superseded_by_relationship_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revoked_at TEXT,
                    UNIQUE(owner_id, relationship_id)
                );

                CREATE TABLE IF NOT EXISTS relationship_evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evidence_id TEXT NOT NULL UNIQUE,
                    relationship_id TEXT NOT NULL,
                    evidence_type TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    summary TEXT,
                    confidence_delta REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(relationship_id) REFERENCES relationship_edges(relationship_id)
                );
                """
            )

    def upsert_entity(
        self,
        *,
        owner_id: str,
        entity: RelationshipEntityInput,
    ) -> RelationshipEntityView:
        now = _now()
        with self._connect() as conn:
            self._upsert_entity_with_conn(conn, owner_id=owner_id, entity=entity, now=now)
            row = conn.execute(
                "SELECT * FROM relationship_entities WHERE owner_id = ? AND entity_id = ? LIMIT 1;",
                (owner_id, entity.entity_id),
            ).fetchone()
        assert row is not None
        return self._entity_view_from_row(row)

    def entity_by_id(self, *, owner_id: str, entity_id: str) -> RelationshipEntityView | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM relationship_entities WHERE owner_id = ? AND entity_id = ? LIMIT 1;",
                (owner_id, entity_id),
            ).fetchone()
        if row is None:
            return None
        return self._entity_view_from_row(row)

    def upsert_edge(
        self,
        *,
        owner_id: str,
        edge: RelationshipEdgeInput,
        evidence: list[RelationshipEdgeEvidenceInput],
    ) -> RelationshipEdgeResponse:
        self._validate_edge_input(edge)
        normalized_status = self._normalized_edge_status(edge)
        now = _now()
        relationship_id = edge.relationship_id or _digest(
            "rel",
            (
                f"{owner_id}:{edge.subject_entity_id}:{edge.relationship_type}:"
                f"{edge.object_entity_id}:{edge.relationship_scope}:{edge.source_type}:{now}"
            ),
        )
        with self._connect() as conn:
            self._upsert_edge_with_conn(
                conn,
                owner_id=owner_id,
                relationship_id=relationship_id,
                edge=edge,
                normalized_status=normalized_status,
                now=now,
            )
            evidence_views = [
                self._insert_evidence(conn, relationship_id=relationship_id, evidence=item)
                for item in evidence
            ]
            row = conn.execute(
                "SELECT * FROM relationship_edges WHERE owner_id = ? AND relationship_id = ? LIMIT 1;",
                (owner_id, relationship_id),
            ).fetchone()
        assert row is not None
        return RelationshipEdgeResponse(
            relationship=self._edge_view_from_row(row, include_restricted_details=False),
            evidence=evidence_views,
        )

    def confirm_edge(
        self,
        *,
        owner_id: str,
        body: RelationshipEdgeConfirmRequest,
    ) -> RelationshipEdgeResponse:
        now = _now()
        with self._connect() as conn:
            row = self._require_edge(conn, owner_id=owner_id, relationship_id=body.relationship_id)
            if row["status"] not in {"provisional", "inferred", "needs_confirmation"}:
                raise RuntimeError("relationship_edge_status_not_confirmable")
            if body.evidence is None:
                raise RuntimeError("relationship_confirmation_evidence_required")
            conn.execute(
                """
                UPDATE relationship_edges
                SET status = 'active', updated_at = ?, revoked_at = NULL
                WHERE owner_id = ? AND relationship_id = ?;
                """,
                (now, owner_id, body.relationship_id),
            )
            evidence_views: list[RelationshipEdgeEvidenceView] = []
            if body.evidence is not None:
                evidence_views.append(
                    self._insert_evidence(conn, relationship_id=body.relationship_id, evidence=body.evidence)
                )
            updated = conn.execute(
                "SELECT * FROM relationship_edges WHERE owner_id = ? AND relationship_id = ? LIMIT 1;",
                (owner_id, body.relationship_id),
            ).fetchone()
        assert updated is not None
        return RelationshipEdgeResponse(
            relationship=self._edge_view_from_row(updated, include_restricted_details=False),
            evidence=evidence_views,
        )

    def bootstrap_apply(
        self,
        *,
        owner_id: str,
        entities: list[RelationshipEntityInput],
        relationships: list[tuple[RelationshipEdgeInput, list[RelationshipEdgeEvidenceInput]]],
        dry_run: bool = False,
    ) -> dict[str, int]:
        conn = self._connect()
        try:
            conn.execute("BEGIN;")
            existing_entity_ids = self._entity_ids_for_owner(conn, owner_id=owner_id)
            seed_entity_ids = {entity.entity_id for entity in entities}
            for edge, _ in relationships:
                if edge.subject_entity_id not in seed_entity_ids and edge.subject_entity_id not in existing_entity_ids:
                    raise RuntimeError(
                        f"relationship_subject_entity_missing:{edge.relationship_id}:{edge.subject_entity_id}"
                    )
                if edge.object_entity_id not in seed_entity_ids and edge.object_entity_id not in existing_entity_ids:
                    raise RuntimeError(
                        f"relationship_object_entity_missing:{edge.relationship_id}:{edge.object_entity_id}"
                    )

            entity_upserts = 0
            relationship_upserts = 0
            evidence_inserted = 0
            evidence_skipped = 0

            for entity in entities:
                self._upsert_entity_with_conn(conn, owner_id=owner_id, entity=entity, now=_now())
                entity_upserts += 1

            for edge, evidence_items in relationships:
                normalized_status = self._normalized_edge_status(edge)
                self._upsert_edge_with_conn(
                    conn,
                    owner_id=owner_id,
                    relationship_id=edge.relationship_id or "",
                    edge=edge,
                    normalized_status=normalized_status,
                    now=_now(),
                )
                relationship_upserts += 1
                relationship_id = edge.relationship_id or ""
                for evidence in evidence_items:
                    _, inserted = self._insert_evidence_deduped(
                        conn,
                        relationship_id=relationship_id,
                        evidence=evidence,
                    )
                    if inserted:
                        evidence_inserted += 1
                    else:
                        evidence_skipped += 1

            if dry_run:
                conn.rollback()
            else:
                conn.commit()
            return {
                "entities_upserted": entity_upserts,
                "relationships_upserted": relationship_upserts,
                "evidence_inserted": evidence_inserted,
                "evidence_skipped": evidence_skipped,
                "errors": 0,
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def revoke_edge(
        self,
        *,
        owner_id: str,
        body: RelationshipEdgeRevokeRequest,
    ) -> RelationshipEdgeResponse:
        now = _now()
        with self._connect() as conn:
            self._require_edge(conn, owner_id=owner_id, relationship_id=body.relationship_id)
            conn.execute(
                """
                UPDATE relationship_edges
                SET status = 'revoked', revoked_at = ?, updated_at = ?
                WHERE owner_id = ? AND relationship_id = ?;
                """,
                (now, now, owner_id, body.relationship_id),
            )
            evidence_views: list[RelationshipEdgeEvidenceView] = []
            if body.evidence is not None:
                evidence_views.append(
                    self._insert_evidence(conn, relationship_id=body.relationship_id, evidence=body.evidence)
                )
            updated = conn.execute(
                "SELECT * FROM relationship_edges WHERE owner_id = ? AND relationship_id = ? LIMIT 1;",
                (owner_id, body.relationship_id),
            ).fetchone()
        assert updated is not None
        return RelationshipEdgeResponse(
            relationship=self._edge_view_from_row(updated, include_restricted_details=False),
            evidence=evidence_views,
        )

    def edge_by_id(
        self,
        *,
        owner_id: str,
        relationship_id: str,
        include_restricted_details: bool = False,
    ) -> RelationshipEdgeView | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM relationship_edges WHERE owner_id = ? AND relationship_id = ? LIMIT 1;",
                (owner_id, relationship_id),
            ).fetchone()
        if row is None:
            return None
        return self._edge_view_from_row(row, include_restricted_details=include_restricted_details)

    def diagnostics(
        self,
        *,
        owner_id: str,
        include_restricted_details: bool = False,
    ) -> RelationshipDiagnosticsResponse:
        _ = include_restricted_details
        with self._connect() as conn:
            entity_rows = conn.execute(
                "SELECT * FROM relationship_entities WHERE owner_id = ? ORDER BY entity_id;",
                (owner_id,),
            ).fetchall()
            edge_rows = conn.execute(
                "SELECT * FROM relationship_edges WHERE owner_id = ? ORDER BY created_at, relationship_id;",
                (owner_id,),
            ).fetchall()
            evidence_rows = conn.execute(
                """
                SELECT relationship_evidence.*, relationship_edges.sensitivity_level
                FROM relationship_evidence
                JOIN relationship_edges ON relationship_edges.relationship_id = relationship_evidence.relationship_id
                WHERE relationship_edges.owner_id = ?
                ORDER BY relationship_evidence.created_at, relationship_evidence.evidence_id;
                """,
                (owner_id,),
            ).fetchall()
        return RelationshipDiagnosticsResponse(
            entities=[self._entity_view_from_row(row) for row in entity_rows],
            relationships=[
                self._edge_view_from_row(row, include_restricted_details=False)
                for row in edge_rows
            ],
            evidence=[
                self._evidence_view_from_row(row, include_restricted_details=False)
                for row in evidence_rows
            ],
        )

    def select_relationships(self, body: RelationshipSelectRequest) -> RelationshipSelectResponse:
        diagnostics = self.diagnostics(owner_id=body.owner_id, include_restricted_details=False)
        persona_id, allowed_scopes = resolve_relationship_persona_scope(
            request_id=body.request_id,
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            surface=body.surface,
            runtime_session_id=body.runtime_session_id,
            active_persona_id=body.active_persona_id,
            requested_scopes=body.requested_scopes,
        )
        entity_map = {entity.entity_id: entity for entity in diagnostics.entities}
        relationships = diagnostics.relationships
        conflict_map = self._relationship_conflict_map(relationships)
        now = datetime.now(UTC)

        selected: list[RelationshipEdgeView] = []
        excluded: list[RelationshipExcludedSummary] = []
        prompt_lines: list[str] = []
        confirmation_required = False
        conflict_ids: list[str] = []
        exclusion_reason_map: dict[str, str] = {}

        for edge in relationships:
            reason: str | None = None
            if body.entity_ids and edge.subject_entity_id not in body.entity_ids and edge.object_entity_id not in body.entity_ids:
                reason = "filtered_entity_id"
            elif body.relationship_types and edge.relationship_type not in body.relationship_types:
                reason = "filtered_relationship_type"
            elif edge.relationship_scope not in allowed_scopes:
                reason = "outside_persona_or_surface_scope"
            elif edge.status != "active":
                if edge.status == "needs_confirmation":
                    confirmation_required = True
                reason = f"status_{edge.status}"
            elif edge.relationship_id in conflict_map:
                conflict_ids.append(edge.relationship_id)
                confirmation_required = True
                reason = "conflicted"
            elif _is_expired(edge.valid_until, now=now):
                reason = "expired"
            elif edge.sensitivity_level in _RESTRICTED_SENSITIVITY_LEVELS:
                confirmation_required = True
                reason = "authorization_required"
            elif edge.mentionability in {"confirm_before_mentioning", "restricted"}:
                confirmation_required = True
                reason = "authorization_required"
            elif edge.confidence < 0.75 and edge.source_type not in _THRESHOLD_EXEMPT_SOURCES:
                reason = "below_confidence_threshold"
            elif edge.allowed_persona_scopes_json and persona_id not in edge.allowed_persona_scopes_json:
                reason = "outside_persona_scope"
            elif persona_id in edge.blocked_persona_scopes_json:
                reason = "blocked_persona_scope"

            if reason is not None:
                exclusion_reason_map[edge.relationship_id] = reason
                excluded.append(
                    RelationshipExcludedSummary(
                        relationship_id=edge.relationship_id,
                        subject_entity_id=edge.subject_entity_id,
                        relationship_type=edge.relationship_type,
                        object_entity_id=edge.object_entity_id,
                        relationship_scope=edge.relationship_scope,
                        status=edge.status,
                        mentionability=edge.mentionability,
                        sensitivity_level=edge.sensitivity_level,
                        confidence=edge.confidence,
                        reason=reason,
                    )
                )
                continue

            selected.append(edge)
            if edge.mentionability == "mentionable":
                subject_label = self._entity_label(entity_map.get(edge.subject_entity_id), edge.subject_entity_id)
                object_label = self._entity_label(entity_map.get(edge.object_entity_id), edge.object_entity_id)
                prompt_lines.append(
                    (
                        f"- {subject_label} {edge.relationship_type} {object_label} "
                        f"(scope={edge.relationship_scope}; confidence={edge.confidence:.2f})"
                    )
                )
            elif edge.mentionability in {"use_for_routing_only", "use_for_filtering_only"}:
                exclusion_reason_map.setdefault(edge.relationship_id, edge.mentionability)

        selected_entity_ids: set[str] = set()
        for edge in selected:
            selected_entity_ids.add(edge.subject_entity_id)
            selected_entity_ids.add(edge.object_entity_id)
        selected_entities = [
            entity_map[entity_id]
            for entity_id in sorted(selected_entity_ids)
            if entity_id in entity_map
        ]
        trace = RelationshipSelectTrace(
            relationship_edges_used=[edge.relationship_id for edge in selected],
            relationship_edges_excluded=[item.relationship_id for item in excluded],
            relationship_exclusion_reasons=exclusion_reason_map,
            relationship_context_overlay_applied=bool(prompt_lines),
            relationship_conflicts=sorted(set(conflict_ids)),
            relationship_confirmation_required=confirmation_required,
            selected_relationship_count=len(selected),
            excluded_relationship_count=len(excluded),
            active_persona_id=persona_id,
            allowed_relationship_scopes=sorted(allowed_scopes),
        )
        return RelationshipSelectResponse(
            selected_entities=selected_entities,
            selected_relationships=selected,
            excluded_relationship_summaries=excluded,
            prompt_content="Relationship context:\n" + "\n".join(prompt_lines) if prompt_lines else None,
            trace=trace,
        )

    def _validate_edge_input(self, edge: RelationshipEdgeInput) -> None:
        if not edge.source_refs_json:
            raise RuntimeError("relationship_source_refs_required")
        if edge.source_type == "model_inference" and edge.status == "active":
            raise RuntimeError("model_inference_cannot_create_active_relationship")
        if edge.relationship_type in _SOCIALISH_RELATIONSHIP_TYPES:
            if edge.status == "active" and edge.source_type not in {
                "explicit_user_confirmation",
                "trusted_config",
                "trusted_integration_metadata",
            }:
                raise RuntimeError("trusted_provenance_required_for_active_socialish_relationship")
            if edge.source_type == "model_inference" and edge.status != "needs_confirmation":
                raise RuntimeError("model_inference_socialish_relationship_requires_confirmation")

    def _upsert_entity_with_conn(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        entity: RelationshipEntityInput,
        now: str,
    ) -> None:
        existing = conn.execute(
            "SELECT 1 FROM relationship_entities WHERE owner_id = ? AND entity_id = ? LIMIT 1;",
            (owner_id, entity.entity_id),
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO relationship_entities (
                    entity_id, owner_id, entity_type, canonical_label, display_label,
                    domain, sensitivity_level, source_type, source_ref,
                    canonical_memory_ref, artifact_ref, status,
                    created_at, updated_at, archived_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    entity.entity_id,
                    owner_id,
                    entity.entity_type,
                    entity.canonical_label,
                    entity.display_label,
                    entity.domain,
                    entity.sensitivity_level,
                    entity.source_type,
                    entity.source_ref,
                    entity.canonical_memory_ref,
                    entity.artifact_ref,
                    entity.status,
                    now,
                    now,
                    entity.archived_at,
                ),
            )
            return
        conn.execute(
            """
            UPDATE relationship_entities
            SET entity_type = ?, canonical_label = ?, display_label = ?, domain = ?,
                sensitivity_level = ?, source_type = ?, source_ref = ?,
                canonical_memory_ref = ?, artifact_ref = ?, status = ?,
                archived_at = ?, updated_at = ?
            WHERE owner_id = ? AND entity_id = ?;
            """,
            (
                entity.entity_type,
                entity.canonical_label,
                entity.display_label,
                entity.domain,
                entity.sensitivity_level,
                entity.source_type,
                entity.source_ref,
                entity.canonical_memory_ref,
                entity.artifact_ref,
                entity.status,
                entity.archived_at,
                now,
                owner_id,
                entity.entity_id,
            ),
        )

    def _upsert_edge_with_conn(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        relationship_id: str,
        edge: RelationshipEdgeInput,
        normalized_status: RelationshipStatus | str,
        now: str,
    ) -> None:
        self._validate_edge_input(edge)
        self._require_entity(conn, owner_id=owner_id, entity_id=edge.subject_entity_id)
        self._require_entity(conn, owner_id=owner_id, entity_id=edge.object_entity_id)
        existing = conn.execute(
            "SELECT 1 FROM relationship_edges WHERE owner_id = ? AND relationship_id = ? LIMIT 1;",
            (owner_id, relationship_id),
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO relationship_edges (
                    relationship_id, owner_id, subject_entity_id, relationship_type,
                    object_entity_id, relationship_scope, source_type, source_refs_json,
                    confidence, status, sensitivity_level, mentionability,
                    allowed_persona_scopes_json, blocked_persona_scopes_json,
                    valid_from, valid_until, superseded_by_relationship_id,
                    created_at, updated_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    relationship_id,
                    owner_id,
                    edge.subject_entity_id,
                    edge.relationship_type,
                    edge.object_entity_id,
                    edge.relationship_scope,
                    edge.source_type,
                    _json(edge.source_refs_json),
                    edge.confidence,
                    normalized_status,
                    edge.sensitivity_level,
                    edge.mentionability,
                    _json(edge.allowed_persona_scopes_json),
                    _json(edge.blocked_persona_scopes_json),
                    edge.valid_from,
                    edge.valid_until,
                    edge.superseded_by_relationship_id,
                    now,
                    now,
                    edge.revoked_at,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE relationship_edges
                SET subject_entity_id = ?, relationship_type = ?, object_entity_id = ?,
                    relationship_scope = ?, source_type = ?, source_refs_json = ?,
                    confidence = ?, status = ?, sensitivity_level = ?, mentionability = ?,
                    allowed_persona_scopes_json = ?, blocked_persona_scopes_json = ?,
                    valid_from = ?, valid_until = ?, superseded_by_relationship_id = ?,
                    revoked_at = ?, updated_at = ?
                WHERE owner_id = ? AND relationship_id = ?;
                """,
                (
                    edge.subject_entity_id,
                    edge.relationship_type,
                    edge.object_entity_id,
                    edge.relationship_scope,
                    edge.source_type,
                    _json(edge.source_refs_json),
                    edge.confidence,
                    normalized_status,
                    edge.sensitivity_level,
                    edge.mentionability,
                    _json(edge.allowed_persona_scopes_json),
                    _json(edge.blocked_persona_scopes_json),
                    edge.valid_from,
                    edge.valid_until,
                    edge.superseded_by_relationship_id,
                    edge.revoked_at,
                    now,
                    owner_id,
                    relationship_id,
                ),
            )
        if edge.supersede_existing_relationship_id:
            self._supersede_existing(
                conn,
                owner_id=owner_id,
                existing_relationship_id=edge.supersede_existing_relationship_id,
                new_relationship_id=relationship_id,
                updated_at=now,
            )

    def _normalized_edge_status(self, edge: RelationshipEdgeInput) -> str:
        status = edge.status
        if _is_model_inference(edge):
            if edge.sensitivity_level in _RESTRICTED_SENSITIVITY_LEVELS:
                return "needs_confirmation"
            if edge.relationship_type in _SOCIALISH_RELATIONSHIP_TYPES:
                return "needs_confirmation"
            if status not in _MODEL_INFERENCE_ALLOWED_STATUSES:
                return "needs_confirmation"
        return status

    def _require_entity(self, conn: sqlite3.Connection, *, owner_id: str, entity_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM relationship_entities WHERE owner_id = ? AND entity_id = ? LIMIT 1;",
            (owner_id, entity_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("relationship_entity_not_found")
        return row

    def _require_edge(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        relationship_id: str,
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM relationship_edges WHERE owner_id = ? AND relationship_id = ? LIMIT 1;",
            (owner_id, relationship_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("relationship_edge_not_found")
        return row

    def _supersede_existing(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        existing_relationship_id: str,
        new_relationship_id: str,
        updated_at: str,
    ) -> None:
        self._require_edge(conn, owner_id=owner_id, relationship_id=existing_relationship_id)
        conn.execute(
            """
            UPDATE relationship_edges
            SET status = 'superseded', superseded_by_relationship_id = ?, updated_at = ?
            WHERE owner_id = ? AND relationship_id = ?;
            """,
            (new_relationship_id, updated_at, owner_id, existing_relationship_id),
        )

    def _insert_evidence(
        self,
        conn: sqlite3.Connection,
        *,
        relationship_id: str,
        evidence: RelationshipEdgeEvidenceInput,
    ) -> RelationshipEdgeEvidenceView:
        now = _now()
        evidence_id = _digest(
            "relev",
            f"{relationship_id}:{evidence.evidence_type}:{evidence.source_ref}:{now}",
        )
        conn.execute(
            """
            INSERT INTO relationship_evidence (
                evidence_id, relationship_id, evidence_type, source_ref, summary,
                confidence_delta, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                evidence_id,
                relationship_id,
                evidence.evidence_type,
                evidence.source_ref,
                evidence.summary,
                evidence.confidence_delta,
                now,
            ),
        )
        row = conn.execute(
            """
            SELECT relationship_evidence.*, relationship_edges.sensitivity_level
            FROM relationship_evidence
            JOIN relationship_edges ON relationship_edges.relationship_id = relationship_evidence.relationship_id
            WHERE evidence_id = ? LIMIT 1;
            """,
            (evidence_id,),
        ).fetchone()
        assert row is not None
        return self._evidence_view_from_row(row, include_restricted_details=False)

    def _insert_evidence_deduped(
        self,
        conn: sqlite3.Connection,
        *,
        relationship_id: str,
        evidence: RelationshipEdgeEvidenceInput,
    ) -> tuple[RelationshipEdgeEvidenceView, bool]:
        existing = conn.execute(
            """
            SELECT relationship_evidence.*, relationship_edges.sensitivity_level
            FROM relationship_evidence
            JOIN relationship_edges ON relationship_edges.relationship_id = relationship_evidence.relationship_id
            WHERE relationship_evidence.relationship_id = ?
              AND relationship_evidence.evidence_type = ?
              AND relationship_evidence.source_ref = ?
              AND COALESCE(relationship_evidence.summary, '') = COALESCE(?, '')
            LIMIT 1;
            """,
            (
                relationship_id,
                evidence.evidence_type,
                evidence.source_ref,
                evidence.summary,
            ),
        ).fetchone()
        if existing is not None:
            return self._evidence_view_from_row(existing, include_restricted_details=False), False
        return self._insert_evidence(conn, relationship_id=relationship_id, evidence=evidence), True

    def _entity_ids_for_owner(self, conn: sqlite3.Connection, *, owner_id: str) -> set[str]:
        rows = conn.execute(
            "SELECT entity_id FROM relationship_entities WHERE owner_id = ?;",
            (owner_id,),
        ).fetchall()
        return {row["entity_id"] for row in rows}

    def _entity_view_from_row(self, row: sqlite3.Row) -> RelationshipEntityView:
        return RelationshipEntityView(
            entity_id=row["entity_id"],
            owner_id=row["owner_id"],
            entity_type=row["entity_type"],
            canonical_label=row["canonical_label"],
            display_label=row["display_label"],
            domain=row["domain"],
            sensitivity_level=row["sensitivity_level"],
            source_type=row["source_type"],
            source_ref=row["source_ref"],
            canonical_memory_ref=row["canonical_memory_ref"],
            artifact_ref=row["artifact_ref"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            archived_at=row["archived_at"],
        )

    def _edge_view_from_row(
        self,
        row: sqlite3.Row,
        *,
        include_restricted_details: bool,
    ) -> RelationshipEdgeView:
        restricted = row["sensitivity_level"] in _RESTRICTED_SENSITIVITY_LEVELS and not include_restricted_details
        source_refs = _load_json(row["source_refs_json"], [])
        return RelationshipEdgeView(
            relationship_id=row["relationship_id"],
            owner_id=row["owner_id"],
            subject_entity_id=row["subject_entity_id"],
            relationship_type=row["relationship_type"],
            object_entity_id=row["object_entity_id"],
            relationship_scope=row["relationship_scope"],
            source_type=row["source_type"],
            source_refs_json=[] if restricted else source_refs,
            source_refs_redacted=restricted,
            confidence=row["confidence"],
            status=row["status"],
            sensitivity_level=row["sensitivity_level"],
            mentionability=row["mentionability"],
            allowed_persona_scopes_json=_load_json(row["allowed_persona_scopes_json"], []),
            blocked_persona_scopes_json=_load_json(row["blocked_persona_scopes_json"], []),
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            superseded_by_relationship_id=row["superseded_by_relationship_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            revoked_at=row["revoked_at"],
        )

    def _evidence_view_from_row(
        self,
        row: sqlite3.Row,
        *,
        include_restricted_details: bool,
    ) -> RelationshipEdgeEvidenceView:
        restricted = row["sensitivity_level"] in _RESTRICTED_SENSITIVITY_LEVELS and not include_restricted_details
        return RelationshipEdgeEvidenceView(
            evidence_id=row["evidence_id"],
            relationship_id=row["relationship_id"],
            evidence_type=row["evidence_type"],
            source_ref=row["source_ref"],
            summary=None if restricted else row["summary"],
            summary_redacted=restricted,
            confidence_delta=row["confidence_delta"],
            created_at=row["created_at"],
        )

    def _relationship_conflict_map(
        self,
        relationships: list[RelationshipEdgeView],
    ) -> dict[str, list[str]]:
        buckets: dict[tuple[str, str, str], list[RelationshipEdgeView]] = {}
        for edge in relationships:
            if edge.status != "active":
                continue
            if edge.superseded_by_relationship_id or edge.revoked_at:
                continue
            key = (edge.subject_entity_id, edge.relationship_type, edge.relationship_scope)
            buckets.setdefault(key, []).append(edge)
        conflicts: dict[str, list[str]] = {}
        for items in buckets.values():
            object_ids = {item.object_entity_id for item in items}
            if len(object_ids) <= 1:
                continue
            ids = [item.relationship_id for item in items]
            for relationship_id in ids:
                conflicts[relationship_id] = [other for other in ids if other != relationship_id]
        return conflicts

    def _entity_label(self, entity: RelationshipEntityView | None, fallback: str) -> str:
        if entity is None:
            return fallback
        return entity.display_label or entity.canonical_label or fallback


def relationship_repository() -> RelationshipRepository:
    global _RELATIONSHIP_REPOSITORY
    if _RELATIONSHIP_REPOSITORY is None:
        _RELATIONSHIP_REPOSITORY = RelationshipRepository()
    return _RELATIONSHIP_REPOSITORY


def clear_relationships_for_tests(db_path: Path | None = None) -> None:
    global _RELATIONSHIP_REPOSITORY
    _RELATIONSHIP_REPOSITORY = RelationshipRepository(db_path=db_path)


def upsert_relationship_entity(
    *,
    owner_id: str,
    entity: RelationshipEntityInput,
) -> RelationshipEntityView:
    return relationship_repository().upsert_entity(owner_id=owner_id, entity=entity)


def upsert_relationship_edge(
    *,
    owner_id: str,
    edge: RelationshipEdgeInput,
    evidence: list[RelationshipEdgeEvidenceInput],
) -> RelationshipEdgeResponse:
    return relationship_repository().upsert_edge(owner_id=owner_id, edge=edge, evidence=evidence)


def confirm_relationship_edge(
    *,
    owner_id: str,
    body: RelationshipEdgeConfirmRequest,
) -> RelationshipEdgeResponse:
    return relationship_repository().confirm_edge(owner_id=owner_id, body=body)


def revoke_relationship_edge(
    *,
    owner_id: str,
    body: RelationshipEdgeRevokeRequest,
) -> RelationshipEdgeResponse:
    return relationship_repository().revoke_edge(owner_id=owner_id, body=body)


def get_relationship_entity(*, owner_id: str, entity_id: str) -> RelationshipEntityView | None:
    return relationship_repository().entity_by_id(owner_id=owner_id, entity_id=entity_id)


def get_relationship_edge(*, owner_id: str, relationship_id: str) -> RelationshipEdgeView | None:
    return relationship_repository().edge_by_id(owner_id=owner_id, relationship_id=relationship_id)


def get_relationship_diagnostics(
    body: RelationshipGraphDiagnosticsRequest,
) -> RelationshipDiagnosticsResponse:
    return relationship_repository().diagnostics(
        owner_id=body.owner_id,
        include_restricted_details=False,
    )


def resolve_relationship_persona_scope(
    *,
    request_id: str,
    owner_id: str,
    conversation_id: str,
    surface: str,
    runtime_session_id: str | None,
    active_persona_id: str | None = None,
    requested_scopes: list[str] | None = None,
) -> tuple[str, set[str]]:
    if active_persona_id is None:
        identity = resolve_runtime_identity(
            RuntimeIdentityResolveRequest(
                request_id=request_id,
                owner_id=owner_id,
                conversation_id=conversation_id,
                surface=surface,
                runtime_session_id=runtime_session_id,
            )
        )
        active_persona_id = identity.runtime_identity.active_persona_id
        surface_type = identity.runtime_identity.surface_type
    else:
        binding = companion_contracts_repository().surface_binding(surface)
        surface_type = binding.surface_type if binding is not None else "unknown_surface"
    persona_scopes = _RELATIONSHIP_SCOPE_ALLOWLISTS.get(
        active_persona_id,
        _RELATIONSHIP_SCOPE_ALLOWLISTS["general_assistant"],
    )
    surface_scopes = _SURFACE_SCOPE_RESTRICTIONS.get(
        surface_type,
        _SURFACE_SCOPE_RESTRICTIONS["unknown_surface"],
    )
    allowed = set(persona_scopes) & set(surface_scopes)
    if requested_scopes:
        allowed &= {scope for scope in requested_scopes if scope}
    return active_persona_id, allowed


def select_relationships(body: RelationshipSelectRequest) -> RelationshipSelectResponse:
    return relationship_repository().select_relationships(body)
