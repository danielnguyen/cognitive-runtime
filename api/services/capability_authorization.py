from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from models import (
    CapabilityAuthorizationRequest,
    CapabilityAuthorizationResponse,
    CapabilityAuthorizationResult,
    CapabilityConfirmationRequest,
    CapabilityConfirmationResponse,
    CapabilityRevalidationSelector,
    RuntimeIdentityResolveRequest,
)
from services.companion_contracts import companion_contracts_repository
from services.relationships import get_relationship_edge
from services.runtime_identity import resolve_runtime_identity
from services.runtime_state import (
    record_runtime_event,
    runtime_session_by_id,
    runtime_state_db_path,
    validate_runtime_turn_session,
)
from services.world_state import get_world_state_diagnostics

_CAPABILITY_AUTH_REPOSITORY: CapabilityAuthorizationRepository | None = None
_RISKY_OPERATION_CLASSES = {"external_write", "destructive", "high_impact"}
_AUTHORIZED_AUTHORITIES = {
    "verified_tool_output": 4,
    "trusted_integration_event": 3,
    "derived_from_multiple_sources": 2,
    "observed_user_report": 1,
    "model_inferred": 0,
    "unverified_assumption": 0,
}
_FRESHNESS_ORDER = {
    "fresh": 0,
    "aging": 1,
    "stale": 2,
    "unknown": 3,
    "expired": 4,
    "superseded": 5,
    "conflicted": 6,
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _digest(prefix: str, material: str) -> str:
    return f"{prefix}_{sha256(material.encode('utf-8')).hexdigest()[:16]}"


def _session_and_turn(body: CapabilityAuthorizationRequest | CapabilityConfirmationRequest) -> None:
    session = runtime_session_by_id(body.runtime_session_id)
    if session is None:
        raise RuntimeError("runtime_session_not_found")
    if (
        session.owner_id != body.owner_id
        or session.conversation_id != body.conversation_id
        or session.surface != body.surface
    ):
        raise RuntimeError("runtime_session_mismatch")
    if body.runtime_turn_id:
        validate_runtime_turn_session(
            runtime_session_id=body.runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
        )


class CapabilityAuthorizationRepository:
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
                CREATE TABLE IF NOT EXISTS capability_confirmation_challenges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    confirmation_challenge_ref TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    runtime_session_id TEXT NOT NULL,
                    originating_runtime_turn_id TEXT,
                    originating_request_id TEXT NOT NULL,
                    capability_id TEXT NOT NULL,
                    operation_class TEXT NOT NULL,
                    argument_digest TEXT NOT NULL,
                    status TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    confirmed_runtime_turn_id TEXT,
                    confirmed_at TEXT,
                    consumed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(owner_id, confirmation_challenge_ref)
                );
                """
            )

    def authorize(self, body: CapabilityAuthorizationRequest) -> CapabilityAuthorizationResponse:
        _session_and_turn(body)
        identity = resolve_runtime_identity(
            RuntimeIdentityResolveRequest(
                request_id=body.request_id,
                owner_id=body.owner_id,
                conversation_id=body.conversation_id,
                surface=body.surface,
                runtime_session_id=body.runtime_session_id,
            )
        )
        reason_codes: list[str] = []
        relationship_ids_used: list[str] = []
        world_state_claim_ids_used: list[str] = []
        revalidation_selector: CapabilityRevalidationSelector | None = None

        binding = companion_contracts_repository().surface_binding(body.surface)
        if binding is None:
            reason_codes.append("unknown_surface")
        if body.active_persona_id != identity.runtime_identity.active_persona_id:
            reason_codes.append("persona_mismatch")
        persona_domains = {
            identity.runtime_identity.capability_domain,
            *identity.runtime_identity.advisory_tool_permission_summary,
        }
        if body.capability_domain not in persona_domains:
            reason_codes.append("capability_domain_denied")
        if body.supported_surfaces and body.surface not in body.supported_surfaces:
            reason_codes.append("surface_unsupported")
        if body.authorization_phase in {"selection", "dispatch"} and not body.argument_digest:
            reason_codes.append("argument_digest_required")

        relationship_reasons = self._relationship_reasons(body)
        reason_codes.extend(relationship_reasons[0])
        relationship_ids_used = relationship_reasons[1]

        world_reasons = self._world_state_reasons(body)
        reason_codes.extend(world_reasons[0])
        world_state_claim_ids_used = world_reasons[1]
        revalidation_selector = world_reasons[2]

        challenge_ref: str | None = body.confirmation_challenge_ref
        confirmation_state = "not_required"
        if body.operation_class in _RISKY_OPERATION_CLASSES:
            if body.authorization_phase == "selection":
                if body.argument_digest and not reason_codes:
                    challenge_ref = self._issue_challenge(body)
                    confirmation_state = "issued"
                    reason_codes.append("confirmation_required")
                else:
                    confirmation_state = "required"
            elif body.authorization_phase == "dispatch":
                confirmation_state = "required"
                challenge_reasons = self._validate_dispatch_challenge(body)
                reason_codes.extend(challenge_reasons)
                if not challenge_reasons:
                    confirmation_state = "accepted"
            else:
                confirmation_state = "required"

        if (
            revalidation_selector is not None
            and "world_state_revalidation_required" not in reason_codes
        ):
            reason_codes.append("world_state_revalidation_required")

        allowed = not reason_codes
        decision_code = "allowed"
        if revalidation_selector is not None:
            decision_code = "revalidation_required"
            allowed = False
        elif "confirmation_required" in reason_codes:
            decision_code = "confirmation_required"
            allowed = False
        elif any(code.startswith("challenge_") for code in reason_codes):
            decision_code = "confirmation_rejected"
            allowed = False
        elif reason_codes:
            decision_code = "authorization_denied"
            allowed = False

        if allowed and body.authorization_phase == "dispatch" and body.confirmation_challenge_ref:
            self._consume_challenge(body)
        result = CapabilityAuthorizationResult(
            phase=body.authorization_phase,
            allowed=allowed,
            decision_code=decision_code,
            reason_codes=reason_codes or ["allowed"],
            confirmation_state=confirmation_state,
            challenge_ref=challenge_ref,
            revalidation_required=revalidation_selector is not None,
            revalidation_selector=revalidation_selector,
            relationship_ids_used=relationship_ids_used,
            world_state_claim_ids_used=world_state_claim_ids_used,
        )
        record_runtime_event(
            runtime_session_id=body.runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
            event_type="capability_authorization_evaluated",
            event_payload_json={
                "request_id": body.request_id,
                "phase": body.authorization_phase,
                "capability_id": body.capability_id,
                "operation_class": body.operation_class,
                "decision_code": result.decision_code,
                "reason_codes": result.reason_codes,
                "relationship_ids_used": relationship_ids_used,
                "world_state_claim_ids_used": world_state_claim_ids_used,
                "confirmation_state": confirmation_state,
                "challenge_ref": challenge_ref,
                "revalidation_required": result.revalidation_required,
            },
        )
        return CapabilityAuthorizationResponse(
            request_id=body.request_id,
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            runtime_session_id=body.runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
            capability_id=body.capability_id,
            result=result,
        )

    def confirm(self, body: CapabilityConfirmationRequest) -> CapabilityConfirmationResponse:
        _session_and_turn(body)
        now = _now()
        with self._connect() as conn:
            row = self._challenge_row(conn, body.owner_id, body.confirmation_challenge_ref)
            if row is None:
                raise RuntimeError("confirmation_challenge_not_found")
            reason = self._challenge_base_mismatch(row, body)
            if reason:
                raise RuntimeError("confirmation_challenge_mismatch")
            if row["consumed_at"]:
                raise RuntimeError("confirmation_challenge_consumed")
            if _parse_ts(row["expires_at"]) <= datetime.now(UTC):
                raise RuntimeError("confirmation_challenge_expired")
            status = "accepted" if body.confirmed else "rejected"
            conn.execute(
                """
                UPDATE capability_confirmation_challenges
                SET status = ?, confirmed_runtime_turn_id = ?, confirmed_at = ?, updated_at = ?
                WHERE owner_id = ? AND confirmation_challenge_ref = ?;
                """,
                (
                    status,
                    body.runtime_turn_id,
                    now,
                    now,
                    body.owner_id,
                    body.confirmation_challenge_ref,
                ),
            )
        record_runtime_event(
            runtime_session_id=body.runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
            event_type="confirmation_challenge_evaluated",
            event_payload_json={
                "request_id": body.request_id,
                "confirmation_challenge_ref": body.confirmation_challenge_ref,
                "capability_id": body.capability_id,
                "operation_class": body.operation_class,
                "confirmation_state": status,
            },
        )
        return CapabilityConfirmationResponse(
            request_id=body.request_id,
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            runtime_session_id=body.runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
            confirmation_challenge_ref=body.confirmation_challenge_ref,
            confirmation_state=status,
        )

    def _relationship_reasons(
        self,
        body: CapabilityAuthorizationRequest,
    ) -> tuple[list[str], list[str]]:
        reasons: list[str] = []
        used: list[str] = []
        for requirement in body.relationship_requirements:
            candidates = [
                rel_id
                for rel_id in body.selected_relationship_ids
                if (edge := get_relationship_edge(owner_id=body.owner_id, relationship_id=rel_id))
                and (
                    not requirement.relationship_scope
                    or edge.relationship_scope == requirement.relationship_scope
                )
                and (
                    not requirement.relationship_type
                    or edge.relationship_type == requirement.relationship_type
                )
            ]
            if not candidates:
                reasons.append("relationship_required")
                continue
            for rel_id in candidates:
                edge = get_relationship_edge(owner_id=body.owner_id, relationship_id=rel_id)
                if edge is None:
                    reasons.append("relationship_not_selected")
                    continue
                if (
                    edge.status != "active"
                    or edge.superseded_by_relationship_id
                    or edge.sensitivity_level in {"high", "restricted"}
                    or edge.mentionability
                    in {"confirm_before_mentioning", "suppress_by_default", "restricted"}
                    or edge.confidence < 0.75
                    or (edge.valid_until and _parse_ts(edge.valid_until) <= datetime.now(UTC))
                ):
                    reasons.append("relationship_not_authorized")
                    continue
                used.append(edge.relationship_id)
        return sorted(set(reasons)), sorted(set(used))

    def _world_state_reasons(
        self,
        body: CapabilityAuthorizationRequest,
    ) -> tuple[list[str], list[str], CapabilityRevalidationSelector | None]:
        diagnostics = get_world_state_diagnostics(
            owner_id=body.owner_id,
            include_sensitive_values=False,
        )
        claim_map = {claim.world_state_claim_id: claim for claim in diagnostics.claims}
        reasons: list[str] = []
        used: list[str] = []
        revalidation_claims: list[str] = []
        revalidator_id: str | None = None
        for requirement in body.world_state_requirements:
            candidates = [
                claim_map[claim_id]
                for claim_id in body.selected_world_state_claim_ids
                if claim_id in claim_map
            ]
            if requirement.domain:
                candidates = [claim for claim in candidates if claim.domain == requirement.domain]
            if requirement.attribute:
                candidates = [
                    claim for claim in candidates if claim.attribute == requirement.attribute
                ]
            if requirement.entity_id:
                candidates = [
                    claim for claim in candidates if claim.entity_id == requirement.entity_id
                ]
            if not candidates:
                reasons.append("world_state_required")
                continue
            for claim in candidates:
                if claim.effective_freshness_state in {
                    "expired",
                    "superseded",
                    "conflicted",
                    "unknown",
                }:
                    reasons.append("world_state_not_authorized")
                    continue
                if claim.effective_freshness_state == "stale" and requirement.revalidator_id:
                    revalidation_claims.append(claim.world_state_claim_id)
                    revalidator_id = requirement.revalidator_id
                    continue
                if requirement.max_freshness_state and (
                    _FRESHNESS_ORDER[claim.effective_freshness_state]
                    > _FRESHNESS_ORDER[requirement.max_freshness_state]
                ):
                    reasons.append("world_state_not_authorized")
                    continue
                if (
                    requirement.min_confidence is not None
                    and claim.confidence < requirement.min_confidence
                ):
                    reasons.append("world_state_not_authorized")
                    continue
                if requirement.min_authority and (
                    _AUTHORIZED_AUTHORITIES[claim.state_authority]
                    < _AUTHORIZED_AUTHORITIES[requirement.min_authority]
                ):
                    reasons.append("world_state_not_authorized")
                    continue
                if body.operation_class in _RISKY_OPERATION_CLASSES and claim.state_authority in {
                    "model_inferred",
                    "unverified_assumption",
                    "observed_user_report",
                }:
                    reasons.append("world_state_not_authorized")
                    continue
                used.append(claim.world_state_claim_id)
        selector = None
        if revalidation_claims and revalidator_id:
            selector = CapabilityRevalidationSelector(
                world_state_claim_ids=sorted(set(revalidation_claims)),
                revalidator_id=revalidator_id,
            )
        return sorted(set(reasons)), sorted(set(used)), selector

    def _issue_challenge(self, body: CapabilityAuthorizationRequest) -> str:
        assert body.argument_digest is not None
        issued_at = _now()
        expires_at = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
        challenge_ref = _digest(
            "capconfirm",
            (
                f"{body.owner_id}:{body.conversation_id}:{body.runtime_session_id}:"
                f"{body.runtime_turn_id}:{body.capability_id}:{body.operation_class}:"
                f"{body.argument_digest}:{issued_at}"
            ),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO capability_confirmation_challenges (
                    confirmation_challenge_ref, owner_id, conversation_id, runtime_session_id,
                    originating_runtime_turn_id, originating_request_id, capability_id,
                    operation_class, argument_digest, status, issued_at, expires_at,
                    confirmed_runtime_turn_id, confirmed_at, consumed_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    challenge_ref,
                    body.owner_id,
                    body.conversation_id,
                    body.runtime_session_id,
                    body.runtime_turn_id,
                    body.request_id,
                    body.capability_id,
                    body.operation_class,
                    body.argument_digest,
                    "issued",
                    issued_at,
                    expires_at,
                    None,
                    None,
                    None,
                    issued_at,
                    issued_at,
                ),
            )
        record_runtime_event(
            runtime_session_id=body.runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
            event_type="confirmation_challenge_evaluated",
            event_payload_json={
                "request_id": body.request_id,
                "confirmation_challenge_ref": challenge_ref,
                "capability_id": body.capability_id,
                "operation_class": body.operation_class,
                "confirmation_state": "issued",
                "expires_at": expires_at,
            },
        )
        return challenge_ref

    def _validate_dispatch_challenge(self, body: CapabilityAuthorizationRequest) -> list[str]:
        if not body.confirmation_challenge_ref:
            return ["challenge_missing"]
        with self._connect() as conn:
            row = self._challenge_row(conn, body.owner_id, body.confirmation_challenge_ref)
        if row is None:
            return ["challenge_missing"]
        mismatch = self._challenge_base_mismatch(row, body)
        if mismatch:
            return [mismatch]
        if row["consumed_at"]:
            return ["challenge_consumed"]
        if _parse_ts(row["expires_at"]) <= datetime.now(UTC):
            return ["challenge_expired"]
        if row["status"] != "accepted":
            return ["challenge_not_confirmed"]
        return []

    def _consume_challenge(self, body: CapabilityAuthorizationRequest) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE capability_confirmation_challenges
                SET consumed_at = ?, status = 'accepted', updated_at = ?
                WHERE owner_id = ? AND confirmation_challenge_ref = ? AND consumed_at IS NULL;
                """,
                (now, now, body.owner_id, body.confirmation_challenge_ref),
            )
        record_runtime_event(
            runtime_session_id=body.runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
            event_type="confirmation_challenge_evaluated",
            event_payload_json={
                "request_id": body.request_id,
                "confirmation_challenge_ref": body.confirmation_challenge_ref,
                "capability_id": body.capability_id,
                "operation_class": body.operation_class,
                "confirmation_state": "consumed",
            },
        )

    def _challenge_row(
        self,
        conn: sqlite3.Connection,
        owner_id: str,
        challenge_ref: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT * FROM capability_confirmation_challenges
            WHERE owner_id = ? AND confirmation_challenge_ref = ?
            LIMIT 1;
            """,
            (owner_id, challenge_ref),
        ).fetchone()

    def _challenge_base_mismatch(
        self,
        row: sqlite3.Row,
        body: CapabilityAuthorizationRequest | CapabilityConfirmationRequest,
    ) -> str | None:
        if (
            row["conversation_id"] != body.conversation_id
            or row["runtime_session_id"] != body.runtime_session_id
            or row["capability_id"] != body.capability_id
            or row["operation_class"] != body.operation_class
            or row["argument_digest"] != body.argument_digest
        ):
            return "challenge_mismatch"
        return None


def capability_authorization_repository() -> CapabilityAuthorizationRepository:
    global _CAPABILITY_AUTH_REPOSITORY
    if _CAPABILITY_AUTH_REPOSITORY is None:
        _CAPABILITY_AUTH_REPOSITORY = CapabilityAuthorizationRepository()
    return _CAPABILITY_AUTH_REPOSITORY


def authorize_capability(body: CapabilityAuthorizationRequest) -> CapabilityAuthorizationResponse:
    return capability_authorization_repository().authorize(body)


def record_capability_confirmation(
    body: CapabilityConfirmationRequest,
) -> CapabilityConfirmationResponse:
    return capability_authorization_repository().confirm(body)


def clear_capability_authorization_for_tests(db_path: Path | None = None) -> None:
    global _CAPABILITY_AUTH_REPOSITORY
    _CAPABILITY_AUTH_REPOSITORY = CapabilityAuthorizationRepository(db_path=db_path)
