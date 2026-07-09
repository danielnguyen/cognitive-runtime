from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from models import (
    CapabilityAuthorizationRequest,
    CapabilityAuthorizationResponse,
    CapabilityAuthorizationResult,
    CapabilityDiscoveryExample,
    CapabilityDiscoveryRequest,
    CapabilityDiscoveryResponse,
    CapabilityDiscoveryResult,
    CapabilityMatchRequest,
    CapabilityMatchResponse,
    CapabilityMatchResult,
    CapabilityConfirmationRequest,
    CapabilityConfirmationResponse,
    CapabilityRegistryRecord,
    CapabilityRevalidationSelector,
    RelationshipSelectRequest,
    RuntimeIdentityResolveRequest,
)
from services.companion_contracts import companion_contracts_repository
from services.relationships import select_relationships
from services.runtime_identity import resolve_runtime_identity
from services.runtime_state import (
    get_runtime_session,
    record_runtime_event,
    runtime_session_by_id,
    runtime_state_db_path,
    validate_runtime_turn_session,
)
from services.world_state import (
    resolve_world_state,
    trusted_world_state_verifier_selector_reason,
)

_CAPABILITY_AUTH_REPOSITORY: CapabilityAuthorizationRepository | None = None
_CAPABILITY_REGISTRY_AVAILABLE = True
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
_HIGH_IMPACT_OPERATION_CLASSES = {"external_write", "destructive", "high_impact"}


@dataclass(frozen=True)
class RegisteredCapability:
    record: CapabilityRegistryRecord
    match_phrases: tuple[str, ...]


_CAPABILITY_REGISTRY: tuple[RegisteredCapability, ...] = (
    RegisteredCapability(
        record=CapabilityRegistryRecord(
            capability_id="service_health_check",
            display_name="Run service health check",
            domain="operations",
            description="Reads local service health status without changing state.",
            operation_kind="read_only",
            risk_level="low_read_only",
            requires_confirmation=False,
            allowed_surfaces=["desktop", "dev", "telegram", "voice_private"],
            allowed_personas=["default_assistant", "home_operator", "technical_architect"],
            reversible=True,
            dry_run_supported=True,
            verification_supported=True,
            audit_required=False,
        ),
        match_phrases=("health check", "service health", "check service status"),
    ),
    RegisteredCapability(
        record=CapabilityRegistryRecord(
            capability_id="office_lights_on",
            display_name="Turn on office lights",
            domain="home_automation",
            description="Turns on the office lights through the local automation layer.",
            operation_kind="state_change",
            risk_level="low_reversible",
            requires_confirmation=False,
            allowed_surfaces=["desktop", "telegram", "voice_private"],
            allowed_personas=["default_assistant", "home_operator"],
            reversible=True,
            dry_run_supported=True,
            verification_supported=True,
            audit_required=True,
        ),
        match_phrases=(
            "turn on office lights",
            "turn on the office lights",
            "office lights on",
            "switch on office lights",
        ),
    ),
    RegisteredCapability(
        record=CapabilityRegistryRecord(
            capability_id="jellyfin_restart",
            display_name="Restart Jellyfin",
            domain="media_operations",
            description="Starts the local Jellyfin restart flow without executing it.",
            operation_kind="restart",
            risk_level="medium_service_interruption",
            requires_confirmation=True,
            allowed_surfaces=["desktop", "dev"],
            allowed_personas=["home_operator", "technical_architect"],
            reversible=False,
            dry_run_supported=True,
            verification_supported=True,
            audit_required=True,
        ),
        match_phrases=("restart jellyfin", "jellyfin restart", "restart media server"),
    ),
    RegisteredCapability(
        record=CapabilityRegistryRecord(
            capability_id="draft_notification",
            display_name="Draft notification",
            domain="communication",
            description="Prepares a notification draft without sending it.",
            operation_kind="draft_or_prepare",
            risk_level="low_prepare_only",
            requires_confirmation=False,
            allowed_surfaces=["desktop", "dev", "telegram"],
            allowed_personas=["default_assistant", "home_operator", "technical_architect"],
            reversible=True,
            dry_run_supported=True,
            verification_supported=False,
            audit_required=False,
        ),
        match_phrases=("draft notification", "prepare notification", "write notification draft"),
    ),
    RegisteredCapability(
        record=CapabilityRegistryRecord(
            capability_id="send_notification",
            display_name="Send notification",
            domain="communication",
            description="Sending notifications is listed for discovery but not executable here.",
            operation_kind="notification",
            risk_level="medium_external_effect",
            requires_confirmation=True,
            allowed_surfaces=[],
            allowed_personas=[],
            reversible=False,
            dry_run_supported=False,
            verification_supported=True,
            audit_required=True,
        ),
        match_phrases=("send notification", "send message", "notify them"),
    ),
    RegisteredCapability(
        record=CapabilityRegistryRecord(
            capability_id="external_purchase",
            display_name="External purchase",
            domain="external_action",
            description="Buying or booking through an external service is blocked in this runtime.",
            operation_kind="blocked_external_action",
            risk_level="blocked_external_effect",
            requires_confirmation=True,
            allowed_surfaces=[],
            allowed_personas=[],
            reversible=False,
            dry_run_supported=False,
            verification_supported=False,
            audit_required=True,
        ),
        match_phrases=("buy", "purchase", "book a", "order"),
    ),
)


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


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _looks_like_raw_capability_name(value: str) -> bool:
    stripped = value.strip()
    if " " in stripped:
        return False
    return any(marker in stripped for marker in ("/", ".", "_", "::"))


def _eligible_reason(
    capability: CapabilityRegistryRecord,
    *,
    surface: str,
    active_persona_id: str,
) -> str | None:
    if surface not in capability.allowed_surfaces:
        return "surface_not_allowed"
    if active_persona_id not in capability.allowed_personas:
        return "persona_not_allowed"
    return None


def _find_registered_capability(text: str) -> RegisteredCapability | None:
    normalized = _normalized_text(text)
    for capability in _CAPABILITY_REGISTRY:
        if any(phrase in normalized for phrase in capability.match_phrases):
            return capability
    return None


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
        world_state_confirmation_required = False

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

        active_persona_id = identity.runtime_identity.active_persona_id

        relationship_reasons = self._relationship_reasons(body, active_persona_id)
        reason_codes.extend(relationship_reasons[0])
        relationship_ids_used = relationship_reasons[1]

        world_reasons = self._world_state_reasons(body, active_persona_id)
        reason_codes.extend(world_reasons[0])
        world_state_claim_ids_used = world_reasons[1]
        revalidation_selector = world_reasons[2]
        world_state_confirmation_required = world_reasons[3]

        challenge_ref: str | None = body.confirmation_challenge_ref
        confirmation_state = "not_required"
        confirmation_needed = (
            body.operation_class in _RISKY_OPERATION_CLASSES
            or world_state_confirmation_required
        )
        challenge_reasons: list[str] = []
        if confirmation_needed:
            if body.authorization_phase == "selection":
                if not body.runtime_turn_id:
                    confirmation_state = "required"
                    reason_codes.append("originating_turn_required")
                elif body.argument_digest and not reason_codes and revalidation_selector is None:
                    challenge_ref = self._issue_challenge(body)
                    confirmation_state = "issued"
                    reason_codes.append("confirmation_required")
                else:
                    confirmation_state = "required"
            elif body.authorization_phase == "dispatch":
                confirmation_state = "required"
            else:
                confirmation_state = "required"

        if revalidation_selector is not None and reason_codes:
            revalidation_selector = None
        if revalidation_selector is not None:
            reason_codes.append("world_state_revalidation_required")

        non_challenge_reasons = [
            code
            for code in reason_codes
            if not code.startswith("challenge_")
            and code not in {"confirmation_required"}
        ]
        if (
            confirmation_needed
            and body.authorization_phase == "dispatch"
            and revalidation_selector is None
            and not non_challenge_reasons
        ):
            challenge_reasons = self._consume_dispatch_challenge_atomic(body)
            reason_codes.extend(challenge_reasons)
            if not challenge_reasons:
                confirmation_state = "accepted"

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
            if row["status"] == "rejected":
                raise RuntimeError("confirmation_challenge_rejected")
            self._validate_current_confirmation_turn(row, body)
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
        active_persona_id: str,
    ) -> tuple[list[str], list[str]]:
        reasons: list[str] = []
        used: list[str] = []
        for requirement in body.relationship_requirements:
            selected = select_relationships(
                RelationshipSelectRequest(
                    request_id=body.request_id,
                    owner_id=body.owner_id,
                    conversation_id=body.conversation_id,
                    surface=body.surface,
                    runtime_session_id=body.runtime_session_id,
                    active_persona_id=active_persona_id,
                    requested_scopes=(
                        [requirement.relationship_scope]
                        if requirement.relationship_scope
                        else []
                    ),
                    relationship_types=(
                        [requirement.relationship_type]
                        if requirement.relationship_type
                        else []
                    ),
                )
            )
            eligible = {edge.relationship_id for edge in selected.selected_relationships}
            requested = set(body.selected_relationship_ids)
            candidates = eligible & requested if requested else eligible
            if not candidates:
                reasons.append(
                    "relationship_not_authorized" if requested else "relationship_required"
                )
                continue
            used.extend(sorted(candidates))
        return sorted(set(reasons)), sorted(set(used))

    def _world_state_reasons(
        self,
        body: CapabilityAuthorizationRequest,
        active_persona_id: str,
    ) -> tuple[list[str], list[str], CapabilityRevalidationSelector | None, bool]:
        requested_domains = sorted(
            {
                requirement.domain
                for requirement in body.world_state_requirements
                if requirement.domain
            }
        )
        resolved = resolve_world_state(
            request_id=body.request_id,
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            surface=body.surface,
            runtime_session_id=body.runtime_session_id,
            active_persona_id=active_persona_id,
            requested_domains=requested_domains,
        )
        claim_map = {claim.world_state_claim_id: claim for claim in resolved.included_claims}
        reasons: list[str] = []
        used: list[str] = []
        revalidation_candidates: dict[str, set[str]] = {}
        confirmation_required = False

        def add_revalidation_candidate(
            *,
            claim_id: str,
            claim: Any,
            revalidator_id: str | None,
            min_authority: str | None = None,
            min_confidence: float | None = None,
            max_freshness_state: str | None = None,
        ) -> None:
            if (
                claim.last_verified_runtime_session_id == body.runtime_session_id
                and claim.last_verified_runtime_turn_id == body.runtime_turn_id
            ):
                reasons.append("world_state_revalidation_inadequate")
                return
            reason = trusted_world_state_verifier_selector_reason(
                verifier_id=revalidator_id,
                claim=claim,
                min_authority=min_authority,
                min_confidence=min_confidence,
                max_freshness_state=max_freshness_state,
            )
            if reason is not None:
                reasons.append(reason)
                return
            assert revalidator_id is not None
            revalidation_candidates.setdefault(revalidator_id, set()).add(claim_id)

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
                if claim.sensitivity in {"high", "restricted"}:
                    reasons.append("world_state_not_authorized")
                    continue
                if claim.effective_freshness_state in {
                    "expired",
                    "superseded",
                    "conflicted",
                }:
                    reasons.append("world_state_not_authorized")
                    continue
                if claim.effective_freshness_state in {"stale", "unknown"}:
                    add_revalidation_candidate(
                        claim_id=claim.world_state_claim_id,
                        claim=claim,
                        revalidator_id=requirement.revalidator_id,
                        max_freshness_state=requirement.max_freshness_state or "aging",
                    )
                    continue
                if requirement.max_freshness_state and (
                    _FRESHNESS_ORDER[claim.effective_freshness_state]
                    > _FRESHNESS_ORDER[requirement.max_freshness_state]
                ):
                    add_revalidation_candidate(
                        claim_id=claim.world_state_claim_id,
                        claim=claim,
                        revalidator_id=requirement.revalidator_id,
                        max_freshness_state=requirement.max_freshness_state,
                    )
                    continue
                if (
                    requirement.min_confidence is not None
                    and claim.confidence < requirement.min_confidence
                ):
                    add_revalidation_candidate(
                        claim_id=claim.world_state_claim_id,
                        claim=claim,
                        revalidator_id=requirement.revalidator_id,
                        min_confidence=requirement.min_confidence,
                    )
                    continue
                if requirement.min_authority and (
                    _AUTHORIZED_AUTHORITIES[claim.state_authority]
                    < _AUTHORIZED_AUTHORITIES[requirement.min_authority]
                ):
                    add_revalidation_candidate(
                        claim_id=claim.world_state_claim_id,
                        claim=claim,
                        revalidator_id=requirement.revalidator_id,
                        min_authority=requirement.min_authority,
                    )
                    continue
                if body.operation_class in _RISKY_OPERATION_CLASSES and claim.state_authority in {
                    "model_inferred",
                    "unverified_assumption",
                    "observed_user_report",
                }:
                    add_revalidation_candidate(
                        claim_id=claim.world_state_claim_id,
                        claim=claim,
                        revalidator_id=requirement.revalidator_id,
                        min_authority="derived_from_multiple_sources",
                    )
                    continue
                if (
                    claim.confirmation_policy == "confirm_before_action"
                    and body.operation_class != "read"
                ):
                    confirmation_required = True
                elif (
                    claim.confirmation_policy == "confirm_before_high_impact_action"
                    and body.operation_class in _HIGH_IMPACT_OPERATION_CLASSES
                ):
                    confirmation_required = True
                elif claim.confirmation_policy == "reverify_before_use":
                    if (
                        claim.last_verified_runtime_session_id != body.runtime_session_id
                        or claim.last_verified_runtime_turn_id != body.runtime_turn_id
                    ):
                        add_revalidation_candidate(
                            claim_id=claim.world_state_claim_id,
                            claim=claim,
                            revalidator_id=requirement.revalidator_id,
                            max_freshness_state=requirement.max_freshness_state or "aging",
                        )
                        continue
                used.append(claim.world_state_claim_id)
        selector = None
        if len(revalidation_candidates) > 1:
            reasons.append("world_state_revalidator_conflict")
        if revalidation_candidates and not reasons:
            revalidator_id, claim_ids = next(iter(revalidation_candidates.items()))
            selector = CapabilityRevalidationSelector(
                world_state_claim_ids=sorted(claim_ids),
                revalidator_id=revalidator_id,
            )
        return sorted(set(reasons)), sorted(set(used)), selector, confirmation_required

    def _issue_challenge(self, body: CapabilityAuthorizationRequest) -> str:
        assert body.argument_digest is not None
        assert body.runtime_turn_id is not None
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

    def _consume_dispatch_challenge_atomic(self, body: CapabilityAuthorizationRequest) -> list[str]:
        if not body.runtime_turn_id:
            return ["dispatch_turn_required"]
        if not body.confirmation_challenge_ref:
            return ["challenge_missing"]
        now = _now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            row = self._challenge_row(conn, body.owner_id, body.confirmation_challenge_ref)
            if row is None:
                conn.rollback()
                return ["challenge_missing"]
            mismatch = self._challenge_base_mismatch(row, body)
            if mismatch:
                conn.rollback()
                return [mismatch]
            if row["consumed_at"]:
                conn.rollback()
                return ["challenge_consumed"]
            if _parse_ts(row["expires_at"]) <= datetime.now(UTC):
                conn.rollback()
                return ["challenge_expired"]
            if row["status"] == "rejected":
                conn.rollback()
                return ["challenge_rejected"]
            if row["status"] != "accepted":
                conn.rollback()
                return ["challenge_not_confirmed"]
            if row["confirmed_runtime_turn_id"] != body.runtime_turn_id:
                conn.rollback()
                return ["challenge_turn_mismatch"]
            turn_current_reason = self._dispatch_turn_current_reason(conn, row, body)
            if turn_current_reason is not None:
                conn.rollback()
                return [turn_current_reason]
            cursor = conn.execute(
                """
                UPDATE capability_confirmation_challenges
                SET consumed_at = ?, updated_at = ?
                WHERE owner_id = ?
                  AND confirmation_challenge_ref = ?
                  AND conversation_id = ?
                  AND runtime_session_id = ?
                  AND capability_id = ?
                  AND operation_class = ?
                  AND argument_digest = ?
                  AND status = 'accepted'
                  AND confirmed_runtime_turn_id = ?
                  AND consumed_at IS NULL
                  AND expires_at > ?;
                """,
                (
                    now,
                    now,
                    body.owner_id,
                    body.confirmation_challenge_ref,
                    body.conversation_id,
                    body.runtime_session_id,
                    body.capability_id,
                    body.operation_class,
                    body.argument_digest,
                    body.runtime_turn_id,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                return ["challenge_consumed"]
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
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
        return []

    def _dispatch_turn_current_reason(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        body: CapabilityAuthorizationRequest,
    ) -> str | None:
        turn = conn.execute(
            """
            SELECT * FROM conversation_runtime_turns
            WHERE runtime_session_id = ? AND runtime_turn_id = ?
            LIMIT 1;
            """,
            (body.runtime_session_id, body.runtime_turn_id),
        ).fetchone()
        if turn is None:
            return "runtime_turn_mismatch"
        if turn["turn_status"] in {"completed", "abandoned"}:
            return "confirmation_turn_not_current"
        if _parse_ts(turn["created_at"]) <= _parse_ts(row["issued_at"]):
            return "confirmation_turn_not_current"
        active_turn = conn.execute(
            """
            SELECT runtime_turn_id FROM conversation_runtime_turns
            WHERE runtime_session_id = ? AND turn_status NOT IN ('completed', 'abandoned')
            ORDER BY id DESC
            LIMIT 1;
            """,
            (body.runtime_session_id,),
        ).fetchone()
        latest_turn = conn.execute(
            """
            SELECT runtime_turn_id FROM conversation_runtime_turns
            WHERE runtime_session_id = ?
            ORDER BY id DESC
            LIMIT 1;
            """,
            (body.runtime_session_id,),
        ).fetchone()
        if active_turn is None or active_turn["runtime_turn_id"] != body.runtime_turn_id:
            return "confirmation_turn_not_current"
        if latest_turn is None or latest_turn["runtime_turn_id"] != body.runtime_turn_id:
            return "confirmation_turn_not_current"
        return None

    def _validate_current_confirmation_turn(
        self,
        row: sqlite3.Row,
        body: CapabilityConfirmationRequest,
    ) -> None:
        if body.runtime_turn_id == row["originating_runtime_turn_id"]:
            raise RuntimeError("confirmation_turn_not_distinct")
        turn = validate_runtime_turn_session(
            runtime_session_id=body.runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
        )
        if _parse_ts(turn.created_at) <= _parse_ts(row["issued_at"]):
            raise RuntimeError("confirmation_turn_not_current")
        diagnostics = get_runtime_session(body.runtime_session_id)
        active_turn = diagnostics.active_turn
        latest_turn = diagnostics.latest_turn
        if active_turn is None or active_turn.runtime_turn_id != body.runtime_turn_id:
            raise RuntimeError("confirmation_turn_not_current")
        if latest_turn is None or latest_turn.runtime_turn_id != body.runtime_turn_id:
            raise RuntimeError("confirmation_turn_not_current")

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


def match_registered_capability(body: CapabilityMatchRequest) -> CapabilityMatchResponse:
    if not _CAPABILITY_REGISTRY_AVAILABLE or not body.registry_enabled:
        result = CapabilityMatchResult(
            capability_matched=False,
            reason_codes=["registry_unavailable"],
        )
    elif _looks_like_raw_capability_name(body.current_user_text):
        result = CapabilityMatchResult(
            capability_matched=False,
            reason_codes=["raw_capability_name_ignored"],
        )
    else:
        capability = _find_registered_capability(body.current_user_text)
        if capability is None:
            result = CapabilityMatchResult(
                capability_matched=False,
                reason_codes=["no_registered_capability"],
            )
        else:
            ineligible_reason = _eligible_reason(
                capability.record,
                surface=body.surface,
                active_persona_id=body.active_persona_id,
            )
            result = CapabilityMatchResult(
                capability_matched=ineligible_reason is None,
                reason_codes=[ineligible_reason or "matched"],
                capability=capability.record,
            )
    return CapabilityMatchResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        active_persona_id=body.active_persona_id,
        result=result,
    )


def discover_registered_capabilities(body: CapabilityDiscoveryRequest) -> CapabilityDiscoveryResponse:
    if not _CAPABILITY_REGISTRY_AVAILABLE or not body.registry_enabled:
        result = CapabilityDiscoveryResult(registry_available=False)
    else:
        allowed_examples: list[CapabilityDiscoveryExample] = []
        blocked_examples: list[CapabilityDiscoveryExample] = []
        for capability in _CAPABILITY_REGISTRY:
            ineligible_reason = _eligible_reason(
                capability.record,
                surface=body.surface,
                active_persona_id=body.active_persona_id,
            )
            example = CapabilityDiscoveryExample(
                capability_id=capability.record.capability_id,
                display_name=capability.record.display_name,
                description=capability.record.description,
                operation_kind=capability.record.operation_kind,
                risk_level=capability.record.risk_level,
                reason_codes=[ineligible_reason or "matched"],
            )
            if ineligible_reason is None:
                allowed_examples.append(example)
            else:
                blocked_examples.append(example)
        result = CapabilityDiscoveryResult(
            registry_available=True,
            allowed_examples=allowed_examples,
            blocked_examples=blocked_examples,
        )
    return CapabilityDiscoveryResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        active_persona_id=body.active_persona_id,
        result=result,
    )


def authorize_capability(body: CapabilityAuthorizationRequest) -> CapabilityAuthorizationResponse:
    return capability_authorization_repository().authorize(body)


def record_capability_confirmation(
    body: CapabilityConfirmationRequest,
) -> CapabilityConfirmationResponse:
    return capability_authorization_repository().confirm(body)


def clear_capability_authorization_for_tests(db_path: Path | None = None) -> None:
    global _CAPABILITY_AUTH_REPOSITORY
    _CAPABILITY_AUTH_REPOSITORY = CapabilityAuthorizationRepository(db_path=db_path)


def configure_capability_registry_for_tests(*, available: bool = True) -> None:
    global _CAPABILITY_REGISTRY_AVAILABLE
    _CAPABILITY_REGISTRY_AVAILABLE = available
