from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from models import (
    RuntimeIdentityResolveRequest,
    WorldStateClaimInput,
    WorldStateClaimSummary,
    WorldStateClaimVerifyRequest,
    WorldStateClaimView,
    WorldStateDiagnosticsResponse,
    WorldStateFreshnessState,
    WorldStateResolveResponse,
    WorldStateResolveTrace,
    WorldStateTransition,
)
from services.companion_contracts import companion_contracts_repository
from services.runtime_identity import resolve_runtime_identity
from services.runtime_state import (
    record_runtime_event,
    runtime_session_by_id,
    runtime_state_db_path,
    validate_runtime_turn_session,
)

_WORLD_STATE_REPOSITORY: WorldStateRepository | None = None
_TERMINAL_FRESHNESS_STATES = {"expired", "superseded"}
_SENSITIVE_LEVELS = {"high", "restricted"}
_VERIFIED_AUTHORITIES = {
    "verified_tool_output",
    "trusted_integration_event",
    "derived_from_multiple_sources",
}
_AUTHORITY_ORDER = {
    "unverified_assumption": 0,
    "model_inferred": 0,
    "observed_user_report": 1,
    "derived_from_multiple_sources": 2,
    "trusted_integration_event": 3,
    "verified_tool_output": 4,
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
_VERIFIER_CONFIG_ENV = "TRUSTED_WORLD_STATE_VERIFIERS_PATH"
_TRUSTED_VERIFICATION_SOURCE_TYPES = {
    "tool_output",
    "integration_event",
    "sensor_update",
    "repository_inspection",
    "calendar_event",
    "automation_workflow",
}
_ACTIVE_FRESHNESS_STATES = {"fresh", "aging", "stale", "unknown"}
_VERIFIER_CONFIG_KEYS = {
    "verifier_id",
    "verification_source_type",
    "allowed_source_refs",
    "max_authority",
    "allowed_domains",
    "allowed_attributes",
    "allowed_entity_ids",
    "max_confidence",
    "max_ttl_seconds",
    "max_revalidation_interval_seconds",
    "max_freshness_state",
}
_MATERIAL_FUTURE_SKEW_SECONDS = 5


@dataclass(frozen=True)
class TrustedWorldStateVerifier:
    verifier_id: str
    verification_source_type: str
    allowed_source_refs: frozenset[str] = field(default_factory=frozenset)
    max_authority: str = "derived_from_multiple_sources"
    allowed_domains: frozenset[str] = field(default_factory=frozenset)
    allowed_attributes: frozenset[str] = field(default_factory=frozenset)
    allowed_entity_ids: frozenset[str] = field(default_factory=frozenset)
    max_confidence: float = 1.0
    max_freshness_state: str = "fresh"
    max_ttl_seconds: int = 300
    max_revalidation_interval_seconds: int = 300


_TRUSTED_VERIFIER_TEST_OVERRIDE: dict[str, TrustedWorldStateVerifier] | None = None
_TRUSTED_VERIFIER_CACHE_PATH: str | None = None
_TRUSTED_VERIFIER_CACHE: dict[str, TrustedWorldStateVerifier] | None = None
_DOMAIN_ALLOWLISTS: dict[str, set[str]] = {
    "general_assistant": {"active_task", "active_project", "pending_action", "runtime_surface"},
    "technical_architect": {
        "active_project",
        "active_repository",
        "active_artifact",
        "active_tool_session",
        "active_external_system",
        "pending_action",
        "runtime_surface",
    },
    "personal_companion": {
        "active_travel_window",
        "active_device_context",
        "active_health_observation",
        "pending_action",
        "runtime_surface",
    },
    "operations_assistant": {
        "active_external_system",
        "active_repository",
        "active_tool_session",
        "active_project",
        "pending_action",
        "runtime_surface",
    },
}
_SURFACE_DOMAIN_RESTRICTIONS: dict[str, set[str]] = {
    "developer_surface": {
        "active_project",
        "active_repository",
        "active_artifact",
        "active_tool_session",
        "active_external_system",
        "pending_action",
        "runtime_surface",
    },
    "ide_extension": {
        "active_project",
        "active_repository",
        "active_artifact",
        "active_tool_session",
        "active_external_system",
        "pending_action",
        "runtime_surface",
    },
    "web_app": {
        "active_task",
        "active_project",
        "pending_action",
        "runtime_surface",
    },
    "unknown_surface": {"active_task", "pending_action", "runtime_surface"},
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


def _parse_aware_ts(value: str | None, *, reason: str) -> datetime:
    if not value:
        raise RuntimeError(reason)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(reason) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError(reason)
    return parsed.astimezone(UTC)


def _finite_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _seconds_between(now: datetime, value: str | None) -> float | None:
    parsed = _parse_ts(value)
    if parsed is None:
        return None
    return max((now - parsed).total_seconds(), 0.0)


def _material_value(value: Any) -> str:
    return _json(value)


def _value_digest_from_material(material: str) -> str:
    return f"wsvalue_{sha256(material.encode('utf-8')).hexdigest()}"


def _record_verification_decision(
    body: WorldStateClaimVerifyRequest,
    *,
    decision: str,
    reason: str | None = None,
    accepted_at: str | None = None,
) -> None:
    if not body.runtime_session_id:
        return
    payload: dict[str, Any] = {
        "request_id": body.request_id,
        "world_state_claim_id": body.world_state_claim_id,
        "decision": decision,
        "verifier_id": body.verifier_id,
        "verification_source_type": body.verification_source_type,
    }
    if reason is not None:
        payload["reason"] = reason
    if decision == "accepted":
        payload.update(
            {
                "verified_at": accepted_at or body.verified_at,
                "resulting_authority": body.resulting_authority,
                "resulting_freshness_state": body.resulting_freshness_state,
            }
        )
    record_runtime_event(
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        event_type="world_state_verification_evaluated",
        event_payload_json=payload,
    )


def configure_trusted_world_state_verifiers_for_tests(
    verifiers: list[TrustedWorldStateVerifier],
) -> None:
    global _TRUSTED_VERIFIER_TEST_OVERRIDE
    _TRUSTED_VERIFIER_TEST_OVERRIDE = {
        verifier.verifier_id: verifier for verifier in verifiers
    }


def clear_trusted_world_state_verifiers_for_tests() -> None:
    global _TRUSTED_VERIFIER_TEST_OVERRIDE, _TRUSTED_VERIFIER_CACHE_PATH, _TRUSTED_VERIFIER_CACHE
    _TRUSTED_VERIFIER_TEST_OVERRIDE = None
    _TRUSTED_VERIFIER_CACHE_PATH = None
    _TRUSTED_VERIFIER_CACHE = None


def _as_non_empty_string_set(value: Any) -> frozenset[str] | None:
    if not isinstance(value, list) or not value:
        return None
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            return None
        items.append(item)
    return frozenset(items)


def _as_optional_string_set(value: Any) -> frozenset[str] | None:
    if value is None:
        return frozenset()
    if not isinstance(value, list):
        return None
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            return None
        items.append(item)
    return frozenset(items)


def _load_trusted_world_state_verifiers(path: Path) -> dict[str, TrustedWorldStateVerifier]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except OSError as exc:
        raise RuntimeError("trusted_verifier_registry_invalid") from exc
    except yaml.YAMLError as exc:
        raise RuntimeError("trusted_verifier_registry_invalid") from exc

    if not isinstance(loaded, dict) or set(loaded) != {"verifiers"}:
        raise RuntimeError("trusted_verifier_registry_invalid")
    entries = loaded.get("verifiers")
    if not isinstance(entries, list):
        raise RuntimeError("trusted_verifier_registry_invalid")

    verifiers: dict[str, TrustedWorldStateVerifier] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - _VERIFIER_CONFIG_KEYS:
            raise RuntimeError("trusted_verifier_registry_invalid")
        required = {
            "verifier_id",
            "verification_source_type",
            "allowed_source_refs",
            "max_authority",
            "allowed_domains",
            "max_confidence",
            "max_ttl_seconds",
            "max_revalidation_interval_seconds",
            "max_freshness_state",
        }
        if not required <= set(entry):
            raise RuntimeError("trusted_verifier_registry_invalid")
        verifier_id = entry["verifier_id"]
        source_type = entry["verification_source_type"]
        max_authority = entry["max_authority"]
        max_freshness_state = entry["max_freshness_state"]
        if (
            not isinstance(verifier_id, str)
            or not verifier_id
            or verifier_id in verifiers
            or source_type not in _TRUSTED_VERIFICATION_SOURCE_TYPES
            or max_authority not in _VERIFIED_AUTHORITIES
            or max_freshness_state not in _ACTIVE_FRESHNESS_STATES
        ):
            raise RuntimeError("trusted_verifier_registry_invalid")
        allowed_source_refs = _as_non_empty_string_set(entry["allowed_source_refs"])
        allowed_domains = _as_non_empty_string_set(entry["allowed_domains"])
        allowed_attributes = _as_optional_string_set(entry.get("allowed_attributes"))
        allowed_entity_ids = _as_optional_string_set(entry.get("allowed_entity_ids"))
        max_ttl_seconds = _finite_positive_int(entry["max_ttl_seconds"])
        max_revalidation_interval_seconds = _finite_positive_int(
            entry["max_revalidation_interval_seconds"]
        )
        max_confidence = entry["max_confidence"]
        if (
            allowed_source_refs is None
            or allowed_domains is None
            or allowed_attributes is None
            or allowed_entity_ids is None
            or max_ttl_seconds is None
            or max_revalidation_interval_seconds is None
            or isinstance(max_confidence, bool)
            or not isinstance(max_confidence, int | float)
            or max_confidence < 0.0
            or max_confidence > 1.0
        ):
            raise RuntimeError("trusted_verifier_registry_invalid")
        verifiers[verifier_id] = TrustedWorldStateVerifier(
            verifier_id=verifier_id,
            verification_source_type=source_type,
            allowed_source_refs=allowed_source_refs,
            max_authority=max_authority,
            allowed_domains=allowed_domains,
            allowed_attributes=allowed_attributes,
            allowed_entity_ids=allowed_entity_ids,
            max_confidence=float(max_confidence),
            max_freshness_state=max_freshness_state,
            max_ttl_seconds=max_ttl_seconds,
            max_revalidation_interval_seconds=max_revalidation_interval_seconds,
        )
    return verifiers


def trusted_world_state_verifiers() -> dict[str, TrustedWorldStateVerifier]:
    global _TRUSTED_VERIFIER_CACHE_PATH, _TRUSTED_VERIFIER_CACHE
    if _TRUSTED_VERIFIER_TEST_OVERRIDE is not None:
        return dict(_TRUSTED_VERIFIER_TEST_OVERRIDE)
    config_path = os.environ.get(_VERIFIER_CONFIG_ENV)
    if not config_path:
        return {}
    if _TRUSTED_VERIFIER_CACHE_PATH == config_path and _TRUSTED_VERIFIER_CACHE is not None:
        return dict(_TRUSTED_VERIFIER_CACHE)
    verifiers = _load_trusted_world_state_verifiers(Path(config_path))
    _TRUSTED_VERIFIER_CACHE_PATH = config_path
    _TRUSTED_VERIFIER_CACHE = verifiers
    return dict(verifiers)


def trusted_world_state_verifier(verifier_id: str | None) -> TrustedWorldStateVerifier | None:
    if verifier_id is None:
        return None
    return trusted_world_state_verifiers().get(verifier_id)


def trusted_world_state_verifier_selector_reason(
    *,
    verifier_id: str | None,
    claim: WorldStateClaimView,
    min_authority: str | None = None,
    min_confidence: float | None = None,
    max_freshness_state: str | None = None,
) -> str | None:
    verifier = trusted_world_state_verifier(verifier_id)
    if verifier_id is None:
        return "world_state_revalidator_required"
    if verifier is None:
        return "world_state_revalidator_not_configured"
    if verifier.allowed_domains and claim.domain not in verifier.allowed_domains:
        return "world_state_revalidator_not_authorized"
    if verifier.allowed_attributes and claim.attribute not in verifier.allowed_attributes:
        return "world_state_revalidator_not_authorized"
    if verifier.allowed_entity_ids and claim.entity_id not in verifier.allowed_entity_ids:
        return "world_state_revalidator_not_authorized"
    if min_authority and _AUTHORITY_ORDER[verifier.max_authority] < _AUTHORITY_ORDER[min_authority]:
        return "world_state_revalidation_inadequate"
    if min_confidence is not None and verifier.max_confidence < min_confidence:
        return "world_state_revalidation_inadequate"
    if (
        max_freshness_state is not None
        and _FRESHNESS_ORDER[verifier.max_freshness_state]
        > _FRESHNESS_ORDER[max_freshness_state]
    ):
        return "world_state_revalidation_inadequate"
    return None


def _verification_policy_reason(
    *,
    verifier: TrustedWorldStateVerifier | None,
    body: WorldStateClaimVerifyRequest,
    row: sqlite3.Row | None = None,
) -> str | None:
    if body.verifier_id is None:
        return "trusted_verifier_required"
    if verifier is None:
        return "unknown_trusted_verifier"
    if body.verification_source_type != verifier.verification_source_type:
        return "verification_source_mismatch"
    if body.verification_source_ref not in verifier.allowed_source_refs:
        return "verification_source_ref_not_allowed"
    if body.resulting_authority not in _VERIFIED_AUTHORITIES:
        return "invalid_verification_authority"
    if _AUTHORITY_ORDER[body.resulting_authority] > _AUTHORITY_ORDER[verifier.max_authority]:
        return "verification_authority_escalation"
    if body.resulting_confidence > verifier.max_confidence:
        return "verification_confidence_escalation"
    if body.resulting_freshness_state in {"expired", "superseded", "conflicted"}:
        return "invalid_verification_freshness"
    if (
        _FRESHNESS_ORDER[body.resulting_freshness_state]
        < _FRESHNESS_ORDER[verifier.max_freshness_state]
    ):
        return "verification_freshness_escalation"
    if body.resulting_ttl_seconds is not None and (
        body.resulting_ttl_seconds <= 0 or body.resulting_ttl_seconds > verifier.max_ttl_seconds
    ):
        return "verification_ttl_escalation"
    if body.resulting_revalidation_interval_seconds is not None and (
        body.resulting_revalidation_interval_seconds <= 0
        or body.resulting_revalidation_interval_seconds
        > verifier.max_revalidation_interval_seconds
    ):
        return "verification_revalidation_interval_escalation"
    if row is not None:
        if verifier.allowed_domains and row["domain"] not in verifier.allowed_domains:
            return "verification_domain_not_allowed"
        if verifier.allowed_attributes and row["attribute"] not in verifier.allowed_attributes:
            return "verification_selector_not_allowed"
        if verifier.allowed_entity_ids and row["entity_id"] not in verifier.allowed_entity_ids:
            return "verification_selector_not_allowed"
    return None


def _computed_freshness_from_age(
    *,
    age_seconds: float,
    ttl_seconds: int,
    revalidation_interval_seconds: int,
) -> WorldStateFreshnessState:
    if age_seconds >= ttl_seconds:
        return "expired"
    if age_seconds >= revalidation_interval_seconds:
        return "stale"
    aging_thresholds = [ttl_seconds * 0.75, revalidation_interval_seconds * 0.75]
    if age_seconds >= min(aging_thresholds):
        return "aging"
    return "fresh"


def _bounded_verification_values(
    *,
    verifier: TrustedWorldStateVerifier,
    body: WorldStateClaimVerifyRequest,
    accepted_at: datetime,
) -> tuple[str, str, int, int]:
    observed_at = _parse_aware_ts(body.observed_at, reason="verification_timestamp_invalid")
    if body.verified_at is not None:
        _parse_aware_ts(body.verified_at, reason="verification_timestamp_invalid")
    future_limit = accepted_at + timedelta(seconds=_MATERIAL_FUTURE_SKEW_SECONDS)
    if observed_at > future_limit:
        raise RuntimeError("verification_timestamp_in_future")
    if body.verified_at is not None:
        verified_at = _parse_aware_ts(body.verified_at, reason="verification_timestamp_invalid")
        if verified_at > future_limit:
            raise RuntimeError("verification_timestamp_in_future")

    effective_ttl_seconds = body.resulting_ttl_seconds or verifier.max_ttl_seconds
    effective_revalidation_interval_seconds = (
        body.resulting_revalidation_interval_seconds
        or verifier.max_revalidation_interval_seconds
    )
    if effective_ttl_seconds > verifier.max_ttl_seconds:
        raise RuntimeError("verification_ttl_escalation")
    if effective_revalidation_interval_seconds > verifier.max_revalidation_interval_seconds:
        raise RuntimeError("verification_revalidation_interval_escalation")

    age_seconds = max((accepted_at - observed_at).total_seconds(), 0.0)
    computed_freshness = _computed_freshness_from_age(
        age_seconds=age_seconds,
        ttl_seconds=effective_ttl_seconds,
        revalidation_interval_seconds=effective_revalidation_interval_seconds,
    )
    if computed_freshness in {"expired", "superseded", "conflicted"}:
        raise RuntimeError("invalid_verification_freshness")
    if _FRESHNESS_ORDER[body.resulting_freshness_state] < _FRESHNESS_ORDER[computed_freshness]:
        raise RuntimeError("verification_freshness_escalation")

    max_expires_at = observed_at + timedelta(seconds=effective_ttl_seconds)
    if body.resulting_expires_at:
        expires_at = _parse_aware_ts(
            body.resulting_expires_at,
            reason="verification_timestamp_invalid",
        )
        if expires_at > max_expires_at:
            raise RuntimeError("verification_expiry_escalation")
    else:
        expires_at = max_expires_at
    return (
        expires_at.isoformat(),
        observed_at.isoformat(),
        effective_ttl_seconds,
        effective_revalidation_interval_seconds,
    )


class WorldStateRepository:
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
                CREATE TABLE IF NOT EXISTS runtime_world_state_claims (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    world_state_claim_id TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    attribute TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    material_value_json TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    verification_verifier_id TEXT,
                    verification_source_type TEXT,
                    verification_source_ref TEXT,
                    confidence REAL NOT NULL,
                    freshness_state TEXT NOT NULL,
                    state_authority TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    last_verified_at TEXT,
                    last_verified_runtime_session_id TEXT,
                    last_verified_runtime_turn_id TEXT,
                    last_verification_request_id TEXT,
                    expires_at TEXT,
                    ttl_seconds INTEGER,
                    revalidation_interval_seconds INTEGER,
                    confirmation_policy TEXT NOT NULL,
                    sensitivity TEXT NOT NULL,
                    scope_labels_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    superseded_by_claim_id TEXT,
                    UNIQUE(owner_id, world_state_claim_id)
                );

                CREATE TABLE IF NOT EXISTS runtime_world_state_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transition_id TEXT NOT NULL UNIQUE,
                    world_state_claim_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    transition_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._ensure_claim_columns(conn)

    def _ensure_claim_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(runtime_world_state_claims);").fetchall()
        }
        if "verification_source_type" not in existing:
            conn.execute(
                "ALTER TABLE runtime_world_state_claims ADD COLUMN verification_source_type TEXT;"
            )
        if "verification_verifier_id" not in existing:
            conn.execute(
                "ALTER TABLE runtime_world_state_claims ADD COLUMN verification_verifier_id TEXT;"
            )
        if "verification_source_ref" not in existing:
            conn.execute(
                "ALTER TABLE runtime_world_state_claims ADD COLUMN verification_source_ref TEXT;"
            )
        if "last_verified_runtime_session_id" not in existing:
            conn.execute(
                "ALTER TABLE runtime_world_state_claims "
                "ADD COLUMN last_verified_runtime_session_id TEXT;"
            )
        if "last_verified_runtime_turn_id" not in existing:
            conn.execute(
                "ALTER TABLE runtime_world_state_claims "
                "ADD COLUMN last_verified_runtime_turn_id TEXT;"
            )
        if "last_verification_request_id" not in existing:
            conn.execute(
                "ALTER TABLE runtime_world_state_claims "
                "ADD COLUMN last_verification_request_id TEXT;"
            )

    def upsert_claim(
        self,
        *,
        owner_id: str,
        claim: WorldStateClaimInput,
    ) -> tuple[WorldStateClaimView, list[WorldStateTransition]]:
        now = _now()
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT * FROM runtime_world_state_claims
                WHERE owner_id = ? AND entity_id = ? AND attribute = ? AND material_value_json = ?
                  AND superseded_by_claim_id IS NULL
                ORDER BY updated_at DESC
                LIMIT 1;
                """,
                (
                    owner_id,
                    claim.entity_id,
                    claim.attribute,
                    _material_value(claim.value_json),
                ),
            ).fetchone()
            transitions: list[WorldStateTransition] = []
            if existing is not None:
                conn.execute(
                    """
                    UPDATE runtime_world_state_claims
                    SET entity_type = ?, domain = ?, value_json = ?,
                        source_type = ?, source_ref = ?,
                        verification_verifier_id = NULL, verification_source_type = NULL,
                        verification_source_ref = NULL,
                        confidence = ?, freshness_state = ?, state_authority = ?, observed_at = ?,
                        last_verified_at = ?, last_verified_runtime_session_id = NULL,
                        last_verified_runtime_turn_id = NULL,
                        last_verification_request_id = NULL,
                        expires_at = ?, ttl_seconds = ?,
                        revalidation_interval_seconds = ?, confirmation_policy = ?,
                        sensitivity = ?, scope_labels_json = ?, updated_at = ?
                    WHERE world_state_claim_id = ?;
                    """,
                    (
                        claim.entity_type,
                        claim.domain,
                        _json(claim.value_json),
                        claim.source_type,
                        claim.source_ref,
                        claim.confidence,
                        claim.freshness_state,
                        claim.state_authority,
                        claim.observed_at,
                        claim.last_verified_at,
                        claim.expires_at,
                        claim.ttl_seconds,
                        claim.revalidation_interval_seconds,
                        claim.confirmation_policy,
                        claim.sensitivity,
                        _json(claim.scope_labels),
                        now,
                        existing["world_state_claim_id"],
                    ),
                )
                transitions.append(
                    self._record_transition(
                        conn,
                        owner_id=owner_id,
                        claim_id=existing["world_state_claim_id"],
                        transition_type="updated",
                        metadata_json={"source_ref": claim.source_ref},
                    )
                )
                claim_id = existing["world_state_claim_id"]
            else:
                claim_id = _digest(
                    "wsclaim",
                    f"{owner_id}:{claim.entity_id}:{claim.attribute}:{_material_value(claim.value_json)}:{now}",
                )
                conn.execute(
                    """
                    INSERT INTO runtime_world_state_claims (
                        world_state_claim_id, owner_id, entity_id, entity_type, domain, attribute,
                        value_json, material_value_json, source_type, source_ref,
                        verification_verifier_id, verification_source_type,
                        verification_source_ref, confidence,
                        freshness_state, state_authority, observed_at, last_verified_at,
                        last_verified_runtime_session_id, last_verified_runtime_turn_id,
                        last_verification_request_id, expires_at, ttl_seconds,
                        revalidation_interval_seconds,
                        confirmation_policy, sensitivity, scope_labels_json,
                        created_at, updated_at, superseded_by_claim_id
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    );
                    """,
                    (
                        claim_id,
                        owner_id,
                        claim.entity_id,
                        claim.entity_type,
                        claim.domain,
                        claim.attribute,
                        _json(claim.value_json),
                        _material_value(claim.value_json),
                        claim.source_type,
                        claim.source_ref,
                        None,
                        None,
                        None,
                        claim.confidence,
                        claim.freshness_state,
                        claim.state_authority,
                        claim.observed_at,
                        claim.last_verified_at,
                        None,
                        None,
                        None,
                        claim.expires_at,
                        claim.ttl_seconds,
                        claim.revalidation_interval_seconds,
                        claim.confirmation_policy,
                        claim.sensitivity,
                        _json(claim.scope_labels),
                        now,
                        now,
                        None,
                    ),
                )
                transitions.append(
                    self._record_transition(
                        conn,
                        owner_id=owner_id,
                        claim_id=claim_id,
                        transition_type="created",
                        metadata_json={"source_ref": claim.source_ref},
                    )
                )

            if claim.supersede_existing_claim_id:
                conn.execute(
                    """
                    UPDATE runtime_world_state_claims
                    SET superseded_by_claim_id = ?, updated_at = ?
                    WHERE owner_id = ? AND world_state_claim_id = ?;
                    """,
                    (claim_id, now, owner_id, claim.supersede_existing_claim_id),
                )
                transitions.append(
                    self._record_transition(
                        conn,
                        owner_id=owner_id,
                        claim_id=claim.supersede_existing_claim_id,
                        transition_type="superseded",
                        metadata_json={"superseded_by_claim_id": claim_id},
                    )
                )

            transitions.extend(
                self._sync_conflict_transitions(
                    conn,
                    owner_id=owner_id,
                    entity_id=claim.entity_id,
                    attribute=claim.attribute,
                )
            )
            row = conn.execute(
                """
                SELECT * FROM runtime_world_state_claims
                WHERE owner_id = ? AND world_state_claim_id = ?
                LIMIT 1;
                """,
                (owner_id, claim_id),
            ).fetchone()
            assert row is not None
            claim_view = self._claim_view_from_row(
                row,
                include_sensitive_values=False,
                now=datetime.now(UTC),
                conflict_map=self._conflict_map(conn, owner_id),
            )
            return claim_view, transitions

    def verify_claim(
        self,
        body: WorldStateClaimVerifyRequest,
    ) -> tuple[WorldStateClaimView, list[WorldStateTransition]]:
        session = (
            runtime_session_by_id(body.runtime_session_id)
            if body.runtime_session_id
            else None
        )
        if body.runtime_session_id and session is None:
            raise RuntimeError("runtime_session_not_found")
        if session is not None and (
            session.owner_id != body.owner_id
            or session.conversation_id != body.conversation_id
            or session.surface != body.surface
        ):
            raise RuntimeError("runtime_session_mismatch")
        if body.runtime_session_id and body.runtime_turn_id:
            validate_runtime_turn_session(
                runtime_session_id=body.runtime_session_id,
                runtime_turn_id=body.runtime_turn_id,
            )
        verifier = trusted_world_state_verifier(body.verifier_id)
        policy_reason = _verification_policy_reason(verifier=verifier, body=body)
        if policy_reason is not None:
            _record_verification_decision(body, decision="rejected", reason=policy_reason)
            raise RuntimeError(policy_reason)
        assert verifier is not None
        accepted_at = datetime.now(UTC)
        (
            effective_expires_at,
            effective_observed_at,
            effective_ttl_seconds,
            effective_revalidation_interval_seconds,
        ) = _bounded_verification_values(
            verifier=verifier,
            body=body,
            accepted_at=accepted_at,
        )

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM runtime_world_state_claims
                WHERE owner_id = ? AND world_state_claim_id = ?
                LIMIT 1;
                """,
                (body.owner_id, body.world_state_claim_id),
            ).fetchone()
            if row is None:
                _record_verification_decision(body, decision="rejected", reason="claim_not_found")
                raise RuntimeError("world_state_claim_not_found")
            policy_reason = _verification_policy_reason(verifier=verifier, body=body, row=row)
            if policy_reason is not None:
                _record_verification_decision(body, decision="rejected", reason=policy_reason)
                raise RuntimeError(policy_reason)
            conflict_map = self._conflict_map(conn, body.owner_id)
            effective = self._effective_freshness_state(
                row,
                now=datetime.now(UTC),
                conflict_ids=conflict_map.get(body.world_state_claim_id, []),
            )
            if effective == "superseded":
                _record_verification_decision(body, decision="rejected", reason="superseded")
                raise RuntimeError("world_state_claim_superseded")
            if effective == "expired":
                _record_verification_decision(body, decision="rejected", reason="expired")
                raise RuntimeError("world_state_claim_expired")
            if effective == "conflicted":
                _record_verification_decision(body, decision="rejected", reason="conflicted")
                raise RuntimeError("world_state_claim_conflicted")
            if (
                _value_digest_from_material(row["material_value_json"])
                != body.expected_value_digest
            ):
                _record_verification_decision(
                    body,
                    decision="rejected",
                    reason="expected_value_mismatch",
                )
                raise RuntimeError("expected_value_mismatch")

            now = accepted_at.isoformat()
            conn.execute(
                """
                UPDATE runtime_world_state_claims
                SET verification_verifier_id = ?, verification_source_type = ?,
                    verification_source_ref = ?,
                    confidence = ?, freshness_state = ?, state_authority = ?,
                    observed_at = ?, last_verified_at = ?,
                    last_verified_runtime_session_id = ?,
                    last_verified_runtime_turn_id = ?, last_verification_request_id = ?,
                    expires_at = ?,
                    ttl_seconds = ?, revalidation_interval_seconds = ?, updated_at = ?
                WHERE owner_id = ? AND world_state_claim_id = ?;
                """,
                (
                    body.verifier_id,
                    body.verification_source_type,
                    body.verification_source_ref,
                    body.resulting_confidence,
                    body.resulting_freshness_state,
                    body.resulting_authority,
                    effective_observed_at,
                    now,
                    body.runtime_session_id,
                    body.runtime_turn_id,
                    body.request_id,
                    effective_expires_at,
                    effective_ttl_seconds,
                    effective_revalidation_interval_seconds,
                    now,
                    body.owner_id,
                    body.world_state_claim_id,
                ),
            )
            transition = self._record_transition(
                conn,
                owner_id=body.owner_id,
                claim_id=body.world_state_claim_id,
                transition_type="verified",
                metadata_json={
                    "verification_source_type": body.verification_source_type,
                    "verification_verifier_id": body.verifier_id,
                    "verification_source_ref": body.verification_source_ref,
                    "verified_at": now,
                    "observed_at": effective_observed_at,
                    "resulting_authority": body.resulting_authority,
                    "resulting_freshness_state": body.resulting_freshness_state,
                },
            )
            updated = conn.execute(
                """
                SELECT * FROM runtime_world_state_claims
                WHERE owner_id = ? AND world_state_claim_id = ?
                LIMIT 1;
                """,
                (body.owner_id, body.world_state_claim_id),
            ).fetchone()
            assert updated is not None
            claim_view = self._claim_view_from_row(
                updated,
                include_sensitive_values=False,
                now=datetime.now(UTC),
                conflict_map=self._conflict_map(conn, body.owner_id),
            )

        _record_verification_decision(body, decision="accepted", accepted_at=now)
        return claim_view, [transition]

    def diagnostics(
        self,
        *,
        owner_id: str,
        include_sensitive_values: bool = False,
    ) -> WorldStateDiagnosticsResponse:
        del include_sensitive_values
        now = datetime.now(UTC)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM runtime_world_state_claims
                WHERE owner_id = ?
                ORDER BY updated_at DESC, world_state_claim_id ASC;
                """,
                (owner_id,),
            ).fetchall()
            transition_rows = conn.execute(
                """
                SELECT * FROM runtime_world_state_transitions
                WHERE owner_id = ?
                ORDER BY created_at ASC, id ASC;
                """,
                (owner_id,),
            ).fetchall()
            conflict_map = self._conflict_map(conn, owner_id)

        claims: list[WorldStateClaimView] = []
        excluded: list[WorldStateClaimSummary] = []
        for row in rows:
            view = self._claim_view_from_row(
                row,
                include_sensitive_values=False,
                now=now,
                conflict_map=conflict_map,
            )
            if view.effective_freshness_state in {"fresh", "aging", "stale", "unknown"}:
                claims.append(view)
            else:
                excluded.append(
                    WorldStateClaimSummary(
                        world_state_claim_id=view.world_state_claim_id,
                        entity_id=view.entity_id,
                        attribute=view.attribute,
                        domain=view.domain,
                        freshness_state=view.freshness_state,
                        effective_freshness_state=view.effective_freshness_state,
                        sensitivity=view.sensitivity,
                        reason=view.effective_freshness_state,
                        superseded_by_claim_id=view.superseded_by_claim_id,
                        conflict_claim_ids=view.conflict_claim_ids,
                    )
                )
        return WorldStateDiagnosticsResponse(
            claims=claims,
            excluded_claims=excluded,
            transitions=[self._transition_from_row(row) for row in transition_rows],
        )

    def _record_transition(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        claim_id: str,
        transition_type: str,
        metadata_json: dict[str, Any],
    ) -> WorldStateTransition:
        created_at = _now()
        transition_id = _digest(
            "wstransition",
            f"{owner_id}:{claim_id}:{transition_type}:{created_at}:{_json(metadata_json)}",
        )
        conn.execute(
            """
            INSERT INTO runtime_world_state_transitions (
                transition_id, world_state_claim_id, owner_id, transition_type,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            (transition_id, claim_id, owner_id, transition_type, _json(metadata_json), created_at),
        )
        return WorldStateTransition(
            transition_id=transition_id,
            world_state_claim_id=claim_id,
            transition_type=transition_type,
            created_at=created_at,
            metadata_json=metadata_json,
        )

    def _sync_conflict_transitions(
        self,
        conn: sqlite3.Connection,
        *,
        owner_id: str,
        entity_id: str,
        attribute: str,
    ) -> list[WorldStateTransition]:
        rows = conn.execute(
            """
            SELECT * FROM runtime_world_state_claims
            WHERE owner_id = ? AND entity_id = ? AND attribute = ?
            ORDER BY updated_at DESC, world_state_claim_id ASC;
            """,
            (owner_id, entity_id, attribute),
        ).fetchall()
        now = datetime.now(UTC)
        active_rows = [
            row
            for row in rows
            if self._effective_freshness_state(row, now=now, conflict_ids=[]) != "expired"
            and not row["superseded_by_claim_id"]
        ]
        material_values = {row["material_value_json"] for row in active_rows}
        transitions: list[WorldStateTransition] = []
        if len(material_values) > 1:
            conflict_ids = [row["world_state_claim_id"] for row in active_rows]
            for row in active_rows:
                last = conn.execute(
                    """
                    SELECT transition_type FROM runtime_world_state_transitions
                    WHERE world_state_claim_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1;
                    """,
                    (row["world_state_claim_id"],),
                ).fetchone()
                if last is None or last["transition_type"] != "conflicted":
                    transitions.append(
                        self._record_transition(
                            conn,
                            owner_id=owner_id,
                            claim_id=row["world_state_claim_id"],
                            transition_type="conflicted",
                            metadata_json={"conflict_claim_ids": conflict_ids},
                        )
                    )
        elif active_rows:
            for row in active_rows:
                last = conn.execute(
                    """
                    SELECT transition_type FROM runtime_world_state_transitions
                    WHERE world_state_claim_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1;
                    """,
                    (row["world_state_claim_id"],),
                ).fetchone()
                if last is not None and last["transition_type"] == "conflicted":
                    transitions.append(
                        self._record_transition(
                            conn,
                            owner_id=owner_id,
                            claim_id=row["world_state_claim_id"],
                            transition_type="resolved",
                            metadata_json={"reason": "material_conflict_cleared"},
                        )
                    )
        return transitions

    def _conflict_map(self, conn: sqlite3.Connection, owner_id: str) -> dict[str, list[str]]:
        rows = conn.execute(
            """
            SELECT * FROM runtime_world_state_claims
            WHERE owner_id = ?
            ORDER BY updated_at DESC, world_state_claim_id ASC;
            """,
            (owner_id,),
        ).fetchall()
        now = datetime.now(UTC)
        grouped: dict[tuple[str, str, str], list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(
                (row["owner_id"], row["entity_id"], row["attribute"]),
                [],
            ).append(row)
        conflict_map: dict[str, list[str]] = {}
        for group_rows in grouped.values():
            active = [
                row
                for row in group_rows
                if not row["superseded_by_claim_id"]
                and self._effective_freshness_state(row, now=now, conflict_ids=[]) != "expired"
            ]
            if len({_material_value(_load_json(row["value_json"], None)) for row in active}) > 1:
                ids = [row["world_state_claim_id"] for row in active]
                for claim_id in ids:
                    conflict_map[claim_id] = [other for other in ids if other != claim_id]
        return conflict_map

    def _effective_freshness_state(
        self,
        row: sqlite3.Row,
        *,
        now: datetime,
        conflict_ids: list[str],
    ) -> WorldStateFreshnessState:
        stored = row["freshness_state"]
        if row["superseded_by_claim_id"]:
            return "superseded"
        if conflict_ids:
            return "conflicted"
        expires_at = _parse_ts(row["expires_at"])
        if expires_at is not None and expires_at <= now:
            return "expired"
        age_seconds = _seconds_between(now, row["observed_at"])
        ttl_seconds = row["ttl_seconds"]
        revalidation_seconds = row["revalidation_interval_seconds"]
        if ttl_seconds is not None and revalidation_seconds is not None and age_seconds is not None:
            return _computed_freshness_from_age(
                age_seconds=age_seconds,
                ttl_seconds=ttl_seconds,
                revalidation_interval_seconds=revalidation_seconds,
            )
        if ttl_seconds is not None and age_seconds is not None:
            if age_seconds >= ttl_seconds:
                return "expired"
            if age_seconds >= ttl_seconds * 0.75:
                return "aging"
            return "fresh"
        if revalidation_seconds is not None and age_seconds is not None:
            if age_seconds >= revalidation_seconds:
                return "stale"
            if age_seconds >= revalidation_seconds * 0.75:
                return "aging"
            return "fresh"
        if stored in _TERMINAL_FRESHNESS_STATES:
            return stored
        return stored

    def _claim_view_from_row(
        self,
        row: sqlite3.Row,
        *,
        include_sensitive_values: bool,
        now: datetime,
        conflict_map: dict[str, list[str]],
    ) -> WorldStateClaimView:
        conflict_ids = conflict_map.get(row["world_state_claim_id"], [])
        effective = self._effective_freshness_state(row, now=now, conflict_ids=conflict_ids)
        value = _load_json(row["value_json"], None)
        redact = row["sensitivity"] in _SENSITIVE_LEVELS and not include_sensitive_values
        return WorldStateClaimView(
            world_state_claim_id=row["world_state_claim_id"],
            owner_id=row["owner_id"],
            entity_id=row["entity_id"],
            entity_type=row["entity_type"],
            domain=row["domain"],
            attribute=row["attribute"],
            value_json=None if redact else value,
            value_redacted=redact,
            value_digest=_value_digest_from_material(row["material_value_json"]),
            source_type=row["source_type"],
            source_ref=row["source_ref"],
            verification_verifier_id=row["verification_verifier_id"],
            verification_source_type=row["verification_source_type"],
            verification_source_ref=row["verification_source_ref"],
            confidence=row["confidence"],
            freshness_state=row["freshness_state"],
            effective_freshness_state=effective,
            state_authority=row["state_authority"],
            observed_at=row["observed_at"],
            last_verified_at=row["last_verified_at"],
            last_verified_runtime_session_id=row["last_verified_runtime_session_id"],
            last_verified_runtime_turn_id=row["last_verified_runtime_turn_id"],
            last_verification_request_id=row["last_verification_request_id"],
            expires_at=row["expires_at"],
            ttl_seconds=row["ttl_seconds"],
            revalidation_interval_seconds=row["revalidation_interval_seconds"],
            confirmation_policy=row["confirmation_policy"],
            sensitivity=row["sensitivity"],
            scope_labels=_load_json(row["scope_labels_json"], []),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            superseded_by_claim_id=row["superseded_by_claim_id"],
            conflict_claim_ids=conflict_ids,
        )

    def _transition_from_row(self, row: sqlite3.Row) -> WorldStateTransition:
        return WorldStateTransition(
            transition_id=row["transition_id"],
            world_state_claim_id=row["world_state_claim_id"],
            transition_type=row["transition_type"],
            created_at=row["created_at"],
            metadata_json=_load_json(row["metadata_json"], {}),
        )


def world_state_repository() -> WorldStateRepository:
    global _WORLD_STATE_REPOSITORY
    if _WORLD_STATE_REPOSITORY is None:
        _WORLD_STATE_REPOSITORY = WorldStateRepository()
    return _WORLD_STATE_REPOSITORY


def upsert_world_state_claim(
    *,
    owner_id: str,
    claim: WorldStateClaimInput,
) -> tuple[WorldStateClaimView, list[WorldStateTransition]]:
    return world_state_repository().upsert_claim(owner_id=owner_id, claim=claim)


def verify_world_state_claim(
    body: WorldStateClaimVerifyRequest,
) -> tuple[WorldStateClaimView, list[WorldStateTransition]]:
    return world_state_repository().verify_claim(body)


def get_world_state_diagnostics(
    *,
    owner_id: str,
    include_sensitive_values: bool = False,
) -> WorldStateDiagnosticsResponse:
    return world_state_repository().diagnostics(
        owner_id=owner_id,
        include_sensitive_values=include_sensitive_values,
    )


def clear_world_state_for_tests(db_path: Path | None = None) -> None:
    global _WORLD_STATE_REPOSITORY
    clear_trusted_world_state_verifiers_for_tests()
    _WORLD_STATE_REPOSITORY = WorldStateRepository(db_path=db_path)


def resolve_world_state_persona_scope(
    *,
    request_id: str,
    owner_id: str,
    conversation_id: str,
    surface: str,
    runtime_session_id: str | None,
    active_persona_id: str | None = None,
    requested_domains: list[str] | None = None,
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
    persona_domains = _DOMAIN_ALLOWLISTS.get(
        active_persona_id,
        _DOMAIN_ALLOWLISTS["general_assistant"],
    )
    surface_domains = _SURFACE_DOMAIN_RESTRICTIONS.get(
        surface_type,
        _SURFACE_DOMAIN_RESTRICTIONS["unknown_surface"],
    )
    allowed = set(persona_domains) & set(surface_domains)
    if requested_domains:
        allowed &= {domain for domain in requested_domains if domain}
    return active_persona_id, allowed


def resolve_world_state(
    *,
    request_id: str,
    owner_id: str,
    conversation_id: str,
    surface: str,
    runtime_session_id: str | None,
    active_persona_id: str | None = None,
    requested_domains: list[str] | None = None,
) -> WorldStateResolveResponse:
    diagnostics = get_world_state_diagnostics(owner_id=owner_id, include_sensitive_values=False)
    persona_id, allowed_domains = resolve_world_state_persona_scope(
        request_id=request_id,
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface=surface,
        runtime_session_id=runtime_session_id,
        active_persona_id=active_persona_id,
        requested_domains=requested_domains,
    )

    included_claims: list[WorldStateClaimView] = []
    excluded_claims = list(diagnostics.excluded_claims)
    for claim in diagnostics.claims:
        if claim.domain not in allowed_domains:
            excluded_claims.append(
                WorldStateClaimSummary(
                    world_state_claim_id=claim.world_state_claim_id,
                    entity_id=claim.entity_id,
                    attribute=claim.attribute,
                    domain=claim.domain,
                    freshness_state=claim.freshness_state,
                    effective_freshness_state=claim.effective_freshness_state,
                    sensitivity=claim.sensitivity,
                    reason="outside_persona_or_surface_scope",
                    superseded_by_claim_id=claim.superseded_by_claim_id,
                    conflict_claim_ids=claim.conflict_claim_ids,
                )
            )
            continue
        if claim.effective_freshness_state == "expired":
            excluded_claims.append(
                WorldStateClaimSummary(
                    world_state_claim_id=claim.world_state_claim_id,
                    entity_id=claim.entity_id,
                    attribute=claim.attribute,
                    domain=claim.domain,
                    freshness_state=claim.freshness_state,
                    effective_freshness_state=claim.effective_freshness_state,
                    sensitivity=claim.sensitivity,
                    reason="expired",
                    superseded_by_claim_id=claim.superseded_by_claim_id,
                    conflict_claim_ids=claim.conflict_claim_ids,
                )
            )
            continue
        included_claims.append(claim)

    prompt_lines: list[str] = []
    confirmation_required = False
    for claim in included_claims:
        label = claim.attribute
        suffix = claim.effective_freshness_state
        if claim.effective_freshness_state in {"aging", "stale"}:
            suffix = (
                f"{claim.effective_freshness_state}; last_known; "
                f"confirmation_policy={claim.confirmation_policy}"
            )
        if claim.confirmation_policy != "none":
            confirmation_required = True
        if claim.value_redacted:
            prompt_lines.append(f"- {claim.domain}/{label}: [REDACTED] ({suffix})")
        else:
            prompt_lines.append(
                (
                    f"- {claim.domain}/{label}: "
                    f"{json.dumps(claim.value_json, sort_keys=True)} ({suffix})"
                )
            )

    trace = WorldStateResolveTrace(
        active_persona_id=persona_id,
        allowed_domains=sorted(allowed_domains),
        included_claim_count=len(included_claims),
        excluded_claim_count=len(excluded_claims),
        stale_count=sum(
            1 for claim in included_claims if claim.effective_freshness_state == "stale"
        ),
        aging_count=sum(
            1 for claim in included_claims if claim.effective_freshness_state == "aging"
        ),
        expired_count=sum(
            1 for claim in excluded_claims if claim.effective_freshness_state == "expired"
        ),
        conflicted_count=sum(
            1 for claim in excluded_claims if claim.effective_freshness_state == "conflicted"
        ),
        confirmation_required=confirmation_required,
    )
    return WorldStateResolveResponse(
        included_claims=included_claims,
        excluded_claim_summaries=excluded_claims,
        prompt_content="World state:\n" + "\n".join(prompt_lines) if prompt_lines else None,
        trace=trace,
    )
