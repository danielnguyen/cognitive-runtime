from __future__ import annotations

import re

from models import (
    ArtifactAccessPolicy,
    PersonaContainmentEvaluateRequest,
    PersonaContainmentEvaluateResponse,
    PersonaContainmentResult,
)
from services.companion_contracts import companion_contracts_repository
from services.runtime_state import (
    record_runtime_event,
    resolve_runtime_session,
    runtime_session_by_id,
)

_SEEDED_DOMAINS = (
    "general",
    "technical",
    "project",
    "infrastructure",
    "vehicle_maintenance",
    "work_professional",
    "personal",
    "health",
    "finance",
)
_CANONICAL_PERSONAS = {
    "general_assistant",
    "technical_architect",
    "operations_assistant",
    "personal_companion",
}
_PERSONA_SCOPE_HINTS = {
    "general_assistant": "general_assistant",
    "technical_operator": "technical_architect",
    "supportive_listener": "personal_companion",
    "careful_decider": "general_assistant",
}
_PERSONA_DEFAULT_DOMAIN = {
    "general_assistant": "general",
    "technical_architect": "technical",
    "operations_assistant": "work_professional",
    "personal_companion": "personal",
}
_PERSONA_BASE_ALLOWED_DOMAINS = {
    "general_assistant": {"general"},
    "technical_architect": {"general", "technical", "project", "infrastructure"},
    "operations_assistant": {"general", "work_professional", "infrastructure"},
    "personal_companion": {"general", "personal"},
}
_PERSONA_ARTIFACT_CONTENT_CLASSES = {
    "general_assistant": {"document", "image", "screenshot"},
    "technical_architect": {"document", "code", "image", "screenshot"},
    "operations_assistant": {"document", "code", "image", "screenshot"},
    "personal_companion": {"document", "image", "screenshot"},
}
_SURFACE_ARTIFACT_CONTENT_CLASSES = {
    "dev": {"document", "code"},
    "vscode": {"document", "code"},
    "web": {"document", "code", "image", "screenshot"},
}
_ARTIFACT_CONTENT_CLASS_ORDER = (
    "document",
    "code",
    "image",
    "screenshot",
    "audio",
    "video",
    "other",
)
_TECHNICAL_MARKERS = (
    "code",
    "repo",
    "repository",
    "function",
    "variable",
    "schema",
    "refactor",
    "implementation",
    "architecture",
    "bug",
    "debug",
    "service",
    "api",
)
_PROJECT_MARKERS = (
    "project",
    "milestone",
    "roadmap",
    "spec",
    "ticket",
    "sprint",
    "task",
)
_INFRASTRUCTURE_MARKERS = (
    "infra",
    "infrastructure",
    "server",
    "production",
    "prod",
    "deploy",
    "deployment",
    "cluster",
    "runbook",
    "incident",
)
_VEHICLE_MARKERS = (
    "vehicle",
    "car",
    "truck",
    "brake",
    "oil change",
    "tire",
    "engine",
    "maintenance",
    "mileage",
    "service interval",
)
_WORK_MARKERS = (
    "work",
    "professional",
    "manager",
    "meeting",
    "office",
    "job",
    "teammate",
    "colleague",
)
_PERSONAL_MARKERS = (
    "personal",
    "family",
    "home",
    "private",
    "work-life",
    "life balance",
)
_HEALTH_MARKERS = (
    "health",
    "medical",
    "doctor",
    "symptom",
    "medication",
    "therapy",
    "sleep",
)
_FINANCE_MARKERS = (
    "finance",
    "financial",
    "money",
    "budget",
    "tax",
    "taxes",
    "expense",
    "expenses",
    "debt",
    "savings",
)
_DOMAIN_SYNONYMS = {
    "general": "general",
    "technical": "technical",
    "tech": "technical",
    "project": "project",
    "projects": "project",
    "infrastructure": "infrastructure",
    "infra": "infrastructure",
    "vehicle": "vehicle_maintenance",
    "vehicle notes": "vehicle_maintenance",
    "vehicle context": "vehicle_maintenance",
    "car": "vehicle_maintenance",
    "maintenance": "vehicle_maintenance",
    "equipment maintenance": "vehicle_maintenance",
    "work": "work_professional",
    "work context": "work_professional",
    "professional": "work_professional",
    "professional context": "work_professional",
    "personal": "personal",
    "health": "health",
    "finances": "finance",
    "finance": "finance",
}
_BRIDGE_PATTERNS = (
    re.compile(r"\bcompare this with my ([a-z_ ]+?)(?: context)?\b"),
    re.compile(r"\buse my ([a-z_ ]+?) notes\b"),
    re.compile(r"\bbring in ([a-z_ ]+?)(?: context)?\b"),
    re.compile(r"\bconnect this to ([a-z_ ]+?)(?: context)?(?:[.!?]|$)"),
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _latest_user_text(body: PersonaContainmentEvaluateRequest) -> str:
    if body.current_user_text:
        return body.current_user_text.strip()
    for message in reversed(body.recent_messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    return ""


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _surface_default_persona(surface: str) -> str | None:
    repository = companion_contracts_repository()
    binding = repository.surface_binding(surface)
    if binding is not None and binding.default_persona_id in _CANONICAL_PERSONAS:
        return binding.default_persona_id
    binding = repository.surface_binding("unknown")
    if binding is not None and binding.default_persona_id in _CANONICAL_PERSONAS:
        return binding.default_persona_id
    return None


def _resolve_persona(body: PersonaContainmentEvaluateRequest) -> tuple[str, list[str]]:
    reasons: list[str] = []
    repository = companion_contracts_repository()

    if body.requested_persona_id:
        if body.requested_persona_id in _CANONICAL_PERSONAS and repository.persona_profile(
            body.requested_persona_id
        ):
            return body.requested_persona_id, ["requested_persona_id"]
        reasons.append("requested_persona_not_canonical")

    if body.active_persona_id:
        if body.active_persona_id in _CANONICAL_PERSONAS and repository.persona_profile(
            body.active_persona_id
        ):
            return body.active_persona_id, ["active_persona_id"]
        reasons.append("active_persona_not_canonical")

    if body.persona_scope_hint:
        mapped = _PERSONA_SCOPE_HINTS.get(body.persona_scope_hint)
        if mapped and repository.persona_profile(mapped):
            return mapped, ["persona_scope_hint"]
        reasons.append("persona_scope_hint_unmapped")

    surface_persona = _surface_default_persona(body.surface)
    if surface_persona is not None:
        return surface_persona, reasons + ["surface_default_persona"]

    return "general_assistant", reasons + ["default_fallback"]


def _extract_bridge_target(text: str) -> str | None:
    for pattern in _BRIDGE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def _normalize_domain_label(raw: str) -> str:
    label = re.sub(r"[^a-z0-9_ ]+", "", raw.lower()).strip()
    label = re.sub(r"\s+", "_", label)
    return label[:64]


def _map_domain_term(term: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9_ ]+", "", term.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized.startswith("my "):
        normalized = normalized[3:]
    if normalized.endswith(" context"):
        normalized = normalized[: -len(" context")]
    mapped = _DOMAIN_SYNONYMS.get(normalized)
    if mapped is not None:
        return mapped
    normalized_label = _normalize_domain_label(normalized)
    if normalized_label in _SEEDED_DOMAINS:
        return normalized_label
    return None


def _capability_domain(text: str, persona_id: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if _contains_any(text, _VEHICLE_MARKERS):
        return "vehicle_maintenance", ["vehicle_maintenance_markers"]
    if _contains_any(text, _FINANCE_MARKERS):
        return "finance", ["finance_markers"]
    if _contains_any(text, _HEALTH_MARKERS):
        return "health", ["health_markers"]
    if _contains_any(text, _PERSONAL_MARKERS):
        return "personal", ["personal_markers"]
    if _contains_any(text, _INFRASTRUCTURE_MARKERS):
        return "infrastructure", ["infrastructure_markers"]
    if _contains_any(text, _TECHNICAL_MARKERS):
        return "technical", ["technical_markers"]
    if _contains_any(text, _PROJECT_MARKERS):
        return "project", ["project_markers"]
    if _contains_any(text, _WORK_MARKERS):
        return "work_professional", ["work_professional_markers"]
    reasons.append("persona_default_domain")
    return _PERSONA_DEFAULT_DOMAIN.get(persona_id, "general"), reasons


def _matched_domains(text: str) -> set[str]:
    matched: set[str] = set()
    if _contains_any(text, _VEHICLE_MARKERS):
        matched.add("vehicle_maintenance")
    if _contains_any(text, _FINANCE_MARKERS):
        matched.add("finance")
    if _contains_any(text, _HEALTH_MARKERS):
        matched.add("health")
    if _contains_any(text, _PERSONAL_MARKERS):
        matched.add("personal")
    if _contains_any(text, _INFRASTRUCTURE_MARKERS):
        matched.add("infrastructure")
    if _contains_any(text, _TECHNICAL_MARKERS):
        matched.add("technical")
    if _contains_any(text, _PROJECT_MARKERS):
        matched.add("project")
    if _contains_any(text, _WORK_MARKERS):
        matched.add("work_professional")
    return matched


def _base_allowed_domains(persona_id: str, capability_domain: str) -> set[str]:
    allowed = set(_PERSONA_BASE_ALLOWED_DOMAINS.get(persona_id, {"general"}))
    if capability_domain in _SEEDED_DOMAINS:
        allowed.add(capability_domain)
    return allowed


def _explicit_sensitive_domain_requests(text: str) -> set[str]:
    allowed: set[str] = set()
    if _contains_any(text, _PERSONAL_MARKERS):
        allowed.add("personal")
    if _contains_any(text, _HEALTH_MARKERS):
        allowed.add("health")
    if _contains_any(text, _FINANCE_MARKERS):
        allowed.add("finance")
    return allowed


def _apply_conservative_restrictions(
    *,
    text: str,
    persona_id: str,
    capability_domain: str,
    allowed_domains: set[str],
) -> set[str]:
    if capability_domain == "work_professional" and not _contains_any(text, _TECHNICAL_MARKERS):
        allowed_domains.discard("technical")
    if capability_domain == "work_professional" and not _contains_any(text, _PROJECT_MARKERS):
        allowed_domains.discard("project")
    if (
        capability_domain == "work_professional"
        and not _contains_any(text, _INFRASTRUCTURE_MARKERS)
    ):
        allowed_domains.discard("infrastructure")
    if capability_domain == "vehicle_maintenance":
        allowed_domains.discard("work_professional")
        allowed_domains.discard("technical")
        allowed_domains.discard("project")
        allowed_domains.discard("infrastructure")
    if persona_id != "personal_companion":
        allowed_domains.discard("personal")
    explicit_sensitive = _explicit_sensitive_domain_requests(text)
    for domain in ("personal", "health", "finance"):
        if domain not in explicit_sensitive and domain != capability_domain:
            allowed_domains.discard(domain)
    return allowed_domains


def _requires_conservative_multi_domain_scope(
    *,
    matched_domains: set[str],
    bridge_target: str | None,
) -> bool:
    if bridge_target is not None:
        return False
    if "vehicle_maintenance" in matched_domains and (
        matched_domains & {"technical", "project", "infrastructure", "work_professional"}
    ):
        return True
    if {"work_professional", "personal"}.issubset(matched_domains):
        return True
    return False


def _sorted_domains(values: set[str]) -> list[str]:
    return sorted(value for value in values if value)


def _ordered_artifact_classes(values: set[str]) -> list[str]:
    return [value for value in _ARTIFACT_CONTENT_CLASS_ORDER if value in values]


def _artifact_access_policy(
    *,
    persona_id: str,
    surface: str,
    allowed_memory_domains: list[str],
    cross_scope_access_allowed: bool,
) -> ArtifactAccessPolicy:
    persona_classes = set(_PERSONA_ARTIFACT_CONTENT_CLASSES.get(persona_id, set()))
    surface_classes = set(_SURFACE_ARTIFACT_CONTENT_CLASSES.get(surface, set()))
    allowed_classes = persona_classes & surface_classes
    reason_codes = ["artifact_policy_applied", "restricted_artifact_access_blocked"]

    if persona_classes != set(_ARTIFACT_CONTENT_CLASS_ORDER):
        reason_codes.append("persona_content_class_limited")
    if surface not in _SURFACE_ARTIFACT_CONTENT_CLASSES:
        reason_codes.append("unknown_surface_no_artifact_access")
    elif surface_classes != set(_ARTIFACT_CONTENT_CLASS_ORDER):
        reason_codes.append("surface_content_class_limited")
    if cross_scope_access_allowed:
        reason_codes.append("cross_scope_domain_authorized")

    return ArtifactAccessPolicy(
        enforcement_mode="mandatory",
        allowed_content_classes=_ordered_artifact_classes(allowed_classes),
        allowed_domains=allowed_memory_domains,
        maximum_sensitivity="high" if surface in _SURFACE_ARTIFACT_CONTENT_CLASSES else "low",
        surface_content_capabilities=_ordered_artifact_classes(surface_classes),
        reason_codes=reason_codes[:8],
    )


def evaluate_persona_containment(
    body: PersonaContainmentEvaluateRequest,
) -> PersonaContainmentEvaluateResponse:
    runtime_session_id = body.runtime_session_id
    if runtime_session_id:
        session = runtime_session_by_id(runtime_session_id)
        if session is None:
            raise ValueError("runtime_session_not_found")
        if (
            session.owner_id != body.owner_id
            or session.conversation_id != body.conversation_id
            or session.surface != body.surface
        ):
            raise ValueError("runtime_session_mismatch")
    else:
        session = resolve_runtime_session(
            request_id=body.request_id,
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            surface=body.surface,
        )
        runtime_session_id = session.runtime_session_id

    raw_text = _latest_user_text(body)
    text = _normalize_text(raw_text)
    persona_id, persona_reasons = _resolve_persona(body)
    capability_domain, capability_reasons = _capability_domain(text, persona_id)
    matched_domains = _matched_domains(text)
    allowed_domains = _base_allowed_domains(persona_id, capability_domain)
    allowed_domains = _apply_conservative_restrictions(
        text=text,
        persona_id=persona_id,
        capability_domain=capability_domain,
        allowed_domains=allowed_domains,
    )

    bridge_target = _extract_bridge_target(text)
    cross_scope_access_allowed = False
    cross_scope_reason = "not_requested"
    extra_blocked_domains: set[str] = set()
    reason_summary: list[str] = [*persona_reasons, *capability_reasons]

    if _requires_conservative_multi_domain_scope(
        matched_domains=matched_domains,
        bridge_target=bridge_target,
    ):
        allowed_domains &= {"general", capability_domain}
        reason_summary.append("multi_domain_signal_conservative_scope")

    if bridge_target:
        mapped_domain = _map_domain_term(bridge_target)
        if mapped_domain is None:
            normalized_label = _normalize_domain_label(bridge_target)
            if normalized_label:
                extra_blocked_domains.add(normalized_label)
            cross_scope_reason = "domain_not_policy_mapped"
            reason_summary.append("domain_not_policy_mapped")
        else:
            allowed_domains.add(mapped_domain)
            cross_scope_access_allowed = True
            cross_scope_reason = "explicit_bridge_request_detected"
            reason_summary.append("explicit_bridge_request_detected")

    if capability_domain not in _SEEDED_DOMAINS:
        capability_domain = "general"
        reason_summary.append("domain_not_policy_mapped")

    blocked_domains = (set(_SEEDED_DOMAINS) - allowed_domains) | extra_blocked_domains
    sorted_allowed_domains = _sorted_domains(set(allowed_domains))
    artifact_access_policy = _artifact_access_policy(
        persona_id=persona_id,
        surface=body.surface,
        allowed_memory_domains=sorted_allowed_domains,
        cross_scope_access_allowed=cross_scope_access_allowed,
    )

    result = PersonaContainmentResult(
        active_persona_id=persona_id,
        capability_domain=capability_domain,
        allowed_memory_domains=sorted_allowed_domains,
        blocked_memory_domains=_sorted_domains(blocked_domains),
        allowed_world_state_domains=sorted_allowed_domains,
        allowed_relationship_domains=sorted_allowed_domains,
        allowed_tool_domains=sorted_allowed_domains,
        cross_scope_access_allowed=cross_scope_access_allowed,
        cross_scope_reason=cross_scope_reason,
        confidence=0.82 if raw_text else 0.46,
        reason_summary=reason_summary[:8] or ["default_fallback"],
        artifact_access_policy=artifact_access_policy,
    )

    record_runtime_event(
        runtime_session_id=runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        event_type="persona_containment_evaluated",
        event_payload_json={
            "request_id": body.request_id,
            "active_persona_id": result.active_persona_id,
            "capability_domain": result.capability_domain,
            "allowed_memory_domains": result.allowed_memory_domains,
            "blocked_memory_domains": result.blocked_memory_domains,
            "allowed_tool_domains": result.allowed_tool_domains,
            "artifact_access_policy": result.artifact_access_policy.model_dump(),
            "cross_scope_access_allowed": result.cross_scope_access_allowed,
            "cross_scope_reason": result.cross_scope_reason,
            "reason_summary": result.reason_summary,
        },
    )

    return PersonaContainmentEvaluateResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        runtime_session_id=runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        result=result,
    )
