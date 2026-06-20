from __future__ import annotations

from models import (
    PrivacyContextEvaluateRequest,
    PrivacyContextEvaluateResponse,
    PrivacyContextResult,
    PrivacySensitivityLevel,
    PrivacySurfaceCategory,
)
from services.runtime_state import (
    record_runtime_event,
    resolve_runtime_session,
    runtime_session_by_id,
    validate_runtime_turn_session,
)

_KNOWN_SURFACE_CATEGORIES = {
    "desktop_private",
    "mobile_private",
    "telegram_private",
    "voice_private",
    "car_voice_possible_passenger",
    "glasses_public_or_semi_public",
    "notification_preview",
    "unknown_surface",
}
_SENSITIVE_DOMAINS = {"personal", "health", "financial", "work"}


def _resolve_surface_category(
    body: PrivacyContextEvaluateRequest,
) -> tuple[PrivacySurfaceCategory, list[str]]:
    reasons: list[str] = []
    explicit_category = body.surface_category
    surface_value = body.surface.strip().lower()
    category_from_surface = surface_value if surface_value in _KNOWN_SURFACE_CATEGORIES else None

    if explicit_category and category_from_surface and explicit_category != category_from_surface:
        raise ValueError("surface_context_mismatch")

    if explicit_category is not None:
        return explicit_category, reasons

    if category_from_surface is not None:
        return category_from_surface, reasons

    reasons.extend(["surface_category_missing", "unknown_surface_conservative"])
    return "unknown_surface", reasons


def _is_sensitive(
    sensitivity_level: PrivacySensitivityLevel,
    sensitivity_domains: list[str],
) -> bool:
    if sensitivity_level in {"sensitive", "highly_sensitive", "unknown"}:
        return True
    return any(domain in _SENSITIVE_DOMAINS for domain in sensitivity_domains)


def _evaluate_result(
    *,
    surface_category: PrivacySurfaceCategory,
    sensitivity_level: PrivacySensitivityLevel,
    sensitivity_domains: list[str],
    initial_reasons: list[str],
) -> PrivacyContextResult:
    reasons = list(initial_reasons)
    sensitive = _is_sensitive(sensitivity_level, sensitivity_domains)
    if sensitivity_level == "highly_sensitive":
        reasons.append("highly_sensitive_content")
    elif sensitivity_level in {"sensitive", "unknown"} or sensitivity_domains:
        reasons.append("sensitive_content")
    if sensitivity_level == "unknown":
        reasons.append("unknown_sensitivity_conservative")

    if surface_category in {"desktop_private", "mobile_private", "telegram_private"}:
        reasons.append("private_surface")
        unknown_requires_summary = sensitivity_level == "unknown"
        return PrivacyContextResult(
            privacy_zone="private",
            surface_type=surface_category,
            sensitivity_level=sensitivity_level,
            sensitive_detail_allowed=not unknown_requires_summary,
            notification_detail_allowed=True,
            voice_detail_allowed=False,
            screen_detail_allowed=True,
            redaction_required=unknown_requires_summary,
            safe_summary_required=unknown_requires_summary,
            reason_codes=list(dict.fromkeys(reasons))[:8],
        )

    if surface_category == "voice_private":
        reasons.extend(["private_surface", "voice_surface"])
        requires_safe_summary = sensitivity_level in {"highly_sensitive", "unknown"}
        return PrivacyContextResult(
            privacy_zone="private",
            surface_type=surface_category,
            sensitivity_level=sensitivity_level,
            sensitive_detail_allowed=not requires_safe_summary,
            notification_detail_allowed=False,
            voice_detail_allowed=not requires_safe_summary,
            screen_detail_allowed=False,
            redaction_required=requires_safe_summary,
            safe_summary_required=requires_safe_summary,
            reason_codes=list(dict.fromkeys(reasons))[:8],
        )

    if surface_category == "car_voice_possible_passenger":
        reasons.extend(["voice_surface", "passenger_presence_unknown"])
        restricted = sensitive
        if restricted:
            reasons.append("safe_summary_required")
        return PrivacyContextResult(
            privacy_zone="shared_or_uncertain",
            surface_type=surface_category,
            sensitivity_level=sensitivity_level,
            sensitive_detail_allowed=not restricted,
            notification_detail_allowed=False,
            voice_detail_allowed=not restricted,
            screen_detail_allowed=True,
            redaction_required=restricted,
            safe_summary_required=restricted,
            reason_codes=list(dict.fromkeys(reasons))[:8],
        )

    if surface_category == "glasses_public_or_semi_public":
        reasons.append("public_or_semi_public_surface")
        restricted = sensitive
        if restricted:
            reasons.append("safe_summary_required")
        return PrivacyContextResult(
            privacy_zone="public_or_semi_public",
            surface_type=surface_category,
            sensitivity_level=sensitivity_level,
            sensitive_detail_allowed=not restricted,
            notification_detail_allowed=False,
            voice_detail_allowed=False,
            screen_detail_allowed=not restricted,
            redaction_required=restricted,
            safe_summary_required=restricted,
            reason_codes=list(dict.fromkeys(reasons))[:8],
        )

    if surface_category == "notification_preview":
        reasons.append("notification_preview_limited")
        restricted = sensitive
        if restricted:
            reasons.append("safe_summary_required")
        return PrivacyContextResult(
            privacy_zone="preview_limited",
            surface_type=surface_category,
            sensitivity_level=sensitivity_level,
            sensitive_detail_allowed=not restricted,
            notification_detail_allowed=not restricted,
            voice_detail_allowed=False,
            screen_detail_allowed=True,
            redaction_required=restricted,
            safe_summary_required=restricted,
            reason_codes=list(dict.fromkeys(reasons))[:8],
        )

    reasons.extend(["unknown_surface_conservative", "safe_summary_required"])
    return PrivacyContextResult(
        privacy_zone="unknown",
        surface_type="unknown_surface",
        sensitivity_level=sensitivity_level,
        sensitive_detail_allowed=False,
        notification_detail_allowed=False,
        voice_detail_allowed=False,
        screen_detail_allowed=True,
        redaction_required=True,
        safe_summary_required=True,
        reason_codes=list(dict.fromkeys(reasons))[:8],
    )


def evaluate_privacy_context(
    body: PrivacyContextEvaluateRequest,
) -> PrivacyContextEvaluateResponse:
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

    if body.runtime_turn_id:
        validate_runtime_turn_session(
            runtime_session_id=runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
        )

    surface_category, initial_reasons = _resolve_surface_category(body)
    result = _evaluate_result(
        surface_category=surface_category,
        sensitivity_level=body.sensitivity_level,
        sensitivity_domains=body.sensitivity_domains,
        initial_reasons=initial_reasons,
    )

    record_runtime_event(
        runtime_session_id=runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        event_type="privacy_context_evaluated",
        event_payload_json={
            "request_id": body.request_id,
            "surface_type": result.surface_type,
            "privacy_zone": result.privacy_zone,
            "sensitivity_level": result.sensitivity_level,
            "sensitive_detail_allowed": result.sensitive_detail_allowed,
            "notification_detail_allowed": result.notification_detail_allowed,
            "voice_detail_allowed": result.voice_detail_allowed,
            "screen_detail_allowed": result.screen_detail_allowed,
            "redaction_required": result.redaction_required,
            "safe_summary_required": result.safe_summary_required,
            "reason_codes": result.reason_codes,
        },
    )

    return PrivacyContextEvaluateResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        runtime_session_id=runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        result=result,
    )
