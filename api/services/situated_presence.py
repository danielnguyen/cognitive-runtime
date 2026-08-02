from __future__ import annotations

from models import (
    SITUATED_PRESENCE_REASON_ORDER,
    InteractionGovernanceResponsePosture,
    SituatedPresenceEvaluateRequest,
    SituatedPresenceEvaluateResponse,
    SituatedPresenceReason,
    SituatedPresenceResult,
)
from services.runtime_state import (
    record_runtime_event,
    runtime_session_by_id,
    validate_runtime_turn_session,
)

_MINIMUM_UPSTREAM_CONFIDENCE = 0.60


def _surface_allows_commentary(body: SituatedPresenceEvaluateRequest) -> bool:
    return (
        body.surface_context.visibility == "private"
        and body.surface_context.constraint == "normal"
    )


def _ordered_reasons(
    reasons: set[SituatedPresenceReason],
) -> list[SituatedPresenceReason]:
    if not reasons:
        reasons.add("ambiguous_context")
    return [reason for reason in SITUATED_PRESENCE_REASON_ORDER if reason in reasons][
        :8
    ]


def _context_reasons(
    body: SituatedPresenceEvaluateRequest,
) -> set[SituatedPresenceReason]:
    reasons: set[SituatedPresenceReason] = set()
    visibility_reason: dict[str, SituatedPresenceReason] = {
        "public": "surface_public",
        "shared": "surface_shared",
        "unknown": "surface_visibility_unknown",
    }
    constraint_reason: dict[str, SituatedPresenceReason] = {
        "constrained": "surface_constrained",
        "unknown": "surface_constraint_unknown",
    }
    if reason := visibility_reason.get(body.surface_context.visibility):
        reasons.add(reason)
    if reason := constraint_reason.get(body.surface_context.constraint):
        reasons.add(reason)

    governance = body.interaction_governance
    restraint = body.restraint
    if not governance.commentary_allowed:
        reasons.add("upstream_commentary_suppressed")
    if not governance.humor_allowed:
        reasons.add("upstream_humor_suppressed")
    if governance.privacy_sensitivity_hint != "normal":
        reasons.add("privacy_sensitive")
    if governance.requires_confirmation:
        reasons.add("confirmation_required")
    if restraint.proactive_output_suppressed:
        reasons.add("proactive_output_suppressed")
    if restraint.personalization_suppressed:
        reasons.add("personalization_suppressed")
    if restraint.brevity_preferred:
        reasons.add("brevity_preferred")
    if restraint.clarification_preferred:
        reasons.add("clarification_preferred")
    return reasons


def _brief_or_upstream_posture(
    body: SituatedPresenceEvaluateRequest,
) -> InteractionGovernanceResponsePosture:
    posture = body.interaction_governance.response_posture
    if body.restraint.brevity_preferred and posture not in {"direct", "tactical"}:
        return "brief"
    return posture


def _evaluate_policy(body: SituatedPresenceEvaluateRequest) -> SituatedPresenceResult:
    governance = body.interaction_governance
    restraint = body.restraint
    surface_allows = _surface_allows_commentary(body)

    if (
        governance.confidence < _MINIMUM_UPSTREAM_CONFIDENCE
        or restraint.confidence < _MINIMUM_UPSTREAM_CONFIDENCE
    ):
        return SituatedPresenceResult(
            commentary_allowed=False,
            humor_allowed=False,
            emotional_attunement_allowed="none",
            challenge_allowed="none",
            silence_preferred=True,
            surface_allows_commentary=surface_allows,
            response_posture="silent_or_minimal",
            action_implication_allowed=False,
            reason_summary=["upstream_confidence_insufficient"],
        )

    reasons = _context_reasons(body)
    commentary_allowed = (
        governance.commentary_allowed
        and surface_allows
        and governance.tension_level == "low"
        and governance.interaction_kind
        not in {"tense_debugging", "high_impact_decision", "ambiguous"}
        and governance.privacy_sensitivity_hint == "normal"
        and not governance.requires_confirmation
        and not restraint.clarification_preferred
    )

    if (
        governance.interaction_kind == "tense_debugging"
        or governance.tension_level == "high"
    ):
        reasons.update({"tense_context", "tactical_response_required"})
        emotional_attunement = (
            "minimal"
            if governance.privacy_sensitivity_hint == "normal"
            and not restraint.personalization_suppressed
            else "none"
        )
        return SituatedPresenceResult(
            commentary_allowed=False,
            humor_allowed=False,
            emotional_attunement_allowed=emotional_attunement,
            challenge_allowed="medium",
            silence_preferred=False,
            surface_allows_commentary=surface_allows,
            response_posture="tactical",
            action_implication_allowed=False,
            reason_summary=_ordered_reasons(reasons),
        )

    if governance.interaction_kind == "high_impact_decision":
        reasons.add("high_impact_context")
        emotional_attunement = (
            "minimal"
            if governance.privacy_sensitivity_hint == "normal"
            and not restraint.personalization_suppressed
            else "none"
        )
        posture: InteractionGovernanceResponsePosture = (
            "brief"
            if restraint.brevity_preferred
            or governance.response_posture in {"brief", "silent_or_minimal"}
            else "direct"
        )
        return SituatedPresenceResult(
            commentary_allowed=False,
            humor_allowed=False,
            emotional_attunement_allowed=emotional_attunement,
            challenge_allowed="low",
            silence_preferred=False,
            surface_allows_commentary=surface_allows,
            response_posture=posture,
            action_implication_allowed=False,
            reason_summary=_ordered_reasons(reasons),
        )

    if governance.interaction_kind in {
        "vent_or_expression",
        "mistake_or_failure_report",
    }:
        commentary_allowed = False
        attunement_allowed = (
            surface_allows
            and governance.privacy_sensitivity_hint in {"normal", "private"}
            and not restraint.clarification_preferred
        )
        if attunement_allowed:
            emotional_attunement = "brief"
            reasons.add("brief_steadying_allowed")
        elif (
            governance.privacy_sensitivity_hint == "sensitive"
            or restraint.clarification_preferred
        ):
            emotional_attunement = "none"
        else:
            emotional_attunement = "minimal"
        silence_preferred = False
        posture = (
            "brief"
            if restraint.brevity_preferred
            or restraint.personalization_suppressed
            or restraint.proactive_output_suppressed
            or governance.privacy_sensitivity_hint != "normal"
            or not surface_allows
            else "supportive"
        )
        return SituatedPresenceResult(
            commentary_allowed=commentary_allowed,
            humor_allowed=False,
            emotional_attunement_allowed=emotional_attunement,
            challenge_allowed=(
                "low"
                if governance.interaction_kind == "mistake_or_failure_report"
                else "none"
            ),
            silence_preferred=silence_preferred,
            surface_allows_commentary=surface_allows,
            response_posture=posture,
            action_implication_allowed=False,
            reason_summary=_ordered_reasons(reasons),
        )

    if governance.interaction_kind == "joke_or_playful":
        humor_allowed = commentary_allowed and governance.humor_allowed
        if humor_allowed:
            reasons.add("light_commentary_allowed")
        elif commentary_allowed:
            reasons.add("low_risk_commentary_allowed")
        silence_preferred = not commentary_allowed and (
            restraint.clarification_preferred or not surface_allows
        )
        return SituatedPresenceResult(
            commentary_allowed=commentary_allowed,
            humor_allowed=humor_allowed,
            emotional_attunement_allowed="none",
            challenge_allowed="low" if commentary_allowed else "none",
            silence_preferred=silence_preferred,
            surface_allows_commentary=surface_allows,
            response_posture=(
                "brief"
                if commentary_allowed and restraint.brevity_preferred
                else "playful"
                if commentary_allowed
                else "silent_or_minimal"
                if silence_preferred
                else _brief_or_upstream_posture(body)
            ),
            action_implication_allowed=False,
            reason_summary=_ordered_reasons(reasons),
        )

    if governance.interaction_kind == "ambiguous":
        reasons.add("ambiguous_context")
        return SituatedPresenceResult(
            commentary_allowed=False,
            humor_allowed=False,
            emotional_attunement_allowed="none",
            challenge_allowed="none",
            silence_preferred=True,
            surface_allows_commentary=surface_allows,
            response_posture="silent_or_minimal",
            action_implication_allowed=False,
            reason_summary=_ordered_reasons(reasons),
        )

    if commentary_allowed:
        reasons.add("low_risk_commentary_allowed")
    silence_preferred = restraint.proactive_output_suppressed and (
        governance.response_posture == "silent_or_minimal"
    )
    return SituatedPresenceResult(
        commentary_allowed=commentary_allowed,
        humor_allowed=False,
        emotional_attunement_allowed="none",
        challenge_allowed=(
            "low" if governance.interaction_kind == "brainstorm" else "none"
        ),
        silence_preferred=silence_preferred,
        surface_allows_commentary=surface_allows,
        response_posture=(
            "silent_or_minimal"
            if silence_preferred
            else _brief_or_upstream_posture(body)
        ),
        action_implication_allowed=False,
        reason_summary=_ordered_reasons(reasons),
    )


def evaluate_situated_presence(
    body: SituatedPresenceEvaluateRequest,
) -> SituatedPresenceEvaluateResponse:
    session = runtime_session_by_id(body.runtime_session_id)
    if session is None:
        raise ValueError("runtime_session_not_found")
    if (
        session.owner_id != body.owner_id
        or session.conversation_id != body.conversation_id
        or session.surface != body.surface
    ):
        raise ValueError("runtime_session_mismatch")
    turn = validate_runtime_turn_session(
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
    )
    if turn.turn_status in {"completed", "abandoned"}:
        raise RuntimeError("runtime_turn_not_current")

    result = _evaluate_policy(body)
    record_runtime_event(
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        event_type="situated_presence_evaluated",
        event_payload_json=result.model_dump(),
    )
    return SituatedPresenceEvaluateResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        result=result,
    )
