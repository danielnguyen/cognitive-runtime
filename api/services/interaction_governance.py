from __future__ import annotations

import re

from models import (
    HistoryFollowupAcquisitionQuestion,
    HistoryFollowupExplanationKind,
    HistoryFollowupPolicyResult,
    InteractionGovernanceEvaluateRequest,
    InteractionGovernanceEvaluateResponse,
    InteractionGovernanceKind,
    InteractionGovernanceResult,
)
from services.runtime_state import (
    record_runtime_event,
    resolve_runtime_session,
    runtime_session_by_id,
    update_runtime_turn_intent_class,
)

_DESTRUCTIVE_MARKERS = (
    "nuke",
    "wipe",
    "drop",
    "delete everything",
    "kill",
    "blow away",
    "burn it down",
)
_DEBUG_MARKERS = (
    "broke",
    "broken",
    "failing",
    "failure",
    "outage",
    "incident",
    "prod",
    "production",
    "error",
    "crash",
    "regression",
    "urgent",
    "server",
    "deploy",
)
_HIGH_IMPACT_MARKERS = (
    "database",
    "credentials",
    "security",
    "legal",
    "medical",
    "finance",
    "financial",
    "payroll",
    "tax",
    "taxes",
    "account access",
    "delete data",
)
_BRAINSTORM_MARKERS = (
    "brainstorm",
    "explore",
    "options",
    "ideas",
    "possibilities",
    "what if",
)
_PLAYFUL_MARKERS = (
    "lol",
    "haha",
    "jk",
    "joking",
    "roast",
    "meme",
    "silly",
    "playful",
)
_CORRECTION_MARKERS = (
    "actually",
    "correction",
    "i meant",
    "that was wrong",
    "to clarify",
)
_CLARIFICATION_REQUEST_PREFIXES = (
    "what do you mean",
    "did you mean",
    "which one",
    "which part",
    "can you clarify",
    "could you clarify",
)
_CONFIRMATION_RESPONSE_MARKERS = (
    "yes",
    "yes please",
    "yeah",
    "yep",
    "no",
    "no thanks",
    "nope",
    "correct",
    "that's right",
    "thats right",
    "exactly",
)
_CONTINUATION_MARKERS = (
    "continue",
    "go on",
    "keep going",
    "tell me more",
    "more detail",
    "more details",
)
_VENT_MARKERS = (
    "frustrated",
    "annoyed",
    "this sucks",
    "i'm upset",
    "i am upset",
    "ugh",
    "wtf",
)
_COMMAND_PREFIXES = (
    "rename ",
    "change ",
    "update ",
    "fix ",
    "add ",
    "remove ",
    "refactor ",
    "create ",
    "write ",
    "show ",
    "list ",
)
_COMMAND_PHRASES = (
    "please ",
    "can you ",
    "could you ",
)
_QUESTION_WORDS = ("what", "why", "how", "when", "where", "who", "which", "should")
_ALL_CAPS_TOKEN = re.compile(r"\b[A-Z]{4,}\b")

HISTORY_FOLLOWUP_HIGH_CONFIDENCE_THRESHOLD = 0.85
HISTORY_FOLLOWUP_CLARIFICATION_THRESHOLD = 0.60

_HISTORY_INTENT_PROJECTION: dict[
    str, tuple[HistoryFollowupExplanationKind, HistoryFollowupAcquisitionQuestion | None]
] = {
    "support_explanation": ("support", None),
    "acquisition_checked": ("acquisition", "checked"),
    "acquisition_coverage": ("acquisition", "coverage"),
    "acquisition_gaps": ("acquisition", "gaps"),
    "new_verification_request": ("support", None),
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _latest_user_text(body: InteractionGovernanceEvaluateRequest) -> str:
    if body.current_user_text:
        return body.current_user_text.strip()
    for message in reversed(body.recent_messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    return ""


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _immediately_preceding_assistant_text(
    body: InteractionGovernanceEvaluateRequest,
) -> str:
    if body.current_user_text:
        if not body.recent_messages:
            return ""
        previous_message = body.recent_messages[-1]
        if previous_message.role == "assistant" and previous_message.content.strip():
            return previous_message.content.strip()
        return ""

    last_user_index = None
    for index in range(len(body.recent_messages) - 1, -1, -1):
        message = body.recent_messages[index]
        if message.role == "user" and message.content.strip():
            last_user_index = index
            break

    if last_user_index in (None, 0):
        return ""

    previous_message = body.recent_messages[last_user_index - 1]
    if previous_message.role == "assistant" and previous_message.content.strip():
        return previous_message.content.strip()
    return ""


def _canonical_turn_text(text: str) -> str:
    return text.rstrip(" .!?")


def _literal_command_confidence(raw_text: str, text: str) -> float:
    if not text:
        return 0.0
    if any(text.startswith(prefix) for prefix in _COMMAND_PREFIXES):
        return 0.86
    if any(text.startswith(prefix) for prefix in _COMMAND_PHRASES):
        return 0.78
    first_word = text.split(" ", 1)[0]
    if first_word in {"rename", "change", "update", "fix", "add", "remove", "refactor", "create"}:
        return 0.82
    if raw_text.endswith("!") and len(text.split()) <= 6:
        return 0.45
    return 0.18


def _is_question(text: str) -> bool:
    return text.endswith("?") or any(text.startswith(f"{word} ") for word in _QUESTION_WORDS)


def _has_tense_markers(raw_text: str, text: str) -> bool:
    return (
        _contains_any(text, _DEBUG_MARKERS)
        or raw_text.count("!") >= 2
        or _ALL_CAPS_TOKEN.search(raw_text) is not None
    )


def _is_failure_report(text: str) -> bool:
    return any(
        marker in text
        for marker in ("broke", "broken", "failing", "failure", "error", "wrong", "crash")
    )


def _is_playful(text: str) -> bool:
    return _contains_any(text, _PLAYFUL_MARKERS)


def _is_brainstorm(text: str) -> bool:
    return _contains_any(text, _BRAINSTORM_MARKERS)


def _is_high_impact(text: str) -> bool:
    return _contains_any(text, _HIGH_IMPACT_MARKERS)


def _is_destructive_ambiguous(text: str) -> bool:
    if not _contains_any(text, _DESTRUCTIVE_MARKERS):
        return False
    return "this" in text or "it" in text or len(text.split()) <= 4


def _is_command(text: str, raw_text: str) -> bool:
    confidence = _literal_command_confidence(raw_text, text)
    if confidence >= 0.75 and not _is_question(text):
        return True
    return False


def _is_vent(text: str) -> bool:
    return _contains_any(text, _VENT_MARKERS)


def _reason_summary(kind: InteractionGovernanceKind, text: str, raw_text: str) -> list[str]:
    reasons: list[str] = []
    if kind == "ambiguous" and _is_destructive_ambiguous(text):
        reasons.extend(["destructive_phrase_detected", "target_missing"])
    if kind == "tense_debugging":
        reasons.append("tense_debugging_markers")
        if "prod" in text or "production" in text:
            reasons.append("possible_production_failure")
    if kind == "mistake_or_failure_report":
        reasons.append("failure_report_markers")
        if any(marker in text for marker in _CORRECTION_MARKERS):
            reasons.append("self_correction_markers")
    if kind == "high_impact_decision":
        reasons.append("high_impact_domain_markers")
    if kind == "brainstorm":
        reasons.append("brainstorm_markers")
    if kind == "joke_or_playful":
        reasons.append("playful_markers")
    if kind == "question":
        reasons.append("question_markers")
    if kind == "command":
        reasons.append("command_markers")
    if kind == "vent_or_expression":
        reasons.append("venting_markers")
    if not reasons:
        reasons.append("insufficient_signal")
    if raw_text.count("!") >= 2 or _ALL_CAPS_TOKEN.search(raw_text):
        reasons.append("elevated_urgency_markers")
    return reasons[:8]


def _persona_scope_hint(kind: InteractionGovernanceKind) -> str | None:
    if kind == "tense_debugging":
        return "technical_operator"
    if kind == "mistake_or_failure_report":
        return "supportive_listener"
    if kind == "high_impact_decision":
        return "careful_decider"
    if kind == "command":
        return "general_assistant"
    return None


def _is_clarification_request(text: str) -> bool:
    canonical = _canonical_turn_text(text)
    return any(
        canonical.startswith(prefix) for prefix in _CLARIFICATION_REQUEST_PREFIXES
    )


def _is_confirmation_response(
    text: str, body: InteractionGovernanceEvaluateRequest
) -> bool:
    if not _immediately_preceding_assistant_text(body).endswith("?"):
        return False
    return _canonical_turn_text(text) in _CONFIRMATION_RESPONSE_MARKERS


def _is_continuation(text: str, body: InteractionGovernanceEvaluateRequest) -> bool:
    if not _immediately_preceding_assistant_text(body):
        return False
    canonical = _canonical_turn_text(text)
    return canonical in _CONTINUATION_MARKERS


def _map_intent_class(
    kind: InteractionGovernanceKind,
    text: str,
    body: InteractionGovernanceEvaluateRequest,
) -> str:
    if _is_clarification_request(text):
        return "clarification_request"
    if _is_confirmation_response(text, body):
        return "confirmation_response"
    if _is_continuation(text, body):
        return "continuation"
    if kind == "question":
        return "information_request"
    if kind == "command":
        return "action_command"
    if kind == "vent_or_expression":
        return "venting_signal"
    if kind == "mistake_or_failure_report" and any(
        marker in text for marker in _CORRECTION_MARKERS
    ):
        return "correction"
    return "low_confidence_unclear"


def _history_confidence_band(
    body: InteractionGovernanceEvaluateRequest,
) -> str:
    candidate = body.history_followup_candidate
    if candidate is None:
        return "not_applicable"
    if candidate.source == "deterministic":
        return "high"
    if candidate.confidence >= HISTORY_FOLLOWUP_HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if candidate.confidence >= HISTORY_FOLLOWUP_CLARIFICATION_THRESHOLD:
        return "medium"
    return "low"


def _history_policy_result(
    body: InteractionGovernanceEvaluateRequest,
) -> HistoryFollowupPolicyResult:
    candidate = body.history_followup_candidate
    if candidate is None:
        return HistoryFollowupPolicyResult(
            status="not_applicable",
            history_lookup_allowed=False,
            new_verification_requested=False,
            new_verification_allowed_after_history_resolution=False,
            clarification_required=False,
            confidence_band="not_applicable",
            reason_codes=["no_candidate"],
        )

    projection = _HISTORY_INTENT_PROJECTION.get(candidate.intent)
    explanation_kind = projection[0] if projection else None
    acquisition_question = projection[1] if projection else None
    confidence_band = _history_confidence_band(body)
    common = {
        "intent": candidate.intent,
        "candidate_source": candidate.source,
        "target_mode": candidate.target_mode,
        "explanation_kind": explanation_kind,
        "acquisition_question": acquisition_question,
        "new_verification_requested": candidate.new_verification_requested,
        "confidence_band": confidence_band,
    }

    if candidate.target_mode == "explicit_reference":
        return HistoryFollowupPolicyResult(
            status="explicit_reference",
            history_lookup_allowed=False,
            new_verification_allowed_after_history_resolution=False,
            clarification_required=False,
            reason_codes=["explicit_reference_routed"],
            **common,
        )
    if candidate.intent == "not_history_followup":
        return HistoryFollowupPolicyResult(
            status="not_applicable",
            history_lookup_allowed=False,
            new_verification_allowed_after_history_resolution=False,
            clarification_required=False,
            reason_codes=["not_history_candidate"],
            **common,
        )
    if candidate.intent == "ambiguous_history_followup":
        return HistoryFollowupPolicyResult(
            status="clarification_required",
            history_lookup_allowed=False,
            new_verification_allowed_after_history_resolution=False,
            clarification_required=True,
            reason_codes=["ambiguous_candidate"],
            **common,
        )
    if candidate.source == "classifier" and confidence_band == "medium":
        return HistoryFollowupPolicyResult(
            status="clarification_required",
            history_lookup_allowed=False,
            new_verification_allowed_after_history_resolution=False,
            clarification_required=True,
            reason_codes=["classifier_confidence_requires_clarification"],
            **common,
        )
    if candidate.source == "classifier" and confidence_band == "low":
        return HistoryFollowupPolicyResult(
            status="rejected",
            history_lookup_allowed=False,
            new_verification_allowed_after_history_resolution=False,
            clarification_required=False,
            reason_codes=["classifier_confidence_rejected"],
            **common,
        )

    reason_code = (
        "deterministic_candidate_accepted"
        if candidate.source == "deterministic"
        else "classifier_candidate_accepted"
    )
    return HistoryFollowupPolicyResult(
        status="accepted",
        history_lookup_allowed=True,
        new_verification_allowed_after_history_resolution=(
            candidate.new_verification_requested
        ),
        clarification_required=False,
        reason_codes=[reason_code],
        **common,
    )


def _classify(body: InteractionGovernanceEvaluateRequest) -> InteractionGovernanceResult:
    raw_text = _latest_user_text(body)
    text = _normalize_text(raw_text)
    literal_command_confidence = _literal_command_confidence(raw_text, text)

    if not text:
        return InteractionGovernanceResult(
            interaction_kind="ambiguous",
            tension_level="low",
            literal_command_confidence=0.0,
            commentary_allowed=False,
            humor_allowed=False,
            clarifying_question_allowed=True,
            action_allowed=False,
            requires_confirmation=False,
            persona_scope_hint=None,
            privacy_sensitivity_hint="normal",
            response_posture="silent_or_minimal",
            confidence=0.2,
            reason_summary=["insufficient_signal"],
            history_followup_policy=_history_policy_result(body),
        )

    if _is_destructive_ambiguous(text):
        kind: InteractionGovernanceKind = "ambiguous"
        tension = "high" if _has_tense_markers(raw_text, text) or _is_vent(text) else "medium"
        posture = "tactical" if tension == "high" else "brief"
        confidence = 0.86
    elif _has_tense_markers(raw_text, text) and _is_failure_report(text):
        kind = "tense_debugging"
        tension = "high"
        posture = "tactical"
        confidence = 0.9
    elif _is_failure_report(text):
        kind = "mistake_or_failure_report"
        tension = "medium"
        posture = "supportive"
        confidence = 0.78
    elif _is_high_impact(text):
        kind = "high_impact_decision"
        tension = "medium"
        posture = "reflective"
        confidence = 0.76
    elif _is_brainstorm(text):
        kind = "brainstorm"
        tension = "low"
        posture = "reflective"
        confidence = 0.82
    elif _is_playful(text):
        kind = "joke_or_playful"
        tension = "low"
        posture = "playful"
        confidence = 0.84
    elif _is_question(text):
        kind = "question"
        tension = "low"
        posture = "direct"
        confidence = 0.8
    elif _is_command(text, raw_text):
        kind = "command"
        tension = "low"
        posture = "direct"
        confidence = 0.83
    elif _is_vent(text):
        kind = "vent_or_expression"
        tension = "medium"
        posture = "supportive"
        confidence = 0.72
    else:
        kind = "ambiguous"
        tension = "low"
        posture = "silent_or_minimal"
        confidence = 0.45

    commentary_allowed = False
    humor_allowed = kind == "joke_or_playful"
    clarifying_question_allowed = kind in {
        "tense_debugging",
        "mistake_or_failure_report",
        "high_impact_decision",
        "ambiguous",
    }
    action_allowed = False
    requires_confirmation = kind in {
        "tense_debugging",
        "mistake_or_failure_report",
        "high_impact_decision",
    } or _is_destructive_ambiguous(text)
    privacy_sensitivity_hint = "normal"
    if kind in {"tense_debugging", "mistake_or_failure_report"}:
        privacy_sensitivity_hint = "private"
    elif kind == "high_impact_decision":
        privacy_sensitivity_hint = "sensitive"

    return InteractionGovernanceResult(
        interaction_kind=kind,
        tension_level=tension,
        literal_command_confidence=literal_command_confidence,
        commentary_allowed=commentary_allowed,
        humor_allowed=humor_allowed,
        clarifying_question_allowed=clarifying_question_allowed,
        action_allowed=action_allowed,
        requires_confirmation=requires_confirmation,
        persona_scope_hint=_persona_scope_hint(kind),
        privacy_sensitivity_hint=privacy_sensitivity_hint,
        response_posture=posture,
        confidence=confidence,
        reason_summary=_reason_summary(kind, text, raw_text),
        history_followup_policy=_history_policy_result(body),
    )


def evaluate_interaction_governance(
    body: InteractionGovernanceEvaluateRequest,
) -> InteractionGovernanceEvaluateResponse:
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
            surface_session_id=body.surface_session_id,
            active_mode=body.active_mode,
        )
        runtime_session_id = session.runtime_session_id

    result = _classify(body)
    normalized_text = _normalize_text(_latest_user_text(body))

    if body.runtime_turn_id:
        history_policy = result.history_followup_policy
        intent_class = _map_intent_class(
            result.interaction_kind, normalized_text, body
        )
        if history_policy.status == "accepted" and history_policy.intent is not None:
            intent_class = history_policy.intent
        elif history_policy.status == "clarification_required":
            intent_class = "ambiguous_history_followup"
        update_runtime_turn_intent_class(
            runtime_session_id=runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
            intent_class=intent_class,
        )

    record_runtime_event(
        runtime_session_id=runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        event_type="interaction_governance_evaluated",
        event_payload_json={
            "request_id": body.request_id,
            "interaction_kind": result.interaction_kind,
            "response_posture": result.response_posture,
            "commentary_allowed": result.commentary_allowed,
            "humor_allowed": result.humor_allowed,
            "action_allowed": result.action_allowed,
            "requires_confirmation": result.requires_confirmation,
            "reason_summary": result.reason_summary,
            "history_followup_policy": result.history_followup_policy.model_dump(),
        },
    )

    return InteractionGovernanceEvaluateResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        runtime_session_id=runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        result=result,
    )
