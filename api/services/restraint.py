from __future__ import annotations

import re
from hashlib import sha256

from models import (
    InteractionGovernanceKind,
    RestraintEvaluateRequest,
    RestraintEvaluateResponse,
    RestraintPolicy,
    RestraintResult,
)
from services.runtime_state import (
    record_runtime_event,
    resolve_runtime_session,
    runtime_session_by_id,
    update_runtime_turn_restraint_policy,
)

_QUESTION_WORDS = ("what", "why", "how", "when", "where", "who", "which", "should")
_DIRECT_COMMAND_PREFIXES = (
    "review ",
    "give me ",
    "check ",
    "show ",
    "list ",
    "fix ",
    "rename ",
    "update ",
    "write ",
)
_AMBIGUOUS_MARKERS = ("fix this", "what now", "check this", "handle this")
_TENSE_DEBUG_MARKERS = (
    "broke",
    "broken",
    "failing",
    "failure",
    "outage",
    "incident",
    "prod",
    "production",
    "server",
    "crash",
    "crashed",
)
_VENT_MARKERS = (
    "exhausting",
    "tired",
    "drained",
    "overwhelmed",
    "frustrated",
    "this sucks",
    "i am so tired",
    "i'm so tired",
)
_RETRIEVAL_REQUEST_MARKERS = (
    "remember",
    "earlier",
    "before",
    "previous",
    "from memory",
    "from my notes",
    "past context",
    "history",
)
_PERSONALIZATION_REQUEST_MARKERS = (
    "for me personally",
    "based on my preferences",
    "for my situation",
    "use my context",
    "personalize",
)
_PROACTIVE_REQUEST_MARKERS = (
    "remind me",
    "follow up",
    "nudge me",
    "check back",
)
_GUIDANCE_MARKERS = (
    "production",
    "prod",
    "database",
    "security",
    "credentials",
    "legal",
    "medical",
    "finance",
    "financial",
    "tax",
    "taxes",
)
_ALL_CAPS_TOKEN = re.compile(r"\b[A-Z]{4,}\b")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _latest_user_text(body: RestraintEvaluateRequest) -> str:
    if body.current_user_text:
        return body.current_user_text.strip()
    for message in reversed(body.recent_messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    return ""


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _is_question(text: str) -> bool:
    return text.endswith("?") or any(text.startswith(f"{word} ") for word in _QUESTION_WORDS)


def _is_direct_request(text: str) -> bool:
    return any(text.startswith(prefix) for prefix in _DIRECT_COMMAND_PREFIXES)


def _is_ambiguous(text: str) -> bool:
    return not text or text in _AMBIGUOUS_MARKERS or len(text.split()) <= 2


def _has_specific_debug_context(raw_text: str, text: str) -> bool:
    return (
        _contains_any(text, _TENSE_DEBUG_MARKERS)
        and len(text.split()) >= 5
        and ("prod" in text or "production" in text or "server" in text or _ALL_CAPS_TOKEN.search(raw_text))
    )


def _infer_interaction_kind(raw_text: str, text: str) -> InteractionGovernanceKind:
    if _contains_any(text, _TENSE_DEBUG_MARKERS):
        return "tense_debugging"
    if _contains_any(text, _VENT_MARKERS):
        return "vent_or_expression"
    if _is_ambiguous(text):
        return "ambiguous"
    if _is_direct_request(text):
        return "command"
    if _is_question(text):
        return "question"
    return "ambiguous"


def _trace_ref(request_id: str, policy: str, domains: list[str], reason: str) -> str:
    material = f"{request_id}:{policy}:{','.join(domains)}:{reason}"
    digest = sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"rtrace_{digest}"


def _result(
    *,
    request_id: str,
    restraint_policy: RestraintPolicy,
    domains: set[str],
    reason: str,
    prompt_overlay: str,
    confidence: float,
    reason_summary: list[str],
    retrieval_suppressed: bool,
    personalization_suppressed: bool,
    proactive_output_suppressed: bool,
    brevity_preferred: bool,
    clarification_preferred: bool,
) -> RestraintResult:
    sorted_domains = sorted(domains)
    return RestraintResult(
        restraint_policy=restraint_policy,
        domains=sorted_domains,
        reason=reason,
        prompt_overlay=prompt_overlay,
        trace_ref=_trace_ref(request_id, restraint_policy, sorted_domains, reason),
        confidence=confidence,
        reason_summary=reason_summary[:8],
        retrieval_suppressed=retrieval_suppressed,
        personalization_suppressed=personalization_suppressed,
        proactive_output_suppressed=proactive_output_suppressed,
        brevity_preferred=brevity_preferred,
        clarification_preferred=clarification_preferred,
    )


def evaluate_restraint(body: RestraintEvaluateRequest) -> RestraintEvaluateResponse:
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
    interaction_kind = body.interaction_kind or _infer_interaction_kind(raw_text, text)

    retrieval_requested = _contains_any(text, _RETRIEVAL_REQUEST_MARKERS)
    personalization_requested = _contains_any(text, _PERSONALIZATION_REQUEST_MARKERS)
    proactive_requested = _contains_any(text, _PROACTIVE_REQUEST_MARKERS)
    required_guidance = _contains_any(text, _GUIDANCE_MARKERS)

    retrieval_suppressed = not retrieval_requested
    personalization_suppressed = not personalization_requested
    proactive_output_suppressed = not proactive_requested

    reason_summary: list[str] = []
    if retrieval_suppressed:
        reason_summary.append("retrieval_not_requested")
    if personalization_suppressed:
        reason_summary.append("personal_framing_not_requested")
    if proactive_output_suppressed:
        reason_summary.append("proactive_not_requested")
    if required_guidance:
        reason_summary.append("required_guidance_preserved")

    if interaction_kind == "ambiguous":
        result = _result(
            request_id=body.request_id,
            restraint_policy="ask_clarifying_question",
            domains={"output", "action", "retrieval"},
            reason="underspecified_request",
            prompt_overlay="Ask one clarifying question instead of assuming details.",
            confidence=0.9 if text else 0.45,
            reason_summary=["underspecified_request", *reason_summary],
            retrieval_suppressed=True,
            personalization_suppressed=True,
            proactive_output_suppressed=proactive_output_suppressed,
            brevity_preferred=False,
            clarification_preferred=True,
        )
    elif interaction_kind == "tense_debugging":
        if _has_specific_debug_context(raw_text, text):
            result = _result(
                request_id=body.request_id,
                restraint_policy="short_answer",
                domains={"output", "affect", "action"},
                reason="tense_debugging_tactical_restraint",
                prompt_overlay=(
                    "Keep the response brief, tactical, and avoid affective intensification."
                ),
                confidence=0.91,
                reason_summary=["tense_debugging_tactical_restraint", *reason_summary],
                retrieval_suppressed=retrieval_suppressed,
                personalization_suppressed=True,
                proactive_output_suppressed=proactive_output_suppressed,
                brevity_preferred=True,
                clarification_preferred=False,
            )
        else:
            result = _result(
                request_id=body.request_id,
                restraint_policy="ask_clarifying_question",
                domains={"output", "affect", "action"},
                reason="tense_debugging_needs_clarification",
                prompt_overlay="Ask one clarifying question before assuming the failure state.",
                confidence=0.82,
                reason_summary=["tense_debugging_needs_clarification", *reason_summary],
                retrieval_suppressed=True,
                personalization_suppressed=True,
                proactive_output_suppressed=proactive_output_suppressed,
                brevity_preferred=False,
                clarification_preferred=True,
            )
    elif interaction_kind == "vent_or_expression":
        result = _result(
            request_id=body.request_id,
            restraint_policy="defer_expansion",
            domains={"output", "personalization", "affect"},
            reason="expression_without_explicit_problem_solving_request",
            prompt_overlay=(
                "Respond briefly and do not turn expression into a full problem-solving plan."
            ),
            confidence=0.84,
            reason_summary=[
                "expression_without_explicit_problem_solving_request",
                *reason_summary,
            ],
            retrieval_suppressed=retrieval_suppressed,
            personalization_suppressed=True,
            proactive_output_suppressed=proactive_output_suppressed,
            brevity_preferred=True,
            clarification_preferred=False,
        )
    elif _is_direct_request(text) or interaction_kind == "command":
        result = _result(
            request_id=body.request_id,
            restraint_policy="short_answer",
            domains={"output"},
            reason="direct_tactical_request",
            prompt_overlay="Keep the response brief and avoid unnecessary elaboration.",
            confidence=0.88,
            reason_summary=["direct_tactical_request", *reason_summary],
            retrieval_suppressed=retrieval_suppressed,
            personalization_suppressed=True,
            proactive_output_suppressed=proactive_output_suppressed,
            brevity_preferred=True,
            clarification_preferred=False,
        )
    elif retrieval_suppressed and not text:
        result = _result(
            request_id=body.request_id,
            restraint_policy="do_not_retrieve",
            domains={"retrieval"},
            reason="retrieval_not_requested",
            prompt_overlay="Do not retrieve or mention memory unless explicitly needed.",
            confidence=0.6,
            reason_summary=["retrieval_not_requested", *reason_summary],
            retrieval_suppressed=True,
            personalization_suppressed=personalization_suppressed,
            proactive_output_suppressed=proactive_output_suppressed,
            brevity_preferred=False,
            clarification_preferred=False,
        )
    else:
        domains = {"output"}
        if retrieval_suppressed:
            domains.add("retrieval")
        if personalization_suppressed:
            domains.add("personalization")
        if proactive_output_suppressed:
            domains.add("proactive")
        result = _result(
            request_id=body.request_id,
            restraint_policy="answer_normally",
            domains=domains,
            reason="normal_response_with_restraint",
            prompt_overlay="",
            confidence=0.74 if text else 0.5,
            reason_summary=["normal_response_with_restraint", *reason_summary],
            retrieval_suppressed=retrieval_suppressed,
            personalization_suppressed=personalization_suppressed,
            proactive_output_suppressed=proactive_output_suppressed,
            brevity_preferred=False,
            clarification_preferred=False,
        )

    if body.runtime_turn_id:
        update_runtime_turn_restraint_policy(
            runtime_session_id=runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
            restraint_policy=result.restraint_policy,
        )

    record_runtime_event(
        runtime_session_id=runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        event_type="restraint_evaluated",
        event_payload_json={
            "request_id": body.request_id,
            "restraint_policy": result.restraint_policy,
            "domains": result.domains,
            "reason": result.reason,
            "confidence": result.confidence,
            "reason_summary": result.reason_summary,
            "retrieval_suppressed": result.retrieval_suppressed,
            "personalization_suppressed": result.personalization_suppressed,
            "proactive_output_suppressed": result.proactive_output_suppressed,
        },
    )

    return RestraintEvaluateResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        runtime_session_id=runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        result=result,
    )
