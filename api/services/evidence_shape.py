from __future__ import annotations

import json
import re
from hashlib import sha256

from models import (
    EvidenceShapeDerivationStatus,
    EvidenceShapeDeriveRequest,
    EvidenceShapeDeriveResponse,
    EvidenceShapeReasonCode,
    EvidenceShapeResult,
    EvidenceTaskShape,
)
from services.runtime_state import (
    record_runtime_event,
    runtime_session_by_id,
    validate_runtime_turn_session,
)

_EVIDENCE_OBJECTS = re.compile(
    r"\b(?:evidence|sources?|records?|files?|documents?|reports?|logs?|data|"
    r"checklists?|requirements?|repositories?|artifacts?|tool outputs?|"
    r"integration records?|world state)\b",
    re.IGNORECASE,
)
_ALTERNATIVE_OBJECTS = re.compile(
    r"\b(?:candidates?|options?|alternatives?|versions?)\b",
    re.IGNORECASE,
)
_DIRECT_EVIDENCE_OPERATORS = re.compile(
    r"\b(?:check(?!\s+out\b)|verify|inspect|audit|research|search|look\s+up|"
    r"trace|ground)\b",
    re.IGNORECASE,
)
_BOUNDED_EVIDENCE_REFERENCE = re.compile(
    r"\b(?:this|that|these|those|the|my|our|your|their|its|available|declared|"
    r"current|specific|given)\b(?:\s+[a-z][a-z-]*){0,4}\s+"
    r"(?:evidence|sources?|records?|files?|documents?|reports?|logs?|data|"
    r"checklists?|requirements?|repositories?|artifacts?|tool outputs?|"
    r"integration records?|world state)\b",
    re.IGNORECASE,
)
_BOUNDED_CONTENT_QUERY = re.compile(
    r"\b(?:summarize|review|examine)\b|"
    r"\bwhat\s+(?:does|do)\b.{0,120}\b(?:say|state|show|contain|report)\b|"
    r"\bwhat\s+(?:is|are|was|were)\b.{0,120}\b"
    r"(?:recorded|listed|documented|reported|shown|contained)\b|"
    r"\bwhich\b.{0,120}\b(?:is|are|was|were)\b.{0,80}\b"
    r"(?:recorded|listed|documented|reported|shown|contained)\b",
    re.IGNORECASE,
)
_CREATIVE_OR_CASUAL = re.compile(
    r"\b(?:write|finish|complete)\s+(?:a|the|this|my)?\s*"
    r"(?:poem|story|sentence|joke)|\brecommend\s+(?:a|an)\s+(?:funny|silly)\b",
    re.IGNORECASE,
)
_UNIVERSAL = re.compile(
    r"\b(?:all|every|entire|complete|fully|full\s+compliance|whole)\b",
    re.IGNORECASE,
)
_BOUNDED_COLLECTION = re.compile(
    r"\b(?:requirements?|items?|records?|documents?|reports?|logs?|sources?|"
    r"files?|checklists?|inventor(?:y|ies)|sets?|scope|coverage|implementations?)\b",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|differences?\s+between|"
    r"differ(?:ence|ences|ent|ently)?|across\s+the\s+two)\b",
    re.IGNORECASE,
)
_CONTRADICTION = re.compile(
    r"\b(?:contradict(?:ion|ions|ory|s|ed)?|conflict(?:ing|s|ed)?|"
    r"inconsisten(?:t|cy|cies)|disagree(?:ment|ments|s|d)?|counterevidence)\b",
    re.IGNORECASE,
)
_ABSENCE = re.compile(
    r"\b(?:no\s+(?:record|evidence)|nothing\s+(?:was\s+)?found|does\s+not\s+exist|"
    r"any\s+missing|what\s+was\s+not\s+covered|which\s+(?:items?\s+)?were\s+not\s+"
    r"examined|none\s+(?:exist|found|recorded)|missing\s+coverage)\b",
    re.IGNORECASE,
)
_HISTORICAL = re.compile(
    r"\b(?:reconstruct|timeline|chronology|sequence|history\s+of|what\s+happened|"
    r"changed\s+over\s+time|before\s+and\s+after)\b",
    re.IGNORECASE,
)
_DECISION = re.compile(
    r"\b(?:which\s+should\s+I\s+choose|which\s+option\s+is\s+better|"
    r"recommend\s+based\s+on|evaluate\s+the\s+trade-?offs?|decide\s+between|"
    r"choose\s+between)\b",
    re.IGNORECASE,
)

_SPECIALIZED_REASON: dict[EvidenceTaskShape, EvidenceShapeReasonCode] = {
    "bounded_exhaustive_review": "exhaustive_scope_requested",
    "cross_source_comparison": "comparison_requested",
    "contradiction_review": "contradiction_requested",
    "absence_or_coverage_check": "absence_scope_requested",
    "historical_reconstruction": "historical_reconstruction_requested",
    "recommendation_or_decision_support": "decision_support_requested",
}
_COMPATIBLE_COMBINATIONS: dict[frozenset[EvidenceTaskShape], EvidenceTaskShape] = {
    frozenset(
        {"recommendation_or_decision_support", "cross_source_comparison"}
    ): "recommendation_or_decision_support",
    frozenset({"contradiction_review", "cross_source_comparison"}): (
        "contradiction_review"
    ),
    frozenset({"bounded_exhaustive_review", "contradiction_review"}): (
        "bounded_exhaustive_review"
    ),
    frozenset({"absence_or_coverage_check", "bounded_exhaustive_review"}): (
        "absence_or_coverage_check"
    ),
}


def _explicit_evidence_language(
    text: str,
    *,
    specialized: set[EvidenceTaskShape],
) -> bool:
    has_evidence_object = bool(_EVIDENCE_OBJECTS.search(text))
    if _DIRECT_EVIDENCE_OPERATORS.search(text) and has_evidence_object:
        return True
    if (
        has_evidence_object
        and _BOUNDED_EVIDENCE_REFERENCE.search(text)
        and _BOUNDED_CONTENT_QUERY.search(text)
    ):
        return True
    if not specialized:
        return False
    return bool(
        has_evidence_object
        or _ALTERNATIVE_OBJECTS.search(text)
        or _BOUNDED_COLLECTION.search(text)
    )


def _specialized_shapes(text: str) -> set[EvidenceTaskShape]:
    shapes: set[EvidenceTaskShape] = set()
    if _UNIVERSAL.search(text) and _BOUNDED_COLLECTION.search(text):
        shapes.add("bounded_exhaustive_review")
    if _COMPARISON.search(text):
        shapes.add("cross_source_comparison")
    if _CONTRADICTION.search(text):
        shapes.add("contradiction_review")
    if _ABSENCE.search(text):
        shapes.add("absence_or_coverage_check")
    if _HISTORICAL.search(text):
        shapes.add("historical_reconstruction")
    if _DECISION.search(text):
        shapes.add("recommendation_or_decision_support")
    return shapes


def _materiality_reasons(
    body: EvidenceShapeDeriveRequest,
    *,
    explicit_evidence: bool,
) -> set[EvidenceShapeReasonCode]:
    context = body.task_context
    reasons: set[EvidenceShapeReasonCode] = set()
    if context.evidence_input_kinds:
        reasons.add("source_context_present")
    if context.external_verification_required:
        reasons.add("external_verification_required")
    if context.freshness_sensitive:
        reasons.add("freshness_sensitive")
    if context.high_stakes_accuracy_required:
        reasons.add("high_stakes_accuracy_required")
    if explicit_evidence:
        reasons.add("explicit_evidence_language")
    return reasons


def _safe_summary(status: EvidenceShapeDerivationStatus) -> str:
    return {
        "derived": (
            "Evidence-scope planning applies and a bounded acquisition mode was "
            "identified."
        ),
        "not_applicable": (
            "This request does not currently require evidence-scope planning."
        ),
        "ambiguous": (
            "The evidence task combines or lacks enough scope information, so it "
            "must be narrowed before planning."
        ),
    }[status]


def _derivation_identity(
    body: EvidenceShapeDeriveRequest,
    *,
    question_digest: str,
    status: EvidenceShapeDerivationStatus,
    task_shape: EvidenceTaskShape | None,
    candidates: list[EvidenceTaskShape],
    evidence_scope_material: bool,
    clarification_required: bool,
    reasons: list[EvidenceShapeReasonCode],
) -> str:
    context = body.task_context.model_dump(mode="json")
    context["evidence_input_kinds"] = sorted(context["evidence_input_kinds"])
    material = {
        "request_id": body.request_id,
        "owner_id": body.owner_id,
        "conversation_id": body.conversation_id,
        "surface": body.surface,
        "runtime_session_id": body.runtime_session_id,
        "runtime_turn_id": body.runtime_turn_id,
        "question_anchor_digest": question_digest,
        "interaction_kind": body.interaction_kind,
        "task_context": context,
        "derivation_status": status,
        "task_shape": task_shape,
        "candidate_task_shapes": candidates,
        "evidence_scope_material": evidence_scope_material,
        "clarification_required": clarification_required,
        "reason_codes": reasons,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"evidence_shape_{sha256(encoded.encode('utf-8')).hexdigest()[:32]}"


def _build_result(
    body: EvidenceShapeDeriveRequest,
    *,
    status: EvidenceShapeDerivationStatus,
    task_shape: EvidenceTaskShape | None,
    candidates: list[EvidenceTaskShape],
    reasons: set[EvidenceShapeReasonCode],
) -> EvidenceShapeResult:
    sorted_candidates = sorted(set(candidates))
    sorted_reasons = sorted(reasons)
    material = status != "not_applicable"
    clarification = status == "ambiguous"
    question_digest = f"sha256:{sha256(body.task_text.encode('utf-8')).hexdigest()}"
    return EvidenceShapeResult(
        derivation_id=_derivation_identity(
            body,
            question_digest=question_digest,
            status=status,
            task_shape=task_shape,
            candidates=sorted_candidates,
            evidence_scope_material=material,
            clarification_required=clarification,
            reasons=sorted_reasons,
        ),
        question_anchor=body.task_text,
        question_anchor_digest=question_digest,
        derivation_status=status,
        task_shape=task_shape,
        candidate_task_shapes=sorted_candidates,
        evidence_scope_material=material,
        clarification_required=clarification,
        reason_codes=sorted_reasons,
        user_safe_summary=_safe_summary(status),
    )


def _derive_result(body: EvidenceShapeDeriveRequest) -> EvidenceShapeResult:
    specialized = _specialized_shapes(body.task_text)
    explicit_evidence = _explicit_evidence_language(
        body.task_text,
        specialized=specialized,
    )
    distinct_evidence_request = explicit_evidence
    context = body.task_context
    reasons = _materiality_reasons(body, explicit_evidence=explicit_evidence)
    context_material = bool(
        context.evidence_input_kinds
        or context.external_verification_required
        or context.freshness_sensitive
        or context.high_stakes_accuracy_required
        or context.continuation_of_prior_evidence_task
    )
    evidence_material = context_material or explicit_evidence
    non_evidence_interaction = body.interaction_kind in {
        "joke_or_playful",
        "vent_or_expression",
    }
    if non_evidence_interaction and not distinct_evidence_request:
        evidence_material = False
        reasons.clear()

    if not evidence_material:
        reasons.add("ordinary_chat_without_material_evidence_scope")
        if non_evidence_interaction or _CREATIVE_OR_CASUAL.search(body.task_text):
            reasons.add("non_evidence_interaction")
        return _build_result(
            body,
            status="not_applicable",
            task_shape=None,
            candidates=[],
            reasons=reasons,
        )

    for shape in specialized:
        reasons.add(_SPECIALIZED_REASON[shape])
    candidates = sorted(specialized)

    if len(specialized) > 1:
        compatible = _COMPATIBLE_COMBINATIONS.get(frozenset(specialized))
        if compatible is None:
            reasons.add("multiple_incompatible_shapes")
            return _build_result(
                body,
                status="ambiguous",
                task_shape=None,
                candidates=candidates,
                reasons=reasons,
            )
        selected = compatible
    elif len(specialized) == 1:
        selected = next(iter(specialized))
    elif context.continuation_of_prior_evidence_task:
        selected = context.prior_task_shape
        reasons.add("prior_shape_inherited")
    elif body.interaction_kind == "ambiguous" and not explicit_evidence:
        reasons.add("ambiguous_interaction_without_shape_signal")
        return _build_result(
            body,
            status="ambiguous",
            task_shape=None,
            candidates=[],
            reasons=reasons,
        )
    elif body.interaction_kind == "brainstorm" and not explicit_evidence:
        reasons.add("ambiguous_interaction_without_shape_signal")
        return _build_result(
            body,
            status="ambiguous",
            task_shape=None,
            candidates=[],
            reasons=reasons,
        )
    else:
        selected = "targeted_lookup"
        reasons.add("targeted_lookup_derived")

    return _build_result(
        body,
        status="derived",
        task_shape=selected,
        candidates=[selected],
        reasons=reasons,
    )


def derive_evidence_shape(
    body: EvidenceShapeDeriveRequest,
) -> EvidenceShapeDeriveResponse:
    session = runtime_session_by_id(body.runtime_session_id)
    if session is None:
        raise RuntimeError("runtime_session_not_found")
    if (
        session.owner_id != body.owner_id
        or session.conversation_id != body.conversation_id
        or session.surface != body.surface
    ):
        raise RuntimeError("runtime_session_mismatch")
    validate_runtime_turn_session(
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
    )

    result = _derive_result(body)
    record_runtime_event(
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        event_type="evidence_shape_derived",
        event_payload_json={
            "request_id": body.request_id,
            "runtime_session_id": body.runtime_session_id,
            "runtime_turn_id": body.runtime_turn_id,
            "derivation_id": result.derivation_id,
            "question_anchor_digest": result.question_anchor_digest,
            "interaction_kind": body.interaction_kind,
            "derivation_status": result.derivation_status,
            "task_shape": result.task_shape,
            "candidate_task_shapes": result.candidate_task_shapes,
            "evidence_scope_material": result.evidence_scope_material,
            "clarification_required": result.clarification_required,
            "evidence_input_count": len(body.task_context.evidence_input_kinds),
            "continuation_of_prior_evidence_task": (
                body.task_context.continuation_of_prior_evidence_task
            ),
            "reason_codes": result.reason_codes,
        },
    )
    return EvidenceShapeDeriveResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        result=result,
    )
