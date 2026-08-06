from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Any, cast

from models import (
    EvidenceAcquisitionPremise,
    EvidenceAnswerConstraint,
    EvidenceConclusionDisposition,
    EvidenceNextStep,
    EvidenceNextStepReasonCode,
    EvidenceNextStepResult,
    EvidenceNextStepSelectRequest,
    EvidenceNextStepSelectResponse,
    EvidenceProviderDisposition,
    EvidenceReacquisitionGuard,
    EvidenceRequirementEvaluation,
    EvidenceSufficiencyStatus,
    EvidenceTaskShape,
    RuntimeEvent,
)
from services.evidence_planning import evidence_acquisition_premise_digest
from services.evidence_sufficiency import evaluated_requirements_digest
from services.runtime_state import (
    get_runtime_session,
    record_runtime_event,
    runtime_session_by_id,
    validate_runtime_turn_session,
)

_SUFFICIENCY_STATUSES = {
    "sufficient_for_declared_scope",
    "sufficient_with_limitations",
    "insufficient",
    "unknown",
}
_TASK_SHAPES = {
    "targeted_lookup",
    "bounded_exhaustive_review",
    "cross_source_comparison",
    "contradiction_review",
    "absence_or_coverage_check",
    "historical_reconstruction",
    "recommendation_or_decision_support",
}
_INTERACTION_KINDS = {
    "command",
    "question",
    "brainstorm",
    "joke_or_playful",
    "vent_or_expression",
    "mistake_or_failure_report",
    "tense_debugging",
    "high_impact_decision",
    "ambiguous",
}
_SHAPE_REASON_CODES = {
    "source_context_present",
    "external_verification_required",
    "freshness_sensitive",
    "high_stakes_accuracy_required",
    "explicit_evidence_language",
    "targeted_lookup_derived",
    "exhaustive_scope_requested",
    "comparison_requested",
    "contradiction_requested",
    "absence_scope_requested",
    "historical_reconstruction_requested",
    "decision_support_requested",
    "prior_shape_inherited",
    "ordinary_chat_without_material_evidence_scope",
    "non_evidence_interaction",
    "ambiguous_interaction_without_shape_signal",
    "multiple_incompatible_shapes",
}
_ANSWER_CONSTRAINTS = {
    "qualify_conclusion",
    "disclose_limitations",
    "identify_unexamined_scope",
    "additional_acquisition_or_clarification_required",
    "withhold_unqualified_conclusion",
    "withhold_exhaustive_conclusion",
    "withhold_absence_conclusion",
    "withhold_contradiction_sensitive_conclusion",
}
_SCOPE_REQUIREMENT_KINDS = {
    "authoritative_inventory",
    "complete_scope_coverage",
    "selected_source_coverage",
    "structured_absence_check",
    "contradiction_search",
    "counterevidence_coverage",
    "historical_scope",
    "historical_sequence_coverage",
    "candidate_evidence_coverage",
    "cross_source_comparison",
}
_ADMINISTRATIVE_REQUIREMENT_KINDS = {
    "authoritative_inventory",
    "context_delivery",
    "no_material_truncation",
}
_REASON_CODE_ORDER: tuple[EvidenceNextStepReasonCode, ...] = (
    "declared_scope_sufficient",
    "optional_limitations_remain",
    "material_uncertainty_requires_clarification",
    "changed_acquisition_premise_available",
    "unchanged_acquisition_premise",
    "acquisition_premise_already_selected",
    "substantive_partial_evidence_available",
    "unexamined_material_scope",
    "unsupported_conclusion_withheld",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalized_evaluations(
    evaluations: list[EvidenceRequirementEvaluation],
) -> list[EvidenceRequirementEvaluation]:
    return sorted(
        evaluations,
        key=lambda item: (
            item.requirement_id,
            item.requirement_kind,
            item.criticality,
            item.effective_outcome,
        ),
    )


def _premise_digest(premise: EvidenceAcquisitionPremise) -> str:
    return evidence_acquisition_premise_digest(
        question_anchor_digest=premise.question_anchor_digest,
        task_shape=premise.task_shape,
        declared_scope=premise.declared_scope,
        source_inventory=premise.source_inventory,
        selected_strategies=premise.selected_strategies,
    )


def _selection_identity(
    body: EvidenceNextStepSelectRequest,
    *,
    requirements_digest: str,
    current_premise_digest: str,
    proposed_premise_digest: str | None,
    clarification_target: str | None,
) -> str:
    material = {
        "request_id": body.request_id,
        "owner_id": body.owner_id,
        "conversation_id": body.conversation_id,
        "surface": body.surface,
        "runtime_session_id": body.runtime_session_id,
        "runtime_turn_id": body.runtime_turn_id,
        "evaluation_id": body.evaluation_id,
        "evidence_plan_id": body.evidence_plan_id,
        "acquisition_manifest_id": body.acquisition_manifest_id,
        "evaluated_requirements_digest": requirements_digest,
        "current_premise_digest": current_premise_digest,
        "proposed_premise_digest": proposed_premise_digest,
        "clarification_target": clarification_target,
    }
    digest = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    return f"evidence_next_step_{digest[:32]}"


def _associated_sufficiency_event(
    events: list[RuntimeEvent],
    body: EvidenceNextStepSelectRequest,
) -> dict[str, Any]:
    candidates = [
        event
        for event in events
        if event.event_type == "evidence_sufficiency_evaluated"
        and event.event_payload_json.get("evaluation_id") == body.evaluation_id
    ]
    if not candidates:
        raise RuntimeError("evidence_sufficiency_evaluation_not_found")

    matches = [
        event
        for event in candidates
        if event.runtime_turn_id == body.runtime_turn_id
        and event.event_payload_json.get("runtime_session_id")
        == body.runtime_session_id
        and event.event_payload_json.get("runtime_turn_id") == body.runtime_turn_id
        and event.event_payload_json.get("evidence_plan_id") == body.evidence_plan_id
        and event.event_payload_json.get("acquisition_manifest_id")
        == body.acquisition_manifest_id
    ]
    if not matches:
        raise RuntimeError("evidence_sufficiency_association_mismatch")
    payload = matches[0].event_payload_json

    status = payload.get("sufficiency_status")
    task_shape = payload.get("task_shape")
    constraints = payload.get("answer_constraints")
    requirements_digest = payload.get("evaluated_requirements_digest")
    if (
        status not in _SUFFICIENCY_STATUSES
        or task_shape not in _TASK_SHAPES
        or not isinstance(constraints, list)
        or any(constraint not in _ANSWER_CONSTRAINTS for constraint in constraints)
        or not isinstance(requirements_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", requirements_digest) is None
    ):
        raise RuntimeError("evidence_sufficiency_event_invalid")
    return payload


def _associated_plan_event(
    events: list[RuntimeEvent],
    body: EvidenceNextStepSelectRequest,
    *,
    task_shape: EvidenceTaskShape,
) -> dict[str, Any]:
    candidates = [
        event
        for event in events
        if event.event_type == "evidence_plan_compiled"
        and event.event_payload_json.get("plan_id") == body.evidence_plan_id
    ]
    matches = [
        event
        for event in candidates
        if event.runtime_turn_id == body.runtime_turn_id
        and event.event_payload_json.get("runtime_session_id")
        == body.runtime_session_id
        and event.event_payload_json.get("runtime_turn_id") == body.runtime_turn_id
        and event.event_payload_json.get("task_shape") == task_shape
    ]
    if len(matches) != 1:
        raise RuntimeError("current_acquisition_premise_mismatch")
    payload = matches[0].event_payload_json
    premise_digest = payload.get("acquisition_premise_digest")
    question_anchor_digest = payload.get("question_anchor_digest")
    if (
        not isinstance(premise_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", premise_digest) is None
        or not isinstance(question_anchor_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", question_anchor_digest) is None
    ):
        raise RuntimeError("current_acquisition_premise_mismatch")
    return payload


def _associated_shape_allows_advisory(
    events: list[RuntimeEvent],
    body: EvidenceNextStepSelectRequest,
    *,
    plan_event: dict[str, Any],
    task_shape: EvidenceTaskShape,
) -> bool:
    if task_shape != "targeted_lookup":
        return False

    plan_positions = [
        index
        for index, event in enumerate(events)
        if event.event_type == "evidence_plan_compiled"
        and event.runtime_session_id == body.runtime_session_id
        and event.runtime_turn_id == body.runtime_turn_id
        and event.event_payload_json.get("runtime_session_id")
        == body.runtime_session_id
        and event.event_payload_json.get("runtime_turn_id") == body.runtime_turn_id
        and event.event_payload_json.get("plan_id") == body.evidence_plan_id
    ]
    if len(plan_positions) != 1:
        return False

    question_anchor_digest = plan_event["question_anchor_digest"]
    associated_shapes = [
        (index, event)
        for index, event in enumerate(events)
        if index < plan_positions[0]
        and event.event_type == "evidence_shape_derived"
        and event.runtime_session_id == body.runtime_session_id
        and event.runtime_turn_id == body.runtime_turn_id
        and event.event_payload_json.get("runtime_session_id")
        == body.runtime_session_id
        and event.event_payload_json.get("runtime_turn_id") == body.runtime_turn_id
        and event.event_payload_json.get("question_anchor_digest")
        == question_anchor_digest
    ]
    if len(associated_shapes) != 1:
        return False

    shape_position, shape_event = associated_shapes[0]
    payload = shape_event.event_payload_json
    derivation_id = payload.get("derivation_id")
    interaction_kind = payload.get("interaction_kind")
    reason_codes = payload.get("reason_codes")
    if (
        not isinstance(derivation_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,119}", derivation_id)
        is None
        or interaction_kind not in _INTERACTION_KINDS
        or payload.get("derivation_status") != "derived"
        or payload.get("task_shape") != "targeted_lookup"
        or payload.get("candidate_task_shapes") != ["targeted_lookup"]
        or payload.get("evidence_scope_material") is not True
        or payload.get("clarification_required") is not False
        or not isinstance(reason_codes, list)
        or reason_codes != sorted(set(reason_codes))
        or any(reason not in _SHAPE_REASON_CODES for reason in reason_codes)
        or "targeted_lookup_derived" not in reason_codes
    ):
        return False

    associated_governance = [
        (index, event)
        for index, event in enumerate(events)
        if event.event_type == "interaction_governance_evaluated"
        and event.runtime_session_id == body.runtime_session_id
        and event.runtime_turn_id == body.runtime_turn_id
    ]
    if len(associated_governance) != 1:
        return False

    governance_position, governance_event = associated_governance[0]
    governance_interaction_kind = governance_event.event_payload_json.get(
        "interaction_kind"
    )
    if (
        governance_position >= shape_position
        or governance_interaction_kind not in _INTERACTION_KINDS
        or governance_interaction_kind != interaction_kind
    ):
        return False
    return (
        governance_interaction_kind != "high_impact_decision"
        and "high_stakes_accuracy_required" not in reason_codes
    )


def _unresolved_material_requirements(
    evaluations: list[EvidenceRequirementEvaluation],
) -> list[str]:
    return sorted(
        evaluation.requirement_id
        for evaluation in evaluations
        if evaluation.criticality == "material"
        and evaluation.effective_outcome != "satisfied"
    )


def _clarification_is_permitted(
    evaluations: list[EvidenceRequirementEvaluation],
) -> bool:
    return any(
        evaluation.criticality == "material"
        and evaluation.effective_outcome in {"missing", "unknown"}
        for evaluation in evaluations
    )


def _safe_partial_answer_is_supported(
    evaluations: list[EvidenceRequirementEvaluation],
) -> bool:
    material = [
        evaluation
        for evaluation in evaluations
        if evaluation.criticality == "material"
    ]
    context_delivered = any(
        evaluation.requirement_kind == "context_delivery"
        and evaluation.effective_outcome == "satisfied"
        for evaluation in material
    )
    substantive_evidence = any(
        evaluation.requirement_kind not in _ADMINISTRATIVE_REQUIREMENT_KINDS
        and evaluation.effective_outcome in {"satisfied", "partial"}
        for evaluation in material
    )
    return context_delivered and substantive_evidence


def _has_unexamined_scope(
    evaluations: list[EvidenceRequirementEvaluation],
) -> bool:
    return any(
        evaluation.criticality == "material"
        and evaluation.requirement_kind in _SCOPE_REQUIREMENT_KINDS
        and evaluation.effective_outcome != "satisfied"
        for evaluation in evaluations
    )


def _proposed_premise_was_selected(
    events: list[RuntimeEvent],
    *,
    proposed_premise_digest: str,
    selection_id: str,
) -> bool:
    return any(
        event.event_type == "evidence_next_step_selected"
        and event.event_payload_json.get("selection_id") != selection_id
        and event.event_payload_json.get("selected_next_step")
        == "perform_additional_acquisition"
        and event.event_payload_json.get("proposed_premise_digest")
        == proposed_premise_digest
        for event in events
    )


def _ordered_reason_codes(
    reason_codes: set[EvidenceNextStepReasonCode],
) -> list[EvidenceNextStepReasonCode]:
    return [code for code in _REASON_CODE_ORDER if code in reason_codes]


def _fallback_selection(
    evaluations: list[EvidenceRequirementEvaluation],
    *,
    guard_reason: EvidenceNextStepReasonCode | None = None,
    advisory_allowed: bool = False,
) -> tuple[
    EvidenceNextStep,
    EvidenceConclusionDisposition,
    EvidenceProviderDisposition,
    list[EvidenceNextStepReasonCode],
    str,
]:
    reasons: set[EvidenceNextStepReasonCode] = set()
    if guard_reason is not None:
        reasons.add(guard_reason)
    if _safe_partial_answer_is_supported(evaluations):
        reasons.add("substantive_partial_evidence_available")
        return (
            "provide_qualified_partial_answer",
            "qualified_partial_only",
            "allowed",
            _ordered_reason_codes(reasons),
            "Delivered substantive evidence supports only a qualified partial answer.",
        )
    if _has_unexamined_scope(evaluations):
        reasons.add("unexamined_material_scope")
        return (
            "disclose_unexamined_scope",
            "requested_conclusion_withheld",
            "blocked",
            _ordered_reason_codes(reasons),
            "The requested conclusion remains withheld while material scope is unexamined.",
        )
    reasons.add("unsupported_conclusion_withheld")
    return (
        "withhold_unsupported_conclusion",
        "requested_conclusion_withheld",
        "allowed" if advisory_allowed else "blocked",
        _ordered_reason_codes(reasons),
        (
            "The requested conclusion remains unsupported; non-authoritative "
            "guidance is permitted without presenting it as verified."
            if advisory_allowed
            else "The requested conclusion remains withheld because material "
            "evidence is unsupported."
        ),
    )


def _selection(
    *,
    status: EvidenceSufficiencyStatus,
    evaluations: list[EvidenceRequirementEvaluation],
    clarification_target: str | None,
    current_premise_digest: str,
    proposed_premise_digest: str | None,
    proposed_premise_was_selected: bool,
    advisory_allowed: bool,
) -> tuple[
    EvidenceNextStep,
    EvidenceConclusionDisposition,
    EvidenceProviderDisposition,
    EvidenceReacquisitionGuard,
    list[EvidenceNextStepReasonCode],
    str,
]:
    if status == "sufficient_for_declared_scope":
        return (
            "answer_within_declared_scope",
            "bounded_conclusion_allowed",
            "allowed",
            "not_applicable",
            ["declared_scope_sufficient"],
            "The requested answer is supported within the declared evidence scope.",
        )
    if status == "sufficient_with_limitations":
        return (
            "provide_qualified_partial_answer",
            "qualified_partial_only",
            "allowed",
            "not_applicable",
            ["optional_limitations_remain"],
            "A qualified answer is permitted while optional evidence limitations remain.",
        )
    if clarification_target is not None and _clarification_is_permitted(evaluations):
        return (
            "ask_narrow_clarification",
            "requested_conclusion_withheld",
            "blocked",
            "not_applicable",
            ["material_uncertainty_requires_clarification"],
            "A narrow clarification is required before the requested conclusion.",
        )
    if proposed_premise_digest is not None:
        if proposed_premise_digest == current_premise_digest:
            fallback = _fallback_selection(
                evaluations,
                guard_reason="unchanged_acquisition_premise",
                advisory_allowed=advisory_allowed,
            )
            return (*fallback[:3], "unchanged_premise_blocked", *fallback[3:])
        if proposed_premise_was_selected:
            fallback = _fallback_selection(
                evaluations,
                guard_reason="acquisition_premise_already_selected",
                advisory_allowed=advisory_allowed,
            )
            return (*fallback[:3], "premise_already_attempted", *fallback[3:])
        return (
            "perform_additional_acquisition",
            "requested_conclusion_withheld",
            "blocked",
            "changed_premise_allowed",
            ["changed_acquisition_premise_available"],
            "Additional acquisition is permitted only under the changed evidence premise.",
        )
    fallback = _fallback_selection(
        evaluations,
        advisory_allowed=advisory_allowed,
    )
    return (*fallback[:3], "not_applicable", *fallback[3:])


def select_evidence_next_step(
    body: EvidenceNextStepSelectRequest,
) -> EvidenceNextStepSelectResponse:
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

    events = get_runtime_session(body.runtime_session_id).events
    sufficiency_event = _associated_sufficiency_event(events, body)
    evaluations = _normalized_evaluations(body.evaluated_requirements)
    requirements_digest = evaluated_requirements_digest(evaluations)
    if requirements_digest != sufficiency_event["evaluated_requirements_digest"]:
        raise RuntimeError("evaluated_requirements_mismatch")

    task_shape = cast(EvidenceTaskShape, sufficiency_event["task_shape"])
    status = cast(EvidenceSufficiencyStatus, sufficiency_event["sufficiency_status"])
    plan_event = _associated_plan_event(events, body, task_shape=task_shape)
    constraints = cast(
        list[EvidenceAnswerConstraint],
        sufficiency_event["answer_constraints"],
    )
    if status in {"insufficient", "unknown"} and (
        "withhold_unqualified_conclusion" not in constraints
        or "additional_acquisition_or_clarification_required" not in constraints
    ):
        raise RuntimeError("evidence_sufficiency_event_invalid")

    current_premise_digest = _premise_digest(body.current_premise)
    if current_premise_digest != plan_event["acquisition_premise_digest"]:
        raise RuntimeError("current_acquisition_premise_mismatch")
    if body.current_premise.question_anchor_digest != plan_event[
        "question_anchor_digest"
    ]:
        raise RuntimeError("current_acquisition_premise_mismatch")
    supplied_proposed_premise_digest = (
        _premise_digest(body.proposed_acquisition_premise)
        if body.proposed_acquisition_premise is not None
        else None
    )
    terminal_sufficiency = status in {
        "sufficient_for_declared_scope",
        "sufficient_with_limitations",
    }
    proposed_premise_digest = (
        None if terminal_sufficiency else supplied_proposed_premise_digest
    )
    clarification_target = None if terminal_sufficiency else body.clarification_target
    selection_id = _selection_identity(
        body,
        requirements_digest=requirements_digest,
        current_premise_digest=current_premise_digest,
        proposed_premise_digest=proposed_premise_digest,
        clarification_target=clarification_target,
    )
    proposed_was_selected = (
        proposed_premise_digest is not None
        and _proposed_premise_was_selected(
            events,
            proposed_premise_digest=proposed_premise_digest,
            selection_id=selection_id,
        )
    )
    advisory_allowed = _associated_shape_allows_advisory(
        events,
        body,
        plan_event=plan_event,
        task_shape=task_shape,
    )
    (
        selected_next_step,
        conclusion_disposition,
        provider_disposition,
        reacquisition_guard,
        reason_codes,
        user_safe_summary,
    ) = _selection(
        status=status,
        evaluations=evaluations,
        clarification_target=clarification_target,
        current_premise_digest=current_premise_digest,
        proposed_premise_digest=proposed_premise_digest,
        proposed_premise_was_selected=proposed_was_selected,
        advisory_allowed=advisory_allowed,
    )
    selected_clarification_target = (
        clarification_target
        if selected_next_step == "ask_narrow_clarification"
        else None
    )
    result = EvidenceNextStepResult(
        selection_id=selection_id,
        evaluation_id=body.evaluation_id,
        evidence_plan_id=body.evidence_plan_id,
        acquisition_manifest_id=body.acquisition_manifest_id,
        task_shape=task_shape,
        sufficiency_status=status,
        selected_next_step=selected_next_step,
        conclusion_disposition=conclusion_disposition,
        provider_disposition=provider_disposition,
        current_premise_digest=current_premise_digest,
        proposed_premise_digest=proposed_premise_digest,
        reacquisition_guard=cast(EvidenceReacquisitionGuard, reacquisition_guard),
        clarification_target=selected_clarification_target,
        unresolved_material_requirement_ids=_unresolved_material_requirements(
            evaluations
        ),
        reason_codes=reason_codes,
        user_safe_summary=user_safe_summary,
    )

    event_exists = any(
        event.event_type == "evidence_next_step_selected"
        and event.event_payload_json.get("selection_id") == selection_id
        for event in events
    )
    if not event_exists:
        record_runtime_event(
            runtime_session_id=body.runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
            event_type="evidence_next_step_selected",
            event_payload_json={
                "request_id": body.request_id,
                "runtime_session_id": body.runtime_session_id,
                "runtime_turn_id": body.runtime_turn_id,
                "selection_id": result.selection_id,
                "evaluation_id": result.evaluation_id,
                "evidence_plan_id": result.evidence_plan_id,
                "acquisition_manifest_id": result.acquisition_manifest_id,
                "task_shape": result.task_shape,
                "sufficiency_status": result.sufficiency_status,
                "selected_next_step": result.selected_next_step,
                "conclusion_disposition": result.conclusion_disposition,
                "provider_disposition": result.provider_disposition,
                "current_premise_digest": result.current_premise_digest,
                "proposed_premise_digest": result.proposed_premise_digest,
                "reacquisition_guard": result.reacquisition_guard,
                "clarification_target": result.clarification_target,
                "unresolved_material_requirement_count": len(
                    result.unresolved_material_requirement_ids
                ),
                "reason_codes": result.reason_codes,
            },
        )

    return EvidenceNextStepSelectResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        result=result,
    )
