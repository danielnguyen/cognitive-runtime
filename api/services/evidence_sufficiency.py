from __future__ import annotations

import json
from hashlib import sha256

from models import (
    EvidenceAcquisitionFact,
    EvidenceAnswerConstraint,
    EvidenceRequirement,
    EvidenceRequirementEvaluation,
    EvidenceSufficiencyEvaluateRequest,
    EvidenceSufficiencyEvaluateResponse,
    EvidenceSufficiencyReasonCode,
    EvidenceSufficiencyResult,
    EvidenceSufficiencyStatus,
)
from services.runtime_state import (
    record_runtime_event,
    runtime_session_by_id,
    validate_runtime_turn_session,
)

_CONCRETE_MATERIAL_FAILURES = {
    "partial",
    "not_attempted",
    "unavailable",
    "unsupported",
    "failed",
    "excluded",
    "filtered",
    "truncated",
    "unresolved_contradiction",
}
_LIMITATION_CONSTRAINTS: list[EvidenceAnswerConstraint] = [
    "qualify_conclusion",
    "disclose_limitations",
    "identify_unexamined_scope",
]
_BLOCKING_CONSTRAINTS: list[EvidenceAnswerConstraint] = [
    *_LIMITATION_CONSTRAINTS,
    "additional_acquisition_or_clarification_required",
    "withhold_unqualified_conclusion",
]
_TASK_CONSTRAINTS: dict[str, EvidenceAnswerConstraint] = {
    "bounded_exhaustive_review": "withhold_exhaustive_conclusion",
    "absence_or_coverage_check": "withhold_absence_conclusion",
    "contradiction_review": "withhold_contradiction_sensitive_conclusion",
}


def _sorted_requirements(
    requirements: list[EvidenceRequirement],
) -> list[EvidenceRequirement]:
    return sorted(
        requirements,
        key=lambda item: (
            item.requirement_id,
            item.requirement_kind,
            item.criticality,
        ),
    )


def _sorted_facts(facts: list[EvidenceAcquisitionFact]) -> list[EvidenceAcquisitionFact]:
    return sorted(facts, key=lambda item: (item.requirement_id, item.outcome))


def _evaluated_requirements(
    requirements: list[EvidenceRequirement],
    facts: list[EvidenceAcquisitionFact],
) -> list[EvidenceRequirementEvaluation]:
    facts_by_requirement = {fact.requirement_id: fact for fact in facts}
    return [
        EvidenceRequirementEvaluation(
            requirement_id=requirement.requirement_id,
            requirement_kind=requirement.requirement_kind,
            criticality=requirement.criticality,
            effective_outcome=(
                facts_by_requirement[requirement.requirement_id].outcome
                if requirement.requirement_id in facts_by_requirement
                else "missing"
            ),
        )
        for requirement in requirements
    ]


def _sufficiency_status(
    evaluations: list[EvidenceRequirementEvaluation],
) -> EvidenceSufficiencyStatus:
    material = [item for item in evaluations if item.criticality == "material"]
    optional = [item for item in evaluations if item.criticality == "optional"]
    if any(item.effective_outcome in _CONCRETE_MATERIAL_FAILURES for item in material):
        return "insufficient"
    if any(item.effective_outcome in {"missing", "unknown"} for item in material):
        return "unknown"
    if any(item.effective_outcome != "satisfied" for item in optional):
        return "sufficient_with_limitations"
    return "sufficient_for_declared_scope"


def _reason_codes(
    evaluations: list[EvidenceRequirementEvaluation],
    *,
    status: EvidenceSufficiencyStatus,
    task_shape: str,
) -> list[EvidenceSufficiencyReasonCode]:
    material = [item for item in evaluations if item.criticality == "material"]
    optional = [item for item in evaluations if item.criticality == "optional"]
    applicable: dict[EvidenceSufficiencyReasonCode, bool] = {
        "all_declared_requirements_satisfied": all(
            item.effective_outcome == "satisfied" for item in evaluations
        ),
        "optional_requirement_incomplete": any(
            item.effective_outcome != "satisfied" for item in optional
        ),
        "material_requirement_not_satisfied": any(
            item.effective_outcome in _CONCRETE_MATERIAL_FAILURES for item in material
        ),
        "material_requirement_unknown": any(
            item.effective_outcome == "unknown" for item in material
        ),
        "material_requirement_missing": any(
            item.effective_outcome == "missing" for item in material
        ),
        "unresolved_material_contradiction": any(
            item.effective_outcome == "unresolved_contradiction" for item in material
        ),
        "exhaustive_scope_incomplete": (
            status in {"insufficient", "unknown"}
            and task_shape == "bounded_exhaustive_review"
        ),
        "absence_scope_unproven": (
            status in {"insufficient", "unknown"}
            and task_shape == "absence_or_coverage_check"
        ),
        "contradiction_sensitive_scope_unresolved": (
            status in {"insufficient", "unknown"}
            and task_shape == "contradiction_review"
        ),
    }
    return [code for code, is_applicable in applicable.items() if is_applicable]


def _answer_constraints(
    status: EvidenceSufficiencyStatus,
    *,
    task_shape: str,
) -> list[EvidenceAnswerConstraint]:
    if status == "sufficient_for_declared_scope":
        return []
    if status == "sufficient_with_limitations":
        return list(_LIMITATION_CONSTRAINTS)
    constraints = list(_BLOCKING_CONSTRAINTS)
    task_constraint = _TASK_CONSTRAINTS.get(task_shape)
    if task_constraint is not None:
        constraints.append(task_constraint)
    return constraints


def _user_safe_summary(status: EvidenceSufficiencyStatus) -> str:
    return {
        "sufficient_for_declared_scope": (
            "The declared evidence requirements were satisfied for the bounded scope."
        ),
        "sufficient_with_limitations": (
            "Material evidence requirements were satisfied, but optional limitations remain."
        ),
        "insufficient": "One or more material evidence requirements were not satisfied.",
        "unknown": (
            "The available acquisition facts do not establish whether all material "
            "evidence requirements were satisfied."
        ),
    }[status]


def _evaluation_identity(
    body: EvidenceSufficiencyEvaluateRequest,
    requirements: list[EvidenceRequirement],
    facts: list[EvidenceAcquisitionFact],
) -> str:
    material = {
        "request_id": body.request_id,
        "owner_id": body.owner_id,
        "conversation_id": body.conversation_id,
        "surface": body.surface,
        "runtime_session_id": body.runtime_session_id,
        "runtime_turn_id": body.runtime_turn_id,
        "evidence_plan_id": body.evidence_plan_id,
        "acquisition_manifest_id": body.acquisition_manifest_id,
        "task_shape": body.task_shape,
        "declared_requirements": [
            requirement.model_dump(mode="json") for requirement in requirements
        ],
        "acquisition_facts": [fact.model_dump(mode="json") for fact in facts],
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = sha256(encoded.encode("utf-8")).hexdigest()
    return f"evidence_eval_{digest[:32]}"


def evaluate_evidence_sufficiency(
    body: EvidenceSufficiencyEvaluateRequest,
) -> EvidenceSufficiencyEvaluateResponse:
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

    requirements = _sorted_requirements(body.declared_requirements)
    facts = _sorted_facts(body.acquisition_facts)
    evaluations = _evaluated_requirements(requirements, facts)
    status = _sufficiency_status(evaluations)
    reasons = _reason_codes(evaluations, status=status, task_shape=body.task_shape)
    constraints = _answer_constraints(status, task_shape=body.task_shape)
    result = EvidenceSufficiencyResult(
        evaluation_id=_evaluation_identity(body, requirements, facts),
        task_shape=body.task_shape,
        sufficiency_status=status,
        evaluated_requirements=evaluations,
        reason_codes=reasons,
        answer_constraints=constraints,
        qualification_required=status != "sufficient_for_declared_scope",
        additional_acquisition_required=status in {"insufficient", "unknown"},
        user_safe_summary=_user_safe_summary(status),
    )

    satisfied_count = sum(
        item.effective_outcome == "satisfied" for item in evaluations
    )
    missing_count = sum(item.effective_outcome == "missing" for item in evaluations)
    record_runtime_event(
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        event_type="evidence_sufficiency_evaluated",
        event_payload_json={
            "request_id": body.request_id,
            "runtime_session_id": body.runtime_session_id,
            "runtime_turn_id": body.runtime_turn_id,
            "evaluation_id": result.evaluation_id,
            "evidence_plan_id": body.evidence_plan_id,
            "acquisition_manifest_id": body.acquisition_manifest_id,
            "task_shape": result.task_shape,
            "sufficiency_status": result.sufficiency_status,
            "total_requirement_count": len(evaluations),
            "material_requirement_count": sum(
                item.criticality == "material" for item in evaluations
            ),
            "optional_requirement_count": sum(
                item.criticality == "optional" for item in evaluations
            ),
            "satisfied_requirement_count": satisfied_count,
            "missing_requirement_count": missing_count,
            "non_satisfactory_requirement_count": len(evaluations) - satisfied_count,
            "reason_codes": result.reason_codes,
            "answer_constraints": result.answer_constraints,
            "qualification_required": result.qualification_required,
            "additional_acquisition_required": result.additional_acquisition_required,
        },
    )
    return EvidenceSufficiencyEvaluateResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        evidence_plan_id=body.evidence_plan_id,
        acquisition_manifest_id=body.acquisition_manifest_id,
        result=result,
    )
