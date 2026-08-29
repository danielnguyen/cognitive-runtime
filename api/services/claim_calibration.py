from __future__ import annotations

import json
from hashlib import sha256

from models import (
    ClaimCalibrationEvaluateRequest,
    ClaimCalibrationEvaluateResponse,
    ClaimCalibrationResult,
    ClaimClass,
    ClaimConfidence,
    ClaimEvidenceAuthority,
    ClaimEvidenceReference,
    ClaimEvidenceStrength,
    ClaimFreshnessSummary,
    ClaimLimitationCode,
    ClaimSupportCalibrationStatus,
    ClaimSupportConclusionDisposition,
    ClaimSupportEvaluateRequest,
    ClaimSupportEvaluateResponse,
    ClaimSupportEvaluationResult,
    ClaimSupportLimitationCode,
)
from services.runtime_state import (
    record_runtime_event,
    runtime_session_by_id,
    validate_runtime_turn_session,
)

_AUTHORITY_RANK: dict[ClaimEvidenceAuthority, int] = {
    "peer_reviewed_evidence": 8,
    "clinical_guidance": 7,
    "manufacturer_guidance": 6,
    "tool_output": 5,
    "trusted_integration": 4,
    "user_report": 3,
    "runtime_inference": 2,
    "speculation": 1,
    "unknown": 0,
}
_MATERIALLY_AUTHORITATIVE = {
    "peer_reviewed_evidence",
    "clinical_guidance",
    "manufacturer_guidance",
    "tool_output",
    "trusted_integration",
}
_POSITIVE_SUPPORT = {"direct", "corroborating"}
_UNUSABLE_FRESHNESS = {"superseded", "corrected"}


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _sorted_references(
    references: list[ClaimEvidenceReference],
) -> list[ClaimEvidenceReference]:
    return sorted(
        references,
        key=lambda item: (
            item.ref_type,
            item.ref_id,
            item.owner_id,
            item.conversation_id or "",
            item.support_kind,
            item.authority,
            item.freshness_state,
        ),
    )


def _positive_references(
    references: list[ClaimEvidenceReference],
) -> list[ClaimEvidenceReference]:
    return [
        reference
        for reference in references
        if reference.support_kind in _POSITIVE_SUPPORT
        and reference.freshness_state not in _UNUSABLE_FRESHNESS
    ]


def _strongest_authority(
    positive: list[ClaimEvidenceReference],
) -> ClaimEvidenceAuthority:
    if not positive:
        return "unknown"
    return max(
        (reference.authority for reference in positive),
        key=lambda authority: _AUTHORITY_RANK[authority],
    )


def _freshness_summary(
    references: list[ClaimEvidenceReference],
) -> ClaimFreshnessSummary:
    if not references:
        return "unknown"
    states = {reference.freshness_state for reference in references}
    if states == {"not_applicable"}:
        return "not_applicable"
    material_states = states - {"not_applicable"}
    if material_states == {"active"}:
        return "current"
    if material_states and material_states <= {"stale", "superseded", "corrected"}:
        return "stale"
    if material_states == {"unknown_freshness"}:
        return "unknown"
    return "mixed"


def _claim_class(
    positive: list[ClaimEvidenceReference],
    *,
    contradictory: bool,
) -> ClaimClass:
    if not positive:
        return "unknown"

    authorities = {reference.authority for reference in positive}
    if authorities == {"speculation"}:
        return "speculation"
    if authorities <= {"runtime_inference", "speculation"}:
        return "runtime_inference"
    if authorities == {"manufacturer_guidance"}:
        return "manufacturer_guidance"

    verified_integration = any(
        reference.support_kind == "direct"
        and reference.freshness_state == "active"
        and (
            (
                reference.ref_type == "tool_output"
                and reference.authority == "tool_output"
            )
            or (
                reference.ref_type == "integration_event"
                and reference.authority == "trusted_integration"
            )
        )
        for reference in positive
    )
    if verified_integration and not contradictory:
        return "verified_fact"

    current_consensus = (
        not contradictory
        and len(positive) >= 2
        and all(reference.freshness_state == "active" for reference in positive)
        and {"peer_reviewed_evidence", "clinical_guidance"} <= authorities
    )
    if current_consensus:
        return "expert_consensus"
    if authorities == {"unknown"}:
        return "unknown"
    return "source_backed_fact"


def _evidence_strength(
    references: list[ClaimEvidenceReference],
    positive: list[ClaimEvidenceReference],
) -> ClaimEvidenceStrength:
    if not positive:
        return "none"
    if any(reference.support_kind == "contradictory" for reference in references):
        return "weak"
    if any(
        reference.freshness_state in {
            "stale",
            "unknown_freshness",
            "superseded",
            "corrected",
        }
        for reference in references
    ):
        return "weak"
    if any(reference.authority not in _MATERIALLY_AUTHORITATIVE for reference in positive):
        return "weak"
    if (
        len(positive) >= 2
        and all(reference.freshness_state == "active" for reference in positive)
    ):
        return "strong"
    return "moderate"


def _limitation_codes(
    references: list[ClaimEvidenceReference],
    positive: list[ClaimEvidenceReference],
) -> list[ClaimLimitationCode]:
    limitations: list[ClaimLimitationCode] = []

    def add(code: ClaimLimitationCode) -> None:
        if code not in limitations:
            limitations.append(code)

    if not positive:
        add("no_supporting_evidence")
    if not positive and any(reference.support_kind == "contextual" for reference in references):
        add("context_only")
    if positive and all(
        reference.authority not in _MATERIALLY_AUTHORITATIVE for reference in positive
    ):
        add("low_authority_evidence")
    if any(reference.freshness_state == "stale" for reference in references):
        add("stale_evidence")
    if any(reference.freshness_state == "unknown_freshness" for reference in references):
        add("unknown_freshness")
    if any(reference.freshness_state in _UNUSABLE_FRESHNESS for reference in references):
        add("superseded_or_corrected_evidence")
    if any(reference.support_kind == "contradictory" for reference in references):
        add("contradictory_evidence")
    if len(positive) == 1:
        add("single_source")
    if positive and all(
        reference.authority in {"runtime_inference", "speculation"}
        for reference in positive
    ):
        add("inference_dominant")
    if positive and all(reference.authority == "speculation" for reference in positive):
        add("speculation_only")
    return limitations


def _user_safe_summary(
    strength: ClaimEvidenceStrength,
    limitations: list[ClaimLimitationCode],
) -> str:
    if strength == "strong":
        return (
            "This claim is strongly supported by multiple current, "
            "authoritative evidence records."
        )
    if strength == "moderate":
        if "single_source" in limitations:
            return "This claim has recorded support, but the evidence set is limited to one source."
        return "This claim has current, authoritative recorded support."
    if strength == "weak":
        if "contradictory_evidence" in limitations or "stale_evidence" in limitations:
            return (
                "This claim has limited support because the recorded evidence is stale "
                "or contradictory."
            )
        return "This claim has limited recorded support and should be presented with uncertainty."
    return "No valid supporting evidence was recorded for this claim."


def _claim_identity(
    body: ClaimCalibrationEvaluateRequest,
    claim_anchor_digest: str,
    references: list[ClaimEvidenceReference],
) -> str:
    material = {
        "request_id": body.request_id,
        "owner_id": body.owner_id,
        "conversation_id": body.conversation_id,
        "surface": body.surface,
        "runtime_session_id": body.runtime_session_id,
        "runtime_turn_id": body.runtime_turn_id,
        "claim_anchor_digest": claim_anchor_digest,
        "evidence_references": [reference.model_dump(mode="json") for reference in references],
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"claim_{_digest(encoded)[:32]}"


def evaluate_claim_calibration(
    body: ClaimCalibrationEvaluateRequest,
) -> ClaimCalibrationEvaluateResponse:
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

    references = _sorted_references(body.evidence_references)
    positive = _positive_references(references)
    contradictory = any(
        reference.support_kind == "contradictory" for reference in references
    )
    strength = _evidence_strength(references, positive)
    confidence: ClaimConfidence = {
        "strong": "high",
        "moderate": "medium",
        "weak": "low",
        "none": "unknown",
    }[strength]
    limitations = _limitation_codes(references, positive)
    claim_anchor_digest = f"sha256:{_digest(body.claim_anchor)}"
    result = ClaimCalibrationResult(
        claim_id=_claim_identity(body, claim_anchor_digest, references),
        claim_anchor=body.claim_anchor,
        claim_anchor_digest=claim_anchor_digest,
        claim_class=_claim_class(positive, contradictory=contradictory),
        calibration_status={
            "strong": "supported",
            "moderate": "supported",
            "weak": "limited",
            "none": "unsupported",
        }[strength],
        evidence_strength=strength,
        confidence=confidence,
        strongest_authority=_strongest_authority(positive),
        freshness_summary=_freshness_summary(references),
        uncertainty_disclosure_required=strength != "strong",
        validated_evidence_references=references,
        limitation_codes=limitations,
        user_safe_summary=_user_safe_summary(strength, limitations),
    )

    record_runtime_event(
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        event_type="claim_calibration_evaluated",
        event_payload_json={
            "request_id": body.request_id,
            "runtime_session_id": body.runtime_session_id,
            "runtime_turn_id": body.runtime_turn_id,
            "claim_id": result.claim_id,
            "claim_anchor_digest": result.claim_anchor_digest,
            "claim_class": result.claim_class,
            "calibration_status": result.calibration_status,
            "evidence_strength": result.evidence_strength,
            "confidence": result.confidence,
            "strongest_authority": result.strongest_authority,
            "freshness_summary": result.freshness_summary,
            "uncertainty_disclosure_required": result.uncertainty_disclosure_required,
            "evidence_count": len(result.validated_evidence_references),
            "limitation_codes": result.limitation_codes,
        },
    )
    return ClaimCalibrationEvaluateResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        result=result,
    )


def _claim_support_identity(
    body: ClaimSupportEvaluateRequest,
    claim_digest: str,
    limitation_codes: list[ClaimSupportLimitationCode],
) -> str:
    proposal = body.proposal
    material = {
        "request_id": body.request_id,
        "owner_id": body.authority_context.owner_id,
        "conversation_id": body.authority_context.conversation_id,
        "surface": body.authority_context.surface,
        "runtime_session_id": body.authority_context.runtime_session_id,
        "runtime_turn_id": body.authority_context.runtime_turn_id,
        "claim_scope_basis": body.authority_context.claim_scope_basis,
        "claim_digest": claim_digest,
        "supporting_evidence_ref_ids": sorted(proposal.supporting_evidence_ref_ids),
        "counterevidence_ref_ids": sorted(proposal.counterevidence_ref_ids),
        "material_exclusions": sorted(
            (
                {
                    "evidence_ref_id": item.evidence_ref_id,
                    "reason": item.reason,
                }
                for item in proposal.material_exclusions
            ),
            key=lambda item: (item["evidence_ref_id"], item["reason"]),
        ),
        "executed_derivation_ref_ids": sorted(
            proposal.executed_derivation_ref_ids
        ),
        "limitation_codes": limitation_codes,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"claim_{_digest(encoded)[:32]}"


def _claim_support_summary(
    calibration_status: ClaimSupportCalibrationStatus,
) -> str:
    if calibration_status == "supported":
        return "The proposed claim has authorized support for the declared bounded scope."
    if calibration_status == "limited":
        return (
            "The proposed claim has bounded support but requires explicit qualification "
            "for the recorded limitations."
        )
    return (
        "The proposed claim does not have enough authorized support for an "
        "unqualified conclusion."
    )


def evaluate_claim_support(
    body: ClaimSupportEvaluateRequest,
) -> ClaimSupportEvaluateResponse:
    context = body.authority_context
    proposal = body.proposal
    session = runtime_session_by_id(context.runtime_session_id)
    if session is None:
        raise RuntimeError("runtime_session_not_found")
    if (
        session.owner_id != context.owner_id
        or session.conversation_id != context.conversation_id
        or session.surface != context.surface
    ):
        raise RuntimeError("runtime_session_mismatch")
    validate_runtime_turn_session(
        runtime_session_id=context.runtime_session_id,
        runtime_turn_id=context.runtime_turn_id,
    )

    evidence_by_id = {item.ref_id: item for item in context.evidence_references}
    derivations_by_id = {
        item.derivation_id: item for item in context.executed_derivations
    }
    support_ids = set(proposal.supporting_evidence_ref_ids)
    counter_ids = set(proposal.counterevidence_ref_ids)
    exclusion_ids = {
        item.evidence_ref_id for item in proposal.material_exclusions
    }
    referenced_derivations = [
        derivations_by_id[item]
        for item in sorted(proposal.executed_derivation_ref_ids)
    ]
    derivation_evidence_ids = {
        evidence_ref_id
        for derivation in referenced_derivations
        for evidence_ref_id in derivation.supporting_evidence_ref_ids
    }
    authority_support_ids = support_ids | derivation_evidence_ids
    mentioned_ids = support_ids | counter_ids | exclusion_ids

    limitations: list[ClaimSupportLimitationCode] = []
    calibration_status: ClaimSupportCalibrationStatus = "supported"
    disposition: ClaimSupportConclusionDisposition = "allowed"

    def add_limitation(
        code: ClaimSupportLimitationCode,
        *,
        severity: ClaimSupportCalibrationStatus,
    ) -> None:
        nonlocal calibration_status, disposition
        if code not in limitations:
            limitations.append(code)
        if severity == "unsupported":
            calibration_status = "unsupported"
            disposition = "withheld"
        elif calibration_status == "supported":
            calibration_status = "limited"
            disposition = "qualified"

    if not support_ids and not referenced_derivations:
        add_limitation("no_supporting_evidence", severity="unsupported")

    for evidence_ref_id in sorted(authority_support_ids):
        reference = evidence_by_id[evidence_ref_id]
        if reference.source_authority == "limited":
            add_limitation("limited_source_authority", severity="limited")
        elif reference.source_authority == "unknown":
            add_limitation("unknown_source_authority", severity="limited")
        if reference.freshness == "stale":
            add_limitation("stale_evidence", severity="limited")
        elif reference.freshness == "unknown":
            add_limitation("unknown_freshness", severity="limited")

    if context.complete_declared_scope_required and (
        context.complete_declared_scope_established is not True
        or context.material_acquisition_limited
    ):
        add_limitation(
            "complete_scope_not_established",
            severity=(
                "limited"
                if context.claim_scope_basis == "supplied_evidence"
                else "unsupported"
            ),
        )
    if context.material_acquisition_limited:
        add_limitation("material_acquisition_limited", severity="limited")

    for reference in sorted(context.evidence_references, key=lambda item: item.ref_id):
        if reference.material_disclosure_required and reference.ref_id not in mentioned_ids:
            add_limitation("material_evidence_omitted", severity="unsupported")
        if reference.material_role == "counterevidence":
            if reference.ref_id in support_ids:
                add_limitation(
                    "material_counterevidence_misclassified",
                    severity="unsupported",
                )
            elif reference.ref_id in counter_ids:
                add_limitation(
                    "material_counterevidence_present",
                    severity="limited",
                )
            elif reference.ref_id in exclusion_ids:
                add_limitation(
                    "material_counterevidence_excluded",
                    severity="limited",
                )
        if reference.material_role == "support" and reference.ref_id in exclusion_ids:
            add_limitation("material_support_excluded", severity="limited")

    if counter_ids:
        add_limitation("declared_counterevidence", severity="limited")
    if exclusion_ids:
        add_limitation("material_exclusion", severity="limited")
    if any(
        derivation.input_basis == "model_interpreted"
        for derivation in referenced_derivations
    ):
        add_limitation("interpretation_dependent_derivation", severity="limited")
    if not context.privacy_policy_allows_claim:
        add_limitation("privacy_constraint", severity="unsupported")
    if not context.consequence_policy_allows_claim:
        add_limitation("consequence_constraint", severity="unsupported")

    claim_digest = f"sha256:{_digest(proposal.proposed_claim)}"
    result = ClaimSupportEvaluationResult(
        claim_id=_claim_support_identity(body, claim_digest, limitations),
        claim_digest=claim_digest,
        calibration_status=calibration_status,
        conclusion_disposition=disposition,
        qualification_required=disposition != "allowed",
        limitation_codes=limitations,
        validated_supporting_evidence_ref_ids=sorted(support_ids),
        validated_counterevidence_ref_ids=sorted(counter_ids),
        validated_material_exclusions=sorted(
            proposal.material_exclusions,
            key=lambda item: (item.evidence_ref_id, item.reason),
        ),
        validated_executed_derivation_ref_ids=sorted(
            proposal.executed_derivation_ref_ids
        ),
        user_safe_summary=_claim_support_summary(calibration_status),
    )

    record_runtime_event(
        runtime_session_id=context.runtime_session_id,
        runtime_turn_id=context.runtime_turn_id,
        event_type="claim_support_evaluated",
        event_payload_json={
            "request_id": body.request_id,
            "runtime_session_id": context.runtime_session_id,
            "runtime_turn_id": context.runtime_turn_id,
            "claim_scope_basis": context.claim_scope_basis,
            "claim_id": result.claim_id,
            "claim_digest": result.claim_digest,
            "calibration_status": result.calibration_status,
            "conclusion_disposition": result.conclusion_disposition,
            "qualification_required": result.qualification_required,
            "limitation_codes": result.limitation_codes,
            "supporting_evidence_count": len(
                result.validated_supporting_evidence_ref_ids
            ),
            "counterevidence_count": len(result.validated_counterevidence_ref_ids),
            "material_exclusion_count": len(result.validated_material_exclusions),
            "executed_derivation_count": len(
                result.validated_executed_derivation_ref_ids
            ),
            "interpretation_dependent_derivation": (
                "interpretation_dependent_derivation" in result.limitation_codes
            ),
        },
    )
    return ClaimSupportEvaluateResponse(
        request_id=body.request_id,
        owner_id=context.owner_id,
        conversation_id=context.conversation_id,
        surface=context.surface,
        runtime_session_id=context.runtime_session_id,
        runtime_turn_id=context.runtime_turn_id,
        result=result,
    )
