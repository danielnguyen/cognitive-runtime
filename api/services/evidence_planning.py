from __future__ import annotations

import json
from hashlib import sha256

from models import (
    EvidenceAcquisitionStrategy,
    EvidenceCompletenessExpectation,
    EvidenceDeclaredScope,
    EvidencePlanCompileRequest,
    EvidencePlanCompileResponse,
    EvidencePlanLimitationCode,
    EvidencePlanResult,
    EvidencePlanStatus,
    EvidenceRequirement,
    EvidenceSourceDescriptor,
)
from services.runtime_state import (
    record_runtime_event,
    runtime_session_by_id,
    validate_runtime_turn_session,
)

_COMPLETENESS_BY_SHAPE: dict[str, EvidenceCompletenessExpectation] = {
    "targeted_lookup": "targeted_scope",
    "bounded_exhaustive_review": "complete_for_declared_scope",
    "cross_source_comparison": "complete_for_selected_sources",
    "contradiction_review": "complete_for_selected_sources",
    "absence_or_coverage_check": "complete_for_declared_scope",
    "historical_reconstruction": "complete_for_time_window",
    "recommendation_or_decision_support": "bounded_decision_support",
}
_CONTRADICTION_SHAPES = {
    "bounded_exhaustive_review",
    "contradiction_review",
    "recommendation_or_decision_support",
}
_AUTHORITATIVE_INVENTORY_SHAPES = {
    "bounded_exhaustive_review",
    "absence_or_coverage_check",
}
_COMPLETE_INVENTORY_SHAPES = {
    "bounded_exhaustive_review",
    "absence_or_coverage_check",
}
_MATERIAL_REQUIREMENT_KINDS: dict[str, tuple[str, ...]] = {
    "targeted_lookup": ("targeted_evidence", "context_delivery"),
    "bounded_exhaustive_review": (
        "authoritative_inventory",
        "complete_scope_coverage",
        "contradiction_search",
        "context_delivery",
        "no_material_truncation",
    ),
    "cross_source_comparison": (
        "selected_source_coverage",
        "cross_source_comparison",
        "context_delivery",
    ),
    "contradiction_review": (
        "contradiction_search",
        "counterevidence_coverage",
        "context_delivery",
        "no_material_truncation",
    ),
    "absence_or_coverage_check": (
        "authoritative_inventory",
        "complete_scope_coverage",
        "structured_absence_check",
        "context_delivery",
        "no_material_truncation",
    ),
    "historical_reconstruction": (
        "historical_scope",
        "historical_sequence_coverage",
        "context_delivery",
        "no_material_truncation",
    ),
    "recommendation_or_decision_support": (
        "candidate_evidence_coverage",
        "cross_source_comparison",
        "contradiction_search",
        "counterevidence_coverage",
        "context_delivery",
        "no_material_truncation",
    ),
}


def _normalized_scope(scope: EvidenceDeclaredScope) -> EvidenceDeclaredScope:
    return scope.model_copy(
        update={
            "source_ids": sorted(scope.source_ids),
            "source_categories": sorted(scope.source_categories),
            "exact_source_refs": sorted(
                scope.exact_source_refs,
                key=lambda reference: (reference.source_ref, reference.source_id),
            ),
        }
    )


def _normalized_inventory(
    inventory: list[EvidenceSourceDescriptor],
) -> list[EvidenceSourceDescriptor]:
    normalized = [
        source.model_copy(
            update={
                "source_categories": sorted(source.source_categories),
                "capabilities": sorted(source.capabilities),
            }
        )
        for source in inventory
    ]
    return sorted(normalized, key=lambda source: source.source_id)


def _within_declared_universe(
    source: EvidenceSourceDescriptor,
    scope: EvidenceDeclaredScope,
) -> bool:
    if scope.exact_source_refs:
        referenced_source_ids = {
            reference.source_id for reference in scope.exact_source_refs
        }
        if source.source_id not in referenced_source_ids:
            return False
        if scope.source_categories:
            return bool(set(source.source_categories) & set(scope.source_categories))
        return True
    if scope.source_ids:
        return source.source_id in scope.source_ids
    if scope.source_categories:
        return bool(set(source.source_categories) & set(scope.source_categories))
    return True


def _eligible_sources(
    inventory: list[EvidenceSourceDescriptor],
    scope: EvidenceDeclaredScope,
) -> list[EvidenceSourceDescriptor]:
    return [
        source
        for source in inventory
        if source.availability == "available"
        and _within_declared_universe(source, scope)
    ]


def _has_capability(
    sources: list[EvidenceSourceDescriptor],
    capability: str,
) -> bool:
    return any(capability in source.capabilities for source in sources)


def _all_have_capability(
    sources: list[EvidenceSourceDescriptor],
    capability: str,
) -> bool:
    return bool(sources) and all(
        capability in source.capabilities for source in sources
    )


def _has_hybrid_source(sources: list[EvidenceSourceDescriptor]) -> bool:
    expansion = {"exact_fetch", "bounded_full_context", "context_expansion"}
    return any(
        "targeted_retrieval" in source.capabilities
        and bool(set(source.capabilities) & expansion)
        for source in sources
    )


def _all_support_hybrid(sources: list[EvidenceSourceDescriptor]) -> bool:
    expansion = {"exact_fetch", "bounded_full_context", "context_expansion"}
    return bool(sources) and all(
        "targeted_retrieval" in source.capabilities
        and bool(set(source.capabilities) & expansion)
        for source in sources
    )


def _select_strategy(
    *,
    task_shape: str,
    scope: EvidenceDeclaredScope,
    eligible: list[EvidenceSourceDescriptor],
) -> EvidenceAcquisitionStrategy | None:
    if task_shape == "targeted_lookup":
        if scope.exact_source_refs:
            referenced_source_ids = {
                reference.source_id for reference in scope.exact_source_refs
            }
            eligible_source_ids = {source.source_id for source in eligible}
            if (
                eligible_source_ids == referenced_source_ids
                and _all_have_capability(eligible, "exact_fetch")
            ):
                return "exact_fetch"
            return None
        if _has_capability(eligible, "targeted_retrieval"):
            return "targeted_retrieval"
        if _has_capability(eligible, "bounded_full_context"):
            return "bounded_full_context"
        return None

    if task_shape == "bounded_exhaustive_review":
        if _all_have_capability(eligible, "structured_query"):
            return "structured_query"
        if _all_have_capability(eligible, "bounded_full_context"):
            return "bounded_full_context"
        if _all_support_hybrid(eligible):
            return "hybrid"
        return None

    if task_shape == "absence_or_coverage_check":
        if _all_have_capability(eligible, "structured_query"):
            return "structured_query"
        if _all_have_capability(eligible, "bounded_full_context"):
            return "bounded_full_context"
        return None

    if task_shape == "cross_source_comparison":
        if len(eligible) < 2:
            return None
        if _all_have_capability(eligible, "targeted_retrieval") and _has_hybrid_source(
            eligible
        ):
            return "hybrid"
        if _all_have_capability(eligible, "structured_query"):
            return "structured_query"
        if _all_have_capability(eligible, "bounded_full_context"):
            return "bounded_full_context"
        if _all_have_capability(eligible, "exact_fetch"):
            return "exact_fetch"
        if scope.source_ids and _all_have_capability(eligible, "targeted_retrieval"):
            return "targeted_retrieval"
        return None

    if task_shape == "contradiction_review":
        one_full_authoritative = any(
            source.authority_role == "authoritative"
            and "bounded_full_context" in source.capabilities
            for source in eligible
        )
        if len(eligible) < 2 and not one_full_authoritative:
            return None
        if _all_support_hybrid(eligible):
            return "hybrid"
        if _all_have_capability(eligible, "bounded_full_context"):
            return "bounded_full_context"
        if _all_have_capability(eligible, "structured_query"):
            return "structured_query"
        if _all_have_capability(eligible, "exact_fetch"):
            return "exact_fetch"
        return None

    if task_shape == "historical_reconstruction":
        if not scope.time_scope_ref:
            return None
        if _all_have_capability(eligible, "structured_query"):
            return "structured_query"
        if _all_support_hybrid(eligible):
            return "hybrid"
        if _all_have_capability(eligible, "bounded_full_context"):
            return "bounded_full_context"
        return None

    if task_shape == "recommendation_or_decision_support":
        if len(eligible) < 2:
            return None
        if _all_support_hybrid(eligible):
            return "hybrid"
        if _all_have_capability(eligible, "structured_query"):
            return "structured_query"
        if _all_have_capability(eligible, "bounded_full_context"):
            return "bounded_full_context"
        return None

    return None


def _base_limitations(
    *,
    scope: EvidenceDeclaredScope,
    inventory: list[EvidenceSourceDescriptor],
    eligible: list[EvidenceSourceDescriptor],
) -> set[EvidencePlanLimitationCode]:
    limitations: set[EvidencePlanLimitationCode] = set()
    inventory_ids = {source.source_id for source in inventory}
    declared_source_ids = (
        {reference.source_id for reference in scope.exact_source_refs}
        if scope.exact_source_refs
        else set(scope.source_ids)
    )
    if any(source_id not in inventory_ids for source_id in declared_source_ids):
        limitations.add("declared_source_missing_from_inventory")

    if scope.source_categories and not any(
        set(source.source_categories) & set(scope.source_categories)
        for source in inventory
    ):
        limitations.add("declared_category_not_available")

    inventory_status_code: dict[str, EvidencePlanLimitationCode] = {
        "partial": "source_inventory_partial",
        "unknown": "source_inventory_unknown",
        "unavailable": "source_inventory_unavailable",
    }
    status_code = inventory_status_code.get(scope.inventory_status)
    if status_code is not None:
        limitations.add(status_code)

    scoped_inventory = [
        source for source in inventory if _within_declared_universe(source, scope)
    ]
    if any(
        source.authority_role == "authoritative"
        and source.availability != "available"
        for source in scoped_inventory
    ):
        limitations.add("authoritative_source_unavailable")
    if any(
        source.authority_role != "authoritative"
        and source.availability != "available"
        for source in scoped_inventory
    ):
        limitations.add("optional_source_unavailable")
    if not eligible and scoped_inventory:
        limitations.add("required_capability_unavailable")
    return limitations


def _shape_limitations(
    *,
    task_shape: str,
    scope: EvidenceDeclaredScope,
    eligible: list[EvidenceSourceDescriptor],
    authoritative: list[EvidenceSourceDescriptor],
    strategy: EvidenceAcquisitionStrategy | None,
) -> set[EvidencePlanLimitationCode]:
    limitations: set[EvidencePlanLimitationCode] = set()
    if task_shape in _AUTHORITATIVE_INVENTORY_SHAPES and not authoritative:
        limitations.add("authoritative_source_missing")
    if strategy is None:
        limitations.add("required_capability_unavailable")

    if task_shape == "bounded_exhaustive_review" and strategy is None:
        if _has_capability(eligible, "targeted_retrieval"):
            limitations.add("targeted_only_not_exhaustive")
    elif task_shape == "absence_or_coverage_check":
        if (
            scope.inventory_status != "complete_for_declared_scope"
            or not authoritative
            or strategy is None
        ):
            limitations.add("absence_scope_not_enumerable")
    elif task_shape == "cross_source_comparison" and len(eligible) < 2:
        limitations.add("insufficient_comparison_scope")
    elif task_shape == "contradiction_review" and strategy is None:
        limitations.add("contradiction_search_not_supported")
    elif task_shape == "historical_reconstruction":
        if not scope.time_scope_ref:
            limitations.add("historical_time_scope_missing")
        if strategy is None:
            limitations.add("historical_sequence_not_supported")
    elif task_shape == "recommendation_or_decision_support" and (
        len(eligible) < 2 or strategy is None
    ):
        limitations.add("decision_support_scope_insufficient")
    return limitations


def _material_plan_supported(
    *,
    task_shape: str,
    scope: EvidenceDeclaredScope,
    eligible: list[EvidenceSourceDescriptor],
    authoritative: list[EvidenceSourceDescriptor],
    strategy: EvidenceAcquisitionStrategy | None,
    limitations: set[EvidencePlanLimitationCode],
) -> bool:
    if strategy is None or not eligible:
        return False
    if "declared_source_missing_from_inventory" in limitations:
        return False
    if task_shape in _COMPLETE_INVENTORY_SHAPES:
        if scope.inventory_status != "complete_for_declared_scope":
            return False
        if not authoritative:
            return False
        if "authoritative_source_unavailable" in limitations:
            return False
        if "declared_source_missing_from_inventory" in limitations:
            return False
    if task_shape in {"cross_source_comparison", "recommendation_or_decision_support"}:
        if len(eligible) < 2:
            return False
        if scope.source_ids and len(eligible) != len(scope.source_ids):
            return False
    if task_shape == "contradiction_review":
        one_full_authoritative = any(
            source.authority_role == "authoritative"
            and "bounded_full_context" in source.capabilities
            for source in eligible
        )
        if len(eligible) < 2 and not one_full_authoritative:
            return False
    if task_shape == "historical_reconstruction" and not scope.time_scope_ref:
        return False
    return True


def _declared_requirements(
    *,
    task_shape: str,
    strategy: EvidenceAcquisitionStrategy | None,
    exact_authoritative_fetch: bool,
    optional_scope_incomplete: bool,
) -> list[EvidenceRequirement]:
    kinds = list(_MATERIAL_REQUIREMENT_KINDS[task_shape])
    if (
        task_shape == "targeted_lookup"
        and strategy == "exact_fetch"
        and exact_authoritative_fetch
        and "exact_authoritative_fetch" not in kinds
    ):
        kinds.append("exact_authoritative_fetch")
    requirements = [
        EvidenceRequirement(
            requirement_id=f"requirement-{kind.replace('_', '-')}",
            requirement_kind=kind,
            criticality="material",
        )
        for kind in kinds
    ]
    if optional_scope_incomplete:
        requirements.append(
            EvidenceRequirement(
                requirement_id="optional-selected-source-coverage",
                requirement_kind="selected_source_coverage",
                criticality="optional",
            )
        )
    return sorted(requirements, key=lambda requirement: requirement.requirement_id)


def _plan_status(
    material_supported: bool,
    limitations: set[EvidencePlanLimitationCode],
) -> EvidencePlanStatus:
    if not material_supported:
        return "unsupported"
    if limitations:
        return "ready_with_limitations"
    return "ready"


def _safe_summary(status: EvidencePlanStatus) -> str:
    return {
        "ready": "A strategy is available for the declared evidence scope.",
        "ready_with_limitations": (
            "A usable strategy exists, but some optional or supplemental scope "
            "is unavailable."
        ),
        "unsupported": (
            "The available source inventory or capabilities cannot safely support "
            "the declared evidence task."
        ),
    }[status]


def _plan_identity(
    *,
    body: EvidencePlanCompileRequest,
    question_digest: str,
    scope: EvidenceDeclaredScope,
    inventory: list[EvidenceSourceDescriptor],
    result_fields: dict[str, object],
) -> str:
    material = {
        "request_id": body.request_id,
        "owner_id": body.owner_id,
        "conversation_id": body.conversation_id,
        "surface": body.surface,
        "runtime_session_id": body.runtime_session_id,
        "runtime_turn_id": body.runtime_turn_id,
        "question_anchor_digest": question_digest,
        "task_shape": body.task_shape,
        "declared_scope": scope.model_dump(mode="json"),
        "source_inventory": [source.model_dump(mode="json") for source in inventory],
        **result_fields,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"evidence_plan_{sha256(encoded.encode('utf-8')).hexdigest()[:32]}"


def compile_evidence_plan(
    body: EvidencePlanCompileRequest,
) -> EvidencePlanCompileResponse:
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

    scope = _normalized_scope(body.declared_scope)
    inventory = _normalized_inventory(body.source_inventory)
    eligible = _eligible_sources(inventory, scope)
    authoritative = [
        source for source in eligible if source.authority_role == "authoritative"
    ]
    strategy = _select_strategy(
        task_shape=body.task_shape,
        scope=scope,
        eligible=eligible,
    )
    limitations = _base_limitations(
        scope=scope,
        inventory=inventory,
        eligible=eligible,
    )
    limitations.update(
        _shape_limitations(
            task_shape=body.task_shape,
            scope=scope,
            eligible=eligible,
            authoritative=authoritative,
            strategy=strategy,
        )
    )
    if (
        body.task_shape == "absence_or_coverage_check"
        and "authoritative_source_unavailable" in limitations
    ):
        limitations.add("absence_scope_not_enumerable")
    material_supported = _material_plan_supported(
        task_shape=body.task_shape,
        scope=scope,
        eligible=eligible,
        authoritative=authoritative,
        strategy=strategy,
        limitations=limitations,
    )
    status = _plan_status(material_supported, limitations)
    optional_scope_limitations = {
        "optional_source_unavailable",
        "source_inventory_partial",
        "source_inventory_unknown",
        "source_inventory_unavailable",
    }
    if body.task_shape not in _COMPLETE_INVENTORY_SHAPES:
        optional_scope_limitations.add("authoritative_source_unavailable")
    optional_scope_incomplete = material_supported and bool(
        limitations & optional_scope_limitations
    )
    exact_authoritative_fetch = any(
        strategy == "exact_fetch"
        and source.authority_role == "authoritative"
        and "exact_fetch" in source.capabilities
        for source in eligible
    )
    requirements = _declared_requirements(
        task_shape=body.task_shape,
        strategy=strategy,
        exact_authoritative_fetch=exact_authoritative_fetch,
        optional_scope_incomplete=optional_scope_incomplete,
    )
    selected_strategies = sorted([strategy] if strategy is not None else [])
    limitation_codes = sorted(limitations)
    completeness = _COMPLETENESS_BY_SHAPE[body.task_shape]
    contradiction_required = body.task_shape in _CONTRADICTION_SHAPES
    question_digest = f"sha256:{sha256(body.question_anchor.encode('utf-8')).hexdigest()}"
    identity_fields: dict[str, object] = {
        "completeness_expectation": completeness,
        "contradiction_search_required": contradiction_required,
        "eligible_source_ids": [source.source_id for source in eligible],
        "authoritative_source_ids": [source.source_id for source in authoritative],
        "selected_strategies": selected_strategies,
        "declared_requirements": [
            requirement.model_dump(mode="json") for requirement in requirements
        ],
        "limitation_codes": limitation_codes,
    }
    result = EvidencePlanResult(
        plan_id=_plan_identity(
            body=body,
            question_digest=question_digest,
            scope=scope,
            inventory=inventory,
            result_fields=identity_fields,
        ),
        question_anchor=body.question_anchor,
        question_anchor_digest=question_digest,
        task_shape=body.task_shape,
        plan_status=status,
        completeness_expectation=completeness,
        contradiction_search_required=contradiction_required,
        eligible_source_ids=identity_fields["eligible_source_ids"],
        authoritative_source_ids=identity_fields["authoritative_source_ids"],
        selected_strategies=selected_strategies,
        declared_requirements=requirements,
        limitation_codes=limitation_codes,
        user_safe_summary=_safe_summary(status),
    )

    record_runtime_event(
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        event_type="evidence_plan_compiled",
        event_payload_json={
            "request_id": body.request_id,
            "runtime_session_id": body.runtime_session_id,
            "runtime_turn_id": body.runtime_turn_id,
            "plan_id": result.plan_id,
            "question_anchor_digest": result.question_anchor_digest,
            "task_shape": result.task_shape,
            "plan_status": result.plan_status,
            "completeness_expectation": result.completeness_expectation,
            "contradiction_search_required": result.contradiction_search_required,
            "source_inventory_count": len(inventory),
            "exact_source_reference_count": len(scope.exact_source_refs),
            "eligible_source_count": len(result.eligible_source_ids),
            "authoritative_source_count": len(result.authoritative_source_ids),
            "material_requirement_count": sum(
                requirement.criticality == "material"
                for requirement in result.declared_requirements
            ),
            "optional_requirement_count": sum(
                requirement.criticality == "optional"
                for requirement in result.declared_requirements
            ),
            "selected_strategies": result.selected_strategies,
            "limitation_codes": result.limitation_codes,
        },
    )
    return EvidencePlanCompileResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        result=result,
    )
