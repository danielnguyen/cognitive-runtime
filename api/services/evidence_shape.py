from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256

from models import (
    AggregateSpec,
    EvidenceShapeDerivationStatus,
    EvidenceShapeDeriveRequest,
    EvidenceShapeDeriveResponse,
    EvidenceShapeReasonCode,
    EvidenceShapeResult,
    EvidenceTaskShape,
    SourceDiscoveryContext,
    SourceDiscoveryEntry,
    SourceMatchResult,
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
_NON_VERIFICATION_FRAMING = re.compile(
    r"^(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
    r"(?:explain|describe|define|write|draft|create|brainstorm|"
    r"review|edit|proofread|tell\s+me\s+(?:a|another)\s+joke)\b",
    re.IGNORECASE,
)
_COMPATIBILITY_RELATIONSHIP_QUESTION = re.compile(
    r"\b(?:will|would|can|could|does|do|is|are)\b.{0,180}\b(?:"
    r"compatible\s+with|interchangeable(?:\s+with)?|"
    r"work\s+(?:with|in|on|together)|fit(?:\s+(?:with|in|on|into))?|"
    r"supports?\b|be\s+used\s+with)\b",
    re.IGNORECASE,
)
_COMPATIBILITY_ARTIFACT = re.compile(
    r"\b(?:parts?|components?|modules?|models?|packages?|versions?|adapters?|"
    r"devices?|runtimes?|platforms?)\b",
    re.IGNORECASE,
)
_DIRECT_FIT_ENDING = re.compile(
    r"\bfit\b\s*[?.!]*$",
    re.IGNORECASE,
)
_PAIRED_COMPATIBILITY = re.compile(
    r"\b(?:two|both|these|those)\b.{0,100}\b(?:"
    r"interchangeable|work\s+together)\b",
    re.IGNORECASE,
)
_SUPPORT_RELATIONSHIP = re.compile(
    r"\bsupports?\b",
    re.IGNORECASE,
)
_VERSIONED_TARGET = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9.+-]*\s+)?\d+(?:\.\d+)+\b",
    re.IGNORECASE,
)
_CURRENT_STATE_QUESTION = re.compile(
    r"\b(?:is|are|was|were|has|have|does|do)\b"
    r"(?=.{0,180}\b(?:this|that|these|those|current|new|capabilit(?:y|ies)|"
    r"implementations?|deployments?|changes?|paths?|repositories?|code|"
    r"boundar(?:y|ies))\b)"
    r".{0,180}\b(?:implemented|wired\s+end[ -]to[ -]end|deployed|enforces?)\b",
    re.IGNORECASE,
)
_PERFORMED_VALIDATION_QUESTION = re.compile(
    r"\b(?:was|were|has|have|did)\b"
    r"(?=.{0,180}\b(?:implementations?|code|request\s+paths?|"
    r"validation\s+paths?|changes?|repositories?|deployments?|"
    r"hosted\s+checks?|final\s+heads?)\b)"
    r".{0,180}\b(?:review(?:ed)?|validat(?:e|ed)|test(?:ed)?|"
    r"checks?\s+run|run\s+against)\b",
    re.IGNORECASE,
)

_IDENTITY_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_IDENTITY_NOISE = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "look",
        "my",
        "of",
        "on",
        "or",
        "our",
        "the",
        "to",
        "with",
    }
)
_GENERIC_SOURCE_TOKENS = frozenset(
    {
        "calendar",
        "data",
        "google",
        "ics",
        "log",
        "logs",
        "record",
        "records",
        "sheet",
        "sheets",
        "source",
        "sources",
        "spreadsheet",
    }
)
_SOURCE_KIND_TOKENS = {
    "calendar": "calendar",
    "log": "log",
    "logs": "log",
    "record": "record",
    "records": "record",
}

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
_SEMANTIC_OPERATION_SHAPES: dict[str, EvidenceTaskShape] = {
    "lookup": "targeted_lookup",
    "latest": "targeted_lookup",
    "comparison": "cross_source_comparison",
    "exhaustive_review": "bounded_exhaustive_review",
    "contradiction_review": "contradiction_review",
    "absence_check": "absence_or_coverage_check",
    "historical_reconstruction": "historical_reconstruction",
    "decision_support": "recommendation_or_decision_support",
}
_AMBIGUOUS_PROBE_TASK_SHAPES: frozenset[EvidenceTaskShape] = frozenset(
    {
        "targeted_lookup",
        "cross_source_comparison",
        "contradiction_review",
        "recommendation_or_decision_support",
    }
)


@dataclass(frozen=True)
class _DeterministicSourceMatch:
    result: SourceMatchResult
    ambiguous_has_distinct_candidate: bool = False
    exact_matched_source_ids: tuple[str, ...] = ()


def _normalized_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _IDENTITY_TOKEN.findall(value))


def _specific_tokens(value: str) -> set[str]:
    return set(_normalized_tokens(value)) - _IDENTITY_NOISE - _GENERIC_SOURCE_TOKENS


def _source_kind_tokens(value: str) -> set[str]:
    return {
        canonical
        for token in _normalized_tokens(value)
        if (canonical := _SOURCE_KIND_TOKENS.get(token)) is not None
    }


def _contains_token_sequence(
    task_tokens: tuple[str, ...],
    candidate_tokens: tuple[str, ...],
) -> bool:
    if not candidate_tokens or len(candidate_tokens) > len(task_tokens):
        return False
    width = len(candidate_tokens)
    return any(
        task_tokens[index : index + width] == candidate_tokens
        for index in range(len(task_tokens) - width + 1)
    )


def _source_identity_fields(source: SourceDiscoveryEntry) -> dict[str, set[str]]:
    scope_values = (
        []
        if source.scope_refs is None
        else [
            value
            for field in ("time", "version", "domain", "project")
            if (value := getattr(source.scope_refs, field)) is not None
        ]
    )
    return {
        "source_id_match": _specific_tokens(source.source_id),
        "display_name_match": _specific_tokens(source.display_name),
        "domain_tag_match": set().union(
            *(_specific_tokens(value) for value in source.domain_tags),
            set(),
        ),
        "scope_reference_match": set().union(
            *(_specific_tokens(value) for value in scope_values),
            set(),
        ),
    }


def _source_identity_kinds(source: SourceDiscoveryEntry) -> set[str]:
    return _source_kind_tokens(source.source_id) | _source_kind_tokens(
        source.display_name
    )


def _derive_deterministic_source_match(
    body: EvidenceShapeDeriveRequest,
) -> _DeterministicSourceMatch | None:
    discovery = body.task_context.source_discovery
    if discovery is None:
        return None
    if discovery.inventory_status == "unknown":
        return _DeterministicSourceMatch(
            result=SourceMatchResult(
                status="inventory_unavailable",
                matched_source_ids=[],
                reason_codes=["inventory_unknown"],
            )
        )
    if discovery.inventory_status == "unavailable":
        return _DeterministicSourceMatch(
            result=SourceMatchResult(
                status="inventory_unavailable",
                matched_source_ids=[],
                reason_codes=["inventory_unavailable"],
            )
        )

    task_tokens = _normalized_tokens(body.task_text)
    task_specific = set(task_tokens) - _IDENTITY_NOISE - _GENERIC_SOURCE_TOKENS
    task_source_kinds = _source_kind_tokens(body.task_text)
    sources = sorted(discovery.sources, key=lambda source: source.source_id)
    fields_by_source = {
        source.source_id: _source_identity_fields(source) for source in sources
    }
    token_sources: dict[str, set[str]] = {}
    for source_id, fields in fields_by_source.items():
        for token in set().union(*fields.values(), set()):
            token_sources.setdefault(token, set()).add(source_id)

    candidates: list[dict[str, object]] = []
    for source in sources:
        fields = fields_by_source[source.source_id]
        matched_by_field = {
            reason: tokens & task_specific
            for reason, tokens in fields.items()
            if tokens & task_specific
        }
        matched_tokens = set().union(*matched_by_field.values(), set())
        exact_source_id = bool(fields["source_id_match"]) and _contains_token_sequence(
            task_tokens,
            _normalized_tokens(source.source_id),
        )
        exact_display_name = bool(fields["display_name_match"]) and (
            _contains_token_sequence(task_tokens, _normalized_tokens(source.display_name))
        )
        unique_match = any(
            len(token_sources[token]) == 1 for token in matched_tokens
        )
        exact_match = exact_source_id or exact_display_name
        multi_token_match = len(matched_tokens) >= 2
        strong = exact_match or multi_token_match
        if matched_tokens:
            candidates.append(
                {
                    "source_id": source.source_id,
                    "strong": strong,
                    "distinct": exact_match or (multi_token_match and unique_match),
                    "exact_match": exact_match,
                    "reasons": set(matched_by_field),
                    "source_kinds": _source_identity_kinds(source),
                }
            )

    strong_candidates = [candidate for candidate in candidates if candidate["strong"]]
    reasons: set[str] = set()
    status = "no_match"
    matched_source_ids: list[str] = []
    if len(strong_candidates) == 1:
        status = "matched"
        matched_source_ids = [str(strong_candidates[0]["source_id"])]
        reasons.update(strong_candidates[0]["reasons"])
    elif len(strong_candidates) > 1:
        if all(candidate["distinct"] for candidate in strong_candidates):
            status = "matched"
            matched_source_ids = sorted(
                str(candidate["source_id"]) for candidate in strong_candidates
            )
            reasons.add("multiple_explicit_source_matches")
            for candidate in strong_candidates:
                reasons.update(candidate["reasons"])
        elif any(candidate["distinct"] for candidate in strong_candidates):
            status = "ambiguous"
            reasons.add("multiple_possible_source_matches")
        else:
            kind_matches = [
                candidate
                for candidate in strong_candidates
                if candidate["source_kinds"] & task_source_kinds
            ]
            if len(kind_matches) == 1:
                status = "matched"
                matched_source_ids = [str(kind_matches[0]["source_id"])]
                reasons.update(kind_matches[0]["reasons"])
            else:
                status = "ambiguous"
                reasons.add("multiple_possible_source_matches")
    elif len(candidates) > 1:
        status = "ambiguous"
        reasons.add("multiple_possible_source_matches")
    else:
        reasons.add("no_source_specific_match")
        connector_tokens = set().union(
            *(_normalized_tokens(source.connector) for source in sources),
            set(),
        )
        if (set(task_tokens) & _GENERIC_SOURCE_TOKENS) or (
            set(task_tokens) & connector_tokens
        ):
            reasons.add("generic_source_signal_rejected")

    if discovery.inventory_status == "partial":
        reasons.add("inventory_partial")
        if status == "no_match":
            status = "inventory_unavailable"
    return _DeterministicSourceMatch(
        result=SourceMatchResult(
            status=status,
            matched_source_ids=matched_source_ids,
            reason_codes=sorted(reasons),
        ),
        ambiguous_has_distinct_candidate=(
            status == "ambiguous"
            and any(candidate["distinct"] for candidate in strong_candidates)
        ),
        exact_matched_source_ids=tuple(
            sorted(
                str(candidate["source_id"])
                for candidate in strong_candidates
                if candidate["exact_match"]
                and candidate["source_id"] in matched_source_ids
            )
        ),
    )


def _authorized_probe_source_ids(
    discovery: SourceDiscoveryContext,
    *,
    candidate_source_ids: list[str],
    effective_task_shape: EvidenceTaskShape | None,
) -> list[str]:
    if (
        discovery.inventory_status != "complete"
        or effective_task_shape not in _AMBIGUOUS_PROBE_TASK_SHAPES
    ):
        return []
    sources_by_id = {source.source_id: source for source in discovery.sources}
    candidates = [sources_by_id[source_id] for source_id in candidate_source_ids]
    if not all(
        source.availability == "available" and "search" in source.capabilities
        for source in candidates
    ):
        return []
    return sorted(candidate_source_ids)


def _derive_source_match(
    body: EvidenceShapeDeriveRequest,
    *,
    effective_task_shape: EvidenceTaskShape | None,
) -> SourceMatchResult | None:
    deterministic = _derive_deterministic_source_match(body)
    if deterministic is None:
        return None
    advisory = body.task_context.semantic_advisory
    discovery = body.task_context.source_discovery
    if advisory is None:
        return deterministic.result
    if discovery.inventory_status in {"unknown", "unavailable"}:
        return deterministic.result
    if deterministic.result.status == "matched":
        if advisory.interpretation_status == "no_match":
            if deterministic.exact_matched_source_ids:
                return deterministic.result
            return SourceMatchResult(
                status=(
                    "inventory_unavailable"
                    if discovery.inventory_status == "partial"
                    else "no_match"
                ),
                matched_source_ids=[],
                reason_codes=sorted(
                    {"semantic_no_match"}
                    | (
                        {"inventory_partial"}
                        if discovery.inventory_status == "partial"
                        else set()
                    )
                ),
            )
        if advisory.interpretation_status == "ambiguous":
            deterministic_source_ids = set(
                deterministic.result.matched_source_ids
            )
            semantic_source_ids = set(advisory.candidate_source_ids)
            if deterministic_source_ids <= semantic_source_ids:
                probe_source_ids = _authorized_probe_source_ids(
                    discovery,
                    candidate_source_ids=advisory.candidate_source_ids,
                    effective_task_shape=effective_task_shape,
                )
                if probe_source_ids:
                    return SourceMatchResult(
                        status="ambiguous",
                        matched_source_ids=[],
                        probe_source_ids=probe_source_ids,
                        reason_codes=["semantic_candidates_ambiguous"],
                    )
        return deterministic.result
    if (
        deterministic.result.status == "ambiguous"
        and deterministic.ambiguous_has_distinct_candidate
    ):
        return deterministic.result

    partial_reasons = (
        {"inventory_partial"}
        if discovery.inventory_status == "partial"
        else set()
    )
    if advisory.interpretation_status == "resolved":
        return SourceMatchResult(
            status="matched",
            matched_source_ids=sorted(advisory.candidate_source_ids),
            reason_codes=sorted({"semantic_candidate_validated"} | partial_reasons),
        )
    if advisory.interpretation_status == "ambiguous":
        probe_source_ids = _authorized_probe_source_ids(
            discovery,
            candidate_source_ids=advisory.candidate_source_ids,
            effective_task_shape=effective_task_shape,
        )
        return SourceMatchResult(
            status="ambiguous",
            matched_source_ids=[],
            probe_source_ids=probe_source_ids,
            reason_codes=sorted(
                {"semantic_candidates_ambiguous"} | partial_reasons
            ),
        )
    if deterministic.result.status == "ambiguous":
        return SourceMatchResult(
            status="ambiguous",
            matched_source_ids=[],
            reason_codes=sorted(
                set(deterministic.result.reason_codes) | {"semantic_no_match"}
            ),
        )
    return SourceMatchResult(
        status=(
            "inventory_unavailable"
            if discovery.inventory_status == "partial"
            else "no_match"
        ),
        matched_source_ids=[],
        reason_codes=sorted({"semantic_no_match"} | partial_reasons),
    )


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


def _effective_task_shape(
    body: EvidenceShapeDeriveRequest,
    *,
    specialized: set[EvidenceTaskShape],
) -> EvidenceTaskShape | None:
    context = body.task_context
    semantic_operation = (
        context.semantic_advisory.operation_hint
        if context.semantic_advisory is not None
        else None
    )
    if semantic_operation == "aggregate":
        return "aggregate"
    if len(specialized) > 1:
        return _COMPATIBLE_COMBINATIONS.get(frozenset(specialized))
    if len(specialized) == 1:
        return next(iter(specialized))
    if context.continuation_of_prior_evidence_task:
        return context.prior_task_shape
    return _SEMANTIC_OPERATION_SHAPES.get(semantic_operation)


def _verification_dependent_request(text: str) -> bool:
    if _NON_VERIFICATION_FRAMING.search(text):
        return False
    compatibility_question = _COMPATIBILITY_RELATIONSHIP_QUESTION.search(text)
    compatibility_artifacts = list(_COMPATIBILITY_ARTIFACT.finditer(text))
    concrete_compatibility = False
    if compatibility_question is not None and compatibility_artifacts:
        direct_fit = _DIRECT_FIT_ENDING.search(text)
        support_relationship = _SUPPORT_RELATIONSHIP.search(text)
        concrete_compatibility = (
            len(compatibility_artifacts) >= 2
            or (
                direct_fit is not None
                and any(
                    artifact.start() < direct_fit.start()
                    for artifact in compatibility_artifacts
                )
            )
            or bool(_PAIRED_COMPATIBILITY.search(text))
            or (
                support_relationship is not None
                and any(
                    artifact.start() < support_relationship.start()
                    for artifact in compatibility_artifacts
                )
                and _VERSIONED_TARGET.search(
                    text,
                    pos=support_relationship.end(),
                )
                is not None
            )
        )
    return concrete_compatibility or any(
        pattern.search(text)
        for pattern in (_CURRENT_STATE_QUESTION, _PERFORMED_VALIDATION_QUESTION)
    )


def _materiality_reasons(
    body: EvidenceShapeDeriveRequest,
    *,
    explicit_evidence: bool,
    verification_dependent: bool,
) -> set[EvidenceShapeReasonCode]:
    context = body.task_context
    reasons: set[EvidenceShapeReasonCode] = set()
    if context.evidence_input_kinds:
        reasons.add("source_context_present")
    if context.external_verification_required or verification_dependent:
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
    aggregate_spec: AggregateSpec | None,
) -> str:
    context = body.task_context.model_dump(mode="json")
    context["evidence_input_kinds"] = sorted(context["evidence_input_kinds"])
    discovery = context.get("source_discovery")
    if discovery is None:
        context.pop("source_discovery", None)
    else:
        for source in discovery["sources"]:
            source["domain_tags"] = sorted(source["domain_tags"])
            source["capabilities"] = sorted(source["capabilities"])
        discovery["sources"] = sorted(
            discovery["sources"], key=lambda source: source["source_id"]
        )
    semantic_advisory = context.get("semantic_advisory")
    if semantic_advisory is None:
        context.pop("semantic_advisory", None)
    else:
        semantic_advisory["candidate_source_ids"] = sorted(
            semantic_advisory["candidate_source_ids"]
        )
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
    if aggregate_spec is not None:
        material["aggregate_spec"] = aggregate_spec.model_dump(mode="json")
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"evidence_shape_{sha256(encoded.encode('utf-8')).hexdigest()[:32]}"


def _build_result(
    body: EvidenceShapeDeriveRequest,
    *,
    status: EvidenceShapeDerivationStatus,
    task_shape: EvidenceTaskShape | None,
    candidates: list[EvidenceTaskShape],
    reasons: set[EvidenceShapeReasonCode],
    source_match: SourceMatchResult | None,
    aggregate_spec: AggregateSpec | None = None,
) -> EvidenceShapeResult:
    sorted_candidates = sorted(set(candidates))
    sorted_reasons = sorted(reasons)
    material = status != "not_applicable"
    clarification = status == "ambiguous"
    question_digest = f"sha256:{sha256(body.task_text.encode('utf-8')).hexdigest()}"
    result_fields = {
        "derivation_id": _derivation_identity(
            body,
            question_digest=question_digest,
            status=status,
            task_shape=task_shape,
            candidates=sorted_candidates,
            evidence_scope_material=material,
            clarification_required=clarification,
            reasons=sorted_reasons,
            aggregate_spec=aggregate_spec,
        ),
        "question_anchor": body.task_text,
        "question_anchor_digest": question_digest,
        "derivation_status": status,
        "task_shape": task_shape,
        "candidate_task_shapes": sorted_candidates,
        "evidence_scope_material": material,
        "clarification_required": clarification,
        "reason_codes": sorted_reasons,
        "user_safe_summary": _safe_summary(status),
        "source_match": source_match,
    }
    if aggregate_spec is not None:
        result_fields["aggregate_spec"] = aggregate_spec
    return EvidenceShapeResult.model_validate(result_fields)


def _derive_result(body: EvidenceShapeDeriveRequest) -> EvidenceShapeResult:
    specialized = _specialized_shapes(body.task_text)
    effective_task_shape = _effective_task_shape(body, specialized=specialized)
    source_match = _derive_source_match(
        body,
        effective_task_shape=effective_task_shape,
    )
    source_context_material = source_match is not None and source_match.status in {
        "matched",
        "ambiguous",
    }
    explicit_evidence = _explicit_evidence_language(
        body.task_text,
        specialized=specialized,
    )
    verification_dependent = _verification_dependent_request(body.task_text)
    distinct_evidence_request = explicit_evidence or verification_dependent
    context = body.task_context
    semantic_operation = (
        context.semantic_advisory.operation_hint
        if context.semantic_advisory is not None
        else None
    )
    semantic_operation_material = semantic_operation not in {None, "unknown"}
    reasons = _materiality_reasons(
        body,
        explicit_evidence=explicit_evidence,
        verification_dependent=verification_dependent,
    )
    if source_context_material:
        reasons.add("source_context_present")
    context_material = bool(
        context.evidence_input_kinds
        or context.external_verification_required
        or context.freshness_sensitive
        or context.high_stakes_accuracy_required
        or context.continuation_of_prior_evidence_task
    )
    evidence_material = (
        context_material
        or distinct_evidence_request
        or source_context_material
        or semantic_operation_material
    )
    non_evidence_interaction = body.interaction_kind in {
        "joke_or_playful",
        "vent_or_expression",
    }
    if (
        non_evidence_interaction
        and not distinct_evidence_request
        and not source_context_material
        and not semantic_operation_material
    ):
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
            source_match=source_match,
        )

    if semantic_operation == "aggregate":
        advisory = context.semantic_advisory
        aggregate_authorized = bool(
            advisory is not None
            and advisory.interpretation_status == "resolved"
            and advisory.aggregate_function is not None
            and advisory.aggregate_field_name is not None
            and context.source_discovery is not None
            and context.source_discovery.inventory_status == "complete"
            and source_match is not None
            and source_match.status == "matched"
            and source_match.matched_source_ids
            == sorted(advisory.candidate_source_ids)
        )
        reasons.add("semantic_operation_hint")
        if aggregate_authorized:
            aggregate_spec = AggregateSpec(
                function=advisory.aggregate_function,
                field_name=advisory.aggregate_field_name,
            )
            return _build_result(
                body,
                status="derived",
                task_shape="aggregate",
                candidates=["aggregate"],
                reasons=reasons,
                source_match=source_match,
                aggregate_spec=aggregate_spec,
            )
        reasons.add("semantic_operation_unsupported")
        return _build_result(
            body,
            status="ambiguous",
            task_shape=None,
            candidates=[],
            reasons=reasons,
            source_match=source_match,
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
                source_match=source_match,
            )
        selected = effective_task_shape
    elif len(specialized) == 1:
        selected = effective_task_shape
    elif context.continuation_of_prior_evidence_task:
        if context.prior_task_shape == "aggregate":
            reasons.add("prior_shape_inherited")
            return _build_result(
                body,
                status="ambiguous",
                task_shape=None,
                candidates=[],
                reasons=reasons,
                source_match=source_match,
            )
        selected = effective_task_shape
        reasons.add("prior_shape_inherited")
    elif semantic_operation in _SEMANTIC_OPERATION_SHAPES:
        selected = effective_task_shape
        reasons.add("semantic_operation_hint")
        if selected == "targeted_lookup":
            reasons.add("targeted_lookup_derived")
        else:
            reasons.add(_SPECIALIZED_REASON[selected])
    elif (
        body.interaction_kind == "ambiguous"
        and not distinct_evidence_request
        and not source_context_material
    ):
        reasons.add("ambiguous_interaction_without_shape_signal")
        return _build_result(
            body,
            status="ambiguous",
            task_shape=None,
            candidates=[],
            reasons=reasons,
            source_match=source_match,
        )
    elif (
        body.interaction_kind == "brainstorm"
        and not distinct_evidence_request
        and not source_context_material
    ):
        reasons.add("ambiguous_interaction_without_shape_signal")
        return _build_result(
            body,
            status="ambiguous",
            task_shape=None,
            candidates=[],
            reasons=reasons,
            source_match=source_match,
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
        source_match=source_match,
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
    event_payload = {
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
    }
    if body.task_context.source_discovery is not None:
        event_payload.update(
            {
                "source_match_status": result.source_match.status,
                "source_match_reason_codes": result.source_match.reason_codes,
                "configured_inventory_status": (
                    body.task_context.source_discovery.inventory_status
                ),
                "configured_source_count": len(
                    body.task_context.source_discovery.sources
                ),
            }
        )
        if result.source_match.status == "matched":
            event_payload["matched_source_ids"] = (
                result.source_match.matched_source_ids
            )
        if result.source_match.probe_source_ids:
            event_payload["probe_source_count"] = len(
                result.source_match.probe_source_ids
            )
    if body.task_context.semantic_advisory is not None:
        event_payload.update(
            {
                "semantic_interpretation_status": (
                    body.task_context.semantic_advisory.interpretation_status
                ),
                "semantic_operation_hint": (
                    body.task_context.semantic_advisory.operation_hint
                ),
                "semantic_candidate_count": len(
                    body.task_context.semantic_advisory.candidate_source_ids
                ),
            }
        )
    record_runtime_event(
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        event_type="evidence_shape_derived",
        event_payload_json=event_payload,
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
