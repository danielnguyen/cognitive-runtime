from __future__ import annotations

import json
from itertools import count

import pytest
from fastapi.testclient import TestClient
from main import app
from models import (
    EvidenceShapeDeriveRequest,
    SemanticEvidenceAdvisory,
    SourceDiscoveryEntry,
    SourceMatchResult,
)
from pydantic import ValidationError
from services.evidence_shape import _derive_result

client = TestClient(app)
_runtime_counter = count()


def _start_runtime() -> dict[str, object]:
    ordinal = next(_runtime_counter)
    response = client.post(
        "/v1/runtime/turns/start",
        json={
            "request_id": f"request-shape-start-{ordinal}",
            "owner_id": f"owner-shape-{ordinal}",
            "conversation_id": f"conversation-shape-{ordinal}",
            "surface": "web",
        },
    )
    assert response.status_code == 200
    body = response.json()
    return {
        "request_id": f"request-shape-derive-{ordinal}",
        "owner_id": f"owner-shape-{ordinal}",
        "conversation_id": f"conversation-shape-{ordinal}",
        "surface": "web",
        "runtime_session_id": body["runtime_session"]["runtime_session_id"],
        "runtime_turn_id": body["runtime_turn"]["runtime_turn_id"],
    }


def _context(
    *,
    evidence_input_kinds: list[str] | None = None,
    external_verification_required: bool = False,
    freshness_sensitive: bool = False,
    high_stakes_accuracy_required: bool = False,
    continuation_of_prior_evidence_task: bool = False,
    prior_task_shape: str | None = None,
    source_discovery: dict[str, object] | None = None,
    semantic_advisory: dict[str, object] | None = None,
) -> dict[str, object]:
    context: dict[str, object] = {
        "evidence_input_kinds": evidence_input_kinds or [],
        "external_verification_required": external_verification_required,
        "freshness_sensitive": freshness_sensitive,
        "high_stakes_accuracy_required": high_stakes_accuracy_required,
        "continuation_of_prior_evidence_task": continuation_of_prior_evidence_task,
        "prior_task_shape": prior_task_shape,
    }
    if source_discovery is not None:
        context["source_discovery"] = source_discovery
    if semantic_advisory is not None:
        context["semantic_advisory"] = semantic_advisory
    return context


def _discovery_source(
    source_id: str,
    display_name: str,
    *,
    connector: str = "google_sheets",
    domain_tags: list[str] | None = None,
    scope_refs: dict[str, str] | None = None,
    content_fields: list[str] | None = None,
    capabilities: list[str] | None = None,
    availability: str = "available",
    authority_role: str = "authoritative",
) -> dict[str, object]:
    source: dict[str, object] = {
        "source_id": source_id,
        "display_name": display_name,
        "connector": connector,
        "domain_tags": domain_tags or [],
        "capabilities": capabilities or ["search", "fetch"],
        "availability": availability,
        "authority_role": authority_role,
    }
    if scope_refs is not None:
        source["scope_refs"] = scope_refs
    if content_fields is not None:
        source["content_fields"] = content_fields
    return source


def _discovery(
    *sources: dict[str, object],
    inventory_status: str = "complete",
) -> dict[str, object]:
    return {"inventory_status": inventory_status, "sources": list(sources)}


def _semantic_advisory(
    interpretation_status: str,
    operation_hint: str,
    *candidate_source_ids: str,
    aggregate_function: str | None = None,
    aggregate_field_name: str | None = None,
) -> dict[str, object]:
    advisory: dict[str, object] = {
        "interpretation_status": interpretation_status,
        "operation_hint": operation_hint,
        "candidate_source_ids": list(candidate_source_ids),
    }
    if aggregate_function is not None:
        advisory["aggregate_function"] = aggregate_function
    if aggregate_field_name is not None:
        advisory["aggregate_field_name"] = aggregate_field_name
    return advisory


def _source_kind_topology() -> tuple[dict[str, object], ...]:
    return (
        _discovery_source(
            "maintenance_log_primary",
            "Maintenance Log - Primary",
            domain_tags=["vehicle", "maintenance"],
            scope_refs={"domain": "vehicle-maintenance", "project": "trail-unit"},
        ),
        _discovery_source(
            "scheduled_actions_trail_unit",
            "Scheduled Actions - Trail Unit",
            domain_tags=["vehicle", "maintenance"],
            scope_refs={"domain": "vehicle-maintenance", "project": "trail-unit"},
        ),
        _discovery_source(
            "maintenance_log_electric",
            "Maintenance Log - Electric",
            domain_tags=["vehicle", "maintenance"],
            scope_refs={
                "domain": "vehicle-maintenance",
                "project": "electric-platform",
            },
        ),
    )


def _derive(
    runtime: dict[str, object],
    *,
    task_text: str = "Check the maintenance record for the latest service date.",
    interaction_kind: str = "question",
    task_context: dict[str, object] | None = None,
    **overrides: object,
):
    payload: dict[str, object] = {
        **runtime,
        "task_text": task_text,
        "interaction_kind": interaction_kind,
        "task_context": task_context or _context(),
    }
    payload.update(overrides)
    return client.post("/v1/runtime/evidence-shapes/derive", json=payload)


def _shape_events(runtime_session_id: object) -> list[dict[str, object]]:
    response = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert response.status_code == 200
    return [
        event
        for event in response.json()["events"]
        if event["event_type"] == "evidence_shape_derived"
    ]


def _assert_not_applicable(result: dict[str, object]) -> None:
    assert result["derivation_status"] == "not_applicable"
    assert result["task_shape"] is None
    assert result["candidate_task_shapes"] == []
    assert result["evidence_scope_material"] is False
    assert result["clarification_required"] is False


def test_positive_request_is_strict_normalized_and_bounded():
    response = _derive(
        _start_runtime(),
        task_text="  Check   the record\nfor the latest date.  ",
        task_context=_context(evidence_input_kinds=["artifact"]),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["question_anchor"] == "Check the record for the latest date."
    assert result["question_anchor_digest"].startswith("sha256:")
    assert len(result["question_anchor_digest"]) == 71
    assert result["task_shape"] == "targeted_lookup"


@pytest.mark.parametrize(
    "extra_field",
    [
        "task_shape",
        "candidate_task_shapes",
        "derivation_status",
        "confidence",
        "reason_codes",
        "metadata",
        "prompt",
        "source_content",
        "advisory_allowed",
        "low_risk",
        "provider_disposition",
    ],
)
def test_caller_selected_or_unrestricted_fields_are_rejected(extra_field: str):
    response = _derive(_start_runtime(), **{extra_field: "caller-selected"})

    assert response.status_code == 422
    assert "extra_forbidden" in response.text


@pytest.mark.parametrize("extra_field", ["metadata", "prompt", "source_content"])
def test_unrestricted_task_context_fields_are_rejected(extra_field: str):
    context = _context()
    context[extra_field] = "not accepted"
    response = _derive(_start_runtime(), task_context=context)

    assert response.status_code == 422
    assert "extra_forbidden" in response.text


def test_duplicate_inputs_invalid_continuations_and_text_bounds_are_rejected():
    runtime = _start_runtime()
    duplicate_inputs = _derive(
        runtime,
        task_context=_context(evidence_input_kinds=["artifact", "artifact"]),
    )
    missing_prior = _derive(
        runtime,
        task_context=_context(continuation_of_prior_evidence_task=True),
    )
    unexpected_prior = _derive(
        runtime,
        task_context=_context(prior_task_shape="targeted_lookup"),
    )
    blank = _derive(runtime, task_text=" \n\t ")
    overlong = _derive(runtime, task_text="x" * 501)

    assert all(
        response.status_code == 422
        for response in (
            duplicate_inputs,
            missing_prior,
            unexpected_prior,
            blank,
            overlong,
        )
    )


@pytest.mark.parametrize(
    ("task_text", "interaction_kind"),
    [
        ("Write a short poem about rain.", "command"),
        ("Tell me a silly joke.", "joke_or_playful"),
        ("I am frustrated and need to vent.", "vent_or_expression"),
        ("Explain why leaves change color.", "question"),
        ("Recommend a funny movie.", "question"),
    ],
)
def test_ordinary_chat_remains_outside_evidence_planning(
    task_text: str,
    interaction_kind: str,
):
    result = _derive(
        _start_runtime(),
        task_text=task_text,
        interaction_kind=interaction_kind,
    ).json()["result"]

    _assert_not_applicable(result)
    assert "ordinary_chat_without_material_evidence_scope" in result["reason_codes"]


def test_playful_and_expressive_interactions_ignore_passive_source_context():
    for task_text, interaction_kind in (
        ("Tell a joke about the report.", "joke_or_playful"),
        ("These records are making me miserable.", "vent_or_expression"),
    ):
        result = _derive(
            _start_runtime(),
            task_text=task_text,
            interaction_kind=interaction_kind,
            task_context=_context(evidence_input_kinds=["artifact"]),
        ).json()["result"]
        _assert_not_applicable(result)


@pytest.mark.parametrize(
    "task_text",
    [
        "What is a report?",
        "Explain what a repository is.",
        "What are requirements?",
        "What options do I have?",
    ],
)
def test_evidence_adjacent_discussion_remains_ordinary_chat(task_text: str):
    result = _derive(_start_runtime(), task_text=task_text).json()["result"]

    _assert_not_applicable(result)


@pytest.mark.parametrize(
    ("task_text", "interaction_kind", "task_context"),
    [
        ("Check my grammar.", "command", _context()),
        (
            "Check out this joke.",
            "joke_or_playful",
            _context(evidence_input_kinds=["artifact"]),
        ),
    ],
)
def test_non_evidentiary_direct_operator_uses_remain_not_applicable(
    task_text: str,
    interaction_kind: str,
    task_context: dict[str, object],
):
    result = _derive(
        _start_runtime(),
        task_text=task_text,
        interaction_kind=interaction_kind,
        task_context=task_context,
    ).json()["result"]

    _assert_not_applicable(result)


@pytest.mark.parametrize(
    "task_text",
    [
        "Verify the record.",
        "What does this document say about the deadline?",
        "Summarize this report.",
    ],
)
def test_genuine_bounded_evidence_queries_derive_targeted_lookup(task_text: str):
    result = _derive(_start_runtime(), task_text=task_text).json()["result"]

    assert result["derivation_status"] == "derived"
    assert result["task_shape"] == "targeted_lookup"
    assert result["evidence_scope_material"] is True


@pytest.mark.parametrize(
    "task_text",
    [
        "Will this part fit?",
        "Will this 2001 control module work in the 2004 model?",
        "Are these two parts interchangeable?",
        "Does this package version support Python 3.14?",
        "Can this adapter be used with that device?",
        "Is this component compatible with that model?",
        "Will these two modules work together?",
        "Is this capability implemented in the current code?",
        "Is the new validation path wired end to end?",
        "Has this change been deployed?",
        "Does the repository currently enforce this boundary?",
        "Was this implementation reviewed end to end?",
        "Were hosted checks run against the final head?",
    ],
)
def test_verification_dependent_questions_derive_targeted_lookup_without_opt_in(
    task_text: str,
):
    runtime = _start_runtime()
    response = _derive(runtime, task_text=task_text)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["derivation_status"] == "derived"
    assert result["task_shape"] == "targeted_lookup"
    assert result["candidate_task_shapes"] == ["targeted_lookup"]
    assert result["evidence_scope_material"] is True
    assert result["clarification_required"] is False
    assert set(result["reason_codes"]) == {
        "external_verification_required",
        "targeted_lookup_derived",
    }


@pytest.mark.parametrize(
    "task_text",
    [
        "Does this sentence work in the paragraph?",
        "Does this name fit in the theme?",
        "Can this joke work in the presentation?",
        "Would this color work with the layout?",
        "Does this explanation work in context?",
        "Is this argument compatible with the premise?",
        "Does this workflow support collaboration?",
        "Can this schedule work with my availability?",
        "Does this plan work in theory?",
        "Will this wording fit in the introduction?",
        "Explain how a climate control module works.",
        "What does implementation mean?",
        "Write a compatibility checklist.",
        "Review this paragraph for grammar.",
        "Could you brainstorm names that fit this theme?",
        "Can you explain how this control module works in a system?",
        "Did that joke work?",
        "Tell me a joke about incompatible components.",
    ],
)
def test_verification_vocabulary_near_misses_remain_not_applicable(
    task_text: str,
):
    result = _derive(_start_runtime(), task_text=task_text).json()["result"]

    _assert_not_applicable(result)
    assert "external_verification_required" not in result["reason_codes"]


def test_verification_dependent_shape_event_is_bounded_and_raw_text_free():
    runtime = _start_runtime()
    task_text = "Does component build 91 support runtime 14?"
    result = _derive(runtime, task_text=task_text).json()["result"]
    event = _shape_events(runtime["runtime_session_id"])[0]
    payload = event["event_payload_json"]

    assert payload["question_anchor_digest"] == result["question_anchor_digest"]
    assert payload["task_shape"] == "targeted_lookup"
    assert payload["reason_codes"] == [
        "external_verification_required",
        "targeted_lookup_derived",
    ]
    serialized = json.dumps(payload, sort_keys=True).lower()
    assert task_text.lower() not in serialized
    assert "component build 91" not in serialized


def test_bounded_alternative_comparison_remains_evidence_material():
    result = _derive(
        _start_runtime(),
        task_text="Compare these versions.",
    ).json()["result"]

    assert result["derivation_status"] == "derived"
    assert result["task_shape"] == "cross_source_comparison"


def test_expressive_text_with_distinct_evidence_request_derives_lookup():
    result = _derive(
        _start_runtime(),
        task_text=(
            "I am frustrated, but verify what this report says about the deadline."
        ),
        interaction_kind="vent_or_expression",
    ).json()["result"]

    assert result["derivation_status"] == "derived"
    assert result["task_shape"] == "targeted_lookup"
    assert result["evidence_scope_material"] is True


def test_targeted_lookup_requires_material_evidence_scope():
    plain = _derive(
        _start_runtime(),
        task_text="What is the latest service date?",
    ).json()["result"]
    contextual = _derive(
        _start_runtime(),
        task_text="What is the latest service date?",
        task_context=_context(
            external_verification_required=True,
            freshness_sensitive=True,
        ),
    ).json()["result"]
    explicit = _derive(
        _start_runtime(),
        task_text="Verify the maintenance record for the latest service date.",
    ).json()["result"]

    _assert_not_applicable(plain)
    assert contextual["task_shape"] == "targeted_lookup"
    assert contextual["derivation_status"] == "derived"
    assert {
        "external_verification_required",
        "freshness_sensitive",
        "targeted_lookup_derived",
    } <= set(contextual["reason_codes"])
    assert explicit["task_shape"] == "targeted_lookup"


@pytest.mark.parametrize(
    ("task_text", "task_context", "expected_shape", "expected_reason"),
    [
        (
            "Check the maintenance record for the latest service date.",
            _context(),
            "targeted_lookup",
            "targeted_lookup_derived",
        ),
        (
            "Check whether every mandatory requirement in the checklist is implemented.",
            _context(),
            "bounded_exhaustive_review",
            "exhaustive_scope_requested",
        ),
        (
            "Compare these two reports and explain the differences between them.",
            _context(),
            "cross_source_comparison",
            "comparison_requested",
        ),
        (
            "Inspect whether these reports contradict each other.",
            _context(),
            "contradiction_review",
            "contradiction_requested",
        ),
        (
            "Check whether there is no record of the change in the declared logs.",
            _context(),
            "absence_or_coverage_check",
            "absence_scope_requested",
        ),
        (
            "Reconstruct what happened across the records last week.",
            _context(),
            "historical_reconstruction",
            "historical_reconstruction_requested",
        ),
        (
            "Which should I choose based on the evidence in these reports?",
            _context(high_stakes_accuracy_required=True),
            "recommendation_or_decision_support",
            "decision_support_requested",
        ),
    ],
)
def test_all_seven_broad_shapes_are_derived_from_distinct_neutral_semantics(
    task_text: str,
    task_context: dict[str, object],
    expected_shape: str,
    expected_reason: str,
):
    result = _derive(
        _start_runtime(),
        task_text=task_text,
        task_context=task_context,
    ).json()["result"]

    assert result["derivation_status"] == "derived"
    assert result["task_shape"] == expected_shape
    assert result["candidate_task_shapes"] == [expected_shape]
    assert expected_reason in result["reason_codes"]


@pytest.mark.parametrize(
    "task_text",
    [
        "Compare the two laboratory reports.",
        "Compare the two travel options using the available records.",
        "Explain the differences between these policy documents.",
    ],
)
def test_generic_operators_work_across_neutral_domains(task_text: str):
    result = _derive(_start_runtime(), task_text=task_text).json()["result"]

    assert result["task_shape"] == "cross_source_comparison"


@pytest.mark.parametrize(
    "task_text",
    [
        "Complete this sentence about the weather.",
        "History is interesting.",
        "No thanks, I am finished.",
    ],
)
def test_generic_shape_near_misses_do_not_trigger_evidence_planning(task_text: str):
    result = _derive(_start_runtime(), task_text=task_text).json()["result"]

    _assert_not_applicable(result)


@pytest.mark.parametrize(
    ("task_text", "expected_shape"),
    [
        (
            "Compare these candidate records and decide between the options.",
            "recommendation_or_decision_support",
        ),
        (
            "Compare these reports and identify conflicting records.",
            "contradiction_review",
        ),
        (
            "Check every report for conflicting records.",
            "bounded_exhaustive_review",
        ),
        (
            "Check the complete checklist for any missing requirements.",
            "absence_or_coverage_check",
        ),
    ],
)
def test_approved_compound_shapes_resolve_to_the_stricter_compatible_shape(
    task_text: str,
    expected_shape: str,
):
    result = _derive(_start_runtime(), task_text=task_text).json()["result"]

    assert result["derivation_status"] == "derived"
    assert result["task_shape"] == expected_shape


@pytest.mark.parametrize(
    "task_text",
    [
        "Reconstruct the timeline and compare the reports.",
        "Reconstruct the history and decide between these options.",
        "Compare every report in the complete set.",
        "Check for any missing evidence and decide between these options.",
    ],
)
def test_incompatible_compound_shapes_remain_ambiguous(task_text: str):
    result = _derive(_start_runtime(), task_text=task_text).json()["result"]

    assert result["derivation_status"] == "ambiguous"
    assert result["task_shape"] is None
    assert len(result["candidate_task_shapes"]) >= 2
    assert result["clarification_required"] is True
    assert "multiple_incompatible_shapes" in result["reason_codes"]


def test_ambiguous_interaction_requires_an_explicit_shape_or_lookup_signal():
    passive = _derive(
        _start_runtime(),
        task_text="This might matter later.",
        interaction_kind="ambiguous",
        task_context=_context(evidence_input_kinds=["artifact"]),
    ).json()["result"]
    explicit_lookup = _derive(
        _start_runtime(),
        task_text="Verify what this document says about the deadline.",
        interaction_kind="ambiguous",
    ).json()["result"]
    explicit_shape = _derive(
        _start_runtime(),
        task_text="Compare these two records.",
        interaction_kind="ambiguous",
    ).json()["result"]

    assert passive["derivation_status"] == "ambiguous"
    assert passive["task_shape"] is None
    assert "ambiguous_interaction_without_shape_signal" in passive["reason_codes"]
    assert explicit_lookup["task_shape"] == "targeted_lookup"
    assert explicit_shape["task_shape"] == "cross_source_comparison"


def test_continuation_inherits_only_when_current_text_has_no_specialized_shape():
    inherited = _derive(
        _start_runtime(),
        task_text="Check the other source too.",
        task_context=_context(
            continuation_of_prior_evidence_task=True,
            prior_task_shape="historical_reconstruction",
        ),
    ).json()["result"]
    no_continuation = _derive(
        _start_runtime(),
        task_text="What about the previous month?",
    ).json()["result"]
    explicit_new = _derive(
        _start_runtime(),
        task_text="Compare the other two records.",
        task_context=_context(
            continuation_of_prior_evidence_task=True,
            prior_task_shape="historical_reconstruction",
        ),
    ).json()["result"]
    incompatible = _derive(
        _start_runtime(),
        task_text="Reconstruct the timeline and compare the records.",
        task_context=_context(
            continuation_of_prior_evidence_task=True,
            prior_task_shape="targeted_lookup",
        ),
    ).json()["result"]

    assert inherited["task_shape"] == "historical_reconstruction"
    assert "prior_shape_inherited" in inherited["reason_codes"]
    _assert_not_applicable(no_continuation)
    assert explicit_new["task_shape"] == "cross_source_comparison"
    assert "prior_shape_inherited" not in explicit_new["reason_codes"]
    assert incompatible["derivation_status"] == "ambiguous"


@pytest.mark.parametrize(
    ("interaction_kind", "task_text", "task_context", "expected_status", "shape"),
    [
        ("question", "Verify the record.", _context(), "derived", "targeted_lookup"),
        ("command", "Check the log.", _context(), "derived", "targeted_lookup"),
        (
            "high_impact_decision",
            "Decide between these options based on the evidence.",
            _context(high_stakes_accuracy_required=True),
            "derived",
            "recommendation_or_decision_support",
        ),
        (
            "tense_debugging",
            "Check the failure logs.",
            _context(),
            "derived",
            "targeted_lookup",
        ),
        (
            "mistake_or_failure_report",
            "Inspect the incident report.",
            _context(),
            "derived",
            "targeted_lookup",
        ),
        (
            "brainstorm",
            "Compare candidate reports for the brainstorm.",
            _context(),
            "derived",
            "cross_source_comparison",
        ),
        ("joke_or_playful", "Tell a joke.", _context(), "not_applicable", None),
        ("vent_or_expression", "I am upset.", _context(), "not_applicable", None),
        (
            "ambiguous",
            "This may matter.",
            _context(evidence_input_kinds=["memory"]),
            "ambiguous",
            None,
        ),
    ],
)
def test_existing_interaction_kinds_are_reused_conservatively(
    interaction_kind: str,
    task_text: str,
    task_context: dict[str, object],
    expected_status: str,
    shape: str | None,
):
    result = _derive(
        _start_runtime(),
        interaction_kind=interaction_kind,
        task_text=task_text,
        task_context=task_context,
    ).json()["result"]

    assert result["derivation_status"] == expected_status
    assert result["task_shape"] == shape


def test_shape_derivation_does_not_modify_runtime_turn_intent_class():
    runtime = _start_runtime()
    before = client.get(f"/v1/runtime/sessions/{runtime['runtime_session_id']}")
    response = _derive(
        runtime,
        task_text="Verify the record.",
        interaction_kind="command",
    )
    after = client.get(f"/v1/runtime/sessions/{runtime['runtime_session_id']}")

    assert response.status_code == 200
    assert before.json()["latest_turn"]["intent_class"] is None
    assert after.json()["latest_turn"]["intent_class"] is None


def _source(
    source_id: str,
    capabilities: list[str],
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_categories": ["records"],
        "capabilities": capabilities,
        "availability": "available",
        "authority_role": "authoritative",
    }


@pytest.mark.parametrize(
    (
        "task_text",
        "task_context",
        "inventory",
        "time_scope_ref",
        "expected_shape",
        "expected_status",
        "expected_strategy",
        "expected_completeness",
        "expected_contradiction_required",
        "expected_requirement_kinds",
        "expected_capability_limitation",
    ),
    [
        (
            "Verify the record for the current value.",
            _context(),
            [_source("source-a", ["targeted_retrieval"])],
            None,
            "targeted_lookup",
            "ready",
            "targeted_retrieval",
            "targeted_scope",
            False,
            {"targeted_evidence", "context_delivery"},
            False,
        ),
        (
            "Check every requirement in the complete checklist.",
            _context(),
            [_source("source-a", ["structured_query"])],
            None,
            "bounded_exhaustive_review",
            "unsupported",
            "structured_query",
            "complete_for_declared_scope",
            True,
            {
                "authoritative_inventory",
                "complete_scope_coverage",
                "contradiction_search",
                "context_delivery",
                "no_material_truncation",
            },
            True,
        ),
        (
            "Check whether there is no record in the declared logs.",
            _context(),
            [_source("source-a", ["structured_query"])],
            None,
            "absence_or_coverage_check",
            "unsupported",
            "structured_query",
            "complete_for_declared_scope",
            False,
            {
                "authoritative_inventory",
                "complete_scope_coverage",
                "structured_absence_check",
                "context_delivery",
                "no_material_truncation",
            },
            True,
        ),
        (
            "Compare the reports and decide between these options.",
            _context(high_stakes_accuracy_required=True),
            [
                _source("source-a", ["targeted_retrieval", "exact_fetch"]),
                _source("source-b", ["targeted_retrieval", "exact_fetch"]),
            ],
            None,
            "recommendation_or_decision_support",
            "unsupported",
            "hybrid",
            "bounded_decision_support",
            True,
            {
                "candidate_evidence_coverage",
                "cross_source_comparison",
                "contradiction_search",
                "counterevidence_coverage",
                "context_delivery",
                "no_material_truncation",
            },
            True,
        ),
    ],
)
def test_derived_shape_and_anchor_submit_unchanged_to_plan_compiler(
    task_text: str,
    task_context: dict[str, object],
    inventory: list[dict[str, object]],
    time_scope_ref: str | None,
    expected_shape: str,
    expected_status: str,
    expected_strategy: str,
    expected_completeness: str,
    expected_contradiction_required: bool,
    expected_requirement_kinds: set[str],
    expected_capability_limitation: bool,
):
    runtime = _start_runtime()
    derived = _derive(
        runtime,
        task_text=task_text,
        task_context=task_context,
    ).json()["result"]
    source_ids = [source["source_id"] for source in inventory]
    compiled = client.post(
        "/v1/runtime/evidence-plans/compile",
        json={
            **runtime,
            "request_id": f"{runtime['request_id']}-compile",
            "question_anchor": derived["question_anchor"],
            "task_shape": derived["task_shape"],
            "declared_scope": {
                "source_ids": source_ids,
                "source_categories": [],
                "inventory_status": "complete_for_declared_scope",
                "time_scope_ref": time_scope_ref,
            },
            "source_inventory": inventory,
        },
    )

    assert derived["task_shape"] == expected_shape
    assert compiled.status_code == 200
    result = compiled.json()["result"]
    assert result["task_shape"] == expected_shape
    assert result["question_anchor"] == derived["question_anchor"]
    assert result["plan_status"] == expected_status
    assert result["selected_strategies"] == [expected_strategy]
    assert result["completeness_expectation"] == expected_completeness
    assert (
        result["contradiction_search_required"]
        is expected_contradiction_required
    )
    assert {
        requirement["requirement_kind"]
        for requirement in result["declared_requirements"]
        if requirement["criticality"] == "material"
    } == expected_requirement_kinds
    assert (
        "required_capability_unavailable" in result["limitation_codes"]
    ) is expected_capability_limitation


def test_non_derived_results_cannot_be_submitted_as_valid_plans():
    runtime = _start_runtime()
    not_applicable = _derive(
        runtime,
        task_text="Tell me a joke.",
        interaction_kind="joke_or_playful",
    ).json()["result"]
    ambiguous = _derive(
        runtime,
        task_text="This might matter.",
        interaction_kind="ambiguous",
        task_context=_context(evidence_input_kinds=["artifact"]),
    ).json()["result"]

    for result in (not_applicable, ambiguous):
        response = client.post(
            "/v1/runtime/evidence-plans/compile",
            json={
                **runtime,
                "request_id": f"{runtime['request_id']}-invalid-plan",
                "question_anchor": result["question_anchor"],
                "task_shape": result["task_shape"],
                "declared_scope": {
                    "source_ids": [],
                    "source_categories": [],
                    "inventory_status": "unknown",
                },
                "source_inventory": [],
            },
        )
        assert response.status_code == 422


def test_equivalent_context_input_orders_produce_identical_complete_results():
    runtime = _start_runtime()
    first = _derive(
        runtime,
        task_text="Check the records.",
        task_context=_context(
            evidence_input_kinds=["tool_output", "artifact", "memory"]
        ),
    )
    second = _derive(
        runtime,
        task_text="Check the records.",
        task_context=_context(
            evidence_input_kinds=["memory", "artifact", "tool_output"]
        ),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["result"] == second.json()["result"]


def test_unknown_runtime_session_returns_bounded_error_and_no_event():
    runtime = _start_runtime()
    original_session_id = runtime["runtime_session_id"]
    runtime["runtime_session_id"] = "rtsession-missing"
    response = _derive(runtime)

    assert response.status_code == 404
    assert response.json() == {"detail": "runtime_session_not_found"}
    assert _shape_events(original_session_id) == []


def test_unknown_runtime_turn_returns_bounded_error_and_no_event():
    runtime = _start_runtime()
    runtime["runtime_turn_id"] = "rtturn-missing"
    response = _derive(runtime)

    assert response.status_code == 404
    assert response.json() == {"detail": "runtime_turn_not_found"}
    assert _shape_events(runtime["runtime_session_id"]) == []


def test_runtime_session_scope_mismatch_returns_bounded_error_and_no_event():
    runtime = _start_runtime()
    runtime["owner_id"] = "another-owner"
    response = _derive(runtime)

    assert response.status_code == 400
    assert response.json() == {"detail": "runtime_session_mismatch"}
    assert _shape_events(runtime["runtime_session_id"]) == []


def test_runtime_turn_session_mismatch_returns_bounded_error_and_no_event():
    first = _start_runtime()
    second = _start_runtime()
    second["runtime_turn_id"] = first["runtime_turn_id"]
    response = _derive(second)

    assert response.status_code == 400
    assert response.json() == {"detail": "runtime_turn_session_mismatch"}
    assert _shape_events(second["runtime_session_id"]) == []


def test_runtime_event_is_exact_private_and_consistent_with_the_result():
    runtime = _start_runtime()
    response = _derive(
        runtime,
        task_text="Compare PRIVATE SOURCE ALPHA with PRIVATE SOURCE BETA.",
        interaction_kind="question",
        task_context=_context(
            evidence_input_kinds=["memory", "tool_output"],
            continuation_of_prior_evidence_task=True,
            prior_task_shape="historical_reconstruction",
        ),
    )
    assert response.status_code == 200
    result = response.json()["result"]
    event = _shape_events(runtime["runtime_session_id"])[0]
    payload = event["event_payload_json"]

    assert set(payload) == {
        "request_id",
        "runtime_session_id",
        "runtime_turn_id",
        "derivation_id",
        "question_anchor_digest",
        "interaction_kind",
        "derivation_status",
        "task_shape",
        "candidate_task_shapes",
        "evidence_scope_material",
        "clarification_required",
        "evidence_input_count",
        "continuation_of_prior_evidence_task",
        "reason_codes",
    }
    for field in (
        "derivation_id",
        "question_anchor_digest",
        "derivation_status",
        "task_shape",
        "candidate_task_shapes",
        "evidence_scope_material",
        "clarification_required",
        "reason_codes",
    ):
        assert payload[field] == result[field]
    assert payload["interaction_kind"] == "question"
    assert payload["evidence_input_count"] == 2
    assert payload["continuation_of_prior_evidence_task"] is True

    serialized = json.dumps(payload, sort_keys=True).lower()
    for excluded in (
        "private source alpha",
        "private source beta",
        "question_anchor\"",
        "task_text",
        "memory",
        "tool_output",
        "historical_reconstruction",
        "source_id",
        "source_content",
        "recent_messages",
        "prompt",
        "private memory",
        "reasoning",
        "metadata",
        "confidence",
        "exception",
    ):
        assert excluded not in serialized


def test_legacy_response_and_event_omit_source_match_entirely():
    runtime = _start_runtime()
    response = _derive(runtime, task_text="How are you today?")

    assert response.status_code == 200
    serialized = response.text
    assert "source_match" not in response.json()["result"]
    assert '"source_match":null' not in serialized
    assert "source_match_status" not in _shape_events(
        runtime["runtime_session_id"]
    )[0]["event_payload_json"]


def test_natural_display_name_match_makes_ordinary_question_targeted():
    discovery = _discovery(
        _discovery_source(
            "trail_vehicle_primary",
            "Trail Vehicle Primary",
            domain_tags=["vehicle", "fuel_economy"],
        ),
        _discovery_source(
            "household_calendar",
            "Household Calendar",
            connector="ics_calendar",
            domain_tags=["household"],
        ),
        _discovery_source(
            "work_schedule",
            "Work Schedule",
            connector="ics_calendar",
            domain_tags=["work"],
        ),
    )
    result = _derive(
        _start_runtime(),
        task_text="What was the median fuel economy for Trail Vehicle Primary?",
        task_context=_context(source_discovery=discovery),
    ).json()["result"]

    assert result["source_match"]["status"] == "matched"
    assert result["source_match"]["matched_source_ids"] == [
        "trail_vehicle_primary"
    ]
    assert result["task_shape"] == "targeted_lookup"
    assert result["evidence_scope_material"] is True
    assert "source_context_present" in result["reason_codes"]


@pytest.mark.parametrize(
    ("task_text", "source", "expected_reason"),
    [
        (
            "Use north-hub for this question.",
            _discovery_source("north_hub", "Northern Archive"),
            "source_id_match",
        ),
        (
            "What is the total for Alpine Metrics?",
            _discovery_source("metric_primary", "Alpine Metrics"),
            "display_name_match",
        ),
        (
            "What is the current nutrition total?",
            _discovery_source(
                "meal_summary", "Meal Summary", domain_tags=["nutrition"]
            ),
            "domain_tag_match",
        ),
        (
            "What changed in release 2026?",
            _discovery_source(
                "release_notes",
                "Release Notes",
                scope_refs={"version": "release_2026"},
            ),
            "scope_reference_match",
        ),
    ],
)
def test_each_source_specific_metadata_field_can_establish_identity(
    task_text: str,
    source: dict[str, object],
    expected_reason: str,
):
    result = _derive(
        _start_runtime(),
        task_text=task_text,
        task_context=_context(source_discovery=_discovery(source)),
    ).json()["result"]

    assert result["source_match"]["status"] == "matched"
    assert result["source_match"]["matched_source_ids"] == [source["source_id"]]
    assert expected_reason in result["source_match"]["reason_codes"]


def test_multiple_distinct_sources_match_in_sorted_order_and_preserve_shape():
    sources = (
        _discovery_source("harbor_metrics", "Harbor Metrics"),
        _discovery_source("alpine_metrics", "Alpine Metrics"),
        _discovery_source("forest_notes", "Forest Notes"),
    )
    result = _derive(
        _start_runtime(),
        task_text="Compare Harbor Metrics with Alpine Metrics.",
        task_context=_context(source_discovery=_discovery(*sources)),
    ).json()["result"]

    assert result["task_shape"] == "cross_source_comparison"
    assert result["source_match"]["status"] == "matched"
    assert result["source_match"]["matched_source_ids"] == [
        "alpine_metrics",
        "harbor_metrics",
    ]
    assert "multiple_explicit_source_matches" in result["source_match"][
        "reason_codes"
    ]


def test_source_kind_disambiguates_only_already_strong_candidates():
    outcomes = []
    sources = _source_kind_topology()
    for ordered_sources in (sources, tuple(reversed(sources))):
        outcomes.append(
            _derive(
                _start_runtime(),
                task_text=(
                    "What is the most recent entry in my vehicle log for Trail Unit?"
                ),
                task_context=_context(
                    source_discovery=_discovery(*ordered_sources)
                ),
            ).json()["result"]
        )

    for result in outcomes:
        assert result["source_match"]["status"] == "matched"
        assert result["source_match"]["matched_source_ids"] == [
            "maintenance_log_primary"
        ]
        assert "scheduled_actions_trail_unit" not in result["source_match"][
            "matched_source_ids"
        ]
        assert "maintenance_log_electric" not in result["source_match"][
            "matched_source_ids"
        ]
        assert result["task_shape"] == "targeted_lookup"
        assert result["evidence_scope_material"] is True
    assert outcomes[0]["source_match"] == outcomes[1]["source_match"]


def test_source_kind_cannot_override_exact_display_name_identity():
    sources = (
        _discovery_source(
            "harbor_metrics",
            "Harbor Metrics",
            domain_tags=["planning"],
            scope_refs={"project": "harbor-project"},
        ),
        _discovery_source(
            "harbor_calendar",
            "Harbor Calendar",
            domain_tags=["planning"],
            scope_refs={"project": "harbor-project"},
        ),
    )
    outcomes = [
        _derive(
            _start_runtime(),
            task_text="What is in Harbor Metrics calendar for Harbor Project?",
            task_context=_context(source_discovery=_discovery(*ordered_sources)),
        ).json()["result"]
        for ordered_sources in (sources, tuple(reversed(sources)))
    ]

    assert all(item["source_match"]["status"] == "ambiguous" for item in outcomes)
    assert all(item["source_match"]["matched_source_ids"] == [] for item in outcomes)
    assert outcomes[0]["source_match"] == outcomes[1]["source_match"]


def test_source_kind_cannot_override_exact_source_id_identity():
    result = _derive(
        _start_runtime(),
        task_text=(
            "What is in harbor_metrics_archive record for Harbor Project?"
        ),
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source(
                    "harbor_metrics_archive",
                    "Harbor Metrics Archive",
                    domain_tags=["planning"],
                    scope_refs={"project": "harbor-project"},
                ),
                _discovery_source(
                    "harbor_record",
                    "Harbor Record",
                    domain_tags=["planning"],
                    scope_refs={"project": "harbor-project"},
                ),
            )
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "ambiguous"
    assert result["source_match"]["matched_source_ids"] == []


def test_source_kind_cannot_override_inventory_unique_identity():
    result = _derive(
        _start_runtime(),
        task_text="What metrics are in the calendar for Harbor Project?",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source(
                    "harbor_metrics",
                    "Harbor Metrics",
                    domain_tags=["planning"],
                    scope_refs={"project": "harbor-project"},
                ),
                _discovery_source(
                    "harbor_calendar",
                    "Harbor Calendar",
                    domain_tags=["planning"],
                    scope_refs={"project": "harbor-project"},
                ),
            )
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "ambiguous"
    assert result["source_match"]["matched_source_ids"] == []


def test_source_kind_does_not_narrow_intentional_multiple_source_match():
    result = _derive(
        _start_runtime(),
        task_text="Compare Harbor Activity Log with Alpine Metrics.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("harbor_activity_log", "Harbor Activity Log"),
                _discovery_source("alpine_metrics", "Alpine Metrics"),
                _discovery_source("forest_notes", "Forest Notes"),
            )
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "matched"
    assert result["source_match"]["matched_source_ids"] == [
        "alpine_metrics",
        "harbor_activity_log",
    ]
    assert "multiple_explicit_source_matches" in result["source_match"][
        "reason_codes"
    ]


def test_calendar_kind_disambiguates_already_strong_project_candidates():
    result = _derive(
        _start_runtime(),
        task_text="What is scheduled for Harbor Project on the calendar?",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source(
                    "harbor_calendar",
                    "Harbor Calendar",
                    connector="ics_calendar",
                    domain_tags=["planning"],
                    scope_refs={"project": "harbor-project"},
                ),
                _discovery_source(
                    "harbor_actions",
                    "Harbor Actions",
                    domain_tags=["planning"],
                    scope_refs={"project": "harbor-project"},
                ),
                _discovery_source(
                    "electric_calendar",
                    "Electric Calendar",
                    connector="ics_calendar",
                    domain_tags=["planning"],
                    scope_refs={"project": "electric-platform"},
                ),
            )
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "matched"
    assert result["source_match"]["matched_source_ids"] == ["harbor_calendar"]


def test_same_kind_strong_candidates_remain_ambiguous_without_order_fallback():
    first = _discovery_source(
        "harbor_activity_log",
        "Harbor Activity Log",
        scope_refs={"project": "harbor-project"},
    )
    second = _discovery_source(
        "harbor_maintenance_log",
        "Harbor Maintenance Log",
        scope_refs={"project": "harbor-project"},
    )
    outcomes = [
        _derive(
            _start_runtime(),
            task_text="What is the latest log entry for Harbor Project?",
            task_context=_context(source_discovery=_discovery(*sources)),
        ).json()["result"]
        for sources in ((first, second), (second, first))
    ]

    assert all(item["source_match"]["status"] == "ambiguous" for item in outcomes)
    assert all(item["source_match"]["matched_source_ids"] == [] for item in outcomes)
    assert outcomes[0]["source_match"] == outcomes[1]["source_match"]


def test_transport_terms_do_not_break_strong_source_ambiguity():
    result = _derive(
        _start_runtime(),
        task_text="What is the latest Google Sheet entry for Harbor Project?",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source(
                    "harbor_primary",
                    "Harbor Primary",
                    connector="google_sheets",
                    scope_refs={"project": "harbor-project"},
                ),
                _discovery_source(
                    "harbor_secondary",
                    "Harbor Secondary",
                    connector="ics_calendar",
                    scope_refs={"project": "harbor-project"},
                ),
            )
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "ambiguous"
    assert result["source_match"]["matched_source_ids"] == []


def test_source_kind_cannot_promote_weak_candidates():
    result = _derive(
        _start_runtime(),
        task_text="What changed in the vehicle log?",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source(
                    "east_vehicle_log",
                    "East Vehicle Log",
                    domain_tags=["vehicle"],
                ),
                _discovery_source(
                    "west_vehicle_log",
                    "West Vehicle Log",
                    domain_tags=["vehicle"],
                ),
            )
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "ambiguous"
    assert result["source_match"]["matched_source_ids"] == []


def test_shared_weak_metadata_is_ambiguous_without_order_fallback():
    first = _discovery_source(
        "east_operations", "East Archive", domain_tags=["operations"]
    )
    second = _discovery_source(
        "west_operations", "West Archive", domain_tags=["operations"]
    )
    outcomes = []
    for sources in ((first, second), (second, first)):
        outcomes.append(
            _derive(
                _start_runtime(),
                task_text="What is the operations total?",
                task_context=_context(source_discovery=_discovery(*sources)),
            ).json()["result"]
        )

    assert all(item["source_match"]["status"] == "ambiguous" for item in outcomes)
    assert all(item["source_match"]["matched_source_ids"] == [] for item in outcomes)
    assert outcomes[0]["source_match"] == outcomes[1]["source_match"]
    assert all(item["task_shape"] == "targeted_lookup" for item in outcomes)


@pytest.mark.parametrize(
    "task_text",
    [
        "Look in the Google Sheet.",
        "Look in the calendar source.",
        "Look in the log.",
    ],
)
def test_generic_source_type_alone_never_selects_a_source(task_text: str):
    discovery = _discovery(
        _discovery_source("alpha_record_log", "Alpha Record Log"),
        _discovery_source("beta_record_log", "Beta Record Log"),
        _discovery_source(
            "team_schedule", "Team Schedule", connector="ics_calendar"
        ),
    )
    result = _derive(
        _start_runtime(),
        task_text=task_text,
        task_context=_context(source_discovery=discovery),
    ).json()["result"]

    assert result["source_match"]["status"] == "no_match"
    assert result["source_match"]["matched_source_ids"] == []
    assert "generic_source_signal_rejected" in result["source_match"][
        "reason_codes"
    ]
    _assert_not_applicable(result)


def test_complete_inventory_no_match_does_not_make_chat_evidence_material():
    result = _derive(
        _start_runtime(),
        task_text="How are you today?",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("alpine_metrics", "Alpine Metrics")
            )
        ),
    ).json()["result"]

    assert result["source_match"] == {
        "status": "no_match",
        "matched_source_ids": [],
        "reason_codes": ["no_source_specific_match"],
    }
    _assert_not_applicable(result)


def test_partial_inventory_positive_matches_and_negative_is_unavailable():
    source = _discovery_source("alpine_metrics", "Alpine Metrics")
    positive = _derive(
        _start_runtime(),
        task_text="What is the Alpine Metrics total?",
        task_context=_context(
            source_discovery=_discovery(source, inventory_status="partial")
        ),
    ).json()["result"]
    negative = _derive(
        _start_runtime(),
        task_text="How are you today?",
        task_context=_context(
            source_discovery=_discovery(source, inventory_status="partial")
        ),
    ).json()["result"]

    assert positive["source_match"]["status"] == "matched"
    assert positive["source_match"]["matched_source_ids"] == ["alpine_metrics"]
    assert "inventory_partial" in positive["source_match"]["reason_codes"]
    assert negative["source_match"]["status"] == "inventory_unavailable"
    assert negative["source_match"]["matched_source_ids"] == []
    assert "inventory_partial" in negative["source_match"]["reason_codes"]
    _assert_not_applicable(negative)


@pytest.mark.parametrize(
    ("inventory_status", "reason"),
    [("unknown", "inventory_unknown"), ("unavailable", "inventory_unavailable")],
)
def test_incomplete_inventory_cannot_establish_identity(
    inventory_status: str,
    reason: str,
):
    result = _derive(
        _start_runtime(),
        task_text="What is the Alpine Metrics total?",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("alpine_metrics", "Alpine Metrics"),
                inventory_status=inventory_status,
            )
        ),
    ).json()["result"]

    assert result["source_match"] == {
        "status": "inventory_unavailable",
        "matched_source_ids": [],
        "reason_codes": [reason],
    }


@pytest.mark.parametrize("availability", ["unavailable", "disabled", "unknown"])
def test_source_availability_does_not_redirect_identity(availability: str):
    result = _derive(
        _start_runtime(),
        task_text="What is the Alpine Metrics total?",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source(
                    "alpine_metrics", "Alpine Metrics", availability=availability
                ),
                _discovery_source("harbor_metrics", "Harbor Metrics"),
            )
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "matched"
    assert result["source_match"]["matched_source_ids"] == ["alpine_metrics"]


def _invalid_discovery_case(case: str) -> dict[str, object]:
    source = _discovery_source("bounded_source", "Bounded Source")
    discovery = _discovery(source)
    if case == "too_many_sources":
        discovery["sources"] = [
            _discovery_source(f"source_{index}", f"Source {index}")
            for index in range(33)
        ]
    elif case == "duplicate_source_ids":
        discovery["sources"] = [source, dict(source)]
    elif case == "overlong_source_id":
        source["source_id"] = "s" * 121
    elif case == "overlong_display_name":
        source["display_name"] = "d" * 241
    elif case == "invalid_domain_tag":
        source["domain_tags"] = ["invalid tag"]
    elif case == "too_many_domain_tags":
        source["domain_tags"] = [f"tag_{index}" for index in range(9)]
    elif case == "duplicate_domain_tags":
        source["domain_tags"] = ["shared", "shared"]
    elif case == "unsupported_capability":
        source["capabilities"] = ["delete"]
    elif case == "duplicate_capability":
        source["capabilities"] = ["search", "search"]
    elif case == "extra_discovery_field":
        discovery["metadata"] = "rejected"
    elif case == "extra_source_field":
        source["description"] = "rejected"
    elif case == "extra_scope_field":
        source["scope_refs"] = {"tenant": "rejected"}
    elif case == "invalid_scope_ref":
        source["scope_refs"] = {"project": "invalid ref"}
    elif case == "empty_scope_refs":
        source["scope_refs"] = {}
    else:
        raise AssertionError(case)
    return discovery


@pytest.mark.parametrize(
    "case",
    [
        "too_many_sources",
        "duplicate_source_ids",
        "overlong_source_id",
        "overlong_display_name",
        "invalid_domain_tag",
        "too_many_domain_tags",
        "duplicate_domain_tags",
        "unsupported_capability",
        "duplicate_capability",
        "extra_discovery_field",
        "extra_source_field",
        "extra_scope_field",
        "invalid_scope_ref",
        "empty_scope_refs",
    ],
)
def test_source_discovery_contract_rejects_unbounded_or_malformed_input(case: str):
    response = _derive(
        _start_runtime(),
        task_context=_context(source_discovery=_invalid_discovery_case(case)),
    )

    assert response.status_code == 422


def test_discovery_inventory_order_is_canonical_for_result_and_identity():
    runtime = _start_runtime()
    first_source = _discovery_source(
        "alpine_metrics",
        "Alpine Metrics",
        domain_tags=["fuel_economy", "vehicle"],
        capabilities=["fetch", "search"],
        scope_refs={"project": "ridge_project", "version": "release_2026"},
    )
    second_source = _discovery_source(
        "harbor_metrics",
        "Harbor Metrics",
        domain_tags=["maritime", "finance"],
        capabilities=["context", "profile"],
    )
    reversed_first = {
        **first_source,
        "domain_tags": list(reversed(first_source["domain_tags"])),
        "capabilities": list(reversed(first_source["capabilities"])),
    }
    reversed_second = {
        **second_source,
        "domain_tags": list(reversed(second_source["domain_tags"])),
        "capabilities": list(reversed(second_source["capabilities"])),
    }
    first = _derive(
        runtime,
        task_text="What is the Alpine Metrics total?",
        task_context=_context(
            source_discovery=_discovery(first_source, second_source)
        ),
    ).json()["result"]
    second = _derive(
        runtime,
        task_text="What is the Alpine Metrics total?",
        task_context=_context(
            source_discovery=_discovery(reversed_second, reversed_first)
        ),
    ).json()["result"]

    assert first["source_match"] == second["source_match"]
    assert first["derivation_id"] == second["derivation_id"]


def test_source_discovery_event_contains_only_bounded_structural_facts():
    runtime = _start_runtime()
    result = _derive(
        runtime,
        task_text="PRIVATE_TASK_SENTINEL Alpine Metrics",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source(
                    "alpine_metrics",
                    "PRIVATE_DISPLAY_SENTINEL Alpine Metrics",
                    domain_tags=["PRIVATE_TAG_SENTINEL"],
                    scope_refs={"project": "PRIVATE_SCOPE_SENTINEL"},
                )
            )
        ),
    ).json()["result"]
    payload = _shape_events(runtime["runtime_session_id"])[0]["event_payload_json"]

    assert payload["source_match_status"] == "matched"
    assert payload["matched_source_ids"] == result["source_match"][
        "matched_source_ids"
    ]
    assert payload["source_match_reason_codes"] == result["source_match"][
        "reason_codes"
    ]
    assert payload["configured_inventory_status"] == "complete"
    assert payload["configured_source_count"] == 1
    serialized = json.dumps(payload, sort_keys=True).lower()
    for sentinel in (
        "private_task_sentinel",
        "private_display_sentinel",
        "private_tag_sentinel",
        "private_scope_sentinel",
        "google_sheets",
    ):
        assert sentinel not in serialized


@pytest.mark.parametrize(
    "updates",
    [
        {"probe_source_ids": ["source_a"]},
        {
            "probe_source_ids": [
                "source_a",
                "source_b",
                "source_c",
                "source_d",
            ]
        },
        {"probe_source_ids": ["source_a", "source_a"]},
        {"probe_source_ids": ["source_b", "source_a"]},
        {
            "status": "matched",
            "matched_source_ids": ["source_a"],
            "probe_source_ids": ["source_a", "source_b"],
        },
        {"status": "no_match", "probe_source_ids": ["source_a", "source_b"]},
        {
            "status": "inventory_unavailable",
            "probe_source_ids": ["source_a", "source_b"],
        },
        {
            "probe_source_ids": ["source_a", "source_b"],
            "reason_codes": ["multiple_possible_source_matches"],
        },
        {
            "matched_source_ids": ["source_c"],
            "probe_source_ids": ["source_a", "source_b"],
        },
    ],
)
def test_source_match_result_rejects_invalid_probe_authorization(
    updates: dict[str, object],
):
    source_match: dict[str, object] = {
        "status": "ambiguous",
        "matched_source_ids": [],
        "probe_source_ids": ["source_a", "source_b"],
        "reason_codes": ["semantic_candidates_ambiguous"],
    }
    source_match.update(updates)

    with pytest.raises(ValidationError):
        SourceMatchResult.model_validate(source_match)


@pytest.mark.parametrize(
    "source_match",
    [
        {
            "status": "matched",
            "matched_source_ids": ["source_a"],
            "reason_codes": ["source_id_match"],
        },
        {
            "status": "ambiguous",
            "matched_source_ids": [],
            "reason_codes": ["multiple_possible_source_matches"],
        },
        {
            "status": "matched",
            "matched_source_ids": ["source_a"],
            "reason_codes": ["semantic_candidate_validated"],
        },
        {
            "status": "no_match",
            "matched_source_ids": [],
            "reason_codes": ["semantic_no_match"],
        },
        {
            "status": "ambiguous",
            "matched_source_ids": [],
            "probe_source_ids": [],
            "reason_codes": ["semantic_candidates_ambiguous"],
        },
        {
            "status": "ambiguous",
            "matched_source_ids": [],
            "reason_codes": [
                "inventory_partial",
                "semantic_candidates_ambiguous",
            ],
        },
    ],
)
def test_source_match_result_omits_empty_probe_authorization_from_wire(
    source_match: dict[str, object],
):
    result = SourceMatchResult.model_validate(source_match)

    assert result.probe_source_ids == []
    assert "probe_source_ids" not in result.model_dump(mode="json")


def test_source_match_result_emits_non_empty_probe_authorization():
    result = SourceMatchResult.model_validate(
        {
            "status": "ambiguous",
            "matched_source_ids": [],
            "probe_source_ids": ["source_a", "source_b"],
            "reason_codes": ["semantic_candidates_ambiguous"],
        }
    )

    assert result.model_dump(mode="json")["probe_source_ids"] == [
        "source_a",
        "source_b",
    ]


def test_semantic_advisory_requires_source_discovery():
    response = _derive(
        _start_runtime(),
        task_context=_context(
            semantic_advisory=_semantic_advisory(
                "resolved", "lookup", "source_a"
            )
        ),
    )

    assert response.status_code == 422
    assert "semantic_advisory_source_discovery_required" in response.text


@pytest.mark.parametrize(
    ("status", "candidates"),
    [
        ("resolved", []),
        ("resolved", ["source_a", "source_b"]),
        ("ambiguous", ["source_a"]),
        ("ambiguous", ["source_a", "source_b", "source_c", "source_d"]),
        ("no_match", ["source_a"]),
        ("ambiguous", ["source_a", "source_a"]),
    ],
)
def test_semantic_advisory_rejects_incoherent_candidate_sets(
    status: str,
    candidates: list[str],
):
    sources = tuple(
        _discovery_source(source_id, source_id.replace("_", " ").title())
        for source_id in {"source_a", "source_b", "source_c", "source_d"}
    )
    response = _derive(
        _start_runtime(),
        task_context=_context(
            source_discovery=_discovery(*sources),
            semantic_advisory=_semantic_advisory(
                status, "lookup", *candidates
            ),
        ),
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "candidate_source_id",
    ["fabricated_source", "invalid source", "s" * 121],
)
def test_semantic_advisory_rejects_unknown_or_invalid_candidate_ids(
    candidate_source_id: str,
):
    response = _derive(
        _start_runtime(),
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Source A")
            ),
            semantic_advisory=_semantic_advisory(
                "resolved", "lookup", candidate_source_id
            ),
        ),
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "advisory_update",
    [
        {"interpretation_status": "unsupported"},
        {"operation_hint": "summarize_anything"},
        {"explanation": "PRIVATE_REASONING_SENTINEL"},
        {"metadata": {"confidence": 0.99}},
    ],
)
def test_semantic_advisory_rejects_unbounded_or_unsupported_fields(
    advisory_update: dict[str, object],
):
    advisory = _semantic_advisory("resolved", "lookup", "source_a")
    advisory.update(advisory_update)
    response = _derive(
        _start_runtime(),
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Source A")
            ),
            semantic_advisory=advisory,
        ),
    )

    assert response.status_code == 422


def test_semantic_candidate_can_resolve_deterministic_no_match():
    result = _derive(
        _start_runtime(),
        task_text="Please answer this bounded question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register"),
                _discovery_source("source_b", "Harbor Register"),
            ),
            semantic_advisory=_semantic_advisory(
                "resolved", "lookup", "source_a"
            ),
        ),
    ).json()["result"]

    assert result["source_match"] == {
        "status": "matched",
        "matched_source_ids": ["source_a"],
        "reason_codes": ["semantic_candidate_validated"],
    }
    assert result["task_shape"] == "targeted_lookup"
    assert "semantic_operation_hint" in result["reason_codes"]


def test_semantic_candidate_can_resolve_non_distinct_primary_ambiguity():
    discovery = _discovery(
        _discovery_source(
            "east_register",
            "East Register",
            domain_tags=["operations"],
            scope_refs={"project": "shared-project"},
        ),
        _discovery_source(
            "west_register",
            "West Register",
            domain_tags=["operations"],
            scope_refs={"project": "shared-project"},
        ),
    )
    result = _derive(
        _start_runtime(),
        task_text="What is the operations result for Shared Project?",
        task_context=_context(
            source_discovery=discovery,
            semantic_advisory=_semantic_advisory(
                "resolved", "lookup", "east_register"
            ),
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "matched"
    assert result["source_match"]["matched_source_ids"] == ["east_register"]
    assert result["source_match"]["reason_codes"] == [
        "semantic_candidate_validated"
    ]


def test_deterministic_match_outranks_conflicting_semantic_candidate():
    result = _derive(
        _start_runtime(),
        task_text="What is in Alpine Register?",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register"),
                _discovery_source("source_b", "Harbor Register"),
            ),
            semantic_advisory=_semantic_advisory(
                "resolved", "lookup", "source_b"
            ),
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "matched"
    assert result["source_match"]["matched_source_ids"] == ["source_a"]
    assert "semantic_candidate_validated" not in result["source_match"][
        "reason_codes"
    ]


def test_deterministic_match_outranks_semantic_probe_candidates():
    result = _derive(
        _start_runtime(),
        task_text="What is in Alpine Register?",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register"),
                _discovery_source("source_b", "Harbor Register"),
                _discovery_source("source_c", "Forest Register"),
            ),
            semantic_advisory=_semantic_advisory(
                "ambiguous", "lookup", "source_b", "source_c"
            ),
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "matched"
    assert result["source_match"]["matched_source_ids"] == ["source_a"]
    assert "probe_source_ids" not in result["source_match"]


def test_primary_distinct_ambiguity_outranks_semantic_candidate():
    discovery = _discovery(
        _discovery_source(
            "harbor_metrics",
            "Harbor Metrics",
            domain_tags=["planning"],
            scope_refs={"project": "harbor-project"},
        ),
        _discovery_source(
            "harbor_calendar",
            "Harbor Calendar",
            domain_tags=["planning"],
            scope_refs={"project": "harbor-project"},
        ),
    )
    result = _derive(
        _start_runtime(),
        task_text="What is in Harbor Metrics calendar for Harbor Project?",
        task_context=_context(
            source_discovery=discovery,
            semantic_advisory=_semantic_advisory(
                "resolved", "lookup", "harbor_calendar"
            ),
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "ambiguous"
    assert result["source_match"]["matched_source_ids"] == []
    assert "semantic_candidate_validated" not in result["source_match"][
        "reason_codes"
    ]


def test_primary_distinct_ambiguity_outranks_semantic_probe_candidates():
    discovery = _discovery(
        _discovery_source(
            "harbor_metrics",
            "Harbor Metrics",
            domain_tags=["planning"],
            scope_refs={"project": "harbor-project"},
        ),
        _discovery_source(
            "harbor_calendar",
            "Harbor Calendar",
            domain_tags=["planning"],
            scope_refs={"project": "harbor-project"},
        ),
    )
    result = _derive(
        _start_runtime(),
        task_text="What is in Harbor Metrics calendar for Harbor Project?",
        task_context=_context(
            source_discovery=discovery,
            semantic_advisory=_semantic_advisory(
                "ambiguous", "lookup", "harbor_calendar", "harbor_metrics"
            ),
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "ambiguous"
    assert result["source_match"]["matched_source_ids"] == []
    assert "semantic_candidates_ambiguous" not in result["source_match"][
        "reason_codes"
    ]
    assert "probe_source_ids" not in result["source_match"]


def test_semantic_ambiguity_authorizes_bounded_two_source_probe():
    result = _derive(
        _start_runtime(),
        task_text="Please handle this question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register"),
                _discovery_source("source_b", "Harbor Register"),
                _discovery_source("source_c", "Forest Register"),
            ),
            semantic_advisory=_semantic_advisory(
                "ambiguous", "lookup", "source_b", "source_a"
            ),
        ),
    ).json()["result"]

    assert result["source_match"] == {
        "status": "ambiguous",
        "matched_source_ids": [],
        "probe_source_ids": ["source_a", "source_b"],
        "reason_codes": ["semantic_candidates_ambiguous"],
    }
    assert result["derivation_status"] == "derived"
    assert result["task_shape"] == "targeted_lookup"
    assert result["candidate_task_shapes"] == ["targeted_lookup"]
    assert result["evidence_scope_material"] is True
    assert result["clarification_required"] is False


@pytest.mark.parametrize("operation_hint", ["lookup", "latest"])
def test_semantic_ambiguity_authorizes_bounded_three_source_probe(
    operation_hint: str,
):
    result = _derive(
        _start_runtime(),
        task_text="Please handle this question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_c", "Forest Register"),
                _discovery_source("source_a", "Alpine Register"),
                _discovery_source("source_b", "Harbor Register"),
            ),
            semantic_advisory=_semantic_advisory(
                "ambiguous", operation_hint, "source_c", "source_a", "source_b"
            ),
        ),
    ).json()["result"]

    assert result["source_match"] == {
        "status": "ambiguous",
        "matched_source_ids": [],
        "probe_source_ids": ["source_a", "source_b", "source_c"],
        "reason_codes": ["semantic_candidates_ambiguous"],
    }
    assert result["derivation_status"] == "derived"
    assert result["task_shape"] == "targeted_lookup"
    assert result["candidate_task_shapes"] == ["targeted_lookup"]
    assert result["evidence_scope_material"] is True
    assert result["clarification_required"] is False


def test_semantic_probe_candidate_and_inventory_order_are_canonical():
    runtime = _start_runtime()
    source_a = _discovery_source(
        "source_a",
        "PRIVATE_DISPLAY_ALPHA",
        domain_tags=["private_alpha", "alpha_domain"],
        capabilities=["fetch", "search"],
    )
    source_b = _discovery_source(
        "source_b",
        "PRIVATE_DISPLAY_BETA",
        domain_tags=["private_beta", "beta_domain"],
        capabilities=["search", "context"],
    )
    first = _derive(
        runtime,
        task_text="Please handle this bounded question.",
        task_context=_context(
            source_discovery=_discovery(source_a, source_b),
            semantic_advisory=_semantic_advisory(
                "ambiguous", "lookup", "source_b", "source_a"
            ),
        ),
    ).json()["result"]
    reversed_a = {
        **source_a,
        "domain_tags": list(reversed(source_a["domain_tags"])),
        "capabilities": list(reversed(source_a["capabilities"])),
    }
    reversed_b = {
        **source_b,
        "domain_tags": list(reversed(source_b["domain_tags"])),
        "capabilities": list(reversed(source_b["capabilities"])),
    }
    second = _derive(
        runtime,
        task_text="Please handle this bounded question.",
        task_context=_context(
            source_discovery=_discovery(reversed_b, reversed_a),
            semantic_advisory=_semantic_advisory(
                "ambiguous", "lookup", "source_a", "source_b"
            ),
        ),
    ).json()["result"]

    assert first["source_match"] == second["source_match"]
    assert first["source_match"]["probe_source_ids"] == ["source_a", "source_b"]
    assert first["derivation_id"] == second["derivation_id"]


def test_semantic_no_match_and_deterministic_ambiguity_compose_conservatively():
    complete_no_match = _derive(
        _start_runtime(),
        task_text="Please handle this question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register")
            ),
            semantic_advisory=_semantic_advisory("no_match", "unknown"),
        ),
    ).json()["result"]
    deterministic_ambiguity = _derive(
        _start_runtime(),
        task_text="What is the operations result?",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source(
                    "east_register", "East Register", domain_tags=["operations"]
                ),
                _discovery_source(
                    "west_register", "West Register", domain_tags=["operations"]
                ),
            ),
            semantic_advisory=_semantic_advisory("no_match", "lookup"),
        ),
    ).json()["result"]

    assert complete_no_match["source_match"] == {
        "status": "no_match",
        "matched_source_ids": [],
        "reason_codes": ["semantic_no_match"],
    }
    _assert_not_applicable(complete_no_match)
    assert deterministic_ambiguity["source_match"]["status"] == "ambiguous"
    assert deterministic_ambiguity["source_match"]["matched_source_ids"] == []
    assert "semantic_no_match" in deterministic_ambiguity["source_match"][
        "reason_codes"
    ]


def test_semantic_partial_inventory_preserves_positive_and_negative_limits():
    discovery = _discovery(
        _discovery_source("source_a", "Alpine Register"),
        inventory_status="partial",
    )
    positive = _derive(
        _start_runtime(),
        task_text="Please handle this question.",
        task_context=_context(
            source_discovery=discovery,
            semantic_advisory=_semantic_advisory(
                "resolved", "lookup", "source_a"
            ),
        ),
    ).json()["result"]
    negative = _derive(
        _start_runtime(),
        task_text="Please handle this question.",
        task_context=_context(
            source_discovery=discovery,
            semantic_advisory=_semantic_advisory("no_match", "lookup"),
        ),
    ).json()["result"]

    assert positive["source_match"] == {
        "status": "matched",
        "matched_source_ids": ["source_a"],
        "reason_codes": ["inventory_partial", "semantic_candidate_validated"],
    }
    assert negative["source_match"] == {
        "status": "inventory_unavailable",
        "matched_source_ids": [],
        "reason_codes": ["inventory_partial", "semantic_no_match"],
    }
    assert negative["evidence_scope_material"] is True


def test_semantic_ambiguity_on_partial_inventory_remains_ambiguous():
    result = _derive(
        _start_runtime(),
        task_text="Please handle this question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register"),
                _discovery_source("source_b", "Harbor Register"),
                inventory_status="partial",
            ),
            semantic_advisory=_semantic_advisory(
                "ambiguous", "lookup", "source_b", "source_a"
            ),
        ),
    ).json()["result"]

    assert result["source_match"] == {
        "status": "ambiguous",
        "matched_source_ids": [],
        "reason_codes": ["inventory_partial", "semantic_candidates_ambiguous"],
    }


@pytest.mark.parametrize("inventory_status", ["unknown", "unavailable"])
def test_semantic_ambiguity_does_not_authorize_probe_for_unusable_inventory(
    inventory_status: str,
):
    result = _derive(
        _start_runtime(),
        task_text="Please handle this question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register"),
                _discovery_source("source_b", "Harbor Register"),
                inventory_status=inventory_status,
            ),
            semantic_advisory=_semantic_advisory(
                "ambiguous", "lookup", "source_a", "source_b"
            ),
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "inventory_unavailable"
    assert "probe_source_ids" not in result["source_match"]


@pytest.mark.parametrize("availability", ["unavailable", "disabled", "unknown"])
def test_semantic_ambiguity_requires_every_probe_candidate_available(
    availability: str,
):
    result = _derive(
        _start_runtime(),
        task_text="Please handle this question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source(
                    "source_a", "Alpine Register", availability=availability
                ),
                _discovery_source("source_b", "Harbor Register"),
            ),
            semantic_advisory=_semantic_advisory(
                "ambiguous", "lookup", "source_a", "source_b"
            ),
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "ambiguous"
    assert result["source_match"]["matched_source_ids"] == []
    assert "probe_source_ids" not in result["source_match"]


def test_semantic_ambiguity_requires_every_probe_candidate_search_capable():
    result = _derive(
        _start_runtime(),
        task_text="Please handle this question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source(
                    "source_a",
                    "Alpine Register",
                    capabilities=["fetch", "context"],
                ),
                _discovery_source("source_b", "Harbor Register"),
            ),
            semantic_advisory=_semantic_advisory(
                "ambiguous", "lookup", "source_a", "source_b"
            ),
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "ambiguous"
    assert result["source_match"]["matched_source_ids"] == []
    assert "probe_source_ids" not in result["source_match"]


@pytest.mark.parametrize(
    "operation_hint",
    [
        "comparison",
        "exhaustive_review",
        "contradiction_review",
        "absence_check",
        "historical_reconstruction",
        "decision_support",
        "aggregate",
        "unknown",
    ],
)
def test_non_lookup_semantic_ambiguity_does_not_authorize_probe(
    operation_hint: str,
):
    result = _derive(
        _start_runtime(),
        task_text="Please handle this question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register"),
                _discovery_source("source_b", "Harbor Register"),
            ),
            semantic_advisory=_semantic_advisory(
                "ambiguous", operation_hint, "source_a", "source_b"
            ),
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "ambiguous"
    assert result["source_match"]["matched_source_ids"] == []
    assert "probe_source_ids" not in result["source_match"]


def test_deterministic_non_lookup_shape_prevents_lookup_probe_authorization():
    result = _derive(
        _start_runtime(),
        task_text="Compare the available records.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register"),
                _discovery_source("source_b", "Harbor Register"),
            ),
            semantic_advisory=_semantic_advisory(
                "ambiguous", "lookup", "source_a", "source_b"
            ),
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "ambiguous"
    assert "probe_source_ids" not in result["source_match"]
    assert result["derivation_status"] == "derived"
    assert result["task_shape"] == "cross_source_comparison"


@pytest.mark.parametrize(
    ("inventory_status", "reason"),
    [("unknown", "inventory_unknown"), ("unavailable", "inventory_unavailable")],
)
def test_semantic_candidate_cannot_override_unusable_inventory(
    inventory_status: str,
    reason: str,
):
    result = _derive(
        _start_runtime(),
        task_text="Please handle this question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register"),
                inventory_status=inventory_status,
            ),
            semantic_advisory=_semantic_advisory(
                "resolved", "lookup", "source_a"
            ),
        ),
    ).json()["result"]

    assert result["source_match"] == {
        "status": "inventory_unavailable",
        "matched_source_ids": [],
        "reason_codes": [reason],
    }


@pytest.mark.parametrize("availability", ["unavailable", "disabled", "unknown"])
def test_semantic_identity_does_not_redirect_individually_unavailable_source(
    availability: str,
):
    result = _derive(
        _start_runtime(),
        task_text="Please handle this question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source(
                    "source_a", "Alpine Register", availability=availability
                ),
                _discovery_source("source_b", "Harbor Register"),
            ),
            semantic_advisory=_semantic_advisory(
                "resolved", "lookup", "source_a"
            ),
        ),
    ).json()["result"]

    assert result["source_match"]["matched_source_ids"] == ["source_a"]


@pytest.mark.parametrize(
    ("operation_hint", "expected_shape", "expected_shape_reason"),
    [
        ("lookup", "targeted_lookup", "targeted_lookup_derived"),
        ("latest", "targeted_lookup", "targeted_lookup_derived"),
        ("comparison", "cross_source_comparison", "comparison_requested"),
        (
            "exhaustive_review",
            "bounded_exhaustive_review",
            "exhaustive_scope_requested",
        ),
        (
            "contradiction_review",
            "contradiction_review",
            "contradiction_requested",
        ),
        ("absence_check", "absence_or_coverage_check", "absence_scope_requested"),
        (
            "historical_reconstruction",
            "historical_reconstruction",
            "historical_reconstruction_requested",
        ),
        (
            "decision_support",
            "recommendation_or_decision_support",
            "decision_support_requested",
        ),
    ],
)
def test_supported_semantic_operation_supplies_governed_task_shape(
    operation_hint: str,
    expected_shape: str,
    expected_shape_reason: str,
):
    result = _derive(
        _start_runtime(),
        task_text="Please handle this bounded question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register")
            ),
            semantic_advisory=_semantic_advisory(
                "resolved", operation_hint, "source_a"
            ),
        ),
    ).json()["result"]

    assert result["derivation_status"] == "derived"
    assert result["task_shape"] == expected_shape
    assert "semantic_operation_hint" in result["reason_codes"]
    assert expected_shape_reason in result["reason_codes"]


@pytest.mark.parametrize(
    ("task_text", "semantic_operation", "expected_shape"),
    [
        (
            "Compare Harbor Metrics with Alpine Metrics.",
            "lookup",
            "cross_source_comparison",
        ),
        (
            "Reconstruct the history of Alpine Metrics.",
            "latest",
            "historical_reconstruction",
        ),
    ],
)
def test_deterministic_task_shape_outranks_semantic_operation(
    task_text: str,
    semantic_operation: str,
    expected_shape: str,
):
    result = _derive(
        _start_runtime(),
        task_text=task_text,
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("alpine_metrics", "Alpine Metrics"),
                _discovery_source("harbor_metrics", "Harbor Metrics"),
            ),
            semantic_advisory=_semantic_advisory(
                "resolved", semantic_operation, "alpine_metrics"
            ),
        ),
    ).json()["result"]

    assert result["task_shape"] == expected_shape
    assert "semantic_operation_hint" not in result["reason_codes"]


def test_aggregate_operation_fails_closed_without_targeted_lookup_downgrade():
    result = _derive(
        _start_runtime(),
        task_text="Please calculate this result.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register")
            ),
            semantic_advisory=_semantic_advisory(
                "resolved", "aggregate", "source_a"
            ),
        ),
    ).json()["result"]

    assert result["source_match"]["matched_source_ids"] == ["source_a"]
    assert result["derivation_status"] == "ambiguous"
    assert result["task_shape"] is None
    assert result["candidate_task_shapes"] == []
    assert result["evidence_scope_material"] is True
    assert result["clarification_required"] is True
    assert "semantic_operation_hint" in result["reason_codes"]
    assert "semantic_operation_unsupported" in result["reason_codes"]


def _assert_aggregate_operation_blocked(result: dict[str, object]) -> None:
    assert result["derivation_status"] == "ambiguous"
    assert result["task_shape"] is None
    assert result["candidate_task_shapes"] == []
    assert result["evidence_scope_material"] is True
    assert result["clarification_required"] is True
    assert "semantic_operation_hint" in result["reason_codes"]
    assert "semantic_operation_unsupported" in result["reason_codes"]


@pytest.mark.parametrize(
    ("task_text", "blocked_shape_reason"),
    [
        (
            "Compare the median values from the available registers.",
            "comparison_requested",
        ),
        (
            "Reconstruct the median history from the available values.",
            "historical_reconstruction_requested",
        ),
        (
            "What was not covered in the available values?",
            "absence_scope_requested",
        ),
    ],
)
def test_aggregate_blocks_deterministic_specialized_shapes(
    task_text: str,
    blocked_shape_reason: str,
):
    result = _derive(
        _start_runtime(),
        task_text=task_text,
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register"),
                _discovery_source("source_b", "Harbor Register"),
            ),
            semantic_advisory=_semantic_advisory(
                "resolved", "aggregate", "source_a"
            ),
        ),
    ).json()["result"]

    _assert_aggregate_operation_blocked(result)
    assert blocked_shape_reason not in result["reason_codes"]
    assert result["source_match"]["status"] == "matched"
    assert result["source_match"]["matched_source_ids"] == ["source_a"]


def test_aggregate_blocks_inherited_continuation_shape():
    result = _derive(
        _start_runtime(),
        task_text="Continue calculating the median.",
        task_context=_context(
            continuation_of_prior_evidence_task=True,
            prior_task_shape="targeted_lookup",
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register")
            ),
            semantic_advisory=_semantic_advisory(
                "resolved", "aggregate", "source_a"
            ),
        ),
    ).json()["result"]

    _assert_aggregate_operation_blocked(result)
    assert "prior_shape_inherited" not in result["reason_codes"]
    assert result["source_match"]["status"] == "matched"
    assert result["source_match"]["matched_source_ids"] == ["source_a"]


def test_aggregate_block_preserves_semantic_source_ambiguity():
    result = _derive(
        _start_runtime(),
        task_text="Calculate the median for the available values.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register"),
                _discovery_source("source_b", "Harbor Register"),
            ),
            semantic_advisory=_semantic_advisory(
                "ambiguous", "aggregate", "source_b", "source_a"
            ),
        ),
    ).json()["result"]

    _assert_aggregate_operation_blocked(result)
    assert result["source_match"]["status"] == "ambiguous"
    assert result["source_match"]["matched_source_ids"] == []


def test_unknown_semantic_operation_has_no_materiality_or_shape_authority():
    result = _derive(
        _start_runtime(),
        task_text="How are you today?",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register")
            ),
            semantic_advisory=_semantic_advisory("no_match", "unknown"),
        ),
    ).json()["result"]

    _assert_not_applicable(result)
    assert "semantic_operation_hint" not in result["reason_codes"]


def test_supported_operation_remains_material_when_semantics_find_no_source():
    result = _derive(
        _start_runtime(),
        task_text="Please handle this bounded question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register")
            ),
            semantic_advisory=_semantic_advisory("no_match", "comparison"),
        ),
    ).json()["result"]

    assert result["source_match"]["status"] == "no_match"
    assert result["source_match"]["matched_source_ids"] == []
    assert result["derivation_status"] == "derived"
    assert result["task_shape"] == "cross_source_comparison"
    assert result["evidence_scope_material"] is True


def test_semantic_candidate_and_inventory_order_are_canonical():
    runtime = _start_runtime()
    source_a = _discovery_source(
        "source_a",
        "PRIVATE_DISPLAY_ALPHA",
        domain_tags=["private_alpha", "alpha_domain"],
        capabilities=["fetch", "search"],
    )
    source_b = _discovery_source(
        "source_b",
        "PRIVATE_DISPLAY_BETA",
        domain_tags=["private_beta", "beta_domain"],
        capabilities=["context", "profile"],
    )
    first = _derive(
        runtime,
        task_text="Please compare this bounded question.",
        task_context=_context(
            source_discovery=_discovery(source_a, source_b),
            semantic_advisory=_semantic_advisory(
                "ambiguous", "comparison", "source_b", "source_a"
            ),
        ),
    ).json()["result"]
    reversed_a = {
        **source_a,
        "domain_tags": list(reversed(source_a["domain_tags"])),
        "capabilities": list(reversed(source_a["capabilities"])),
    }
    reversed_b = {
        **source_b,
        "domain_tags": list(reversed(source_b["domain_tags"])),
        "capabilities": list(reversed(source_b["capabilities"])),
    }
    second = _derive(
        runtime,
        task_text="Please compare this bounded question.",
        task_context=_context(
            source_discovery=_discovery(reversed_b, reversed_a),
            semantic_advisory=_semantic_advisory(
                "ambiguous", "comparison", "source_a", "source_b"
            ),
        ),
    ).json()["result"]

    assert first["source_match"] == second["source_match"]
    assert first["task_shape"] == second["task_shape"]
    assert first["derivation_id"] == second["derivation_id"]


def test_semantic_event_is_structural_and_hides_ambiguous_candidate_ids():
    runtime = _start_runtime()
    _derive(
        runtime,
        task_text="SECRETQUESTIONMARKER please answer this question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source(
                    "private_candidate_alpha",
                    "PRIVATE_DISPLAY_SENTINEL Alpha",
                    domain_tags=["PRIVATE_TAG_SENTINEL"],
                    scope_refs={"project": "PRIVATE_SCOPE_SENTINEL"},
                ),
                _discovery_source(
                    "private_candidate_beta",
                    "Private Beta",
                ),
            ),
            semantic_advisory=_semantic_advisory(
                "ambiguous",
                "lookup",
                "private_candidate_beta",
                "private_candidate_alpha",
            ),
        ),
    )
    payload = _shape_events(runtime["runtime_session_id"])[0]["event_payload_json"]
    serialized = json.dumps(payload, sort_keys=True).lower()

    assert payload["semantic_interpretation_status"] == "ambiguous"
    assert payload["semantic_operation_hint"] == "lookup"
    assert payload["semantic_candidate_count"] == 2
    assert payload["probe_source_count"] == 2
    assert "matched_source_ids" not in payload
    assert "probe_source_ids" not in payload
    for sentinel in (
        "secretquestionmarker",
        "private_display_sentinel",
        "private_tag_sentinel",
        "private_scope_sentinel",
        "private_candidate_alpha",
        "private_candidate_beta",
    ):
        assert sentinel not in serialized


@pytest.mark.parametrize(
    "semantic_advisory",
    [
        _semantic_advisory("resolved", "lookup", "source_a"),
        _semantic_advisory("no_match", "unknown"),
        _semantic_advisory("ambiguous", "comparison", "source_a", "source_b"),
    ],
)
def test_runtime_event_omits_probe_count_without_authorization(
    semantic_advisory: dict[str, object],
):
    runtime = _start_runtime()
    _derive(
        runtime,
        task_text="Please handle this question.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source("source_a", "Alpine Register"),
                _discovery_source("source_b", "Harbor Register"),
            ),
            semantic_advisory=semantic_advisory,
        ),
    )
    payload = _shape_events(runtime["runtime_session_id"])[0]["event_payload_json"]

    assert "probe_source_count" not in payload
    assert "probe_source_ids" not in payload


def test_absent_semantic_advisory_preserves_legacy_identity_and_wire_shape():
    request = EvidenceShapeDeriveRequest.model_validate(
        {
            "request_id": "request-legacy-fixed",
            "owner_id": "owner-legacy-fixed",
            "conversation_id": "conversation-legacy-fixed",
            "surface": "web",
            "runtime_session_id": "session-legacy-fixed",
            "runtime_turn_id": "turn-legacy-fixed",
            "task_text": "How are you today?",
            "interaction_kind": "question",
            "task_context": _context(),
        }
    )
    result = _derive_result(request)
    serialized = result.model_dump(mode="json", exclude_none=True)

    assert result.derivation_id == "evidence_shape_1cb58e5b82b64cddec197a154e54f053"
    assert "source_match" not in serialized
    assert "semantic_advisory" not in serialized
    assert all(not reason.startswith("semantic_") for reason in result.reason_codes)


def _source_entry_payload(**overrides: object) -> dict[str, object]:
    payload = _discovery_source("source_a", "Alpine Register")
    payload.update(overrides)
    return payload


def _fixed_shape_request(
    *,
    task_text: str,
    source_discovery: dict[str, object],
    semantic_advisory: dict[str, object] | None = None,
) -> EvidenceShapeDeriveRequest:
    return EvidenceShapeDeriveRequest.model_validate(
        {
            "request_id": "request-aggregate-contract-fixed",
            "owner_id": "owner-aggregate-contract-fixed",
            "conversation_id": "conversation-aggregate-contract-fixed",
            "surface": "web",
            "runtime_session_id": "session-aggregate-contract-fixed",
            "runtime_turn_id": "turn-aggregate-contract-fixed",
            "task_text": task_text,
            "interaction_kind": "question",
            "task_context": _context(
                source_discovery=source_discovery,
                semantic_advisory=semantic_advisory,
            ),
        }
    )


def test_source_discovery_content_fields_are_optional_and_omitted_when_absent():
    source = SourceDiscoveryEntry.model_validate(_source_entry_payload())

    assert source.content_fields is None
    assert "content_fields" not in source.model_dump(mode="json")


def test_source_discovery_accepts_sorted_exact_content_fields():
    source = SourceDiscoveryEntry.model_validate(
        _source_entry_payload(content_fields=["Date", "Fuel (L)", "Odometer"])
    )

    assert source.content_fields == ["Date", "Fuel (L)", "Odometer"]
    assert source.model_dump(mode="json")["content_fields"] == [
        "Date",
        "Fuel (L)",
        "Odometer",
    ]


def test_source_discovery_accepts_dsa_valid_outer_whitespace_unchanged():
    source = SourceDiscoveryEntry.model_validate(
        _source_entry_payload(content_fields=[" Fuel (L)", "Date", "Fuel (L) "])
    )

    assert source.content_fields == [" Fuel (L)", "Date", "Fuel (L) "]
    assert source.model_dump(mode="json")["content_fields"] == [
        " Fuel (L)",
        "Date",
        "Fuel (L) ",
    ]


def test_non_aggregate_shape_accepts_dsa_valid_outer_whitespace_metadata():
    request = _fixed_shape_request(
        task_text="Please handle this bounded question.",
        source_discovery=_discovery(
            _discovery_source(
                "source_a",
                "Alpine Register",
                content_fields=[" Fuel (L)"],
            )
        ),
        semantic_advisory=_semantic_advisory("resolved", "lookup", "source_a"),
    )

    result = _derive_result(request).model_dump(mode="json")
    assert request.task_context.source_discovery.sources[0].content_fields == [
        " Fuel (L)"
    ]
    assert result["derivation_status"] == "derived"
    assert result["task_shape"] == "targeted_lookup"
    assert result["source_match"]["matched_source_ids"] == ["source_a"]


@pytest.mark.parametrize(
    "content_fields",
    [
        None,
        [""],
        ["   "],
        ["Fuel\n(L)"],
        [1],
        ["x" * 121],
        [f"Field {index:02d}" for index in range(25)],
        ["Date", "Date"],
        ["Odometer", "Date"],
    ],
)
def test_source_discovery_rejects_invalid_content_fields(content_fields: object):
    with pytest.raises(ValidationError):
        SourceDiscoveryEntry.model_validate(
            _source_entry_payload(content_fields=content_fields)
        )


def test_legacy_aggregate_advisory_is_accepted_and_omits_new_fields():
    advisory = SemanticEvidenceAdvisory.model_validate(
        _semantic_advisory("resolved", "aggregate", "source_a")
    )

    assert advisory.aggregate_function is None
    assert advisory.aggregate_field_name is None
    assert advisory.model_dump(mode="json") == {
        "interpretation_status": "resolved",
        "operation_hint": "aggregate",
        "candidate_source_ids": ["source_a"],
    }


@pytest.mark.parametrize(
    "aggregate_function",
    ["median", "mean", "count", "sum", "minimum", "maximum"],
)
def test_enriched_aggregate_advisory_accepts_closed_functions(
    aggregate_function: str,
):
    advisory = SemanticEvidenceAdvisory.model_validate(
        _semantic_advisory(
            "resolved",
            "aggregate",
            "source_a",
            aggregate_function=aggregate_function,
            aggregate_field_name="Fuel (L)",
        )
    )

    assert advisory.aggregate_function == aggregate_function
    assert advisory.aggregate_field_name == "Fuel (L)"


@pytest.mark.parametrize("aggregate_function", ["average", "min", "max", "mode"])
def test_enriched_aggregate_advisory_rejects_unknown_functions(
    aggregate_function: str,
):
    with pytest.raises(ValidationError):
        SemanticEvidenceAdvisory.model_validate(
            _semantic_advisory(
                "resolved",
                "aggregate",
                "source_a",
                aggregate_function=aggregate_function,
                aggregate_field_name="Fuel (L)",
            )
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "interpretation_status": "resolved",
            "operation_hint": "aggregate",
            "candidate_source_ids": ["source_a"],
            "aggregate_function": "median",
        },
        {
            "interpretation_status": "resolved",
            "operation_hint": "aggregate",
            "candidate_source_ids": ["source_a"],
            "aggregate_field_name": "Fuel (L)",
        },
        {
            "interpretation_status": "resolved",
            "operation_hint": "aggregate",
            "candidate_source_ids": ["source_a"],
            "aggregate_function": None,
            "aggregate_field_name": "Fuel (L)",
        },
        {
            "interpretation_status": "resolved",
            "operation_hint": "aggregate",
            "candidate_source_ids": ["source_a"],
            "aggregate_function": "median",
            "aggregate_field_name": None,
        },
    ],
)
def test_aggregate_advisory_rejects_one_sided_or_null_details(
    payload: dict[str, object],
):
    with pytest.raises(ValidationError):
        SemanticEvidenceAdvisory.model_validate(payload)


@pytest.mark.parametrize("operation_hint", ["lookup", "latest", "comparison"])
def test_non_aggregate_advisory_rejects_aggregate_details(operation_hint: str):
    with pytest.raises(ValidationError):
        SemanticEvidenceAdvisory.model_validate(
            _semantic_advisory(
                "resolved",
                operation_hint,
                "source_a",
                aggregate_function="median",
                aggregate_field_name="Fuel (L)",
            )
        )


@pytest.mark.parametrize(
    "aggregate_field_name",
    ["", "   ", " Fuel (L)", "Fuel (L) ", "Fuel\x7f(L)", "x" * 121],
)
def test_aggregate_advisory_rejects_invalid_field_names(
    aggregate_field_name: str,
):
    with pytest.raises(ValidationError):
        SemanticEvidenceAdvisory.model_validate(
            _semantic_advisory(
                "resolved",
                "aggregate",
                "source_a",
                aggregate_function="median",
                aggregate_field_name=aggregate_field_name,
            )
        )


def test_enriched_resolved_aggregate_requires_exact_candidate_field():
    request = _fixed_shape_request(
        task_text="Calculate the requested statistic.",
        source_discovery=_discovery(
            _discovery_source(
                "source_a",
                "Alpine Register",
                content_fields=["Date", "Fuel (L)", "Odometer"],
            )
        ),
        semantic_advisory=_semantic_advisory(
            "resolved",
            "aggregate",
            "source_a",
            aggregate_function="median",
            aggregate_field_name="Fuel (L)",
        ),
    )

    result = _derive_result(request).model_dump(mode="json")
    _assert_aggregate_operation_blocked(result)
    assert result["source_match"]["matched_source_ids"] == ["source_a"]
    assert "probe_source_ids" not in result["source_match"]


@pytest.mark.parametrize(
    ("source_a_fields", "source_b_fields", "aggregate_field_name"),
    [
        (None, ["Fuel (L)"], "Fuel (L)"),
        (["Date"], ["Fuel (L)"], "Fuel (L)"),
        ([" Fuel (L)"], ["Fuel (L)"], "Fuel (L)"),
        (["Fuel (L)"], ["Fuel (L)"], "fuel (l)"),
        (["Fuel (L)"], ["Fuel (L)"], "Fuel"),
    ],
)
def test_enriched_aggregate_rejects_missing_or_inexact_candidate_field(
    source_a_fields: list[str] | None,
    source_b_fields: list[str],
    aggregate_field_name: str,
):
    with pytest.raises(ValidationError):
        _fixed_shape_request(
            task_text="Calculate the requested statistic.",
            source_discovery=_discovery(
                _discovery_source(
                    "source_a",
                    "Alpine Register",
                    content_fields=source_a_fields,
                ),
                _discovery_source(
                    "source_b",
                    "Harbor Register",
                    content_fields=source_b_fields,
                ),
            ),
            semantic_advisory=_semantic_advisory(
                "resolved",
                "aggregate",
                "source_a",
                aggregate_function="median",
                aggregate_field_name=aggregate_field_name,
            ),
        )


def test_enriched_aggregate_rejects_field_found_only_on_unrelated_source():
    with pytest.raises(ValidationError):
        _fixed_shape_request(
            task_text="Calculate the requested statistic.",
            source_discovery=_discovery(
                _discovery_source(
                    "source_a", "Alpine Register", content_fields=["Date"]
                ),
                _discovery_source(
                    "source_b",
                    "Harbor Register",
                    content_fields=["Fuel (L)"],
                ),
            ),
            semantic_advisory=_semantic_advisory(
                "resolved",
                "aggregate",
                "source_a",
                aggregate_function="median",
                aggregate_field_name="Fuel (L)",
            ),
        )


@pytest.mark.parametrize("candidate_count", [2, 3])
def test_enriched_aggregate_ambiguity_requires_field_on_every_candidate(
    candidate_count: int,
):
    sources = [
        _discovery_source(
            f"source_{index}",
            f"Register {index}",
            content_fields=["Date", "Fuel (L)"],
        )
        for index in range(candidate_count)
    ]
    candidate_ids = [f"source_{index}" for index in range(candidate_count)]
    request = _fixed_shape_request(
        task_text="Calculate the requested statistic.",
        source_discovery=_discovery(*sources),
        semantic_advisory=_semantic_advisory(
            "ambiguous",
            "aggregate",
            *candidate_ids,
            aggregate_function="median",
            aggregate_field_name="Fuel (L)",
        ),
    )

    result = _derive_result(request).model_dump(mode="json")
    _assert_aggregate_operation_blocked(result)
    assert result["source_match"]["status"] == "ambiguous"
    assert result["source_match"]["matched_source_ids"] == []
    assert "probe_source_ids" not in result["source_match"]


def test_enriched_aggregate_ambiguity_rejects_field_missing_from_one_candidate():
    with pytest.raises(ValidationError):
        _fixed_shape_request(
            task_text="Calculate the requested statistic.",
            source_discovery=_discovery(
                _discovery_source(
                    "source_a", "Alpine Register", content_fields=["Fuel (L)"]
                ),
                _discovery_source(
                    "source_b", "Harbor Register", content_fields=["Date"]
                ),
            ),
            semantic_advisory=_semantic_advisory(
                "ambiguous",
                "aggregate",
                "source_a",
                "source_b",
                aggregate_function="median",
                aggregate_field_name="Fuel (L)",
            ),
        )


@pytest.mark.parametrize("enriched", [False, True])
def test_aggregate_no_match_accepts_legacy_and_enriched_contracts(enriched: bool):
    aggregate_details = (
        {"aggregate_function": "median", "aggregate_field_name": "Fuel (L)"}
        if enriched
        else {}
    )
    request = _fixed_shape_request(
        task_text="Calculate the requested statistic.",
        source_discovery=_discovery(),
        semantic_advisory=_semantic_advisory(
            "no_match",
            "aggregate",
            **aggregate_details,
        ),
    )

    result = _derive_result(request).model_dump(mode="json")
    _assert_aggregate_operation_blocked(result)
    assert result["source_match"]["status"] == "no_match"
    assert result["source_match"]["matched_source_ids"] == []


def test_unknown_aggregate_candidate_remains_rejected_by_inventory_binding():
    with pytest.raises(ValidationError):
        _fixed_shape_request(
            task_text="Calculate the requested statistic.",
            source_discovery=_discovery(
                _discovery_source(
                    "source_a", "Alpine Register", content_fields=["Fuel (L)"]
                )
            ),
            semantic_advisory=_semantic_advisory(
                "resolved",
                "aggregate",
                "source_unknown",
                aggregate_function="median",
                aggregate_field_name="Fuel (L)",
            ),
        )


def test_legacy_contract_fields_preserve_serialized_semantics():
    request = _fixed_shape_request(
        task_text="Calculate the requested statistic.",
        source_discovery=_discovery(
            _discovery_source("source_a", "Alpine Register")
        ),
        semantic_advisory=_semantic_advisory(
            "resolved", "aggregate", "source_a"
        ),
    )
    serialized_context = request.task_context.model_dump(mode="json")

    assert "content_fields" not in serialized_context["source_discovery"]["sources"][0]
    assert "aggregate_function" not in serialized_context["semantic_advisory"]
    assert "aggregate_field_name" not in serialized_context["semantic_advisory"]
    result = _derive_result(request).model_dump(mode="json")
    _assert_aggregate_operation_blocked(result)


def test_enriched_aggregate_derivation_is_order_independent_and_meaning_bound():
    sources = (
        _discovery_source(
            "source_a",
            "Alpine Register",
            content_fields=["Date", "Fuel (L)", "Odometer"],
        ),
        _discovery_source(
            "source_b",
            "Harbor Register",
            content_fields=["Date", "Fuel (L)", "Odometer"],
        ),
    )

    def derive(
        *,
        inventory: tuple[dict[str, object], ...],
        candidates: tuple[str, ...],
        function: str = "median",
        field: str = "Fuel (L)",
    ) -> dict[str, object]:
        request = _fixed_shape_request(
            task_text="Calculate the requested statistic.",
            source_discovery=_discovery(*inventory),
            semantic_advisory=_semantic_advisory(
                "ambiguous",
                "aggregate",
                *candidates,
                aggregate_function=function,
                aggregate_field_name=field,
            ),
        )
        return _derive_result(request).model_dump(mode="json")

    first = derive(inventory=sources, candidates=("source_b", "source_a"))
    reordered = derive(
        inventory=tuple(reversed(sources)),
        candidates=("source_a", "source_b"),
    )
    changed_function = derive(
        inventory=sources,
        candidates=("source_a", "source_b"),
        function="mean",
    )
    changed_field = derive(
        inventory=tuple(
            {
                **source,
                "content_fields": ["Date", "Fuel (L)", "Odometer"],
            }
            for source in sources
        ),
        candidates=("source_a", "source_b"),
        field="Odometer",
    )

    assert first["source_match"] == reordered["source_match"]
    assert first["derivation_id"] == reordered["derivation_id"]
    assert first["derivation_id"] != changed_function["derivation_id"]
    assert first["derivation_id"] != changed_field["derivation_id"]


def test_enriched_aggregate_event_omits_contract_metadata():
    runtime = _start_runtime()
    response = _derive(
        runtime,
        task_text="Calculate the requested statistic.",
        task_context=_context(
            source_discovery=_discovery(
                _discovery_source(
                    "source_a",
                    "Alpine Register",
                    content_fields=["PRIVATE_AGGREGATE_FIELD"],
                )
            ),
            semantic_advisory=_semantic_advisory(
                "resolved",
                "aggregate",
                "source_a",
                aggregate_function="median",
                aggregate_field_name="PRIVATE_AGGREGATE_FIELD",
            ),
        ),
    )
    assert response.status_code == 200

    payload = _shape_events(runtime["runtime_session_id"])[0]["event_payload_json"]
    serialized = json.dumps(payload, sort_keys=True)
    assert payload["semantic_operation_hint"] == "aggregate"
    assert "aggregate_function" not in payload
    assert "aggregate_field_name" not in payload
    assert "content_fields" not in payload
    assert "PRIVATE_AGGREGATE_FIELD" not in serialized
