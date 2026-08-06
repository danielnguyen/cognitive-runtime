from __future__ import annotations

import json
from itertools import count

import pytest
from fastapi.testclient import TestClient
from main import app

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
) -> dict[str, object]:
    return {
        "evidence_input_kinds": evidence_input_kinds or [],
        "external_verification_required": external_verification_required,
        "freshness_sensitive": freshness_sensitive,
        "high_stakes_accuracy_required": high_stakes_accuracy_required,
        "continuation_of_prior_evidence_task": continuation_of_prior_evidence_task,
        "prior_task_shape": prior_task_shape,
    }


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
