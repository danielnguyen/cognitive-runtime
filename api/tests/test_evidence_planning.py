from __future__ import annotations

import json
from itertools import count

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
_runtime_counter = count()


def _start_runtime(
    *,
    owner_id: str | None = None,
    conversation_id: str | None = None,
    surface: str = "web",
) -> dict[str, object]:
    ordinal = next(_runtime_counter)
    resolved_owner = owner_id or f"owner-plan-{ordinal}"
    resolved_conversation = conversation_id or f"conversation-plan-{ordinal}"
    response = client.post(
        "/v1/runtime/turns/start",
        json={
            "request_id": f"request-plan-start-{ordinal}",
            "owner_id": resolved_owner,
            "conversation_id": resolved_conversation,
            "surface": surface,
        },
    )
    assert response.status_code == 200
    body = response.json()
    return {
        "request_id": f"request-plan-compile-{ordinal}",
        "owner_id": resolved_owner,
        "conversation_id": resolved_conversation,
        "surface": surface,
        "runtime_session_id": body["runtime_session"]["runtime_session_id"],
        "runtime_turn_id": body["runtime_turn"]["runtime_turn_id"],
    }


def _scope(
    *,
    source_ids: list[str] | None = None,
    source_categories: list[str] | None = None,
    inventory_status: str = "complete_for_declared_scope",
    time_scope_ref: str | None = None,
) -> dict[str, object]:
    return {
        "source_ids": source_ids or [],
        "source_categories": source_categories or [],
        "inventory_status": inventory_status,
        "time_scope_ref": time_scope_ref,
    }


def _source(
    source_id: str,
    *,
    categories: list[str] | None = None,
    capabilities: list[str] | None = None,
    availability: str = "available",
    authority_role: str = "authoritative",
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_categories": categories or ["records"],
        "capabilities": capabilities or ["targeted_retrieval"],
        "availability": availability,
        "authority_role": authority_role,
    }


def _compile(
    runtime: dict[str, object],
    *,
    task_shape: str = "targeted_lookup",
    declared_scope: dict[str, object] | None = None,
    source_inventory: list[dict[str, object]] | None = None,
    question_anchor: str = "Which retained setting is active?",
    **overrides: object,
):
    payload: dict[str, object] = {
        **runtime,
        "question_anchor": question_anchor,
        "task_shape": task_shape,
        "declared_scope": declared_scope or _scope(),
        "source_inventory": source_inventory
        if source_inventory is not None
        else [
            _source(
                "source-a",
                capabilities=["targeted_retrieval", "exact_fetch"],
            )
        ],
    }
    payload.update(overrides)
    return client.post("/v1/runtime/evidence-plans/compile", json=payload)


def _plan_events(runtime_session_id: object) -> list[dict[str, object]]:
    response = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert response.status_code == 200
    return [
        event
        for event in response.json()["events"]
        if event["event_type"] == "evidence_plan_compiled"
    ]


def _requirement_kinds(result: dict[str, object]) -> set[str]:
    return {
        requirement["requirement_kind"]
        for requirement in result["declared_requirements"]
    }


@pytest.mark.parametrize(
    (
        "task_shape",
        "declared_scope",
        "inventory",
        "completeness",
        "contradiction_required",
        "strategy",
        "required_kinds",
    ),
    [
        (
            "targeted_lookup",
            _scope(source_ids=["source-a"]),
            [_source("source-a", capabilities=["exact_fetch"])],
            "targeted_scope",
            False,
            "exact_fetch",
            {"exact_authoritative_fetch", "context_delivery"},
        ),
        (
            "bounded_exhaustive_review",
            _scope(source_ids=["source-a"]),
            [_source("source-a", capabilities=["structured_query"])],
            "complete_for_declared_scope",
            True,
            "structured_query",
            {"authoritative_inventory", "complete_scope_coverage"},
        ),
        (
            "cross_source_comparison",
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "exact_fetch"],
                ),
                _source("source-b", capabilities=["targeted_retrieval"]),
            ],
            "complete_for_selected_sources",
            False,
            "hybrid",
            {"selected_source_coverage", "cross_source_comparison"},
        ),
        (
            "contradiction_review",
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
                _source(
                    "source-b",
                    capabilities=["targeted_retrieval", "exact_fetch"],
                ),
            ],
            "complete_for_selected_sources",
            True,
            "hybrid",
            {"contradiction_search", "counterevidence_coverage"},
        ),
        (
            "absence_or_coverage_check",
            _scope(source_ids=["source-a"]),
            [_source("source-a", capabilities=["structured_query"])],
            "complete_for_declared_scope",
            False,
            "structured_query",
            {"authoritative_inventory", "structured_absence_check"},
        ),
        (
            "historical_reconstruction",
            _scope(source_ids=["source-a"], time_scope_ref="window-2024"),
            [_source("source-a", capabilities=["structured_query"])],
            "complete_for_time_window",
            False,
            "structured_query",
            {"historical_scope", "historical_sequence_coverage"},
        ),
        (
            "recommendation_or_decision_support",
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "exact_fetch"],
                ),
                _source(
                    "source-b",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
            ],
            "bounded_decision_support",
            True,
            "hybrid",
            {"candidate_evidence_coverage", "counterevidence_coverage"},
        ),
    ],
)
def test_each_task_shape_produces_a_distinct_ready_plan(
    task_shape: str,
    declared_scope: dict[str, object],
    inventory: list[dict[str, object]],
    completeness: str,
    contradiction_required: bool,
    strategy: str,
    required_kinds: set[str],
):
    response = _compile(
        _start_runtime(),
        task_shape=task_shape,
        declared_scope=declared_scope,
        source_inventory=inventory,
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["plan_status"] == "ready"
    assert result["completeness_expectation"] == completeness
    assert result["contradiction_search_required"] is contradiction_required
    assert result["selected_strategies"] == [strategy]
    assert required_kinds <= _requirement_kinds(result)
    assert "context_delivery" in _requirement_kinds(result)


def test_question_anchor_is_normalized_and_exact_authoritative_fetch_is_preferred():
    response = _compile(
        _start_runtime(),
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[
            _source(
                "source-a",
                capabilities=["targeted_retrieval", "exact_fetch"],
            )
        ],
        question_anchor="  Which   setting\n is active?  ",
    )
    result = response.json()["result"]

    assert result["question_anchor"] == "Which setting is active?"
    assert result["question_anchor_digest"].startswith("sha256:")
    assert len(result["question_anchor_digest"]) == 71
    assert result["plan_status"] == "ready"
    assert result["selected_strategies"] == ["exact_fetch"]
    assert result["eligible_source_ids"] == ["source-a"]
    assert result["authoritative_source_ids"] == ["source-a"]
    assert "exact_authoritative_fetch" in _requirement_kinds(result)


def test_targeted_only_inventory_is_not_an_exhaustive_plan():
    response = _compile(
        _start_runtime(),
        task_shape="bounded_exhaustive_review",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[_source("source-a", capabilities=["targeted_retrieval"])],
    )
    result = response.json()["result"]

    assert result["plan_status"] == "unsupported"
    assert result["selected_strategies"] == []
    assert "targeted_only_not_exhaustive" in result["limitation_codes"]
    assert "complete_scope_coverage" in _requirement_kinds(result)
    assert "authoritative_inventory" in _requirement_kinds(result)


def test_capability_union_does_not_upgrade_each_source_to_exhaustive_support():
    result = _compile(
        _start_runtime(),
        task_shape="bounded_exhaustive_review",
        declared_scope=_scope(source_ids=["source-a", "source-b"]),
        source_inventory=[
            _source("source-a", capabilities=["structured_query"]),
            _source("source-b", capabilities=["targeted_retrieval"]),
        ],
    ).json()["result"]

    assert result["plan_status"] == "unsupported"
    assert result["selected_strategies"] == []
    assert "required_capability_unavailable" in result["limitation_codes"]


@pytest.mark.parametrize("inventory_status", ["partial", "unknown", "unavailable"])
def test_absence_plan_requires_complete_inventory(inventory_status: str):
    response = _compile(
        _start_runtime(),
        task_shape="absence_or_coverage_check",
        declared_scope=_scope(
            source_ids=["source-a"],
            inventory_status=inventory_status,
        ),
        source_inventory=[_source("source-a", capabilities=["structured_query"])],
    )
    result = response.json()["result"]

    assert result["plan_status"] == "unsupported"
    assert "absence_scope_not_enumerable" in result["limitation_codes"]
    assert f"source_inventory_{inventory_status}" in result["limitation_codes"]


def test_absence_plan_requires_authority_and_enumeration_capability():
    no_authority = _compile(
        _start_runtime(),
        task_shape="absence_or_coverage_check",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[
            _source(
                "source-a",
                capabilities=["structured_query"],
                authority_role="supplemental",
            )
        ],
    ).json()["result"]
    targeted_only = _compile(
        _start_runtime(),
        task_shape="absence_or_coverage_check",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[_source("source-a", capabilities=["targeted_retrieval"])],
    ).json()["result"]

    assert no_authority["plan_status"] == "unsupported"
    assert "authoritative_source_missing" in no_authority["limitation_codes"]
    assert targeted_only["plan_status"] == "unsupported"
    assert "absence_scope_not_enumerable" in targeted_only["limitation_codes"]
    assert "required_capability_unavailable" in targeted_only["limitation_codes"]


def test_comparison_requires_two_sources_and_contradiction_rejects_targeted_only():
    comparison_negative = _compile(
        _start_runtime(),
        task_shape="cross_source_comparison",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[_source("source-a")],
    ).json()["result"]
    comparison_positive = _compile(
        _start_runtime(),
        task_shape="cross_source_comparison",
        declared_scope=_scope(source_ids=["source-a", "source-b"]),
        source_inventory=[_source("source-a"), _source("source-b")],
    ).json()["result"]
    comparison_exact = _compile(
        _start_runtime(),
        task_shape="cross_source_comparison",
        declared_scope=_scope(source_ids=["source-a", "source-b"]),
        source_inventory=[
            _source("source-a", capabilities=["exact_fetch"]),
            _source("source-b", capabilities=["exact_fetch"]),
        ],
    ).json()["result"]
    contradiction_negative = _compile(
        _start_runtime(),
        task_shape="contradiction_review",
        declared_scope=_scope(source_ids=["source-a", "source-b"]),
        source_inventory=[_source("source-a"), _source("source-b")],
    ).json()["result"]
    contradiction_positive = _compile(
        _start_runtime(),
        task_shape="contradiction_review",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[
            _source("source-a", capabilities=["bounded_full_context"])
        ],
    ).json()["result"]

    assert comparison_negative["plan_status"] == "unsupported"
    assert "insufficient_comparison_scope" in comparison_negative["limitation_codes"]
    assert comparison_positive["plan_status"] == "ready"
    assert comparison_positive["selected_strategies"] == ["targeted_retrieval"]
    assert comparison_exact["plan_status"] == "ready"
    assert comparison_exact["selected_strategies"] == ["exact_fetch"]
    assert contradiction_negative["plan_status"] == "unsupported"
    assert "contradiction_search_not_supported" in contradiction_negative[
        "limitation_codes"
    ]
    assert contradiction_positive["plan_status"] == "ready"
    assert contradiction_positive["selected_strategies"] == [
        "bounded_full_context"
    ]


def test_historical_and_decision_support_prerequisites_are_enforced():
    historical_missing_time = _compile(
        _start_runtime(),
        task_shape="historical_reconstruction",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[_source("source-a", capabilities=["structured_query"])],
    ).json()["result"]
    historical_targeted = _compile(
        _start_runtime(),
        task_shape="historical_reconstruction",
        declared_scope=_scope(source_ids=["source-a"], time_scope_ref="window-1"),
        source_inventory=[_source("source-a", capabilities=["targeted_retrieval"])],
    ).json()["result"]
    decision_negative = _compile(
        _start_runtime(),
        task_shape="recommendation_or_decision_support",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[
            _source(
                "source-a",
                capabilities=["targeted_retrieval", "exact_fetch"],
            )
        ],
    ).json()["result"]

    assert historical_missing_time["plan_status"] == "unsupported"
    assert "historical_time_scope_missing" in historical_missing_time[
        "limitation_codes"
    ]
    assert historical_targeted["plan_status"] == "unsupported"
    assert "historical_sequence_not_supported" in historical_targeted[
        "limitation_codes"
    ]
    assert decision_negative["plan_status"] == "unsupported"
    assert "decision_support_scope_insufficient" in decision_negative[
        "limitation_codes"
    ]
    assert {
        "candidate_evidence_coverage",
        "cross_source_comparison",
        "counterevidence_coverage",
    } <= _requirement_kinds(decision_negative)


def test_declared_universe_and_available_eligible_sources_remain_distinct():
    result = _compile(
        _start_runtime(),
        declared_scope=_scope(
            source_ids=["source-a", "source-b", "source-c", "source-missing"]
        ),
        source_inventory=[
            _source(
                "source-a",
                capabilities=["exact_fetch"],
            ),
            _source(
                "source-b",
                capabilities=["exact_fetch"],
                availability="disabled",
                authority_role="supplemental",
            ),
            _source(
                "source-c",
                capabilities=["exact_fetch"],
                availability="unknown",
                authority_role="authoritative",
            ),
            _source(
                "source-other",
                categories=["other"],
                capabilities=["exact_fetch"],
            ),
        ],
    ).json()["result"]

    assert result["eligible_source_ids"] == ["source-a"]
    assert result["authoritative_source_ids"] == ["source-a"]
    assert result["plan_status"] == "unsupported"
    assert {
        "declared_source_missing_from_inventory",
        "optional_source_unavailable",
        "authoritative_source_unavailable",
    } <= set(result["limitation_codes"])
    assert "source-other" not in result["eligible_source_ids"]


def test_category_mismatch_is_visible_and_does_not_redefine_scope():
    result = _compile(
        _start_runtime(),
        declared_scope=_scope(source_categories=["required-category"]),
        source_inventory=[
            _source(
                "source-a",
                categories=["different-category"],
                capabilities=["exact_fetch"],
            )
        ],
    ).json()["result"]

    assert result["plan_status"] == "unsupported"
    assert result["eligible_source_ids"] == []
    assert "declared_category_not_available" in result["limitation_codes"]


def test_empty_inventory_is_an_unsupported_plan_not_proof_of_empty_scope():
    result = _compile(
        _start_runtime(),
        task_shape="absence_or_coverage_check",
        declared_scope=_scope(inventory_status="unknown"),
        source_inventory=[],
    ).json()["result"]

    assert result["plan_status"] == "unsupported"
    assert result["eligible_source_ids"] == []
    assert "source_inventory_unknown" in result["limitation_codes"]
    assert "absence_scope_not_enumerable" in result["limitation_codes"]


def _evaluate_compiled_requirements(
    runtime: dict[str, object],
    plan: dict[str, object],
    facts: list[dict[str, str]],
):
    return client.post(
        "/v1/runtime/evidence-sufficiency/evaluate",
        json={
            **runtime,
            "request_id": f"{runtime['request_id']}-sufficiency",
            "evidence_plan_id": plan["plan_id"],
            "acquisition_manifest_id": "manifest-compatibility",
            "task_shape": plan["task_shape"],
            "declared_requirements": plan["declared_requirements"],
            "acquisition_facts": facts,
        },
    )


def test_compiled_requirements_submit_unchanged_to_sufficiency_evaluator():
    runtime = _start_runtime()
    plan = _compile(
        runtime,
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[_source("source-a", capabilities=["exact_fetch"])],
    ).json()["result"]
    facts = [
        {"requirement_id": requirement["requirement_id"], "outcome": "satisfied"}
        for requirement in plan["declared_requirements"]
    ]
    complete = _evaluate_compiled_requirements(runtime, plan, facts)
    missing = _evaluate_compiled_requirements(runtime, plan, facts[1:])
    failed_facts = list(facts)
    failed_facts[0] = {**failed_facts[0], "outcome": "failed"}
    failed = _evaluate_compiled_requirements(runtime, plan, failed_facts)

    assert complete.status_code == 200
    assert complete.json()["result"]["sufficiency_status"] == (
        "sufficient_for_declared_scope"
    )
    assert missing.json()["result"]["sufficiency_status"] == "unknown"
    assert failed.json()["result"]["sufficiency_status"] == "insufficient"


def test_compiled_optional_requirement_produces_sufficient_with_limitations():
    runtime = _start_runtime()
    plan = _compile(
        runtime,
        declared_scope=_scope(
            source_ids=["source-a"],
            inventory_status="partial",
        ),
        source_inventory=[_source("source-a", capabilities=["exact_fetch"])],
    ).json()["result"]
    assert plan["plan_status"] == "ready_with_limitations"
    optional = [
        requirement
        for requirement in plan["declared_requirements"]
        if requirement["criticality"] == "optional"
    ]
    assert len(optional) == 1
    facts = [
        {
            "requirement_id": requirement["requirement_id"],
            "outcome": (
                "unavailable"
                if requirement["criticality"] == "optional"
                else "satisfied"
            ),
        }
        for requirement in plan["declared_requirements"]
    ]
    response = _evaluate_compiled_requirements(runtime, plan, facts)

    assert response.status_code == 200
    assert response.json()["result"]["sufficiency_status"] == (
        "sufficient_with_limitations"
    )
    assert response.json()["result"]["additional_acquisition_required"] is False


def test_equivalent_reordered_inputs_produce_identical_complete_results():
    runtime = _start_runtime()
    scope = _scope(
        source_ids=["source-b", "source-a"],
        source_categories=["records", "reference"],
    )
    inventory = [
        _source(
            "source-b",
            categories=["reference", "records"],
            capabilities=["exact_fetch", "targeted_retrieval"],
        ),
        _source(
            "source-a",
            categories=["records", "reference"],
            capabilities=["targeted_retrieval", "exact_fetch"],
        ),
    ]
    first = _compile(
        runtime,
        task_shape="cross_source_comparison",
        declared_scope=scope,
        source_inventory=inventory,
    )
    second_scope = {
        **scope,
        "source_ids": list(reversed(scope["source_ids"])),
        "source_categories": list(reversed(scope["source_categories"])),
    }
    second_inventory = [
        {
            **source,
            "source_categories": list(reversed(source["source_categories"])),
            "capabilities": list(reversed(source["capabilities"])),
        }
        for source in reversed(inventory)
    ]
    second = _compile(
        runtime,
        task_shape="cross_source_comparison",
        declared_scope=second_scope,
        source_inventory=second_inventory,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["result"] == second.json()["result"]


@pytest.mark.parametrize(
    "extra_field",
    [
        "metadata",
        "description",
        "url",
        "credentials",
        "connector_config",
        "result_count",
        "citation_count",
        "provider_text",
        "confidence",
        "plan_status",
        "contradiction_search_required",
    ],
)
def test_caller_selected_or_unrestricted_top_level_fields_are_rejected(
    extra_field: str,
):
    response = _compile(
        _start_runtime(),
        **{extra_field: "caller-supplied"},
    )

    assert response.status_code == 422
    assert "extra_forbidden" in response.text


@pytest.mark.parametrize(
    "extra_field",
    [
        "metadata",
        "description",
        "display_text",
        "url",
        "credentials",
        "connector_config",
        "source_content",
        "prompt",
        "confidence",
        "result_count",
        "provider_text",
    ],
)
def test_unrestricted_source_fields_are_rejected(extra_field: str):
    source = _source("source-a")
    source[extra_field] = "not accepted"
    response = _compile(_start_runtime(), source_inventory=[source])

    assert response.status_code == 422
    assert "extra_forbidden" in response.text


def test_duplicate_and_over_limit_inventory_values_are_rejected():
    runtime = _start_runtime()
    duplicate_sources = _compile(
        runtime,
        source_inventory=[_source("source-a"), _source("source-a")],
    )
    duplicate_scope_ids = _compile(
        runtime,
        declared_scope=_scope(source_ids=["source-a", "source-a"]),
    )
    duplicate_scope_categories = _compile(
        runtime,
        declared_scope=_scope(source_categories=["records", "records"]),
    )
    duplicate_categories = _source(
        "source-a",
        categories=["records", "records"],
    )
    duplicate_capabilities = _source(
        "source-a",
        capabilities=["exact_fetch", "exact_fetch"],
    )
    too_many_sources = [_source(f"source-{index}") for index in range(33)]
    too_many_ids = [f"source-{index}" for index in range(33)]
    too_many_categories = [f"category-{index}" for index in range(17)]

    responses = [
        duplicate_sources,
        duplicate_scope_ids,
        duplicate_scope_categories,
        _compile(runtime, source_inventory=[duplicate_categories]),
        _compile(runtime, source_inventory=[duplicate_capabilities]),
        _compile(runtime, source_inventory=too_many_sources),
        _compile(runtime, declared_scope=_scope(source_ids=too_many_ids)),
        _compile(
            runtime,
            declared_scope=_scope(source_categories=too_many_categories),
        ),
    ]
    assert all(response.status_code == 422 for response in responses)


def test_identifier_question_and_nested_scope_bounds_are_enforced():
    runtime = _start_runtime()
    unsafe_source = _source("https://example.invalid/source?token=private")
    blank_question = _compile(runtime, question_anchor=" \n \t ")
    long_question = _compile(runtime, question_anchor="q" * 501)
    unsafe_scope_ref = _scope(time_scope_ref="https://example.invalid/window?q=1")
    too_many_source_categories = _source(
        "source-a",
        categories=[f"category-{index}" for index in range(9)],
    )

    responses = [
        _compile(runtime, source_inventory=[unsafe_source]),
        blank_question,
        long_question,
        _compile(runtime, declared_scope=unsafe_scope_ref),
        _compile(runtime, source_inventory=[too_many_source_categories]),
    ]
    assert all(response.status_code == 422 for response in responses)


def test_unknown_runtime_session_returns_bounded_error_and_no_event():
    runtime = _start_runtime()
    original_session_id = runtime["runtime_session_id"]
    runtime["runtime_session_id"] = "rtsession-missing"
    response = _compile(runtime)

    assert response.status_code == 404
    assert response.json() == {"detail": "runtime_session_not_found"}
    assert _plan_events(original_session_id) == []


def test_unknown_runtime_turn_returns_bounded_error_and_no_event():
    runtime = _start_runtime()
    runtime["runtime_turn_id"] = "rtturn-missing"
    response = _compile(runtime)

    assert response.status_code == 404
    assert response.json() == {"detail": "runtime_turn_not_found"}
    assert _plan_events(runtime["runtime_session_id"]) == []


def test_runtime_session_scope_mismatch_returns_bounded_error_and_no_event():
    runtime = _start_runtime()
    runtime["owner_id"] = "another-owner"
    response = _compile(runtime)

    assert response.status_code == 400
    assert response.json() == {"detail": "runtime_session_mismatch"}
    assert _plan_events(runtime["runtime_session_id"]) == []


def test_runtime_turn_session_mismatch_returns_bounded_error_and_no_event():
    first = _start_runtime()
    second = _start_runtime()
    second["runtime_turn_id"] = first["runtime_turn_id"]
    response = _compile(second)

    assert response.status_code == 400
    assert response.json() == {"detail": "runtime_turn_session_mismatch"}
    assert _plan_events(second["runtime_session_id"]) == []


def test_response_and_runtime_event_are_bounded_private_and_consistent():
    runtime = _start_runtime()
    response = _compile(
        runtime,
        declared_scope=_scope(
            source_ids=["private-source"],
            source_categories=["private-category"],
        ),
        source_inventory=[
            _source(
                "private-source",
                categories=["private-category"],
                capabilities=["exact_fetch"],
            )
        ],
        question_anchor="PRIVATE QUESTION CONTENT",
    )
    assert response.status_code == 200
    body = response.json()
    result = body["result"]
    assert set(body) == {
        "request_id",
        "owner_id",
        "conversation_id",
        "surface",
        "runtime_session_id",
        "runtime_turn_id",
        "result",
    }
    assert set(result) == {
        "plan_id",
        "question_anchor",
        "question_anchor_digest",
        "task_shape",
        "plan_status",
        "completeness_expectation",
        "contradiction_search_required",
        "eligible_source_ids",
        "authoritative_source_ids",
        "selected_strategies",
        "declared_requirements",
        "limitation_codes",
        "user_safe_summary",
    }

    event = _plan_events(runtime["runtime_session_id"])[0]
    payload = event["event_payload_json"]
    assert set(payload) == {
        "request_id",
        "runtime_session_id",
        "runtime_turn_id",
        "plan_id",
        "question_anchor_digest",
        "task_shape",
        "plan_status",
        "completeness_expectation",
        "contradiction_search_required",
        "source_inventory_count",
        "eligible_source_count",
        "authoritative_source_count",
        "material_requirement_count",
        "optional_requirement_count",
        "selected_strategies",
        "limitation_codes",
    }
    for field in (
        "plan_id",
        "question_anchor_digest",
        "task_shape",
        "plan_status",
        "completeness_expectation",
        "contradiction_search_required",
        "selected_strategies",
        "limitation_codes",
    ):
        assert payload[field] == result[field]
    assert payload["source_inventory_count"] == 1
    assert payload["eligible_source_count"] == 1
    assert payload["authoritative_source_count"] == 1
    assert payload["material_requirement_count"] == len(
        result["declared_requirements"]
    )
    assert payload["optional_requirement_count"] == 0

    serialized = json.dumps(payload, sort_keys=True).lower()
    for excluded in (
        "private question content",
        "private-source",
        "private-category",
        "question_anchor\"",
        '"source_inventory":',
        '"source_ids":',
        '"source_categories":',
        "requirement_id",
        "requirement_kind",
        "connector",
        "credentials",
        "prompt",
        "private memory",
        "reasoning",
        "metadata",
        "exception",
    ):
        assert excluded not in serialized
