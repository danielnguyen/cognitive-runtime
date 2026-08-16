from __future__ import annotations

import json
from itertools import count

import pytest
from fastapi.testclient import TestClient
from main import app
from models import (
    AggregateSpec,
    EvidenceAcquisitionPremise,
    EvidencePlanCompileRequest,
    EvidencePlanResult,
    EvidenceSourceDescriptor,
)
from pydantic import ValidationError
from services.evidence_planning import (
    _normalized_inventory,
    _normalized_scope,
    _plan_identity,
    evidence_acquisition_premise_digest,
)

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
    exact_source_refs: list[dict[str, str]] | None = None,
    inventory_status: str = "complete_for_declared_scope",
    time_scope_ref: str | None = None,
) -> dict[str, object]:
    return {
        "source_ids": source_ids or [],
        "source_categories": source_categories or [],
        "exact_source_refs": exact_source_refs or [],
        "inventory_status": inventory_status,
        "time_scope_ref": time_scope_ref,
    }


def _exact_ref(source_id: str, native_ref: str) -> dict[str, str]:
    return {
        "source_id": source_id,
        "source_ref": f"connector:{source_id}:{native_ref}",
    }


def _source(
    source_id: str,
    *,
    categories: list[str] | None = None,
    capabilities: list[str] | None = None,
    availability: str = "available",
    authority_role: str = "authoritative",
    content_fields: list[str] | None = None,
) -> dict[str, object]:
    source: dict[str, object] = {
        "source_id": source_id,
        "source_categories": categories or ["records"],
        "capabilities": capabilities or ["targeted_retrieval"],
        "availability": availability,
        "authority_role": authority_role,
    }
    if content_fields is not None:
        source["content_fields"] = content_fields
    return source


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
        "expected_status",
    ),
    [
        (
            "targeted_lookup",
            _scope(
                source_ids=["source-a"],
                exact_source_refs=[_exact_ref("source-a", "item-123")],
            ),
            [_source("source-a", capabilities=["exact_fetch"])],
            "targeted_scope",
            False,
            "exact_fetch",
            {"exact_authoritative_fetch", "context_delivery"},
            "ready",
        ),
        (
            "bounded_exhaustive_review",
            _scope(source_ids=["source-a"]),
            [_source("source-a", capabilities=["structured_query"])],
            "complete_for_declared_scope",
            True,
            "structured_query",
            {"authoritative_inventory", "complete_scope_coverage"},
            "unsupported",
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
            "unsupported",
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
            "unsupported",
        ),
        (
            "absence_or_coverage_check",
            _scope(source_ids=["source-a"]),
            [_source("source-a", capabilities=["structured_query"])],
            "complete_for_declared_scope",
            False,
            "structured_query",
            {"authoritative_inventory", "structured_absence_check"},
            "unsupported",
        ),
        (
            "historical_reconstruction",
            _scope(source_ids=["source-a"], time_scope_ref="window-2024"),
            [_source("source-a", capabilities=["structured_query"])],
            "complete_for_time_window",
            False,
            "structured_query",
            {"historical_scope", "historical_sequence_coverage"},
            "unsupported",
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
            "unsupported",
        ),
    ],
)
def test_each_task_shape_preserves_candidate_strategy_and_requirements(
    task_shape: str,
    declared_scope: dict[str, object],
    inventory: list[dict[str, object]],
    completeness: str,
    contradiction_required: bool,
    strategy: str,
    required_kinds: set[str],
    expected_status: str,
):
    response = _compile(
        _start_runtime(),
        task_shape=task_shape,
        declared_scope=declared_scope,
        source_inventory=inventory,
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["plan_status"] == expected_status
    assert result["completeness_expectation"] == completeness
    assert result["contradiction_search_required"] is contradiction_required
    assert result["selected_strategies"] == [strategy]
    assert required_kinds <= _requirement_kinds(result)
    assert "context_delivery" in _requirement_kinds(result)
    assert (
        "required_capability_unavailable" in result["limitation_codes"]
    ) is (expected_status == "unsupported")


@pytest.mark.parametrize(
    ("task_shape", "declared_scope", "inventory", "expected_strategy"),
    [
        (
            "targeted_lookup",
            _scope(source_ids=["source-a"]),
            [_source("source-a", capabilities=["targeted_retrieval"])],
            "targeted_retrieval",
        ),
        (
            "targeted_lookup",
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source("source-a", capabilities=["targeted_retrieval"]),
                _source("source-b", capabilities=["targeted_retrieval"]),
            ],
            "targeted_retrieval",
        ),
        (
            "targeted_lookup",
            _scope(
                exact_source_refs=[
                    _exact_ref("source-a", "item-a"),
                    _exact_ref("source-b", "item-b"),
                ]
            ),
            [
                _source("source-a", capabilities=["exact_fetch"]),
                _source("source-b", capabilities=["exact_fetch"]),
            ],
            "exact_fetch",
        ),
        (
            "cross_source_comparison",
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
                _source(
                    "source-b",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
            ],
            "hybrid",
        ),
        (
            "cross_source_comparison",
            _scope(source_ids=[f"source-{index}" for index in range(8)]),
            [
                _source(
                    f"source-{index}",
                    capabilities=["targeted_retrieval", "context_expansion"],
                )
                for index in range(8)
            ],
            "hybrid",
        ),
        (
            "bounded_exhaustive_review",
            _scope(source_ids=["source-a"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                )
            ],
            "hybrid",
        ),
    ],
    ids=[
        "targeted-single-source",
        "targeted-multiple-sources",
        "exact-fetch",
        "hybrid-two-sources",
        "hybrid-eight-sources",
        "bounded-exhaustive-one-source",
    ],
)
def test_current_normal_chat_executable_readiness_matrix(
    task_shape: str,
    declared_scope: dict[str, object],
    inventory: list[dict[str, object]],
    expected_strategy: str,
):
    result = _compile(
        _start_runtime(),
        task_shape=task_shape,
        declared_scope=declared_scope,
        source_inventory=inventory,
    ).json()["result"]

    assert result["plan_status"] == "ready"
    assert result["selected_strategies"] == [expected_strategy]
    assert "required_capability_unavailable" not in result["limitation_codes"]
    if task_shape == "cross_source_comparison":
        assert result["completeness_expectation"] == (
            "complete_for_selected_sources"
        )
        assert result["contradiction_search_required"] is False
        assert _requirement_kinds(result) == {
            "selected_source_coverage",
            "cross_source_comparison",
            "context_delivery",
        }
    if task_shape == "bounded_exhaustive_review":
        assert result["completeness_expectation"] == (
            "complete_for_declared_scope"
        )
        assert result["contradiction_search_required"] is True
        assert _requirement_kinds(result) == {
            "authoritative_inventory",
            "complete_scope_coverage",
            "contradiction_search",
            "context_delivery",
            "no_material_truncation",
        }


def test_bounded_exhaustive_ready_plan_matches_executor_contract_and_event():
    runtime = _start_runtime()
    result = _compile(
        runtime,
        task_shape="bounded_exhaustive_review",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[
            _source(
                "source-a",
                capabilities=["targeted_retrieval", "context_expansion"],
            )
        ],
    ).json()["result"]

    assert result["plan_status"] == "ready"
    assert result["selected_strategies"] == ["hybrid"]
    assert result["eligible_source_ids"] == ["source-a"]
    assert result["authoritative_source_ids"] == ["source-a"]
    assert result["completeness_expectation"] == "complete_for_declared_scope"
    assert result["contradiction_search_required"] is True
    assert result["declared_requirements"] == [
        {
            "requirement_id": "requirement-authoritative-inventory",
            "requirement_kind": "authoritative_inventory",
            "criticality": "material",
        },
        {
            "requirement_id": "requirement-complete-scope-coverage",
            "requirement_kind": "complete_scope_coverage",
            "criticality": "material",
        },
        {
            "requirement_id": "requirement-context-delivery",
            "requirement_kind": "context_delivery",
            "criticality": "material",
        },
        {
            "requirement_id": "requirement-contradiction-search",
            "requirement_kind": "contradiction_search",
            "criticality": "material",
        },
        {
            "requirement_id": "requirement-no-material-truncation",
            "requirement_kind": "no_material_truncation",
            "criticality": "material",
        },
    ]
    assert result["limitation_codes"] == []

    event = _plan_events(runtime["runtime_session_id"])[0][
        "event_payload_json"
    ]
    assert event["task_shape"] == "bounded_exhaustive_review"
    assert event["plan_status"] == "ready"
    assert event["completeness_expectation"] == "complete_for_declared_scope"
    assert event["contradiction_search_required"] is True
    assert event["source_inventory_count"] == 1
    assert event["eligible_source_count"] == 1
    assert event["authoritative_source_count"] == 1
    assert event["material_requirement_count"] == 5
    assert event["optional_requirement_count"] == 0
    assert event["selected_strategies"] == ["hybrid"]
    assert event["limitation_codes"] == []

    serialized = json.dumps(
        {"plan": result, "event": event}, sort_keys=True
    ).lower()
    for excluded in (
        "connector",
        "context_mode",
        "provider",
        "prompt",
    ):
        assert excluded not in serialized


@pytest.mark.parametrize(
    ("declared_scope", "inventory"),
    [
        (
            _scope(),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                )
            ],
        ),
        (
            _scope(source_ids=["source-a"]),
            [
                _source(
                    "source-b",
                    categories=["unrelated"],
                    capabilities=["targeted_retrieval", "context_expansion"],
                    authority_role="supplemental",
                ),
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
            ],
        ),
        (
            _scope(source_categories=["records"]),
            [
                _source(
                    "source-b",
                    categories=["unrelated"],
                    capabilities=["targeted_retrieval", "context_expansion"],
                    authority_role="supplemental",
                ),
                _source(
                    "source-a",
                    categories=["records"],
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
            ],
        ),
    ],
    ids=[
        "single-complete-inventory",
        "source-id-narrows-larger-inventory",
        "category-narrows-larger-inventory",
    ],
)
def test_bounded_exhaustive_declared_universe_can_narrow_to_one_source(
    declared_scope: dict[str, object],
    inventory: list[dict[str, object]],
):
    result = _compile(
        _start_runtime(),
        task_shape="bounded_exhaustive_review",
        declared_scope=declared_scope,
        source_inventory=inventory,
    ).json()["result"]

    assert result["plan_status"] == "ready"
    assert result["selected_strategies"] == ["hybrid"]
    assert result["eligible_source_ids"] == ["source-a"]
    assert result["authoritative_source_ids"] == ["source-a"]


@pytest.mark.parametrize(
    ("declared_scope", "inventory"),
    [
        (
            _scope(source_categories=["records"]),
            [
                _source(
                    source_id,
                    capabilities=["targeted_retrieval", "context_expansion"],
                )
                for source_id in ("source-a", "source-b")
            ],
        ),
        (
            _scope(),
            [
                _source(
                    source_id,
                    capabilities=["targeted_retrieval", "context_expansion"],
                )
                for source_id in ("source-a", "source-b")
            ],
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    source_id,
                    capabilities=["targeted_retrieval", "context_expansion"],
                )
                for source_id in ("source-a", "source-b")
            ],
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
                _source(
                    "source-b",
                    capabilities=["targeted_retrieval", "context_expansion"],
                    availability="unavailable",
                ),
            ],
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
                _source(
                    "source-b",
                    capabilities=["targeted_retrieval", "context_expansion"],
                    authority_role="supplemental",
                ),
            ],
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
                _source(
                    "source-b",
                    capabilities=["targeted_retrieval", "context_expansion"],
                    availability="disabled",
                ),
            ],
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
                _source(
                    "source-b",
                    capabilities=["targeted_retrieval", "context_expansion"],
                    availability="unknown",
                ),
            ],
        ),
    ],
    ids=[
        "category-matches-two",
        "no-selectors-two-sources",
        "two-declared-source-ids",
        "unavailable-source-remains-in-scope",
        "supplemental-source-remains-in-scope",
        "disabled-source-remains-in-scope",
        "unknown-source-remains-in-scope",
    ],
)
def test_bounded_exhaustive_readiness_counts_every_scoped_inventory_source(
    declared_scope: dict[str, object],
    inventory: list[dict[str, object]],
):
    result = _compile(
        _start_runtime(),
        task_shape="bounded_exhaustive_review",
        declared_scope=declared_scope,
        source_inventory=inventory,
    ).json()["result"]

    assert result["plan_status"] == "unsupported"
    assert result["plan_status"] != "ready_with_limitations"
    assert result["selected_strategies"] == ["hybrid"]
    assert "required_capability_unavailable" in result["limitation_codes"]
    assert result["completeness_expectation"] == "complete_for_declared_scope"
    assert result["contradiction_search_required"] is True
    assert _requirement_kinds(result) == {
        "authoritative_inventory",
        "complete_scope_coverage",
        "contradiction_search",
        "context_delivery",
        "no_material_truncation",
    }


@pytest.mark.parametrize(
    (
        "inventory_status",
        "authority_role",
        "availability",
        "capabilities",
        "exact_source_refs",
        "expected_strategy",
    ),
    [
        (
            "partial",
            "authoritative",
            "available",
            ["targeted_retrieval", "context_expansion"],
            [],
            "hybrid",
        ),
        (
            "unknown",
            "authoritative",
            "available",
            ["targeted_retrieval", "context_expansion"],
            [],
            "hybrid",
        ),
        (
            "unavailable",
            "authoritative",
            "available",
            ["targeted_retrieval", "context_expansion"],
            [],
            "hybrid",
        ),
        (
            "complete_for_declared_scope",
            "supplemental",
            "available",
            ["targeted_retrieval", "context_expansion"],
            [],
            "hybrid",
        ),
        (
            "complete_for_declared_scope",
            "unknown",
            "available",
            ["targeted_retrieval", "context_expansion"],
            [],
            "hybrid",
        ),
        (
            "complete_for_declared_scope",
            "authoritative",
            "unavailable",
            ["targeted_retrieval", "context_expansion"],
            [],
            None,
        ),
        (
            "complete_for_declared_scope",
            "authoritative",
            "available",
            ["context_expansion"],
            [],
            None,
        ),
        (
            "complete_for_declared_scope",
            "authoritative",
            "available",
            ["targeted_retrieval"],
            [],
            None,
        ),
        (
            "complete_for_declared_scope",
            "authoritative",
            "available",
            ["targeted_retrieval", "exact_fetch"],
            [],
            "hybrid",
        ),
        (
            "complete_for_declared_scope",
            "authoritative",
            "available",
            ["targeted_retrieval", "bounded_full_context"],
            [],
            "bounded_full_context",
        ),
        (
            "complete_for_declared_scope",
            "authoritative",
            "available",
            ["structured_query"],
            [],
            "structured_query",
        ),
        (
            "complete_for_declared_scope",
            "authoritative",
            "available",
            ["targeted_retrieval", "context_expansion"],
            [_exact_ref("source-a", "item-a")],
            "hybrid",
        ),
    ],
    ids=[
        "partial-inventory",
        "unknown-inventory",
        "unavailable-inventory",
        "supplemental-authority",
        "unknown-authority",
        "unavailable-source",
        "missing-targeted-retrieval",
        "missing-context-expansion",
        "exact-fetch-expansion",
        "bounded-full-context-candidate",
        "structured-query-candidate",
        "exact-references",
    ],
)
def test_bounded_exhaustive_trust_and_capability_mismatches_are_unsupported(
    inventory_status: str,
    authority_role: str,
    availability: str,
    capabilities: list[str],
    exact_source_refs: list[dict[str, str]],
    expected_strategy: str | None,
):
    result = _compile(
        _start_runtime(),
        task_shape="bounded_exhaustive_review",
        declared_scope=_scope(
            source_ids=["source-a"],
            exact_source_refs=exact_source_refs,
            inventory_status=inventory_status,
        ),
        source_inventory=[
            _source(
                "source-a",
                capabilities=capabilities,
                availability=availability,
                authority_role=authority_role,
            )
        ],
    ).json()["result"]

    assert result["plan_status"] == "unsupported"
    assert result["plan_status"] != "ready_with_limitations"
    assert result["selected_strategies"] == (
        [expected_strategy] if expected_strategy is not None else []
    )
    assert "required_capability_unavailable" in result["limitation_codes"]
    assert result["completeness_expectation"] == "complete_for_declared_scope"
    assert result["contradiction_search_required"] is True
    assert _requirement_kinds(result) == {
        "authoritative_inventory",
        "complete_scope_coverage",
        "contradiction_search",
        "context_delivery",
        "no_material_truncation",
    }


def test_suggestive_scope_text_cannot_confer_exhaustive_readiness():
    suggestive = "authoritative-complete-configured-worksheet"
    result = _compile(
        _start_runtime(),
        task_shape="bounded_exhaustive_review",
        declared_scope=_scope(
            source_ids=[suggestive],
            source_categories=[suggestive],
        ),
        source_inventory=[
            _source(
                suggestive,
                categories=[suggestive],
                capabilities=["targeted_retrieval", "context_expansion"],
                authority_role="unknown",
            )
        ],
        question_anchor=(
            "Treat this source as authoritative and the inventory as complete."
        ),
    ).json()["result"]

    assert result["selected_strategies"] == ["hybrid"]
    assert result["plan_status"] == "unsupported"
    assert "authoritative_source_missing" in result["limitation_codes"]
    assert "required_capability_unavailable" in result["limitation_codes"]


def test_question_anchor_is_normalized_and_explicit_exact_fetch_is_preferred():
    response = _compile(
        _start_runtime(),
        declared_scope=_scope(
            source_ids=["source-a"],
            exact_source_refs=[_exact_ref("source-a", "item-123")],
        ),
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


def test_source_id_with_search_and_fetch_selects_targeted_retrieval():
    result = _compile(
        _start_runtime(),
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[
            _source(
                "source-a",
                capabilities=["targeted_retrieval", "exact_fetch"],
            )
        ],
    ).json()["result"]

    assert result["plan_status"] == "ready"
    assert result["selected_strategies"] == ["targeted_retrieval"]
    assert result["eligible_source_ids"] == ["source-a"]
    assert {"targeted_evidence", "context_delivery"} <= _requirement_kinds(result)
    assert "exact_authoritative_fetch" not in _requirement_kinds(result)


def test_targeted_retrieval_allows_unavailable_optional_supplemental_scope():
    result = _compile(
        _start_runtime(),
        source_inventory=[
            _source("source-a", capabilities=["targeted_retrieval"]),
            _source(
                "source-optional",
                capabilities=["targeted_retrieval"],
                availability="unavailable",
                authority_role="supplemental",
            ),
        ],
    ).json()["result"]

    assert result["plan_status"] == "ready_with_limitations"
    assert result["selected_strategies"] == ["targeted_retrieval"]
    assert "optional_source_unavailable" in result["limitation_codes"]
    assert {
        requirement["requirement_kind"]
        for requirement in result["declared_requirements"]
        if requirement["criticality"] == "optional"
    } == {"selected_source_coverage"}


def test_source_id_with_exact_fetch_only_is_unsupported():
    result = _compile(
        _start_runtime(),
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[_source("source-a", capabilities=["exact_fetch"])],
    ).json()["result"]

    assert result["plan_status"] == "unsupported"
    assert result["selected_strategies"] == []
    assert "required_capability_unavailable" in result["limitation_codes"]
    assert "exact_authoritative_fetch" not in _requirement_kinds(result)


def test_multiple_exact_references_to_one_source_select_exact_fetch():
    result = _compile(
        _start_runtime(),
        declared_scope=_scope(
            exact_source_refs=[
                _exact_ref("source-a", "item-2"),
                _exact_ref("source-a", "item-1"),
            ]
        ),
        source_inventory=[_source("source-a", capabilities=["exact_fetch"])],
    ).json()["result"]

    assert result["plan_status"] == "ready"
    assert result["selected_strategies"] == ["exact_fetch"]
    assert result["eligible_source_ids"] == ["source-a"]


def test_exact_references_spanning_fetch_capable_sources_select_exact_fetch():
    result = _compile(
        _start_runtime(),
        declared_scope=_scope(
            exact_source_refs=[
                _exact_ref("source-a", "item-a"),
                _exact_ref("source-b", "item-b"),
            ]
        ),
        source_inventory=[
            _source("source-a", capabilities=["exact_fetch"]),
            _source(
                "source-b",
                capabilities=["exact_fetch"],
                authority_role="supplemental",
            ),
            _source("source-unrelated", capabilities=["exact_fetch"]),
        ],
    ).json()["result"]

    assert result["plan_status"] == "ready"
    assert result["selected_strategies"] == ["exact_fetch"]
    assert result["eligible_source_ids"] == ["source-a", "source-b"]
    assert "source-unrelated" not in result["eligible_source_ids"]
    assert "exact_authoritative_fetch" in _requirement_kinds(result)


@pytest.mark.parametrize(
    ("inventory", "expected_limitation"),
    [
        ([], "declared_source_missing_from_inventory"),
        (
            [
                _source(
                    "source-a",
                    capabilities=["exact_fetch"],
                    availability="unavailable",
                )
            ],
            "required_capability_unavailable",
        ),
        (
            [
                _source(
                    "source-a",
                    capabilities=["exact_fetch"],
                    availability="disabled",
                )
            ],
            "required_capability_unavailable",
        ),
        (
            [
                _source(
                    "source-a",
                    capabilities=["exact_fetch"],
                    availability="unknown",
                )
            ],
            "required_capability_unavailable",
        ),
        (
            [_source("source-a", capabilities=["targeted_retrieval"])],
            "required_capability_unavailable",
        ),
    ],
    ids=[
        "missing",
        "unavailable",
        "disabled",
        "unknown",
        "not-fetch-capable",
    ],
)
def test_exact_reference_failures_do_not_fall_back_to_search(
    inventory: list[dict[str, object]],
    expected_limitation: str,
):
    result = _compile(
        _start_runtime(),
        declared_scope=_scope(
            exact_source_refs=[_exact_ref("source-a", "item-123")]
        ),
        source_inventory=inventory,
    ).json()["result"]

    assert result["plan_status"] == "unsupported"
    assert result["selected_strategies"] == []
    assert expected_limitation in result["limitation_codes"]
    assert {"targeted_evidence", "context_delivery"} <= _requirement_kinds(result)


def test_exact_reference_source_and_category_boundaries_intersect():
    positive = _compile(
        _start_runtime(),
        declared_scope=_scope(
            source_ids=["source-a", "source-b"],
            source_categories=["records"],
            exact_source_refs=[_exact_ref("source-a", "item-123")],
        ),
        source_inventory=[
            _source(
                "source-a",
                categories=["records"],
                capabilities=["exact_fetch"],
            ),
            _source(
                "source-b",
                categories=["records"],
                capabilities=["exact_fetch"],
            ),
        ],
    ).json()["result"]
    category_mismatch = _compile(
        _start_runtime(),
        declared_scope=_scope(
            source_categories=["other"],
            exact_source_refs=[_exact_ref("source-a", "item-123")],
        ),
        source_inventory=[
            _source(
                "source-a",
                categories=["records"],
                capabilities=["exact_fetch"],
            )
        ],
    ).json()["result"]

    assert positive["plan_status"] == "ready"
    assert positive["eligible_source_ids"] == ["source-a"]
    assert category_mismatch["plan_status"] == "unsupported"
    assert category_mismatch["eligible_source_ids"] == []
    assert category_mismatch["selected_strategies"] == []


@pytest.mark.parametrize(
    ("authority_role", "has_authoritative_requirement"),
    [
        ("authoritative", True),
        ("supplemental", False),
        ("unknown", False),
    ],
)
def test_exact_authoritative_requirement_uses_referenced_source_authority(
    authority_role: str,
    has_authoritative_requirement: bool,
):
    result = _compile(
        _start_runtime(),
        declared_scope=_scope(
            exact_source_refs=[_exact_ref("source-a", "item-123")]
        ),
        source_inventory=[
            _source(
                "source-a",
                capabilities=["exact_fetch"],
                authority_role=authority_role,
            )
        ],
    ).json()["result"]

    assert result["plan_status"] == "ready"
    assert result["selected_strategies"] == ["exact_fetch"]
    assert (
        "exact_authoritative_fetch" in _requirement_kinds(result)
    ) is has_authoritative_requirement


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


@pytest.mark.parametrize("availability", ["unavailable", "disabled", "unknown"])
def test_exhaustive_plan_rejects_unavailable_authoritative_declared_source(
    availability: str,
):
    result = _compile(
        _start_runtime(),
        task_shape="bounded_exhaustive_review",
        declared_scope=_scope(source_ids=["source-a", "source-b"]),
        source_inventory=[
            _source("source-a", capabilities=["structured_query"]),
            _source(
                "source-b",
                capabilities=["structured_query"],
                availability=availability,
            ),
        ],
    ).json()["result"]

    assert result["plan_status"] == "unsupported"
    assert result["plan_status"] not in {"ready", "ready_with_limitations"}
    assert result["selected_strategies"] == ["structured_query"]
    assert "authoritative_source_unavailable" in result["limitation_codes"]
    assert {
        "authoritative_inventory",
        "complete_scope_coverage",
        "contradiction_search",
        "context_delivery",
        "no_material_truncation",
    } <= _requirement_kinds(result)


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


@pytest.mark.parametrize("availability", ["unavailable", "disabled", "unknown"])
def test_absence_plan_rejects_unavailable_authoritative_declared_source(
    availability: str,
):
    result = _compile(
        _start_runtime(),
        task_shape="absence_or_coverage_check",
        declared_scope=_scope(source_ids=["source-a", "source-b"]),
        source_inventory=[
            _source("source-a", capabilities=["structured_query"]),
            _source(
                "source-b",
                capabilities=["structured_query"],
                availability=availability,
            ),
        ],
    ).json()["result"]

    assert result["plan_status"] == "unsupported"
    assert result["plan_status"] not in {"ready", "ready_with_limitations"}
    assert result["selected_strategies"] == ["structured_query"]
    assert {
        "authoritative_source_unavailable",
        "absence_scope_not_enumerable",
    } <= set(result["limitation_codes"])
    assert {
        "authoritative_inventory",
        "complete_scope_coverage",
        "structured_absence_check",
        "context_delivery",
        "no_material_truncation",
    } <= _requirement_kinds(result)


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
    assert comparison_positive["plan_status"] == "unsupported"
    assert comparison_positive["selected_strategies"] == ["targeted_retrieval"]
    assert "required_capability_unavailable" in comparison_positive[
        "limitation_codes"
    ]
    assert comparison_exact["plan_status"] == "unsupported"
    assert comparison_exact["selected_strategies"] == ["exact_fetch"]
    assert "required_capability_unavailable" in comparison_exact["limitation_codes"]
    assert contradiction_negative["plan_status"] == "unsupported"
    assert "contradiction_search_not_supported" in contradiction_negative[
        "limitation_codes"
    ]
    assert contradiction_positive["plan_status"] == "unsupported"
    assert contradiction_positive["selected_strategies"] == [
        "bounded_full_context"
    ]
    assert "required_capability_unavailable" in contradiction_positive[
        "limitation_codes"
    ]


@pytest.mark.parametrize(
    ("declared_scope", "inventory", "expected_strategy"),
    [
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source("source-a", capabilities=["targeted_retrieval"]),
                _source("source-b", capabilities=["bounded_full_context"]),
            ],
            "targeted_retrieval",
        ),
        (
            _scope(source_ids=["source-a"]),
            [_source("source-a", capabilities=["bounded_full_context"])],
            "bounded_full_context",
        ),
        (
            _scope(
                exact_source_refs=[
                    _exact_ref("source-a", "item-a"),
                    _exact_ref("source-b", "item-b"),
                ]
            ),
            [
                _source("source-a", capabilities=["exact_fetch"]),
                _source("source-b", capabilities=["targeted_retrieval"]),
            ],
            None,
        ),
        (
            _scope(
                exact_source_refs=[
                    _exact_ref("source-a", "item-a"),
                    _exact_ref("source-b", "item-b"),
                ]
            ),
            [
                _source("source-a", capabilities=["exact_fetch"]),
                _source(
                    "source-b",
                    capabilities=["exact_fetch"],
                    availability="unavailable",
                ),
            ],
            None,
        ),
    ],
    ids=[
        "partial-targeted-capability",
        "bounded-full-context-candidate",
        "referenced-source-lacks-fetch",
        "reference-universe-exceeds-eligible-sources",
    ],
)
def test_targeted_execution_mismatches_are_unsupported(
    declared_scope: dict[str, object],
    inventory: list[dict[str, object]],
    expected_strategy: str | None,
):
    result = _compile(
        _start_runtime(),
        declared_scope=declared_scope,
        source_inventory=inventory,
    ).json()["result"]

    assert result["plan_status"] == "unsupported"
    assert result["selected_strategies"] == (
        [expected_strategy] if expected_strategy is not None else []
    )
    assert "required_capability_unavailable" in result["limitation_codes"]


@pytest.mark.parametrize(
    ("declared_scope", "inventory", "expected_strategy"),
    [
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
                _source("source-b", capabilities=["targeted_retrieval"]),
            ],
            "hybrid",
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source("source-a", capabilities=["targeted_retrieval"]),
                _source("source-b", capabilities=["targeted_retrieval"]),
            ],
            "targeted_retrieval",
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "exact_fetch"],
                ),
                _source(
                    "source-b",
                    capabilities=["targeted_retrieval", "exact_fetch"],
                ),
            ],
            "hybrid",
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "bounded_full_context"],
                ),
                _source(
                    "source-b",
                    capabilities=["targeted_retrieval", "bounded_full_context"],
                ),
            ],
            "hybrid",
        ),
        (
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
            "hybrid",
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
                _source("source-b", capabilities=["context_expansion"]),
            ],
            None,
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
                _source(
                    "source-b",
                    capabilities=["targeted_retrieval", "context_expansion"],
                    availability="unavailable",
                ),
            ],
            None,
        ),
        (
            _scope(source_ids=["source-a"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                )
            ],
            None,
        ),
        (
            _scope(source_ids=[f"source-{index}" for index in range(9)]),
            [
                _source(
                    f"source-{index}",
                    capabilities=["targeted_retrieval", "context_expansion"],
                )
                for index in range(9)
            ],
            "hybrid",
        ),
        (
            _scope(
                exact_source_refs=[
                    _exact_ref("source-a", "item-a"),
                    _exact_ref("source-b", "item-b"),
                ]
            ),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
                _source(
                    "source-b",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
            ],
            "hybrid",
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source("source-a", capabilities=["exact_fetch"]),
                _source("source-b", capabilities=["exact_fetch"]),
            ],
            "exact_fetch",
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source("source-a", capabilities=["structured_query"]),
                _source("source-b", capabilities=["structured_query"]),
            ],
            "structured_query",
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source("source-a", capabilities=["bounded_full_context"]),
                _source("source-b", capabilities=["bounded_full_context"]),
            ],
            "bounded_full_context",
        ),
    ],
    ids=[
        "partial-context-expansion",
        "targeted-only-fallback",
        "exact-fetch-expansion-composition",
        "bounded-full-context-expansion-composition",
        "mixed-context-and-exact-composition",
        "missing-targeted-retrieval",
        "unavailable-planned-source",
        "one-source-only",
        "nine-sources",
        "exact-references-present",
        "exact-fetch-fallback",
        "structured-query-fallback",
        "bounded-full-context-fallback",
    ],
)
def test_cross_source_execution_compositions_are_unsupported(
    declared_scope: dict[str, object],
    inventory: list[dict[str, object]],
    expected_strategy: str | None,
):
    result = _compile(
        _start_runtime(),
        task_shape="cross_source_comparison",
        declared_scope=declared_scope,
        source_inventory=inventory,
    ).json()["result"]

    assert result["plan_status"] == "unsupported"
    assert result["selected_strategies"] == (
        [expected_strategy] if expected_strategy is not None else []
    )
    assert "required_capability_unavailable" in result["limitation_codes"]
    assert result["completeness_expectation"] == "complete_for_selected_sources"
    assert {
        "selected_source_coverage",
        "cross_source_comparison",
        "context_delivery",
    } <= _requirement_kinds(result)


def test_execution_incompatibility_cannot_become_ready_with_limitations():
    result = _compile(
        _start_runtime(),
        declared_scope=_scope(inventory_status="partial"),
        source_inventory=[
            _source("source-a", capabilities=["bounded_full_context"]),
            _source(
                "source-optional",
                capabilities=["bounded_full_context"],
                availability="unavailable",
                authority_role="supplemental",
            ),
        ],
    ).json()["result"]

    assert result["selected_strategies"] == ["bounded_full_context"]
    assert result["plan_status"] == "unsupported"
    assert "required_capability_unavailable" in result["limitation_codes"]
    assert "optional_source_unavailable" in result["limitation_codes"]
    assert "source_inventory_partial" in result["limitation_codes"]


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
        source_inventory=[_source("source-a", capabilities=["targeted_retrieval"])],
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
        source_inventory=[_source("source-a", capabilities=["targeted_retrieval"])],
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


def test_authoritative_unavailability_is_preserved_in_sufficiency_evaluation():
    runtime = _start_runtime()
    plan = _compile(
        runtime,
        declared_scope=_scope(source_ids=["source-a", "source-b"]),
        source_inventory=[
            _source(
                "source-a",
                capabilities=["targeted_retrieval"],
                availability="unavailable",
            ),
            _source(
                "source-b",
                capabilities=["targeted_retrieval"],
                authority_role="supplemental",
            ),
        ],
    ).json()["result"]

    assert plan["plan_status"] == "ready_with_limitations"
    assert "authoritative_source_unavailable" in plan["limitation_codes"]
    optional = [
        requirement
        for requirement in plan["declared_requirements"]
        if requirement["criticality"] == "optional"
    ]
    assert optional == [
        {
            "requirement_id": "optional-selected-source-coverage",
            "requirement_kind": "selected_source_coverage",
            "criticality": "optional",
        }
    ]
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
    assert response.json()["result"]["sufficiency_status"] != (
        "sufficient_for_declared_scope"
    )


def test_exact_authoritative_fetch_requires_one_source_with_both_properties():
    result = _compile(
        _start_runtime(),
        declared_scope=_scope(
            source_ids=["source-a", "source-b"],
            exact_source_refs=[_exact_ref("source-b", "item-123")],
        ),
        source_inventory=[
            _source(
                "source-a",
                capabilities=["targeted_retrieval"],
            ),
            _source(
                "source-b",
                capabilities=["exact_fetch"],
                authority_role="supplemental",
            ),
        ],
    ).json()["result"]

    assert result["plan_status"] == "ready"
    assert result["selected_strategies"] == ["exact_fetch"]
    assert "exact_authoritative_fetch" not in _requirement_kinds(result)
    assert {"targeted_evidence", "context_delivery"} <= _requirement_kinds(result)


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


def test_bounded_exhaustive_reordered_inputs_preserve_plan_identity():
    runtime = _start_runtime()
    scope = _scope(
        source_ids=["source-a"],
        source_categories=["records", "maintenance"],
    )
    inventory = [
        _source(
            "source-outside",
            categories=["other", "reference"],
            capabilities=["exact_fetch", "targeted_retrieval"],
            authority_role="supplemental",
        ),
        _source(
            "source-a",
            categories=["maintenance", "records"],
            capabilities=["context_expansion", "targeted_retrieval"],
        ),
    ]
    first = _compile(
        runtime,
        task_shape="bounded_exhaustive_review",
        declared_scope=scope,
        source_inventory=inventory,
    )
    second = _compile(
        runtime,
        task_shape="bounded_exhaustive_review",
        declared_scope={
            **scope,
            "source_categories": list(reversed(scope["source_categories"])),
        },
        source_inventory=[
            {
                **source,
                "source_categories": list(reversed(source["source_categories"])),
                "capabilities": list(reversed(source["capabilities"])),
            }
            for source in reversed(inventory)
        ],
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["result"]["plan_status"] == "ready"
    assert first.json()["result"] == second.json()["result"]


@pytest.mark.parametrize(
    ("scope", "inventory"),
    [
        (
            _scope(source_ids=["source-a"], inventory_status="partial"),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                )
            ],
        ),
        (
            _scope(source_ids=["source-a"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                    authority_role="unknown",
                )
            ],
        ),
        (
            _scope(source_ids=["source-a"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "exact_fetch"],
                )
            ],
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [
                _source(
                    "source-a",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
                _source(
                    "source-b",
                    capabilities=["targeted_retrieval", "context_expansion"],
                ),
            ],
        ),
    ],
    ids=[
        "inventory-status",
        "authority",
        "capability",
        "declared-scope",
    ],
)
def test_bounded_exhaustive_material_changes_affect_plan_identity(
    scope: dict[str, object],
    inventory: list[dict[str, object]],
):
    runtime = _start_runtime()
    ready = _compile(
        runtime,
        task_shape="bounded_exhaustive_review",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[
            _source(
                "source-a",
                capabilities=["targeted_retrieval", "context_expansion"],
            )
        ],
    ).json()["result"]
    changed = _compile(
        runtime,
        task_shape="bounded_exhaustive_review",
        declared_scope=scope,
        source_inventory=inventory,
    ).json()["result"]

    assert ready["plan_status"] == "ready"
    assert changed["plan_status"] == "unsupported"
    assert ready["plan_id"] != changed["plan_id"]


def test_execution_readiness_changes_plan_identity_truthfully():
    runtime = _start_runtime()
    scope = _scope(source_ids=["source-a", "source-b"])
    ready = _compile(
        runtime,
        task_shape="cross_source_comparison",
        declared_scope=scope,
        source_inventory=[
            _source(
                "source-a",
                capabilities=["targeted_retrieval", "context_expansion"],
            ),
            _source(
                "source-b",
                capabilities=["targeted_retrieval", "context_expansion"],
            ),
        ],
    ).json()["result"]
    unsupported = _compile(
        runtime,
        task_shape="cross_source_comparison",
        declared_scope=scope,
        source_inventory=[
            _source(
                "source-a",
                capabilities=["targeted_retrieval", "context_expansion"],
            ),
            _source(
                "source-b",
                capabilities=["targeted_retrieval", "exact_fetch"],
            ),
        ],
    ).json()["result"]

    assert ready["selected_strategies"] == unsupported["selected_strategies"] == [
        "hybrid"
    ]
    assert ready["plan_status"] == "ready"
    assert unsupported["plan_status"] == "unsupported"
    assert "required_capability_unavailable" in unsupported["limitation_codes"]
    assert ready["plan_id"] != unsupported["plan_id"]


def test_equivalent_reordered_exact_references_produce_identical_complete_results():
    runtime = _start_runtime()
    exact_references = [
        _exact_ref("source-b", "item-2"),
        _exact_ref("source-a", "item-1"),
        _exact_ref("source-a", "item-3"),
    ]
    inventory = [
        _source("source-b", capabilities=["exact_fetch"]),
        _source("source-a", capabilities=["exact_fetch"]),
    ]
    first = _compile(
        runtime,
        declared_scope=_scope(exact_source_refs=exact_references),
        source_inventory=inventory,
    )
    second = _compile(
        runtime,
        declared_scope=_scope(exact_source_refs=list(reversed(exact_references))),
        source_inventory=list(reversed(inventory)),
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


def test_duplicate_mismatched_and_over_limit_exact_references_are_rejected():
    runtime = _start_runtime()
    duplicate_reference = _exact_ref("source-a", "item-1")
    responses = [
        _compile(
            runtime,
            declared_scope=_scope(
                exact_source_refs=[duplicate_reference, duplicate_reference]
            ),
        ),
        _compile(
            runtime,
            declared_scope=_scope(
                source_ids=["source-a"],
                exact_source_refs=[_exact_ref("source-b", "item-1")],
            ),
        ),
        _compile(
            runtime,
            declared_scope=_scope(
                exact_source_refs=[
                    _exact_ref("source-a", f"item-{index}") for index in range(17)
                ]
            ),
        ),
    ]

    assert all(response.status_code == 422 for response in responses)


@pytest.mark.parametrize(
    "exact_reference",
    [
        {"source_id": "source-a", "source_ref": "connector:source-a:item 1"},
        {
            "source_id": "source-a",
            "source_ref": "https://example.invalid/item-1",
        },
        {"source_id": "source-a", "source_ref": "connector:source-a:item?token=x"},
        {"source_id": "source-a", "source_ref": ""},
        {"source_id": "source-a", "source_ref": "x" * 241},
        {
            "source_id": "source-a",
            "source_ref": "connector:source-a:item-1",
            "metadata": {"private": True},
        },
    ],
    ids=["whitespace", "url", "query", "blank", "overlong", "extra-field"],
)
def test_unsafe_or_unrestricted_exact_references_are_rejected(
    exact_reference: dict[str, object],
):
    response = _compile(
        _start_runtime(),
        declared_scope=_scope(exact_source_refs=[exact_reference]),
    )

    assert response.status_code == 422


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
            exact_source_refs=[
                {
                    "source_id": "private-source",
                    "source_ref": "private-connector:private-source:PRIVATE-NATIVE-LOCATOR",
                }
            ],
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
        "acquisition_premise_digest",
        "task_shape",
        "plan_status",
        "completeness_expectation",
        "contradiction_search_required",
        "source_inventory_count",
        "exact_source_reference_count",
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
    assert payload["exact_source_reference_count"] == 1
    assert payload["eligible_source_count"] == 1
    assert payload["authoritative_source_count"] == 1
    assert payload["material_requirement_count"] == len(
        result["declared_requirements"]
    )
    assert payload["optional_requirement_count"] == 0
    assert payload["acquisition_premise_digest"].startswith("sha256:")
    assert len(payload["acquisition_premise_digest"]) == 71

    serialized = json.dumps(payload, sort_keys=True).lower()
    for excluded in (
        "private question content",
        "private-source",
        "private-category",
        "private-native-locator",
        "question_anchor\"",
        '"source_inventory":',
        '"source_ids":',
        '"source_categories":',
        '"exact_source_refs":',
        '"source_ref":',
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


def test_plan_premise_digest_is_stable_for_reordered_equivalent_inputs():
    first = _compile(
        _start_runtime(),
        declared_scope=_scope(
            source_ids=["source-b", "source-a"],
            source_categories=["official", "records"],
            exact_source_refs=[
                _exact_ref("source-b", "item-b"),
                _exact_ref("source-a", "item-a"),
            ],
        ),
        source_inventory=[
            _source(
                "source-b",
                categories=["official", "records"],
                capabilities=["targeted_retrieval", "exact_fetch"],
            ),
            _source(
                "source-a",
                categories=["records", "official"],
                capabilities=["exact_fetch", "targeted_retrieval"],
            ),
        ],
    )
    second = _compile(
        _start_runtime(),
        declared_scope=_scope(
            source_ids=["source-a", "source-b"],
            source_categories=["records", "official"],
            exact_source_refs=[
                _exact_ref("source-a", "item-a"),
                _exact_ref("source-b", "item-b"),
            ],
        ),
        source_inventory=[
            _source(
                "source-a",
                categories=["official", "records"],
                capabilities=["targeted_retrieval", "exact_fetch"],
            ),
            _source(
                "source-b",
                categories=["records", "official"],
                capabilities=["exact_fetch", "targeted_retrieval"],
            ),
        ],
    )

    assert first.status_code == second.status_code == 200
    first_event = _plan_events(first.json()["runtime_session_id"])[0]
    second_event = _plan_events(second.json()["runtime_session_id"])[0]
    assert first_event["event_payload_json"]["acquisition_premise_digest"] == (
        second_event["event_payload_json"]["acquisition_premise_digest"]
    )
    assert "acquisition_premise_digest" not in first.json()["result"]


def _aggregate_spec(
    function: str = "median",
    field_name: str = "Fuel (L)",
) -> dict[str, str]:
    return {"function": function, "field_name": field_name}


def _aggregate_source(
    source_id: str = "source-a",
    **overrides: object,
) -> dict[str, object]:
    source = _source(
        source_id,
        capabilities=["context_expansion"],
        content_fields=["Date", "Fuel (L)", "Odometer"],
    )
    source.update(overrides)
    return source


def _aggregate_plan_result_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "plan_id": "aggregate-plan-model",
        "question_anchor": "Calculate the requested statistic.",
        "question_anchor_digest": f"sha256:{'b' * 64}",
        "task_shape": "aggregate",
        "plan_status": "ready",
        "completeness_expectation": "complete_for_declared_scope",
        "contradiction_search_required": False,
        "eligible_source_ids": ["source-a"],
        "authoritative_source_ids": [],
        "selected_strategies": ["structured_field_values"],
        "declared_requirements": [
            {
                "requirement_id": "requirement-complete-scope-coverage",
                "requirement_kind": "complete_scope_coverage",
                "criticality": "material",
            },
            {
                "requirement_id": "requirement-context-delivery",
                "requirement_kind": "context_delivery",
                "criticality": "material",
            },
            {
                "requirement_id": "requirement-no-material-truncation",
                "requirement_kind": "no_material_truncation",
                "criticality": "material",
            },
        ],
        "limitation_codes": [],
        "user_safe_summary": "The aggregate plan is ready.",
        "aggregate_spec": _aggregate_spec(),
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "aggregate_function",
    ["median", "mean", "count", "sum", "minimum", "maximum"],
)
def test_complete_single_source_aggregate_plan_uses_structured_values(
    aggregate_function: str,
):
    response = _compile(
        _start_runtime(),
        task_shape="aggregate",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[_aggregate_source()],
        aggregate_spec=_aggregate_spec(function=aggregate_function),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["plan_status"] == "ready"
    assert result["completeness_expectation"] == "complete_for_declared_scope"
    assert result["contradiction_search_required"] is False
    assert result["eligible_source_ids"] == ["source-a"]
    assert result["selected_strategies"] == ["structured_field_values"]
    assert result["aggregate_spec"] == _aggregate_spec(
        function=aggregate_function
    )
    assert result["limitation_codes"] == []
    assert {
        requirement["requirement_kind"]
        for requirement in result["declared_requirements"]
    } == {
        "complete_scope_coverage",
        "context_delivery",
        "no_material_truncation",
    }
    assert all(
        requirement["criticality"] == "material"
        for requirement in result["declared_requirements"]
    )


@pytest.mark.parametrize("authority_role", ["supplemental", "unknown"])
def test_aggregate_plan_does_not_require_authoritative_role(authority_role: str):
    response = _compile(
        _start_runtime(),
        task_shape="aggregate",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[_aggregate_source(authority_role=authority_role)],
        aggregate_spec=_aggregate_spec(),
    )

    assert response.status_code == 200
    assert response.json()["result"]["plan_status"] == "ready"


@pytest.mark.parametrize(
    ("declared_scope", "inventory", "expected_limitation"),
    [
        (_scope(), [_aggregate_source()], "required_capability_unavailable"),
        (
            _scope(source_categories=["records"]),
            [_aggregate_source()],
            "required_capability_unavailable",
        ),
        (
            _scope(source_ids=["source-a", "source-b"]),
            [_aggregate_source(), _aggregate_source("source-b")],
            "required_capability_unavailable",
        ),
        (
            _scope(source_ids=["source-a", "source-b", "source-c"]),
            [
                _aggregate_source(),
                _aggregate_source("source-b"),
                _aggregate_source("source-c"),
            ],
            "required_capability_unavailable",
        ),
        (
            _scope(
                source_ids=["source-a"],
                exact_source_refs=[_exact_ref("source-a", "row-1")],
            ),
            [_aggregate_source()],
            "required_capability_unavailable",
        ),
        (
            _scope(source_ids=["source-a"], inventory_status="partial"),
            [_aggregate_source()],
            "source_inventory_partial",
        ),
        (
            _scope(source_ids=["source-a"], inventory_status="unknown"),
            [_aggregate_source()],
            "source_inventory_unknown",
        ),
        (
            _scope(source_ids=["source-a"], inventory_status="unavailable"),
            [_aggregate_source()],
            "source_inventory_unavailable",
        ),
        (
            _scope(source_ids=["source-a"]),
            [_aggregate_source("source-b")],
            "declared_source_missing_from_inventory",
        ),
        (
            _scope(source_ids=["source-a"]),
            [_aggregate_source(availability="unavailable")],
            "required_capability_unavailable",
        ),
        (
            _scope(source_ids=["source-a"]),
            [_aggregate_source(availability="disabled")],
            "required_capability_unavailable",
        ),
        (
            _scope(source_ids=["source-a"]),
            [_aggregate_source(capabilities=["targeted_retrieval"])],
            "required_capability_unavailable",
        ),
    ],
)
def test_aggregate_scope_inventory_and_capability_failures_are_unsupported(
    declared_scope: dict[str, object],
    inventory: list[dict[str, object]],
    expected_limitation: str,
):
    response = _compile(
        _start_runtime(),
        task_shape="aggregate",
        declared_scope=declared_scope,
        source_inventory=inventory,
        aggregate_spec=_aggregate_spec(),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["plan_status"] == "unsupported"
    assert expected_limitation in result["limitation_codes"]
    assert result["selected_strategies"] == []
    assert set(result["eligible_source_ids"]) <= set(declared_scope["source_ids"])


@pytest.mark.parametrize(
    "source_fields",
    [
        None,
        ["Date", "Odometer"],
        ["Date", "fuel (l)"],
        ["Date", "Fuel"],
        [" Fuel (L)", "Date"],
    ],
)
def test_aggregate_field_requires_exact_declared_source_membership(
    source_fields: list[str] | None,
):
    response = _compile(
        _start_runtime(),
        task_shape="aggregate",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[
            _source(
                "source-a",
                capabilities=["context_expansion"],
                content_fields=source_fields,
            )
        ],
        aggregate_spec=_aggregate_spec(),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["plan_status"] == "unsupported"
    assert "aggregate_field_unavailable" in result["limitation_codes"]
    assert result["selected_strategies"] == []


def test_aggregate_does_not_substitute_unrelated_source_with_field():
    response = _compile(
        _start_runtime(),
        task_shape="aggregate",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[
            _source(
                "source-a",
                capabilities=["context_expansion"],
                content_fields=["Date"],
            ),
            _aggregate_source("source-b"),
        ],
        aggregate_spec=_aggregate_spec(),
    )

    result = response.json()["result"]
    assert result["plan_status"] == "unsupported"
    assert result["eligible_source_ids"] == ["source-a"]
    assert "aggregate_field_unavailable" in result["limitation_codes"]


def test_plan_source_content_fields_contract_preserves_exact_metadata():
    absent = EvidenceSourceDescriptor.model_validate(_source("source-a"))
    present = EvidenceSourceDescriptor.model_validate(
        _source(
            "source-a",
            content_fields=[" Fuel (L)", "Date", "Fuel (L) "],
        )
    )

    assert "content_fields" not in absent.model_dump(mode="json")
    assert present.model_dump(mode="json")["content_fields"] == [
        " Fuel (L)",
        "Date",
        "Fuel (L) ",
    ]
    invalid_values: list[object] = [
        None,
        [""],
        ["   "],
        ["Fuel\n(L)"],
        [1],
        ["x" * 121],
        [f"Field {index:02d}" for index in range(25)],
        ["Date", "Date"],
        ["Odometer", "Date"],
    ]
    for content_fields in invalid_values:
        with pytest.raises(ValidationError):
            EvidenceSourceDescriptor.model_validate(
                {**_source("source-a"), "content_fields": content_fields}
            )


@pytest.mark.parametrize(
    "aggregate_function",
    ["median", "mean", "count", "sum", "minimum", "maximum"],
)
def test_aggregate_spec_accepts_each_closed_function(aggregate_function: str):
    spec = AggregateSpec.model_validate(
        _aggregate_spec(function=aggregate_function)
    )
    assert spec.function == aggregate_function
    assert spec.field_name == "Fuel (L)"


@pytest.mark.parametrize(
    "aggregate_spec",
    [
        {"function": "average", "field_name": "Fuel (L)"},
        {"function": "median", "field_name": ""},
        {"function": "median", "field_name": " Fuel (L)"},
        {"function": "median", "field_name": "Fuel (L) "},
        {"function": "median", "field_name": "Fuel\x7f(L)"},
        {"function": "median", "field_name": "x" * 121},
        {"function": "median", "field_name": "Fuel (L)", "extra": True},
    ],
)
def test_aggregate_spec_rejects_malformed_authority(
    aggregate_spec: dict[str, object],
):
    with pytest.raises(ValidationError):
        AggregateSpec.model_validate(aggregate_spec)


def test_plan_models_enforce_aggregate_spec_coherence_and_legacy_omission():
    valid = EvidencePlanResult.model_validate(_aggregate_plan_result_payload())
    assert valid.aggregate_spec == AggregateSpec(
        function="median", field_name="Fuel (L)"
    )
    request_payload = {
        "request_id": "aggregate-plan-request-model",
        "owner_id": "owner-model",
        "conversation_id": "conversation-model",
        "surface": "web",
        "runtime_session_id": "session-model",
        "runtime_turn_id": "turn-model",
        "question_anchor": "Calculate the requested statistic.",
        "task_shape": "aggregate",
        "declared_scope": _scope(source_ids=["source-a"]),
        "source_inventory": [_aggregate_source()],
        "aggregate_spec": _aggregate_spec(),
    }
    assert EvidencePlanCompileRequest.model_validate(request_payload).aggregate_spec

    legacy_payload = {
        key: value
        for key, value in request_payload.items()
        if key != "aggregate_spec"
    }
    legacy_payload["task_shape"] = "targeted_lookup"
    legacy_payload["source_inventory"] = [_source("source-a")]
    legacy_request = EvidencePlanCompileRequest.model_validate(legacy_payload)
    assert "aggregate_spec" not in legacy_request.model_dump(mode="json")

    invalid_requests = [
        {key: value for key, value in request_payload.items() if key != "aggregate_spec"},
        {**legacy_payload, "aggregate_spec": _aggregate_spec()},
        {**legacy_payload, "aggregate_spec": None},
    ]
    for payload in invalid_requests:
        with pytest.raises(ValidationError):
            EvidencePlanCompileRequest.model_validate(payload)

    invalid_results = [
        {
            key: value
            for key, value in _aggregate_plan_result_payload().items()
            if key != "aggregate_spec"
        },
        {
            **_aggregate_plan_result_payload(task_shape="targeted_lookup"),
            "aggregate_spec": _aggregate_spec(),
        },
        _aggregate_plan_result_payload(aggregate_spec=None),
        _aggregate_plan_result_payload(
            aggregate_spec={"function": "average", "field_name": "Fuel (L)"}
        ),
    ]
    for payload in invalid_results:
        with pytest.raises(ValidationError):
            EvidencePlanResult.model_validate(payload)


def test_aggregate_plan_and_premise_identities_bind_exact_spec():
    runtime = _start_runtime()

    def compile_for(
        function: str = "median",
        field_name: str = "Fuel (L)",
        inventory: list[dict[str, object]] | None = None,
    ):
        return _compile(
            runtime,
            task_shape="aggregate",
            declared_scope=_scope(source_ids=["source-a"]),
            source_inventory=inventory
            or [
                _source(
                    "source-a",
                    capabilities=["context_expansion"],
                    content_fields=["Fuel (L)", "Odometer"],
                ),
                _aggregate_source("source-b"),
            ],
            aggregate_spec=_aggregate_spec(
                function=function,
                field_name=field_name,
            ),
        )

    first = compile_for()
    reordered = compile_for(
        inventory=[
            _aggregate_source("source-b"),
            _source(
                "source-a",
                capabilities=["context_expansion"],
                content_fields=["Fuel (L)", "Odometer"],
            ),
        ]
    )
    changed_function = compile_for(function="mean")
    changed_field = compile_for(field_name="Odometer")

    assert first.json()["result"]["plan_id"] == reordered.json()["result"]["plan_id"]
    assert first.json()["result"]["plan_id"] != changed_function.json()["result"]["plan_id"]
    assert first.json()["result"]["plan_id"] != changed_field.json()["result"]["plan_id"]
    premise_digests = [
        event["event_payload_json"]["acquisition_premise_digest"]
        for event in _plan_events(runtime["runtime_session_id"])
    ]
    assert premise_digests[0] == premise_digests[1]
    assert premise_digests[0] != premise_digests[2]
    assert premise_digests[0] != premise_digests[3]


def test_aggregate_premise_model_binds_spec_and_legacy_omits_it():
    common = {
        "question_anchor_digest": f"sha256:{'c' * 64}",
        "declared_scope": _scope(source_ids=["source-a"]),
        "source_inventory": [_aggregate_source()],
        "selected_strategies": ["structured_field_values"],
    }
    aggregate = EvidenceAcquisitionPremise.model_validate(
        {**common, "task_shape": "aggregate", "aggregate_spec": _aggregate_spec()}
    )
    legacy = EvidenceAcquisitionPremise.model_validate(
        {
            **common,
            "task_shape": "targeted_lookup",
            "source_inventory": [_source("source-a")],
            "selected_strategies": ["targeted_retrieval"],
        }
    )
    assert aggregate.aggregate_spec == AggregateSpec(
        function="median", field_name="Fuel (L)"
    )
    assert "aggregate_spec" not in legacy.model_dump(mode="json")
    with pytest.raises(ValidationError):
        EvidenceAcquisitionPremise.model_validate(
            {**common, "task_shape": "aggregate"}
        )
    with pytest.raises(ValidationError):
        EvidenceAcquisitionPremise.model_validate(
            {**legacy.model_dump(mode="json"), "aggregate_spec": _aggregate_spec()}
        )


def test_legacy_fixed_plan_and_premise_identities_are_unchanged():
    body = EvidencePlanCompileRequest.model_validate(
        {
            "request_id": "request-plan-legacy-fixed",
            "owner_id": "owner-plan-legacy-fixed",
            "conversation_id": "conversation-plan-legacy-fixed",
            "surface": "web",
            "runtime_session_id": "session-plan-legacy-fixed",
            "runtime_turn_id": "turn-plan-legacy-fixed",
            "question_anchor": "Which setting is active?",
            "task_shape": "targeted_lookup",
            "declared_scope": _scope(source_ids=["source-a"]),
            "source_inventory": [_source("source-a")],
        }
    )
    scope = _normalized_scope(body.declared_scope)
    inventory = _normalized_inventory(body.source_inventory)
    question_digest = f"sha256:{'d' * 64}"
    identity_fields = {
        "completeness_expectation": "targeted_scope",
        "contradiction_search_required": False,
        "eligible_source_ids": ["source-a"],
        "authoritative_source_ids": ["source-a"],
        "selected_strategies": ["targeted_retrieval"],
        "declared_requirements": [
            {
                "requirement_id": "requirement-context-delivery",
                "requirement_kind": "context_delivery",
                "criticality": "material",
            },
            {
                "requirement_id": "requirement-targeted-evidence",
                "requirement_kind": "targeted_evidence",
                "criticality": "material",
            },
        ],
        "limitation_codes": [],
    }
    assert _plan_identity(
        body=body,
        question_digest=question_digest,
        scope=scope,
        inventory=inventory,
        result_fields=identity_fields,
    ) == "evidence_plan_cfd23e0d0da1c1f6a98709cc25990bf3"
    assert evidence_acquisition_premise_digest(
        question_anchor_digest=question_digest,
        task_shape="targeted_lookup",
        declared_scope=scope,
        source_inventory=inventory,
        selected_strategies=["targeted_retrieval"],
    ) == "sha256:3965129ab2599a23d5a23e559d5849439b8066eb5a5f169b49b5498bbf9438de"


def test_aggregate_plan_event_keeps_spec_and_content_fields_private():
    runtime = _start_runtime()
    response = _compile(
        runtime,
        task_shape="aggregate",
        declared_scope=_scope(source_ids=["source-a"]),
        source_inventory=[
            _source(
                "source-a",
                capabilities=["context_expansion"],
                content_fields=["PRIVATE_AGGREGATE_FIELD"],
            )
        ],
        aggregate_spec=_aggregate_spec(field_name="PRIVATE_AGGREGATE_FIELD"),
    )

    assert response.status_code == 200
    event = _plan_events(runtime["runtime_session_id"])[0]["event_payload_json"]
    serialized = json.dumps(event, sort_keys=True)
    assert event["task_shape"] == "aggregate"
    assert event["selected_strategies"] == ["structured_field_values"]
    assert "PRIVATE_AGGREGATE_FIELD" not in serialized
    assert "aggregate_spec" not in serialized
    assert "content_fields" not in serialized
    assert '"function"' not in serialized
