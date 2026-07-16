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
    resolved_owner = owner_id or f"owner-{ordinal}"
    resolved_conversation = conversation_id or f"conversation-{ordinal}"
    response = client.post(
        "/v1/runtime/turns/start",
        json={
            "request_id": f"request-start-{ordinal}",
            "owner_id": resolved_owner,
            "conversation_id": resolved_conversation,
            "surface": surface,
        },
    )
    assert response.status_code == 200
    body = response.json()
    return {
        "request_id": f"request-evaluate-{ordinal}",
        "owner_id": resolved_owner,
        "conversation_id": resolved_conversation,
        "surface": surface,
        "runtime_session_id": body["runtime_session"]["runtime_session_id"],
        "runtime_turn_id": body["runtime_turn"]["runtime_turn_id"],
        "evidence_plan_id": f"plan-{ordinal}",
        "acquisition_manifest_id": f"manifest-{ordinal}",
        "task_shape": "targeted_lookup",
    }


def _requirement(
    requirement_id: str,
    *,
    requirement_kind: str = "exact_authoritative_fetch",
    criticality: str = "material",
) -> dict[str, str]:
    return {
        "requirement_id": requirement_id,
        "requirement_kind": requirement_kind,
        "criticality": criticality,
    }


def _fact(requirement_id: str, outcome: str = "satisfied") -> dict[str, str]:
    return {"requirement_id": requirement_id, "outcome": outcome}


def _evaluate(
    scope: dict[str, object],
    requirements: list[dict[str, str]],
    facts: list[dict[str, str]],
    **overrides: object,
):
    payload: dict[str, object] = {
        **scope,
        "declared_requirements": requirements,
        "acquisition_facts": facts,
    }
    payload.update(overrides)
    return client.post("/v1/runtime/evidence-sufficiency/evaluate", json=payload)


def _evaluation_events(runtime_session_id: object) -> list[dict[str, object]]:
    response = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert response.status_code == 200
    return [
        event
        for event in response.json()["events"]
        if event["event_type"] == "evidence_sufficiency_evaluated"
    ]


@pytest.mark.parametrize(
    ("requirements", "facts", "expected_status"),
    [
        (
            [_requirement("material-1"), _requirement("optional-1", criticality="optional")],
            [_fact("material-1"), _fact("optional-1")],
            "sufficient_for_declared_scope",
        ),
        (
            [_requirement("material-1"), _requirement("optional-1", criticality="optional")],
            [_fact("material-1")],
            "sufficient_with_limitations",
        ),
        (
            [_requirement("material-1")],
            [_fact("material-1", "partial")],
            "insufficient",
        ),
        (
            [_requirement("material-1")],
            [_fact("material-1", "unknown")],
            "unknown",
        ),
    ],
)
def test_four_sufficiency_statuses_are_derived_exactly(
    requirements: list[dict[str, str]],
    facts: list[dict[str, str]],
    expected_status: str,
):
    response = _evaluate(_start_runtime(), requirements, facts)

    assert response.status_code == 200
    assert response.json()["result"]["sufficiency_status"] == expected_status


def test_small_complete_scope_beats_large_incomplete_scope():
    small = _evaluate(
        _start_runtime(),
        [_requirement("authoritative-record")],
        [_fact("authoritative-record")],
    )
    large_requirements = [
        _requirement(f"requirement-{index:02d}", requirement_kind="bounded-source")
        for index in range(32)
    ]
    large_facts = [
        _fact(
            f"requirement-{index:02d}",
            "truncated" if index == 31 else "satisfied",
        )
        for index in range(32)
    ]
    large = _evaluate(_start_runtime(), large_requirements, large_facts)

    assert small.json()["result"]["sufficiency_status"] == (
        "sufficient_for_declared_scope"
    )
    assert large.json()["result"]["sufficiency_status"] == "insufficient"
    assert "material_requirement_not_satisfied" in large.json()["result"]["reason_codes"]


def test_optional_incompleteness_limits_without_requiring_more_acquisition():
    response = _evaluate(
        _start_runtime(),
        [
            _requirement("material-1"),
            _requirement("optional-1", criticality="optional"),
        ],
        [_fact("material-1"), _fact("optional-1", "failed")],
    )
    result = response.json()["result"]

    assert result["sufficiency_status"] == "sufficient_with_limitations"
    assert result["answer_constraints"] == [
        "qualify_conclusion",
        "disclose_limitations",
        "identify_unexamined_scope",
    ]
    assert result["qualification_required"] is True
    assert result["additional_acquisition_required"] is False
    assert result["reason_codes"] == ["optional_requirement_incomplete"]


def test_concrete_material_failure_wins_over_missing_material_fact():
    response = _evaluate(
        _start_runtime(),
        [_requirement("failed-1"), _requirement("missing-1")],
        [_fact("failed-1", "failed")],
    )
    result = response.json()["result"]

    assert result["sufficiency_status"] == "insufficient"
    assert "material_requirement_not_satisfied" in result["reason_codes"]
    assert "material_requirement_missing" in result["reason_codes"]


@pytest.mark.parametrize(
    ("task_shape", "outcome", "constraint", "reason"),
    [
        (
            "bounded_exhaustive_review",
            "partial",
            "withhold_exhaustive_conclusion",
            "exhaustive_scope_incomplete",
        ),
        (
            "absence_or_coverage_check",
            "unknown",
            "withhold_absence_conclusion",
            "absence_scope_unproven",
        ),
        (
            "contradiction_review",
            "unresolved_contradiction",
            "withhold_contradiction_sensitive_conclusion",
            "contradiction_sensitive_scope_unresolved",
        ),
    ],
)
def test_scope_sensitive_tasks_receive_exact_withholding_constraints(
    task_shape: str,
    outcome: str,
    constraint: str,
    reason: str,
):
    scope = _start_runtime()
    scope["task_shape"] = task_shape
    response = _evaluate(
        scope,
        [_requirement("scope-1", requirement_kind="complete-item-coverage")],
        [_fact("scope-1", outcome)],
    )
    result = response.json()["result"]

    assert result["sufficiency_status"] in {"insufficient", "unknown"}
    assert constraint in result["answer_constraints"]
    assert reason in result["reason_codes"]
    assert result["additional_acquisition_required"] is True


@pytest.mark.parametrize(
    ("task_shape", "withholding_constraint"),
    [
        ("bounded_exhaustive_review", "withhold_exhaustive_conclusion"),
        ("absence_or_coverage_check", "withhold_absence_conclusion"),
        ("contradiction_review", "withhold_contradiction_sensitive_conclusion"),
    ],
)
def test_complete_scope_has_no_withholding_constraint(
    task_shape: str,
    withholding_constraint: str,
):
    scope = _start_runtime()
    scope["task_shape"] = task_shape
    response = _evaluate(
        scope,
        [_requirement("scope-1")],
        [_fact("scope-1")],
    )
    result = response.json()["result"]

    assert result["sufficiency_status"] == "sufficient_for_declared_scope"
    assert result["answer_constraints"] == []
    assert withholding_constraint not in result["answer_constraints"]


def test_missing_absence_scope_is_unknown_not_proof_of_absence():
    scope = _start_runtime()
    scope["task_shape"] = "absence_or_coverage_check"
    response = _evaluate(
        scope,
        [_requirement("authoritative-inventory", requirement_kind="authoritative-inventory")],
        [],
    )
    result = response.json()["result"]

    assert result["sufficiency_status"] == "unknown"
    assert result["evaluated_requirements"][0]["effective_outcome"] == "missing"
    assert "material_requirement_missing" in result["reason_codes"]
    assert "withhold_absence_conclusion" in result["answer_constraints"]


@pytest.mark.parametrize(
    "extra_field",
    [
        "model_confidence",
        "citation_count",
        "result_count",
        "context_size",
        "evidence_volume",
        "answer_text",
        "sufficiency_status",
        "reason_codes",
        "answer_constraints",
        "metadata",
    ],
)
def test_caller_selected_or_upgrade_fields_are_rejected(extra_field: str):
    response = _evaluate(
        _start_runtime(),
        [_requirement("material-1")],
        [_fact("material-1")],
        **{extra_field: "caller-supplied"},
    )

    assert response.status_code == 422
    assert "extra_forbidden" in response.text


@pytest.mark.parametrize(
    ("target", "extra_field"),
    [
        ("requirement", "raw_content"),
        ("requirement", "prompt"),
        ("fact", "snippet"),
        ("fact", "reasoning"),
        ("fact", "exception"),
        ("fact", "metadata"),
    ],
)
def test_nested_raw_or_unrestricted_fields_are_rejected(target: str, extra_field: str):
    requirement = _requirement("material-1")
    fact = _fact("material-1")
    if target == "requirement":
        requirement[extra_field] = "not accepted"
    else:
        fact[extra_field] = "not accepted"

    response = _evaluate(_start_runtime(), [requirement], [fact])

    assert response.status_code == 422
    assert "extra_forbidden" in response.text


def test_duplicate_and_undeclared_requirement_facts_are_rejected():
    scope = _start_runtime()
    duplicate_requirements = _evaluate(
        scope,
        [_requirement("same-1"), _requirement("same-1", requirement_kind="other-kind")],
        [],
    )
    duplicate_facts = _evaluate(
        scope,
        [_requirement("same-1")],
        [_fact("same-1"), _fact("same-1", "partial")],
    )
    undeclared = _evaluate(
        scope,
        [_requirement("declared-1")],
        [_fact("other-1")],
    )

    assert duplicate_requirements.status_code == 422
    assert "duplicate_evidence_requirement" in duplicate_requirements.text
    assert duplicate_facts.status_code == 422
    assert "duplicate_acquisition_fact" in duplicate_facts.text
    assert undeclared.status_code == 422
    assert "undeclared_evidence_requirement" in undeclared.text


def test_identifier_and_list_bounds_are_enforced():
    empty = _evaluate(_start_runtime(), [], [])
    too_many_requirements = _evaluate(
        _start_runtime(),
        [_requirement(f"requirement-{index}") for index in range(33)],
        [],
    )
    too_many_facts = _evaluate(
        _start_runtime(),
        [_requirement(f"requirement-{index}") for index in range(32)],
        [_fact(f"requirement-{index % 32}") for index in range(33)],
    )
    unsafe_identifier = _evaluate(
        _start_runtime(),
        [_requirement("https://example.invalid/source?token=private")],
        [],
    )
    overlong_identifier = _evaluate(
        _start_runtime(),
        [_requirement("r" * 121)],
        [],
    )

    assert empty.status_code == 422
    assert too_many_requirements.status_code == 422
    assert too_many_facts.status_code == 422
    assert unsafe_identifier.status_code == 422
    assert overlong_identifier.status_code == 422


def test_requirement_kind_is_bounded_but_not_a_closed_catalog():
    response = _evaluate(
        _start_runtime(),
        [_requirement("custom-1", requirement_kind="neutral-custom-capability")],
        [_fact("custom-1")],
    )

    assert response.status_code == 200
    assert response.json()["result"]["evaluated_requirements"][0][
        "requirement_kind"
    ] == "neutral-custom-capability"


def test_reordered_equivalent_inputs_produce_identical_results():
    scope = _start_runtime()
    requirements = [
        _requirement("material-b", requirement_kind="context-delivery"),
        _requirement("material-a", requirement_kind="authoritative-inventory"),
        _requirement("optional-c", criticality="optional"),
    ]
    facts = [
        _fact("material-b"),
        _fact("material-a"),
        _fact("optional-c", "unavailable"),
    ]

    first = _evaluate(scope, requirements, facts)
    second = _evaluate(scope, list(reversed(requirements)), list(reversed(facts)))

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["result"] == second.json()["result"]
    assert [
        item["requirement_id"] for item in first.json()["result"]["evaluated_requirements"]
    ] == ["material-a", "material-b", "optional-c"]


def test_unknown_runtime_session_returns_bounded_error_and_no_event():
    scope = _start_runtime()
    original_session_id = scope["runtime_session_id"]
    scope["runtime_session_id"] = "rtsession-missing"
    response = _evaluate(scope, [_requirement("material-1")], [])

    assert response.status_code == 404
    assert response.json() == {"detail": "runtime_session_not_found"}
    assert _evaluation_events(original_session_id) == []


def test_unknown_runtime_turn_returns_bounded_error_and_no_event():
    scope = _start_runtime()
    scope["runtime_turn_id"] = "rtturn-missing"
    response = _evaluate(scope, [_requirement("material-1")], [])

    assert response.status_code == 404
    assert response.json() == {"detail": "runtime_turn_not_found"}
    assert _evaluation_events(scope["runtime_session_id"]) == []


def test_runtime_session_scope_mismatch_returns_bounded_error_and_no_event():
    scope = _start_runtime()
    scope["owner_id"] = "another-owner"
    response = _evaluate(scope, [_requirement("material-1")], [])

    assert response.status_code == 400
    assert response.json() == {"detail": "runtime_session_mismatch"}
    assert _evaluation_events(scope["runtime_session_id"]) == []


def test_runtime_turn_session_mismatch_returns_bounded_error_and_no_event():
    first_scope = _start_runtime()
    second_scope = _start_runtime()
    second_scope["runtime_turn_id"] = first_scope["runtime_turn_id"]
    response = _evaluate(second_scope, [_requirement("material-1")], [])

    assert response.status_code == 400
    assert response.json() == {"detail": "runtime_turn_session_mismatch"}
    assert _evaluation_events(second_scope["runtime_session_id"]) == []


def test_response_and_runtime_event_are_bounded_private_and_consistent():
    scope = _start_runtime()
    scope["task_shape"] = "contradiction_review"
    response = _evaluate(
        scope,
        [
            _requirement(
                "private-requirement-id",
                requirement_kind="private-requirement-kind",
            )
        ],
        [_fact("private-requirement-id", "unresolved_contradiction")],
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
        "evidence_plan_id",
        "acquisition_manifest_id",
        "result",
    }
    assert set(result) == {
        "evaluation_id",
        "task_shape",
        "sufficiency_status",
        "evaluated_requirements",
        "reason_codes",
        "answer_constraints",
        "qualification_required",
        "additional_acquisition_required",
        "user_safe_summary",
    }

    event = _evaluation_events(scope["runtime_session_id"])[0]
    payload = event["event_payload_json"]
    assert set(payload) == {
        "request_id",
        "runtime_session_id",
        "runtime_turn_id",
        "evaluation_id",
        "evidence_plan_id",
        "acquisition_manifest_id",
        "task_shape",
        "sufficiency_status",
        "total_requirement_count",
        "material_requirement_count",
        "optional_requirement_count",
        "satisfied_requirement_count",
        "missing_requirement_count",
        "non_satisfactory_requirement_count",
        "reason_codes",
        "answer_constraints",
        "qualification_required",
        "additional_acquisition_required",
    }
    for field in (
        "evaluation_id",
        "task_shape",
        "sufficiency_status",
        "reason_codes",
        "answer_constraints",
        "qualification_required",
        "additional_acquisition_required",
    ):
        assert payload[field] == result[field]
    assert payload["total_requirement_count"] == 1
    assert payload["material_requirement_count"] == 1
    assert payload["optional_requirement_count"] == 0
    assert payload["satisfied_requirement_count"] == 0
    assert payload["missing_requirement_count"] == 0
    assert payload["non_satisfactory_requirement_count"] == 1

    serialized = json.dumps(payload, sort_keys=True).lower()
    for excluded in (
        "private-requirement-id",
        "private-requirement-kind",
        "acquisition_facts",
        "declared_requirements",
        "raw evidence",
        "source content",
        "prompt",
        "private memory",
        "reasoning",
        "metadata",
        "exception",
    ):
        assert excluded not in serialized
