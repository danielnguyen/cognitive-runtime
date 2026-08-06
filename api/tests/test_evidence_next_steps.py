from __future__ import annotations

import copy
import json
from hashlib import sha256
from itertools import count

import pytest
from fastapi.testclient import TestClient
from main import app
from models import EvidenceAcquisitionPremise, EvidenceNextStepResult
from pydantic import ValidationError
from services import evidence_next_steps
from services.evidence_planning import evidence_acquisition_premise_digest

client = TestClient(app)
_runtime_counter = count()
_DEFAULT_QUESTION = "What evidence supports the conclusion?"


def _question_digest(value: str) -> str:
    return f"sha256:{sha256(value.encode('utf-8')).hexdigest()}"


def _start_runtime(
    *,
    question_anchor: str = _DEFAULT_QUESTION,
    premise: dict[str, object] | None = None,
    derive_shape: bool = False,
    interaction_kind: str = "question",
    shape_context: dict[str, object] | None = None,
) -> dict[str, object]:
    ordinal = next(_runtime_counter)
    planned_premise = copy.deepcopy(premise or _premise(question=question_anchor))
    assert planned_premise["question_anchor_digest"] == _question_digest(
        question_anchor
    )
    response = client.post(
        "/v1/runtime/turns/start",
        json={
            "request_id": f"start-{ordinal}",
            "owner_id": f"owner-{ordinal}",
            "conversation_id": f"conversation-{ordinal}",
            "surface": "web",
        },
    )
    assert response.status_code == 200
    body = response.json()
    scope: dict[str, object] = {
        "request_id": f"evaluate-{ordinal}",
        "owner_id": f"owner-{ordinal}",
        "conversation_id": f"conversation-{ordinal}",
        "surface": "web",
        "runtime_session_id": body["runtime_session"]["runtime_session_id"],
        "runtime_turn_id": body["runtime_turn"]["runtime_turn_id"],
        "acquisition_manifest_id": f"manifest-{ordinal}",
    }
    if derive_shape:
        shape_response = client.post(
            "/v1/runtime/evidence-shapes/derive",
            json={
                "request_id": f"derive-{ordinal}",
                "owner_id": scope["owner_id"],
                "conversation_id": scope["conversation_id"],
                "surface": scope["surface"],
                "runtime_session_id": scope["runtime_session_id"],
                "runtime_turn_id": scope["runtime_turn_id"],
                "task_text": question_anchor,
                "interaction_kind": interaction_kind,
                "task_context": shape_context or _shape_context(),
            },
        )
        assert shape_response.status_code == 200
        shape_result = shape_response.json()["result"]
        assert shape_result["question_anchor"] == question_anchor
        assert shape_result["task_shape"] == planned_premise["task_shape"]
        scope["_shape_result"] = shape_result
    plan_response = client.post(
        "/v1/runtime/evidence-plans/compile",
        json={
            "request_id": f"compile-{ordinal}",
            "owner_id": scope["owner_id"],
            "conversation_id": scope["conversation_id"],
            "surface": scope["surface"],
            "runtime_session_id": scope["runtime_session_id"],
            "runtime_turn_id": scope["runtime_turn_id"],
            "question_anchor": question_anchor,
            "task_shape": planned_premise["task_shape"],
            "declared_scope": planned_premise["declared_scope"],
            "source_inventory": planned_premise["source_inventory"],
        },
    )
    assert plan_response.status_code == 200
    plan_result = plan_response.json()["result"]
    current_premise = {
        **planned_premise,
        "question_anchor_digest": plan_result["question_anchor_digest"],
        "selected_strategies": plan_result["selected_strategies"],
    }
    scope["evidence_plan_id"] = plan_result["plan_id"]
    scope["_current_premise"] = current_premise
    return scope


def _requirement(
    requirement_id: str,
    requirement_kind: str,
    *,
    criticality: str = "material",
) -> dict[str, str]:
    return {
        "requirement_id": requirement_id,
        "requirement_kind": requirement_kind,
        "criticality": criticality,
    }


def _fact(requirement_id: str, outcome: str) -> dict[str, str]:
    return {"requirement_id": requirement_id, "outcome": outcome}


def _evaluate(
    scope: dict[str, object],
    requirements: list[dict[str, str]],
    facts: list[dict[str, str]],
    *,
    task_shape: str = "targeted_lookup",
) -> dict[str, object]:
    response = client.post(
        "/v1/runtime/evidence-sufficiency/evaluate",
        json={
            **{
                key: scope[key]
                for key in (
                    "request_id",
                    "owner_id",
                    "conversation_id",
                    "surface",
                    "runtime_session_id",
                    "runtime_turn_id",
                    "evidence_plan_id",
                    "acquisition_manifest_id",
                )
            },
            "task_shape": task_shape,
            "declared_requirements": requirements,
            "acquisition_facts": facts,
        },
    )
    assert response.status_code == 200
    return response.json()


def _source(
    source_id: str = "source-a",
    *,
    categories: list[str] | None = None,
    capabilities: list[str] | None = None,
    availability: str = "available",
    authority_role: str = "authoritative",
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_categories": categories or ["records"],
        "capabilities": capabilities or [
            "targeted_retrieval",
            "context_expansion",
        ],
        "availability": availability,
        "authority_role": authority_role,
    }


def _shape_context(*, high_stakes_accuracy_required: bool = False) -> dict[str, object]:
    return {
        "evidence_input_kinds": [],
        "external_verification_required": False,
        "freshness_sensitive": False,
        "high_stakes_accuracy_required": high_stakes_accuracy_required,
        "continuation_of_prior_evidence_task": False,
        "prior_task_shape": None,
    }


def _premise(
    *,
    question: str = _DEFAULT_QUESTION,
    task_shape: str = "targeted_lookup",
    source_ids: list[str] | None = None,
    source_categories: list[str] | None = None,
    exact_source_refs: list[dict[str, str]] | None = None,
    inventory_status: str = "complete_for_declared_scope",
    source_inventory: list[dict[str, object]] | None = None,
    selected_strategies: list[str] | None = None,
) -> dict[str, object]:
    return {
        "question_anchor_digest": _question_digest(question),
        "task_shape": task_shape,
        "declared_scope": {
            "source_ids": source_ids or [],
            "source_categories": source_categories or [],
            "exact_source_refs": exact_source_refs or [],
            "inventory_status": inventory_status,
            "time_scope_ref": None,
            "version_scope_ref": None,
            "domain_scope_ref": None,
            "project_scope_ref": None,
        },
        "source_inventory": source_inventory or [_source()],
        "selected_strategies": selected_strategies or ["targeted_retrieval"],
    }


def _premise_digest(premise: dict[str, object]) -> str:
    model = EvidenceAcquisitionPremise.model_validate(premise)
    return evidence_acquisition_premise_digest(
        question_anchor_digest=model.question_anchor_digest,
        task_shape=model.task_shape,
        declared_scope=model.declared_scope,
        source_inventory=model.source_inventory,
        selected_strategies=model.selected_strategies,
    )


def _mutated_premise(
    premise: dict[str, object],
    mutation: str,
) -> dict[str, object]:
    changed = copy.deepcopy(premise)
    if mutation == "question":
        changed["question_anchor_digest"] = _question_digest("Different question")
    elif mutation == "task_shape":
        changed["task_shape"] = "cross_source_comparison"
    elif mutation == "source_ids":
        changed["declared_scope"]["source_ids"] = ["source-a"]
    elif mutation == "source_categories":
        changed["declared_scope"]["source_categories"] = ["other"]
    elif mutation == "exact_reference":
        changed["declared_scope"]["exact_source_refs"] = [
            {"source_ref": "sheet:a:2", "source_id": "source-a"}
        ]
    elif mutation == "inventory_status":
        changed["declared_scope"]["inventory_status"] = "partial"
    elif mutation in {
        "time_scope",
        "version_scope",
        "domain_scope",
        "project_scope",
    }:
        changed["declared_scope"][f"{mutation}_ref"] = f"{mutation}-ref"
    elif mutation == "descriptor_categories":
        changed["source_inventory"][0]["source_categories"] = ["other"]
    elif mutation == "availability":
        changed["source_inventory"][0]["availability"] = "unavailable"
    elif mutation == "authority":
        changed["source_inventory"][0]["authority_role"] = "supplemental"
    elif mutation == "capability":
        changed["source_inventory"][0]["capabilities"] = ["targeted_retrieval"]
    else:
        changed["selected_strategies"] = ["hybrid"]
    return changed


def _next_step_payload(
    scope: dict[str, object],
    evaluation: dict[str, object],
    *,
    current_premise: dict[str, object] | None = None,
    **overrides: object,
) -> dict[str, object]:
    result = evaluation["result"]
    assert isinstance(result, dict)
    payload: dict[str, object] = {
        **{
            key: scope[key]
            for key in (
                "request_id",
                "owner_id",
                "conversation_id",
                "surface",
                "runtime_session_id",
                "runtime_turn_id",
                "evidence_plan_id",
                "acquisition_manifest_id",
            )
        },
        "request_id": f"select-{scope['request_id']}",
        "evaluation_id": result["evaluation_id"],
        "evaluated_requirements": result["evaluated_requirements"],
        "current_premise": current_premise
        or copy.deepcopy(scope["_current_premise"]),
    }
    payload.update(overrides)
    return payload


def _select(payload: dict[str, object]):
    return client.post("/v1/runtime/evidence-next-steps/select", json=payload)


def _selection_events(runtime_session_id: object) -> list[dict[str, object]]:
    response = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert response.status_code == 200
    return [
        event
        for event in response.json()["events"]
        if event["event_type"] == "evidence_next_step_selected"
    ]


def _plan_events(runtime_session_id: object) -> list[dict[str, object]]:
    response = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert response.status_code == 200
    return [
        event
        for event in response.json()["events"]
        if event["event_type"] == "evidence_plan_compiled"
    ]


@pytest.mark.parametrize(
    (
        "facts",
        "expected_status",
        "expected_step",
        "expected_conclusion",
        "expected_provider",
    ),
    [
        (
            [_fact("material", "satisfied"), _fact("optional", "satisfied")],
            "sufficient_for_declared_scope",
            "answer_within_declared_scope",
            "bounded_conclusion_allowed",
            "allowed",
        ),
        (
            [_fact("material", "satisfied"), _fact("optional", "unavailable")],
            "sufficient_with_limitations",
            "provide_qualified_partial_answer",
            "qualified_partial_only",
            "allowed",
        ),
    ],
)
def test_sufficient_statuses_select_answer_without_more_acquisition(
    facts: list[dict[str, str]],
    expected_status: str,
    expected_step: str,
    expected_conclusion: str,
    expected_provider: str,
):
    question = "Does this package version support Python 3.14?"
    scope = _start_runtime(question_anchor=question, derive_shape=True)
    evaluation = _evaluate(
        scope,
        [
            _requirement("material", "targeted_evidence"),
            _requirement("optional", "selected_source_coverage", criticality="optional"),
        ],
        facts,
    )
    payload = _next_step_payload(
        scope,
        evaluation,
        proposed_acquisition_premise=_premise(question="A changed question"),
    )
    response = _select(payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "request_id",
        "owner_id",
        "conversation_id",
        "surface",
        "runtime_session_id",
        "runtime_turn_id",
        "result",
    }
    result = body["result"]
    assert set(result) == {
        "selection_id",
        "evaluation_id",
        "evidence_plan_id",
        "acquisition_manifest_id",
        "task_shape",
        "sufficiency_status",
        "selected_next_step",
        "conclusion_disposition",
        "provider_disposition",
        "current_premise_digest",
        "proposed_premise_digest",
        "reacquisition_guard",
        "clarification_target",
        "unresolved_material_requirement_ids",
        "reason_codes",
        "user_safe_summary",
    }
    assert result["sufficiency_status"] == expected_status
    assert result["selected_next_step"] == expected_step
    assert result["conclusion_disposition"] == expected_conclusion
    assert result["provider_disposition"] == expected_provider
    assert result["reacquisition_guard"] == "not_applicable"
    assert result["proposed_premise_digest"] is None


@pytest.mark.parametrize("outcome", ["missing", "unknown"])
def test_missing_or_unknown_material_requirement_permits_narrow_clarification(
    outcome: str,
):
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("missing-scope", "targeted_evidence")],
        [] if outcome == "missing" else [_fact("missing-scope", outcome)],
    )
    response = _select(
        _next_step_payload(
            scope,
            evaluation,
            clarification_target="source_scope",
        )
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["selected_next_step"] == "ask_narrow_clarification"
    assert result["clarification_target"] == "source_scope"
    assert result["provider_disposition"] == "blocked"
    assert result["conclusion_disposition"] == "requested_conclusion_withheld"


@pytest.mark.parametrize(
    "outcome",
    [
        "failed",
        "filtered",
        "truncated",
        "unsupported",
        "unavailable",
        "unresolved_contradiction",
    ],
)
def test_concrete_failure_cannot_force_clarification(outcome: str):
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("failed", "targeted_evidence")],
        [_fact("failed", outcome)],
    )
    response = _select(
        _next_step_payload(
            scope,
            evaluation,
            clarification_target="question_scope",
        )
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["selected_next_step"] == "withhold_unsupported_conclusion"
    assert result["clarification_target"] is None


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [("failed", "insufficient"), ("unknown", "unknown")],
)
def test_associated_low_risk_targeted_gap_withholds_conclusion_and_allows_guidance(
    outcome: str,
    expected_status: str,
):
    question = "Does this package version support Python 3.14?"
    scope = _start_runtime(question_anchor=question, derive_shape=True)
    evaluation = _evaluate(
        scope,
        [_requirement("targeted", "targeted_evidence")],
        [_fact("targeted", outcome)],
    )

    response = _select(_next_step_payload(scope, evaluation))

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["task_shape"] == "targeted_lookup"
    assert result["sufficiency_status"] == expected_status
    assert result["selected_next_step"] == "withhold_unsupported_conclusion"
    assert result["conclusion_disposition"] == "requested_conclusion_withheld"
    assert result["provider_disposition"] == "allowed"
    assert result["reacquisition_guard"] == "not_applicable"
    assert result["reason_codes"] == ["unsupported_conclusion_withheld"]
    assert result["user_safe_summary"] == (
        "The requested conclusion remains unsupported; non-authoritative "
        "guidance is permitted without presenting it as verified."
    )
    event = _selection_events(scope["runtime_session_id"])[0][
        "event_payload_json"
    ]
    assert event["selected_next_step"] == result["selected_next_step"]
    assert event["conclusion_disposition"] == result["conclusion_disposition"]
    assert event["provider_disposition"] == "allowed"
    serialized = json.dumps(event, sort_keys=True).lower()
    assert question.lower() not in serialized
    assert "non-authoritative guidance" not in serialized


@pytest.mark.parametrize(
    ("interaction_kind", "high_stakes"),
    [("high_impact_decision", False), ("question", True)],
)
def test_associated_high_risk_targeted_gap_keeps_provider_blocked(
    interaction_kind: str,
    high_stakes: bool,
):
    question = "Can this adapter be used with that device?"
    scope = _start_runtime(
        question_anchor=question,
        derive_shape=True,
        interaction_kind=interaction_kind,
        shape_context=_shape_context(
            high_stakes_accuracy_required=high_stakes,
        ),
    )
    evaluation = _evaluate(
        scope,
        [_requirement("targeted", "targeted_evidence")],
        [_fact("targeted", "failed")],
    )

    result = _select(_next_step_payload(scope, evaluation)).json()["result"]

    assert result["selected_next_step"] == "withhold_unsupported_conclusion"
    assert result["conclusion_disposition"] == "requested_conclusion_withheld"
    assert result["provider_disposition"] == "blocked"


def test_absent_associated_shape_event_keeps_provider_blocked():
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("targeted", "targeted_evidence")],
        [_fact("targeted", "failed")],
    )

    result = _select(_next_step_payload(scope, evaluation)).json()["result"]

    assert result["selected_next_step"] == "withhold_unsupported_conclusion"
    assert result["provider_disposition"] == "blocked"


@pytest.mark.parametrize(
    "event_mutation",
    [
        "session",
        "turn",
        "question_anchor",
        "task_shape",
        "malformed_reasons",
        "duplicate",
    ],
)
def test_mismatched_or_malformed_shape_association_keeps_provider_blocked(
    event_mutation: str,
    monkeypatch: pytest.MonkeyPatch,
):
    question = "Does this package version support Python 3.14?"
    scope = _start_runtime(question_anchor=question, derive_shape=True)
    evaluation = _evaluate(
        scope,
        [_requirement("targeted", "targeted_evidence")],
        [_fact("targeted", "failed")],
    )
    original_get_runtime_session = evidence_next_steps.get_runtime_session

    def _altered_diagnostics(runtime_session_id: str):
        diagnostics = original_get_runtime_session(runtime_session_id)
        for index, event in enumerate(diagnostics.events):
            if event.event_type != "evidence_shape_derived":
                continue
            if event_mutation == "session":
                event.event_payload_json["runtime_session_id"] = "other-session"
            elif event_mutation == "turn":
                event.event_payload_json["runtime_turn_id"] = "other-turn"
            elif event_mutation == "question_anchor":
                event.event_payload_json["question_anchor_digest"] = (
                    _question_digest("Different question")
                )
            elif event_mutation == "task_shape":
                event.event_payload_json["task_shape"] = "cross_source_comparison"
            elif event_mutation == "malformed_reasons":
                event.event_payload_json["reason_codes"] = ["not-a-reason"]
            else:
                duplicate = event.model_copy(deep=True)
                duplicate.event_id = f"{event.event_id}-duplicate"
                diagnostics.events.insert(index + 1, duplicate)
            break
        return diagnostics

    monkeypatch.setattr(
        evidence_next_steps,
        "get_runtime_session",
        _altered_diagnostics,
    )
    response = _select(_next_step_payload(scope, evaluation))

    assert response.status_code == 200
    assert response.json()["result"]["provider_disposition"] == "blocked"


def test_changed_premise_precedes_advisory_permission():
    question = "Does this package version support Python 3.14?"
    scope = _start_runtime(question_anchor=question, derive_shape=True)
    evaluation = _evaluate(
        scope,
        [_requirement("targeted", "targeted_evidence")],
        [_fact("targeted", "failed")],
    )

    result = _select(
        _next_step_payload(
            scope,
            evaluation,
            proposed_acquisition_premise=_premise(
                question="Does this other version support Python 3.14?"
            ),
        )
    ).json()["result"]

    assert result["selected_next_step"] == "perform_additional_acquisition"
    assert result["provider_disposition"] == "blocked"
    assert result["reacquisition_guard"] == "changed_premise_allowed"


def test_exhausted_changed_premise_can_reach_advisory_fallback():
    question = "Does this package version support Python 3.14?"
    scope = _start_runtime(question_anchor=question, derive_shape=True)
    evaluation = _evaluate(
        scope,
        [_requirement("targeted", "targeted_evidence")],
        [_fact("targeted", "failed")],
    )
    proposed = _premise(
        question="Does this other version support Python 3.14?"
    )
    first_payload = _next_step_payload(
        scope,
        evaluation,
        proposed_acquisition_premise=proposed,
    )
    first = _select(first_payload).json()["result"]
    repeated = _select(
        {
            **first_payload,
            "request_id": f"{first_payload['request_id']}-repeated",
        }
    ).json()["result"]

    assert first["selected_next_step"] == "perform_additional_acquisition"
    assert first["provider_disposition"] == "blocked"
    assert repeated["selected_next_step"] == "withhold_unsupported_conclusion"
    assert repeated["conclusion_disposition"] == "requested_conclusion_withheld"
    assert repeated["provider_disposition"] == "allowed"
    assert repeated["reacquisition_guard"] == "premise_already_attempted"
    assert repeated["reason_codes"] == [
        "acquisition_premise_already_selected",
        "unsupported_conclusion_withheld",
    ]


def test_permitted_clarification_precedes_advisory_permission():
    question = "Does this package version support Python 3.14?"
    scope = _start_runtime(question_anchor=question, derive_shape=True)
    evaluation = _evaluate(
        scope,
        [_requirement("targeted", "targeted_evidence")],
        [_fact("targeted", "unknown")],
    )

    result = _select(
        _next_step_payload(
            scope,
            evaluation,
            clarification_target="version_scope",
        )
    ).json()["result"]

    assert result["selected_next_step"] == "ask_narrow_clarification"
    assert result["provider_disposition"] == "blocked"
    assert result["clarification_target"] == "version_scope"


def test_unchanged_premise_blocks_additional_acquisition():
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("failed", "targeted_evidence")],
        [_fact("failed", "failed")],
    )
    premise = copy.deepcopy(scope["_current_premise"])
    response = _select(
        _next_step_payload(
            scope,
            evaluation,
            current_premise=premise,
            proposed_acquisition_premise=copy.deepcopy(premise),
        )
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["selected_next_step"] != "perform_additional_acquisition"
    assert result["reacquisition_guard"] == "unchanged_premise_blocked"
    assert result["current_premise_digest"] == result["proposed_premise_digest"]
    assert result["current_premise_digest"] == _plan_events(
        scope["runtime_session_id"]
    )[0]["event_payload_json"]["acquisition_premise_digest"]


def test_changed_premise_is_selected_once_and_a_second_change_is_permitted():
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("scope-gap", "complete_scope_coverage")],
        [_fact("scope-gap", "failed")],
    )
    first_payload = _next_step_payload(
        scope,
        evaluation,
        proposed_acquisition_premise=_premise(question="Changed question one"),
    )
    first = _select(first_payload)
    repeated_payload = {
        **first_payload,
        "request_id": f"{first_payload['request_id']}-repeated",
    }
    repeated = _select(repeated_payload)
    second_change = _select(
        {
            **first_payload,
            "request_id": f"{first_payload['request_id']}-second-change",
            "proposed_acquisition_premise": _premise(
                question="Changed question two"
            ),
        }
    )

    assert first.json()["result"]["selected_next_step"] == (
        "perform_additional_acquisition"
    )
    assert first.json()["result"]["reacquisition_guard"] == "changed_premise_allowed"
    assert repeated.json()["result"]["selected_next_step"] != (
        "perform_additional_acquisition"
    )
    assert repeated.json()["result"]["reacquisition_guard"] == (
        "premise_already_attempted"
    )
    assert second_change.json()["result"]["selected_next_step"] == (
        "perform_additional_acquisition"
    )


def test_exact_duplicate_request_is_idempotent_and_records_one_event():
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("failed", "targeted_evidence")],
        [_fact("failed", "failed")],
    )
    payload = _next_step_payload(
        scope,
        evaluation,
        proposed_acquisition_premise=_premise(question="Changed question"),
    )

    first = _select(payload)
    second = _select(payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    events = _selection_events(scope["runtime_session_id"])
    assert len(events) == 1


def test_equivalent_reordered_premises_have_identical_digests():
    first = _premise(
        source_ids=["source-b", "source-a"],
        source_categories=["records", "official"],
        exact_source_refs=[
            {"source_ref": "sheet:b:2", "source_id": "source-b"},
            {"source_ref": "sheet:a:2", "source_id": "source-a"},
        ],
        source_inventory=[
            _source(
                "source-b",
                categories=["official", "records"],
                capabilities=["context_expansion", "targeted_retrieval"],
            ),
            _source(
                "source-a",
                categories=["records", "official"],
                capabilities=["targeted_retrieval", "context_expansion"],
            ),
        ],
        selected_strategies=["hybrid", "targeted_retrieval"],
    )
    second = _premise(
        source_ids=["source-a", "source-b"],
        source_categories=["official", "records"],
        exact_source_refs=[
            {"source_ref": "sheet:a:2", "source_id": "source-a"},
            {"source_ref": "sheet:b:2", "source_id": "source-b"},
        ],
        source_inventory=[
            _source(
                "source-a",
                categories=["official", "records"],
                capabilities=["context_expansion", "targeted_retrieval"],
            ),
            _source(
                "source-b",
                categories=["records", "official"],
                capabilities=["targeted_retrieval", "context_expansion"],
            ),
        ],
        selected_strategies=["targeted_retrieval", "hybrid"],
    )

    assert _premise_digest(first) == _premise_digest(second)


@pytest.mark.parametrize(
    "mutation",
    [
        "question",
        "task_shape",
        "source_ids",
        "source_categories",
        "exact_reference",
        "inventory_status",
        "time_scope",
        "version_scope",
        "domain_scope",
        "project_scope",
        "descriptor_categories",
        "availability",
        "authority",
        "capability",
        "strategy",
    ],
)
def test_material_premise_changes_change_the_digest(mutation: str):
    original = _premise()
    changed = _mutated_premise(original, mutation)

    original_digest = _premise_digest(original)
    changed_digest = _premise_digest(changed)
    assert changed_digest != original_digest


@pytest.mark.parametrize(
    "mutation",
    [
        "question",
        "task_shape",
        "source_ids",
        "source_categories",
        "exact_reference",
        "inventory_status",
        "time_scope",
        "version_scope",
        "domain_scope",
        "project_scope",
        "descriptor_categories",
        "availability",
        "authority",
        "capability",
        "strategy",
    ],
)
def test_current_premise_mutation_fails_compiled_plan_association(mutation: str):
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("failed", "targeted_evidence")],
        [_fact("failed", "failed")],
    )
    claimed_current = _mutated_premise(scope["_current_premise"], mutation)

    response = _select(
        _next_step_payload(
            scope,
            evaluation,
            current_premise=claimed_current,
        )
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "current_acquisition_premise_mismatch"}
    assert _selection_events(scope["runtime_session_id"]) == []


def test_caller_cannot_relabel_attempted_premise_as_changed():
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("failed", "targeted_evidence")],
        [_fact("failed", "failed")],
    )
    actual_premise = copy.deepcopy(scope["_current_premise"])
    claimed_current = _mutated_premise(actual_premise, "question")

    response = _select(
        _next_step_payload(
            scope,
            evaluation,
            current_premise=claimed_current,
            proposed_acquisition_premise=actual_premise,
        )
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "current_acquisition_premise_mismatch"}
    assert _selection_events(scope["runtime_session_id"]) == []


def test_association_identifiers_do_not_enter_premise_digest():
    premise = _premise()
    first = _premise_digest(premise)

    association_only = {
        "request_id": "request-other",
        "runtime_session_id": "session-other",
        "runtime_turn_id": "turn-other",
        "evidence_plan_id": "plan-other",
        "acquisition_manifest_id": "manifest-other",
        "evaluation_id": "evaluation-other",
    }

    assert first == _premise_digest(premise)
    assert not set(association_only) & set(premise)


def test_safe_partial_answer_requires_delivered_substantive_evidence():
    question = "Does this package version support Python 3.14?"
    scope = _start_runtime(question_anchor=question, derive_shape=True)
    evaluation = _evaluate(
        scope,
        [
            _requirement("delivery", "context_delivery"),
            _requirement("evidence", "targeted_evidence"),
            _requirement("remaining", "no_material_truncation"),
        ],
        [
            _fact("delivery", "satisfied"),
            _fact("evidence", "partial"),
            _fact("remaining", "failed"),
        ],
    )
    response = _select(_next_step_payload(scope, evaluation))

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["selected_next_step"] == "provide_qualified_partial_answer"
    assert result["provider_disposition"] == "allowed"
    assert result["conclusion_disposition"] == "qualified_partial_only"


def test_administrative_only_satisfaction_does_not_permit_partial_answer():
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [
            _requirement("inventory", "authoritative_inventory"),
            _requirement("delivery", "context_delivery"),
            _requirement("truncation", "no_material_truncation"),
        ],
        [
            _fact("inventory", "satisfied"),
            _fact("delivery", "satisfied"),
            _fact("truncation", "failed"),
        ],
    )
    response = _select(_next_step_payload(scope, evaluation))

    assert response.status_code == 200
    assert response.json()["result"]["selected_next_step"] == (
        "withhold_unsupported_conclusion"
    )
    assert response.json()["result"]["provider_disposition"] == "blocked"


def test_unresolved_scope_selects_disclosure_and_hard_failure_selects_withholding():
    scope_gap_scope = _start_runtime()
    scope_gap_evaluation = _evaluate(
        scope_gap_scope,
        [_requirement("coverage", "complete_scope_coverage")],
        [_fact("coverage", "unavailable")],
    )
    hard_failure_scope = _start_runtime()
    hard_failure_evaluation = _evaluate(
        hard_failure_scope,
        [_requirement("targeted", "targeted_evidence")],
        [_fact("targeted", "unsupported")],
    )

    disclosure = _select(
        _next_step_payload(scope_gap_scope, scope_gap_evaluation)
    ).json()["result"]
    withholding = _select(
        _next_step_payload(hard_failure_scope, hard_failure_evaluation)
    ).json()["result"]

    assert disclosure["selected_next_step"] == "disclose_unexamined_scope"
    assert disclosure["provider_disposition"] == "blocked"
    assert withholding["selected_next_step"] == "withhold_unsupported_conclusion"
    assert withholding["provider_disposition"] == "blocked"


@pytest.mark.parametrize(
    ("question", "task_shape", "requirement_kind"),
    [
        (
            "Check every requirement in the complete checklist.",
            "bounded_exhaustive_review",
            "complete_scope_coverage",
        ),
        (
            "Check whether there is no record in the declared logs.",
            "absence_or_coverage_check",
            "structured_absence_check",
        ),
        (
            "Inspect whether these reports contradict each other.",
            "contradiction_review",
            "contradiction_search",
        ),
        (
            "Reconstruct what happened across the records last week.",
            "historical_reconstruction",
            "historical_scope",
        ),
        (
            "Which should I choose based on the evidence in these reports?",
            "recommendation_or_decision_support",
            "candidate_evidence_coverage",
        ),
    ],
)
def test_unsupported_non_targeted_shapes_keep_provider_blocked(
    question: str,
    task_shape: str,
    requirement_kind: str,
):
    scope = _start_runtime(
        question_anchor=question,
        premise=_premise(question=question, task_shape=task_shape),
        derive_shape=True,
    )
    evaluation = _evaluate(
        scope,
        [_requirement("material", requirement_kind)],
        [_fact("material", "failed")],
        task_shape=task_shape,
    )

    result = _select(_next_step_payload(scope, evaluation)).json()["result"]

    assert result["conclusion_disposition"] == "requested_conclusion_withheld"
    assert result["provider_disposition"] == "blocked"


def test_advisory_selection_is_idempotent_and_records_one_event():
    question = "Does this package version support Python 3.14?"
    scope = _start_runtime(question_anchor=question, derive_shape=True)
    evaluation = _evaluate(
        scope,
        [_requirement("targeted", "targeted_evidence")],
        [_fact("targeted", "failed")],
    )
    payload = _next_step_payload(scope, evaluation)

    first = _select(payload)
    second = _select(payload)

    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["result"]["provider_disposition"] == "allowed"
    assert len(_selection_events(scope["runtime_session_id"])) == 1


def _advisory_result_data() -> dict[str, object]:
    return {
        "selection_id": "selection-valid",
        "evaluation_id": "evaluation-valid",
        "evidence_plan_id": "plan-valid",
        "acquisition_manifest_id": "manifest-valid",
        "task_shape": "targeted_lookup",
        "sufficiency_status": "insufficient",
        "selected_next_step": "withhold_unsupported_conclusion",
        "conclusion_disposition": "requested_conclusion_withheld",
        "provider_disposition": "allowed",
        "current_premise_digest": f"sha256:{'a' * 64}",
        "proposed_premise_digest": None,
        "reacquisition_guard": "not_applicable",
        "clarification_target": None,
        "unresolved_material_requirement_ids": ["material"],
        "reason_codes": ["unsupported_conclusion_withheld"],
        "user_safe_summary": "The conclusion is unsupported; guidance is bounded.",
    }


def test_strict_result_model_accepts_only_the_targeted_advisory_form():
    result = EvidenceNextStepResult.model_validate(_advisory_result_data())

    assert result.selected_next_step == "withhold_unsupported_conclusion"
    assert result.conclusion_disposition == "requested_conclusion_withheld"
    assert result.provider_disposition == "allowed"


@pytest.mark.parametrize(
    "updates",
    [
        {"conclusion_disposition": "bounded_conclusion_allowed"},
        {"task_shape": "bounded_exhaustive_review"},
        {"task_shape": "absence_or_coverage_check"},
        {"task_shape": "contradiction_review"},
        {"task_shape": "historical_reconstruction"},
        {"task_shape": "recommendation_or_decision_support"},
        {"reason_codes": ["unexamined_material_scope"]},
        {
            "selected_next_step": "disclose_unexamined_scope",
            "reason_codes": ["unexamined_material_scope"],
        },
        {
            "selected_next_step": "ask_narrow_clarification",
            "clarification_target": "source_scope",
            "reason_codes": ["material_uncertainty_requires_clarification"],
        },
        {
            "selected_next_step": "perform_additional_acquisition",
            "proposed_premise_digest": f"sha256:{'b' * 64}",
            "reacquisition_guard": "changed_premise_allowed",
            "reason_codes": ["changed_acquisition_premise_available"],
        },
        {"reason_codes": ["unsupported_conclusion_withheld"] * 2},
    ],
)
def test_strict_result_model_rejects_incoherent_provider_permission(
    updates: dict[str, object],
):
    data = {**_advisory_result_data(), **updates}

    with pytest.raises(ValidationError):
        EvidenceNextStepResult.model_validate(data)


@pytest.mark.parametrize(
    ("override", "expected_status", "expected_detail"),
    [
        (
            {"evaluation_id": "missing-evaluation"},
            404,
            "evidence_sufficiency_evaluation_not_found",
        ),
        (
            {"evidence_plan_id": "wrong-plan"},
            400,
            "evidence_sufficiency_association_mismatch",
        ),
        (
            {"acquisition_manifest_id": "wrong-manifest"},
            400,
            "evidence_sufficiency_association_mismatch",
        ),
    ],
)
def test_unknown_or_mismatched_sufficiency_association_fails_boundedly(
    override: dict[str, str],
    expected_status: int,
    expected_detail: str,
):
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("missing", "targeted_evidence")],
        [],
    )
    response = _select(_next_step_payload(scope, evaluation, **override))

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    assert _selection_events(scope["runtime_session_id"]) == []


@pytest.mark.parametrize("plan_id_source", ["unknown", "other-session"])
def test_unknown_or_mismatched_plan_event_fails_boundedly(plan_id_source: str):
    scope = _start_runtime()
    mismatched_scope = copy.deepcopy(scope)
    if plan_id_source == "unknown":
        mismatched_scope["evidence_plan_id"] = "unknown-plan"
    else:
        mismatched_scope["evidence_plan_id"] = _start_runtime()["evidence_plan_id"]
    evaluation = _evaluate(
        mismatched_scope,
        [_requirement("failed", "targeted_evidence")],
        [_fact("failed", "failed")],
    )

    response = _select(_next_step_payload(mismatched_scope, evaluation))

    assert response.status_code == 400
    assert response.json() == {"detail": "current_acquisition_premise_mismatch"}
    assert _selection_events(scope["runtime_session_id"]) == []


@pytest.mark.parametrize(
    "event_mutation",
    ["missing_digest", "malformed_digest", "mismatched_task_shape"],
)
def test_invalid_plan_event_fails_before_selection(
    event_mutation: str,
    monkeypatch: pytest.MonkeyPatch,
):
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("failed", "targeted_evidence")],
        [_fact("failed", "failed")],
    )
    original_get_runtime_session = evidence_next_steps.get_runtime_session

    def _altered_diagnostics(runtime_session_id: str):
        diagnostics = original_get_runtime_session(runtime_session_id)
        for event in diagnostics.events:
            if (
                event.event_type == "evidence_plan_compiled"
                and event.event_payload_json.get("plan_id")
                == scope["evidence_plan_id"]
            ):
                if event_mutation == "missing_digest":
                    event.event_payload_json.pop("acquisition_premise_digest")
                elif event_mutation == "malformed_digest":
                    event.event_payload_json["acquisition_premise_digest"] = (
                        "sha256:not-a-digest"
                    )
                else:
                    event.event_payload_json["task_shape"] = (
                        "cross_source_comparison"
                    )
        return diagnostics

    monkeypatch.setattr(
        evidence_next_steps,
        "get_runtime_session",
        _altered_diagnostics,
    )
    response = _select(_next_step_payload(scope, evaluation))

    assert response.status_code == 400
    assert response.json() == {"detail": "current_acquisition_premise_mismatch"}
    assert _selection_events(scope["runtime_session_id"]) == []


def test_mismatched_turn_and_runtime_scope_fail_boundedly():
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("missing", "targeted_evidence")],
        [],
    )
    completion = client.post(
        "/v1/runtime/turns/complete",
        json={
            "request_id": f"complete-{scope['request_id']}",
            "runtime_session_id": scope["runtime_session_id"],
            "runtime_turn_id": scope["runtime_turn_id"],
            "turn_status": "completed",
        },
    )
    assert completion.status_code == 200
    next_turn = client.post(
        "/v1/runtime/turns/start",
        json={
            "request_id": f"new-turn-{scope['request_id']}",
            "owner_id": scope["owner_id"],
            "conversation_id": scope["conversation_id"],
            "surface": scope["surface"],
        },
    ).json()
    wrong_turn = next_turn["runtime_turn"]["runtime_turn_id"]
    turn_response = _select(
        _next_step_payload(
            scope,
            evaluation,
            runtime_turn_id=wrong_turn,
        )
    )
    owner_response = _select(
        _next_step_payload(
            scope,
            evaluation,
            owner_id="wrong-owner",
        )
    )

    assert turn_response.status_code == 400
    assert turn_response.json() == {
        "detail": "evidence_sufficiency_association_mismatch"
    }
    assert owner_response.status_code == 400
    assert owner_response.json() == {"detail": "runtime_session_mismatch"}


def test_tampered_evaluated_requirements_fail_digest_association():
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("missing", "targeted_evidence")],
        [],
    )
    payload = _next_step_payload(scope, evaluation)
    payload["evaluated_requirements"][0]["effective_outcome"] = "satisfied"
    response = _select(payload)

    assert response.status_code == 400
    assert response.json() == {"detail": "evaluated_requirements_mismatch"}
    assert _selection_events(scope["runtime_session_id"]) == []


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "sufficiency_status",
        "reason_codes",
        "answer_constraints",
        "provider_disposition",
        "selected_next_step",
        "conclusion_disposition",
        "advisory_allowed",
        "low_risk",
    ],
)
def test_caller_selected_policy_fields_are_rejected(forbidden_field: str):
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("missing", "targeted_evidence")],
        [],
    )
    payload = _next_step_payload(scope, evaluation)
    payload[forbidden_field] = "caller-selected"

    response = _select(payload)

    assert response.status_code == 422


def test_conflicting_inputs_and_private_premise_fields_are_rejected():
    scope = _start_runtime()
    evaluation = _evaluate(
        scope,
        [_requirement("missing", "targeted_evidence")],
        [],
    )
    conflicting = _next_step_payload(
        scope,
        evaluation,
        proposed_acquisition_premise=_premise(question="Changed"),
        clarification_target="source_scope",
    )
    private = _next_step_payload(scope, evaluation)
    private["current_premise"]["provider_text"] = "private provider text"

    assert _select(conflicting).status_code == 422
    assert _select(private).status_code == 422


def test_event_is_structural_private_and_deterministic_for_reordered_inputs():
    planned_premise = _premise(
        source_categories=["private-category", "records"],
        source_inventory=[
            _source(
                "private-source",
                categories=["records", "private-category"],
            )
        ],
    )
    scope = _start_runtime(premise=planned_premise)
    evaluation = _evaluate(
        scope,
        [
            _requirement("scope", "complete_scope_coverage"),
            _requirement("delivery", "context_delivery"),
        ],
        [_fact("scope", "failed"), _fact("delivery", "satisfied")],
    )
    first_premise = copy.deepcopy(scope["_current_premise"])
    second_premise = copy.deepcopy(first_premise)
    second_premise["declared_scope"]["source_categories"].reverse()
    second_premise["source_inventory"][0]["source_categories"].reverse()
    second_premise["source_inventory"][0]["capabilities"].reverse()
    first_payload = _next_step_payload(
        scope,
        evaluation,
        current_premise=first_premise,
    )
    second_payload = copy.deepcopy(first_payload)
    second_payload["evaluated_requirements"].reverse()
    second_payload["current_premise"] = second_premise

    first = _select(first_payload)
    second = _select(second_payload)

    assert first.status_code == 200
    assert first.json() == second.json()
    events = _selection_events(scope["runtime_session_id"])
    assert len(events) == 1
    payload = events[0]["event_payload_json"]
    assert set(payload) == {
        "request_id",
        "runtime_session_id",
        "runtime_turn_id",
        "selection_id",
        "evaluation_id",
        "evidence_plan_id",
        "acquisition_manifest_id",
        "task_shape",
        "sufficiency_status",
        "selected_next_step",
        "conclusion_disposition",
        "provider_disposition",
        "current_premise_digest",
        "proposed_premise_digest",
        "reacquisition_guard",
        "clarification_target",
        "unresolved_material_requirement_count",
        "reason_codes",
    }
    serialized = json.dumps(payload, sort_keys=True).lower()
    for excluded in (
        "private-source",
        "private-category",
        "source_ref",
        "source_inventory",
        "provider text",
        "source content",
        "https://",
        "/private/path",
        "credential",
    ):
        assert excluded not in serialized
