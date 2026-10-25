from __future__ import annotations

import copy
import json
from hashlib import sha256
from itertools import count

import pytest
from fastapi.testclient import TestClient
from main import app
from models import (
    EvidenceAcquisitionPremise,
    EvidenceNextStepResult,
    EvidenceTerminalEvaluateResponseV2,
)
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
    interaction_kind: str | None = None,
    shape_context: dict[str, object] | None = None,
    include_governance: bool = True,
    governance_text: str | None = None,
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
        governance_result: dict[str, object] | None = None
        if include_governance:
            governance_response = client.post(
                "/v1/runtime/interaction-governance/evaluate",
                json={
                    "request_id": f"governance-{ordinal}",
                    "owner_id": scope["owner_id"],
                    "conversation_id": scope["conversation_id"],
                    "surface": scope["surface"],
                    "runtime_session_id": scope["runtime_session_id"],
                    "runtime_turn_id": scope["runtime_turn_id"],
                    "current_user_text": governance_text or question_anchor,
                    "recent_messages": [],
                },
            )
            assert governance_response.status_code == 200
            governance_result = governance_response.json()["result"]
            scope["_governance_result"] = governance_result
        shape_interaction_kind = interaction_kind or (
            governance_result["interaction_kind"]
            if governance_result is not None
            else "question"
        )
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
                "interaction_kind": shape_interaction_kind,
                "task_context": shape_context or _shape_context(),
            },
        )
        assert shape_response.status_code == 200
        shape_result = shape_response.json()["result"]
        assert shape_result["question_anchor"] == question_anchor
        assert shape_result["task_shape"] == planned_premise["task_shape"]
        scope["_shape_result"] = shape_result
    plan_payload = {
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
    }
    if "aggregate_spec" in planned_premise:
        plan_payload["aggregate_spec"] = planned_premise["aggregate_spec"]
    plan_response = client.post(
        "/v1/runtime/evidence-plans/compile",
        json=plan_payload,
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
    content_fields: list[str] | None = None,
) -> dict[str, object]:
    source: dict[str, object] = {
        "source_id": source_id,
        "source_categories": categories or ["records"],
        "capabilities": capabilities or [
            "targeted_retrieval",
            "context_expansion",
        ],
        "availability": availability,
        "authority_role": authority_role,
    }
    if content_fields is not None:
        source["content_fields"] = content_fields
    return source


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
    aggregate_spec: dict[str, str] | None = None,
) -> dict[str, object]:
    premise: dict[str, object] = {
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
    if aggregate_spec is not None:
        premise["aggregate_spec"] = aggregate_spec
    return premise


def _premise_digest(premise: dict[str, object]) -> str:
    model = EvidenceAcquisitionPremise.model_validate(premise)
    return evidence_acquisition_premise_digest(
        question_anchor_digest=model.question_anchor_digest,
        task_shape=model.task_shape,
        declared_scope=model.declared_scope,
        source_inventory=model.source_inventory,
        selected_strategies=model.selected_strategies,
        aggregate_spec=model.aggregate_spec,
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


def _govern(
    scope: dict[str, object],
    *,
    current_user_text: str,
    request_suffix: str,
):
    return client.post(
        "/v1/runtime/interaction-governance/evaluate",
        json={
            "request_id": f"governance-{request_suffix}-{scope['request_id']}",
            "owner_id": scope["owner_id"],
            "conversation_id": scope["conversation_id"],
            "surface": scope["surface"],
            "runtime_session_id": scope["runtime_session_id"],
            "runtime_turn_id": scope["runtime_turn_id"],
            "current_user_text": current_user_text,
            "recent_messages": [],
        },
    )


def _selection_events(runtime_session_id: object) -> list[dict[str, object]]:
    response = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert response.status_code == 200
    return [
        event
        for event in response.json()["events"]
        if event["event_type"] == "evidence_next_step_selected"
    ]


def _governance_events(runtime_session_id: object) -> list[dict[str, object]]:
    response = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert response.status_code == 200
    return [
        event
        for event in response.json()["events"]
        if event["event_type"] == "interaction_governance_evaluated"
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
    assert scope["_governance_result"]["interaction_kind"] == "question"
    assert scope["_shape_result"]["task_shape"] == "targeted_lookup"
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
    assert scope["_governance_result"]["interaction_kind"] == "question"
    assert scope["_shape_result"]["task_shape"] == "targeted_lookup"
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
    governance_events = _governance_events(scope["runtime_session_id"])
    assert len(governance_events) == 1
    assert governance_events[0]["event_payload_json"]["interaction_kind"] == (
        "question"
    )
    assert question.lower() not in json.dumps(
        governance_events[0]["event_payload_json"],
        sort_keys=True,
    ).lower()


def test_associated_high_stakes_targeted_gap_keeps_provider_blocked():
    question = "Can this adapter be used with that device?"
    scope = _start_runtime(
        question_anchor=question,
        derive_shape=True,
        shape_context=_shape_context(
            high_stakes_accuracy_required=True,
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


def test_matching_high_impact_governance_keeps_provider_blocked():
    question = "Does this payroll module support version 3.14?"
    scope = _start_runtime(question_anchor=question, derive_shape=True)

    assert scope["_governance_result"]["interaction_kind"] == (
        "high_impact_decision"
    )
    assert scope["_shape_result"]["task_shape"] == "targeted_lookup"
    evaluation = _evaluate(
        scope,
        [_requirement("targeted", "targeted_evidence")],
        [_fact("targeted", "failed")],
    )

    result = _select(_next_step_payload(scope, evaluation)).json()["result"]

    assert result["selected_next_step"] == "withhold_unsupported_conclusion"
    assert result["conclusion_disposition"] == "requested_conclusion_withheld"
    assert result["provider_disposition"] == "blocked"


def test_caller_cannot_downgrade_high_impact_governance_for_advisory_authority():
    question = "Does this payroll module support version 3.14?"
    scope = _start_runtime(
        question_anchor=question,
        derive_shape=True,
        interaction_kind="question",
    )

    assert scope["_governance_result"]["interaction_kind"] == (
        "high_impact_decision"
    )
    assert scope["_shape_result"]["task_shape"] == "targeted_lookup"
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


def test_absent_associated_governance_event_keeps_provider_blocked():
    question = "Does this package version support Python 3.14?"
    scope = _start_runtime(
        question_anchor=question,
        derive_shape=True,
        include_governance=False,
    )
    evaluation = _evaluate(
        scope,
        [_requirement("targeted", "targeted_evidence")],
        [_fact("targeted", "failed")],
    )

    result = _select(_next_step_payload(scope, evaluation)).json()["result"]

    assert result["selected_next_step"] == "withhold_unsupported_conclusion"
    assert result["provider_disposition"] == "blocked"


def test_governance_event_after_shape_and_plan_keeps_provider_blocked():
    question = "Does this package version support Python 3.14?"
    scope = _start_runtime(
        question_anchor=question,
        derive_shape=True,
        include_governance=False,
    )
    late_governance = _govern(
        scope,
        current_user_text=question,
        request_suffix="late",
    )
    assert late_governance.status_code == 200
    assert late_governance.json()["result"]["interaction_kind"] == "question"
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


@pytest.mark.parametrize("event_mutation", ["duplicate", "malformed"])
def test_duplicate_or_malformed_governance_event_keeps_provider_blocked(
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
            if event.event_type != "interaction_governance_evaluated":
                continue
            if event_mutation == "duplicate":
                duplicate = event.model_copy(deep=True)
                duplicate.event_id = f"{event.event_id}-duplicate"
                diagnostics.events.insert(index + 1, duplicate)
            else:
                event.event_payload_json["interaction_kind"] = "not-a-kind"
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
        "policy_version",
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


def _aggregate_premise(
    *,
    function: str = "median",
    field_name: str = "Fuel (L)",
) -> dict[str, object]:
    return _premise(
        question="Calculate the requested statistic.",
        task_shape="aggregate",
        source_ids=["source-a"],
        source_inventory=[
            _source(
                capabilities=["context_expansion"],
                content_fields=["Fuel (L)", "Odometer"],
            )
        ],
        selected_strategies=["structured_field_values"],
        aggregate_spec={"function": function, "field_name": field_name},
    )


def _aggregate_requirements() -> list[dict[str, str]]:
    return [
        _requirement("aggregate-complete-scope", "complete_scope_coverage"),
        _requirement("aggregate-context-delivery", "context_delivery"),
        _requirement("aggregate-no-truncation", "no_material_truncation"),
    ]


def test_sufficient_aggregate_selects_bounded_conclusion_permission():
    scope = _start_runtime(
        question_anchor="Calculate the requested statistic.",
        premise=_aggregate_premise(),
    )
    requirements = _aggregate_requirements()
    evaluation = _evaluate(
        scope,
        requirements,
        [
            _fact(requirement["requirement_id"], "satisfied")
            for requirement in requirements
        ],
        task_shape="aggregate",
    )
    response = _select(_next_step_payload(scope, evaluation))

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["task_shape"] == "aggregate"
    assert result["sufficiency_status"] == "sufficient_for_declared_scope"
    assert result["selected_next_step"] == "answer_within_declared_scope"
    assert result["conclusion_disposition"] == "bounded_conclusion_allowed"
    assert result["provider_disposition"] == "allowed"
    plan_event = _plan_events(scope["runtime_session_id"])[0]["event_payload_json"]
    assert result["current_premise_digest"] == plan_event[
        "acquisition_premise_digest"
    ]


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [("failed", "insufficient"), ("unknown", "unknown")],
)
def test_incomplete_aggregate_withholds_and_never_gets_targeted_advisory(
    outcome: str,
    expected_status: str,
):
    scope = _start_runtime(
        question_anchor="Calculate the requested statistic.",
        premise=_aggregate_premise(),
    )
    requirements = _aggregate_requirements()
    facts = [
        _fact(
            requirement["requirement_id"],
            outcome
            if requirement["requirement_id"] == "aggregate-complete-scope"
            else "satisfied",
        )
        for requirement in requirements
    ]
    evaluation = _evaluate(
        scope,
        requirements,
        facts,
        task_shape="aggregate",
    )
    response = _select(_next_step_payload(scope, evaluation))

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["sufficiency_status"] == expected_status
    assert result["selected_next_step"] in {
        "disclose_unexamined_scope",
        "withhold_unsupported_conclusion",
    }
    assert result["conclusion_disposition"] == "requested_conclusion_withheld"
    assert result["provider_disposition"] == "blocked"


def test_aggregate_premise_digest_binds_function_and_exact_field():
    median_fuel = _premise_digest(_aggregate_premise())
    mean_fuel = _premise_digest(_aggregate_premise(function="mean"))
    median_odometer = _premise_digest(
        _aggregate_premise(field_name="Odometer")
    )

    assert median_fuel != mean_fuel
    assert median_fuel != median_odometer
    assert median_fuel == _premise_digest(_aggregate_premise())


@pytest.fixture(autouse=True)
def evidence_calls_stay_local(monkeypatch: pytest.MonkeyPatch):
    import socket

    def forbidden_network(*args, **kwargs):
        pytest.fail("Evidence decisions must not contact another service")

    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden_network)


def _terminal_payload(scope, requirements=None, facts=None, **overrides):
    payload = {key: value for key, value in scope.items() if not key.startswith("_")}
    payload.update(
        task_shape=scope["_current_premise"]["task_shape"],
        declared_requirements=(
            requirements
            if requirements is not None
            else [_requirement("material", "targeted_evidence")]
        ),
        acquisition_facts=facts if facts is not None else [],
        current_premise=copy.deepcopy(scope["_current_premise"]),
    )
    payload.update(overrides)
    return payload


def _terminal(payload, version="v1"):
    return client.post(f"/{version}/runtime/evidence-terminal/evaluate", json=payload)


def _sequential(payload):
    sufficiency = client.post(
        "/v1/runtime/evidence-sufficiency/evaluate",
        json={
            key: value
            for key, value in payload.items()
            if key
            not in {
                "current_premise",
                "proposed_acquisition_premise",
                "clarification_target",
            }
        },
    )
    if sufficiency.status_code != 200:
        return sufficiency, None
    evaluation = sufficiency.json()
    next_step = _select(
        {
            **{
                key: payload[key]
                for key in (
                    "request_id",
                    "owner_id",
                    "conversation_id",
                    "surface",
                    "runtime_session_id",
                    "runtime_turn_id",
                    "evidence_plan_id",
                    "acquisition_manifest_id",
                    "current_premise",
                )
            },
            "evaluation_id": evaluation["result"]["evaluation_id"],
            "evaluated_requirements": evaluation["result"]["evaluated_requirements"],
            **{
                key: payload[key]
                for key in (
                    "proposed_acquisition_premise",
                    "clarification_target",
                )
                if key in payload
            },
        }
    )
    return sufficiency, next_step


def _diagnostics(scope):
    response = client.get(f"/v1/runtime/sessions/{scope['runtime_session_id']}")
    assert response.status_code == 200
    return response.json()


def _assert_v2_event_references(response, scope):
    assert EvidenceTerminalEvaluateResponseV2.model_validate(response).model_dump(
        mode="json"
    ) == response
    events = _diagnostics(scope)["events"]
    for reference in (
        response["sufficiency_provenance"]["persisted_event"],
        response["next_step_provenance"]["persisted_event"],
        response["next_step_provenance"]["consulted_sufficiency_event"],
    ):
        matches = [event for event in events if event["event_id"] == reference["event_id"]]
        assert len(matches) == 1
        event = matches[0]
        payload = event["event_payload_json"]
        assert reference == {
            "event_id": event["event_id"],
            "event_type": event["event_type"],
            **{key: payload[key] for key in (
                "request_id", "runtime_session_id", "runtime_turn_id",
                "evidence_plan_id", "acquisition_manifest_id", "evaluation_id",
            )},
            "policy_version": payload.get("policy_version", "UNKNOWN"),
            **({"selection_id": payload["selection_id"]}
               if event["event_type"] == "evidence_next_step_selected" else {}),
        }
        assert event["runtime_session_id"] == scope["runtime_session_id"]
        assert event["runtime_turn_id"] == scope["runtime_turn_id"]


def _assert_terminal_parity(payload, scope):
    before = _diagnostics(scope)
    response = _terminal(payload)
    assert response.status_code == 200, response.text
    composite = response.json()
    assert set(composite) == {"schema_version", "sufficiency", "next_step"}
    assert composite["schema_version"] == "evidence-terminal.v1"
    after = _diagnostics(scope)
    new_events = after["events"][len(before["events"]) :]
    assert [event["event_type"] for event in new_events] == [
        "evidence_sufficiency_evaluated",
        "evidence_next_step_selected",
    ]
    assert new_events[0]["event_id"] != new_events[1]["event_id"]
    assert after["latest_turn"] == before["latest_turn"]
    for key in ("owner_id", "conversation_id", "surface", "runtime_session_id"):
        assert after["runtime_session"][key] == payload[key]
    sufficiency = composite["sufficiency"]
    selection = composite["next_step"]
    for key in (
        "request_id",
        "owner_id",
        "conversation_id",
        "surface",
        "runtime_session_id",
        "runtime_turn_id",
    ):
        assert sufficiency[key] == selection[key] == payload[key]
    for key in ("evidence_plan_id", "acquisition_manifest_id"):
        assert sufficiency[key] == selection["result"][key] == payload[key]
    evaluation = sufficiency["result"]
    result = selection["result"]
    assert result["evaluation_id"] == evaluation["evaluation_id"]
    assert result["selection_id"] != evaluation["evaluation_id"]
    assert result["sufficiency_status"] == evaluation["sufficiency_status"]
    assert result["task_shape"] == evaluation["task_shape"] == payload["task_shape"]
    for event in new_events:
        assert event["runtime_session_id"] == payload["runtime_session_id"]
        assert event["runtime_turn_id"] == payload["runtime_turn_id"]
        association = event["event_payload_json"]
        for key in (
            "request_id",
            "runtime_session_id",
            "runtime_turn_id",
            "evidence_plan_id",
            "acquisition_manifest_id",
        ):
            assert association[key] == payload[key]
        assert association["evaluation_id"] == evaluation["evaluation_id"]
    evaluation_event = new_events[0]["event_payload_json"]
    selection_event = new_events[1]["event_payload_json"]
    assert evaluation_event["policy_version"] == "evidence-sufficiency-policy.v1"
    assert selection_event["policy_version"] == "evidence-next-step-policy.v1"
    assert len({evaluation_event["policy_version"], selection_event["policy_version"],
                composite["schema_version"]}) == 3
    # Join each returned identity to its persisted event using the exact scope.
    for event_type, id_field, decision in (
        ("evidence_sufficiency_evaluated", "evaluation_id", evaluation),
        ("evidence_next_step_selected", "selection_id", result),
    ):
        matches = [event for event in after["events"]
                   if event["event_type"] == event_type
                   and event["runtime_session_id"] == payload["runtime_session_id"]
                   and event["runtime_turn_id"] == payload["runtime_turn_id"]
                   and event["event_payload_json"].get(id_field) == decision[id_field]]
        assert len(matches) == 1
        assert matches[0] in new_events
    assert "policy_version" not in json.dumps(composite)
    assert evaluation_event["reason_codes"] == evaluation["reason_codes"]
    assert evaluation_event["answer_constraints"] == evaluation["answer_constraints"]
    # Independently compute the wire digest rather than calling the owning helper.
    encoded = json.dumps(
        evaluation["evaluated_requirements"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert evaluation_event["evaluated_requirements_digest"] == (
        f"sha256:{sha256(encoded.encode('utf-8')).hexdigest()}"
    )
    for key in (
        "selection_id",
        "reason_codes",
        "provider_disposition",
        "conclusion_disposition",
        "current_premise_digest",
        "proposed_premise_digest",
        "selected_next_step",
        "reacquisition_guard",
    ):
        assert selection_event[key] == result[key]
    assert (
        result["current_premise_digest"]
        == _plan_events(scope["runtime_session_id"])[0]["event_payload_json"][
            "acquisition_premise_digest"
        ]
    )
    original_evaluation, original_selection = _sequential(payload)
    assert original_evaluation.status_code == original_selection.status_code == 200
    assert sufficiency == original_evaluation.json()
    assert selection == original_selection.json()
    final_events = _diagnostics(scope)["events"]
    # The individual evaluator appends; the individual selector deduplicates.
    assert len(final_events) == len(after["events"]) + 1
    assert final_events[-1]["event_type"] == "evidence_sufficiency_evaluated"
    assert final_events[-1]["event_payload_json"] == evaluation_event
    assert final_events[-1]["event_id"] != new_events[0]["event_id"]
    # Exercise v2 against every independently checked policy outcome in this
    # matrix, retaining the complete decisions from the original endpoints.
    v2 = _terminal(payload, "v2")
    assert v2.status_code == 200, v2.text
    assert v2.json()["sufficiency"] == sufficiency
    assert v2.json()["next_step"] == selection
    assert v2.json()["schema_version"] == "evidence-terminal.v2"
    _assert_v2_event_references(v2.json(), scope)
    provenance = v2.json()["next_step_provenance"]
    assert provenance["persistence"] == "reused"
    assert provenance["consulted_sufficiency_event"]["event_id"] == new_events[0]["event_id"]
    assert provenance["persisted_event"]["event_id"] == new_events[1]["event_id"]
    assert len(_diagnostics(scope)["events"]) == len(final_events) + 1
    return composite


@pytest.mark.parametrize(
    "outcome,expected_status,expected_step,expected_reasons,expected_constraints",
    [
        (
            "satisfied",
            "sufficient_for_declared_scope",
            "answer_within_declared_scope",
            ["all_declared_requirements_satisfied"],
            [],
        ),
        (
            "limited",
            "sufficient_with_limitations",
            "provide_qualified_partial_answer",
            ["optional_requirement_incomplete"],
            ["qualify_conclusion", "disclose_limitations", "identify_unexamined_scope"],
        ),
        (
            "failed",
            "insufficient",
            "withhold_unsupported_conclusion",
            ["material_requirement_not_satisfied"],
            [
                "qualify_conclusion",
                "disclose_limitations",
                "identify_unexamined_scope",
                "additional_acquisition_or_clarification_required",
                "withhold_unqualified_conclusion",
            ],
        ),
        (
            "unknown",
            "unknown",
            "withhold_unsupported_conclusion",
            ["material_requirement_unknown"],
            [
                "qualify_conclusion",
                "disclose_limitations",
                "identify_unexamined_scope",
                "additional_acquisition_or_clarification_required",
                "withhold_unqualified_conclusion",
            ],
        ),
        (
            "missing",
            "unknown",
            "withhold_unsupported_conclusion",
            ["material_requirement_missing"],
            [
                "qualify_conclusion",
                "disclose_limitations",
                "identify_unexamined_scope",
                "additional_acquisition_or_clarification_required",
                "withhold_unqualified_conclusion",
            ],
        ),
    ],
)
def test_terminal_exact_sequential_parity_and_event_provenance(
    outcome,
    expected_status,
    expected_step,
    expected_reasons,
    expected_constraints,
):
    scope = _start_runtime()
    payload = _terminal_payload(
        scope,
        [
            _requirement("material", "targeted_evidence"),
            _requirement("optional", "freshness", criticality="optional"),
        ],
        (
            [_fact("material", "satisfied"), _fact("optional", "unavailable")]
            if outcome == "limited"
            else [_fact("optional", "satisfied")]
            + ([] if outcome == "missing" else [_fact("material", outcome)])
        ),
    )
    composite = _assert_terminal_parity(payload, scope)
    evaluation = composite["sufficiency"]["result"]
    result = composite["next_step"]["result"]
    assert evaluation["sufficiency_status"] == expected_status
    assert evaluation["reason_codes"] == expected_reasons
    assert evaluation["answer_constraints"] == expected_constraints
    assert result["selected_next_step"] == expected_step
    assert result["provider_disposition"] == (
        "allowed" if outcome in {"satisfied", "limited"} else "blocked"
    )
    assert result["reason_codes"] == {
        "satisfied": ["declared_scope_sufficient"],
        "limited": ["optional_limitations_remain"],
    }.get(outcome, ["unsupported_conclusion_withheld"])


@pytest.mark.parametrize("mode", ["clarify", "changed", "unchanged", "partial"])
def test_terminal_clarification_reacquisition_and_partial_answer_parity(mode):
    scope = _start_runtime()
    payload = _terminal_payload(scope)
    if mode == "clarify":
        payload["clarification_target"] = "source_scope"
    elif mode in {"changed", "unchanged"}:
        payload["proposed_acquisition_premise"] = (
            _mutated_premise(scope["_current_premise"], "question")
            if mode == "changed"
            else scope["_current_premise"]
        )
    else:
        payload["declared_requirements"] += [_requirement("delivery", "context_delivery")]
        payload["acquisition_facts"] = [
            _fact("material", "partial"),
            _fact("delivery", "satisfied"),
        ]
    composite = _assert_terminal_parity(payload, scope)
    result = composite["next_step"]["result"]
    assert (
        result["selected_next_step"]
        == {
            "clarify": "ask_narrow_clarification",
            "changed": "perform_additional_acquisition",
            "unchanged": "withhold_unsupported_conclusion",
            "partial": "provide_qualified_partial_answer",
        }[mode]
    )
    assert result["provider_disposition"] == ("allowed" if mode == "partial" else "blocked")
    assert result["reacquisition_guard"] == {
        "changed": "changed_premise_allowed",
        "unchanged": "unchanged_premise_blocked",
    }.get(mode, "not_applicable")


@pytest.mark.parametrize("outcome", ["satisfied", "failed", "unknown"])
def test_terminal_aggregate_parity(outcome):
    scope = _start_runtime(
        question_anchor="Calculate the requested statistic.", premise=_aggregate_premise()
    )
    requirements = _aggregate_requirements()
    payload = _terminal_payload(
        scope, requirements, [_fact(item["requirement_id"], outcome) for item in requirements]
    )
    result = _assert_terminal_parity(payload, scope)["next_step"]["result"]
    assert result["provider_disposition"] == ("allowed" if outcome == "satisfied" else "blocked")
    assert result["selected_next_step"] == (
        "answer_within_declared_scope" if outcome == "satisfied" else "disclose_unexamined_scope"
    )


@pytest.mark.parametrize("high_stakes,high_impact", [(False, False), (True, False), (False, True)])
def test_terminal_advisory_authority_parity(high_stakes, high_impact):
    question = (
        "Does this payroll module support version 3.14?"
        if high_impact else "Does this package version support Python 3.14?"
    )
    scope = _start_runtime(
        question_anchor=question,
        derive_shape=True,
        shape_context=_shape_context(high_stakes_accuracy_required=high_stakes),
    )
    assert scope["_governance_result"]["interaction_kind"] == (
        "high_impact_decision" if high_impact else "question"
    )
    composite = _assert_terminal_parity(
        _terminal_payload(scope, facts=[_fact("material", "failed")]),
        scope,
    )
    result = composite["next_step"]["result"]
    assert result["selected_next_step"] == "withhold_unsupported_conclusion"
    assert result["conclusion_disposition"] == "requested_conclusion_withheld"
    assert result["provider_disposition"] == (
        "blocked" if high_stakes or high_impact else "allowed"
    )


@pytest.mark.parametrize(
    "task_shape,kind,constraint",
    [
        ("bounded_exhaustive_review", "complete_scope_coverage", "withhold_exhaustive_conclusion"),
        ("absence_or_coverage_check", "structured_absence_check", "withhold_absence_conclusion"),
        (
            "contradiction_review",
            "contradiction_search",
            "withhold_contradiction_sensitive_conclusion",
        ),
    ],
)
def test_terminal_scope_sensitive_restrictions(task_shape, kind, constraint):
    scope = _start_runtime(premise=_premise(task_shape=task_shape))
    payload = _terminal_payload(scope, [_requirement("scope", kind)], [])
    composite = _assert_terminal_parity(payload, scope)
    assert constraint in composite["sufficiency"]["result"]["answer_constraints"]
    assert composite["next_step"]["result"]["provider_disposition"] == "blocked"
    assert composite["next_step"]["result"]["selected_next_step"] == "disclose_unexamined_scope"


@pytest.mark.parametrize(
    "field",
    [
        "evaluation_id",
        "evaluated_requirements",
        "sufficiency_status",
        "answer_constraints",
        "selected_next_step",
        "provider_disposition",
        "conclusion_disposition",
        "reason_codes",
        "policy_version",
        "schema_version",
        "advisory_allowed",
        "unknown_field",
    ],
)
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_terminal_rejects_caller_authority(field, version):
    scope = _start_runtime()
    before = _diagnostics(scope)
    response = _terminal(_terminal_payload(scope, **{field: "forged"}), version)
    assert response.status_code == 422
    assert _diagnostics(scope) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_context",
        "empty_requirements",
        "too_many_requirements",
        "duplicate_requirement",
        "undeclared_fact",
        "duplicate_fact",
        "bad_outcome",
        "bad_shape",
        "numeric_owner",
        "nested_authority",
        "conflicting_hints",
        "bad_clarification",
        "bad_digest",
    ],
)
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_terminal_rejects_malformed_inputs_before_events(mutation, version):
    scope = _start_runtime()
    payload = _terminal_payload(scope)
    if mutation == "missing_context":
        payload.pop("runtime_turn_id")
    elif mutation == "empty_requirements":
        payload["declared_requirements"] = []
    elif mutation == "too_many_requirements":
        payload["declared_requirements"] = [
            _requirement(f"item-{i}", "targeted_evidence") for i in range(33)
        ]
    elif mutation == "duplicate_requirement":
        payload["declared_requirements"] *= 2
    elif mutation == "undeclared_fact":
        payload["acquisition_facts"] = [_fact("undeclared", "satisfied")]
    elif mutation == "duplicate_fact":
        payload["acquisition_facts"] = [_fact("material", "satisfied")] * 2
    elif mutation == "bad_outcome":
        payload["acquisition_facts"] = [_fact("material", "authorized")]
    elif mutation == "bad_shape":
        payload["task_shape"] = "unknown"
    elif mutation == "numeric_owner":
        payload["owner_id"] = 42
    elif mutation == "nested_authority":
        payload["current_premise"]["provider_disposition"] = "allowed"
    elif mutation == "conflicting_hints":
        payload["clarification_target"] = "source_scope"
        payload["proposed_acquisition_premise"] = payload["current_premise"]
    elif mutation == "bad_clarification":
        payload["clarification_target"] = "anything"
    else:
        payload["current_premise"]["question_anchor_digest"] = "sha256:forged"
    before = _diagnostics(scope)
    assert _terminal(payload, version).status_code == 422
    assert _diagnostics(scope) == before


@pytest.mark.parametrize(
    "field,value,status,detail",
    [
        ("owner_id", "wrong-owner", 400, "runtime_session_mismatch"),
        ("conversation_id", "wrong-conversation", 400, "runtime_session_mismatch"),
        ("surface", "telegram", 400, "runtime_session_mismatch"),
        ("runtime_session_id", "missing-session", 404, "runtime_session_not_found"),
        ("runtime_turn_id", "missing-turn", 404, "runtime_turn_not_found"),
        ("runtime_turn_id", "other-turn", 400, "runtime_turn_session_mismatch"),
        ("runtime_session_id", "other-session", 400, "runtime_session_mismatch"),
        ("evidence_plan_id", "wrong-plan", 400, "current_acquisition_premise_mismatch"),
    ],
)
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_terminal_context_failure_matches_individual_calls(field, value, status, detail, version):
    scope = _start_runtime()
    other = _start_runtime()
    if value == "other-turn":
        value = other["runtime_turn_id"]
    elif value == "other-session":
        value = other["runtime_session_id"]
    payload = _terminal_payload(scope, **{field: value})
    response = _terminal(payload, version)
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert _selection_events(scope["runtime_session_id"]) == []
    events = _diagnostics(scope)["events"]
    sufficiency_events = [e for e in events if e["event_type"] == "evidence_sufficiency_evaluated"]
    assert len(sufficiency_events) == (1 if field == "evidence_plan_id" else 0)
    evaluation, selection = _sequential(payload)
    failed = selection if selection is not None else evaluation
    assert failed.status_code == response.status_code
    assert failed.json() == response.json()


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
        "strategies",
    ],
)
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_terminal_changed_current_premise_fails_without_partial_success(mutation, version):
    scope = _start_runtime()
    payload = _terminal_payload(
        scope,
        current_premise=_mutated_premise(
            scope["_current_premise"],
            mutation,
        ),
    )
    response = _terminal(payload, version)
    assert response.status_code == 400
    assert response.json() == {"detail": "current_acquisition_premise_mismatch"}
    events = _diagnostics(scope)["events"]
    assert events[-1]["event_type"] == "evidence_sufficiency_evaluated"
    assert _selection_events(scope["runtime_session_id"]) == []
    _, selection = _sequential(payload)
    assert selection.status_code == response.status_code
    assert selection.json() == response.json()


@pytest.mark.parametrize(
    "event_type,mutation,status,detail",
    [
        (
            "evidence_sufficiency_evaluated",
            "missing",
            404,
            "evidence_sufficiency_evaluation_not_found",
        ),
        ("evidence_sufficiency_evaluated", "status", 400, "evidence_sufficiency_event_invalid"),
        (
            "evidence_sufficiency_evaluated",
            "constraints",
            400,
            "evidence_sufficiency_event_invalid",
        ),
        ("evidence_sufficiency_evaluated", "digest", 400, "evaluated_requirements_mismatch"),
        (
            "evidence_sufficiency_evaluated",
            "session",
            400,
            "evidence_sufficiency_association_mismatch",
        ),
        (
            "evidence_sufficiency_evaluated",
            "turn",
            400,
            "evidence_sufficiency_association_mismatch",
        ),
        (
            "evidence_sufficiency_evaluated",
            "plan",
            400,
            "evidence_sufficiency_association_mismatch",
        ),
        (
            "evidence_sufficiency_evaluated",
            "manifest",
            400,
            "evidence_sufficiency_association_mismatch",
        ),
        ("evidence_plan_compiled", "missing", 400, "current_acquisition_premise_mismatch"),
        ("evidence_plan_compiled", "digest", 400, "current_acquisition_premise_mismatch"),
        ("evidence_shape_derived", "missing", 200, None),
        ("evidence_shape_derived", "malformed", 200, None),
        ("interaction_governance_evaluated", "missing", 200, None),
        ("interaction_governance_evaluated", "malformed", 200, None),
    ],
)
def test_terminal_authoritative_event_failures_preserve_conservative_behavior(
    event_type,
    mutation,
    status,
    detail,
    monkeypatch,
):
    question = "Does this package version support Python 3.14?"
    scope = _start_runtime(question_anchor=question, derive_shape=True)
    payload = _terminal_payload(scope, facts=[_fact("material", "failed")])
    original = evidence_next_steps.get_runtime_session
    observed = []

    def altered_diagnostics(session_id):
        diagnostics = original(session_id)
        # This read happens after sufficiency has durably recorded its event.
        assert diagnostics.events[-1].event_type == "evidence_sufficiency_evaluated"
        observed.append(diagnostics.events[-1].event_payload_json["evaluation_id"])
        if mutation == "missing":
            diagnostics.events = [e for e in diagnostics.events if e.event_type != event_type]
        else:
            for event in diagnostics.events:
                if event.event_type != event_type:
                    continue
                field, replacement = {
                    "status": ("sufficiency_status", "authorized"),
                    "constraints": ("answer_constraints", []),
                    "digest": ("evaluated_requirements_digest", f"sha256:{'0' * 64}"),
                    "session": ("runtime_session_id", "wrong-session"),
                    "turn": ("runtime_turn_id", "wrong-turn"),
                    "plan": ("evidence_plan_id", "wrong-plan"),
                    "manifest": ("acquisition_manifest_id", "wrong-manifest"),
                    "malformed": ("interaction_kind", "unknown-kind"),
                }[mutation]
                if event_type == "evidence_plan_compiled":
                    field, replacement = "acquisition_premise_digest", "not-a-digest"
                event.event_payload_json[field] = replacement
        return diagnostics

    monkeypatch.setattr(evidence_next_steps, "get_runtime_session", altered_diagnostics)
    response = _terminal(payload)
    assert response.status_code == status
    assert len(observed) == 1
    events = _diagnostics(scope)["events"]
    sufficiency_event = next(
        e for e in events if e["event_type"] == "evidence_sufficiency_evaluated"
    )
    assert sufficiency_event["event_payload_json"]["evaluation_id"] == observed[0]
    if status != 200:
        assert response.json() == {"detail": detail}
        assert _selection_events(scope["runtime_session_id"]) == []
    else:
        result = response.json()["next_step"]["result"]
        assert result["provider_disposition"] == "blocked"
        assert result["conclusion_disposition"] == "requested_conclusion_withheld"
    # Exercise the original endpoints under the same fault, independently.
    evaluation, selection = _sequential(payload)
    assert evaluation.status_code == 200
    assert selection.status_code == status
    if detail is not None:
        assert selection.json() == response.json()
    else:
        assert selection.json() == response.json()["next_step"]


@pytest.mark.parametrize("stage", ["sufficiency", "next_step"])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_terminal_unexpected_constituent_failure_is_not_success(stage, monkeypatch, version):
    from services import evidence_sufficiency

    scope = _start_runtime()
    before = _diagnostics(scope)
    original_record = evidence_sufficiency.record_runtime_event

    attempted_records = []

    def fail_record(**kwargs):
        attempted_records.append(copy.deepcopy(kwargs))
        raise RuntimeError("storage_unavailable")

    target = evidence_sufficiency if stage == "sufficiency" else evidence_next_steps
    monkeypatch.setattr(target, "record_runtime_event", fail_record)
    with pytest.raises(RuntimeError, match="storage_unavailable"):
        _terminal(_terminal_payload(scope), version)
    diagnostics = _diagnostics(scope)
    assert diagnostics["latest_turn"] == before["latest_turn"]
    events = diagnostics["events"]
    assert len([e for e in events if e["event_type"] == "evidence_sufficiency_evaluated"]) == (
        0 if stage == "sufficiency" else 1
    )
    assert _selection_events(scope["runtime_session_id"]) == []
    if stage == "next_step":
        retained = [event for event in events
                    if event["event_type"] == "evidence_sufficiency_evaluated"][0]
        payload = retained["event_payload_json"]
        assert payload["policy_version"] == "evidence-sufficiency-policy.v1"
        assert payload["runtime_session_id"] == retained["runtime_session_id"]
        assert payload["runtime_turn_id"] == retained["runtime_turn_id"]
        assert payload["evidence_plan_id"] == scope["evidence_plan_id"]
        assert payload["acquisition_manifest_id"] == scope["acquisition_manifest_id"]
        assert attempted_records[0]["event_type"] == "evidence_next_step_selected"
        attempted = attempted_records[0]["event_payload_json"]
        assert attempted["evaluation_id"] == payload["evaluation_id"]
        assert attempted["policy_version"] == "evidence-next-step-policy.v1"
        for key in ("runtime_session_id", "runtime_turn_id", "evidence_plan_id",
                    "acquisition_manifest_id"):
            assert attempted[key] == payload[key]
        assert "provider_disposition" not in payload
        assert "selection_id" not in payload
    if stage == "sufficiency":
        monkeypatch.setattr(target, "record_runtime_event", original_record)


def test_terminal_repeated_equivalent_requests_preserve_identity_and_events():
    scope = _start_runtime()
    payload = _terminal_payload(
        scope,
        [_requirement("z", "targeted_evidence"), _requirement("a", "context_delivery")],
        [_fact("z", "partial"), _fact("a", "satisfied")],
    )
    first = _terminal(payload)
    assert first.status_code == 200
    initial_events = _diagnostics(scope)["events"]
    reordered = copy.deepcopy(payload)
    reordered["declared_requirements"].reverse()
    reordered["acquisition_facts"].reverse()
    second = _terminal(reordered)
    assert second.json() == first.json()
    events = _diagnostics(scope)["events"]
    assert len(events) == len(initial_events) + 1
    assert events[-1]["event_type"] == "evidence_sufficiency_evaluated"
    assert events[-1]["event_payload_json"] == initial_events[-2]["event_payload_json"]
    assert len(_selection_events(scope["runtime_session_id"])) == 1
    evaluated = [event["event_payload_json"] for event in events
                 if event["event_type"] == "evidence_sufficiency_evaluated"]
    assert len(evaluated) == 2
    assert {event["policy_version"] for event in evaluated} == {"evidence-sufficiency-policy.v1"}
    selected = _selection_events(scope["runtime_session_id"])[0]["event_payload_json"]
    assert selected["policy_version"] == "evidence-next-step-policy.v1"
    # A new admitted turn must not reuse the prior plan or terminal decision.
    completion = client.post(
        "/v1/runtime/turns/complete",
        json={
            "request_id": "complete-repeat",
            "runtime_session_id": scope["runtime_session_id"],
            "runtime_turn_id": scope["runtime_turn_id"],
            "turn_status": "completed",
        },
    )
    assert completion.status_code == 200
    new_turn = client.post(
        "/v1/runtime/turns/start",
        json={
            "request_id": "start-repeat",
            "owner_id": scope["owner_id"],
            "conversation_id": scope["conversation_id"],
            "surface": scope["surface"],
        },
    ).json()["runtime_turn"]["runtime_turn_id"]
    payload["runtime_turn_id"] = new_turn
    response = _terminal(payload)
    assert response.status_code == 400
    assert response.json() == {"detail": "current_acquisition_premise_mismatch"}
    new_evaluation = _diagnostics(scope)["events"][-1]["event_payload_json"]
    assert new_evaluation["evaluation_id"] != first.json()["sufficiency"]["result"]["evaluation_id"]
    assert new_evaluation["runtime_turn_id"] == new_turn
    _, selection = _sequential(payload)
    assert selection.status_code == response.status_code
    assert selection.json() == response.json()
    assert len(_selection_events(scope["runtime_session_id"])) == 1


def test_terminal_reacquisition_guard_retains_individual_request_semantics():
    scope = _start_runtime()
    payload = _terminal_payload(
        scope,
        proposed_acquisition_premise=_mutated_premise(
            scope["_current_premise"],
            "question",
        ),
    )
    first = _terminal(payload)
    assert first.status_code == 200
    assert (
        first.json()["next_step"]["result"]["selected_next_step"]
        == "perform_additional_acquisition"
    )
    assert _terminal(payload).json() == first.json()
    payload["request_id"] = "another-request"
    second = _terminal(payload)
    assert second.status_code == 200
    result = second.json()["next_step"]["result"]
    assert result["reacquisition_guard"] == "premise_already_attempted"
    assert result["provider_disposition"] == "blocked"
    assert result["selected_next_step"] != "perform_additional_acquisition"
    assert result["selection_id"] != first.json()["next_step"]["result"]["selection_id"]
    evaluation, selection = _sequential(payload)
    assert evaluation.json() == second.json()["sufficiency"]
    assert selection.json() == second.json()["next_step"]
    assert len(_selection_events(scope["runtime_session_id"])) == 2


def test_terminal_policy_versions_survive_reload_and_match_returned_decisions():
    from services.evidence_next_steps import EVIDENCE_NEXT_STEP_POLICY_VERSION
    from services.evidence_sufficiency import EVIDENCE_SUFFICIENCY_POLICY_VERSION
    from services.runtime_state import clear_states_for_tests, runtime_state_repository

    assert EVIDENCE_SUFFICIENCY_POLICY_VERSION == "evidence-sufficiency-policy.v1"
    assert EVIDENCE_NEXT_STEP_POLICY_VERSION == "evidence-next-step-policy.v1"
    scope = _start_runtime()
    response = _assert_terminal_parity(
        _terminal_payload(scope, facts=[_fact("material", "satisfied")]), scope,
    )
    before = _diagnostics(scope)
    clear_states_for_tests(db_path=runtime_state_repository().db_path)
    after = _diagnostics(scope)
    assert after == before
    selection = response["next_step"]["result"]
    selections = [event for event in after["events"]
                  if event["event_type"] == "evidence_next_step_selected"
                  and event["event_payload_json"].get("selection_id") == selection["selection_id"]]
    assert len(selections) == 1
    assert selections[0]["event_payload_json"]["policy_version"] == (
        EVIDENCE_NEXT_STEP_POLICY_VERSION
    )
    evaluations = [event["event_payload_json"] for event in after["events"]
                   if event["event_type"] == "evidence_sufficiency_evaluated"
                   and event["event_payload_json"].get("evaluation_id")
                   == selection["evaluation_id"]]
    # Existing append behavior allows multiple event rows for one evaluation ID;
    # each new row truthfully records the same producing policy implementation.
    assert len(evaluations) == 3  # v1 composite, sequential evaluation, v2 composite
    assert {event["policy_version"] for event in evaluations} == {
        EVIDENCE_SUFFICIENCY_POLICY_VERSION
    }


def test_legacy_unversioned_events_are_not_backfilled_on_read_or_terminal_retry(monkeypatch):
    from services import evidence_sufficiency
    from services.runtime_state import clear_states_for_tests, runtime_state_repository

    scope = _start_runtime()
    payload = _terminal_payload(scope, facts=[_fact("material", "satisfied")])
    record = evidence_sufficiency.record_runtime_event

    def legacy_record(**kwargs):
        # Explicit old-record fixture: historical services did not record a version.
        kwargs["event_payload_json"] = dict(kwargs["event_payload_json"])
        kwargs["event_payload_json"].pop("policy_version")
        return record(**kwargs)

    with monkeypatch.context() as historical:
        historical.setattr(evidence_sufficiency, "record_runtime_event", legacy_record)
        historical.setattr(evidence_next_steps, "record_runtime_event", legacy_record)
        first = _terminal(payload)
    assert first.status_code == 200
    before = _diagnostics(scope)
    decisions = [event for event in before["events"] if event["event_type"] in {
        "evidence_sufficiency_evaluated", "evidence_next_step_selected",
    }]
    assert len(decisions) == 2
    assert all("policy_version" not in event["event_payload_json"] for event in decisions)
    clear_states_for_tests(db_path=runtime_state_repository().db_path)
    assert _diagnostics(scope) == before
    repeated = _terminal(payload)
    assert repeated.status_code == 200
    assert repeated.json() == first.json()
    after = _diagnostics(scope)
    assert after["events"][:-1] == before["events"]
    appended = after["events"][-1]
    assert appended["event_type"] == "evidence_sufficiency_evaluated"
    assert appended["event_payload_json"]["policy_version"] == "evidence-sufficiency-policy.v1"
    assert appended["event_payload_json"]["evaluation_id"] == first.json()["sufficiency"]["result"][
        "evaluation_id"
    ]
    # Selection deduplicates against the original immutable record; no current
    # version may be asserted for that historical selection retroactively.
    assert _selection_events(scope["runtime_session_id"]) == [decisions[1]]
    assert "policy_version" not in json.dumps(repeated.json())


def test_new_selection_over_legacy_sufficiency_records_only_its_own_policy(monkeypatch):
    from services import evidence_sufficiency

    scope = _start_runtime()
    record = evidence_sufficiency.record_runtime_event

    def legacy_record(**kwargs):
        kwargs["event_payload_json"].pop("policy_version")
        return record(**kwargs)

    with monkeypatch.context() as historical:
        historical.setattr(evidence_sufficiency, "record_runtime_event", legacy_record)
        evaluation = _evaluate(scope, [_requirement("material", "targeted_evidence")],
                               [_fact("material", "satisfied")])
    before = _diagnostics(scope)
    legacy = before["events"][-1]
    assert "policy_version" not in legacy["event_payload_json"]
    response = _select(_next_step_payload(scope, evaluation))
    assert response.status_code == 200
    after = _diagnostics(scope)
    assert after["events"][:-1] == before["events"]
    selection = after["events"][-1]["event_payload_json"]
    assert selection["policy_version"] == "evidence-next-step-policy.v1"
    assert selection["selection_id"] == response.json()["result"]["selection_id"]
    assert selection["evaluation_id"] == evaluation["result"]["evaluation_id"]
    assert "policy_version" not in json.dumps(response.json())


def test_terminal_v2_fresh_exact_writer_events_and_v1_json_parity(monkeypatch):
    from services import evidence_sufficiency

    scope = _start_runtime()
    payload = _terminal_payload(scope, facts=[_fact("material", "satisfied")])
    before = _diagnostics(scope)
    committed = []
    record = evidence_sufficiency.record_runtime_event

    def capture_record(**kwargs):
        event = record(**kwargs)
        committed.append(event)
        return event

    monkeypatch.setattr(evidence_sufficiency, "record_runtime_event", capture_record)
    monkeypatch.setattr(evidence_next_steps, "record_runtime_event", capture_record)
    response = _terminal(payload, "v2")
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"schema_version", "sufficiency", "next_step",
                         "sufficiency_provenance", "next_step_provenance"}
    assert body["schema_version"] == "evidence-terminal.v2"
    assert len(committed) == 2
    assert [event.event_type for event in committed] == [
        "evidence_sufficiency_evaluated", "evidence_next_step_selected",
    ]
    sufficiency = body["sufficiency_provenance"]
    selection = body["next_step_provenance"]
    assert sufficiency["computation_policy_version"] == "evidence-sufficiency-policy.v1"
    assert selection["computation_policy_version"] == "evidence-next-step-policy.v1"
    assert sufficiency["persistence"] == selection["persistence"] == "committed"
    assert sufficiency["persisted_event"]["event_id"] == committed[0].event_id
    assert selection["persisted_event"]["event_id"] == committed[1].event_id
    assert selection["consulted_sufficiency_event"] == sufficiency["persisted_event"]
    _assert_v2_event_references(body, scope)
    assert len(_diagnostics(scope)["events"]) == len(before["events"]) + 2
    assert _diagnostics(scope)["latest_turn"] == before["latest_turn"]
    v1 = _terminal(payload)
    evaluation, next_step = _sequential(payload)
    assert v1.json() == {
        "schema_version": "evidence-terminal.v1",
        "sufficiency": body["sufficiency"], "next_step": body["next_step"],
    }
    assert body["sufficiency"] == evaluation.json()
    assert body["next_step"] == next_step.json()
    assert body["next_step"]["result"]["provider_disposition"] == "allowed"
    assert body["next_step"]["result"]["selected_next_step"] == "answer_within_declared_scope"
    # V1's exact serialized JSON contract still contains only its original fields.
    assert v1.content == json.dumps(v1.json(), separators=(",", ":")).encode()


def test_terminal_v2_duplicate_evaluation_occurrences_and_reload():
    from services.runtime_state import clear_states_for_tests, runtime_state_repository

    scope = _start_runtime()
    payload = _terminal_payload(scope)
    first = _terminal(payload, "v2")
    second = _terminal(payload, "v2")
    assert first.status_code == second.status_code == 200
    a, b = first.json(), second.json()
    assert a["sufficiency"] == b["sufficiency"]
    assert a["next_step"] == b["next_step"]
    assert (a["sufficiency"]["result"]["evaluation_id"]
            == b["sufficiency"]["result"]["evaluation_id"])
    original = a["sufficiency_provenance"]["persisted_event"]
    current = b["sufficiency_provenance"]["persisted_event"]
    assert original["event_id"] != current["event_id"]
    assert b["next_step_provenance"]["consulted_sufficiency_event"] == original
    assert (b["next_step_provenance"]["persisted_event"]
            == a["next_step_provenance"]["persisted_event"])
    assert a["next_step_provenance"]["persistence"] == "committed"
    assert b["next_step_provenance"]["persistence"] == "reused"
    before = _diagnostics(scope)
    clear_states_for_tests(db_path=runtime_state_repository().db_path)
    assert _diagnostics(scope) == before
    _assert_v2_event_references(a, scope)
    _assert_v2_event_references(b, scope)


def test_terminal_v2_historical_unversioned_counterexample(monkeypatch):
    from services import evidence_sufficiency
    from services.runtime_state import clear_states_for_tests, runtime_state_repository

    scope = _start_runtime()
    payload = _terminal_payload(scope, facts=[_fact("material", "satisfied")])
    record = evidence_sufficiency.record_runtime_event

    def historical_record(**kwargs):
        # Reproduce the old service's real persisted payloads and API results.
        # Both events lack policy metadata, as in the independent counterexample.
        kwargs["event_payload_json"].pop("policy_version")
        return record(**kwargs)

    with monkeypatch.context() as historical:
        historical.setattr(evidence_sufficiency, "record_runtime_event", historical_record)
        historical.setattr(evidence_next_steps, "record_runtime_event", historical_record)
        legacy = _terminal(payload)
    assert legacy.status_code == 200
    before = _diagnostics(scope)
    old_evaluation, old_selection = before["events"][-2:]
    assert "policy_version" not in old_evaluation["event_payload_json"]
    assert "policy_version" not in old_selection["event_payload_json"]
    clear_states_for_tests(db_path=runtime_state_repository().db_path)
    computations = []
    select = evidence_next_steps._selection

    def compute_selection(**kwargs):
        computations.append(kwargs)
        return select(**kwargs)

    monkeypatch.setattr(evidence_next_steps, "_selection", compute_selection)
    response = _terminal(payload, "v2")
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(computations) == 1  # Recomputed, despite reusing the old event.
    assert body["sufficiency"] == legacy.json()["sufficiency"]
    assert body["next_step"] == legacy.json()["next_step"]
    current = body["sufficiency_provenance"]
    selection = body["next_step_provenance"]
    assert current["computation_policy_version"] == "evidence-sufficiency-policy.v1"
    assert current["persisted_event"]["policy_version"] == "evidence-sufficiency-policy.v1"
    assert current["persisted_event"]["event_id"] != old_evaluation["event_id"]
    assert current["persisted_event"]["evaluation_id"] == old_evaluation["event_payload_json"][
        "evaluation_id"
    ]
    assert selection["computation_policy_version"] == "evidence-next-step-policy.v1"
    assert selection["persistence"] == "reused"
    assert selection["persisted_event"]["event_id"] == old_selection["event_id"]
    assert selection["persisted_event"]["policy_version"] == "UNKNOWN"
    assert selection["consulted_sufficiency_event"]["event_id"] == old_evaluation["event_id"]
    assert selection["consulted_sufficiency_event"]["policy_version"] == "UNKNOWN"
    after = _diagnostics(scope)
    assert after["events"][:-1] == before["events"]
    assert len(after["events"]) == len(before["events"]) + 1
    assert after["events"][-1]["event_id"] == current["persisted_event"]["event_id"]
    clear_states_for_tests(db_path=runtime_state_repository().db_path)
    assert _diagnostics(scope) == after
    _assert_v2_event_references(body, scope)


def test_terminal_v2_ambiguous_retained_selection_fails_without_changing_v1():
    from services.runtime_state import record_runtime_event

    scope = _start_runtime()
    payload = _terminal_payload(scope, facts=[_fact("material", "satisfied")])
    original = _terminal(payload)
    assert original.status_code == 200
    retained = _selection_events(scope["runtime_session_id"])[0]
    duplicate = record_runtime_event(
        runtime_session_id=scope["runtime_session_id"],
        runtime_turn_id=scope["runtime_turn_id"],
        event_type=retained["event_type"],
        event_payload_json=retained["event_payload_json"],
    )
    assert duplicate.event_id != retained["event_id"]
    assert _terminal(payload).content == original.content
    before = _diagnostics(scope)
    response = _terminal(payload, "v2")
    assert response.status_code == 409
    assert response.json() == {"detail": "evidence_terminal_provenance_ambiguous"}
    after = _diagnostics(scope)
    assert after["events"][:-1] == before["events"]
    assert after["events"][-1]["event_type"] == "evidence_sufficiency_evaluated"
    assert len(_selection_events(scope["runtime_session_id"])) == 2


@pytest.mark.parametrize("field,value", [
    ("request_id", "wrong-request"),
    ("runtime_session_id", "wrong-session"),
    ("runtime_turn_id", "wrong-turn"),
    ("evidence_plan_id", "wrong-plan"),
    ("acquisition_manifest_id", "wrong-manifest"),
    ("evaluation_id", "wrong-evaluation"),
    ("policy_version", "evidence-terminal.v2"),
    ("policy_version", None),
])
def test_terminal_v2_invalid_retained_event_provenance_fails_closed(field, value):
    from services.runtime_state import runtime_state_repository

    scope = _start_runtime()
    payload = _terminal_payload(scope, facts=[_fact("material", "satisfied")])
    assert _terminal(payload).status_code == 200
    selected = _selection_events(scope["runtime_session_id"])[0]
    corrupt = {**selected["event_payload_json"], field: value}
    with runtime_state_repository()._connect() as conn:
        conn.execute(
            "UPDATE conversation_runtime_events SET event_payload_json=? WHERE event_id=?",
            (json.dumps(corrupt), selected["event_id"]),
        )
    before = _diagnostics(scope)
    response = _terminal(payload, "v2")
    assert response.status_code == 409
    assert response.json() == {"detail": "evidence_terminal_provenance_invalid"}
    after = _diagnostics(scope)
    assert after["events"][:-1] == before["events"]
    assert len(_selection_events(scope["runtime_session_id"])) == 1


@pytest.mark.parametrize("stage", ["sufficiency", "next_step"])
def test_terminal_v2_forged_writer_event_reference_is_not_published(stage, monkeypatch):
    from services import evidence_sufficiency

    scope = _start_runtime()
    payload = _terminal_payload(scope)
    target = evidence_sufficiency if stage == "sufficiency" else evidence_next_steps
    record = target.record_runtime_event

    def forged_reference(**kwargs):
        event = record(**kwargs)
        return event.model_copy(update={"event_id": "event-not-persisted"})

    monkeypatch.setattr(target, "record_runtime_event", forged_reference)
    response = _terminal(payload, "v2")
    assert response.status_code == 409
    assert response.json() == {"detail": "evidence_terminal_provenance_invalid"}
    # An invalid sufficiency receipt must be rejected before selection. A bad
    # selection receipt can leave both writes durable without terminal success.
    decisions = [event["event_type"] for event in _diagnostics(scope)["events"]
                 if event["event_type"] in {
                     "evidence_sufficiency_evaluated", "evidence_next_step_selected",
                 }]
    assert decisions == (["evidence_sufficiency_evaluated"] if stage == "sufficiency" else [
        "evidence_sufficiency_evaluated", "evidence_next_step_selected",
    ])


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("field", ["sufficiency_provenance", "next_step_provenance",
                                   "persisted_event", "computation_policy_version"])
def test_terminal_provenance_is_server_owned_not_request_authority(version, field):
    scope = _start_runtime()
    before = _diagnostics(scope)
    response = _terminal(_terminal_payload(scope, **{field: {"persistence": "committed"}}), version)
    assert response.status_code == 422
    assert _diagnostics(scope) == before


def test_terminal_v2_response_provenance_is_strict_and_bound():
    scope = _start_runtime()
    response = _terminal(_terminal_payload(scope), "v2")
    assert response.status_code == 200
    valid = response.json()
    mutations = [
        (("schema_version",), "evidence-terminal.v1"),
        (("extra",), True),
        (("sufficiency_provenance", "computation_policy_version"), "evidence-terminal.v2"),
        (("next_step_provenance", "computation_policy_version"), "UNKNOWN"),
        (("sufficiency_provenance", "persistence"), "reused"),
        (("next_step_provenance", "persistence"), True),
        (("sufficiency_provenance", "persisted_event", "policy_version"), "UNKNOWN"),
        (("next_step_provenance", "persisted_event", "policy_version"), "UNKNOWN"),
        (("next_step_provenance", "persisted_event", "extra"), "forged"),
        (("next_step_provenance", "persisted_event", "event_id"), ""),
        (("next_step_provenance", "persisted_event", "event_type"),
         "evidence_sufficiency_evaluated"),
        (("next_step_provenance", "persisted_event", "evaluation_id"), "wrong-evaluation"),
        (("next_step_provenance", "persisted_event", "selection_id"), "wrong-selection"),
        (("next_step_provenance", "consulted_sufficiency_event", "runtime_session_id"), "wrong"),
        (("next_step_provenance", "consulted_sufficiency_event", "runtime_turn_id"), "wrong"),
        (("next_step_provenance", "consulted_sufficiency_event", "evidence_plan_id"), "wrong"),
        (("next_step_provenance", "consulted_sufficiency_event", "acquisition_manifest_id"),
         "wrong"),
        (("next_step", "owner_id"), "wrong-owner"),
        (("next_step", "conversation_id"), "wrong-conversation"),
        (("next_step", "surface"), "wrong-surface"),
    ]
    for path, value in mutations:
        forged = copy.deepcopy(valid)
        target = forged
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValidationError):
            EvidenceTerminalEvaluateResponseV2.model_validate(forged)
    for missing in ("sufficiency", "next_step", "sufficiency_provenance", "next_step_provenance"):
        forged = {key: value for key, value in valid.items() if key != missing}
        with pytest.raises(ValidationError):
            EvidenceTerminalEvaluateResponseV2.model_validate(forged)


@pytest.mark.parametrize("write_new_event", [True, False], ids=["writes-new", "no-write"])
def test_terminal_v2_rejects_stale_sufficiency_commit_receipt(write_new_event, monkeypatch):
    from models import RuntimeEvent
    from services import evidence_sufficiency

    scope = _start_runtime()
    payload = _terminal_payload(scope, facts=[_fact("material", "satisfied")])
    evaluation = client.post("/v1/runtime/evidence-sufficiency/evaluate", json={
        key: value for key, value in payload.items() if key != "current_premise"
    })
    assert evaluation.status_code == 200
    before = _diagnostics(scope)
    older = RuntimeEvent.model_validate(before["events"][-1])
    assert older.event_payload_json["policy_version"] == "evidence-sufficiency-policy.v1"
    assert older.event_payload_json["evaluation_id"] == evaluation.json()["result"]["evaluation_id"]
    record = evidence_sufficiency.record_runtime_event
    actual_writes = []
    selection_calls = []
    select = evidence_next_steps.select_evidence_next_step_with_events

    def stale_receipt(**kwargs):
        assert kwargs["event_payload_json"] == older.event_payload_json
        if write_new_event:
            actual_writes.append(record(**kwargs))
            assert actual_writes[-1].event_id != older.event_id
        return older

    def observe_selection(*args, **kwargs):
        selection_calls.append(True)
        return select(*args, **kwargs)

    monkeypatch.setattr(evidence_sufficiency, "record_runtime_event", stale_receipt)
    monkeypatch.setattr(
        evidence_next_steps, "select_evidence_next_step_with_events", observe_selection,
    )
    response = _terminal(payload, "v2")
    assert response.status_code == 409, response.text
    assert response.json() == {"detail": "evidence_terminal_provenance_invalid"}
    assert selection_calls == []
    after = _diagnostics(scope)
    assert after["events"][:len(before["events"])] == before["events"]
    assert after["events"][len(before["events"]):] == [
        event.model_dump(mode="json") for event in actual_writes
    ]
    assert len(actual_writes) == int(write_new_event)
    assert _selection_events(scope["runtime_session_id"]) == []


@pytest.mark.parametrize("write_new_event", [True, False], ids=["writes-new", "no-write"])
def test_terminal_v2_rejects_stale_selection_commit_receipt(write_new_event, monkeypatch):
    from models import RuntimeEvent

    scope = _start_runtime()
    payload = _terminal_payload(scope, facts=[_fact("material", "satisfied")])
    assert _terminal(payload).status_code == 200
    before = _diagnostics(scope)
    older = RuntimeEvent.model_validate(before["events"][-1])
    record = evidence_next_steps.record_runtime_event
    get_session = evidence_next_steps.get_runtime_session
    select = evidence_next_steps.select_evidence_next_step_with_events
    actual_writes = []
    inside_selection = False

    def stale_selection_read(session_id):
        diagnostics = get_session(session_id)
        if inside_selection:
            # Simulate a stale selection read so the original dedup check
            # cannot see the retained occurrence and attempts a new write.
            diagnostics.events = [event for event in diagnostics.events
                                  if event.event_id != older.event_id]
        return diagnostics

    def select_with_stale_read(*args, **kwargs):
        nonlocal inside_selection
        inside_selection = True
        try:
            return select(*args, **kwargs)
        finally:
            inside_selection = False

    def stale_receipt(**kwargs):
        assert kwargs["event_payload_json"] == older.event_payload_json
        if write_new_event:
            actual_writes.append(record(**kwargs))
            assert actual_writes[-1].event_id != older.event_id
        return older

    monkeypatch.setattr(evidence_next_steps, "get_runtime_session", stale_selection_read)
    monkeypatch.setattr(
        evidence_next_steps, "select_evidence_next_step_with_events", select_with_stale_read,
    )
    monkeypatch.setattr(evidence_next_steps, "record_runtime_event", stale_receipt)
    response = _terminal(payload, "v2")
    assert response.status_code == 409
    assert response.json() == {"detail": "evidence_terminal_provenance_invalid"}
    after = _diagnostics(scope)
    assert after["events"][:len(before["events"])] == before["events"]
    added = after["events"][len(before["events"]):]
    assert added[0]["event_type"] == "evidence_sufficiency_evaluated"
    assert added[1:] == [event.model_dump(mode="json") for event in actual_writes]
    assert len(actual_writes) == int(write_new_event)


@pytest.mark.parametrize("stage", ["sufficiency", "next_step"])
@pytest.mark.parametrize("change", ["append", "remove", "mutate", "bool_type"])
def test_terminal_v2_concurrent_additions_allow_bound_commits_but_mutations_reject(
    stage, change, monkeypatch,
):
    from services import evidence_sufficiency
    from services.runtime_state import runtime_state_repository

    scope = _start_runtime()
    prior = _diagnostics(scope)["events"][0]
    if change == "bool_type":
        seeded = _terminal_payload(scope, facts=[_fact("material", "satisfied")])
        evaluation = client.post("/v1/runtime/evidence-sufficiency/evaluate", json={
            key: value for key, value in seeded.items() if key != "current_premise"
        })
        assert evaluation.status_code == 200
        prior = _diagnostics(scope)["events"][-1]
        assert prior["event_payload_json"]["qualification_required"] is False
    target = evidence_sufficiency if stage == "sufficiency" else evidence_next_steps
    record = target.record_runtime_event

    def intervening_writer(**kwargs):
        receipt = record(**kwargs)
        if change == "append":
            # Identical payload and decision ID, genuinely different occurrence.
            concurrent = record(**{key: value for key, value in kwargs.items()
                                   if key != "expected_event_id"})
            assert concurrent.event_id != receipt.event_id
        else:
            with runtime_state_repository()._connect() as conn:
                if change == "remove":
                    conn.execute("DELETE FROM conversation_runtime_events WHERE event_id=?",
                                 (prior["event_id"],))
                else:
                    payload = dict(prior["event_payload_json"])
                    if change == "bool_type":
                        # Python considers False == 0; exact JSON records do not.
                        payload["qualification_required"] = 0
                    else:
                        payload["changed"] = True
                    conn.execute(
                        "UPDATE conversation_runtime_events SET event_payload_json=? "
                        "WHERE event_id=?",
                        (json.dumps(payload), prior["event_id"]),
                    )
        return receipt

    monkeypatch.setattr(target, "record_runtime_event", intervening_writer)
    response = _terminal(_terminal_payload(scope), "v2")
    if change == "append":
        assert response.status_code == 200, response.text
        _assert_v2_event_references(response.json(), scope)
        return
    assert response.status_code == 409
    assert response.json() == {"detail": "evidence_terminal_provenance_invalid"}
    if stage == "sufficiency":
        assert _selection_events(scope["runtime_session_id"]) == []


@pytest.mark.parametrize("stage", ["sufficiency", "next_step"])
def test_terminal_v2_missing_write_receipt_cannot_produce_success(stage, monkeypatch):
    from services import evidence_sufficiency

    scope = _start_runtime()
    target = evidence_sufficiency if stage == "sufficiency" else evidence_next_steps
    monkeypatch.setattr(target, "record_runtime_event", lambda **kwargs: None)
    response = _terminal(_terminal_payload(scope), "v2")
    assert response.status_code == 409
    assert response.json() == {"detail": "evidence_terminal_provenance_invalid"}
    assert _selection_events(scope["runtime_session_id"]) == []


@pytest.mark.parametrize("field,value", [
    ("event_id", "invalid-id!"),
    ("policy_version", "evidence-sufficiency-policy.v2"),
    ("runtime_session_id", "wrong-session"),
    ("runtime_turn_id", "wrong-turn"),
    ("evidence_plan_id", "wrong-plan"),
    ("acquisition_manifest_id", "wrong-manifest"),
    ("evaluation_id", "wrong-evaluation"),
    ("sufficiency_status", "unknown"),
    ("evaluated_requirements_digest", f"sha256:{'0' * 64}"),
])
def test_terminal_v2_invalid_new_sufficiency_receipt_never_reaches_selection(
    field, value, monkeypatch,
):
    from services import evidence_sufficiency
    from services.runtime_state import runtime_state_repository

    scope = _start_runtime()
    record = evidence_sufficiency.record_runtime_event

    def malformed_write(**kwargs):
        if field == "event_id":
            receipt = record(**kwargs)
            with runtime_state_repository()._connect() as conn:
                conn.execute("UPDATE conversation_runtime_events SET event_id=? WHERE event_id=?",
                             (value, receipt.event_id))
            return receipt.model_copy(update={"event_id": value})
        kwargs["event_payload_json"][field] = value
        return record(**kwargs)

    def forbidden_selection(*args, **kwargs):
        pytest.fail("Invalid sufficiency provenance reached selection")

    monkeypatch.setattr(evidence_sufficiency, "record_runtime_event", malformed_write)
    monkeypatch.setattr(
        evidence_next_steps, "select_evidence_next_step_with_events", forbidden_selection,
    )
    response = _terminal(_terminal_payload(scope, facts=[_fact("material", "satisfied")]), "v2")
    assert response.status_code == 409
    assert response.json() == {"detail": "evidence_terminal_provenance_invalid"}
    assert _selection_events(scope["runtime_session_id"]) == []


def test_terminal_v2_concurrent_evaluation_receipt_does_not_attest_own_write(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, get_ident

    from models import EvidenceSufficiencyEvaluateRequest
    from services import evidence_sufficiency

    scope = _start_runtime()
    payload = _terminal_payload(scope, facts=[_fact("material", "satisfied")])
    other_body = EvidenceSufficiencyEvaluateRequest.model_validate({
        key: value for key, value in payload.items() if key != "current_premise"
    })
    current_thread = get_ident()
    ready = Event()
    record = evidence_sufficiency.record_runtime_event
    actual_commits = []

    def foreign_receipt(**kwargs):
        if get_ident() != current_thread:
            event = record(**kwargs)
            actual_commits.append((get_ident(), event))
            return event
        # This invocation does not write. The separate concurrent evaluation
        # commits one matching occurrence after this invocation's snapshot.
        ready.set()
        return other.result(timeout=10)[1]

    def independent_evaluation():
        assert ready.wait(timeout=10)
        return evidence_sufficiency.evaluate_evidence_sufficiency_with_event(other_body)

    monkeypatch.setattr(evidence_sufficiency, "record_runtime_event", foreign_receipt)
    with ThreadPoolExecutor(max_workers=1) as pool:
        other = pool.submit(independent_evaluation)
        response = _terminal(payload, "v2")
    assert len(actual_commits) == 1
    assert actual_commits[0][0] != current_thread
    if response.status_code == 200:
        provenance = response.json()["sufficiency_provenance"]
        assert provenance["persisted_event"]["event_id"] == actual_commits[0][1].event_id
        assert provenance["persistence"] == "committed"
    assert response.status_code == 409, response.text
    assert _selection_events(scope["runtime_session_id"]) == []


def test_terminal_v2_concurrent_invocations_keep_exact_commit_identities(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from models import EvidenceTerminalEvaluateRequest
    from services import evidence_sufficiency
    from services.runtime_state import RuntimeStateRepository, runtime_state_repository

    scope = _start_runtime()
    payload = _terminal_payload(scope, facts=[_fact("material", "satisfied")])
    before = _diagnostics(scope)
    barrier = Barrier(2)
    record = evidence_sufficiency.record_runtime_event
    receipts = {}

    def synchronized_write(**kwargs):
        expected = kwargs["expected_event_id"]
        barrier.wait(timeout=10)
        event = record(**kwargs)
        assert event.event_id == expected
        receipts[kwargs["event_payload_json"]["request_id"]] = event
        return event

    monkeypatch.setattr(evidence_sufficiency, "record_runtime_event", synchronized_write)

    def evaluate(index):
        body = EvidenceTerminalEvaluateRequest.model_validate({
            **payload, "request_id": f"concurrent-terminal-{index}",
        })
        return evidence_next_steps.evaluate_evidence_terminal_v2(body).model_dump(mode="json")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(evaluate, range(2)))
    assert len({event.event_id for event in receipts.values()}) == 2
    for result in results:
        _assert_v2_event_references(result, scope)
        event = receipts[result["sufficiency"]["request_id"]]
        assert result["sufficiency_provenance"]["persisted_event"]["event_id"] == event.event_id
        consulted = result["next_step_provenance"]["consulted_sufficiency_event"]
        assert consulted["event_id"] == event.event_id
        assert result["next_step_provenance"]["persistence"] == "committed"
        assert result["next_step"]["result"]["provider_disposition"] == "allowed"
    persisted = RuntimeStateRepository(runtime_state_repository().db_path).diagnostics(
        scope["runtime_session_id"],
    )
    assert len(persisted.events) == len(before["events"]) + 4
    for result in results:
        for key in ("sufficiency_provenance", "next_step_provenance"):
            reference = result[key]["persisted_event"]
            event = next(e for e in persisted.events if e.event_id == reference["event_id"])
            assert event.event_payload_json["policy_version"] == reference["policy_version"]


@pytest.mark.parametrize("stage", ["sufficiency", "next_step"])
def test_terminal_v2_storage_rollback_cannot_publish_success(stage, monkeypatch):
    from services.runtime_state import runtime_state_repository

    scope = _start_runtime()
    repo = runtime_state_repository()
    original = repo._record_event
    before = _diagnostics(scope)

    def fail_after_insert(conn, **kwargs):
        event = original(conn, **kwargs)
        if kwargs["event_type"] == ("evidence_sufficiency_evaluated" if stage == "sufficiency"
                                    else "evidence_next_step_selected"):
            assert event.event_id == kwargs["expected_event_id"]
            raise RuntimeError("storage_failure_after_insert")
        return event

    monkeypatch.setattr(repo, "_record_event", fail_after_insert)
    with pytest.raises(RuntimeError, match="storage_failure_after_insert"):
        _terminal(_terminal_payload(scope), "v2")
    after = _diagnostics(scope)
    added = after["events"][len(before["events"]):]
    assert [e["event_type"] for e in added] == (
        [] if stage == "sufficiency" else ["evidence_sufficiency_evaluated"]
    )
    assert _selection_events(scope["runtime_session_id"]) == []


@pytest.mark.parametrize("stage", ["sufficiency", "next_step"])
def test_terminal_v2_wrong_expected_identity_cannot_claim_committed(stage, monkeypatch):
    from services import evidence_sufficiency
    from services.runtime_state import new_runtime_event_id

    scope = _start_runtime()
    target = evidence_sufficiency if stage == "sufficiency" else evidence_next_steps
    record = target.record_runtime_event
    writes = []

    def wrong_identity(**kwargs):
        expected = kwargs["expected_event_id"]
        event = record(**{**kwargs, "expected_event_id": new_runtime_event_id()})
        assert event.event_id != expected
        writes.append(event)
        return event

    monkeypatch.setattr(target, "record_runtime_event", wrong_identity)
    response = _terminal(_terminal_payload(scope), "v2")
    assert response.status_code == 409
    assert response.json() == {"detail": "evidence_terminal_provenance_invalid"}
    assert len(writes) == 1
    if stage == "sufficiency":
        assert _selection_events(scope["runtime_session_id"]) == []
