from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def _start_runtime(
    *,
    request_id: str = "request-start",
    owner_id: str = "owner-1",
    conversation_id: str = "conversation-1",
    surface: str = "web",
) -> dict[str, str]:
    response = client.post(
        "/v1/runtime/turns/start",
        json={
            "request_id": request_id,
            "owner_id": owner_id,
            "conversation_id": conversation_id,
            "surface": surface,
        },
    )
    assert response.status_code == 200
    body = response.json()
    return {
        "request_id": "request-calibration",
        "owner_id": owner_id,
        "conversation_id": conversation_id,
        "surface": surface,
        "runtime_session_id": body["runtime_session"]["runtime_session_id"],
        "runtime_turn_id": body["runtime_turn"]["runtime_turn_id"],
        "claim_anchor": "The bounded claim under evaluation.",
    }


def _reference(
    ref_id: str,
    *,
    owner_id: str = "owner-1",
    conversation_id: str | None = "conversation-1",
    ref_type: str = "external_source",
    support_kind: str = "direct",
    authority: str = "peer_reviewed_evidence",
    freshness_state: str = "active",
) -> dict[str, object]:
    return {
        "ref_type": ref_type,
        "ref_id": ref_id,
        "owner_id": owner_id,
        "conversation_id": conversation_id,
        "support_kind": support_kind,
        "authority": authority,
        "freshness_state": freshness_state,
    }


def _evaluate(
    scope: dict[str, str],
    evidence_references: list[dict[str, object]],
    **overrides: object,
):
    payload: dict[str, object] = {
        **scope,
        "evidence_references": evidence_references,
    }
    payload.update(overrides)
    return client.post("/v1/runtime/claim-calibration/evaluate", json=payload)


def _support_evidence(
    ref_id: str,
    *,
    owner_id: str = "owner-1",
    conversation_id: str = "conversation-1",
    source_authority: str = "established",
    freshness: str = "current",
    material_disclosure_required: bool = False,
    material_role: str = "neutral",
) -> dict[str, object]:
    return {
        "ref_id": ref_id,
        "owner_id": owner_id,
        "conversation_id": conversation_id,
        "source_authority": source_authority,
        "freshness": freshness,
        "material_disclosure_required": material_disclosure_required,
        "material_role": material_role,
    }


def _executed_derivation(
    scope: dict[str, str],
    *,
    derivation_id: str = "derivation-1",
    evidence_ref_ids: list[str] | None = None,
    input_basis: str = "system_established",
) -> dict[str, object]:
    return {
        "derivation_id": derivation_id,
        "owner_id": scope["owner_id"],
        "conversation_id": scope["conversation_id"],
        "runtime_session_id": scope["runtime_session_id"],
        "runtime_turn_id": scope["runtime_turn_id"],
        "operation": "divide",
        "canonical_inputs": ["5", "8"],
        "canonical_result": "0.625",
        "execution_status": "executed",
        "execution_digest": f"sha256:{'a' * 64}",
        "executor_version": "decimal-v1",
        "supporting_evidence_ref_ids": evidence_ref_ids or [],
        "input_basis": input_basis,
    }


def _support_payload(
    scope: dict[str, str],
    *,
    evidence_references: list[dict[str, object]] | None = None,
    proposal: dict[str, object] | None = None,
    executed_derivations: list[dict[str, object]] | None = None,
    **authority_overrides: object,
) -> dict[str, object]:
    authority_context: dict[str, object] = {
        "owner_id": scope["owner_id"],
        "conversation_id": scope["conversation_id"],
        "surface": scope["surface"],
        "runtime_session_id": scope["runtime_session_id"],
        "runtime_turn_id": scope["runtime_turn_id"],
        "evidence_references": evidence_references or [],
        "complete_declared_scope_required": False,
        "complete_declared_scope_established": None,
        "material_acquisition_limited": False,
        "privacy_policy_allows_claim": True,
        "consequence_policy_allows_claim": True,
        "executed_derivations": executed_derivations or [],
    }
    authority_context.update(authority_overrides)
    return {
        "request_id": "request-claim-support",
        "authority_context": authority_context,
        "proposal": proposal
        or {
            "proposed_claim": "The bounded records support the proposed result.",
            "supporting_evidence_ref_ids": [],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
    }


def _evaluate_support(payload: dict[str, object]):
    return client.post("/v1/runtime/claim-support/evaluate", json=payload)


def test_single_current_manufacturer_source_remains_moderately_supported():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [
            _reference(
                "manufacturer-record-1",
                authority="manufacturer_guidance",
            )
        ],
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["claim_class"] == "manufacturer_guidance"
    assert result["calibration_status"] == "supported"
    assert result["evidence_strength"] == "moderate"
    assert result["confidence"] == "medium"
    assert result["uncertainty_disclosure_required"] is True
    assert "single_source" in result["limitation_codes"]


def test_multiple_current_authoritative_sources_are_strong_and_order_independent():
    scope = _start_runtime()
    references = [
        _reference("clinical-1", authority="clinical_guidance", support_kind="corroborating"),
        _reference(
            "peer-reviewed-1",
            authority="peer_reviewed_evidence",
            support_kind="corroborating",
        ),
    ]

    first = _evaluate(scope, references)
    second = _evaluate(scope, list(reversed(references)))

    assert first.status_code == 200
    assert second.status_code == 200
    first_result = first.json()["result"]
    second_result = second.json()["result"]
    assert first_result["calibration_status"] == "supported"
    assert first_result["claim_class"] == "expert_consensus"
    assert first_result["evidence_strength"] == "strong"
    assert first_result["confidence"] == "high"
    assert first_result["uncertainty_disclosure_required"] is False
    assert first_result == second_result
    assert [item["ref_id"] for item in first_result["validated_evidence_references"]] == [
        "clinical-1",
        "peer-reviewed-1",
    ]


@pytest.mark.parametrize(
    ("ref_type", "authority"),
    [
        ("tool_output", "tool_output"),
        ("integration_event", "trusted_integration"),
    ],
)
def test_current_direct_verified_system_evidence_may_be_verified_fact(
    ref_type: str,
    authority: str,
):
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [
            _reference(
                "verified-record-1",
                ref_type=ref_type,
                authority=authority,
            )
        ],
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["claim_class"] == "verified_fact"
    assert result["evidence_strength"] == "moderate"
    assert result["confidence"] == "medium"


def test_user_report_only_is_not_verified_and_remains_weak():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [_reference("user-report-1", ref_type="message", authority="user_report")],
    )

    result = response.json()["result"]
    assert result["claim_class"] == "source_backed_fact"
    assert result["evidence_strength"] == "weak"
    assert result["confidence"] == "low"
    assert "low_authority_evidence" in result["limitation_codes"]


def test_runtime_inference_only_remains_inference_with_low_confidence():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [_reference("inference-1", authority="runtime_inference")],
    )

    result = response.json()["result"]
    assert result["claim_class"] == "runtime_inference"
    assert result["evidence_strength"] == "weak"
    assert result["confidence"] == "low"
    assert "inference_dominant" in result["limitation_codes"]


def test_speculation_only_remains_speculation_and_limited():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [_reference("speculation-1", authority="speculation")],
    )

    result = response.json()["result"]
    assert result["claim_class"] == "speculation"
    assert result["calibration_status"] == "limited"
    assert result["evidence_strength"] == "weak"
    assert "speculation_only" in result["limitation_codes"]


def test_contextual_only_evidence_does_not_count_as_support():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [_reference("context-1", support_kind="contextual")],
    )

    result = response.json()["result"]
    assert result["claim_class"] == "unknown"
    assert result["calibration_status"] == "unsupported"
    assert result["evidence_strength"] == "none"
    assert result["confidence"] == "unknown"
    assert result["strongest_authority"] == "unknown"
    assert result["limitation_codes"][:2] == ["no_supporting_evidence", "context_only"]


@pytest.mark.parametrize(
    ("freshness_state", "limitation"),
    [
        ("stale", "stale_evidence"),
        ("unknown_freshness", "unknown_freshness"),
    ],
)
def test_uncertain_freshness_lowers_calibration(
    freshness_state: str,
    limitation: str,
):
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [_reference("freshness-1", freshness_state=freshness_state)],
    )

    result = response.json()["result"]
    assert result["evidence_strength"] == "weak"
    assert result["confidence"] == "low"
    assert result["uncertainty_disclosure_required"] is True
    assert limitation in result["limitation_codes"]


def test_contradictory_evidence_prevents_strong_calibration():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [
            _reference("support-1"),
            _reference("support-2", support_kind="corroborating"),
            _reference("contradiction-1", support_kind="contradictory"),
        ],
    )

    result = response.json()["result"]
    assert result["evidence_strength"] == "weak"
    assert result["confidence"] == "low"
    assert result["claim_class"] != "verified_fact"
    assert "contradictory_evidence" in result["limitation_codes"]


def test_contradictory_only_evidence_has_no_supporting_authority():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [_reference("contradiction-only-1", support_kind="contradictory")],
    )

    result = response.json()["result"]
    assert result["evidence_strength"] == "none"
    assert result["strongest_authority"] == "unknown"
    assert "contradictory_evidence" in result["limitation_codes"]


def test_contradiction_cannot_replace_manufacturer_supporting_authority():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [
            _reference(
                "manufacturer-support-1",
                authority="manufacturer_guidance",
            ),
            _reference(
                "peer-contradiction-1",
                support_kind="contradictory",
                authority="peer_reviewed_evidence",
            ),
        ],
    )

    result = response.json()["result"]
    assert result["strongest_authority"] == "manufacturer_guidance"
    assert result["evidence_strength"] == "weak"
    assert "contradictory_evidence" in result["limitation_codes"]


def test_context_cannot_replace_user_report_supporting_authority():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [
            _reference(
                "user-support-1",
                ref_type="message",
                authority="user_report",
            ),
            _reference(
                "peer-context-1",
                support_kind="contextual",
                authority="peer_reviewed_evidence",
            ),
        ],
    )

    result = response.json()["result"]
    assert result["strongest_authority"] == "user_report"
    assert result["evidence_strength"] == "weak"
    assert len(result["validated_evidence_references"]) == 2


def test_cross_owner_evidence_is_rejected():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [_reference("cross-owner-1", owner_id="owner-2")],
    )

    assert response.status_code == 422
    assert "evidence_owner_mismatch" in response.text


def test_cross_conversation_evidence_is_rejected():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [_reference("cross-conversation-1", conversation_id="conversation-2")],
    )

    assert response.status_code == 422
    assert "evidence_conversation_mismatch" in response.text


def test_duplicate_evidence_reference_is_rejected():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [
            _reference("duplicate-1"),
            _reference("duplicate-1", support_kind="corroborating"),
        ],
    )

    assert response.status_code == 422
    assert "duplicate_evidence_reference" in response.text


@pytest.mark.parametrize(
    "extra_field",
    [
        "snippet",
        "raw_content",
        "prompt",
        "reasoning",
        "metadata",
        "model_confidence",
        "authority_score",
    ],
)
def test_unbounded_evidence_fields_are_rejected(extra_field: str):
    scope = _start_runtime()
    reference = _reference("unsafe-extra-1")
    reference[extra_field] = "not accepted"

    response = _evaluate(scope, [reference])

    assert response.status_code == 422
    assert "extra_forbidden" in response.text


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_id": "r" * 121},
        {"claim_anchor": "c" * 501},
        {"limitation_codes": ["caller_supplied"]},
    ],
)
def test_unbounded_or_caller_selected_calibration_fields_are_rejected(
    overrides: dict[str, object],
):
    scope = _start_runtime()
    response = _evaluate(scope, [_reference("bounded-1")], **overrides)

    assert response.status_code == 422


def test_overlong_reference_id_and_reference_list_are_rejected():
    scope = _start_runtime()
    overlong_id = _evaluate(scope, [_reference("r" * 121)])
    too_many = _evaluate(scope, [_reference(f"record-{index}") for index in range(17)])

    assert overlong_id.status_code == 422
    assert too_many.status_code == 422


@pytest.mark.parametrize(
    "reference_override",
    [
        {"ref_type": "url"},
        {"ref_id": "https://example.invalid/source?token=private"},
        {"freshness_state": "fresh"},
    ],
)
def test_malformed_reference_vocabulary_or_identifier_is_rejected(
    reference_override: dict[str, object],
):
    scope = _start_runtime()
    reference = _reference("well-formed-1")
    reference.update(reference_override)

    response = _evaluate(scope, [reference])

    assert response.status_code == 422


def test_blank_claim_anchor_is_rejected_after_normalization():
    scope = _start_runtime()
    response = _evaluate(scope, [], claim_anchor=" \n\t ")

    assert response.status_code == 422


@pytest.mark.parametrize("freshness_state", ["superseded", "corrected"])
def test_superseded_or_corrected_evidence_is_not_positive_support(
    freshness_state: str,
):
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [_reference("invalidated-1", freshness_state=freshness_state)],
    )

    result = response.json()["result"]
    assert result["claim_class"] == "unknown"
    assert result["evidence_strength"] == "none"
    assert result["confidence"] == "unknown"
    assert result["strongest_authority"] == "unknown"
    assert result["freshness_summary"] == "stale"
    assert "superseded_or_corrected_evidence" in result["limitation_codes"]


def test_not_applicable_freshness_is_reported_without_being_exaggerated():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [
            _reference(
                "timeless-guidance-1",
                authority="manufacturer_guidance",
                freshness_state="not_applicable",
            )
        ],
    )

    result = response.json()["result"]
    assert result["claim_class"] == "manufacturer_guidance"
    assert result["evidence_strength"] == "moderate"
    assert result["confidence"] == "medium"
    assert result["freshness_summary"] == "not_applicable"


def test_response_contains_only_typed_scope_and_calibration_fields():
    scope = _start_runtime()
    response = _evaluate(scope, [_reference("typed-1")])

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
    assert set(body["result"]) == {
        "claim_id",
        "claim_anchor",
        "claim_anchor_digest",
        "claim_class",
        "calibration_status",
        "evidence_strength",
        "confidence",
        "strongest_authority",
        "freshness_summary",
        "uncertainty_disclosure_required",
        "validated_evidence_references",
        "limitation_codes",
        "user_safe_summary",
    }


def test_runtime_event_excludes_claim_and_raw_or_private_content():
    scope = _start_runtime()
    scope["claim_anchor"] = "A private but bounded claim anchor."
    response = _evaluate(scope, [_reference("privacy-1")])
    assert response.status_code == 200

    diagnostics = client.get(f"/v1/runtime/sessions/{scope['runtime_session_id']}")
    event = next(
        item
        for item in diagnostics.json()["events"]
        if item["event_type"] == "claim_calibration_evaluated"
    )
    payload = event["event_payload_json"]
    serialized = json.dumps(payload, sort_keys=True).lower()
    assert "claim_anchor" not in payload
    assert "a private but bounded claim anchor" not in serialized
    assert "evidence_references" not in payload
    assert "prompt" not in serialized
    assert "private_memory" not in serialized
    assert "reasoning" not in serialized
    assert "exception" not in serialized
    assert "metadata" not in serialized


def test_unknown_runtime_session_returns_bounded_error():
    scope = {
        "request_id": "request-calibration",
        "owner_id": "owner-1",
        "conversation_id": "conversation-1",
        "surface": "web",
        "runtime_session_id": "rtsession-missing",
        "runtime_turn_id": "rtturn-missing",
        "claim_anchor": "A bounded claim.",
    }
    response = _evaluate(scope, [])

    assert response.status_code == 404
    assert response.json() == {"detail": "runtime_session_not_found"}


def test_unknown_runtime_turn_returns_bounded_error():
    scope = _start_runtime()
    scope["runtime_turn_id"] = "rtturn-missing"
    response = _evaluate(scope, [])

    assert response.status_code == 404
    assert response.json() == {"detail": "runtime_turn_not_found"}


def test_runtime_session_scope_mismatch_returns_bounded_error():
    scope = _start_runtime()
    scope["owner_id"] = "owner-2"
    response = _evaluate(scope, [])

    assert response.status_code == 400
    assert response.json() == {"detail": "runtime_session_mismatch"}


def test_runtime_turn_session_mismatch_returns_bounded_error():
    first_scope = _start_runtime()
    second_scope = _start_runtime(
        request_id="request-start-2",
        conversation_id="conversation-2",
    )
    second_scope["runtime_turn_id"] = first_scope["runtime_turn_id"]
    response = _evaluate(
        second_scope,
        [
            _reference(
                "mismatch-1",
                conversation_id="conversation-2",
            )
        ],
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "runtime_turn_session_mismatch"}


def test_api_response_and_runtime_event_agree_on_structural_outcome():
    scope = _start_runtime()
    response = _evaluate(
        scope,
        [_reference("agreement-1", authority="manufacturer_guidance")],
    )
    result = response.json()["result"]

    diagnostics = client.get(f"/v1/runtime/sessions/{scope['runtime_session_id']}")
    event = next(
        item
        for item in diagnostics.json()["events"]
        if item["event_type"] == "claim_calibration_evaluated"
    )
    payload = event["event_payload_json"]
    for field in (
        "claim_id",
        "claim_anchor_digest",
        "claim_class",
        "calibration_status",
        "evidence_strength",
        "confidence",
        "strongest_authority",
        "freshness_summary",
        "uncertainty_disclosure_required",
        "limitation_codes",
    ):
        assert payload[field] == result[field]
    assert payload["evidence_count"] == 1


def test_generic_claim_support_allows_ordinary_claim_without_task_shape():
    scope = _start_runtime()
    payload = _support_payload(
        scope,
        evidence_references=[_support_evidence("evidence-1")],
        proposal={
            "proposed_claim": "  A bounded contextual claim is supported.  ",
            "supporting_evidence_ref_ids": ["evidence-1"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
        complete_declared_scope_established=False,
    )

    response = _evaluate_support(payload)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["calibration_status"] == "supported"
    assert result["conclusion_disposition"] == "allowed"
    assert result["qualification_required"] is False
    assert result["limitation_codes"] == []
    assert result["validated_supporting_evidence_ref_ids"] == ["evidence-1"]
    assert "task_shape" not in json.dumps(payload)


@pytest.mark.parametrize(
    "proposal_field",
    [
        "provenance",
        "source_authority",
        "freshness",
        "completeness",
        "execution_status",
        "confidence",
        "conclusion_disposition",
        "action_permission",
        "tool_permission",
        "observation_status",
        "claim_scope_basis",
    ],
)
def test_generic_proposal_rejects_authority_escalation_fields(
    proposal_field: str,
):
    scope = _start_runtime()
    proposal = {
        "proposed_claim": "A bounded claim.",
        "supporting_evidence_ref_ids": ["evidence-1"],
        "counterevidence_ref_ids": [],
        "material_exclusions": [],
        "executed_derivation_ref_ids": [],
        proposal_field: "provider-selected",
    }
    payload = _support_payload(
        scope,
        evidence_references=[_support_evidence("evidence-1")],
        proposal=proposal,
    )

    response = _evaluate_support(payload)

    assert response.status_code == 422
    assert "extra_forbidden" in response.text


@pytest.mark.parametrize(
    ("proposal_update", "expected_error"),
    [
        (
            {"supporting_evidence_ref_ids": ["missing-evidence"]},
            "proposal_evidence_reference_not_authorized",
        ),
        (
            {"supporting_evidence_ref_ids": ["evidence-1", "evidence-1"]},
            "duplicate_proposal_reference",
        ),
        (
            {
                "supporting_evidence_ref_ids": ["evidence-1"],
                "counterevidence_ref_ids": ["evidence-1"],
            },
            "conflicting_proposal_evidence_role",
        ),
    ],
)
def test_generic_proposal_references_are_closed_and_unique(
    proposal_update: dict[str, object],
    expected_error: str,
):
    scope = _start_runtime()
    proposal: dict[str, object] = {
        "proposed_claim": "A bounded claim.",
        "supporting_evidence_ref_ids": [],
        "counterevidence_ref_ids": [],
        "material_exclusions": [],
        "executed_derivation_ref_ids": [],
    }
    proposal.update(proposal_update)
    payload = _support_payload(
        scope,
        evidence_references=[_support_evidence("evidence-1")],
        proposal=proposal,
    )

    response = _evaluate_support(payload)

    assert response.status_code == 422
    assert expected_error in response.text


def test_generic_supporting_evidence_may_disclose_partial_material_exclusion():
    scope = _start_runtime()
    payload = _support_payload(
        scope,
        evidence_references=[_support_evidence("collection-evidence")],
        proposal={
            "proposed_claim": "The usable entries support a bounded estimate.",
            "supporting_evidence_ref_ids": ["collection-evidence"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [
                {
                    "evidence_ref_id": "collection-evidence",
                    "reason": "One entry was ambiguous and excluded.",
                }
            ],
            "executed_derivation_ref_ids": [],
        },
    )

    response = _evaluate_support(payload)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["calibration_status"] == "limited"
    assert result["conclusion_disposition"] == "qualified"
    assert result["qualification_required"] is True
    assert result["validated_supporting_evidence_ref_ids"] == [
        "collection-evidence"
    ]
    assert result["validated_material_exclusions"] == [
        {
            "evidence_ref_id": "collection-evidence",
            "reason": "One entry was ambiguous and excluded.",
        }
    ]
    assert "material_exclusion" in result["limitation_codes"]


def test_generic_counterevidence_cannot_be_relabelled_as_exclusion():
    scope = _start_runtime()
    payload = _support_payload(
        scope,
        evidence_references=[
            _support_evidence(
                "material-counter",
                material_disclosure_required=True,
                material_role="counterevidence",
            )
        ],
        proposal={
            "proposed_claim": "A bounded claim.",
            "supporting_evidence_ref_ids": [],
            "counterevidence_ref_ids": ["material-counter"],
            "material_exclusions": [
                {
                    "evidence_ref_id": "material-counter",
                    "reason": "The material conflicts with the claim.",
                }
            ],
            "executed_derivation_ref_ids": [],
        },
    )

    response = _evaluate_support(payload)

    assert response.status_code == 422
    assert "conflicting_proposal_evidence_role" in response.text


@pytest.mark.parametrize(
    ("authority_update", "expected_error"),
    [
        ({"owner_id": "owner-2"}, "evidence_owner_mismatch"),
        ({"conversation_id": "conversation-2"}, "evidence_conversation_mismatch"),
    ],
)
def test_generic_authority_rejects_evidence_association_mismatch(
    authority_update: dict[str, object],
    expected_error: str,
):
    scope = _start_runtime()
    reference = _support_evidence("evidence-1")
    reference.update(authority_update)
    payload = _support_payload(scope, evidence_references=[reference])

    response = _evaluate_support(payload)

    assert response.status_code == 422
    assert expected_error in response.text


def test_generic_complete_scope_requirement_is_claim_sensitive():
    ordinary_scope = _start_runtime(request_id="ordinary-start")
    ordinary = _support_payload(
        ordinary_scope,
        evidence_references=[_support_evidence("ordinary-evidence")],
        proposal={
            "proposed_claim": "A bounded ordinary claim.",
            "supporting_evidence_ref_ids": ["ordinary-evidence"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
        complete_declared_scope_established=False,
    )
    exhaustive_scope = _start_runtime(
        request_id="exhaustive-start",
        conversation_id="conversation-exhaustive",
    )
    exhaustive = _support_payload(
        exhaustive_scope,
        evidence_references=[
            _support_evidence(
                "exhaustive-evidence",
                conversation_id="conversation-exhaustive",
            )
        ],
        proposal={
            "proposed_claim": "The declared scope contains no other records.",
            "supporting_evidence_ref_ids": ["exhaustive-evidence"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
        complete_declared_scope_required=True,
        complete_declared_scope_established=False,
    )

    ordinary_result = _evaluate_support(ordinary).json()["result"]
    exhaustive_result = _evaluate_support(exhaustive).json()["result"]

    assert "claim_scope_basis" not in exhaustive["authority_context"]
    assert ordinary_result["conclusion_disposition"] == "allowed"
    assert exhaustive_result["calibration_status"] == "unsupported"
    assert exhaustive_result["conclusion_disposition"] == "withheld"
    assert "complete_scope_not_established" in exhaustive_result["limitation_codes"]


def test_generic_supplied_evidence_scope_qualifies_incomplete_broader_scope():
    scope = _start_runtime()
    payload = _support_payload(
        scope,
        evidence_references=[_support_evidence("bounded-evidence")],
        proposal={
            "proposed_claim": (
                "Among the supplied records, the available entries support "
                "the bounded comparison."
            ),
            "supporting_evidence_ref_ids": ["bounded-evidence"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
        claim_scope_basis="supplied_evidence",
        complete_declared_scope_required=True,
        complete_declared_scope_established=False,
    )

    response = _evaluate_support(payload)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["calibration_status"] == "limited"
    assert result["conclusion_disposition"] == "qualified"
    assert result["qualification_required"] is True
    assert "complete_scope_not_established" in result["limitation_codes"]
    assert result["validated_supporting_evidence_ref_ids"] == ["bounded-evidence"]


def test_generic_supplied_evidence_scope_discloses_material_acquisition_limit():
    scope = _start_runtime()
    payload = _support_payload(
        scope,
        evidence_references=[_support_evidence("bounded-evidence")],
        proposal={
            "proposed_claim": "The supplied records support a bounded claim.",
            "supporting_evidence_ref_ids": ["bounded-evidence"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
        claim_scope_basis="supplied_evidence",
        complete_declared_scope_required=True,
        complete_declared_scope_established=False,
        material_acquisition_limited=True,
    )

    result = _evaluate_support(payload).json()["result"]

    assert result["calibration_status"] == "limited"
    assert result["conclusion_disposition"] == "qualified"
    assert result["qualification_required"] is True
    assert "complete_scope_not_established" in result["limitation_codes"]
    assert "material_acquisition_limited" in result["limitation_codes"]


def test_generic_supplied_evidence_scope_does_not_force_qualification():
    scope = _start_runtime()
    payload = _support_payload(
        scope,
        evidence_references=[_support_evidence("complete-evidence")],
        proposal={
            "proposed_claim": "The supplied records support the complete result.",
            "supporting_evidence_ref_ids": ["complete-evidence"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
        claim_scope_basis="supplied_evidence",
        complete_declared_scope_required=True,
        complete_declared_scope_established=True,
    )

    result = _evaluate_support(payload).json()["result"]

    assert result["calibration_status"] == "supported"
    assert result["conclusion_disposition"] == "allowed"
    assert result["qualification_required"] is False
    assert result["limitation_codes"] == []


def test_generic_claim_scope_basis_changes_claim_identity_not_digest():
    scope = _start_runtime()
    common = {
        "evidence_references": [_support_evidence("complete-evidence")],
        "proposal": {
            "proposed_claim": "The complete evidence supports this claim.",
            "supporting_evidence_ref_ids": ["complete-evidence"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
        "complete_declared_scope_required": True,
        "complete_declared_scope_established": True,
    }
    declared = _support_payload(
        scope,
        claim_scope_basis="declared_scope",
        **common,
    )
    supplied = _support_payload(
        scope,
        claim_scope_basis="supplied_evidence",
        **common,
    )

    declared_result = _evaluate_support(declared).json()["result"]
    supplied_result = _evaluate_support(supplied).json()["result"]

    assert declared_result["claim_digest"] == supplied_result["claim_digest"]
    assert declared_result["limitation_codes"] == supplied_result["limitation_codes"]
    assert declared_result["claim_id"] != supplied_result["claim_id"]


def test_generic_claim_support_response_fields_remain_compatible():
    scope = _start_runtime()
    payload = _support_payload(
        scope,
        evidence_references=[_support_evidence("evidence-1")],
        proposal={
            "proposed_claim": "A bounded claim.",
            "supporting_evidence_ref_ids": ["evidence-1"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
        claim_scope_basis="supplied_evidence",
    )

    result = _evaluate_support(payload).json()["result"]

    assert set(result) == {
        "claim_id",
        "claim_digest",
        "calibration_status",
        "conclusion_disposition",
        "qualification_required",
        "limitation_codes",
        "validated_supporting_evidence_ref_ids",
        "validated_counterevidence_ref_ids",
        "validated_material_exclusions",
        "validated_executed_derivation_ref_ids",
        "user_safe_summary",
    }
    assert "claim_scope_basis" not in result


def test_generic_claim_scope_basis_rejects_unknown_value():
    scope = _start_runtime()
    payload = _support_payload(scope, claim_scope_basis="provider_selected")

    response = _evaluate_support(payload)

    assert response.status_code == 422
    assert "literal_error" in response.text


@pytest.mark.parametrize(
    ("authority_update", "expected_limitation"),
    [
        ({}, "no_supporting_evidence"),
        ({"privacy_policy_allows_claim": False}, "privacy_constraint"),
        ({"consequence_policy_allows_claim": False}, "consequence_constraint"),
    ],
)
def test_generic_supplied_evidence_scope_preserves_hard_authority_blockers(
    authority_update: dict[str, object],
    expected_limitation: str,
):
    scope = _start_runtime()
    evidence = [] if not authority_update else [_support_evidence("evidence-1")]
    support_ids = [] if not authority_update else ["evidence-1"]
    payload = _support_payload(
        scope,
        evidence_references=evidence,
        proposal={
            "proposed_claim": "A bounded claim.",
            "supporting_evidence_ref_ids": support_ids,
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
        claim_scope_basis="supplied_evidence",
        complete_declared_scope_required=True,
        complete_declared_scope_established=False,
        **authority_update,
    )

    result = _evaluate_support(payload).json()["result"]

    assert result["calibration_status"] == "unsupported"
    assert result["conclusion_disposition"] == "withheld"
    assert expected_limitation in result["limitation_codes"]


def test_generic_omitted_required_material_evidence_withholds_claim():
    scope = _start_runtime()
    payload = _support_payload(
        scope,
        evidence_references=[
            _support_evidence("support-1"),
            _support_evidence(
                "material-counter-1",
                material_disclosure_required=True,
                material_role="counterevidence",
            ),
        ],
        proposal={
            "proposed_claim": "A claim that omits known material evidence.",
            "supporting_evidence_ref_ids": ["support-1"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
        claim_scope_basis="supplied_evidence",
    )

    result = _evaluate_support(payload).json()["result"]

    assert result["calibration_status"] == "unsupported"
    assert result["conclusion_disposition"] == "withheld"
    assert "material_evidence_omitted" in result["limitation_codes"]


def test_generic_supplied_evidence_scope_cannot_misclassify_counterevidence():
    scope = _start_runtime()
    payload = _support_payload(
        scope,
        evidence_references=[
            _support_evidence(
                "material-counter",
                material_disclosure_required=True,
                material_role="counterevidence",
            )
        ],
        proposal={
            "proposed_claim": "A bounded claim.",
            "supporting_evidence_ref_ids": ["material-counter"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
        claim_scope_basis="supplied_evidence",
    )

    result = _evaluate_support(payload).json()["result"]

    assert result["calibration_status"] == "unsupported"
    assert result["conclusion_disposition"] == "withheld"
    assert "material_counterevidence_misclassified" in result["limitation_codes"]


def test_generic_declared_material_counterevidence_requires_qualification():
    scope = _start_runtime()
    payload = _support_payload(
        scope,
        evidence_references=[
            _support_evidence("support-1"),
            _support_evidence(
                "material-counter-1",
                material_disclosure_required=True,
                material_role="counterevidence",
            ),
        ],
        proposal={
            "proposed_claim": "A claim with disclosed counterevidence.",
            "supporting_evidence_ref_ids": ["support-1"],
            "counterevidence_ref_ids": ["material-counter-1"],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
    )

    result = _evaluate_support(payload).json()["result"]

    assert result["calibration_status"] == "limited"
    assert result["conclusion_disposition"] == "qualified"
    assert result["qualification_required"] is True
    assert "material_counterevidence_present" in result["limitation_codes"]


def test_generic_actual_system_derivation_is_validated_and_order_independent():
    scope = _start_runtime()
    evidence = [_support_evidence("evidence-1")]
    derivation = _executed_derivation(scope, evidence_ref_ids=["evidence-1"])
    proposal = {
        "proposed_claim": "The deterministic result is 0.625.",
        "supporting_evidence_ref_ids": ["evidence-1"],
        "counterevidence_ref_ids": [],
        "material_exclusions": [],
        "executed_derivation_ref_ids": ["derivation-1"],
    }
    payload = _support_payload(
        scope,
        evidence_references=evidence,
        proposal=proposal,
        executed_derivations=[derivation],
    )

    first = _evaluate_support(payload)
    second = _evaluate_support(payload)

    assert first.status_code == 200
    assert first.json()["result"] == second.json()["result"]
    assert first.json()["result"]["validated_executed_derivation_ref_ids"] == [
        "derivation-1"
    ]
    assert first.json()["result"]["conclusion_disposition"] == "allowed"


@pytest.mark.parametrize(
    ("derivation_update", "proposal_refs", "expected_error"),
    [
        ({}, ["missing-derivation"], "proposal_derivation_reference_not_executed"),
        (
            {"execution_status": "proposed"},
            ["derivation-1"],
            "literal_error",
        ),
        (
            {"runtime_turn_id": "rtturn-other"},
            ["derivation-1"],
            "derivation_turn_mismatch",
        ),
    ],
)
def test_generic_fabricated_or_mismatched_derivations_are_rejected(
    derivation_update: dict[str, object],
    proposal_refs: list[str],
    expected_error: str,
):
    scope = _start_runtime()
    derivation = _executed_derivation(scope)
    derivation.update(derivation_update)
    payload = _support_payload(
        scope,
        proposal={
            "proposed_claim": "A bounded derived claim.",
            "supporting_evidence_ref_ids": [],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": proposal_refs,
        },
        executed_derivations=[derivation],
    )

    response = _evaluate_support(payload)

    assert response.status_code == 422
    assert expected_error in response.text


def test_generic_interpretation_dependent_arithmetic_remains_qualified():
    scope = _start_runtime()
    evidence = [_support_evidence("semantic-source-1")]
    derivation = _executed_derivation(
        scope,
        evidence_ref_ids=["semantic-source-1"],
        input_basis="model_interpreted",
    )
    payload = _support_payload(
        scope,
        evidence_references=evidence,
        proposal={
            "proposed_claim": "The mechanically derived result is 0.625.",
            "supporting_evidence_ref_ids": ["semantic-source-1"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": ["derivation-1"],
        },
        executed_derivations=[derivation],
    )

    result = _evaluate_support(payload).json()["result"]

    assert result["calibration_status"] == "limited"
    assert result["conclusion_disposition"] == "qualified"
    assert "interpretation_dependent_derivation" in result["limitation_codes"]
    assert "verified" not in result["user_safe_summary"].lower()


@pytest.mark.parametrize(
    ("policy_field", "limitation"),
    [
        ("privacy_policy_allows_claim", "privacy_constraint"),
        ("consequence_policy_allows_claim", "consequence_constraint"),
    ],
)
def test_generic_system_policy_constraints_withhold_claim(
    policy_field: str,
    limitation: str,
):
    scope = _start_runtime()
    payload = _support_payload(
        scope,
        evidence_references=[_support_evidence("evidence-1")],
        proposal={
            "proposed_claim": "A bounded claim.",
            "supporting_evidence_ref_ids": ["evidence-1"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
        **{policy_field: False},
    )

    result = _evaluate_support(payload).json()["result"]

    assert result["conclusion_disposition"] == "withheld"
    assert limitation in result["limitation_codes"]


@pytest.mark.parametrize("claim_scope_basis", ["declared_scope", "supplied_evidence"])
def test_generic_runtime_event_is_bounded_and_excludes_claim_content_and_refs(
    claim_scope_basis: str,
):
    scope = _start_runtime()
    private_claim = "PRIVATE CLAIM SENTINEL"
    payload = _support_payload(
        scope,
        evidence_references=[_support_evidence("private-ref-1")],
        proposal={
            "proposed_claim": private_claim,
            "supporting_evidence_ref_ids": ["private-ref-1"],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
        claim_scope_basis=claim_scope_basis,
    )
    response = _evaluate_support(payload)
    assert response.status_code == 200

    diagnostics = client.get(f"/v1/runtime/sessions/{scope['runtime_session_id']}")
    event = next(
        item
        for item in diagnostics.json()["events"]
        if item["event_type"] == "claim_support_evaluated"
    )
    event_payload = event["event_payload_json"]
    serialized = json.dumps(event_payload, sort_keys=True)
    assert set(event_payload) == {
        "request_id",
        "runtime_session_id",
        "runtime_turn_id",
        "claim_scope_basis",
        "claim_id",
        "claim_digest",
        "calibration_status",
        "conclusion_disposition",
        "qualification_required",
        "limitation_codes",
        "supporting_evidence_count",
        "counterevidence_count",
        "material_exclusion_count",
        "executed_derivation_count",
        "interpretation_dependent_derivation",
    }
    assert event_payload["claim_scope_basis"] == claim_scope_basis
    assert private_claim not in serialized
    assert "private-ref-1" not in serialized
    assert "prompt" not in serialized.lower()
    assert "reasoning" not in serialized.lower()


def test_generic_unknown_session_and_turn_return_bounded_errors():
    scope = _start_runtime()
    missing_session_payload = _support_payload(scope)
    missing_session_payload["authority_context"]["runtime_session_id"] = (
        "rtsession-missing"
    )
    missing_session_payload["authority_context"]["runtime_turn_id"] = "rtturn-missing"
    missing_turn_payload = _support_payload(scope)
    missing_turn_payload["authority_context"]["runtime_turn_id"] = "rtturn-missing"

    missing_session = _evaluate_support(missing_session_payload)
    missing_turn = _evaluate_support(missing_turn_payload)

    assert missing_session.status_code == 404
    assert missing_session.json() == {"detail": "runtime_session_not_found"}
    assert missing_turn.status_code == 404
    assert missing_turn.json() == {"detail": "runtime_turn_not_found"}


def test_generic_runtime_scope_mismatch_returns_bounded_error():
    scope = _start_runtime()
    payload = _support_payload(scope)
    payload["authority_context"]["owner_id"] = "owner-2"

    response = _evaluate_support(payload)

    assert response.status_code == 400
    assert response.json() == {"detail": "runtime_session_mismatch"}


@pytest.mark.parametrize("invalid_ref", [True, 503, "contains whitespace"])
def test_generic_proposal_rejects_coercive_or_malformed_reference_ids(
    invalid_ref: object,
):
    scope = _start_runtime()
    payload = _support_payload(
        scope,
        evidence_references=[_support_evidence("evidence-1")],
        proposal={
            "proposed_claim": "A bounded claim.",
            "supporting_evidence_ref_ids": [invalid_ref],
            "counterevidence_ref_ids": [],
            "material_exclusions": [],
            "executed_derivation_ref_ids": [],
        },
    )

    response = _evaluate_support(payload)

    assert response.status_code == 422
