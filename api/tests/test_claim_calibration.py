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
