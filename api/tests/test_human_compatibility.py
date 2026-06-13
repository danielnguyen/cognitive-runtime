from __future__ import annotations

import asyncio
import inspect

import pytest
from pydantic import ValidationError

import main as main_module
from main import (
    runtime_human_compatibility_diagnostics,
    runtime_human_compatibility_review,
)
from models import HumanCompatibilityDiagnosticsRequest, HumanCompatibilityReviewRequest
from services import human_compatibility


def _base() -> dict[str, object]:
    return {
        "request_id": "human-compatibility-review-request",
        "owner_id": "owner",
    }


def _review_payload(**overrides) -> dict[str, object]:
    payload: dict[str, object] = {
        **_base(),
        "feature_ref": "human_compatibility.review.surface",
        "spec_ref": "human_compatibility_v1",
        "review_surfaces": ["social_memory"],
        "proposed_behavior_summary": "Use social memory as a bounded review surface.",
        "risk_level": "medium",
        "review_notes": "Operator review for human compatibility trace coverage.",
        "mitigations_json": {"status": "not_required"},
        "runtime_turn_id": "rtturn_hc_1",
        "interaction_risk_flags": [],
    }
    payload.update(overrides)
    return payload


def test_creates_review_and_includes_all_human_compatibility_principles():
    response = asyncio.run(
        runtime_human_compatibility_review(
            HumanCompatibilityReviewRequest.model_validate(_review_payload())
        )
    )

    body = response.model_dump()
    assert body["review_result"] == "approved"
    assert body["review_id"].startswith("hcrev_")
    assert body["flags_recorded"] == 0
    assert body["principles_checked"] == [
        "usefulness_over_attachment",
        "continuity_without_possession",
        "warmth_without_reciprocity_claims",
        "transparency_over_hidden_influence",
        "restraint_over_intensity",
        "user_agency_first",
    ]


def test_records_risk_flags():
    response = asyncio.run(
        runtime_human_compatibility_review(
            HumanCompatibilityReviewRequest.model_validate(
                _review_payload(
            interaction_risk_flags=[
                {
                    "flag_id": "flag-1",
                    "risk_type": "over_personalization",
                    "severity": "medium",
                    "triggering_policy": "policy_trace",
                },
                {
                    "flag_id": "flag-2",
                    "risk_type": "intensity_escalation",
                    "severity": "medium",
                    "triggering_policy": "policy_trace",
                },
            ]
                )
            )
        )
    )
    diagnostics = asyncio.run(
        runtime_human_compatibility_diagnostics(
            HumanCompatibilityDiagnosticsRequest.model_validate(
                {**_base(), "feature_ref": "human_compatibility.review.surface", "runtime_turn_id": "rtturn_hc_1"}
            )
        )
    )

    assert response.flags_recorded == 2
    flags = diagnostics.model_dump()["interaction_risk_flags"]
    assert [flag["flag_id"] for flag in flags] == ["flag-1", "flag-2"]


def test_high_risk_review_requires_mitigations():
    with pytest.raises(ValidationError) as exc_info:
        HumanCompatibilityReviewRequest.model_validate(
            _review_payload(risk_level="high", mitigations_json={})
        )

    assert "human_compatibility_mitigations_required_for_high_risk" in str(exc_info.value)


def test_high_risk_review_with_mitigations_requires_human_review():
    response = asyncio.run(
        runtime_human_compatibility_review(
            HumanCompatibilityReviewRequest.model_validate(
                _review_payload(
            risk_level="high",
            mitigations_json={"restraint": "manual signoff required"},
            interaction_risk_flags=[
                {
                    "risk_type": "over_personalization",
                    "severity": "high",
                    "triggering_policy": "manual_review",
                }
            ],
                )
            )
        )
    )

    body = response.model_dump()
    assert body["review_result"] == "requires_human_review"
    assert body["mitigations_required"] is True


def test_unknown_surface_rejected():
    with pytest.raises(ValidationError):
        HumanCompatibilityReviewRequest.model_validate(
            _review_payload(review_surfaces=["unknown_surface"])
        )


def test_unknown_risk_type_rejected():
    with pytest.raises(ValidationError):
        HumanCompatibilityReviewRequest.model_validate(
            _review_payload(
            interaction_risk_flags=[
                {
                    "risk_type": "unknown_risk",
                    "severity": "medium",
                    "triggering_policy": "policy_trace",
                }
            ]
            )
        )


def test_unknown_risk_severity_rejected():
    with pytest.raises(ValidationError):
        HumanCompatibilityReviewRequest.model_validate(
            _review_payload(
            interaction_risk_flags=[
                {
                    "risk_type": "over_personalization",
                    "severity": "urgent",
                    "triggering_policy": "policy_trace",
                }
            ]
            )
        )


def test_empty_feature_ref_rejected():
    with pytest.raises(ValidationError):
        HumanCompatibilityReviewRequest.model_validate(_review_payload(feature_ref="   "))


def test_empty_proposed_behavior_summary_rejected():
    with pytest.raises(ValidationError):
        HumanCompatibilityReviewRequest.model_validate(
            _review_payload(proposed_behavior_summary="   ")
        )


def test_explicit_operator_labels_require_notes():
    with pytest.raises(ValidationError) as exc_info:
        HumanCompatibilityReviewRequest.model_validate(
            _review_payload(
            review_notes=None,
            interaction_risk_flags=[
                {
                    "risk_type": "attachment_pressure",
                    "severity": "medium",
                    "triggering_policy": "operator_label",
                }
            ],
            )
        )

    assert "human_compatibility_review_notes_required" in str(exc_info.value)


def test_diagnostics_returns_submitted_checklist_data_only():
    asyncio.run(
        runtime_human_compatibility_review(
            HumanCompatibilityReviewRequest.model_validate(
                _review_payload(
            review_id="review-diag",
            review_notes="Tagged for reciprocity_claim review.",
            interaction_risk_flags=[
                {
                    "flag_id": "flag-diag",
                    "risk_type": "reciprocity_claim",
                    "severity": "medium",
                    "triggering_policy": "operator_label",
                }
            ],
                )
            )
        )
    )

    response = asyncio.run(
        runtime_human_compatibility_diagnostics(
            HumanCompatibilityDiagnosticsRequest.model_validate(
                {**_base(), "feature_ref": "human_compatibility.review.surface", "runtime_turn_id": "rtturn_hc_1"}
            )
        )
    )

    body = response.model_dump()
    assert body["reviews"][0]["review_id"] == "review-diag"
    assert body["interaction_risk_flags"][0]["flag_id"] == "flag-diag"
    payload_text = str(body)
    assert "vulnerability" not in payload_text
    assert "psychology" not in payload_text
    assert "profile_score" not in payload_text


def test_supplied_review_and_flag_ids_are_idempotent_per_owner():
    payload = _review_payload(
        review_id="review-fixed",
        interaction_risk_flags=[
            {
                "flag_id": "flag-fixed",
                "risk_type": "over_personalization",
                "severity": "medium",
                "triggering_policy": "policy_trace",
            }
        ],
    )

    first = asyncio.run(
        runtime_human_compatibility_review(HumanCompatibilityReviewRequest.model_validate(payload))
    )
    second = asyncio.run(
        runtime_human_compatibility_review(HumanCompatibilityReviewRequest.model_validate(payload))
    )
    diagnostics = asyncio.run(
        runtime_human_compatibility_diagnostics(
            HumanCompatibilityDiagnosticsRequest.model_validate(
                {**_base(), "feature_ref": "human_compatibility.review.surface", "runtime_turn_id": "rtturn_hc_1"}
            )
        )
    )

    assert first.review_id == "review-fixed"
    assert second.review_id == "review-fixed"
    body = diagnostics.model_dump()
    assert len(body["reviews"]) == 1
    assert len(body["interaction_risk_flags"]) == 1


def test_omitted_flag_ids_generate_unique_flags():
    response = asyncio.run(
        runtime_human_compatibility_review(
            HumanCompatibilityReviewRequest.model_validate(
                _review_payload(
            interaction_risk_flags=[
                {
                    "risk_type": "over_personalization",
                    "severity": "medium",
                    "triggering_policy": "policy_trace",
                },
                {
                    "risk_type": "over_personalization",
                    "severity": "medium",
                    "triggering_policy": "policy_trace",
                },
            ]
                )
            )
        )
    )
    diagnostics = asyncio.run(
        runtime_human_compatibility_diagnostics(
            HumanCompatibilityDiagnosticsRequest.model_validate(
                {**_base(), "feature_ref": "human_compatibility.review.surface", "runtime_turn_id": "rtturn_hc_1"}
            )
        )
    )

    assert response.flags_recorded == 2
    flags = diagnostics.model_dump()["interaction_risk_flags"]
    assert len(flags) == 2
    assert flags[0]["flag_id"] != flags[1]["flag_id"]


def test_no_llm_or_model_call():
    def _explode(*args, **kwargs):
        raise AssertionError("unexpected_model_path")

    original_compile_policy = main_module.compile_policy
    original_resolve_interaction_contract = main_module.resolve_interaction_contract
    main_module.compile_policy = _explode
    main_module.resolve_interaction_contract = _explode
    try:
        response = asyncio.run(
            runtime_human_compatibility_review(
                HumanCompatibilityReviewRequest.model_validate(_review_payload())
            )
        )
    finally:
        main_module.compile_policy = original_compile_policy
        main_module.resolve_interaction_contract = original_resolve_interaction_contract

    assert response.review_result == "approved"


def test_no_bms_dependency():
    source = inspect.getsource(human_compatibility)

    assert "basic_memory_store" not in source
    assert "basic-memory-store" not in source
