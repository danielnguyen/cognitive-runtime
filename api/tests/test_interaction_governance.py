import httpx
import pytest
from main import app
from services.runtime_state import runtime_state_repository


def _base(**overrides):
    payload = {
        "request_id": "rid-governance",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
        "recent_messages": [],
    }
    payload.update(overrides)
    return payload


def _candidate(**overrides):
    candidate = {
        "source": "deterministic",
        "intent": "support_explanation",
        "confidence": 1.0,
        "target_mode": "immediate_previous",
        "new_verification_requested": False,
    }
    candidate.update(overrides)
    return candidate


async def _post(path: str, payload: dict[str, object]):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post(path, json=payload)


async def _get(path: str):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_tense_failure_report_suppresses_humor_and_commentary_and_uses_tactical_posture():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text="I think I broke the server and prod is failing"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "tense_debugging"
    assert result["humor_allowed"] is False
    assert result["commentary_allowed"] is False
    assert result["response_posture"] == "tactical"


@pytest.mark.asyncio
async def test_normal_question_classifies_as_question_and_stays_non_actioning():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text="What does this function do?"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "question"
    assert result["action_allowed"] is False
    assert result["humor_allowed"] is False
    assert result["commentary_allowed"] is False


@pytest.mark.asyncio
async def test_recent_messages_latest_user_text_is_used_when_current_user_text_is_omitted():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            recent_messages=[
                {"role": "user", "content": "Can you take a look?"},
                {"role": "assistant", "content": "What seems wrong?"},
                {
                    "role": "user",
                    "content": "I think I broke the server and prod is failing",
                },
            ],
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "tense_debugging"
    assert result["humor_allowed"] is False
    assert result["commentary_allowed"] is False
    assert result["response_posture"] == "tactical"


@pytest.mark.asyncio
async def test_playful_low_risk_message_allows_commentary_and_humor_without_forcing_output():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text="lol roast my tiny todo list"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "joke_or_playful"
    assert result["tension_level"] == "low"
    assert result["humor_allowed"] is True
    assert result["commentary_allowed"] is True
    assert result["action_allowed"] is False
    assert result["requires_confirmation"] is False
    assert result["privacy_sensitivity_hint"] == "normal"
    assert result["response_posture"] == "playful"


@pytest.mark.parametrize(
    ("current_user_text", "expected_kind"),
    [
        ("What does this function do?", "question"),
        ("rename this variable to count", "command"),
        ("brainstorm options for this name", "brainstorm"),
        ("Ugh, this sucks and I am upset.", "vent_or_expression"),
        ("That was wrong in the report.", "mistake_or_failure_report"),
        ("I think I broke the server and prod is failing", "tense_debugging"),
        ("Should I change payroll taxes?", "high_impact_decision"),
        ("Maybe later", "ambiguous"),
    ],
)
@pytest.mark.asyncio
async def test_non_playful_interactions_do_not_receive_commentary_permission(
    current_user_text: str,
    expected_kind: str,
):
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text=current_user_text),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == expected_kind
    assert result["commentary_allowed"] is False


@pytest.mark.asyncio
async def test_destructive_frustrated_phrase_requires_confirmation_and_stays_non_actioning():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text="nuke this"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "ambiguous"
    assert result["action_allowed"] is False
    assert result["requires_confirmation"] is True


@pytest.mark.asyncio
async def test_direct_low_risk_command_classifies_as_command_but_is_not_actionable():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text="rename this variable to count"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "command"
    assert result["literal_command_confidence"] >= 0.75
    assert result["action_allowed"] is False


@pytest.mark.asyncio
async def test_runtime_event_payload_is_summarized_only():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text="I think I broke the server and prod is failing badly"),
    )

    assert response.status_code == 200
    runtime_session_id = response.json()["runtime_session_id"]

    diagnostics = await _get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    events = diagnostics.json()["events"]
    governance_event = next(
        event for event in events if event["event_type"] == "interaction_governance_evaluated"
    )
    payload = governance_event["event_payload_json"]
    assert set(payload.keys()) == {
        "request_id",
        "interaction_kind",
        "response_posture",
        "commentary_allowed",
        "humor_allowed",
        "action_allowed",
        "requires_confirmation",
        "reason_summary",
        "history_followup_policy",
    }
    assert set(payload["history_followup_policy"]) == {
        "status",
        "intent",
        "candidate_source",
        "target_mode",
        "explanation_kind",
        "acquisition_question",
        "history_lookup_allowed",
        "new_verification_requested",
        "new_verification_allowed_after_history_resolution",
        "clarification_required",
        "confidence_band",
        "reason_codes",
    }
    assert "current_user_text" not in str(payload)
    assert "broke the server" not in str(payload)


@pytest.mark.asyncio
async def test_ambiguous_fallback_is_conservative_and_non_actioning():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "ambiguous"
    assert result["action_allowed"] is False
    assert result["humor_allowed"] is False
    assert result["commentary_allowed"] is False


@pytest.mark.asyncio
async def test_runtime_turn_integration_updates_intent_class_and_records_event():
    started = await _post(
        "/v1/runtime/turns/start",
        {
            "request_id": "rid-turn-start",
            "owner_id": "owner",
            "conversation_id": "conv-1",
            "surface": "dev",
            "input_message_id": "msg-1",
        },
    )
    assert started.status_code == 200
    runtime_session_id = started.json()["runtime_session"]["runtime_session_id"]
    runtime_turn_id = started.json()["runtime_turn"]["runtime_turn_id"]

    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            request_id="rid-turn-governance",
            runtime_session_id=runtime_session_id,
            runtime_turn_id=runtime_turn_id,
            current_user_text="rename this variable to count",
        ),
    )

    assert response.status_code == 200
    diagnostics = await _get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    body = diagnostics.json()
    assert body["latest_turn"]["intent_class"] == "action_command"
    assert any(
        event["event_type"] == "interaction_governance_evaluated" for event in body["events"]
    )


async def _governance_turn_diagnostics(
    *,
    request_id: str,
    current_user_text: str,
    recent_messages: list[dict[str, str]] | None = None,
    history_followup_candidate: dict[str, object] | None = None,
):
    started = await _post(
        "/v1/runtime/turns/start",
        {
            "request_id": f"{request_id}-start",
            "owner_id": "owner",
            "conversation_id": "conv-1",
            "surface": "dev",
            "input_message_id": "msg-1",
        },
    )
    assert started.status_code == 200
    runtime_session_id = started.json()["runtime_session"]["runtime_session_id"]
    runtime_turn_id = started.json()["runtime_turn"]["runtime_turn_id"]

    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            request_id=request_id,
            runtime_session_id=runtime_session_id,
            runtime_turn_id=runtime_turn_id,
            current_user_text=current_user_text,
            recent_messages=recent_messages or [],
            history_followup_candidate=history_followup_candidate,
        ),
    )

    assert response.status_code == 200
    diagnostics = await _get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    return response.json()["result"], diagnostics.json()


@pytest.mark.asyncio
async def test_clarification_request_projection_uses_question_form_clarification_markers():
    result, diagnostics = await _governance_turn_diagnostics(
        request_id="rid-clarification",
        current_user_text="What do you mean by policy hints?",
    )

    assert result["interaction_kind"] == "question"
    assert diagnostics["latest_turn"]["intent_class"] == "clarification_request"


@pytest.mark.asyncio
async def test_confirmation_response_projection_requires_immediately_preceding_assistant_question():
    result, diagnostics = await _governance_turn_diagnostics(
        request_id="rid-confirmation-adjacent",
        current_user_text="Yes.",
        recent_messages=[
            {"role": "assistant", "content": "Do you want me to proceed?"},
        ],
    )

    assert result["interaction_kind"] == "ambiguous"
    assert diagnostics["latest_turn"]["intent_class"] == "confirmation_response"


@pytest.mark.asyncio
async def test_older_assistant_question_separated_by_user_message_does_not_enable_confirmation_response(  # noqa: E501
):
    result, diagnostics = await _governance_turn_diagnostics(
        request_id="rid-confirmation-stale",
        current_user_text="Yes.",
        recent_messages=[
            {"role": "assistant", "content": "Do you want me to proceed?"},
            {"role": "user", "content": "Let's park that for now."},
        ],
    )

    assert result["interaction_kind"] == "ambiguous"
    assert diagnostics["latest_turn"]["intent_class"] == "low_confidence_unclear"


@pytest.mark.asyncio
async def test_continuation_projection_requires_immediately_preceding_assistant_message():
    result, diagnostics = await _governance_turn_diagnostics(
        request_id="rid-continuation-adjacent",
        current_user_text="Go on.",
        recent_messages=[
            {"role": "assistant", "content": "The first issue is in the retry path."},
        ],
    )

    assert result["interaction_kind"] == "ambiguous"
    assert diagnostics["latest_turn"]["intent_class"] == "continuation"


@pytest.mark.asyncio
async def test_older_assistant_message_separated_by_user_message_does_not_enable_continuation():
    result, diagnostics = await _governance_turn_diagnostics(
        request_id="rid-continuation-stale",
        current_user_text="Continue.",
        recent_messages=[
            {"role": "assistant", "content": "The first issue is in the retry path."},
            {"role": "user", "content": "Okay, noted."},
        ],
    )

    assert result["interaction_kind"] == "ambiguous"
    assert diagnostics["latest_turn"]["intent_class"] == "low_confidence_unclear"


@pytest.mark.asyncio
async def test_unsupported_ambiguous_reply_still_falls_back_to_low_confidence_unclear():
    result, diagnostics = await _governance_turn_diagnostics(
        request_id="rid-ambiguous-fallback",
        current_user_text="Okay.",
        recent_messages=[
            {"role": "assistant", "content": "Do you want me to proceed?"},
        ],
    )

    assert result["interaction_kind"] == "ambiguous"
    assert diagnostics["latest_turn"]["intent_class"] == "low_confidence_unclear"


@pytest.mark.asyncio
async def test_existing_supported_intent_class_mappings_remain_unchanged():
    cases = [
        (
            "rid-information-request",
            "What does this function do?",
            [],
            "information_request",
        ),
        (
            "rid-action-command",
            "rename this variable to count",
            [],
            "action_command",
        ),
        (
            "rid-correction",
            "Actually, the wrong service failed; I meant the worker queue.",
            [],
            "correction",
        ),
        (
            "rid-venting",
            "Ugh, this sucks and I'm upset.",
            [],
            "venting_signal",
        ),
    ]

    for request_id, current_user_text, recent_messages, expected_intent_class in cases:
        _, diagnostics = await _governance_turn_diagnostics(
            request_id=request_id,
            current_user_text=current_user_text,
            recent_messages=recent_messages,
        )
        assert diagnostics["latest_turn"]["intent_class"] == expected_intent_class
        completion = await _post(
            "/v1/runtime/turns/complete",
            {
                "request_id": f"{request_id}-complete",
                "runtime_session_id": diagnostics["runtime_session"][
                    "runtime_session_id"
                ],
                "runtime_turn_id": diagnostics["latest_turn"]["runtime_turn_id"],
                "turn_status": "completed",
            },
        )
        assert completion.status_code == 200


@pytest.mark.asyncio
async def test_no_candidate_preserves_generic_result_and_returns_not_applicable_policy():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(current_user_text="What does this function do?"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["interaction_kind"] == "question"
    assert result["response_posture"] == "direct"
    assert result["confidence"] == 0.8
    assert result["history_followup_policy"] == {
        "status": "not_applicable",
        "intent": None,
        "candidate_source": None,
        "target_mode": None,
        "explanation_kind": None,
        "acquisition_question": None,
        "history_lookup_allowed": False,
        "new_verification_requested": False,
        "new_verification_allowed_after_history_resolution": False,
        "clarification_required": False,
        "confidence_band": "not_applicable",
        "reason_codes": ["no_candidate"],
    }


@pytest.mark.asyncio
async def test_unknown_top_level_fields_remain_backward_compatible():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="What does this function do?",
            future_optional_field={"ignored": True},
        ),
    )

    assert response.status_code == 200
    assert response.json()["result"]["interaction_kind"] == "question"


def test_interaction_governance_route_is_not_duplicated_or_replaced():
    paths = [
        route.path
        for route in app.routes
        if route.path == "/v1/runtime/interaction-governance/evaluate"
    ]

    assert paths == ["/v1/runtime/interaction-governance/evaluate"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intent", "new_verification_requested"),
    [
        ("not_history_followup", False),
        ("support_explanation", False),
        ("acquisition_checked", False),
        ("acquisition_coverage", False),
        ("acquisition_gaps", False),
        ("new_verification_request", True),
        ("ambiguous_history_followup", False),
    ],
)
async def test_all_closed_history_intent_labels_validate(
    intent,
    new_verification_requested,
):
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="Current user turn",
            history_followup_candidate=_candidate(
                intent=intent,
                new_verification_requested=new_verification_requested,
            ),
        ),
    )

    assert response.status_code == 200
    assert response.json()["result"]["history_followup_policy"]["intent"] == intent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("intent", "history_guess"),
        ("source", "provider"),
        ("target_mode", "selected_record"),
    ],
)
async def test_unknown_candidate_enums_are_rejected(field, value):
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="Current user turn",
            history_followup_candidate=_candidate(**{field: value}),
        ),
    )

    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_field",
    [
        "prose",
        "explanation",
        "reasoning",
        "record_id",
        "message_id",
        "assistant_message_id",
        "request_id",
        "claim_id",
        "trace_id",
        "manifest_id",
        "acquisition_manifest_id",
        "classifier_prompt",
        "source_names",
        "source_references",
        "evidence",
        "retained_content",
        "proposed_wording",
        "model_response",
        "provider",
        "provider_response",
    ],
)
async def test_candidate_extra_content_identifiers_and_provider_fields_are_rejected(
    extra_field,
):
    candidate = _candidate()
    candidate[extra_field] = "forbidden"
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="Current user turn",
            history_followup_candidate=candidate,
        ),
    )

    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("confidence", [-0.000001, 1.000001])
async def test_candidate_confidence_must_be_bounded(confidence):
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="Current user turn",
            history_followup_candidate=_candidate(
                source="classifier",
                confidence=confidence,
            ),
        ),
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_deterministic_candidate_requires_full_confidence():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="Current user turn",
            history_followup_candidate=_candidate(confidence=0.999999),
        ),
    )

    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intent", "new_verification_requested"),
    [
        ("new_verification_request", False),
        ("not_history_followup", True),
        ("ambiguous_history_followup", True),
    ],
)
async def test_inconsistent_candidate_verification_flags_are_rejected(
    intent,
    new_verification_requested,
):
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="Current user turn",
            history_followup_candidate=_candidate(
                intent=intent,
                new_verification_requested=new_verification_requested,
            ),
        ),
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_candidate_requires_current_user_turn():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(history_followup_candidate=_candidate()),
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_latest_user_message_satisfies_candidate_current_turn_requirement():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            recent_messages=[{"role": "user", "content": "What was that based on?"}],
            history_followup_candidate=_candidate(),
        ),
    )

    assert response.status_code == 200
    assert response.json()["result"]["history_followup_policy"]["status"] == "accepted"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intent", "explanation_kind", "acquisition_question", "verification_requested"),
    [
        ("support_explanation", "support", None, False),
        ("acquisition_checked", "acquisition", "checked", False),
        ("acquisition_coverage", "acquisition", "coverage", False),
        ("acquisition_gaps", "acquisition", "gaps", False),
        ("new_verification_request", "support", None, True),
    ],
)
async def test_actionable_deterministic_candidates_map_to_closed_policy(
    intent,
    explanation_kind,
    acquisition_question,
    verification_requested,
):
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="Current user turn only",
            recent_messages=[],
            history_followup_candidate=_candidate(
                intent=intent,
                new_verification_requested=verification_requested,
            ),
        ),
    )

    assert response.status_code == 200
    policy = response.json()["result"]["history_followup_policy"]
    assert policy["status"] == "accepted"
    assert policy["explanation_kind"] == explanation_kind
    assert policy["acquisition_question"] == acquisition_question
    assert policy["history_lookup_allowed"] is True
    assert policy["new_verification_requested"] is verification_requested
    assert policy["new_verification_allowed_after_history_resolution"] is (
        verification_requested
    )
    assert policy["reason_codes"] == ["deterministic_candidate_accepted"]


@pytest.mark.asyncio
async def test_compound_acquisition_and_verification_preserves_question_and_gate():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="What did you check? Verify it again now.",
            history_followup_candidate=_candidate(
                intent="acquisition_checked",
                new_verification_requested=True,
            ),
        ),
    )

    policy = response.json()["result"]["history_followup_policy"]
    assert policy["status"] == "accepted"
    assert policy["acquisition_question"] == "checked"
    assert policy["new_verification_requested"] is True
    assert policy["new_verification_allowed_after_history_resolution"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("confidence", "status", "band", "clarification_required"),
    [
        (0.85, "accepted", "high", False),
        (0.849999, "clarification_required", "medium", True),
        (0.60, "clarification_required", "medium", True),
        (0.599999, "rejected", "low", False),
    ],
)
async def test_classifier_confidence_boundaries(
    confidence,
    status,
    band,
    clarification_required,
):
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="What was that based on?",
            history_followup_candidate=_candidate(
                source="classifier",
                confidence=confidence,
            ),
        ),
    )

    assert response.status_code == 200
    policy = response.json()["result"]["history_followup_policy"]
    assert policy["status"] == status
    assert policy["confidence_band"] == band
    assert policy["clarification_required"] is clarification_required
    assert policy["history_lookup_allowed"] is (status == "accepted")
    assert policy["new_verification_allowed_after_history_resolution"] is False


@pytest.mark.asyncio
async def test_high_confidence_ambiguous_candidate_always_requires_clarification():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="Can you explain that?",
            history_followup_candidate=_candidate(
                source="classifier",
                intent="ambiguous_history_followup",
                confidence=0.99,
            ),
        ),
    )

    policy = response.json()["result"]["history_followup_policy"]
    assert policy["status"] == "clarification_required"
    assert policy["clarification_required"] is True
    assert policy["history_lookup_allowed"] is False


@pytest.mark.asyncio
async def test_high_confidence_not_history_candidate_preserves_ordinary_handling():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="What does this function do?",
            history_followup_candidate=_candidate(
                source="classifier",
                intent="not_history_followup",
                confidence=0.99,
            ),
        ),
    )

    result = response.json()["result"]
    assert result["interaction_kind"] == "question"
    assert result["history_followup_policy"]["status"] == "not_applicable"
    assert result["history_followup_policy"]["history_lookup_allowed"] is False


@pytest.mark.asyncio
async def test_explicit_reference_is_routed_without_immediate_history_authority():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="Explain the answer I explicitly selected.",
            history_followup_candidate=_candidate(target_mode="explicit_reference"),
        ),
    )

    policy = response.json()["result"]["history_followup_policy"]
    assert policy["status"] == "explicit_reference"
    assert policy["target_mode"] == "explicit_reference"
    assert policy["history_lookup_allowed"] is False
    assert policy["new_verification_allowed_after_history_resolution"] is False
    assert policy["reason_codes"] == ["explicit_reference_routed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "confidence"),
    [("classifier", 0.99), ("deterministic", 1.0)],
)
async def test_not_history_precedes_explicit_reference_and_preserves_generic_intent(
    source,
    confidence,
):
    result, diagnostics = await _governance_turn_diagnostics(
        request_id=f"rid-not-history-explicit-{source}",
        current_user_text="rename this variable to count",
        history_followup_candidate=_candidate(
            source=source,
            intent="not_history_followup",
            confidence=confidence,
            target_mode="explicit_reference",
        ),
    )

    policy = result["history_followup_policy"]
    assert policy["status"] == "not_applicable"
    assert policy["history_lookup_allowed"] is False
    assert policy["new_verification_allowed_after_history_resolution"] is False
    assert policy["reason_codes"] == ["not_history_candidate"]
    assert diagnostics["latest_turn"]["intent_class"] == "action_command"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "confidence"),
    [("classifier", 0.99), ("deterministic", 1.0)],
)
async def test_ambiguity_precedes_explicit_reference_and_projects_clarification(
    source,
    confidence,
):
    result, diagnostics = await _governance_turn_diagnostics(
        request_id=f"rid-ambiguous-explicit-{source}",
        current_user_text="Can you explain that?",
        history_followup_candidate=_candidate(
            source=source,
            intent="ambiguous_history_followup",
            confidence=confidence,
            target_mode="explicit_reference",
        ),
    )

    policy = result["history_followup_policy"]
    assert policy["status"] == "clarification_required"
    assert policy["history_lookup_allowed"] is False
    assert policy["new_verification_allowed_after_history_resolution"] is False
    assert policy["clarification_required"] is True
    assert policy["reason_codes"] == ["ambiguous_candidate"]
    assert diagnostics["latest_turn"]["intent_class"] == (
        "ambiguous_history_followup"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("confidence", "status", "clarification_required", "reason_code"),
    [
        (0.599999, "rejected", False, "classifier_confidence_rejected"),
        (
            0.60,
            "clarification_required",
            True,
            "classifier_confidence_requires_clarification",
        ),
        (
            0.849999,
            "clarification_required",
            True,
            "classifier_confidence_requires_clarification",
        ),
        (0.85, "explicit_reference", False, "explicit_reference_routed"),
    ],
)
async def test_classifier_confidence_precedes_explicit_reference_target_mode(
    confidence,
    status,
    clarification_required,
    reason_code,
):
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="What did you check? Verify it again now.",
            history_followup_candidate=_candidate(
                source="classifier",
                intent="acquisition_checked",
                confidence=confidence,
                target_mode="explicit_reference",
                new_verification_requested=True,
            ),
        ),
    )

    assert response.status_code == 200
    policy = response.json()["result"]["history_followup_policy"]
    assert policy["status"] == status
    assert policy["clarification_required"] is clarification_required
    assert policy["history_lookup_allowed"] is False
    assert policy["new_verification_requested"] is True
    assert policy["new_verification_allowed_after_history_resolution"] is False
    assert policy["reason_codes"] == [reason_code]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent",
    [
        "support_explanation",
        "acquisition_checked",
        "acquisition_coverage",
        "acquisition_gaps",
        "new_verification_request",
    ],
)
async def test_accepted_history_candidate_updates_exact_runtime_turn_intent(intent):
    _, diagnostics = await _governance_turn_diagnostics(
        request_id=f"rid-history-{intent}",
        current_user_text="Current history follow-up",
        history_followup_candidate=_candidate(
            intent=intent,
            new_verification_requested=intent == "new_verification_request",
        ),
    )

    assert diagnostics["latest_turn"]["intent_class"] == intent


@pytest.mark.asyncio
async def test_history_clarification_updates_runtime_turn_intent():
    _, diagnostics = await _governance_turn_diagnostics(
        request_id="rid-history-clarification",
        current_user_text="Can you explain that?",
        history_followup_candidate=_candidate(
            source="classifier",
            confidence=0.70,
        ),
    )

    assert diagnostics["latest_turn"]["intent_class"] == (
        "ambiguous_history_followup"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate",
    [
        None,
        _candidate(source="classifier", confidence=0.40),
        _candidate(
            source="classifier",
            intent="not_history_followup",
            confidence=0.99,
        ),
        _candidate(target_mode="explicit_reference"),
    ],
)
async def test_non_applicable_and_rejected_history_preserve_generic_intent(candidate):
    _, diagnostics = await _governance_turn_diagnostics(
        request_id=f"rid-generic-{candidate is None}-{str(candidate)[:12]}",
        current_user_text="rename this variable to count",
        history_followup_candidate=candidate,
    )

    assert diagnostics["latest_turn"]["intent_class"] == "action_command"


@pytest.mark.asyncio
async def test_history_projection_modifies_only_the_requested_runtime_turn():
    first = await _post(
        "/v1/runtime/turns/start",
        {
            "request_id": "rid-unrelated-first",
            "owner_id": "owner-unrelated",
            "conversation_id": "conv-unrelated",
            "surface": "dev",
            "input_message_id": "msg-first",
            "intent_class": "existing_intent",
        },
    )
    assert first.status_code == 200
    first_turn_id = first.json()["runtime_turn"]["runtime_turn_id"]
    completion = await _post(
        "/v1/runtime/turns/complete",
        {
            "request_id": "rid-unrelated-first-complete",
            "runtime_session_id": first.json()["runtime_session"]["runtime_session_id"],
            "runtime_turn_id": first_turn_id,
            "turn_status": "completed",
        },
    )
    assert completion.status_code == 200
    second = await _post(
        "/v1/runtime/turns/start",
        {
            "request_id": "rid-unrelated-second",
            "owner_id": "owner-unrelated",
            "conversation_id": "conv-unrelated",
            "surface": "dev",
            "input_message_id": "msg-second",
            "intent_class": "second_existing_intent",
        },
    )
    assert second.status_code == 200
    runtime_session_id = second.json()["runtime_session"]["runtime_session_id"]
    second_turn_id = second.json()["runtime_turn"]["runtime_turn_id"]

    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            request_id="rid-unrelated-evaluate",
            owner_id="owner-unrelated",
            conversation_id="conv-unrelated",
            runtime_session_id=runtime_session_id,
            runtime_turn_id=second_turn_id,
            current_user_text="What was that based on?",
            history_followup_candidate=_candidate(),
        ),
    )

    assert response.status_code == 200
    repository = runtime_state_repository()
    assert repository.turn_by_id(first_turn_id).intent_class == "existing_intent"
    assert repository.turn_by_id(second_turn_id).intent_class == "support_explanation"


@pytest.mark.asyncio
async def test_runtime_event_history_summary_is_bounded_and_private():
    current_sentinel = "PRIVATE CURRENT USER SENTINEL"
    assistant_sentinel = "PRIVATE RECENT ASSISTANT SENTINEL"
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            request_id="rid-private-history-event",
            current_user_text=current_sentinel,
            recent_messages=[{"role": "assistant", "content": assistant_sentinel}],
            history_followup_candidate=_candidate(
                intent="acquisition_checked",
                new_verification_requested=True,
            ),
        ),
    )
    diagnostics = await _get(
        f"/v1/runtime/sessions/{response.json()['runtime_session_id']}"
    )
    event = next(
        item
        for item in diagnostics.json()["events"]
        if item["event_type"] == "interaction_governance_evaluated"
        and item["event_payload_json"]["request_id"] == "rid-private-history-event"
    )
    summary = event["event_payload_json"]["history_followup_policy"]

    assert set(summary) == {
        "status",
        "intent",
        "candidate_source",
        "target_mode",
        "explanation_kind",
        "acquisition_question",
        "history_lookup_allowed",
        "new_verification_requested",
        "new_verification_allowed_after_history_resolution",
        "clarification_required",
        "confidence_band",
        "reason_codes",
    }
    serialized = str(event["event_payload_json"])
    assert current_sentinel not in serialized
    assert assistant_sentinel not in serialized
    for forbidden in (
        "record_id",
        "message_id",
        "claim_id",
        "trace_id",
        "manifest_id",
        "provider",
        "model_response",
        "source_references",
        "exception",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_identical_inputs_produce_identical_complete_policy_results():
    payload = _base(
        request_id="rid-deterministic-policy",
        current_user_text="What did you check? Verify it again now.",
        history_followup_candidate=_candidate(
            intent="acquisition_checked",
            new_verification_requested=True,
        ),
    )

    first = await _post("/v1/runtime/interaction-governance/evaluate", payload)
    second = await _post("/v1/runtime/interaction-governance/evaluate", payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["result"]["history_followup_policy"] == second.json()["result"][
        "history_followup_policy"
    ]


@pytest.mark.asyncio
async def test_policy_result_never_claims_history_or_external_work_occurred():
    response = await _post(
        "/v1/runtime/interaction-governance/evaluate",
        _base(
            current_user_text="Verify that again now.",
            history_followup_candidate=_candidate(
                intent="new_verification_request",
                new_verification_requested=True,
            ),
        ),
    )

    policy = response.json()["result"]["history_followup_policy"]
    assert policy["status"] == "accepted"
    assert policy["history_lookup_allowed"] is True
    assert policy["new_verification_allowed_after_history_resolution"] is True
    assert set(policy) == {
        "status",
        "intent",
        "candidate_source",
        "target_mode",
        "explanation_kind",
        "acquisition_question",
        "history_lookup_allowed",
        "new_verification_requested",
        "new_verification_allowed_after_history_resolution",
        "clarification_required",
        "confidence_band",
        "reason_codes",
    }
    serialized = str(policy)
    for forbidden in (
        "record_exists",
        "record",
        "answer",
        "historical_explanation",
        "provider_call",
        "bms_call",
        "dsa_call",
        "retrieval_call",
        "verification_performed",
    ):
        assert forbidden not in serialized
