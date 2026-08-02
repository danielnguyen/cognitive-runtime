from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from main import app
from models import SituatedPresenceEvaluateResponse, SituatedPresenceResult
from pydantic import ValidationError
from services import situated_presence
from services.runtime_state import (
    clear_states_for_tests,
    runtime_state_db_path,
)

client = TestClient(app)


def _start_scope(
    *,
    owner_id: str = "owner-presence",
    conversation_id: str = "conversation-presence",
    surface: str = "web",
    request_id: str = "presence-turn-start",
) -> dict[str, str]:
    response = client.post(
        "/v1/runtime/turns/start",
        json={
            "request_id": request_id,
            "owner_id": owner_id,
            "conversation_id": conversation_id,
            "surface": surface,
            "input_message_id": "message-presence",
        },
    )
    assert response.status_code == 200
    body = response.json()
    return {
        "request_id": "presence-evaluate",
        "owner_id": owner_id,
        "conversation_id": conversation_id,
        "surface": surface,
        "runtime_session_id": body["runtime_session"]["runtime_session_id"],
        "runtime_turn_id": body["runtime_turn"]["runtime_turn_id"],
    }


def _governance(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "interaction_kind": "joke_or_playful",
        "tension_level": "low",
        "commentary_allowed": True,
        "humor_allowed": True,
        "action_allowed": False,
        "requires_confirmation": False,
        "privacy_sensitivity_hint": "normal",
        "response_posture": "playful",
        "confidence": 0.9,
    }
    value.update(overrides)
    return value


def _restraint(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "restraint_policy": "answer_normally",
        "proactive_output_suppressed": False,
        "personalization_suppressed": False,
        "brevity_preferred": False,
        "clarification_preferred": False,
        "confidence": 0.9,
    }
    value.update(overrides)
    return value


def _request(
    scope: dict[str, str] | None = None,
    *,
    visibility: str = "private",
    constraint: str = "normal",
    governance: dict[str, object] | None = None,
    restraint: dict[str, object] | None = None,
    **overrides: object,
) -> dict[str, object]:
    value: dict[str, object] = {
        **(scope or _start_scope()),
        "surface_context": {
            "visibility": visibility,
            "constraint": constraint,
        },
        "interaction_governance": governance or _governance(),
        "restraint": restraint or _restraint(),
    }
    value.update(overrides)
    return value


def _evaluate(payload: dict[str, object]):
    return client.post("/v1/runtime/situated-presence/evaluate", json=payload)


def _table_rows(
    db_path: Path,
) -> tuple[tuple[tuple[object, ...], ...], ...]:
    with sqlite3.connect(db_path) as conn:
        return tuple(
            tuple(tuple(row) for row in conn.execute(query).fetchall())
            for query in (
                "SELECT * FROM conversation_runtime_sessions ORDER BY id;",
                "SELECT * FROM conversation_runtime_threads ORDER BY id;",
                "SELECT * FROM conversation_runtime_turns ORDER BY id;",
                "SELECT * FROM conversation_runtime_events ORDER BY id;",
            )
        )


def _policy_rows(
    db_path: Path,
) -> tuple[tuple[tuple[object, ...], ...], ...]:
    return _table_rows(db_path)[:3]


def _presence_events(runtime_session_id: str) -> list[dict[str, object]]:
    response = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert response.status_code == 200
    return [
        event
        for event in response.json()["events"]
        if event["event_type"] == "situated_presence_evaluated"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda body: body.update(extra="forbidden"),
        lambda body: body["surface_context"].update(extra="forbidden"),
        lambda body: body["interaction_governance"].update(extra="forbidden"),
        lambda body: body["restraint"].update(extra="forbidden"),
        lambda body: body.update(request_id=""),
        lambda body: body.update(owner_id="o" * 121),
        lambda body: body.update(conversation_id=" "),
        lambda body: body.update(surface="s" * 65),
        lambda body: body.update(runtime_session_id=""),
        lambda body: body.update(runtime_turn_id="t" * 121),
        lambda body: body["surface_context"].update(visibility="restricted"),
        lambda body: body["surface_context"].update(constraint="tight"),
        lambda body: body["interaction_governance"].update(commentary_allowed=1),
        lambda body: body["restraint"].update(proactive_output_suppressed=0),
        lambda body: body["interaction_governance"].update(confidence="0.9"),
        lambda body: body["interaction_governance"].update(confidence=1),
        lambda body: body["restraint"].update(confidence=1.1),
    ],
)
def test_strict_request_rejects_invalid_or_extra_fields(mutation):
    payload = _request()
    mutation(payload)

    response = _evaluate(payload)

    assert response.status_code == 422


def test_result_and_response_models_reject_extras_and_incoherent_gates():
    valid_result = {
        "commentary_allowed": True,
        "humor_allowed": True,
        "emotional_attunement_allowed": "none",
        "challenge_allowed": "low",
        "silence_preferred": False,
        "surface_allows_commentary": True,
        "response_posture": "playful",
        "action_implication_allowed": False,
        "reason_summary": ["light_commentary_allowed"],
        "policy_version": "situated-presence.v1",
    }
    valid_response = {
        "schema_version": "situated-presence.v1",
        "request_id": "presence-response",
        "owner_id": "owner",
        "conversation_id": "conversation",
        "surface": "web",
        "runtime_session_id": "session",
        "runtime_turn_id": "turn",
        "result": valid_result,
    }

    with pytest.raises(ValidationError):
        SituatedPresenceResult.model_validate({**valid_result, "extra": True})
    with pytest.raises(ValidationError):
        SituatedPresenceEvaluateResponse.model_validate(
            {**valid_response, "extra": True}
        )
    with pytest.raises(ValidationError):
        SituatedPresenceResult.model_validate(
            {**valid_result, "action_implication_allowed": True}
        )
    with pytest.raises(ValidationError):
        SituatedPresenceResult.model_validate(
            {**valid_result, "action_implication_allowed": 0}
        )
    with pytest.raises(ValidationError):
        SituatedPresenceResult.model_validate(
            {**valid_result, "commentary_allowed": False}
        )
    with pytest.raises(ValidationError):
        SituatedPresenceResult.model_validate(
            {
                **valid_result,
                "humor_allowed": False,
                "silence_preferred": True,
            }
        )
    with pytest.raises(ValidationError):
        SituatedPresenceResult.model_validate(
            {
                **valid_result,
                "reason_summary": [
                    "surface_public",
                    "tense_context",
                ],
            }
        )
    with pytest.raises(ValidationError):
        SituatedPresenceResult.model_validate(
            {
                **valid_result,
                "reason_summary": [
                    "light_commentary_allowed",
                    "light_commentary_allowed",
                ],
            }
        )


def test_valid_admitted_scope_returns_strict_bounded_response():
    scope = _start_scope()

    response = _evaluate(_request(scope))

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "schema_version",
        "request_id",
        "owner_id",
        "conversation_id",
        "surface",
        "runtime_session_id",
        "runtime_turn_id",
        "result",
    }
    assert body["schema_version"] == "situated-presence.v1"
    assert body["runtime_session_id"] == scope["runtime_session_id"]
    assert body["runtime_turn_id"] == scope["runtime_turn_id"]
    assert set(body["result"]) == {
        "commentary_allowed",
        "humor_allowed",
        "emotional_attunement_allowed",
        "challenge_allowed",
        "silence_preferred",
        "surface_allows_commentary",
        "response_posture",
        "action_implication_allowed",
        "reason_summary",
        "policy_version",
    }
    assert not {"text", "content", "prompt", "message"}.intersection(body["result"])


def test_missing_session_fails_without_creating_state_or_event():
    db_path = runtime_state_db_path()
    payload = _request()
    before = _table_rows(db_path)
    payload["runtime_session_id"] = "missing-session"

    response = _evaluate(payload)

    assert response.status_code == 404
    assert response.json() == {"detail": "runtime_session_not_found"}
    assert _table_rows(db_path) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("owner_id", "different-owner"),
        ("conversation_id", "different-conversation"),
        ("surface", "different-surface"),
    ],
)
def test_session_scope_mismatch_fails_without_event(field: str, value: str):
    scope = _start_scope()
    payload = _request(scope)
    payload[field] = value

    response = _evaluate(payload)

    assert response.status_code == 400
    assert response.json() == {"detail": "runtime_session_mismatch"}
    assert _presence_events(scope["runtime_session_id"]) == []


def test_missing_and_cross_session_turn_fail_without_event():
    first = _start_scope(request_id="first-start")
    missing = _request(first, runtime_turn_id="missing-turn")

    missing_response = _evaluate(missing)

    assert missing_response.status_code == 404
    assert missing_response.json() == {"detail": "runtime_turn_not_found"}
    assert _presence_events(first["runtime_session_id"]) == []

    second = _start_scope(
        owner_id="owner-second",
        conversation_id="conversation-second",
        surface="mobile",
        request_id="second-start",
    )
    cross = _request(second, runtime_turn_id=first["runtime_turn_id"])

    cross_response = _evaluate(cross)

    assert cross_response.status_code == 400
    assert cross_response.json() == {"detail": "runtime_turn_session_mismatch"}
    assert _presence_events(second["runtime_session_id"]) == []


def test_terminal_turn_is_not_accepted_as_current_admitted_scope():
    scope = _start_scope()
    completed = client.post(
        "/v1/runtime/turns/complete",
        json={
            "request_id": "presence-turn-complete",
            "runtime_session_id": scope["runtime_session_id"],
            "runtime_turn_id": scope["runtime_turn_id"],
            "turn_status": "completed",
        },
    )
    assert completed.status_code == 200

    response = _evaluate(_request(scope))

    assert response.status_code == 409
    assert response.json() == {"detail": "runtime_turn_not_current"}
    assert _presence_events(scope["runtime_session_id"]) == []


def test_evaluation_only_adds_one_summarized_event():
    scope = _start_scope()
    db_path = runtime_state_db_path()
    before = _policy_rows(db_path)

    response = _evaluate(_request(scope))

    assert response.status_code == 200
    assert _policy_rows(db_path) == before
    events = _presence_events(scope["runtime_session_id"])
    assert len(events) == 1
    assert events[0]["runtime_turn_id"] == scope["runtime_turn_id"]


def test_private_playful_low_risk_allows_light_commentary_and_humor():
    response = _evaluate(_request())

    assert response.status_code == 200
    result = response.json()["result"]
    assert result == {
        "commentary_allowed": True,
        "humor_allowed": True,
        "emotional_attunement_allowed": "none",
        "challenge_allowed": "low",
        "silence_preferred": False,
        "surface_allows_commentary": True,
        "response_posture": "playful",
        "action_implication_allowed": False,
        "reason_summary": ["light_commentary_allowed"],
        "policy_version": "situated-presence.v1",
    }


def test_brevity_keeps_allowed_playfulness_bounded_to_brief_posture():
    response = _evaluate(
        _request(restraint=_restraint(brevity_preferred=True))
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["commentary_allowed"] is True
    assert result["humor_allowed"] is True
    assert result["response_posture"] == "brief"
    assert "brevity_preferred" in result["reason_summary"]


@pytest.mark.parametrize(
    "governance",
    [
        _governance(
            interaction_kind="tense_debugging",
            tension_level="high",
            response_posture="playful",
        ),
        _governance(
            interaction_kind="joke_or_playful",
            tension_level="high",
            response_posture="playful",
        ),
    ],
)
def test_tense_context_suppresses_humor_and_requires_tactical_response(governance):
    response = _evaluate(_request(governance=governance))

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["commentary_allowed"] is False
    assert result["humor_allowed"] is False
    assert result["challenge_allowed"] == "medium"
    assert result["silence_preferred"] is False
    assert result["response_posture"] == "tactical"
    assert {"tense_context", "tactical_response_required"}.issubset(
        result["reason_summary"]
    )


def test_high_impact_clamps_upstream_commentary_humor_and_action():
    governance = _governance(
        interaction_kind="high_impact_decision",
        tension_level="medium",
        action_allowed=True,
        requires_confirmation=True,
        response_posture="playful",
    )

    response = _evaluate(_request(governance=governance))

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["commentary_allowed"] is False
    assert result["humor_allowed"] is False
    assert result["action_implication_allowed"] is False
    assert result["response_posture"] in {"direct", "brief"}
    assert result["silence_preferred"] is False
    assert "high_impact_context" in result["reason_summary"]
    assert "confirmation_required" in result["reason_summary"]


@pytest.mark.parametrize(
    ("kind", "challenge"),
    [
        ("vent_or_expression", "none"),
        ("mistake_or_failure_report", "low"),
    ],
)
def test_private_emotional_or_mistake_context_allows_only_brief_attunement(
    kind: str,
    challenge: str,
):
    governance = _governance(
        interaction_kind=kind,
        tension_level="medium",
        humor_allowed=True,
        response_posture="supportive",
    )

    response = _evaluate(_request(governance=governance))

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["commentary_allowed"] is False
    assert result["humor_allowed"] is False
    assert result["emotional_attunement_allowed"] == "brief"
    assert result["challenge_allowed"] == challenge
    assert result["response_posture"] == "supportive"
    assert "brief_steadying_allowed" in result["reason_summary"]


@pytest.mark.parametrize(
    ("governance_overrides", "restraint_overrides", "expected_reason"),
    [
        (
            {"privacy_sensitivity_hint": "sensitive"},
            {},
            "privacy_sensitive",
        ),
        (
            {},
            {"personalization_suppressed": True},
            "personalization_suppressed",
        ),
    ],
)
def test_privacy_or_personalization_clamp_reduces_emotional_attunement(
    governance_overrides,
    restraint_overrides,
    expected_reason,
):
    governance = _governance(
        interaction_kind="mistake_or_failure_report",
        tension_level="medium",
        humor_allowed=False,
        response_posture="supportive",
        **governance_overrides,
    )

    response = _evaluate(
        _request(
            governance=governance,
            restraint=_restraint(**restraint_overrides),
        )
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["commentary_allowed"] is False
    assert result["humor_allowed"] is False
    assert result["emotional_attunement_allowed"] != "brief"
    assert expected_reason in result["reason_summary"]


@pytest.mark.parametrize(
    ("visibility", "constraint", "reason"),
    [
        ("shared", "normal", "surface_shared"),
        ("public", "normal", "surface_public"),
        ("unknown", "normal", "surface_visibility_unknown"),
        ("private", "constrained", "surface_constrained"),
        ("private", "unknown", "surface_constraint_unknown"),
    ],
)
def test_surface_clamps_commentary_without_displacing_direct_help(
    visibility: str,
    constraint: str,
    reason: str,
):
    governance = _governance(
        interaction_kind="question",
        humor_allowed=False,
        response_posture="direct",
    )

    response = _evaluate(
        _request(
            visibility=visibility,
            constraint=constraint,
            governance=governance,
        )
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["surface_allows_commentary"] is False
    assert result["commentary_allowed"] is False
    assert result["humor_allowed"] is False
    assert result["response_posture"] == "direct"
    assert result["silence_preferred"] is False
    assert reason in result["reason_summary"]


def test_public_tense_context_keeps_tactical_help_while_suppressing_commentary():
    governance = _governance(
        interaction_kind="tense_debugging",
        tension_level="high",
        response_posture="playful",
    )

    response = _evaluate(
        _request(visibility="public", governance=governance)
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["surface_allows_commentary"] is False
    assert result["commentary_allowed"] is False
    assert result["humor_allowed"] is False
    assert result["silence_preferred"] is False
    assert result["response_posture"] == "tactical"


@pytest.mark.parametrize(
    ("governance", "restraint", "reason", "commentary_allowed"),
    [
        (
            _governance(commentary_allowed=False),
            _restraint(),
            "upstream_commentary_suppressed",
            False,
        ),
        (
            _governance(humor_allowed=False),
            _restraint(),
            "upstream_humor_suppressed",
            True,
        ),
        (
            _governance(),
            _restraint(proactive_output_suppressed=True),
            "proactive_output_suppressed",
            False,
        ),
        (
            _governance(),
            _restraint(personalization_suppressed=True),
            "personalization_suppressed",
            False,
        ),
        (
            _governance(commentary_allowed=False),
            _restraint(brevity_preferred=True),
            "brevity_preferred",
            False,
        ),
        (
            _governance(),
            _restraint(clarification_preferred=True),
            "clarification_preferred",
            False,
        ),
        (
            _governance(requires_confirmation=True),
            _restraint(),
            "confirmation_required",
            False,
        ),
    ],
)
def test_upstream_governance_and_restraint_can_only_tighten(
    governance,
    restraint,
    reason,
    commentary_allowed,
):
    response = _evaluate(
        _request(governance=governance, restraint=restraint)
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["commentary_allowed"] is commentary_allowed
    assert result["humor_allowed"] is False
    assert result["action_implication_allowed"] is False
    assert reason in result["reason_summary"]


def test_ambiguous_and_low_confidence_inputs_prefer_silence():
    ambiguous = _evaluate(
        _request(
            governance=_governance(
                interaction_kind="ambiguous",
                commentary_allowed=True,
                humor_allowed=True,
                response_posture="playful",
            )
        )
    )
    low_confidence = _evaluate(
        _request(
            _start_scope(
                owner_id="owner-low-confidence",
                conversation_id="conversation-low-confidence",
                request_id="low-confidence-start",
            ),
            governance=_governance(confidence=0.59),
        )
    )

    for response, reason in (
        (ambiguous, "ambiguous_context"),
        (low_confidence, "upstream_confidence_insufficient"),
    ):
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["commentary_allowed"] is False
        assert result["humor_allowed"] is False
        assert result["emotional_attunement_allowed"] == "none"
        assert result["challenge_allowed"] == "none"
        assert result["silence_preferred"] is True
        assert result["response_posture"] == "silent_or_minimal"
        assert reason in result["reason_summary"]


def test_success_event_contains_only_bounded_gates_and_no_inferred_feeling():
    scope = _start_scope()
    response = _evaluate(
        _request(
            scope,
            governance=_governance(
                interaction_kind="mistake_or_failure_report",
                tension_level="medium",
                humor_allowed=False,
                response_posture="supportive",
            ),
        )
    )

    assert response.status_code == 200
    event = _presence_events(scope["runtime_session_id"])[0]
    payload = event["event_payload_json"]
    assert set(payload) == {
        "commentary_allowed",
        "humor_allowed",
        "emotional_attunement_allowed",
        "challenge_allowed",
        "silence_preferred",
        "surface_allows_commentary",
        "response_posture",
        "action_implication_allowed",
        "reason_summary",
        "policy_version",
    }
    serialized = str(payload).lower()
    for forbidden in (
        "request_id",
        "current_user_text",
        "recent_messages",
        "prompt_overlay",
        "content",
        "title",
        "feeling",
        "emotion_label",
        "surface_context",
    ):
        assert forbidden not in serialized


def test_event_survives_repository_reopen_without_policy_state_mutation():
    scope = _start_scope()
    db_path = runtime_state_db_path()
    before = _policy_rows(db_path)
    response = _evaluate(_request(scope))
    assert response.status_code == 200

    clear_states_for_tests(db_path=db_path)

    assert _policy_rows(db_path) == before
    events = _presence_events(scope["runtime_session_id"])
    assert len(events) == 1
    assert events[0]["event_type"] == "situated_presence_evaluated"


def test_service_has_no_raw_text_classifier_or_external_dependency():
    request_fields = set(_request().keys())
    source = inspect.getsource(situated_presence)

    assert not {"current_user_text", "recent_messages", "content", "summary"}.intersection(
        request_fields
    )
    for forbidden in (
        "import re",
        "httpx",
        "provider",
        "semantic",
        "retrieval",
        "basic_memory",
        "data_source",
        "adapter",
    ):
        assert forbidden not in source.lower()
