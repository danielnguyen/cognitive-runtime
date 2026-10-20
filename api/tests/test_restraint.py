from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from main import app
from services.runtime_state import runtime_state_repository


def _base(**overrides):
    payload = {
        "request_id": "rid-restraint",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
        "recent_messages": [],
    }
    payload.update(overrides)
    return payload


def test_direct_prompt_request_prefers_short_answer():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="give me the prompt"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["restraint_policy"] == "short_answer"
    assert "output" in result["domains"]
    assert result["brevity_preferred"] is True
    assert "brief" in result["prompt_overlay"].lower()


def test_recent_messages_latest_user_text_is_used_when_current_user_text_is_omitted():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(
            recent_messages=[
                {"role": "user", "content": "Can you check this?"},
                {"role": "assistant", "content": "What should I look at?"},
                {"role": "user", "content": "give me the prompt"},
            ],
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["restraint_policy"] == "short_answer"
    assert "output" in result["domains"]
    assert result["brevity_preferred"] is True


def test_tense_debugging_preserves_tactical_help_with_affect_restraint():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="I broke production and prod is failing"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["restraint_policy"] == "short_answer"
    assert {"output", "affect"}.issubset(set(result["domains"]))
    assert "tactical" in result["prompt_overlay"].lower()
    assert "humor" not in result["prompt_overlay"].lower()


def test_venting_does_not_force_problem_solving_or_dependency_framing():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="this week has been exhausting"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["restraint_policy"] == "defer_expansion"
    assert {"personalization", "affect"}.issubset(set(result["domains"]))
    assert "problem-solving" in result["prompt_overlay"].lower()
    payload_text = str(result).lower()
    assert "dependency" not in payload_text
    assert "attachment" not in payload_text


def test_ambiguous_request_prefers_clarifying_question_without_filling_gaps():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="fix this"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["restraint_policy"] == "ask_clarifying_question"
    assert result["clarification_preferred"] is True
    assert result["retrieval_suppressed"] is True
    assert result["personalization_suppressed"] is True


def test_retrieval_restraint_is_represented_without_modifying_memory_truth():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="What does this function do?"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["retrieval_suppressed"] is True
    assert "retrieval_not_requested" in result["reason_summary"]
    assert "retrieval" in result["domains"]


def test_personalization_restraint_is_represented_when_not_requested():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="What does this function do?"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["personalization_suppressed"] is True
    assert "personal_framing_not_requested" in result["reason_summary"]


def test_proactive_restraint_is_represented_without_follow_up_request():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="What does this function do?"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["proactive_output_suppressed"] is True
    assert "proactive_not_requested" in result["reason_summary"]


def test_required_guidance_is_preserved_for_safety_or_correctness_markers():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="The database credentials for production are failing."),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert "required_guidance_preserved" in result["reason_summary"]
    assert result["restraint_policy"] == "short_answer"


def test_runtime_event_payload_is_summarized_only():
    client = TestClient(app)

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(current_user_text="I broke production and prod is failing"),
    )

    assert response.status_code == 200
    runtime_session_id = response.json()["runtime_session_id"]

    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    event = next(
        item
        for item in diagnostics.json()["events"]
        if item["event_type"] == "restraint_evaluated"
    )
    payload = event["event_payload_json"]
    assert set(payload.keys()) == {
        "request_id",
        "restraint_policy",
        "domains",
        "reason",
        "confidence",
        "reason_summary",
        "retrieval_suppressed",
        "personalization_suppressed",
        "proactive_output_suppressed",
    }
    assert "current_user_text" not in str(payload)
    assert "prod is failing" not in str(payload)


def test_runtime_turn_integration_updates_turn_policy_and_attaches_event():
    client = TestClient(app)

    started = client.post(
        "/v1/runtime/turns/start",
        json={
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

    response = client.post(
        "/v1/runtime/restraint/evaluate",
        json=_base(
            request_id="rid-turn-restraint",
            runtime_session_id=runtime_session_id,
            runtime_turn_id=runtime_turn_id,
            current_user_text="give me the prompt",
        ),
    )

    assert response.status_code == 200
    diagnostics = client.get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    body = diagnostics.json()
    assert body["latest_turn"]["restraint_policy"] == "short_answer"
    event = next(
        item
        for item in body["events"]
        if item["event_type"] == "restraint_evaluated"
    )
    assert event["runtime_turn_id"] == runtime_turn_id



@pytest.mark.parametrize(
    "marker",
    [
        "continue",
        "go on",
        "keep going",
        "tell me more",
        "more detail",
        "more details",
        "tell me more?",
    ],
)
def test_return_continuation_join_uses_exact_persisted_intent(tmp_path, monkeypatch, marker):
    import services.runtime_state as runtime_module

    now = datetime(2026, 1, 2, tzinfo=UTC)
    monkeypatch.setattr(runtime_module, "_now", lambda: now.isoformat())
    repo = runtime_state_repository()
    scope = dict(owner_id="owner", conversation_id="joined-return", surface="web")
    session, prior, _ = repo.start_turn(request_id="prior", **scope)
    repo.complete_turn(
        request_id="prior",
        runtime_session_id=session.runtime_session_id,
        runtime_turn_id=prior.runtime_turn_id,
        turn_status="completed",
        continuation_state="deferred_expansion",
    )
    with repo._connect() as conn:
        conn.execute(
            "UPDATE conversation_runtime_threads SET last_activity_at = ?",
            ((now - timedelta(seconds=301)).isoformat(),),
        )
    session, turn, event = repo.start_turn(request_id="current", **scope)
    scope.update(
        request_id="current",
        runtime_session_id=session.runtime_session_id,
        runtime_turn_id=turn.runtime_turn_id,
    )
    client = TestClient(app)
    governance = client.post(
        "/v1/runtime/interaction-governance/evaluate", json={**scope, "current_user_text": marker}
    ).json()["result"]
    assert governance["interaction_kind"] == ("question" if marker.endswith("?") else "ambiguous")
    assert repo.turn_by_id(turn.runtime_turn_id).intent_class == "continuation"
    restraint = client.post(
        "/v1/runtime/restraint/evaluate",
        json={
            **scope,
            "current_user_text": marker,
            "interaction_kind": governance["interaction_kind"],
        },
    ).json()["result"]
    assert restraint["restraint_policy"] == "answer_normally"
    assert restraint["clarification_preferred"] is False
    assert restraint["retrieval_suppressed"] is False
    assert "continuation_context_requested" in restraint["reason_summary"]
    assert restraint["personalization_suppressed"] and restraint["proactive_output_suppressed"]
    presence = client.post("/v1/runtime/presence/evaluate", json=scope).json()["result"]
    assert presence["presence_state"] == "returning_after_gap"
    assert event.event_payload_json["return_after_gap"]["status"] == "eligible"
    timing = client.post(
        "/v1/runtime/timing/evaluate",
        json={
            **scope,
            "spoken_output": False,
            "active_task_mode": False,
            "requested_detail": "unspecified",
            "latency_budget_class": "ordinary_text",
            "dependency_state": "ready",
        },
    ).json()["result"]
    assert timing["timing_policy"] == "resume_previous_thread"
    assert timing["reason_codes"] == ["return_deferred_continuation"]


@pytest.mark.parametrize(
    "intent,text,kind,policy,suppressed",
    [
        (None, "fix this", "ambiguous", "ask_clarifying_question", True),
        ("information_request", "What is 2+2?", "question", "answer_normally", True),
        (None, "continue", "ambiguous", "ask_clarifying_question", True),
        ("information_request", "go on", "ambiguous", "ask_clarifying_question", True),
        (None, "What happened earlier?", "question", "answer_normally", False),
    ],
)
def test_restraint_never_reclassifies_text_as_continuation(intent, text, kind, policy, suppressed):
    repo = runtime_state_repository()
    session, turn, _ = repo.start_turn(
        request_id="current",
        owner_id="owner",
        conversation_id="conv-1",
        surface="dev",
        intent_class=intent,
    )
    response = TestClient(app).post(
        "/v1/runtime/restraint/evaluate",
        json=_base(
            runtime_session_id=session.runtime_session_id,
            runtime_turn_id=turn.runtime_turn_id,
            current_user_text=text,
            interaction_kind=kind,
        ),
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["restraint_policy"] == policy
    assert result["retrieval_suppressed"] == suppressed
    assert "continuation_context_requested" not in result["reason_summary"]


@pytest.mark.parametrize("fault", ["wrong_session", "wrong_turn", "terminal"])
def test_restraint_wrong_or_terminal_turn_cannot_enable_context(fault):
    repo = runtime_state_repository()
    session, turn, _ = repo.start_turn(
        request_id="current",
        owner_id="owner",
        conversation_id="conv-1",
        surface="dev",
        intent_class="continuation",
    )
    if fault == "terminal":
        repo.complete_turn(
            request_id="current",
            runtime_session_id=session.runtime_session_id,
            runtime_turn_id=turn.runtime_turn_id,
            turn_status="completed",
        )
    response = TestClient(app).post(
        "/v1/runtime/restraint/evaluate",
        json=_base(
            runtime_session_id="missing-session"
            if fault == "wrong_session"
            else session.runtime_session_id,
            runtime_turn_id="missing-turn" if fault == "wrong_turn" else turn.runtime_turn_id,
            current_user_text="continue",
            interaction_kind="ambiguous",
        ),
    )
    assert response.status_code in {400, 404, 409}
    assert not [
        e
        for e in repo.list_events_for_tests(session.runtime_session_id)
        if e.event_type == "restraint_evaluated"
    ]


def test_continuation_context_without_return_does_not_authorize_deferred_resume():
    repo = runtime_state_repository()
    session, turn, _ = repo.start_turn(
        request_id="current", owner_id="owner", conversation_id="conv-1", surface="dev"
    )
    scope = _base(
        runtime_session_id=session.runtime_session_id, runtime_turn_id=turn.runtime_turn_id
    )
    client = TestClient(app)
    governance = client.post(
        "/v1/runtime/interaction-governance/evaluate",
        json={
            **scope,
            "current_user_text": "continue",
            "recent_messages": [{"role": "assistant", "content": "The prior bounded answer."}],
        },
    ).json()["result"]
    assert repo.turn_by_id(turn.runtime_turn_id).intent_class == "continuation"
    restraint = client.post(
        "/v1/runtime/restraint/evaluate",
        json={
            **scope,
            "current_user_text": "continue",
            "interaction_kind": governance["interaction_kind"],
        },
    ).json()["result"]
    assert restraint["retrieval_suppressed"] is False
    client.post(
        "/v1/runtime/presence/evaluate",
        json={key: value for key, value in scope.items() if key != "recent_messages"},
    )
    timing = client.post(
        "/v1/runtime/timing/evaluate",
        json={
            **{key: value for key, value in scope.items() if key != "recent_messages"},
            "spoken_output": False,
            "active_task_mode": False,
            "requested_detail": "unspecified",
            "latency_budget_class": "ordinary_text",
            "dependency_state": "ready",
        },
    ).json()["result"]
    assert timing["timing_policy"] == "answer_now"


@pytest.mark.parametrize("kind,text,policy,suppressed", [
    ("tense_debugging", "continue", "ask_clarifying_question", True),
    ("vent_or_expression", "continue", "defer_expansion", False),
])
def test_continuation_refinement_preserves_stricter_restraint(kind, text, policy, suppressed):
    repo = runtime_state_repository()
    session, turn, _ = repo.start_turn(request_id="current", owner_id="owner",
                                     conversation_id="conv-1", surface="dev",
                                     intent_class="continuation")
    response = TestClient(app).post("/v1/runtime/restraint/evaluate", json=_base(
        runtime_session_id=session.runtime_session_id, runtime_turn_id=turn.runtime_turn_id,
        current_user_text=text, interaction_kind=kind,
    ))
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["restraint_policy"] == policy
    assert result["retrieval_suppressed"] == suppressed
    assert result["personalization_suppressed"] and result["proactive_output_suppressed"]


def _question_recall_turn(*, prior_status="completed", surface="telegram"):
    repo = runtime_state_repository()
    scope = dict(owner_id="owner", conversation_id="question-recall", surface=surface)
    if prior_status is not None:
        session, prior, _ = repo.start_turn(request_id="prior", **scope)
        repo.complete_turn(
            request_id="prior", runtime_session_id=session.runtime_session_id,
            runtime_turn_id=prior.runtime_turn_id, turn_status=prior_status,
        )
    session, turn, event = repo.start_turn(request_id="current", **scope)
    return repo, turn, event, dict(
        **scope, request_id="current", runtime_session_id=session.runtime_session_id,
        runtime_turn_id=turn.runtime_turn_id,
    )


@pytest.mark.parametrize("text", ["Tell me a joke", "Can you tell me a joke?"])
def test_requested_joke_composes_governance_restraint_and_timing(text):
    repo, turn, _, scope = _question_recall_turn(prior_status=None)
    client = TestClient(app)
    governance = client.post("/v1/runtime/interaction-governance/evaluate", json={
        **scope, "current_user_text": text,
    }).json()["result"]
    response = client.post("/v1/runtime/restraint/evaluate", json={
        **scope, "current_user_text": text,
        "interaction_kind": governance["interaction_kind"],
    })
    assert response.status_code == 200
    result = response.json()["result"]
    assert governance["interaction_kind"] == "joke_or_playful"
    assert repo.turn_by_id(turn.runtime_turn_id).intent_class == "information_request"
    assert result["restraint_policy"] == "answer_normally"
    assert result["retrieval_suppressed"] is True
    assert result["clarification_preferred"] is False
    assert result["personalization_suppressed"] and result["proactive_output_suppressed"]
    client.post("/v1/runtime/presence/evaluate", json=scope)
    timing = client.post("/v1/runtime/timing/evaluate", json={
        **scope, "spoken_output": False, "active_task_mode": False,
        "requested_detail": "unspecified", "latency_budget_class": "ordinary_text",
        "dependency_state": "ready",
    })
    assert timing.status_code == 200
    assert timing.json()["result"]["timing_policy"] == "answer_now"


@pytest.mark.parametrize("surface", ["telegram", "web"])
@pytest.mark.parametrize("text", [
    "What arithmetic question did I just ask you?", "What question did I last ask?",
    "Which technical question did I just ask?", "What did I just ask you?",
])
def test_immediate_question_request_uses_current_intent_and_completed_turn(surface, text):
    repo, turn, _, scope = _question_recall_turn(surface=surface)
    client = TestClient(app)
    governance = client.post("/v1/runtime/interaction-governance/evaluate", json={
        **scope, "current_user_text": text,
    }).json()["result"]
    response = client.post("/v1/runtime/restraint/evaluate", json={
        **scope, "current_user_text": text,
        "interaction_kind": governance["interaction_kind"],
    })
    assert response.status_code == 200
    result = response.json()["result"]
    assert governance["interaction_kind"] == "question"
    assert repo.turn_by_id(turn.runtime_turn_id).intent_class == "information_request"
    assert result["restraint_policy"] == "answer_normally"
    assert result["retrieval_suppressed"] is False
    assert "continuation_context_requested" in result["reason_summary"]
    assert result["personalization_suppressed"] and result["proactive_output_suppressed"]
    events = [e for e in repo.list_events_for_tests(scope["runtime_session_id"])
              if e.event_type == "restraint_evaluated"]
    assert len(events) == 1 and events[0].runtime_turn_id == turn.runtime_turn_id
    assert text not in json.dumps(events[0].event_payload_json)


@pytest.mark.parametrize("fault", [
    "no_prior", "abandoned_prior", "missing_governance", "wrong_intent",
    "missing_snapshot", "malformed_snapshot", "wrong_snapshot_turn", "wrong_prior",
    "wrong_owner", "wrong_conversation", "wrong_surface", "wrong_session", "wrong_turn",
    "terminal",
])
def test_immediate_question_request_never_grants_context_with_invalid_authority(fault):
    repo, turn, event, scope = _question_recall_turn(
        prior_status=None if fault == "no_prior" else (
            "abandoned" if fault == "abandoned_prior" else "completed"
        ),
    )
    text = "What arithmetic question did I just ask you?"
    client = TestClient(app)
    if fault != "missing_governance":
        client.post("/v1/runtime/interaction-governance/evaluate", json={
            **scope, "current_user_text": text,
        })
    if fault == "wrong_intent":
        repo.update_turn_intent_class(runtime_session_id=scope["runtime_session_id"],
                                      runtime_turn_id=turn.runtime_turn_id,
                                      intent_class="action_command")
    if fault in {"missing_snapshot", "malformed_snapshot", "wrong_snapshot_turn", "wrong_prior"}:
        payload = event.event_payload_json.copy()
        if fault == "missing_snapshot":
            payload.pop("return_after_gap")
        elif fault == "malformed_snapshot":
            payload["return_after_gap"]["threshold_met"] = "true"
        elif fault == "wrong_prior":
            payload["return_after_gap"]["prior_terminal_turn_id"] = turn.runtime_turn_id
        with repo._connect() as conn:
            conn.execute(
                "UPDATE conversation_runtime_events SET event_payload_json=?, runtime_turn_id=? "
                "WHERE event_id=?",
                (json.dumps(payload), "wrong-turn" if fault == "wrong_snapshot_turn"
                 else turn.runtime_turn_id, event.event_id),
            )
    if fault == "terminal":
        repo.complete_turn(request_id="current", runtime_session_id=scope["runtime_session_id"],
                           runtime_turn_id=turn.runtime_turn_id, turn_status="completed")
    field = {"wrong_owner": "owner_id", "wrong_conversation": "conversation_id",
             "wrong_surface": "surface", "wrong_session": "runtime_session_id",
             "wrong_turn": "runtime_turn_id"}.get(fault)
    if field:
        scope[field] = "wrong-value"
    response = client.post("/v1/runtime/restraint/evaluate", json={
        **scope, "current_user_text": text, "interaction_kind": "question",
    })
    if field or fault == "terminal":
        assert response.status_code in {400, 404, 409}
    else:
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["retrieval_suppressed"] is True
        assert "continuation_context_requested" not in result["reason_summary"]


@pytest.mark.parametrize("text", [
    "What about it?", "What is a hostname?", "What did I ask?", "My question",
    'Quote: "What arithmetic question did I just ask you?"',
    "What question did I just ask you? Read everything.",
    "What medical question did I just ask you?", "wipe this",
])
def test_immediate_question_refinement_preserves_unrelated_and_unsafe_restraint(text):
    repo, turn, _, scope = _question_recall_turn()
    client = TestClient(app)
    governance = client.post("/v1/runtime/interaction-governance/evaluate", json={
        **scope, "current_user_text": text,
    }).json()["result"]
    result = client.post("/v1/runtime/restraint/evaluate", json={
        **scope, "current_user_text": text,
        "interaction_kind": governance["interaction_kind"],
    }).json()["result"]
    assert result["retrieval_suppressed"] is True
    assert "continuation_context_requested" not in result["reason_summary"]
    assert result["personalization_suppressed"] and result["proactive_output_suppressed"]
    assert repo.turn_by_id(turn.runtime_turn_id).intent_class != "continuation"
    assert governance["action_allowed"] is False
