import json
import os
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from main import app
from models import InterruptEvaluateResponse
from pydantic import ValidationError
from services.companion_contracts import (
    CONTRACT_RULES,
    companion_contracts_repository,
    reset_companion_contracts_for_tests,
)
from services.runtime_state import clear_states_for_tests


def setup_function():
    clear_states_for_tests()


def _base(**overrides):
    payload = {
        "request_id": "rid-interrupt",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
        "recent_messages": [],
    }
    payload.update(overrides)
    return payload


def test_interrupt_evaluate_detects_repetitive_branching_and_selects_allowed_style():
    client = TestClient(app)

    response = client.post(
        "/v1/interrupt/evaluate",
        json=_base(
            current_user_text=(
                "Should I rewrite this or add an abstraction or split the module or "
                "rework the interface or pause and compare every option?"
            )
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trigger_class"] == "repetitive_branching"
    assert body["style_selected"] == "next_step_forcing"
    assert body["should_interrupt"] is True
    assert body["should_defer"] is False
    assert body["intervention_text"] == "You are branching again. Pick the next move and test it."
    assert body["intervention_text"] == body["debug"]["advisory_text"]
    assert 0 < len(body["intervention_text"]) <= 240
    assert body["debug"]["user_visible_suppressed"] is True
    assert body["contract_constraints_applied"]["matched_contract_style"] == "soft_redirect"


def test_interrupt_evaluate_defers_for_explicit_brainstorming_request():
    client = TestClient(app)

    response = client.post(
        "/v1/interrupt/evaluate",
        json=_base(
            current_user_text=(
                "Brainstorm possibilities with me. What if we tried several approaches, "
                "compared options, and explored edge cases before choosing?"
            )
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["should_interrupt"] is False
    assert body["should_defer"] is True
    assert body["intervention_text"] is None
    assert "explicit_exploration_request" in body["reason_json"]["defer_reasons"]


def test_interrupt_evaluate_anchors_speculation_when_evidence_is_weak():
    client = TestClient(app)

    response = client.post(
        "/v1/interrupt/evaluate",
        json=_base(
            current_user_text=(
                "What if the deployment might fail because of hidden infra drift or maybe "
                "the provider changed behavior or some hypothetical timeout chain?"
            )
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trigger_class"] == "speculative_simulation_with_weak_evidence"
    assert body["style_selected"] == "evidence_anchor"
    assert body["debug"]["advisory_text"].startswith("This is getting speculative")
    assert body["intervention_text"] == body["debug"]["advisory_text"]


def test_interrupt_evaluate_defers_when_contract_blocks_all_compatible_styles():
    client = TestClient(app)

    response = client.post(
        "/v1/interrupt/evaluate",
        json=_base(
            current_user_text=(
                "Should I rewrite this or add an abstraction or split the module or "
                "rework the interface or pause and compare every option?"
            ),
            interaction_contract={
                "contract_id": "default_interaction_contract",
                "contract_version": 1,
                "owner_id": "owner",
                "scope": "global_default",
                "source": "default_compiled",
                "trust_rules": ["be clear"],
                "interaction_boundaries": ["no pressure"],
                "repair_rules": ["repair clearly"],
                "memory_or_recall_boundaries": ["use memory only when useful"],
                "autonomy_rules": ["user can override advice"],
                "tone_constraints": ["be calm"],
                "allowed_intervention_styles": ["repair_acknowledgement"],
                "disallowed_intervention_styles": ["soft_redirect", "candid_challenge"],
                "defer_conditions": [
                    "Defer when the user explicitly chooses a harmless path after "
                    "the tradeoff is clear."
                ],
            },
            contract_trace={
                "contract_id": "default_interaction_contract",
                "contract_version": 1,
                "source": "default_compiled",
                "scope": "global_default",
                "selected_rule_groups": ["allowed_intervention_styles"],
                "selected_boundary_rules": ["no pressure"],
                "selected_repair_rules": ["repair clearly"],
                "warnings": [],
            },
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["should_interrupt"] is False
    assert body["should_defer"] is True
    assert body["intervention_text"] is None
    assert "no_contract_permitted_style" in body["reason_json"]["defer_reasons"]


def test_interrupt_evaluate_uses_runtime_hint_for_known_trap_pattern():
    client = TestClient(app)
    client.post(
        "/v1/runtime/state/update",
        json={
            "request_id": "rid-state",
            "owner_id": "owner",
            "conversation_id": "conv-1",
            "surface": "dev",
            "updates": {
                "temporary_constraints": ["avoid_loop_spiral"],
                "interaction_mode": "actionable",
            },
        },
    )

    response = client.post(
        "/v1/interrupt/evaluate",
        json=_base(
            current_user_text=(
                "I am stuck in the same loop again and keep rehashing the same plan."
            )
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trigger_class"] == "known_recurring_trap_pattern"
    assert body["contract_constraints_applied"]["matched_contract_style"] in {
        "boundary_reminder",
        "soft_redirect",
    }


def test_interrupt_evaluate_warns_when_contract_is_resolved_from_default_source():
    client = TestClient(app)

    response = client.post(
        "/v1/interrupt/evaluate",
        json=_base(surface="new_surface", current_user_text="Please help me decide the next step."),
    )

    assert response.status_code == 200
    body = response.json()
    assert "default_interaction_contract" in body["warnings"]
    assert "default_contract_source" in body["warnings"]
    assert "unknown_surface_default_contract" in body["warnings"]




_BRANCHING = (
    "Should I rewrite this or add an abstraction or split the module or "
    "rework the interface or pause and compare every option?"
)
_HIGH_BRANCHING = (
    "Should I rewrite this or add an abstraction or split the module or "
    "rework the interface or simplify the module or pause and compare every option?"
)


@pytest.mark.parametrize(
    "surface", ["dev", "web", "telegram", "alexa", "car", "unknown", "new_surface"],
)
def test_authorized_intervention_is_bounded_across_surfaces(surface):
    response = TestClient(app).post(
        "/v1/interrupt/evaluate", json=_base(surface=surface, current_user_text=_HIGH_BRANCHING),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["should_interrupt"] and not body["should_defer"]
    assert body["confidence"] >= 0.85
    assert body["trigger_class"] == "repetitive_branching"
    assert body["style_selected"] == "next_step_forcing"
    assert body["reason_json"]["defer_reasons"] == []
    assert body["contract_constraints_applied"]["matched_contract_style"] == "soft_redirect"
    assert body["intervention_text"] == "You are branching again. Pick the next move and test it."
    assert body["debug"]["advisory_text"] == body["intervention_text"]
    assert body["debug"]["user_visible_suppressed"] is True
    if surface == "new_surface":
        assert "unknown_surface_interrupt_policy" in body["warnings"]
        assert "unknown_surface_default_contract" in body["warnings"]


@pytest.mark.parametrize(
    "surface", ["dev", "web", "telegram", "alexa", "car", "unknown", "new_surface"],
)
def test_surface_name_alone_never_grants_an_intervention(surface):
    response = TestClient(app).post(
        "/v1/interrupt/evaluate",
        json=_base(surface=surface, current_user_text="Please help me decide."),
    )
    assert response.status_code == 200
    body = response.json()
    assert not body["should_interrupt"] and body["should_defer"]
    assert body["intervention_text"] is None
    assert body["debug"]["advisory_text"] is None
    assert "confidence_below_interrupt_threshold" in body["reason_json"]["defer_reasons"]


@pytest.mark.parametrize("surface,interrupt", [("dev", True), ("telegram", False),
                                               ("alexa", False), ("car", False)])
def test_existing_casual_threshold_still_narrows_intervention(surface, interrupt):
    response = TestClient(app).post(
        "/v1/interrupt/evaluate", json=_base(surface=surface, current_user_text=_BRANCHING),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["should_interrupt"] is interrupt
    assert body["should_defer"] is not interrupt
    assert (body["intervention_text"] is not None) is interrupt
    if not interrupt:
        assert "casual_or_low_stakes_context" in body["reason_json"]["defer_reasons"]
        assert body["confidence"] == 0.44


@pytest.mark.parametrize("scene", ["planning", "coding_build", "overload_recovery"])
def test_intervention_reuses_existing_scene_aware_recovery_guidance(scene):
    response = TestClient(app).post(
        "/v1/interrupt/evaluate", json=_base(requested_scene=scene, current_user_text=_BRANCHING),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["should_interrupt"]
    assert body["requested_scene"] == scene
    assert body["reason_json"]["requested_scene"] == scene
    assert body["intervention_text"] == (
        "You are branching again. Pick the next move and test it."
        " Keep it to the next concrete step."
    )
    assert body["intervention_text"] == body["debug"]["advisory_text"]
    assert len(body["intervention_text"]) <= 240


@pytest.mark.parametrize("payload", [
    {}, {"current_user_text": None}, {"current_user_text": "   "},
    {"recent_messages": [{"role": "user", "content": ""}]},
])
def test_missing_usable_input_has_no_intervention(payload):
    response = TestClient(app).post("/v1/interrupt/evaluate", json=_base(**payload))
    assert response.status_code == 200
    body = response.json()
    assert body["should_defer"] and not body["should_interrupt"]
    assert body["intervention_text"] is None
    assert body["debug"]["advisory_text"] is None
    assert body["debug"]["degraded"] is True
    assert "missing_user_text" in body["warnings"]


def test_empty_explicit_text_retains_request_validation():
    response = TestClient(app).post("/v1/interrupt/evaluate", json=_base(current_user_text=""))
    assert response.status_code == 422
    assert "intervention_text" not in response.json()


def test_intervention_does_not_copy_user_or_provider_text():
    client = TestClient(app)
    normal = client.post(
        "/v1/interrupt/evaluate", json=_base(current_user_text=_HIGH_BRANCHING),
    ).json()
    altered = client.post("/v1/interrupt/evaluate", json=_base(
        request_id="rid-interrupt-private",
        current_user_text=_HIGH_BRANCHING + " PRIVATE-USER-SENTINEL",
        recent_messages=[{"role": "assistant", "content": "PRIVATE-PROVIDER-SENTINEL "
                          "I'm always listening. I need you here. Don't leave me."}],
    )).json()
    for field in ["trigger_class", "confidence", "style_selected", "should_interrupt",
                  "should_defer", "intervention_text"]:
        assert normal[field] == altered[field]
    for prohibited in ["PRIVATE-USER-SENTINEL", "PRIVATE-PROVIDER-SENTINEL", "always listening",
                       "I need you", "Don't leave me"]:
        assert prohibited not in altered["intervention_text"]


def test_authorized_recurring_trap_has_usable_constraint_reset_text():
    client = TestClient(app)
    text = ("I am stuck in the same loop again and keep rehashing the same plan "
            "without making a concrete decision about the next useful step.")
    updated = client.post("/v1/runtime/state/update", json={
        "request_id": "trap-state", "owner_id": "owner",
        "conversation_id": "conv-1", "surface": "dev",
        "updates": {"temporary_constraints": ["avoid_loop_spiral"],
                    "interaction_mode": "actionable"},
    })
    assert updated.status_code == 200
    response = client.post("/v1/interrupt/evaluate", json=_base(
        current_user_text=text,
        recent_messages=[{"role": "user", "content": text} for _ in range(5)],
    ))
    assert response.status_code == 200
    body = response.json()
    assert body["should_interrupt"] and not body["should_defer"]
    assert body["trigger_class"] == "known_recurring_trap_pattern"
    assert body["confidence"] == 1.0
    assert body["style_selected"] == "constraint_reset"
    assert body["intervention_text"] == "Reset to the immediate objective."
    assert body["debug"]["advisory_text"] == body["intervention_text"]


@pytest.mark.parametrize("overrides", [
    {"intervention_text": None}, {"intervention_text": ""}, {"intervention_text": "   "},
    {"intervention_text": "x" * 241}, {"intervention_text": True},
    {"should_defer": True}, {"style_selected": None}, {"trigger_class": None},
    {"should_interrupt": False, "should_defer": True},
])
def test_response_contract_rejects_incoherent_intervention(overrides):
    response = TestClient(app).post(
        "/v1/interrupt/evaluate", json=_base(current_user_text=_HIGH_BRANCHING),
    )
    assert response.status_code == 200
    with pytest.raises(ValidationError):
        InterruptEvaluateResponse.model_validate({**response.json(), **overrides})


def test_production_candidate_can_be_read_without_diagnostic_text():
    response = TestClient(app).post(
        "/v1/interrupt/evaluate", json=_base(current_user_text=_HIGH_BRANCHING),
    ).json()
    response["debug"]["advisory_text"] = None
    parsed = InterruptEvaluateResponse.model_validate(response)
    assert parsed.intervention_text == "You are branching again. Pick the next move and test it."


@pytest.mark.parametrize("surface,exploration", [
    ("web", False), ("telegram", False), ("alexa", False), ("car", False),
    ("new_surface", False), ("unknown", False), ("web", True), ("new_surface", True),
])
@pytest.mark.parametrize("malformed", ['[]', '{private-invalid-json'])
def test_interrupt_uses_canonical_default_for_malformed_persisted_contract(
    surface, exploration, malformed,
):
    path = os.environ["COMPANION_CONTRACTS_DB_PATH"]
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE interaction_contracts SET trust_rules_json = ?", (malformed,))
        before = conn.execute("SELECT * FROM interaction_contracts ORDER BY id").fetchall()
    text = (
        "Brainstorm possibilities with me. What if we tried several approaches, "
        "compared options, and explored edge cases before choosing?"
        if exploration else
        "Should I rewrite this or add an abstraction or split the module or "
        "rework the interface or simplify the module or compare every option?"
    )
    response = TestClient(app).post(
        "/v1/interrupt/evaluate",
        json=_base(surface=surface, requested_scene="planning", current_user_text=text),
    )
    assert response.status_code == 200
    body = response.json()
    contract = body["interaction_contract"]
    assert {field: contract[field] for field in CONTRACT_RULES} == CONTRACT_RULES
    assert contract["source"] == "default_compiled"
    assert contract["scope"] == "global_default"
    assert contract["owner_id"] == "owner"
    warning = "malformed_interaction_contract_defaulted"
    assert body["warnings"].count(warning) == body["contract_trace"]["warnings"].count(warning) == 1
    assert "default_contract_applied" in body["warnings"]
    assert "private-invalid-json" not in response.text
    if surface == "new_surface":
        assert "unknown_surface_default_contract" in body["warnings"]
        assert "unknown_surface_interrupt_policy" in body["warnings"]
    if exploration:
        assert body["should_interrupt"] is False
        assert body["should_defer"] is True
        assert body["intervention_text"] is None
        assert "explicit_exploration_request" in body["reason_json"]["defer_reasons"]
    else:
        assert body["should_interrupt"] is True
        assert body["should_defer"] is False
        assert body["confidence"] >= 0.85
        assert body["style_selected"] == "next_step_forcing"
        assert body["contract_constraints_applied"]["matched_contract_style"] == "soft_redirect"
        assert body["intervention_text"] == (
            "You are branching again. Pick the next move and test it. "
            "Keep it to the next concrete step."
        )
        assert 0 < len(body["intervention_text"]) <= 240
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT * FROM interaction_contracts ORDER BY id").fetchall() == before


def _evaluate(request="eval-first", text=_HIGH_BRANCHING, **scope):
    return TestClient(app).post(
        "/v1/interrupt/evaluate",
        json=_base(
            request_id=request,
            current_user_text=text,
            **scope,
        ),
    )


def _receipt(evaluation, **changes):
    body = evaluation.json() if hasattr(evaluation, "json") else evaluation
    payload = {
        key: body[key]
        for key in (
            "request_id",
            "owner_id",
            "conversation_id",
            "surface",
            "trigger_class",
            "style_selected",
            "intervention_text",
        )
    }
    return {**payload, **changes}


def _execute(evaluation, **changes):
    return TestClient(app).post("/v1/interrupt/execute", json=_receipt(evaluation, **changes))


def _events():
    return companion_contracts_repository().list_interaction_boundary_events_for_tests()


def _debug(request="eval-first", owner="owner", conversation="conv-1"):
    return TestClient(app).get(
        f"/v1/interrupt/debug/{request}",
        params={
            "owner_id": owner,
            "conversation_id": conversation,
        },
    )


def test_evaluation_is_structural_idempotent_and_conflicting_reuse_rejected():
    first = _evaluate(text=_HIGH_BRANCHING + " PRIVATE-USER-TEXT")
    again = _evaluate(text=_HIGH_BRANCHING + " PRIVATE-USER-TEXT")
    assert first.status_code == again.status_code == 200
    assert first.json()["should_interrupt"] is True
    assert first.json()["lifecycle"]["state"] == "none"
    assert again.json()["lifecycle"] == first.json()["lifecycle"]
    assert len(_events()) == 1
    stored = _events()[0]
    assert stored["check_type"] == "interrupt_evaluation"
    assert stored["result"] == "authorized"
    serialized = str(stored)
    assert "PRIVATE-USER-TEXT" not in serialized
    assert first.json()["intervention_text"] not in serialized
    assert "advisory_text" not in serialized and "detector_signals" not in serialized
    assert _evaluate(text="What is 2+2?").status_code == 409
    assert len(_events()) == 1


def test_execution_receipt_and_idempotency_survive_recovery_without_resetting_state():
    evaluated = _evaluate()
    first, duplicate = _execute(evaluated), _execute(evaluated)
    assert first.status_code == duplicate.status_code == 200
    assert first.json()["execution_recorded"] is True
    assert first.json()["idempotent_replay"] is False
    assert duplicate.json()["idempotent_replay"] is True
    assert first.json()["lifecycle"]["state"] == "awaiting_feedback"
    assert [row["check_type"] for row in _events()] == [
        "interrupt_evaluation",
        "interrupt_execution",
    ]
    assert evaluated.json()["intervention_text"] not in str(_events())
    next_turn = _evaluate("eval-next")
    assert next_turn.json()["lifecycle"]["state"] == "overridden"
    again = _execute(evaluated)
    assert again.json()["idempotent_replay"] is True
    assert again.json()["lifecycle"]["state"] == "overridden"
    assert len([e for e in _events() if e["check_type"] == "interrupt_execution"]) == 1
    assert _evaluate("eval-third").json()["lifecycle"]["state"] == "repeat_suppressed"


@pytest.mark.parametrize(
    "changes,status",
    [
        ({"trigger_class": "known_recurring_trap_pattern"}, 409),
        ({"style_selected": "soft_redirect"}, 409),
        ({"intervention_text": "Different bounded text."}, 409),
        ({"request_id": "unknown"}, 404),
        ({"owner_id": "another-owner"}, 404),
        ({"conversation_id": "another-conversation"}, 404),
        ({"surface": "telegram"}, 404),
        ({"intervention_text": " "}, 422),
        ({"intervention_text": "x" * 241}, 422),
        ({"extra": "private"}, 422),
        ({"trigger_class": "unsupported"}, 422),
        ({"request_id": True}, 422),
    ],
)
def test_invalid_receipt_cannot_create_execution(changes, status):
    evaluated = _evaluate()
    assert _execute(evaluated, **changes).status_code == status
    assert len(_events()) == 1


def test_deferred_exploration_has_no_execution_authority():
    response = _evaluate(text="Brainstorm possibilities with me. What if we explored options?")
    assert response.json()["should_defer"] is True
    assert response.json()["intervention_text"] is None
    assert _events()[0]["result"] == "deferred"
    assert (
        _execute(
            response,
            intervention_text="A fabricated redirect.",
            trigger_class="repetitive_branching",
            style_selected="next_step_forcing",
        ).status_code
        == 409
    )
    assert len(_events()) == 1


def test_nonrecurring_pattern_is_accepted_and_debug_is_private():
    first = _evaluate()
    assert _execute(first).status_code == 200
    second = _evaluate("eval-next", "What is 2+2?")
    assert second.status_code == 200
    assert second.json()["lifecycle"]["state"] == "accepted"
    assert second.json()["lifecycle"]["candidate_suppressed"] is False
    assert [e["result"] for e in _events() if e["check_type"] == "interrupt_recovery"] == [
        "accepted"
    ]
    debug = _debug()
    assert debug.status_code == 200
    assert debug.json()["execution_state"] == "executed"
    assert debug.json()["recovery_outcome"] == "accepted"
    assert debug.json()["lifecycle"]["prior_executed_request_id"] == "eval-first"
    for forbidden in (
        "intervention_text",
        "candidate_digest",
        "input_digest",
        "reason_json",
        "detector_signals",
        "debug",
        first.json()["intervention_text"],
    ):
        assert forbidden not in debug.text


def test_override_suppresses_repeats_until_pattern_breaks_then_rearms():
    first = _evaluate()
    assert _execute(first).status_code == 200
    for request, state, count in (
        ("eval-second", "overridden", 1),
        ("eval-third", "repeat_suppressed", 2),
        ("eval-fourth", "repeat_suppressed", 3),
    ):
        response = _evaluate(request)
        assert response.status_code == 200
        body = response.json()
        assert body["trigger_class"] == "repetitive_branching"
        assert body["style_selected"] == "next_step_forcing" and body["confidence"] >= 0.85
        assert body["should_interrupt"] is False and body["should_defer"] is True
        assert body["intervention_text"] is None
        assert body["lifecycle"]["state"] == state
        assert body["lifecycle"]["repeated_trigger_count"] == count
        assert body["lifecycle"]["candidate_suppressed"] is True
        assert body["reason_json"]["defer_reasons"] == []  # Existing CO enum remains unchanged.
        repeated = _evaluate(request)
        assert repeated.json()["lifecycle"] == body["lifecycle"]
        assert (
            _execute(response, intervention_text=first.json()["intervention_text"]).status_code
            == 409
        )
    assert _debug().json()["recovery_outcome"] == "overridden"
    recovered = _evaluate("eval-break", "What is 2+2?")
    assert recovered.json()["lifecycle"]["state"] == "recovered"
    assert recovered.json()["lifecycle"]["candidate_suppressed"] is False
    assert _debug().json()["recovery_outcome"] == "recovered"
    rearmed = _evaluate("eval-rearmed")
    assert rearmed.json()["should_interrupt"] is True
    assert rearmed.json()["lifecycle"]["state"] == "none"
    assert [e["result"] for e in _events() if e["check_type"] == "interrupt_recovery"] == [
        "overridden",
        "recovered",
    ]


def test_cross_surface_lifecycle_is_conversation_scoped_but_owners_and_threads_are_isolated():
    first = _evaluate(surface="telegram")
    assert _execute(first).status_code == 200
    assert (
        _evaluate("eval-other-owner", owner_id="another-owner").json()["should_interrupt"] is True
    )
    assert (
        _evaluate("eval-other-thread", conversation_id="another-conversation").json()[
            "should_interrupt"
        ]
        is True
    )
    current = _evaluate("eval-web", surface="web")
    assert current.json()["lifecycle"]["state"] == "overridden"
    assert current.json()["lifecycle"]["prior_executed_request_id"] == "eval-first"
    assert current.json()["should_interrupt"] is False
    events = [e for e in _events() if e["owner_id"] == "owner" and e["conversation_id"] == "conv-1"]
    assert [(e["check_type"], e["surface"]) for e in events] == [
        ("interrupt_evaluation", "telegram"),
        ("interrupt_execution", "telegram"),
        ("interrupt_recovery", "web"),
        ("interrupt_evaluation", "web"),
    ]


def test_lifecycle_survives_repository_replacement():
    first = _evaluate()
    assert _execute(first).status_code == 200
    reset_companion_contracts_for_tests(db_path=Path(os.environ["COMPANION_CONTRACTS_DB_PATH"]))
    assert _debug().json()["execution_state"] == "executed"
    current = _evaluate("eval-after-restart")
    assert current.status_code == 200
    assert current.json()["lifecycle"]["state"] == "overridden"
    assert current.json()["should_interrupt"] is False


@pytest.mark.parametrize(
    "corruption",
    [
        "bad_json",
        "wrong_trigger",
        "foreign_evaluation",
        "wrong_surface",
        "unknown_result",
        "unknown_type",
        "unexpected_key",
        "missing_version",
        "forged_lifecycle",
        "duplicate_execution",
    ],
)
def test_malformed_history_fails_closed_without_exposing_storage(corruption):
    first = _evaluate()
    assert _execute(first).status_code == 200
    with sqlite3.connect(os.environ["COMPANION_CONTRACTS_DB_PATH"]) as conn:
        execution = conn.execute(
            "SELECT id,reason_json FROM interaction_boundary_events "
            "WHERE check_type='interrupt_execution'"
        ).fetchone()
        proof = json.loads(execution[1])
        if corruption == "bad_json":
            conn.execute(
                "UPDATE interaction_boundary_events SET reason_json=? WHERE id=?",
                ("PRIVATE-MALFORMED-JSON", execution[0]),
            )
        elif corruption == "wrong_surface":
            conn.execute(
                "UPDATE interaction_boundary_events SET surface='alexa' WHERE id=?", (execution[0],)
            )
        elif corruption == "unknown_type":
            conn.execute("UPDATE interaction_boundary_events SET check_type='interrupt_unknown' "
                         "WHERE id=?", (execution[0],))
        elif corruption == "unknown_result":
            conn.execute(
                "UPDATE interaction_boundary_events SET result='private-invalid' WHERE id=?",
                (execution[0],),
            )
        elif corruption == "duplicate_execution":
            columns = [
                r[1] for r in conn.execute("PRAGMA table_info(interaction_boundary_events)")
            ][1:]
            names = ",".join(columns)
            conn.execute(
                f"INSERT INTO interaction_boundary_events ({names}) SELECT {names} "
                "FROM interaction_boundary_events WHERE id=?",
                (execution[0],),
            )
        elif corruption == "forged_lifecycle":
            row = conn.execute(
                "SELECT id,reason_json FROM interaction_boundary_events "
                "WHERE check_type='interrupt_evaluation'"
            ).fetchone()
            data = json.loads(row[1])
            data["lifecycle"]["candidate_suppressed"] = True
            conn.execute(
                "UPDATE interaction_boundary_events SET reason_json=? WHERE id=?",
                (json.dumps(data), row[0]),
            )
        else:
            if corruption == "wrong_trigger":
                proof["trigger_class"] = "known_recurring_trap_pattern"
            elif corruption == "foreign_evaluation":
                foreign = _evaluate("eval-foreign", owner_id="foreign-owner")
                assert foreign.status_code == 200
                proof["evaluation_id"] = max(e["id"] for e in _events())
            elif corruption == "missing_version":
                del proof["schema_version"]
            else:
                proof["raw_private"] = "PRIVATE-MALFORMED-JSON"
            conn.execute(
                "UPDATE interaction_boundary_events SET reason_json=? WHERE id=?",
                (json.dumps(proof), execution[0]),
            )
    current = _evaluate("eval-after-corruption")
    assert current.status_code == 200
    assert current.json()["lifecycle"]["state"] == "history_unavailable"
    assert current.json()["lifecycle"]["prior_executed_request_id"] is None
    assert current.json()["should_interrupt"] is False and current.json()["should_defer"] is True
    assert current.json()["intervention_text"] is None
    assert "PRIVATE-MALFORMED-JSON" not in current.text
    assert "private-invalid" not in current.text
    assert _execute(first).status_code == 409
    assert _debug().status_code == 409
    assert "PRIVATE-MALFORMED-JSON" not in _debug().text
    assert not [e for e in _events() if e["check_type"] == "interrupt_recovery"]


@pytest.mark.parametrize(
    "request_key,owner,conversation",
    [
        ("unknown", "owner", "conv-1"),
        ("eval-first", "other-owner", "conv-1"),
        ("eval-first", "owner", "other-conversation"),
    ],
)
def test_debug_isolation_has_indistinguishable_not_found(request_key, owner, conversation):
    _evaluate()
    response = _debug(request_key, owner, conversation)
    assert response.status_code == 404
    assert response.json() == {"detail": "interrupt_request_not_found"}


def test_debug_requires_scope_and_bounded_request():
    client = TestClient(app)
    assert client.get("/v1/interrupt/debug/eval-first").status_code == 422
    assert _debug("x" * 121).status_code == 422


def test_unrelated_lifecycle_storage_failure_is_not_success(monkeypatch):
    def unavailable():
        raise sqlite3.OperationalError("PRIVATE-STORAGE-ERROR")

    monkeypatch.setattr(
        companion_contracts_repository(), "interaction_boundary_transaction", unavailable
    )
    with pytest.raises(sqlite3.OperationalError):
        _evaluate()
