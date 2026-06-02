from fastapi.testclient import TestClient
from main import app
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


def test_interrupt_evaluate_accepts_legacy_default_static_contract_source():
    client = TestClient(app)

    response = client.post(
        "/v1/interrupt/evaluate",
        json=_base(
            current_user_text=(
                "Should I rewrite this or add an abstraction or split the module or "
                "rework the interface or pause and compare every option?"
            ),
            interaction_contract={
                "contract_id": "interaction_contract_r19_default_static",
                "contract_version": 1,
                "owner_id": "owner",
                "scope": "global_default",
                "source": "default_static",
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
                "contract_id": "interaction_contract_r19_default_static",
                "contract_version": 1,
                "source": "default_static",
                "scope": "global_default",
                "selected_rule_groups": ["allowed_intervention_styles"],
                "selected_boundary_rules": ["no pressure"],
                "selected_repair_rules": ["repair clearly"],
                "warnings": ["default_static_contract"],
            },
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["should_defer"] is True
    assert "no_contract_permitted_style" in body["reason_json"]["defer_reasons"]
    assert "default_contract_source" in body["warnings"]
    assert "default_contract_applied" in body["warnings"]
    assert "default_static_contract" not in body["warnings"]
