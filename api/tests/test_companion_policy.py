import json
import os
import sqlite3

import pytest
from fastapi.testclient import TestClient
from main import app
from services.companion_contracts import (
    CONTRACT_RULES,
    DEFAULT_CONTRACT_ID,
    DEFAULT_CONTRACT_VERSION,
    DEFAULT_PROFILE_VERSION,
    companion_contracts_repository,
)
from services.companion_policy import compile_policy, resolve_interaction_contract
from services.runtime_state import clear_states_for_tests, resolve_state


def setup_function():
    clear_states_for_tests()


def _base():
    return {
        "request_id": "rid-companion",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
    }


def test_compile_returns_bounded_ordered_policy_overlays():
    client = TestClient(app)

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    body = response.json()
    assert body["profile_id"] == "default_companion_profile"
    assert body["profile_version"] == DEFAULT_PROFILE_VERSION
    assert body["contract_id"] == "default_interaction_contract"
    assert body["contract_version"] == 2
    assert body["scene_id"] == "general"
    assert body["scene_source"] == "general"
    assert body["warnings"] == ["default_contract_applied"]
    assert [overlay["overlay_type"] for overlay in body["overlays"]] == [
        "interaction_contract",
        "companion_profile",
        "scene_policy",
    ]
    for overlay in body["overlays"]:
        assert overlay["role"] == "system"
        assert 0 < len(overlay["content"]) <= 1200


def test_compile_returns_structured_interaction_contract():
    client = TestClient(app)

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    contract = response.json()["interaction_contract"]
    assert contract["contract_id"] == "default_interaction_contract"
    assert contract["contract_version"] == 2
    assert contract["owner_id"] == "owner"
    assert contract["scope"] == "global_default"
    assert contract["source"] == "default_compiled"
    for field in (
        "trust_rules",
        "interaction_boundaries",
        "repair_rules",
        "memory_or_recall_boundaries",
        "autonomy_rules",
        "tone_constraints",
        "allowed_intervention_styles",
        "disallowed_intervention_styles",
        "defer_conditions",
    ):
        assert contract[field]
    assert "perform_closeness" not in contract["allowed_intervention_styles"]
    assert "performative_memory" in contract["disallowed_intervention_styles"]


def test_compile_returns_inspectable_contract_trace():
    client = TestClient(app)

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    trace = response.json()["contract_trace"]
    assert trace["contract_id"] == "default_interaction_contract"
    assert trace["contract_version"] == 2
    assert trace["source"] == "default_compiled"
    assert trace["scope"] == "global_default"
    assert trace["warnings"] == ["default_contract_applied"]
    assert trace["selected_rule_groups"] == [
        "trust_rules",
        "interaction_boundaries",
        "repair_rules",
        "memory_or_recall_boundaries",
        "autonomy_rules",
        "tone_constraints",
        "allowed_intervention_styles",
        "disallowed_intervention_styles",
        "defer_conditions",
    ]
    assert trace["selected_boundary_rules"]
    assert trace["selected_repair_rules"]


def test_interaction_contract_overlay_is_concise_and_not_interrupt_execution():
    client = TestClient(app)

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    overlay = response.json()["overlays"][0]
    assert overlay["overlay_type"] == "interaction_contract"
    content = overlay["content"]
    assert "Interaction contract:" in content
    assert "Memory:" in content
    assert "Repair:" in content
    blocked_phrases = (
        "detect loops",
        "interrupt when",
        "trigger class",
        "grounding trigger",
        "speculation-loop",
    )
    assert all(phrase not in content.lower() for phrase in blocked_phrases)


def test_explicit_scene_resolves_scene_policy():
    client = TestClient(app)

    response = client.post(
        "/v1/companion/policy/compile",
        json={**_base(), "requested_scene": "planning"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "planning"
    assert body["scene_confidence"] == 1.0
    assert body["scene_source"] == "requested_scene"
    assert body["warnings"] == ["default_contract_applied"]
    assert "Scene policy: planning" in body["overlays"][2]["content"]


def test_requested_scene_aliases_return_canonical_scene_ids():
    client = TestClient(app)

    aliases = {
        "coding": "coding_build",
        "coding_build_mode": "coding_build",
        "reflective_conversation": "reflective",
        "notifications_briefings": "briefing",
    }
    for alias, canonical in aliases.items():
        response = client.post(
            "/v1/companion/policy/compile",
            json={**_base(), "requested_scene": alias},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["scene_id"] == canonical
        assert body["scene_source"] == "requested_scene"
        assert body["warnings"] == ["default_contract_applied"]


def test_runtime_scene_alias_returns_canonical_scene_id():
    client = TestClient(app)
    client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"active_scene": "coding_build_mode"}},
    )

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "coding_build"
    assert body["scene_source"] == "runtime_state"
    assert body["warnings"] == ["default_contract_applied"]


def test_unknown_requested_scene_falls_back_without_using_runtime_scene():
    client = TestClient(app)
    client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"active_scene": "planning"}},
    )

    response = client.post(
        "/v1/companion/policy/compile",
        json={**_base(), "requested_scene": "not_a_scene"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "general"
    assert body["scene_source"] == "fallback_general"
    assert body["warnings"] == ["unknown_requested_scene", "default_contract_applied"]
    assert body["contract_trace"]["warnings"] == [
        "default_contract_applied",
        "unknown_requested_scene",
    ]


def test_unknown_surface_falls_back_to_default_contract_with_warning():
    client = TestClient(app)

    response = client.post(
        "/v1/companion/policy/compile",
        json={**_base(), "surface": "new_surface"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["interaction_contract"]["source"] == "default_compiled"
    assert body["contract_trace"]["warnings"] == [
        "default_contract_applied",
        "unknown_surface_default_contract",
    ]
    assert body["warnings"] == [
        "default_contract_applied",
        "unknown_surface_default_contract",
    ]


def test_runtime_scene_used_only_when_requested_scene_absent():
    client = TestClient(app)
    client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"active_scene": "planning"}},
    )

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "planning"
    assert body["scene_source"] == "runtime_state"
    assert body["warnings"] == ["default_contract_applied"]


def test_unknown_runtime_scene_falls_back_with_warning():
    client = TestClient(app)
    client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"active_scene": "not_a_scene"}},
    )

    response = client.post("/v1/companion/policy/compile", json=_base())

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "general"
    assert body["scene_source"] == "fallback_general"
    assert body["warnings"] == ["unknown_runtime_scene", "default_contract_applied"]
    assert body["contract_trace"]["warnings"] == [
        "default_contract_applied",
        "unknown_runtime_scene",
    ]


def test_compile_output_is_deterministic():
    client = TestClient(app)
    payload = {**_base(), "requested_scene": "work_triage"}

    first = client.post("/v1/companion/policy/compile", json=payload)
    second = client.post("/v1/companion/policy/compile", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["overlays"] == second.json()["overlays"]
    assert first.json()["interaction_contract"] == second.json()["interaction_contract"]
    assert first.json()["contract_trace"] == second.json()["contract_trace"]
    assert first.json()["scene_id"] == second.json()["scene_id"]


def test_structurally_invalid_requested_scene_is_rejected():
    client = TestClient(app)

    for value in ({"scene": "planning"}, ["planning"], 123):
        response = client.post(
            "/v1/companion/policy/compile",
            json={**_base(), "requested_scene": value},
        )
        assert response.status_code == 422


def test_oversized_requested_scene_is_rejected():
    client = TestClient(app)

    response = client.post(
        "/v1/companion/policy/compile",
        json={**_base(), "requested_scene": "x" * 65},
    )

    assert response.status_code == 422


def test_active_profile_endpoint_returns_seeded_default_record():
    client = TestClient(app)

    response = client.get("/v1/companion/profile/active")

    assert response.status_code == 200
    body = response.json()
    assert body["profile_id"] == "default_companion_profile"
    assert body["profile_version"] == DEFAULT_PROFILE_VERSION
    assert body["scope"] == "global_default"
    assert body["source"] == "seeded_default"
    assert body["status"] == "active"
    assert body["role_label"] == "personal_intelligence_companion"
    assert body["core_traits_json"]["directness"] == "high"


def test_profile_compile_endpoint_matches_policy_compile_alias():
    client = TestClient(app)
    payload = {**_base(), "requested_scene": "planning"}

    profile_response = client.post("/v1/companion/profile/compile", json=payload)
    policy_response = client.post("/v1/companion/policy/compile", json=payload)

    assert profile_response.status_code == 200
    assert policy_response.status_code == 200
    assert profile_response.json() == policy_response.json()


def test_compile_uses_seeded_general_scene_for_unknown_scene():
    client = TestClient(app)

    response = client.post(
        "/v1/companion/profile/compile",
        json={**_base(), "requested_scene": "unknown_scene"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scene_id"] == "general"
    assert body["scene_source"] == "fallback_general"
    assert "Scene policy: use the general operating mode" in body["overlays"][2]["content"]
    assert body["profile_version"] == DEFAULT_PROFILE_VERSION
    assert body["contract_version"] == 2


@pytest.mark.parametrize("column,value", [
    ("trust_rules_json", "[]"),
    ("interaction_boundaries_json", "[]"),
    ("memory_or_recall_boundaries_json", "[]"),
    ("autonomy_rules_json", "[]"),
    ("tone_constraints_json", "[]"),
    ("repair_rules_json", '["Only one repair rule"]'),
    ("trust_rules_json", '{private-invalid-json'),
    ("trust_rules_json", '{"wrong": "container"}'),
    ("trust_rules_json", 'null'),
    ("trust_rules_json", '[42]'),
    ("scope", "invalid-scope"),
])
def test_compile_defaults_malformed_contract_without_rewriting(column, value):
    path = os.environ["COMPANION_CONTRACTS_DB_PATH"]
    with sqlite3.connect(path) as conn:
        conn.execute(f"UPDATE interaction_contracts SET {column} = ?", (value,))
        before = conn.execute("SELECT * FROM interaction_contracts ORDER BY id").fetchall()
    client = TestClient(app)
    first = client.post("/v1/companion/policy/compile", json=_base())
    second = client.post("/v1/companion/policy/compile", json=_base())
    assert first.status_code == second.status_code == 200
    body = first.json()
    contract = body["interaction_contract"]
    assert {field: contract[field] for field in CONTRACT_RULES} == CONTRACT_RULES
    assert contract["contract_id"] == DEFAULT_CONTRACT_ID
    assert contract["contract_version"] == DEFAULT_CONTRACT_VERSION
    assert contract["owner_id"] == "owner"
    assert contract["source"] == "default_compiled"
    assert contract["scope"] == "global_default"
    profile = companion_contracts_repository().active_profile()
    assert (body["profile_id"], body["profile_version"]) == (profile.profile_id, profile.version)
    expected = ["default_contract_applied", "malformed_interaction_contract_defaulted"]
    assert body["warnings"] == body["contract_trace"]["warnings"] == expected
    assert second.json()["contract_trace"] == body["contract_trace"]
    assert second.json()["overlays"] == body["overlays"]
    assert "private-invalid-json" not in first.text
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT * FROM interaction_contracts ORDER BY id").fetchall() == before


def test_compile_valid_persisted_contract_is_preserved():
    rules = ["Use this valid configured trust rule."]
    with sqlite3.connect(os.environ["COMPANION_CONTRACTS_DB_PATH"]) as conn:
        conn.execute(
            "UPDATE interaction_contracts SET trust_rules_json = ?, contract_version = 3",
            (json.dumps(rules),),
        )
    response = TestClient(app).post("/v1/companion/policy/compile", json=_base())
    assert response.status_code == 200
    body = response.json()
    assert body["interaction_contract"]["trust_rules"] == rules
    assert body["contract_version"] == 3
    assert body["contract_trace"]["warnings"] == body["warnings"] == ["default_contract_applied"]


@pytest.mark.parametrize("surface,scene,expected", [
    ("new_surface", None, ["default_contract_applied", "unknown_surface_default_contract",
                           "malformed_interaction_contract_defaulted"]),
    ("dev", "not_a_scene", ["unknown_requested_scene", "default_contract_applied",
                            "malformed_interaction_contract_defaulted"]),
])
def test_malformed_contract_warning_composes_with_surface_and_scene(surface, scene, expected):
    with sqlite3.connect(os.environ["COMPANION_CONTRACTS_DB_PATH"]) as conn:
        conn.execute("UPDATE interaction_contracts SET trust_rules_json = '[]'")
    payload = {**_base(), "surface": surface}
    if scene:
        payload["requested_scene"] = scene
    response = TestClient(app).post("/v1/companion/policy/compile", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["warnings"] == expected
    assert body["contract_trace"]["warnings"].count("malformed_interaction_contract_defaulted") == 1
    assert body["scene_id"] == "general"
    assert body["scene_source"] == ("fallback_general" if scene else "general")
    assert body["interaction_contract"]["trust_rules"] == CONTRACT_RULES["trust_rules"]


@pytest.mark.parametrize("error", [
    sqlite3.OperationalError("storage-unavailable"),
    RuntimeError("unexpected-repository-failure"),
    RuntimeError("active_interaction_contract_missing"),
])
@pytest.mark.parametrize("consumer", ["compile", "interrupt_contract"])
def test_contract_operational_failures_do_not_default(monkeypatch, error, consumer):
    repository = companion_contracts_repository()
    state = resolve_state(owner_id="owner", conversation_id="conv-1", surface="dev")

    def unavailable(**kwargs):
        raise error

    monkeypatch.setattr(repository, "active_interaction_contract", unavailable)
    with pytest.raises(type(error), match=str(error)):
        if consumer == "compile":
            compile_policy(state=state)
        else:
            resolve_interaction_contract(
                owner_id="owner", surface="dev", requested_scene=None, runtime_state=state,
            )


@pytest.mark.parametrize(
    ("mode", "expected_scene"),
    [
        ("coding_build", "coding_build"),
        ("coding", "coding_build"),
        ("coding_build_mode", "coding_build"),
        ("planning", "planning"),
        ("work_triage", "work_triage"),
    ],
)
def test_compile_infers_only_registered_work_modes_without_state_mutation(mode, expected_scene):
    client = TestClient(app)
    updated = client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"interaction_mode": mode}},
    )
    assert updated.status_code == 200
    state_before = resolve_state(**{key: _base()[key] for key in (
        "owner_id", "conversation_id", "surface",
    )}).model_dump()

    inferred = client.post("/v1/companion/policy/compile", json=_base())
    explicit = client.post(
        "/v1/companion/policy/compile",
        json={**_base(), "requested_scene": expected_scene},
    )

    assert inferred.status_code == explicit.status_code == 200
    body = inferred.json()
    assert body["scene_id"] == expected_scene
    assert body["scene_source"] == "runtime_interaction_mode"
    assert body["scene_confidence"] == 0.6
    assert body["runtime_state"] == state_before
    assert body["overlays"] == explicit.json()["overlays"]
    assert body["interaction_contract"] == explicit.json()["interaction_contract"]
    assert resolve_state(**{key: _base()[key] for key in (
        "owner_id", "conversation_id", "surface",
    )}).model_dump() == state_before
    assert companion_contracts_repository().list_scene_resolution_events_for_tests() == []


@pytest.mark.parametrize("mode", [
    None, "", "unknown", "driving", "overload_recovery", "reflective",
    "reflective_conversation", "notifications_briefings", "medical_urgency",
    "voice_private", "technical_architect", "Coding", " coding ",
])
def test_compile_does_not_infer_sensitive_or_unrecognized_modes(mode):
    client = TestClient(app)
    assert client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"interaction_mode": mode}},
    ).status_code == 200
    response = client.post("/v1/companion/policy/compile", json=_base())
    assert response.status_code == 200
    assert response.json()["scene_id"] == "general"
    assert response.json()["scene_source"] == "general"
    assert response.json()["runtime_state"]["active_scene"] is None


def test_scene_selection_across_requests_preserves_saved_choice_and_identity():
    client = TestClient(app)
    scope = {key: _base()[key] for key in ("owner_id", "conversation_id", "surface")}
    assert client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {
            "active_scene": "planning", "interaction_mode": "coding",
            "temporary_constraints": ["do_not_personalize"],
        }},
    ).status_code == 200
    saved_state = resolve_state(**scope).model_dump()

    def compile_request(request_id, **fields):
        response = client.post(
            "/v1/companion/policy/compile",
            json={**_base(), "request_id": request_id, **fields},
        )
        assert response.status_code == 200
        return response.json()

    saved = compile_request("scene-saved")
    override = compile_request("scene-override", requested_scene="work_triage")
    restored = compile_request("scene-restored")
    assert [body["scene_id"] for body in (saved, override, restored)] == [
        "planning", "work_triage", "planning",
    ]
    assert [body["scene_source"] for body in (saved, override, restored)] == [
        "runtime_state", "requested_scene", "runtime_state",
    ]
    assert saved["overlays"][2]["content"] != override["overlays"][2]["content"]
    assert saved["overlays"] == restored["overlays"]
    assert resolve_state(**scope).model_dump() == saved_state

    assert client.post("/v1/runtime/state/reset", json=_base()).status_code == 200
    general = compile_request("scene-reset")
    assert general["scene_id"] == "general"
    assert general["runtime_state"]["active_scene"] is None
    assert general["runtime_state"]["interaction_mode"] is None
    for mode, expected_scene in (("coding", "coding_build"), ("work_triage", "work_triage")):
        assert client.post(
            "/v1/runtime/state/update",
            json={**_base(), "updates": {"interaction_mode": mode}},
        ).status_code == 200
        inferred = compile_request("scene-inferred-" + mode)
        assert inferred["scene_id"] == expected_scene
        assert inferred["runtime_state"]["active_scene"] is None
        assert inferred["runtime_state"]["runtime_state_id"] == saved_state["runtime_state_id"]
        assert (inferred["profile_id"], inferred["profile_version"]) == (
            saved["profile_id"], saved["profile_version"],
        )
        assert inferred["interaction_contract"] == saved["interaction_contract"]
    transient = compile_request("scene-transient", requested_scene="planning")
    assert transient["scene_id"] == "planning"
    assert compile_request("scene-after-transient")["scene_id"] == "work_triage"


@pytest.mark.parametrize(
    ("requested", "saved", "mode", "expected", "source"),
    [
        ("work_triage", "planning", "coding", "work_triage", "requested_scene"),
        ("invalid", "planning", "coding", "general", "fallback_general"),
        (None, "planning", "coding", "planning", "runtime_state"),
        (None, "invalid", "coding", "general", "fallback_general"),
        (None, None, "coding", "coding_build", "runtime_interaction_mode"),
        (None, None, "planning", "planning", "runtime_interaction_mode"),
        (None, None, "work_triage", "work_triage", "runtime_interaction_mode"),
        (None, None, "driving", "general", "general"),
    ],
)
def test_bound_companion_compile_preserves_scene_precedence_and_read_only_state(
    requested, saved, mode, expected, source,
):
    client = TestClient(app)
    started = client.post("/v1/runtime/turns/start", json=_base())
    assert started.status_code == 200
    binding = {
        "runtime_session_id": started.json()["runtime_session"]["runtime_session_id"],
        "runtime_turn_id": started.json()["runtime_turn"]["runtime_turn_id"],
    }
    updated = client.post(
        "/v1/runtime/state/update",
        json={**_base(), "updates": {"active_scene": saved, "interaction_mode": mode}},
    )
    before = updated.json()["runtime_state"]
    request = {**_base(), **binding, "requested_scene": requested}
    first = client.post("/v1/companion/profile/compile", json=request)
    repeat = client.post("/v1/companion/policy/compile", json=request)
    legacy = client.post(
        "/v1/companion/profile/compile", json={**_base(), "requested_scene": requested},
    )
    assert first.status_code == repeat.status_code == legacy.status_code == 200
    assert first.json() == repeat.json() == legacy.json()
    assert first.json()["scene_id"] == expected
    assert first.json()["scene_source"] == source
    assert first.json()["runtime_state"] == before
    assert client.post("/v1/runtime/state/resolve", json=_base()).json()["runtime_state"] == before
    events = companion_contracts_repository().list_scene_resolution_events_for_tests()
    assert len(events) == (0 if expected == "general" else 1)
    if events:
        assert events[0]["resolved_scene_id"] == expected
        assert json.loads(events[0]["signals_json"])["transition"]["previous_scene_id"] == "general"


def _scene_turn(client, **scope):
    request = {**_base(), **scope}
    started = client.post("/v1/runtime/turns/start", json=request)
    assert started.status_code == 200, started.json()
    return {
        **request,
        "runtime_session_id": started.json()["runtime_session"]["runtime_session_id"],
        "runtime_turn_id": started.json()["runtime_turn"]["runtime_turn_id"],
    }


def _scene_transitions():
    return [
        json.loads(event["signals_json"])["transition"]
        for event in companion_contracts_repository().list_scene_resolution_events_for_tests()
        if "transition" in json.loads(event["signals_json"])
    ]


def test_persistent_scene_override_composes_policy_and_transitions_mid_session():
    client = TestClient(app)
    request = _scene_turn(client)
    assert client.post("/v1/runtime/state/update", json={
        **_base(), "updates": {"temporary_constraints": ["do_not_personalize"]},
    }).status_code == 200
    original = client.post("/v1/companion/profile/compile", json=request).json()
    assert original["scene_id"] == "general"
    assert _scene_transitions() == []

    planning = client.post("/v1/companion/scene/override", json={
        **request, "scene_id": "planning",
    })
    assert planning.status_code == 200
    assert planning.json()["runtime_state"]["active_scene"] == "planning"
    saved_state = planning.json()["runtime_state"]
    assert client.post("/v1/companion/scene/override", json={
        **request, "scene_id": "planning",
    }).json() == planning.json()
    assert len(_scene_transitions()) == 1
    assert client.post("/v1/runtime/state/update", json={
        **_base(), "updates": {"interaction_mode": "work_triage"},
    }).status_code == 200
    compiled = client.post("/v1/companion/policy/compile", json=request).json()
    assert compiled["scene_id"] == "planning"
    assert compiled["scene_source"] == "runtime_state"
    assert "Scene policy: planning" in compiled["overlays"][2]["content"]
    assert len(_scene_transitions()) == 1

    # Complete this turn, then clear the choice on a later turn in the same session.
    assert client.post("/v1/runtime/turns/complete", json={
        "request_id": "finish-planning", "runtime_session_id": request["runtime_session_id"],
        "runtime_turn_id": request["runtime_turn_id"],
    }).status_code == 200
    next_request = _scene_turn(client, request_id="next-scene")
    assert next_request["runtime_session_id"] == request["runtime_session_id"]
    cleared = client.post("/v1/companion/scene/override", json={
        **next_request, "scene_id": None,
    })
    assert cleared.status_code == 200
    assert cleared.json()["scene_id"] == "work_triage"
    assert cleared.json()["scene_source"] == "runtime_interaction_mode"
    triage = client.post("/v1/companion/policy/compile", json=next_request).json()
    assert "Scene policy: work triage" in triage["overlays"][2]["content"]
    assert client.post("/v1/runtime/state/update", json={
        **_base(), "updates": {"interaction_mode": "ordinary"},
    }).status_code == 200
    general = client.post("/v1/companion/profile/compile", json=next_request).json()
    assert general["scene_id"] == "general"
    assert general["scene_source"] == "general"
    assert general["overlays"] == original["overlays"]
    assert general["runtime_state"]["active_scene"] is None
    for body in (compiled, triage, general):
        assert (body["profile_id"], body["profile_version"]) == (
            original["profile_id"], original["profile_version"],
        )
        assert body["interaction_contract"] == original["interaction_contract"]
        assert all(
            a["content"] == b["content"]
            for a, b in zip(body["overlays"][:2], original["overlays"][:2])
        )
        for key in ("owner_id", "conversation_id", "surface", "runtime_state_id",
                    "temporary_constraints", "attention_focus", "trace_refs", "reset_after_turn"):
            assert body["runtime_state"][key] == saved_state[key]
    transitions = _scene_transitions()
    assert [(t["previous_scene_id"], t["scene_id"], t["source"]) for t in transitions] == [
        ("general", "planning", "runtime_state"),
        ("planning", "work_triage", "runtime_interaction_mode"),
        ("work_triage", "general", "general"),
    ]
    assert all(t["previous_scene_version"] == t["scene_version"] == 1 for t in transitions)
    assert all(set(t["policy_delta"]) == {"content"} for t in transitions)
    assert all(t["fallback_reason"] is None for t in transitions)
    events = companion_contracts_repository().list_scene_resolution_events_for_tests()
    assert [json.loads(e["signals_json"])["runtime_turn_id"] for e in events] == [
        request["runtime_turn_id"],
        next_request["runtime_turn_id"], next_request["runtime_turn_id"],
    ]
    assert all(e["owner_id"] == "owner" and e["conversation_id"] == "conv-1"
               and e["surface"] == "dev" for e in events)


@pytest.mark.parametrize("mode,expected", [("coding", "coding_build"), ("driving", "general")])
@pytest.mark.parametrize("saved", ["planning", "general"])
def test_temporary_selection_preserves_override_and_clear_restores_inference(mode, expected, saved):
    client = TestClient(app)
    request = _scene_turn(client)
    assert client.post("/v1/runtime/state/update", json={
        **_base(), "updates": {"interaction_mode": mode},
    }).status_code == 200
    assert client.post("/v1/companion/scene/override", json={
        **request, "scene_id": saved,
    }).status_code == 200
    temporary = client.post("/v1/companion/policy/compile", json={
        **request, "requested_scene": "work_triage",
    })
    assert temporary.json()["scene_id"] == "work_triage"
    assert temporary.json()["runtime_state"]["active_scene"] == saved
    restored = client.post("/v1/companion/profile/compile", json=request)
    assert restored.json()["scene_id"] == saved
    cleared = client.post("/v1/companion/scene/override", json={**request, "scene_id": None})
    assert cleared.status_code == 200
    assert cleared.json()["scene_id"] == expected
    assert cleared.json()["runtime_state"]["interaction_mode"] == mode
    assert cleared.json()["runtime_state"]["active_scene"] is None
    before = _scene_transitions()
    assert client.post("/v1/companion/scene/override", json={
        **request, "scene_id": None,
    }).json() == cleared.json()
    assert _scene_transitions() == before


@pytest.mark.parametrize("unknown_source", ["requested_scene", "runtime_scene"])
def test_bound_scene_fallback_transition_records_reason_without_unknown_input(unknown_source):
    client = TestClient(app)
    request = _scene_turn(client)
    client.post("/v1/companion/scene/override", json={**request, "scene_id": "planning"})
    fields = {"requested_scene": "unrecognized-private-label"}
    if unknown_source == "runtime_scene":
        assert client.post("/v1/runtime/state/update", json={
            **_base(), "updates": {"active_scene": "unrecognized-private-label"},
        }).status_code == 200
        fields = {}
    response = client.post("/v1/companion/policy/compile", json={**request, **fields})
    assert response.status_code == 200
    assert response.json()["scene_id"] == "general"
    assert response.json()["runtime_state"]["active_scene"] == (
        "planning" if fields else "unrecognized-private-label"
    )
    transition = _scene_transitions()[-1]
    assert (transition["previous_scene_id"], transition["scene_id"]) == ("planning", "general")
    assert transition["fallback_reason"] == "unknown_" + unknown_source
    assert transition["source"] == "fallback_general"
    assert "unrecognized-private-label" not in json.dumps(
        companion_contracts_repository().list_scene_resolution_events_for_tests(),
    )


def test_scene_transition_reports_only_changed_registered_policy_dimensions():
    client = TestClient(app)
    request = _scene_turn(client)
    with sqlite3.connect(os.environ["COMPANION_CONTRACTS_DB_PATH"]) as conn:
        conn.execute("UPDATE scene_policies SET format_policy_json = ? WHERE scene_id = ?",
                     (json.dumps({"format": "steps"}), "planning"))
    response = client.post("/v1/companion/scene/override", json={**request, "scene_id": "planning"})
    assert response.status_code == 200
    delta = _scene_transitions()[-1]["policy_delta"]
    assert set(delta) == {"content", "format"}
    assert delta["format"] == {"previous": {}, "current": {"format": "steps"}}
    assert delta["content"]["current"] == response.json()["policy"]["content"]


def test_diagnostic_scene_resolutions_do_not_become_applied_transitions():
    client = TestClient(app)
    request = _scene_turn(client)
    for _ in range(2):
        assert client.post("/v1/companion/scene/resolve", json={
            **request, "requested_scene": "planning",
        }).status_code == 200
    compiled = client.post("/v1/companion/policy/compile", json=request)
    assert compiled.status_code == 200
    assert compiled.json()["scene_id"] == "general"
    assert _scene_transitions() == []
    assert client.post("/v1/companion/scene/override", json={
        **request, "scene_id": "planning",
    }).status_code == 200
    assert client.post("/v1/companion/scene/resolve", json={
        **request, "requested_scene": "work_triage",
    }).status_code == 200
    before = _scene_transitions()
    assert client.post("/v1/companion/policy/compile", json=request).status_code == 200
    assert _scene_transitions() == before


def test_scene_override_persists_canonical_registered_alias():
    client = TestClient(app)
    request = _scene_turn(client)
    response = client.post("/v1/companion/scene/override", json={**request, "scene_id": "coding"})
    assert response.status_code == 200
    assert response.json()["scene_id"] == "coding_build"
    assert response.json()["runtime_state"]["active_scene"] == "coding_build"
    before = _scene_transitions()
    assert client.post("/v1/companion/scene/override", json={
        **request, "scene_id": "coding_build_mode",
    }).json() == response.json()
    assert _scene_transitions() == before
