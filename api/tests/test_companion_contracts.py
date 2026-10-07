from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from services.companion_contracts import (
    DEFAULT_DB_PATH,
    PERSONA_PROFILES,
    SCENE_POLICIES,
    SURFACE_BINDINGS,
    CompanionContractsRepository,
    companion_contracts_db_path,
)


def test_default_db_path_is_local_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("COMPANION_CONTRACTS_DB_PATH", raising=False)

    assert companion_contracts_db_path() == Path(DEFAULT_DB_PATH)


def test_repository_creates_parent_directory_and_seed_records(tmp_path):
    db_path = tmp_path / "missing" / "nested" / "companion_contracts.sqlite3"

    assert not db_path.parent.exists()

    repository = CompanionContractsRepository(db_path=db_path)

    assert db_path.parent.is_dir()
    assert db_path.exists()
    assert repository.record_counts() == {
        "companion_profiles": 1,
        "scene_policies": len(SCENE_POLICIES),
        "interaction_contracts": 1,
        "persona_profiles": len(PERSONA_PROFILES),
        "surface_bindings": len(SURFACE_BINDINGS),
        "scene_resolution_events": 0,
        "interaction_boundary_events": 0,
    }


def test_repository_initialization_is_idempotent(tmp_path):
    db_path = tmp_path / "contracts" / "companion_contracts.sqlite3"

    first = CompanionContractsRepository(db_path=db_path)
    second = CompanionContractsRepository(db_path=db_path)

    assert first.record_counts() == second.record_counts()
    assert second.record_counts() == {
        "companion_profiles": 1,
        "scene_policies": len(SCENE_POLICIES),
        "interaction_contracts": 1,
        "persona_profiles": len(PERSONA_PROFILES),
        "surface_bindings": len(SURFACE_BINDINGS),
        "scene_resolution_events": 0,
        "interaction_boundary_events": 0,
    }


def test_repository_resolves_seeded_records(tmp_path):
    repository = CompanionContractsRepository(
        db_path=tmp_path / "contracts" / "companion_contracts.sqlite3"
    )

    profile = repository.active_profile()
    scene = repository.scene_policy("coding")
    contract = repository.active_interaction_contract(
        profile_id=profile.profile_id,
        profile_version=profile.version,
    )
    persona = repository.persona_profile("personal_companion")
    surface_binding = repository.surface_binding("vscode")

    assert profile.profile_id == "default_companion_profile"
    assert profile.version == 2
    assert scene is not None
    assert scene.scene_id == "coding_build"
    assert contract.contract_id == "default_interaction_contract"
    assert contract.profile_id == profile.profile_id
    assert persona is not None
    assert persona.persona_id == "personal_companion"
    assert persona.persona_owns_durable_memory is False
    assert surface_binding is not None
    assert surface_binding.default_persona_id == "technical_architect"



def _event_fields():
    return dict(request_id="bounded-receipt", owner_id="owner", conversation_id="conversation",
                surface="web", contract_id="contract", contract_version=1,
                check_type="neutral_receipt", severity="none", input_summary="bounded receipt",
                result="recorded", reason_json={"reason_code": "recorded"}, idempotent=True)


def test_boundary_event_idempotency_is_atomic_across_repository_instances(tmp_path):
    db = tmp_path / "shared.sqlite3"
    first = CompanionContractsRepository(db_path=db)
    second = CompanionContractsRepository(db_path=db)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(repo.record_interaction_boundary_event, **_event_fields())
                   for repo in (first, second)]
        ids = [future.result() for future in futures]
    assert ids[0] == ids[1]
    assert len(first.list_interaction_boundary_events_for_tests()) == 1
    with pytest.raises(ValueError, match="interaction_boundary_event_conflict"):
        second.record_interaction_boundary_event(**{**_event_fields(), "result": "conflicting"})
    assert len(first.list_interaction_boundary_events_for_tests()) == 1


def test_boundary_event_transaction_rolls_back_partial_transition(tmp_path):
    repository = CompanionContractsRepository(db_path=tmp_path / "rollback.sqlite3")
    with pytest.raises(RuntimeError, match="abort"):
        with repository.interaction_boundary_transaction() as connection:
            repository.record_interaction_boundary_event(**_event_fields(), connection=connection)
            raise RuntimeError("abort")
    assert repository.list_interaction_boundary_events_for_tests() == []
