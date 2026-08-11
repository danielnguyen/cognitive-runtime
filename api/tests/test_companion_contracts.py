from __future__ import annotations

from pathlib import Path

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
