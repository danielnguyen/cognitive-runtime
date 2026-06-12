from __future__ import annotations

import pytest
from services.companion_contracts import reset_companion_contracts_for_tests
from services.runtime_state import clear_states_for_tests
from services.world_state import clear_world_state_for_tests


@pytest.fixture(autouse=True)
def isolated_runtime_storage(tmp_path, monkeypatch):
    contracts_db_path = tmp_path / "contracts" / "companion_contracts.sqlite3"
    runtime_db_path = tmp_path / "runtime" / "runtime_state.sqlite3"
    monkeypatch.setenv("COMPANION_CONTRACTS_DB_PATH", str(contracts_db_path))
    monkeypatch.setenv("COGNITIVE_RUNTIME_DB_PATH", str(runtime_db_path))
    reset_companion_contracts_for_tests(db_path=contracts_db_path)
    clear_states_for_tests(db_path=runtime_db_path)
    clear_world_state_for_tests(db_path=runtime_db_path)
