from __future__ import annotations

import pytest
from services.companion_contracts import reset_companion_contracts_for_tests
from services.runtime_state import clear_states_for_tests


@pytest.fixture(autouse=True)
def isolated_runtime_storage(tmp_path, monkeypatch):
    db_path = tmp_path / "contracts" / "companion_contracts.sqlite3"
    monkeypatch.setenv("COMPANION_CONTRACTS_DB_PATH", str(db_path))
    reset_companion_contracts_for_tests(db_path=db_path)
    clear_states_for_tests()
