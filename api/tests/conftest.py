from __future__ import annotations

import anyio
import httpx
import pytest
import fastapi.testclient as fastapi_testclient
import starlette.testclient as starlette_testclient
from services.companion_contracts import reset_companion_contracts_for_tests
from services.human_compatibility import clear_human_compatibility_for_tests
from services.relationships import clear_relationships_for_tests
from services.runtime_state import clear_states_for_tests
from services.social_context import clear_social_context_for_tests
from services.world_state import clear_world_state_for_tests


class _CompatClient:
    __test__ = False

    def __init__(self, app, base_url: str = "http://testserver", **_: object) -> None:
        self.app = app
        self.base_url = base_url

    async def _request_async(
        self,
        method: str,
        path: str,
        **kwargs: object,
    ) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url=self.base_url,
        ) as client:
            return await client.request(method, path, **kwargs)

    def request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        async def _runner() -> httpx.Response:
            return await self._request_async(method, path, **kwargs)

        return anyio.run(_runner)

    def get(self, path: str, **kwargs: object) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: object) -> httpx.Response:
        return self.request("POST", path, **kwargs)


fastapi_testclient.TestClient = _CompatClient
starlette_testclient.TestClient = _CompatClient


@pytest.fixture(autouse=True)
def isolated_runtime_storage(tmp_path, monkeypatch):
    contracts_db_path = tmp_path / "contracts" / "companion_contracts.sqlite3"
    runtime_db_path = tmp_path / "runtime" / "runtime_state.sqlite3"
    monkeypatch.setenv("COMPANION_CONTRACTS_DB_PATH", str(contracts_db_path))
    monkeypatch.setenv("COGNITIVE_RUNTIME_DB_PATH", str(runtime_db_path))
    reset_companion_contracts_for_tests(db_path=contracts_db_path)
    clear_states_for_tests(db_path=runtime_db_path)
    clear_world_state_for_tests(db_path=runtime_db_path)
    clear_relationships_for_tests(db_path=runtime_db_path)
    clear_social_context_for_tests(db_path=runtime_db_path)
    clear_human_compatibility_for_tests(db_path=runtime_db_path)
