from __future__ import annotations

import httpx
import pytest

from main import app


def _base(**overrides):
    payload = {
        "request_id": "rid-memory-hygiene",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
        "items": [],
    }
    payload.update(overrides)
    return payload


def _item(ref_id: str, freshness_state: str | None, **overrides):
    payload = {
        "item_ref": {
            "ref_type": "message",
            "ref_id": ref_id,
        },
        "freshness_state": freshness_state,
    }
    payload.update(overrides)
    return payload


async def _post(path: str, payload: dict[str, object]):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post(path, json=payload)


async def _get(path: str):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_active_is_usable_and_current():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(items=[_item("msg-1", "active")]),
    )

    assert response.status_code == 200
    decision = response.json()["result"]["decisions"][0]
    assert decision["use_allowed"] is True
    assert decision["mention_as_current_allowed"] is True
    assert decision["framing"] == "current"


@pytest.mark.asyncio
async def test_parked_is_usable_but_not_current():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(items=[_item("msg-1", "parked")]),
    )

    assert response.status_code == 200
    decision = response.json()["result"]["decisions"][0]
    assert decision["use_allowed"] is True
    assert decision["mention_as_current_allowed"] is False
    assert decision["framing"] == "parked_or_historical"


@pytest.mark.asyncio
async def test_stale_is_usable_only_with_stale_framing():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(items=[_item("msg-1", "stale")]),
    )

    assert response.status_code == 200
    decision = response.json()["result"]["decisions"][0]
    assert decision["use_allowed"] is True
    assert decision["mention_as_current_allowed"] is False
    assert decision["framing"] == "stale_or_unverified"


@pytest.mark.asyncio
async def test_corrected_is_usable_as_replacement():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(items=[_item("msg-2", "corrected", memory_id="memory-2", supersedes="memory-1")]),
    )

    assert response.status_code == 200
    decision = response.json()["result"]["decisions"][0]
    assert decision["use_allowed"] is True
    assert decision["mention_as_current_allowed"] is True
    assert decision["framing"] == "corrected_replacement"


@pytest.mark.asyncio
async def test_superseded_is_omitted():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(items=[_item("msg-1", "superseded")]),
    )

    assert response.status_code == 200
    decision = response.json()["result"]["decisions"][0]
    assert decision["use_allowed"] is False
    assert decision["mention_as_current_allowed"] is False
    assert decision["framing"] == "omit"


@pytest.mark.asyncio
async def test_forgotten_or_demoted_is_omitted():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(items=[_item("msg-1", "forgotten_or_demoted")]),
    )

    assert response.status_code == 200
    decision = response.json()["result"]["decisions"][0]
    assert decision["use_allowed"] is False
    assert decision["mention_as_current_allowed"] is False
    assert decision["framing"] == "omit"


@pytest.mark.asyncio
async def test_unknown_freshness_is_conservative_and_not_current():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(items=[_item("msg-1", "unknown_freshness")]),
    )

    assert response.status_code == 200
    decision = response.json()["result"]["decisions"][0]
    assert decision["use_allowed"] is True
    assert decision["mention_as_current_allowed"] is False
    assert decision["framing"] == "unknown_or_unverified"


@pytest.mark.asyncio
async def test_missing_or_invalid_freshness_never_becomes_current():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(
            items=[
                _item("msg-1", None),
                _item("msg-2", "not-a-real-state"),
            ]
        ),
    )

    assert response.status_code == 200
    decisions = response.json()["result"]["decisions"]
    for decision in decisions:
        assert decision["freshness_state"] == "unknown_freshness"
        assert decision["mention_as_current_allowed"] is False


@pytest.mark.asyncio
async def test_replacement_suppresses_using_memory_id_when_source_ids_differ():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(
            items=[
                _item("older-source-message", "active", memory_id="older-memory-item-id"),
                _item(
                    "replacement-source-message",
                    "corrected",
                    memory_id="memory-item-id",
                    supersedes="older-memory-item-id",
                ),
            ]
        ),
    )

    assert response.status_code == 200
    decisions = {
        item["item_ref"]["ref_id"]: item for item in response.json()["result"]["decisions"]
    }
    assert decisions["older-source-message"]["use_allowed"] is False
    assert decisions["older-source-message"]["framing"] == "omit"
    assert decisions["replacement-source-message"]["use_allowed"] is True
    assert decisions["replacement-source-message"]["mention_as_current_allowed"] is True


@pytest.mark.asyncio
async def test_superseded_by_suppresses_using_memory_id_when_source_ids_differ():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(
            items=[
                _item(
                    "older-source-message",
                    "corrected",
                    memory_id="older-memory-item-id",
                    superseded_by="newer-memory-item-id",
                ),
                _item(
                    "newer-source-message",
                    "active",
                    memory_id="newer-memory-item-id",
                ),
            ]
        ),
    )

    assert response.status_code == 200
    decisions = {
        item["item_ref"]["ref_id"]: item for item in response.json()["result"]["decisions"]
    }
    assert decisions["older-source-message"]["use_allowed"] is False
    assert decisions["older-source-message"]["mention_as_current_allowed"] is False
    assert decisions["newer-source-message"]["mention_as_current_allowed"] is True


@pytest.mark.asyncio
async def test_same_ref_id_in_different_namespaces_does_not_cross_suppress():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(
            items=[
                _item("shared-ref", "active", memory_id="message-memory-id"),
                _item(
                    "shared-ref",
                    "corrected",
                    memory_id="derived-memory-id",
                    supersedes="different-unsubmitted-memory-id",
                    item_ref={"ref_type": "derived_text", "ref_id": "shared-ref"},
                ),
            ]
        ),
    )

    assert response.status_code == 200
    decisions = response.json()["result"]["decisions"]
    by_type = {item["item_ref"]["ref_type"]: item for item in decisions}
    assert by_type["message"]["use_allowed"] is True
    assert by_type["derived_text"]["use_allowed"] is True


@pytest.mark.asyncio
async def test_missing_memory_id_does_not_crash():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(
            items=[
                _item("msg-1", "active"),
                _item("msg-2", "corrected", supersedes="older-memory-item-id"),
            ]
        ),
    )

    assert response.status_code == 200
    decisions = {
        item["item_ref"]["ref_id"]: item for item in response.json()["result"]["decisions"]
    }
    assert decisions["msg-1"]["use_allowed"] is True
    assert decisions["msg-2"]["use_allowed"] is True


@pytest.mark.asyncio
async def test_dangling_memory_relationship_ids_do_not_suppress_unrelated_items():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(
            items=[
                _item("msg-1", "active", memory_id="active-memory-id", superseded_by="missing-memory-id"),
                _item(
                    "msg-2",
                    "corrected",
                    memory_id="corrected-memory-id",
                    supersedes="other-missing-memory-id",
                ),
            ]
        ),
    )

    assert response.status_code == 200
    decisions = {
        item["item_ref"]["ref_id"]: item for item in response.json()["result"]["decisions"]
    }
    assert decisions["msg-1"]["use_allowed"] is True
    assert decisions["msg-1"]["mention_as_current_allowed"] is True
    assert decisions["msg-2"]["use_allowed"] is True
    assert decisions["msg-2"]["mention_as_current_allowed"] is True


@pytest.mark.asyncio
async def test_runtime_event_payload_is_aggregate_only_without_item_identities_or_content():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(
            items=[
                _item("msg-1", "active", memory_id="memory-1"),
                _item("msg-2", "stale", memory_id="memory-2", source_kind="message", confidence=0.8),
            ]
        ),
    )

    assert response.status_code == 200
    runtime_session_id = response.json()["runtime_session_id"]

    diagnostics = await _get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    event = next(
        item
        for item in diagnostics.json()["events"]
        if item["event_type"] == "memory_hygiene_evaluated"
    )
    payload = event["event_payload_json"]
    assert set(payload.keys()) == {
        "request_id",
        "evaluated_item_count",
        "usable_item_count",
        "current_mention_allowed_count",
        "restricted_or_omitted_count",
        "counts_by_freshness_state",
        "reason_codes",
        "supersession_handling_applied",
    }
    assert "memory-1" not in str(payload)
    assert "memory-2" not in str(payload)
    assert "msg-1" not in str(payload)
    assert "msg-2" not in str(payload)
    assert "source_kind" not in str(payload)
    assert "content" not in str(payload)


@pytest.mark.asyncio
async def test_endpoint_works_without_auth_headers_following_current_behavior():
    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(items=[_item("msg-1", "active")]),
    )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_request_id_handling_matches_current_validation_behavior():
    payload = _base(items=[_item("msg-1", "active")])
    payload.pop("request_id")

    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        payload,
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_runtime_turn_is_validated_against_session_and_event_attaches_to_turn():
    started = await _post(
        "/v1/runtime/turns/start",
        {
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

    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(
            request_id="rid-turn-memory-hygiene",
            runtime_session_id=runtime_session_id,
            runtime_turn_id=runtime_turn_id,
            items=[_item("msg-1", "active")],
        ),
    )

    assert response.status_code == 200
    diagnostics = await _get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    event = next(
        item
        for item in diagnostics.json()["events"]
        if item["event_type"] == "memory_hygiene_evaluated"
    )
    assert event["runtime_turn_id"] == runtime_turn_id


@pytest.mark.asyncio
async def test_mismatched_runtime_turn_returns_not_found_style_error():
    started_a = await _post(
        "/v1/runtime/turns/start",
        {
            "request_id": "rid-turn-start-a",
            "owner_id": "owner",
            "conversation_id": "conv-1",
            "surface": "dev",
            "input_message_id": "msg-1",
        },
    )
    started_b = await _post(
        "/v1/runtime/turns/start",
        {
            "request_id": "rid-turn-start-b",
            "owner_id": "owner",
            "conversation_id": "conv-2",
            "surface": "dev",
            "input_message_id": "msg-2",
        },
    )
    assert started_a.status_code == 200
    assert started_b.status_code == 200

    runtime_session_id = started_a.json()["runtime_session"]["runtime_session_id"]
    runtime_turn_id = started_b.json()["runtime_turn"]["runtime_turn_id"]

    response = await _post(
        "/v1/runtime/memory-hygiene/evaluate",
        _base(
            conversation_id="conv-1",
            runtime_session_id=runtime_session_id,
            runtime_turn_id=runtime_turn_id,
            items=[_item("msg-1", "active")],
        ),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "runtime_turn_session_mismatch"
