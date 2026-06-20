from __future__ import annotations

import httpx
import pytest

from main import app


def _base(**overrides):
    payload = {
        "request_id": "rid-privacy-context",
        "owner_id": "owner",
        "conversation_id": "conv-1",
        "surface": "dev",
        "surface_category": "desktop_private",
        "sensitivity_level": "normal",
        "sensitivity_domains": [],
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
async def test_desktop_private_allows_normal_detail():
    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(surface_category="desktop_private", sensitivity_level="normal"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["privacy_zone"] == "private"
    assert result["sensitive_detail_allowed"] is True
    assert result["screen_detail_allowed"] is True
    assert result["redaction_required"] is False


@pytest.mark.asyncio
async def test_mobile_private_allows_normal_detail():
    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(surface_category="mobile_private", sensitivity_level="normal"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["privacy_zone"] == "private"
    assert result["screen_detail_allowed"] is True
    assert result["redaction_required"] is False


@pytest.mark.asyncio
async def test_telegram_private_allows_normal_detail():
    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(surface_category="telegram_private", sensitivity_level="normal"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["privacy_zone"] == "private"
    assert result["sensitive_detail_allowed"] is True
    assert result["redaction_required"] is False


@pytest.mark.asyncio
async def test_voice_private_differs_from_car_voice_possible_passenger():
    private_response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(surface_category="voice_private", sensitivity_level="sensitive"),
    )
    car_response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(
            surface_category="car_voice_possible_passenger",
            sensitivity_level="sensitive",
            sensitivity_domains=["personal"],
        ),
    )

    assert private_response.status_code == 200
    assert car_response.status_code == 200
    assert private_response.json()["result"]["voice_detail_allowed"] is True
    assert car_response.json()["result"]["voice_detail_allowed"] is False


@pytest.mark.asyncio
async def test_car_voice_suppresses_sensitive_personal_detail():
    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(
            surface_category="car_voice_possible_passenger",
            sensitivity_level="sensitive",
            sensitivity_domains=["personal"],
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["sensitive_detail_allowed"] is False
    assert result["voice_detail_allowed"] is False
    assert result["redaction_required"] is True
    assert result["safe_summary_required"] is True


@pytest.mark.asyncio
async def test_car_voice_suppresses_sensitive_health_detail():
    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(
            surface_category="car_voice_possible_passenger",
            sensitivity_level="sensitive",
            sensitivity_domains=["health"],
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["voice_detail_allowed"] is False
    assert result["redaction_required"] is True


@pytest.mark.asyncio
async def test_car_voice_suppresses_sensitive_financial_detail():
    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(
            surface_category="car_voice_possible_passenger",
            sensitivity_level="sensitive",
            sensitivity_domains=["financial"],
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["voice_detail_allowed"] is False
    assert result["redaction_required"] is True


@pytest.mark.asyncio
async def test_car_voice_suppresses_sensitive_work_detail():
    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(
            surface_category="car_voice_possible_passenger",
            sensitivity_level="sensitive",
            sensitivity_domains=["work"],
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["voice_detail_allowed"] is False
    assert result["redaction_required"] is True


@pytest.mark.asyncio
async def test_notification_preview_suppresses_sensitive_detail():
    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(
            surface_category="notification_preview",
            sensitivity_level="sensitive",
            sensitivity_domains=["personal"],
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["notification_detail_allowed"] is False
    assert result["safe_summary_required"] is True


@pytest.mark.asyncio
async def test_glasses_public_requires_redaction_or_safe_summary():
    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(
            surface_category="glasses_public_or_semi_public",
            sensitivity_level="sensitive",
            sensitivity_domains=["work"],
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["screen_detail_allowed"] is False
    assert result["redaction_required"] is True
    assert result["safe_summary_required"] is True


@pytest.mark.asyncio
async def test_unknown_surface_defaults_conservatively():
    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(surface="mystery-surface", surface_category=None, sensitivity_level="normal"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["surface_type"] == "unknown_surface"
    assert result["sensitive_detail_allowed"] is False
    assert result["notification_detail_allowed"] is False
    assert result["voice_detail_allowed"] is False
    assert result["redaction_required"] is True
    assert result["safe_summary_required"] is True


@pytest.mark.asyncio
async def test_non_sensitive_content_is_not_unnecessarily_suppressed():
    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(
            surface_category="glasses_public_or_semi_public",
            sensitivity_level="normal",
        ),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["screen_detail_allowed"] is True
    assert result["redaction_required"] is False
    assert result["safe_summary_required"] is False


@pytest.mark.asyncio
async def test_missing_or_invalid_sensitivity_does_not_become_normal():
    missing_response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(surface_category="desktop_private", sensitivity_level=None),
    )
    invalid_response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(surface_category="desktop_private", sensitivity_level="not-real"),
    )

    assert missing_response.status_code == 200
    assert invalid_response.status_code == 200
    assert missing_response.json()["result"]["sensitivity_level"] == "unknown"
    assert missing_response.json()["result"]["sensitive_detail_allowed"] is False
    assert invalid_response.json()["result"]["sensitivity_level"] == "unknown"
    assert invalid_response.json()["result"]["sensitive_detail_allowed"] is False


@pytest.mark.asyncio
async def test_missing_or_invalid_surface_does_not_become_private():
    missing_response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(surface="non-r57-surface", surface_category=None),
    )
    invalid_response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(surface="non-r57-surface", surface_category="not-real"),
    )

    assert missing_response.status_code == 200
    assert invalid_response.status_code == 200
    assert missing_response.json()["result"]["surface_type"] == "unknown_surface"
    assert invalid_response.json()["result"]["surface_type"] == "unknown_surface"


@pytest.mark.asyncio
async def test_conflicting_surface_context_fails_conservatively():
    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(
            surface="desktop_private",
            surface_category="car_voice_possible_passenger",
            sensitivity_level="sensitive",
        ),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "surface_context_mismatch"


@pytest.mark.asyncio
async def test_runtime_event_contains_bounded_policy_outcomes_without_content():
    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(
            surface_category="notification_preview",
            sensitivity_level="sensitive",
            sensitivity_domains=["personal"],
        ),
    )

    assert response.status_code == 200
    runtime_session_id = response.json()["runtime_session_id"]
    diagnostics = await _get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    event = next(
        item
        for item in diagnostics.json()["events"]
        if item["event_type"] == "privacy_context_evaluated"
    )
    payload = event["event_payload_json"]
    assert set(payload.keys()) == {
        "request_id",
        "surface_type",
        "privacy_zone",
        "sensitivity_level",
        "sensitive_detail_allowed",
        "notification_detail_allowed",
        "voice_detail_allowed",
        "screen_detail_allowed",
        "redaction_required",
        "safe_summary_required",
        "reason_codes",
    }
    assert "items" not in str(payload)
    assert "personal" not in str(payload)
    assert "surface_category" not in str(payload)


@pytest.mark.asyncio
async def test_runtime_session_validation_matches_existing_policy_behavior():
    started = await _post(
        "/v1/runtime/sessions/resolve",
        {
            "request_id": "rid-privacy-session",
            "owner_id": "owner",
            "conversation_id": "conv-1",
            "surface": "dev",
        },
    )
    assert started.status_code == 200
    runtime_session_id = started.json()["runtime_session"]["runtime_session_id"]

    response = await _post(
        "/v1/runtime/privacy-context/evaluate",
        _base(
            runtime_session_id=runtime_session_id,
            conversation_id="different-conversation",
        ),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "runtime_session_mismatch"


@pytest.mark.asyncio
async def test_runtime_turn_validation_matches_existing_policy_behavior():
    started = await _post(
        "/v1/runtime/turns/start",
        {
            "request_id": "rid-privacy-turn-start",
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
        "/v1/runtime/privacy-context/evaluate",
        _base(
            runtime_session_id=runtime_session_id,
            runtime_turn_id=runtime_turn_id,
        ),
    )

    assert response.status_code == 200
    diagnostics = await _get(f"/v1/runtime/sessions/{runtime_session_id}")
    assert diagnostics.status_code == 200
    event = next(
        item
        for item in diagnostics.json()["events"]
        if item["event_type"] == "privacy_context_evaluated"
    )
    assert event["runtime_turn_id"] == runtime_turn_id
