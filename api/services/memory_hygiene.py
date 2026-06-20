from __future__ import annotations

from models import (
    MemoryHygieneAggregate,
    MemoryHygieneDecision,
    MemoryHygieneEvaluateRequest,
    MemoryHygieneEvaluateResponse,
    MemoryHygieneFraming,
    MemoryHygieneFreshnessState,
    MemoryHygieneItemInput,
    MemoryHygieneResult,
)
from services.runtime_state import (
    record_runtime_event,
    resolve_runtime_session,
    runtime_session_by_id,
    validate_runtime_turn_session,
)

_BASE_DECISIONS: dict[
    MemoryHygieneFreshnessState,
    tuple[bool, bool, MemoryHygieneFraming, list[str]],
] = {
    "active": (True, True, "current", ["active_current_allowed"]),
    "parked": (True, False, "parked_or_historical", ["parked_not_current"]),
    "stale": (True, False, "stale_or_unverified", ["stale_not_current"]),
    "corrected": (True, True, "corrected_replacement", ["corrected_replacement_allowed"]),
    "superseded": (False, False, "omit", ["superseded_omitted"]),
    "forgotten_or_demoted": (False, False, "omit", ["demoted_omitted"]),
    "unknown_freshness": (
        True,
        False,
        "unknown_or_unverified",
        ["unknown_freshness_not_current"],
    ),
}


def _evaluate_item(
    item: MemoryHygieneItemInput,
    known_ref_ids: set[str],
    superseded_target_ids: set[str],
) -> MemoryHygieneDecision:
    use_allowed, mention_as_current_allowed, framing, reason_codes = _BASE_DECISIONS[
        item.freshness_state
    ]
    reasons = list(reason_codes)

    if item.superseded_by and item.superseded_by in known_ref_ids:
        use_allowed = False
        mention_as_current_allowed = False
        framing = "omit"
        reasons.append("superseded_by_submitted_replacement")

    if item.item_ref.ref_id in superseded_target_ids:
        use_allowed = False
        mention_as_current_allowed = False
        framing = "omit"
        reasons.append("superseded_within_submitted_set")

    return MemoryHygieneDecision(
        item_ref=item.item_ref,
        freshness_state=item.freshness_state,
        use_allowed=use_allowed,
        mention_as_current_allowed=mention_as_current_allowed,
        framing=framing,
        reason_codes=list(dict.fromkeys(reasons))[:8],
    )


def _aggregate(decisions: list[MemoryHygieneDecision]) -> MemoryHygieneAggregate:
    counts_by_freshness_state: dict[MemoryHygieneFreshnessState, int] = {
        "active": 0,
        "parked": 0,
        "stale": 0,
        "superseded": 0,
        "corrected": 0,
        "forgotten_or_demoted": 0,
        "unknown_freshness": 0,
    }
    distinct_reason_codes: list[str] = []
    for decision in decisions:
        counts_by_freshness_state[decision.freshness_state] += 1
        for reason in decision.reason_codes:
            if reason not in distinct_reason_codes:
                distinct_reason_codes.append(reason)

    usable_item_count = sum(1 for decision in decisions if decision.use_allowed)
    current_mention_allowed_count = sum(
        1 for decision in decisions if decision.mention_as_current_allowed
    )
    restricted_or_omitted_count = sum(
        1
        for decision in decisions
        if (not decision.use_allowed) or (not decision.mention_as_current_allowed)
    )

    return MemoryHygieneAggregate(
        evaluated_item_count=len(decisions),
        usable_item_count=usable_item_count,
        current_mention_allowed_count=current_mention_allowed_count,
        restricted_or_omitted_count=restricted_or_omitted_count,
        counts_by_freshness_state=counts_by_freshness_state,
        reason_codes=distinct_reason_codes[:16],
        supersession_handling_applied=any(
            (
                "superseded_by_submitted_replacement" in decision.reason_codes
                or "superseded_within_submitted_set" in decision.reason_codes
            )
            for decision in decisions
        ),
    )


def evaluate_memory_hygiene(
    body: MemoryHygieneEvaluateRequest,
) -> MemoryHygieneEvaluateResponse:
    runtime_session_id = body.runtime_session_id
    if runtime_session_id:
        session = runtime_session_by_id(runtime_session_id)
        if session is None:
            raise ValueError("runtime_session_not_found")
        if (
            session.owner_id != body.owner_id
            or session.conversation_id != body.conversation_id
            or session.surface != body.surface
        ):
            raise ValueError("runtime_session_mismatch")
    else:
        session = resolve_runtime_session(
            request_id=body.request_id,
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            surface=body.surface,
        )
        runtime_session_id = session.runtime_session_id

    if body.runtime_turn_id:
        validate_runtime_turn_session(
            runtime_session_id=runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
        )

    known_ref_ids = {item.item_ref.ref_id for item in body.items}
    superseded_target_ids = {
        item.supersedes for item in body.items if item.supersedes and item.supersedes in known_ref_ids
    }
    decisions = [
        _evaluate_item(
            item,
            known_ref_ids=known_ref_ids,
            superseded_target_ids=superseded_target_ids,
        )
        for item in body.items
    ]
    aggregate = _aggregate(decisions)
    result = MemoryHygieneResult(decisions=decisions, aggregate=aggregate)

    record_runtime_event(
        runtime_session_id=runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        event_type="memory_hygiene_evaluated",
        event_payload_json={
            "request_id": body.request_id,
            "evaluated_item_count": aggregate.evaluated_item_count,
            "usable_item_count": aggregate.usable_item_count,
            "current_mention_allowed_count": aggregate.current_mention_allowed_count,
            "restricted_or_omitted_count": aggregate.restricted_or_omitted_count,
            "counts_by_freshness_state": aggregate.counts_by_freshness_state,
            "reason_codes": aggregate.reason_codes,
            "supersession_handling_applied": aggregate.supersession_handling_applied,
        },
    )

    return MemoryHygieneEvaluateResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        runtime_session_id=runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        result=result,
    )
