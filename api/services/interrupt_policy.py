from __future__ import annotations

import json
import re
from datetime import datetime
from hashlib import sha256
from typing import Annotated, Any, Literal

from models import (
    InteractionContract,
    InterruptEvaluateRequest,
    InterruptEvaluateResponse,
    InterruptExecutionRequest,
    InterruptExecutionResponse,
    InterruptLifecycle,
    InterruptLifecycleDebugResponse,
    InterruptStyle,
    InterruptTriggerClass,
    RuntimeState,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from services.companion_contracts import companion_contracts_repository
from services.companion_policy import resolve_interaction_contract

_STYLE_COMPATIBILITY = {
    "soft_redirect": ["soft_redirect"],
    "crisp_callout": ["candid_challenge", "soft_redirect"],
    "constraint_reset": ["boundary_reminder", "soft_redirect"],
    "next_step_forcing": ["soft_redirect", "candid_challenge"],
    "evidence_anchor": ["candid_challenge", "soft_redirect"],
    "scene_aware_simplification": ["soft_redirect", "boundary_reminder"],
}

_TRIGGER_STYLE_ORDER = {
    "repetitive_branching": ["next_step_forcing", "soft_redirect"],
    "speculative_simulation_with_weak_evidence": ["evidence_anchor", "soft_redirect"],
    "avoidance_disguised_as_analysis": ["crisp_callout", "next_step_forcing", "soft_redirect"],
    "complexity_expansion_beyond_task_value": ["constraint_reset", "scene_aware_simplification"],
    "rising_agitation_with_shrinking_informational_gain": [
        "soft_redirect",
        "scene_aware_simplification",
    ],
    "mismatch_between_context_and_answer_depth": ["scene_aware_simplification", "soft_redirect"],
    "known_recurring_trap_pattern": ["constraint_reset", "soft_redirect"],
}

_EXPLORATION_MARKERS = (
    "brainstorm",
    "explore",
    "think aloud",
    "open ended",
    "possibilities",
    "speculate",
    "hypothesize",
)

_EVIDENCE_MARKERS = (
    "evidence",
    "log",
    "trace",
    "measured",
    "actual",
    "data",
    "stack trace",
    "test result",
    "error output",
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _last_user_text(body: InterruptEvaluateRequest) -> str:
    if body.current_user_text:
        return body.current_user_text.strip()
    for message in reversed(body.recent_messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    return ""


def _count_recent_repetition(messages: list[dict[str, str]], text: str) -> int:
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    total = 0
    for message in messages[-5:-1]:
        if message.get("role") != "user":
            continue
        prior = _normalize_text(message.get("content", ""))
        if prior and prior == normalized:
            total += 1
    return total


def _exploration_requested(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _EXPLORATION_MARKERS)


def _casual_or_low_stakes(surface: str, scene: str | None, text: str) -> bool:
    if surface in {"telegram", "alexa", "car"}:
        return True
    if scene in {"media_co_commentary", "briefing"}:
        return True
    return len(text) < 100


def _build_advisory(trigger_class: str, style: str, scene: str | None) -> str | None:
    scene_hint = ""
    if scene in {"planning", "coding_build", "overload_recovery"}:
        scene_hint = " Keep it to the next concrete step."
    templates = {
        (
            "repetitive_branching",
            "next_step_forcing",
        ): "You are branching again. Pick the next move and test it.",
        (
            "repetitive_branching",
            "soft_redirect",
        ): "This is branching out. Narrow it to the next useful decision.",
        (
            "speculative_simulation_with_weak_evidence",
            "evidence_anchor",
        ): "This is getting speculative. Anchor to the strongest real signal first.",
        (
            "avoidance_disguised_as_analysis",
            "crisp_callout",
        ): "Analysis is now replacing action. Make the next concrete decision.",
        (
            "complexity_expansion_beyond_task_value",
            "constraint_reset",
        ): "The scope is expanding beyond the task value. Reset to the immediate objective.",
        (
            "known_recurring_trap_pattern",
            "constraint_reset",
        ): "Reset to the immediate objective.",
        (
            "mismatch_between_context_and_answer_depth",
            "scene_aware_simplification",
        ): "The context calls for a lighter pass. Keep only what changes the next step.",
    }
    base = templates.get((trigger_class, style))
    if base is None:
        if style == "soft_redirect":
            base = "This is drifting. Return to the highest-value next step."
        elif style == "scene_aware_simplification":
            base = "The context favors a simpler answer. Reduce optional depth."
        else:
            return None
    return f"{base}{scene_hint}"[:240]


def _detector_scores(
    *,
    text: str,
    surface: str,
    scene: str | None,
    interaction_mode: str | None,
    recent_messages: list[dict[str, str]],
    runtime_state: RuntimeState | None,
) -> dict[str, dict[str, Any]]:
    lowered = text.lower()
    words = text.split()
    branch_count = lowered.count(" or ") + lowered.count(" option ") + lowered.count(" maybe ")
    question_count = text.count("?")
    evidence_hits = sum(1 for marker in _EVIDENCE_MARKERS if marker in lowered)
    repetition = _count_recent_repetition(recent_messages, text)
    all_caps_tokens = sum(1 for token in words if len(token) > 3 and token.isupper())
    exclamations = text.count("!")
    abstraction_hits = sum(
        lowered.count(marker)
        for marker in (
            "framework",
            "taxonomy",
            "matrix",
            "comprehensive",
            "exhaustive",
            "every edge case",
        )
    )
    speculative_hits = sum(
        lowered.count(marker)
        for marker in ("what if", "suppose", "imagine", "could", "might", "maybe", "hypothetical")
    )
    avoidance_hits = sum(
        lowered.count(marker)
        for marker in (
            "before doing",
            "not ready to act",
            "keep analyzing",
            "more analysis",
            "every angle",
        )
    )
    trap_hint = 0
    if runtime_state is not None:
        joined_constraints = " ".join(runtime_state.temporary_constraints).lower()
        joined_refs = " ".join(runtime_state.trace_refs).lower()
        if any(
            marker in joined_constraints or marker in joined_refs
            for marker in ("loop", "overthinking", "spiral", "trap")
        ):
            trap_hint = 1

    mismatch_context = int(
        surface in {"car", "alexa", "telegram"}
        or scene in {"driving", "overload_recovery", "media_co_commentary"}
    )
    interaction_constrained = int(interaction_mode in {"actionable", "brief"})
    long_text = int(len(text) > 420)

    return {
        "repetitive_branching": {
            "score": min(1.0, 0.18 * branch_count + 0.12 * question_count + 0.2 * repetition),
            "signals": {
                "branch_count": branch_count,
                "question_count": question_count,
                "repetition_count": repetition,
            },
        },
        "speculative_simulation_with_weak_evidence": {
            "score": min(1.0, 0.14 * speculative_hits + 0.12 * max(0, 2 - evidence_hits)),
            "signals": {
                "speculative_hits": speculative_hits,
                "evidence_hits": evidence_hits,
            },
        },
        "avoidance_disguised_as_analysis": {
            "score": min(1.0, 0.2 * avoidance_hits + 0.1 * branch_count + 0.18 * repetition),
            "signals": {
                "avoidance_hits": avoidance_hits,
                "branch_count": branch_count,
                "repetition_count": repetition,
            },
        },
        "complexity_expansion_beyond_task_value": {
            "score": min(1.0, 0.22 * long_text + 0.15 * abstraction_hits + 0.15 * branch_count),
            "signals": {
                "long_text": bool(long_text),
                "abstraction_hits": abstraction_hits,
                "branch_count": branch_count,
            },
        },
        "rising_agitation_with_shrinking_informational_gain": {
            "score": min(1.0, 0.16 * exclamations + 0.12 * all_caps_tokens + 0.18 * repetition),
            "signals": {
                "exclamations": exclamations,
                "all_caps_tokens": all_caps_tokens,
                "repetition_count": repetition,
            },
        },
        "mismatch_between_context_and_answer_depth": {
            "score": min(
                1.0,
                0.3 * mismatch_context + 0.2 * interaction_constrained + 0.18 * long_text,
            ),
            "signals": {
                "surface_or_scene_constrained": bool(mismatch_context),
                "interaction_mode_constrained": bool(interaction_constrained),
                "long_text": bool(long_text),
            },
        },
        "known_recurring_trap_pattern": {
            "score": min(1.0, 0.45 * trap_hint + 0.18 * repetition),
            "signals": {
                "runtime_trap_hint": bool(trap_hint),
                "repetition_count": repetition,
            },
        },
    }


def _select_style(
    trigger_class: str,
    contract: InteractionContract,
) -> tuple[str | None, dict[str, Any]]:
    allowed = set(contract.allowed_intervention_styles)
    disallowed = set(contract.disallowed_intervention_styles)
    blocked = []
    for style in _TRIGGER_STYLE_ORDER[trigger_class]:
        compatible = _STYLE_COMPATIBILITY[style]
        blocked_by_contract = [name for name in compatible if name in disallowed]
        if blocked_by_contract:
            blocked.append({"style": style, "blocked_by": blocked_by_contract})
            continue
        if allowed and not any(name in allowed for name in compatible):
            blocked.append({"style": style, "missing_allowed_match": compatible})
            continue
        matched = next((name for name in compatible if not allowed or name in allowed), None)
        return style, {
            "allowed_styles": contract.allowed_intervention_styles,
            "disallowed_styles": contract.disallowed_intervention_styles,
            "matched_contract_style": matched,
            "blocked_candidates": blocked,
        }
    return None, {
        "allowed_styles": contract.allowed_intervention_styles,
        "disallowed_styles": contract.disallowed_intervention_styles,
        "matched_contract_style": None,
        "blocked_candidates": blocked,
    }


def evaluate_interrupt_policy(body: InterruptEvaluateRequest) -> InterruptEvaluateResponse:
    runtime_state = body.runtime_state
    warnings: list[str] = []
    degraded = False
    if runtime_state is None:
        from services.runtime_state import resolve_state

        runtime_state = resolve_state(
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            surface=body.surface,
        )

    interaction_contract = body.interaction_contract
    contract_trace = body.contract_trace
    if interaction_contract is None or contract_trace is None:
        warnings.append("default_interaction_contract")
        interaction_contract, contract_trace = resolve_interaction_contract(
            owner_id=body.owner_id,
            surface=body.surface,
            requested_scene=body.requested_scene,
            runtime_state=runtime_state,
        )

    text = _last_user_text(body)
    recent_messages = [message.model_dump() for message in body.recent_messages]
    requested_scene = body.requested_scene or runtime_state.active_scene
    exploration_requested = _exploration_requested(text)
    low_stakes = _casual_or_low_stakes(body.surface, requested_scene, text)
    scores = _detector_scores(
        text=text,
        surface=body.surface,
        scene=requested_scene,
        interaction_mode=runtime_state.interaction_mode,
        recent_messages=recent_messages,
        runtime_state=runtime_state,
    )
    trigger_class = None
    confidence = 0.0
    winning_signals: dict[str, Any] = {}
    for candidate, details in scores.items():
        score = float(details["score"])
        if score > confidence:
            confidence = score
            trigger_class = candidate
            winning_signals = details["signals"]

    if not text:
        warnings.append("missing_user_text")
        degraded = True

    defer_reasons = []
    if exploration_requested and confidence < 0.9:
        defer_reasons.append("explicit_exploration_request")
        confidence = min(confidence, 0.49)
    if low_stakes and confidence < 0.85:
        defer_reasons.append("casual_or_low_stakes_context")
        confidence = min(confidence, 0.44)
    if confidence < 0.72:
        defer_reasons.append("confidence_below_interrupt_threshold")
    if text and len(text) < 40 and confidence < 0.85:
        defer_reasons.append("insufficient_context")

    selected_style = None
    contract_constraints: dict[str, Any] = {}
    if trigger_class is not None:
        selected_style, contract_constraints = _select_style(trigger_class, interaction_contract)
        if selected_style is None:
            defer_reasons.append("no_contract_permitted_style")

    if any(
        "Defer when the user explicitly chooses a harmless path" in rule
        for rule in interaction_contract.defer_conditions
    ):
        if "explicit_exploration_request" in defer_reasons:
            contract_constraints["defer_condition_matched"] = "explicit_harmless_exploration"

    advisory_text = None
    should_interrupt = False
    if trigger_class and selected_style and not defer_reasons and confidence >= 0.72:
        should_interrupt = True
        advisory_text = _build_advisory(trigger_class, selected_style, requested_scene)

    if body.surface not in {"unknown", "dev", "vscode", "web", "telegram", "alexa", "car"}:
        warnings.append("unknown_surface_interrupt_policy")
    if interaction_contract.source == "default_compiled":
        warnings.append("default_contract_source")

    lifecycle = _evaluate_lifecycle(
        body,
        interaction_contract,
        trigger_class,
        selected_style,
        round(confidence, 4),
        advisory_text,
    )
    if lifecycle.candidate_suppressed:
        should_interrupt = False
        advisory_text = None
        warnings.append(
            "interrupt_lifecycle_history_unavailable"
            if lifecycle.state == "history_unavailable"
            else "repeat_interrupt_suppressed"
        )

    detector_signals = {
        "winning_trigger_signals": winning_signals,
        "all_scores": {name: round(float(details["score"]), 4) for name, details in scores.items()},
        "exploration_requested": exploration_requested,
        "casual_or_low_stakes": low_stakes,
        "message_count": len(recent_messages),
    }

    return InterruptEvaluateResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        requested_scene=body.requested_scene,
        runtime_state=runtime_state,
        interaction_contract=interaction_contract,
        contract_trace=contract_trace,
        trigger_class=trigger_class,
        confidence=round(confidence, 4),
        style_selected=selected_style,
        should_interrupt=should_interrupt,
        should_defer=not should_interrupt,
        intervention_text=advisory_text,
        reason_json={
            "defer_reasons": defer_reasons,
            "trigger_class": trigger_class,
            "requested_scene": requested_scene,
        },
        contract_constraints_applied=contract_constraints,
        warnings=list(dict.fromkeys(warnings + contract_trace.warnings)),
        lifecycle=lifecycle,
        debug={
            "detector_signals": detector_signals,
            "advisory_text": advisory_text,
            "user_visible_suppressed": True,
            "degraded": degraded,
        },
    )


# These payloads are persisted evidence, never a copy of policy inputs or prose.
_Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_EVENT_TYPES = ("interrupt_evaluation", "interrupt_execution", "interrupt_recovery")


class _Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["interrupt-lifecycle.v1"]


class _EvaluationEvidence(_Evidence):
    input_digest: _Digest
    trigger_class: InterruptTriggerClass | None
    style_selected: InterruptStyle | None
    confidence: float = Field(ge=0, le=1)
    candidate_digest: _Digest | None
    lifecycle: InterruptLifecycle


class _ExecutionEvidence(_Evidence):
    evaluation_id: int = Field(gt=0)
    trigger_class: InterruptTriggerClass
    style_selected: InterruptStyle
    candidate_digest: _Digest


class _RecoveryEvidence(_Evidence):
    execution_id: int = Field(gt=0)
    trigger_class: InterruptTriggerClass
    repeated_trigger_count: int = Field(ge=0, le=2147483647)
    reason_code: Literal["trigger_not_recurred", "trigger_recurred", "pattern_broken"]


class InterruptHistoryInvalid(ValueError):
    """Stored lifecycle evidence cannot safely be consumed."""


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _scope(row: dict) -> tuple:
    return tuple(row[name] for name in ("request_id", "owner_id", "conversation_id", "surface"))


def _decode_event(row: dict):
    try:
        for name, limit in (
            ("request_id", 120),
            ("owner_id", 120),
            ("conversation_id", 120),
            ("surface", 64),
            ("contract_id", 120),
        ):
            value = row[name]
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise InterruptHistoryInvalid()
        if (
            type(row["id"]) is not int
            or row["id"] <= 0
            or type(row["contract_version"]) is not int
            or row["contract_version"] <= 0
            or row["severity"] != "none"
            or row["input_summary"] != "interrupt lifecycle"
            or not isinstance(row["created_at"], str)
            or len(row["created_at"]) > 40
            or datetime.fromisoformat(row["created_at"]).utcoffset() is None
            or not isinstance(row["reason_json"], str)
            or len(row["reason_json"]) > 4096
        ):
            raise InterruptHistoryInvalid()

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise InterruptHistoryInvalid()
                result[key] = value
            return result

        raw = json.loads(row["reason_json"], object_pairs_hook=unique_object)
        model = {
            "interrupt_evaluation": _EvaluationEvidence,
            "interrupt_execution": _ExecutionEvidence,
            "interrupt_recovery": _RecoveryEvidence,
        }[row["check_type"]]
        payload = model.model_validate(raw)
        if isinstance(payload, _EvaluationEvidence):
            if row["result"] not in {"authorized", "deferred", "repeat_suppressed"}:
                raise InterruptHistoryInvalid()
            if (row["result"] == "repeat_suppressed") != payload.lifecycle.candidate_suppressed:
                raise InterruptHistoryInvalid()
            if row["result"] == "authorized" and (
                payload.trigger_class is None
                or payload.style_selected is None
                or payload.candidate_digest is None
                or payload.confidence < 0.72
            ):
                raise InterruptHistoryInvalid()
            if row["result"] == "deferred" and payload.candidate_digest is not None:
                raise InterruptHistoryInvalid()
        elif isinstance(payload, _ExecutionEvidence):
            if row["result"] != "executed":
                raise InterruptHistoryInvalid()
        elif row["result"] not in {"accepted", "overridden", "recovered"}:
            raise InterruptHistoryInvalid()
        return payload
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise InterruptHistoryInvalid("interrupt_history_invalid") from error


def _projection(active, state: str, count: int = 0) -> InterruptLifecycle:
    return InterruptLifecycle(
        state=state,
        prior_executed_request_id=active["row"]["request_id"],
        prior_trigger=active["payload"].trigger_class,
        repeated_trigger_count=count,
        candidate_suppressed=state in {"overridden", "repeat_suppressed"},
        reason_code={
            "awaiting_feedback": "execution_recorded",
            "accepted": "trigger_not_recurred",
            "overridden": "trigger_recurred",
            "repeat_suppressed": "repeat_trigger_suppressed",
            "recovered": "pattern_broken",
        }[state],
    )


def _transition(active, trigger):
    if active is None or active["state"] in {"accepted", "recovered"}:
        return InterruptLifecycle(), active, None
    same = trigger == active["payload"].trigger_class
    if active["state"] == "awaiting_feedback":
        state, count = ("overridden", 1) if same else ("accepted", 0)
        recovery = state
    elif same:
        state, count, recovery = "repeat_suppressed", active["count"] + 1, None
    else:
        state, count, recovery = "recovered", active["count"], "recovered"
    updated = {**active, "state": state, "count": count}
    return _projection(updated, state, count), updated, recovery


def _history(rows):
    evaluations, executions, outcomes = {}, {}, {}
    active, pending = None, None
    keys = set()
    for row in rows:
        payload = _decode_event(row)
        key = (*_scope(row), row["check_type"])
        if key in keys:
            raise InterruptHistoryInvalid("interrupt_history_duplicate")
        keys.add(key)
        if isinstance(payload, _RecoveryEvidence):
            if pending is not None or active is None or payload.execution_id != active["row"]["id"]:
                raise InterruptHistoryInvalid("interrupt_recovery_unbound")
            pending = row, payload
        elif isinstance(payload, _EvaluationEvidence):
            if payload.lifecycle.state == "history_unavailable":
                # Conservative diagnostics are never a state transition or authority.
                if pending is not None:
                    raise InterruptHistoryInvalid("interrupt_recovery_unbound")
            else:
                expected, updated, recovery = _transition(active, payload.trigger_class)
                if payload.lifecycle != expected or (pending is not None) != (recovery is not None):
                    raise InterruptHistoryInvalid("interrupt_transition_inconsistent")
                if pending is not None:
                    observed, proof = pending
                    if (
                        _scope(observed) != _scope(row)
                        or observed["result"] != recovery
                        or proof.trigger_class != expected.prior_trigger
                        or proof.repeated_trigger_count != expected.repeated_trigger_count
                        or proof.reason_code != expected.reason_code
                        or (observed["contract_id"], observed["contract_version"])
                        != (active["row"]["contract_id"], active["row"]["contract_version"])
                    ):
                        raise InterruptHistoryInvalid("interrupt_recovery_inconsistent")
                    pending = None
                active = updated
                if expected.prior_executed_request_id is not None:
                    outcomes[active["row"]["id"]] = expected
            evaluations[row["id"]] = row, payload
        else:
            evaluation = evaluations.get(payload.evaluation_id)
            if pending is not None or evaluation is None or payload.evaluation_id in executions:
                raise InterruptHistoryInvalid("interrupt_execution_unbound")
            evaluated, proof = evaluation
            if (
                _scope(evaluated) != _scope(row)
                or evaluated["result"] != "authorized"
                or (proof.trigger_class, proof.style_selected, proof.candidate_digest)
                != (payload.trigger_class, payload.style_selected, payload.candidate_digest)
                or (evaluated["contract_id"], evaluated["contract_version"])
                != (row["contract_id"], row["contract_version"])
            ):
                raise InterruptHistoryInvalid("interrupt_execution_inconsistent")
            active = {"row": row, "payload": payload, "state": "awaiting_feedback", "count": 0}
            executions[payload.evaluation_id] = row, payload
            outcomes[row["id"]] = _projection(active, "awaiting_feedback")
    if pending is not None:
        raise InterruptHistoryInvalid("interrupt_recovery_incomplete")
    return evaluations, executions, outcomes, active


def _append(repository, connection, body, contract_id, contract_version, kind, result, payload):
    return repository.record_interaction_boundary_event(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        contract_id=contract_id,
        contract_version=contract_version,
        check_type=kind,
        severity="none",
        input_summary="interrupt lifecycle",
        result=result,
        reason_json=payload.model_dump(mode="json"),
        connection=connection,
        idempotent=True,
    )


def _evaluate_lifecycle(body, contract, trigger, style, confidence, candidate):
    repository = companion_contracts_repository()
    input_digest = _digest(
        json.dumps(body.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    )
    candidate_digest = _digest(candidate) if candidate is not None else None
    with repository.interaction_boundary_transaction() as connection:
        rows = repository.interaction_boundary_history(
            connection=connection,
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            check_types=_EVENT_TYPES,
            check_type_prefix="interrupt_",
        )
        existing = [
            row
            for row in rows
            if row["check_type"] == "interrupt_evaluation"
            and row["request_id"] == body.request_id
            and row["surface"] == body.surface
        ]
        try:
            evaluations, _, _, active = _history(rows)
        except InterruptHistoryInvalid:
            lifecycle = InterruptLifecycle(
                state="history_unavailable",
                candidate_suppressed=True,
                reason_code="lifecycle_history_invalid",
            )
            # Never overwrite or duplicate a corrupt/previous authority record.
            if existing:
                return lifecycle
            recovery = None
        else:
            if existing:
                row, proof = evaluations[existing[0]["id"]]
                if proof.input_digest != input_digest or (
                    proof.trigger_class,
                    proof.style_selected,
                    proof.confidence,
                    proof.candidate_digest,
                ) != (trigger, style, confidence, candidate_digest):
                    raise ValueError("interrupt_evaluation_conflict")
                return proof.lifecycle
            lifecycle, _, recovery = _transition(active, trigger)
            if recovery is not None:
                _append(
                    repository,
                    connection,
                    body,
                    active["row"]["contract_id"],
                    active["row"]["contract_version"],
                    "interrupt_recovery",
                    recovery,
                    _RecoveryEvidence(
                        schema_version="interrupt-lifecycle.v1",
                        execution_id=active["row"]["id"],
                        trigger_class=lifecycle.prior_trigger,
                        repeated_trigger_count=lifecycle.repeated_trigger_count,
                        reason_code=lifecycle.reason_code,
                    ),
                )
        result = (
            "repeat_suppressed"
            if lifecycle.candidate_suppressed
            else ("authorized" if candidate is not None else "deferred")
        )
        _append(
            repository,
            connection,
            body,
            contract.contract_id,
            contract.contract_version,
            "interrupt_evaluation",
            result,
            _EvaluationEvidence(
                schema_version="interrupt-lifecycle.v1",
                input_digest=input_digest,
                trigger_class=trigger,
                style_selected=style,
                confidence=confidence,
                candidate_digest=candidate_digest,
                lifecycle=lifecycle,
            ),
        )
        return lifecycle


def execute_interrupt(body: InterruptExecutionRequest) -> InterruptExecutionResponse:
    repository = companion_contracts_repository()
    with repository.interaction_boundary_transaction() as connection:
        rows = repository.interaction_boundary_history(
            connection=connection,
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            check_types=_EVENT_TYPES,
            check_type_prefix="interrupt_",
        )
        if not any(
            _scope(row) == (body.request_id, body.owner_id, body.conversation_id, body.surface)
            and row["check_type"] == "interrupt_evaluation"
            for row in rows
        ):
            raise LookupError("interrupt_evaluation_not_found")
        evaluations, executions, outcomes, _ = _history(rows)
        found = [
            (row, proof)
            for row, proof in evaluations.values()
            if _scope(row) == (body.request_id, body.owner_id, body.conversation_id, body.surface)
        ]
        if len(found) != 1:
            raise LookupError("interrupt_evaluation_not_found")
        row, proof = found[0]
        digest = _digest(body.intervention_text)
        if row["result"] != "authorized" or (
            proof.trigger_class,
            proof.style_selected,
            proof.candidate_digest,
        ) != (body.trigger_class, body.style_selected, digest):
            raise ValueError("interrupt_execution_conflict")
        replay = row["id"] in executions
        _append(
            repository,
            connection,
            body,
            row["contract_id"],
            row["contract_version"],
            "interrupt_execution",
            "executed",
            _ExecutionEvidence(
                schema_version="interrupt-lifecycle.v1",
                evaluation_id=row["id"],
                trigger_class=body.trigger_class,
                style_selected=body.style_selected,
                candidate_digest=digest,
            ),
        )
    active = {"row": row, "payload": proof}
    lifecycle = (
        outcomes[executions[row["id"]][0]["id"]]
        if replay
        else (_projection(active, "awaiting_feedback"))
    )
    return InterruptExecutionResponse(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        execution_recorded=True,
        idempotent_replay=replay,
        lifecycle=lifecycle,
    )


def interrupt_debug(*, request_id: str, owner_id: str, conversation_id: str):
    repository = companion_contracts_repository()
    with repository.interaction_boundary_transaction() as connection:
        rows = repository.interaction_boundary_history(
            connection=connection,
            owner_id=owner_id,
            conversation_id=conversation_id,
            check_types=_EVENT_TYPES,
            check_type_prefix="interrupt_",
        )
        # Check binding before interpreting any private lifecycle history.
        if not any(
            row["request_id"] == request_id and row["check_type"] == "interrupt_evaluation"
            for row in rows
        ):
            raise LookupError("interrupt_request_not_found")
        evaluations, executions, outcomes, _ = _history(rows)
        found = [
            (row, proof) for row, proof in evaluations.values() if row["request_id"] == request_id
        ]
        if len(found) != 1:
            raise LookupError("interrupt_request_not_found")
        row, proof = found[0]
        execution = executions.get(row["id"])
        lifecycle = outcomes[execution[0]["id"]] if execution else proof.lifecycle
    return InterruptLifecycleDebugResponse(
        request_id=request_id,
        owner_id=owner_id,
        conversation_id=conversation_id,
        surface=row["surface"],
        trigger_class=proof.trigger_class,
        style_selected=proof.style_selected,
        evaluation_result=row["result"],
        execution_state="executed" if execution else "not_executed",
        recovery_outcome=lifecycle.state
        if lifecycle.state in {"accepted", "overridden", "recovered"}
        else ("overridden" if lifecycle.state == "repeat_suppressed" else None),
        lifecycle=lifecycle,
    )
