from __future__ import annotations

import re
from typing import Any

from models import (
    InteractionContract,
    InterruptEvaluateRequest,
    InterruptEvaluateResponse,
    RuntimeState,
)
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

    detector_signals = {
        "winning_trigger_signals": winning_signals,
        "all_scores": {
            name: round(float(details["score"]), 4)
            for name, details in scores.items()
        },
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
        reason_json={
            "defer_reasons": defer_reasons,
            "trigger_class": trigger_class,
            "requested_scene": requested_scene,
        },
        contract_constraints_applied=contract_constraints,
        warnings=list(dict.fromkeys(warnings + contract_trace.warnings)),
        debug={
            "detector_signals": detector_signals,
            "advisory_text": advisory_text,
            "user_visible_suppressed": True,
            "degraded": degraded,
        },
    )
