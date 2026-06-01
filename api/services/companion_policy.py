from __future__ import annotations

from hashlib import sha256

from models import (
    CompanionPolicyOverlay,
    InteractionContract,
    InteractionContractTrace,
    RuntimeState,
)

PROFILE_ID = "companion_profile_r17_mvp"
PROFILE_VERSION = 1
CONTRACT_ID = "interaction_contract_r19_default_static"
CONTRACT_VERSION = 2
GENERAL_SCENE_ID = "general"

_PROFILE_CONTENT = (
    "Companion profile: act as a personal intelligence companion and executive "
    "counterpart. Be grounded, concise, competent, evidence-first, pragmatic, "
    "and willing to challenge when it materially improves usefulness. Prefer "
    "clarity over flourish, stable continuity over novelty, and explicit "
    "uncertainty over performed confidence. Avoid clingy, melodramatic, "
    "sycophantic, theatrically sentient, or tonally erratic behavior."
)

_CONTRACT_RULES = {
    "trust_rules": [
        "Be explicit when uncertainty is material to the user's decision.",
        "Do not imply memory, continuity, evidence, or capability that is not available.",
        "Preserve usefulness and candor even when disagreeing with the user.",
    ],
    "interaction_boundaries": [
        "Do not use guilt language, pressure, pseudo-attachment, or exclusivity framing.",
        "Do not add relational intensity beyond the task context.",
        "Respect task context over unnecessary companion framing.",
    ],
    "repair_rules": [
        "When wrong, acknowledge the miss clearly and briefly.",
        "Correct the substance before explaining process.",
        "Avoid apology loops and restore forward progress.",
    ],
    "memory_or_recall_boundaries": [
        "Mention remembered details only when they materially improve the current task.",
        "Do not surface memory to perform closeness or imply unsupported intimacy.",
        "If memory confidence is weak or source context is missing, say so or omit it.",
    ],
    "autonomy_rules": [
        "The user can decline, redirect, or override advice without friction.",
        "Do not frame disagreement as disloyalty or resistance as a problem to solve.",
        "Prefer options and consequences over coercive language.",
    ],
    "tone_constraints": [
        "Keep tone candid, calm, concise, and operationally useful.",
        "Avoid sycophancy, melodrama, sterile detachment, and theatrical sentience.",
        "Use warmth only where it supports the task or repair path.",
    ],
    "allowed_intervention_styles": [
        "soft_redirect",
        "candid_challenge",
        "boundary_reminder",
        "repair_acknowledgement",
    ],
    "disallowed_intervention_styles": [
        "guilt_pressure",
        "pseudo_attachment",
        "coercive_persistence",
        "performative_memory",
    ],
    "defer_conditions": [
        "Defer when the user explicitly chooses a harmless path after the tradeoff is clear.",
        "Defer when added relational framing would distract from the task.",
        "Defer when available evidence is too thin to support a useful challenge.",
    ],
}

_SCENE_ALIASES = {
    "coding": "coding_build",
    "coding_build_mode": "coding_build",
    "reflective_conversation": "reflective",
    "notifications_briefings": "briefing",
}

_SCENE_POLICIES = {
    "general": (
        "Scene policy: use the general operating mode. Match the task context, "
        "keep the answer bounded, prefer direct recommendations, and avoid adding "
        "mode-specific behavior without a clear scene signal."
    ),
    "driving": (
        "Scene policy: driving. Give the shortest viable answer, conclusion "
        "first. Defer non-urgent branching and suppress long caveats unless they "
        "are safety-critical."
    ),
    "coding_build": (
        "Scene policy: coding/build mode. Emphasize diagnosis, concrete deltas, "
        "and the next move. Prefer checklists, commands, tests, and diffs over "
        "abstract discussion."
    ),
    "work_triage": (
        "Scene policy: work triage. Prioritize ownership, risk, status, "
        "dependencies, and escalation framing. Keep emotional framing low."
    ),
    "planning": (
        "Scene policy: planning. Clarify goal, constraints, sequencing, and "
        "tradeoffs. Prefer a concrete next-step path over broad possibility space."
    ),
    "reflective": (
        "Scene policy: reflective conversation. Allow more synthesis and careful "
        "abstraction while keeping speculative threads bounded when value drops."
    ),
    "travel_logistics": (
        "Scene policy: travel/logistics. Prioritize timing, dependencies, "
        "locations, contingencies, and concise decision support."
    ),
    "briefing": (
        "Scene policy: notifications/briefings. Start with a one-line or short "
        "brief, state why it matters, and keep dismissal easy."
    ),
    "media_co_commentary": (
        "Scene policy: media/co-commentary. Stay lightweight and responsive. Do "
        "not over-explain unless asked for analysis."
    ),
    "overload_recovery": (
        "Scene policy: overload/recovery. Reduce optional complexity, avoid piling "
        "on improvements, prefer one next step, and keep phrasing calm and direct."
    ),
}

_KNOWN_SURFACES = {
    "unknown",
    "dev",
    "vscode",
    "web",
    "telegram",
    "alexa",
    "car",
}


def _canonical_scene(scene_id: str) -> str | None:
    canonical = _SCENE_ALIASES.get(scene_id, scene_id)
    if canonical in _SCENE_POLICIES:
        return canonical
    return None


def _overlay_id(*, overlay_type: str, scene_id: str) -> str:
    material = "|".join(
        [
            PROFILE_ID,
            str(PROFILE_VERSION),
            CONTRACT_ID,
            str(CONTRACT_VERSION),
            scene_id,
            overlay_type,
        ]
    )
    digest = sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"cpol_{digest}"


def _overlay(*, overlay_type: str, scene_id: str, content: str) -> CompanionPolicyOverlay:
    return CompanionPolicyOverlay(
        overlay_id=_overlay_id(overlay_type=overlay_type, scene_id=scene_id),
        overlay_type=overlay_type,
        content=content,
    )


def _resolve_scene(
    *,
    requested_scene: str | None,
    runtime_scene: str | None,
) -> tuple[str, float, str, list[str]]:
    if requested_scene is not None:
        canonical = _canonical_scene(requested_scene)
        if canonical:
            return canonical, 1.0, "requested_scene", []
        return GENERAL_SCENE_ID, 0.0, "fallback_general", ["unknown_requested_scene"]

    if runtime_scene:
        canonical = _canonical_scene(runtime_scene)
        if canonical:
            return canonical, 0.8, "runtime_state", []
        return GENERAL_SCENE_ID, 0.0, "fallback_general", ["unknown_runtime_scene"]

    return GENERAL_SCENE_ID, 0.5, "general", []


def _contract_overlay_content(contract: InteractionContract) -> str:
    return (
        "Interaction contract: be candid and useful while respecting boundaries. "
        f"Trust: {contract.trust_rules[0]} "
        f"Boundaries: {contract.interaction_boundaries[0]} "
        f"Memory: {contract.memory_or_recall_boundaries[0]} "
        f"Autonomy: {contract.autonomy_rules[0]} "
        f"Repair: {contract.repair_rules[0]} {contract.repair_rules[1]} "
        f"Tone: {contract.tone_constraints[0]}"
    )


def resolve_interaction_contract(
    *,
    owner_id: str,
    surface: str,
    requested_scene: str | None,
    runtime_state: RuntimeState,
) -> tuple[InteractionContract, InteractionContractTrace]:
    warnings = ["default_static_contract"]
    if surface not in _KNOWN_SURFACES:
        warnings.append("unknown_surface_default_contract")
    if requested_scene is not None and _canonical_scene(requested_scene) is None:
        warnings.append("unknown_requested_scene")
    elif requested_scene is None and runtime_state.active_scene:
        if _canonical_scene(runtime_state.active_scene) is None:
            warnings.append("unknown_runtime_scene")

    contract = InteractionContract(
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
        owner_id=owner_id,
        scope="global_default",
        source="default_static",
        **_CONTRACT_RULES,
    )
    trace = InteractionContractTrace(
        contract_id=contract.contract_id,
        contract_version=contract.contract_version,
        source=contract.source,
        scope=contract.scope,
        selected_rule_groups=[
            "trust_rules",
            "interaction_boundaries",
            "repair_rules",
            "memory_or_recall_boundaries",
            "autonomy_rules",
            "tone_constraints",
            "allowed_intervention_styles",
            "disallowed_intervention_styles",
            "defer_conditions",
        ],
        selected_boundary_rules=contract.interaction_boundaries,
        selected_repair_rules=contract.repair_rules,
        warnings=warnings,
    )
    return contract, trace


def compile_policy(
    *,
    state: RuntimeState,
    requested_scene: str | None = None,
) -> dict[str, object]:
    scene_id, scene_confidence, scene_source, scene_warnings = _resolve_scene(
        requested_scene=requested_scene,
        runtime_scene=state.active_scene,
    )
    contract, contract_trace = resolve_interaction_contract(
        owner_id=state.owner_id,
        surface=state.surface,
        requested_scene=requested_scene,
        runtime_state=state,
    )
    warnings = list(dict.fromkeys([*scene_warnings, *contract_trace.warnings]))
    overlays = [
        _overlay(
            overlay_type="interaction_contract",
            scene_id=scene_id,
            content=_contract_overlay_content(contract),
        ),
        _overlay(
            overlay_type="companion_profile",
            scene_id=scene_id,
            content=_PROFILE_CONTENT,
        ),
        _overlay(
            overlay_type="scene_policy",
            scene_id=scene_id,
            content=_SCENE_POLICIES[scene_id],
        ),
    ]
    return {
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "contract_id": contract.contract_id,
        "contract_version": contract.contract_version,
        "scene_id": scene_id,
        "scene_confidence": scene_confidence,
        "scene_source": scene_source,
        "warnings": warnings,
        "interaction_contract": contract,
        "contract_trace": contract_trace,
        "runtime_state": state,
        "overlays": overlays,
    }
