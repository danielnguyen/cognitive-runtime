from __future__ import annotations

from hashlib import sha256

from models import CompanionPolicyOverlay, RuntimeState

PROFILE_ID = "companion_profile_r17_mvp"
PROFILE_VERSION = 1
CONTRACT_ID = "interaction_contract_r19_mvp"
CONTRACT_VERSION = 1
GENERAL_SCENE_ID = "general"

_PROFILE_CONTENT = (
    "Companion profile: act as a personal intelligence companion and executive "
    "counterpart. Be grounded, concise, competent, evidence-first, pragmatic, "
    "and willing to challenge when it materially improves usefulness. Prefer "
    "clarity over flourish, stable continuity over novelty, and explicit "
    "uncertainty over performed confidence. Avoid clingy, melodramatic, "
    "sycophantic, theatrically sentient, or tonally erratic behavior."
)

_CONTRACT_CONTENT = (
    "Interaction contract: preserve the user's attention and time. Be candid, "
    "reliable, and useful under disagreement. Do not imply memory that does not "
    "exist, use continuity as performance, pressure continued engagement, or add "
    "relational framing that the task does not call for. When wrong, acknowledge "
    "the miss clearly, correct the substance, avoid apology loops, and restore "
    "forward progress."
)

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
        "abstraction while interrupting speculative loops when value drops."
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


def compile_policy(
    *,
    state: RuntimeState,
    requested_scene: str | None = None,
) -> dict[str, object]:
    scene_id, scene_confidence, scene_source, warnings = _resolve_scene(
        requested_scene=requested_scene,
        runtime_scene=state.active_scene,
    )
    overlays = [
        _overlay(
            overlay_type="interaction_contract",
            scene_id=scene_id,
            content=_CONTRACT_CONTENT,
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
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "scene_id": scene_id,
        "scene_confidence": scene_confidence,
        "scene_source": scene_source,
        "warnings": warnings,
        "runtime_state": state,
        "overlays": overlays,
    }
