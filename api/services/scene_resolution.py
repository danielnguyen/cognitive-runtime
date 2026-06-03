from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.companion_contracts import (
    GENERAL_SCENE_ID,
    CompanionContractsRepository,
    ScenePolicyRecord,
)


@dataclass(frozen=True)
class SceneResolutionResult:
    scene: ScenePolicyRecord
    confidence: float
    source: str
    warnings: list[str]
    signals_json: dict[str, Any]
    used_fallback: bool
    used_default_scene: bool


def resolve_scene_policy(
    *,
    repository: CompanionContractsRepository,
    requested_scene: str | None,
    runtime_scene: str | None,
) -> SceneResolutionResult:
    if requested_scene is not None:
        scene = repository.scene_policy(requested_scene)
        if scene is not None:
            return SceneResolutionResult(
                scene=scene,
                confidence=1.0,
                source="requested_scene",
                warnings=[],
                signals_json=_signals_json(
                    requested_scene=requested_scene,
                    runtime_scene=runtime_scene,
                    scene=scene,
                    source="requested_scene",
                ),
                used_fallback=False,
                used_default_scene=scene.scene_id == GENERAL_SCENE_ID,
            )
        default_scene = repository.default_scene_policy()
        return SceneResolutionResult(
            scene=default_scene,
            confidence=0.0,
            source="fallback_general",
            warnings=["unknown_requested_scene"],
            signals_json=_signals_json(
                requested_scene=requested_scene,
                runtime_scene=runtime_scene,
                scene=default_scene,
                source="fallback_general",
            ),
            used_fallback=True,
            used_default_scene=True,
        )

    if runtime_scene:
        scene = repository.scene_policy(runtime_scene)
        if scene is not None:
            return SceneResolutionResult(
                scene=scene,
                confidence=0.8,
                source="runtime_state",
                warnings=[],
                signals_json=_signals_json(
                    requested_scene=requested_scene,
                    runtime_scene=runtime_scene,
                    scene=scene,
                    source="runtime_state",
                ),
                used_fallback=False,
                used_default_scene=scene.scene_id == GENERAL_SCENE_ID,
            )
        default_scene = repository.default_scene_policy()
        return SceneResolutionResult(
            scene=default_scene,
            confidence=0.0,
            source="fallback_general",
            warnings=["unknown_runtime_scene"],
            signals_json=_signals_json(
                requested_scene=requested_scene,
                runtime_scene=runtime_scene,
                scene=default_scene,
                source="fallback_general",
            ),
            used_fallback=True,
            used_default_scene=True,
        )

    default_scene = repository.default_scene_policy()
    return SceneResolutionResult(
        scene=default_scene,
        confidence=0.5,
        source="general",
        warnings=[],
        signals_json=_signals_json(
            requested_scene=requested_scene,
            runtime_scene=runtime_scene,
            scene=default_scene,
            source="general",
        ),
        used_fallback=False,
        used_default_scene=True,
    )


def _signals_json(
    *,
    requested_scene: str | None,
    runtime_scene: str | None,
    scene: ScenePolicyRecord,
    source: str,
) -> dict[str, Any]:
    trigger_value = requested_scene if requested_scene is not None else runtime_scene
    alias_used = bool(
        trigger_value
        and trigger_value != scene.scene_id
        and trigger_value in scene.aliases
    )
    return {
        "requested_scene": requested_scene,
        "runtime_scene": runtime_scene,
        "requested_scene_present": requested_scene is not None,
        "runtime_scene_present": bool(runtime_scene),
        "resolved_scene_id": scene.scene_id,
        "source": source,
        "alias_used": alias_used,
        "matched_alias": trigger_value if alias_used else None,
        "scene_version": scene.version,
        "used_default_scene": scene.scene_id == GENERAL_SCENE_ID,
    }
