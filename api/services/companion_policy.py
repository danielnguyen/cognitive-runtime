from __future__ import annotations

from hashlib import sha256

from models import (
    CompanionPolicyOverlay,
    InteractionContract,
    InteractionContractTrace,
    RuntimeState,
)
from services.companion_contracts import (
    DEFAULT_CONTRACT_WARNING,
    CompanionProfileRecord,
    InteractionContractRecord,
    ScenePolicyRecord,
    companion_contracts_repository,
)
from services.scene_resolution import SceneResolutionResult, resolve_scene_policy

_KNOWN_SURFACES = {
    "unknown",
    "dev",
    "vscode",
    "web",
    "telegram",
    "alexa",
    "car",
}


def active_profile() -> CompanionProfileRecord:
    return companion_contracts_repository().active_profile()


def resolve_scene(
    *,
    requested_scene: str | None,
    runtime_scene: str | None,
) -> SceneResolutionResult:
    return resolve_scene_policy(
        repository=companion_contracts_repository(),
        requested_scene=requested_scene,
        runtime_scene=runtime_scene,
    )


def _overlay_id(
    *,
    profile: CompanionProfileRecord,
    contract: InteractionContractRecord,
    scene: ScenePolicyRecord,
    overlay_type: str,
) -> str:
    material = "|".join(
        [
            profile.profile_id,
            str(profile.version),
            contract.contract_id,
            str(contract.contract_version),
            scene.scene_id,
            str(scene.version),
            overlay_type,
        ]
    )
    digest = sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"cpol_{digest}"


def _overlay(
    *,
    profile: CompanionProfileRecord,
    contract: InteractionContractRecord,
    scene: ScenePolicyRecord,
    overlay_type: str,
    content: str,
) -> CompanionPolicyOverlay:
    return CompanionPolicyOverlay(
        overlay_id=_overlay_id(
            profile=profile,
            contract=contract,
            scene=scene,
            overlay_type=overlay_type,
        ),
        overlay_type=overlay_type,
        content=content,
    )


def _contract_model(
    *,
    owner_id: str,
    contract: InteractionContractRecord,
) -> InteractionContract:
    return InteractionContract(
        contract_id=contract.contract_id,
        contract_version=contract.contract_version,
        owner_id=owner_id,
        scope=contract.scope,
        source=contract.source,
        trust_rules=contract.trust_rules,
        interaction_boundaries=contract.interaction_boundaries,
        repair_rules=contract.repair_rules,
        memory_or_recall_boundaries=contract.memory_or_recall_boundaries,
        autonomy_rules=contract.autonomy_rules,
        tone_constraints=contract.tone_constraints,
        allowed_intervention_styles=contract.allowed_intervention_styles,
        disallowed_intervention_styles=contract.disallowed_intervention_styles,
        defer_conditions=contract.defer_conditions,
    )


def _validate_interaction_contract(contract: InteractionContract) -> None:
    if not contract.trust_rules:
        raise RuntimeError("invalid_interaction_contract_record: trust_rules")
    if not contract.interaction_boundaries:
        raise RuntimeError("invalid_interaction_contract_record: interaction_boundaries")
    if not contract.memory_or_recall_boundaries:
        raise RuntimeError("invalid_interaction_contract_record: memory_or_recall_boundaries")
    if not contract.autonomy_rules:
        raise RuntimeError("invalid_interaction_contract_record: autonomy_rules")
    if len(contract.repair_rules) < 2:
        raise RuntimeError("invalid_interaction_contract_record: repair_rules")
    if not contract.tone_constraints:
        raise RuntimeError("invalid_interaction_contract_record: tone_constraints")


def _contract_overlay_content(contract: InteractionContract) -> str:
    _validate_interaction_contract(contract)
    return (
        "Interaction contract: be candid and useful while respecting boundaries. "
        f"Trust: {contract.trust_rules[0]} "
        f"Boundaries: {contract.interaction_boundaries[0]} "
        f"Memory: {contract.memory_or_recall_boundaries[0]} "
        f"Autonomy: {contract.autonomy_rules[0]} "
        f"Repair: {contract.repair_rules[0]} {contract.repair_rules[1]} "
        f"Tone: {contract.tone_constraints[0]}"
    )


def _contract_warnings(
    *,
    surface: str,
    requested_scene: str | None,
    runtime_state: RuntimeState,
) -> list[str]:
    repository = companion_contracts_repository()
    warnings = [DEFAULT_CONTRACT_WARNING]
    if surface not in _KNOWN_SURFACES:
        warnings.append("unknown_surface_default_contract")
    if requested_scene is not None and repository.scene_policy(requested_scene) is None:
        warnings.append("unknown_requested_scene")
    elif requested_scene is None and runtime_state.active_scene:
        if repository.scene_policy(runtime_state.active_scene) is None:
            warnings.append("unknown_runtime_scene")
    return warnings


def resolve_interaction_contract(
    *,
    owner_id: str,
    surface: str,
    requested_scene: str | None,
    runtime_state: RuntimeState,
) -> tuple[InteractionContract, InteractionContractTrace]:
    profile = active_profile()
    contract_record = companion_contracts_repository().active_interaction_contract(
        profile_id=profile.profile_id,
        profile_version=profile.version,
    )
    contract = _contract_model(owner_id=owner_id, contract=contract_record)
    _validate_interaction_contract(contract)
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
        warnings=_contract_warnings(
            surface=surface,
            requested_scene=requested_scene,
            runtime_state=runtime_state,
        ),
    )
    return contract, trace


def compile_policy(
    *,
    state: RuntimeState,
    requested_scene: str | None = None,
) -> dict[str, object]:
    profile = active_profile()
    scene_resolution = resolve_scene(
        requested_scene=requested_scene,
        runtime_scene=state.active_scene,
    )
    scene = scene_resolution.scene
    scene_confidence = scene_resolution.confidence
    scene_source = scene_resolution.source
    scene_warnings = scene_resolution.warnings
    contract_record = companion_contracts_repository().active_interaction_contract(
        profile_id=profile.profile_id,
        profile_version=profile.version,
    )
    contract = _contract_model(owner_id=state.owner_id, contract=contract_record)
    _validate_interaction_contract(contract)
    contract_trace = InteractionContractTrace(
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
        warnings=_contract_warnings(
            surface=state.surface,
            requested_scene=requested_scene,
            runtime_state=state,
        ),
    )
    warnings = list(dict.fromkeys([*scene_warnings, *contract_trace.warnings]))
    overlays = [
        _overlay(
            profile=profile,
            contract=contract_record,
            scene=scene,
            overlay_type="interaction_contract",
            content=_contract_overlay_content(contract),
        ),
        _overlay(
            profile=profile,
            contract=contract_record,
            scene=scene,
            overlay_type="companion_profile",
            content=profile.content,
        ),
        _overlay(
            profile=profile,
            contract=contract_record,
            scene=scene,
            overlay_type="scene_policy",
            content=scene.content,
        ),
    ]
    return {
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "contract_id": contract.contract_id,
        "contract_version": contract.contract_version,
        "scene_id": scene.scene_id,
        "scene_confidence": scene_confidence,
        "scene_source": scene_source,
        "warnings": warnings,
        "interaction_contract": contract,
        "contract_trace": contract_trace,
        "runtime_state": state,
        "overlays": overlays,
    }
