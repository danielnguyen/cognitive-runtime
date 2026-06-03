from __future__ import annotations

from fastapi import FastAPI, HTTPException
from models import (
    CompanionPolicyCompileRequest,
    CompanionPolicyCompileResponse,
    CompanionProfileActiveResponse,
    InteractionContractValidateRequest,
    InteractionContractValidateResponse,
    InterruptEvaluateRequest,
    InterruptEvaluateResponse,
    RepairSimulateRequest,
    RepairSimulateResponse,
    RuntimeOverlayResponse,
    RuntimeStateResetRequest,
    RuntimeStateResetResponse,
    RuntimeStateResolveRequest,
    RuntimeStateResponse,
    RuntimeStateUpdateRequest,
    ScenePolicyDetail,
    SceneResolveRequest,
    SceneResolveResponse,
)
from services.companion_contracts import companion_contracts_repository
from services.companion_policy import (
    active_profile,
    compile_policy,
    resolve_interaction_contract,
    resolve_scene,
)
from services.interaction_diagnostics import (
    simulate_repair_text,
    summarize_text,
    validate_interaction_text,
)
from services.interrupt_policy import evaluate_interrupt_policy
from services.runtime_state import (
    build_overlay,
    reset_state,
    resolve_state,
    update_state,
)

app = FastAPI(title="Cognitive Runtime", version="0.1.0")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "cognitive-runtime"}


@app.post("/v1/runtime/state/resolve", response_model=RuntimeStateResponse)
async def resolve_runtime_state(body: RuntimeStateResolveRequest) -> RuntimeStateResponse:
    state = resolve_state(
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
    )
    return RuntimeStateResponse(runtime_state=state)


@app.post("/v1/runtime/state/update", response_model=RuntimeStateResponse)
async def update_runtime_state(body: RuntimeStateUpdateRequest) -> RuntimeStateResponse:
    state = update_state(
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        updates=body.updates,
    )
    return RuntimeStateResponse(runtime_state=state)


@app.post("/v1/runtime/state/reset", response_model=RuntimeStateResetResponse)
async def reset_runtime_state(body: RuntimeStateResetRequest) -> RuntimeStateResetResponse:
    state = reset_state(
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
    )
    return RuntimeStateResetResponse(runtime_state=state, reset=True)


@app.post("/v1/runtime/overlay", response_model=RuntimeOverlayResponse)
async def runtime_overlay(body: RuntimeStateResolveRequest) -> RuntimeOverlayResponse:
    state = resolve_state(
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
    )
    overlay, omission_reason = build_overlay(state)
    return RuntimeOverlayResponse(
        runtime_state=state,
        overlay=overlay,
        omitted=overlay is None,
        omission_reason=omission_reason,
    )


@app.get("/v1/companion/profile/active", response_model=CompanionProfileActiveResponse)
async def companion_profile_active() -> CompanionProfileActiveResponse:
    profile = active_profile()
    return CompanionProfileActiveResponse(
        profile_id=profile.profile_id,
        profile_version=profile.version,
        name=profile.name,
        scope=profile.scope,
        source=profile.source,
        status=profile.status,
        role_label=profile.role_label,
        core_traits_json=profile.core_traits_json,
        behavioral_laws_json=profile.behavioral_laws_json,
        style_constraints_json=profile.style_constraints_json,
        surface_overrides_json=profile.surface_overrides_json,
    )


@app.post("/v1/companion/scene/resolve", response_model=SceneResolveResponse)
async def companion_scene_resolve(body: SceneResolveRequest) -> SceneResolveResponse:
    state = resolve_state(
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
    )
    resolution = resolve_scene(
        requested_scene=body.requested_scene,
        runtime_scene=state.active_scene,
    )
    companion_contracts_repository().record_scene_resolution_event(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        requested_scene=body.requested_scene,
        runtime_scene=state.active_scene,
        resolved_scene_id=resolution.scene.scene_id,
        confidence=resolution.confidence,
        source=resolution.source,
        signals_json=resolution.signals_json,
        used_fallback=resolution.used_fallback,
        used_default_scene=resolution.used_default_scene,
    )
    return SceneResolveResponse(
        scene_id=resolution.scene.scene_id,
        scene_version=resolution.scene.version,
        scene_confidence=resolution.confidence,
        scene_source=resolution.source,
        warnings=resolution.warnings,
        signals_json=resolution.signals_json,
        used_fallback=resolution.used_fallback,
        used_default_scene=resolution.used_default_scene,
        policy=_scene_policy_detail(resolution.scene),
    )


@app.get("/v1/companion/scene/{scene_id}", response_model=ScenePolicyDetail)
async def companion_scene_detail(scene_id: str) -> ScenePolicyDetail:
    scene = companion_contracts_repository().scene_policy(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="scene_not_found")
    return _scene_policy_detail(scene)


@app.post(
    "/v1/companion/interaction-contract/validate",
    response_model=InteractionContractValidateResponse,
)
async def companion_interaction_contract_validate(
    body: InteractionContractValidateRequest,
) -> InteractionContractValidateResponse:
    state = resolve_state(
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
    )
    if body.interaction_contract is not None:
        contract = body.interaction_contract
    else:
        contract, _ = resolve_interaction_contract(
            owner_id=body.owner_id,
            surface=body.surface,
            requested_scene=body.requested_scene,
            runtime_state=state,
        )
    validation = validate_interaction_text(text=body.text, contract=contract)
    companion_contracts_repository().record_interaction_boundary_event(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        contract_id=contract.contract_id,
        contract_version=contract.contract_version,
        check_type="interaction_contract_validation",
        severity=validation["severity"],
        input_summary=summarize_text(body.text),
        result=validation["result"],
        reason_json=validation["reason_json"],
    )
    return InteractionContractValidateResponse(diagnostic_only=True, **validation)


@app.post("/v1/companion/repair/simulate", response_model=RepairSimulateResponse)
async def companion_repair_simulate(body: RepairSimulateRequest) -> RepairSimulateResponse:
    state = resolve_state(
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
    )
    contract, _ = resolve_interaction_contract(
        owner_id=body.owner_id,
        surface=body.surface,
        requested_scene=body.requested_scene,
        runtime_state=state,
    )
    repair = simulate_repair_text(
        miss_description=body.miss_description,
        corrected_substance=body.corrected_substance,
    )
    validation = validate_interaction_text(text=repair["repair_text"], contract=contract)
    reason_json = dict(repair["reason_json"])
    reason_json["validation"] = validation["reason_json"]
    companion_contracts_repository().record_interaction_boundary_event(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        contract_id=contract.contract_id,
        contract_version=contract.contract_version,
        check_type="repair_simulation",
        severity=validation["severity"],
        input_summary=summarize_text(
            f"{body.miss_description} {body.corrected_substance}"
        ),
        result=validation["result"],
        reason_json=reason_json,
    )
    return RepairSimulateResponse(
        diagnostic_only=True,
        repair_text=repair["repair_text"],
        result=validation["result"],
        severity=validation["severity"],
        warnings=validation["warnings"],
        contract_id=contract.contract_id,
        contract_version=contract.contract_version,
        reason_json=reason_json,
    )


def _compile_companion_profile(
    body: CompanionPolicyCompileRequest,
) -> CompanionPolicyCompileResponse:
    state = resolve_state(
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
    )
    return CompanionPolicyCompileResponse(
        **compile_policy(state=state, requested_scene=body.requested_scene)
    )


@app.post("/v1/companion/profile/compile", response_model=CompanionPolicyCompileResponse)
async def companion_profile_compile(
    body: CompanionPolicyCompileRequest,
) -> CompanionPolicyCompileResponse:
    return _compile_companion_profile(body)


@app.post("/v1/companion/policy/compile", response_model=CompanionPolicyCompileResponse)
async def companion_policy_compile(
    body: CompanionPolicyCompileRequest,
) -> CompanionPolicyCompileResponse:
    return _compile_companion_profile(body)


@app.post("/v1/interrupt/evaluate", response_model=InterruptEvaluateResponse)
async def interrupt_evaluate(body: InterruptEvaluateRequest) -> InterruptEvaluateResponse:
    return evaluate_interrupt_policy(body)


def _scene_policy_detail(scene) -> ScenePolicyDetail:
    return ScenePolicyDetail(
        scene_id=scene.scene_id,
        scene_version=scene.version,
        aliases=scene.aliases,
        content=scene.content,
        active=scene.active,
        status=scene.status,
        constraints_json=scene.constraints_json,
        initiative_policy_json=scene.initiative_policy_json,
        interrupt_policy_json=scene.interrupt_policy_json,
        recall_policy_json=scene.recall_policy_json,
        format_policy_json=scene.format_policy_json,
    )
