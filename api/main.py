from __future__ import annotations

from fastapi import FastAPI, HTTPException
from models import (
    CompanionPolicyCompileRequest,
    CompanionPolicyCompileResponse,
    CompanionProfileActiveResponse,
    HumanCompatibilityDiagnosticsRequest,
    HumanCompatibilityDiagnosticsResponse,
    HumanCompatibilityReviewRequest,
    HumanCompatibilityReviewResponse,
    RelationshipDiagnosticsResponse,
    RelationshipEdgeConfirmRequest,
    RelationshipEdgeResponse,
    RelationshipEdgeUpsertRequest,
    RelationshipEdgeRevokeRequest,
    RelationshipEntityResponse,
    RelationshipEntityUpsertRequest,
    RelationshipGraphDiagnosticsRequest,
    RelationshipSelectRequest,
    RelationshipSelectResponse,
    SocialContextDiagnosticsRequest,
    SocialContextDiagnosticsResponse,
    SocialContextItemResponse,
    SocialContextItemUpsertRequest,
    SocialContextUsageEventRecordRequest,
    SocialContextUsageEventResponse,
    InteractionContractValidateRequest,
    InteractionContractValidateResponse,
    InterruptEvaluateRequest,
    InterruptEvaluateResponse,
    RepairSimulateRequest,
    RepairSimulateResponse,
    RuntimeIdentityResolveRequest,
    RuntimeIdentityResolveResponse,
    RuntimeOverlayResponse,
    RuntimeSessionDiagnosticsResponse,
    RuntimeSessionResolveRequest,
    RuntimeSessionResponse,
    RuntimeStateResetRequest,
    RuntimeStateResetResponse,
    RuntimeStateResolveRequest,
    RuntimeStateResponse,
    RuntimeStateUpdateRequest,
    RuntimeTurnCompleteRequest,
    RuntimeTurnResponse,
    RuntimeTurnStartRequest,
    RuntimeTurnUpdateRequest,
    ScenePolicyDetail,
    SceneResolveRequest,
    SceneResolveResponse,
    WorldStateClaimResponse,
    WorldStateClaimUpsertRequest,
    WorldStateDiagnosticsRequest,
    WorldStateDiagnosticsResponse,
    WorldStateResolveRequest,
    WorldStateResolveResponse,
)
from services.companion_contracts import companion_contracts_repository
from services.companion_policy import (
    active_profile,
    compile_policy,
    resolve_interaction_contract,
    resolve_scene,
)
from services.human_compatibility import (
    get_human_compatibility_diagnostics,
    submit_human_compatibility_review,
)
from services.interaction_diagnostics import (
    simulate_repair_text,
    summarize_text,
    validate_interaction_text,
)
from services.interrupt_policy import evaluate_interrupt_policy
from services.relationships import (
    confirm_relationship_edge,
    get_relationship_diagnostics,
    get_relationship_edge,
    get_relationship_entity,
    revoke_relationship_edge,
    select_relationships,
    upsert_relationship_edge,
    upsert_relationship_entity,
)
from services.runtime_identity import resolve_runtime_identity
from services.runtime_state import (
    build_overlay,
    complete_turn,
    get_runtime_session,
    record_runtime_event,
    resolve_runtime_session,
    reset_state,
    resolve_state,
    start_turn,
    update_turn,
    update_state,
)
from services.social_context import (
    get_social_context_diagnostics,
    record_social_context_usage_event,
    upsert_social_context_item,
)
from services.world_state import (
    get_world_state_diagnostics,
    resolve_world_state,
    upsert_world_state_claim,
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


@app.post("/v1/runtime/sessions/resolve", response_model=RuntimeSessionResponse)
async def resolve_runtime_session_endpoint(
    body: RuntimeSessionResolveRequest,
) -> RuntimeSessionResponse:
    session = resolve_runtime_session(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        surface_session_id=body.surface_session_id,
        active_mode=body.active_mode,
    )
    return RuntimeSessionResponse(runtime_session=session)


@app.get(
    "/v1/runtime/sessions/{runtime_session_id}",
    response_model=RuntimeSessionDiagnosticsResponse,
)
async def runtime_session_diagnostics(
    runtime_session_id: str,
) -> RuntimeSessionDiagnosticsResponse:
    return get_runtime_session(runtime_session_id)


@app.post("/v1/runtime/turns/start", response_model=RuntimeTurnResponse)
async def runtime_turn_start(body: RuntimeTurnStartRequest) -> RuntimeTurnResponse:
    session, turn, event = start_turn(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        surface_session_id=body.surface_session_id,
        active_mode=body.active_mode,
        input_message_id=body.input_message_id,
        intent_class=body.intent_class,
        timing_policy=body.timing_policy,
        restraint_policy=body.restraint_policy,
        continuation_state=body.continuation_state,
    )
    return RuntimeTurnResponse(runtime_session=session, runtime_turn=turn, event=event)


@app.post("/v1/runtime/turns/update", response_model=RuntimeTurnResponse)
async def runtime_turn_update(body: RuntimeTurnUpdateRequest) -> RuntimeTurnResponse:
    session, turn, event = update_turn(
        request_id=body.request_id,
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        turn_status=body.turn_status,
        timing_policy=body.timing_policy,
        restraint_policy=body.restraint_policy,
        continuation_state=body.continuation_state,
    )
    return RuntimeTurnResponse(runtime_session=session, runtime_turn=turn, event=event)


@app.post("/v1/runtime/turns/complete", response_model=RuntimeTurnResponse)
async def runtime_turn_complete(body: RuntimeTurnCompleteRequest) -> RuntimeTurnResponse:
    session, turn, event = complete_turn(
        request_id=body.request_id,
        runtime_session_id=body.runtime_session_id,
        runtime_turn_id=body.runtime_turn_id,
        turn_status=body.turn_status,
        continuation_state=body.continuation_state,
    )
    return RuntimeTurnResponse(runtime_session=session, runtime_turn=turn, event=event)


@app.post(
    "/v1/runtime/human-compatibility/review",
    response_model=HumanCompatibilityReviewResponse,
)
async def runtime_human_compatibility_review(
    body: HumanCompatibilityReviewRequest,
) -> HumanCompatibilityReviewResponse:
    return submit_human_compatibility_review(body)


@app.post(
    "/v1/runtime/human-compatibility/diagnostics",
    response_model=HumanCompatibilityDiagnosticsResponse,
)
async def runtime_human_compatibility_diagnostics(
    body: HumanCompatibilityDiagnosticsRequest,
) -> HumanCompatibilityDiagnosticsResponse:
    return get_human_compatibility_diagnostics(body)


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


@app.post("/v1/runtime/identity/resolve", response_model=RuntimeIdentityResolveResponse)
async def runtime_identity_resolve(
    body: RuntimeIdentityResolveRequest,
) -> RuntimeIdentityResolveResponse:
    resolution = resolve_runtime_identity(body)
    record_runtime_event(
        runtime_session_id=resolution.runtime_session.runtime_session_id,
        runtime_turn_id=None,
        event_type="identity_resolved",
        event_payload_json={
            "request_id": body.request_id,
            "active_persona_id": resolution.trace.active_persona_id,
            "persona_resolution_reason": resolution.trace.persona_resolution_reason,
            "surface_id": resolution.trace.surface_id,
        },
    )
    return resolution


@app.post("/v1/relationships/entities/upsert", response_model=RelationshipEntityResponse)
async def relationship_entity_upsert(
    body: RelationshipEntityUpsertRequest,
) -> RelationshipEntityResponse:
    entity = upsert_relationship_entity(owner_id=body.owner_id, entity=body.entity)
    return RelationshipEntityResponse(entity=entity)


@app.post("/v1/relationships/edges/upsert", response_model=RelationshipEdgeResponse)
async def relationship_edge_upsert(
    body: RelationshipEdgeUpsertRequest,
) -> RelationshipEdgeResponse:
    return upsert_relationship_edge(owner_id=body.owner_id, edge=body.edge, evidence=body.evidence)


@app.post("/v1/relationships/edges/confirm", response_model=RelationshipEdgeResponse)
async def relationship_edge_confirm(
    body: RelationshipEdgeConfirmRequest,
) -> RelationshipEdgeResponse:
    return confirm_relationship_edge(owner_id=body.owner_id, body=body)


@app.post("/v1/relationships/edges/revoke", response_model=RelationshipEdgeResponse)
async def relationship_edge_revoke(
    body: RelationshipEdgeRevokeRequest,
) -> RelationshipEdgeResponse:
    return revoke_relationship_edge(owner_id=body.owner_id, body=body)


@app.get("/v1/relationships/entities/{entity_id}", response_model=RelationshipEntityResponse)
async def relationship_entity_get(entity_id: str, owner_id: str) -> RelationshipEntityResponse:
    entity = get_relationship_entity(owner_id=owner_id, entity_id=entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="relationship_entity_not_found")
    return RelationshipEntityResponse(entity=entity)


@app.get("/v1/relationships/edges/{relationship_id}", response_model=RelationshipEdgeResponse)
async def relationship_edge_get(relationship_id: str, owner_id: str) -> RelationshipEdgeResponse:
    relationship = get_relationship_edge(owner_id=owner_id, relationship_id=relationship_id)
    if relationship is None:
        raise HTTPException(status_code=404, detail="relationship_edge_not_found")
    return RelationshipEdgeResponse(relationship=relationship, evidence=[])


@app.post("/v1/relationships/diagnostics", response_model=RelationshipDiagnosticsResponse)
async def relationship_diagnostics(
    body: RelationshipGraphDiagnosticsRequest,
) -> RelationshipDiagnosticsResponse:
    return get_relationship_diagnostics(body)


@app.post("/v1/relationships/select", response_model=RelationshipSelectResponse)
async def relationship_select(body: RelationshipSelectRequest) -> RelationshipSelectResponse:
    return select_relationships(body)


@app.post("/v1/social-context/items/upsert", response_model=SocialContextItemResponse)
async def social_context_item_upsert(
    body: SocialContextItemUpsertRequest,
) -> SocialContextItemResponse:
    item = upsert_social_context_item(owner_id=body.owner_id, item=body.item)
    return SocialContextItemResponse(item=item)


@app.post("/v1/social-context/usage-events/record", response_model=SocialContextUsageEventResponse)
async def social_context_usage_event_record(
    body: SocialContextUsageEventRecordRequest,
) -> SocialContextUsageEventResponse:
    event = record_social_context_usage_event(owner_id=body.owner_id, event=body.event)
    return SocialContextUsageEventResponse(event=event)


@app.post("/v1/social-context/diagnostics", response_model=SocialContextDiagnosticsResponse)
async def social_context_diagnostics(
    body: SocialContextDiagnosticsRequest,
) -> SocialContextDiagnosticsResponse:
    return get_social_context_diagnostics(body)


@app.post("/v1/world-state/claims/upsert", response_model=WorldStateClaimResponse)
async def world_state_claim_upsert(
    body: WorldStateClaimUpsertRequest,
) -> WorldStateClaimResponse:
    claim, transitions = upsert_world_state_claim(
        owner_id=body.owner_id,
        claim=body.claim,
    )
    return WorldStateClaimResponse(claim=claim, transitions=transitions)


@app.post("/v1/world-state/diagnostics", response_model=WorldStateDiagnosticsResponse)
async def world_state_diagnostics(
    body: WorldStateDiagnosticsRequest,
) -> WorldStateDiagnosticsResponse:
    return get_world_state_diagnostics(
        owner_id=body.owner_id,
        include_sensitive_values=False,
    )


@app.post("/v1/world-state/resolve", response_model=WorldStateResolveResponse)
async def world_state_resolve(
    body: WorldStateResolveRequest,
) -> WorldStateResolveResponse:
    return resolve_world_state(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        runtime_session_id=body.runtime_session_id,
        active_persona_id=body.active_persona_id,
        requested_domains=body.requested_domains,
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
