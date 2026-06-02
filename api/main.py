from __future__ import annotations

from fastapi import FastAPI
from models import (
    CompanionPolicyCompileRequest,
    CompanionPolicyCompileResponse,
    InterruptEvaluateRequest,
    InterruptEvaluateResponse,
    RuntimeOverlayResponse,
    RuntimeStateResetRequest,
    RuntimeStateResetResponse,
    RuntimeStateResolveRequest,
    RuntimeStateResponse,
    RuntimeStateUpdateRequest,
)
from services.companion_policy import compile_policy
from services.interrupt_policy import evaluate_interrupt_policy
from services.runtime_state import build_overlay, reset_state, resolve_state, update_state

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

@app.post("/v1/companion/policy/compile", response_model=CompanionPolicyCompileResponse)
async def companion_policy_compile(
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


@app.post("/v1/interrupt/evaluate", response_model=InterruptEvaluateResponse)
async def interrupt_evaluate(body: InterruptEvaluateRequest) -> InterruptEvaluateResponse:
    return evaluate_interrupt_policy(body)
