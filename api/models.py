from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

AttentionStatus = Literal["active", "paused", "resolved"]
BoundedLabel = Annotated[str, Field(min_length=1, max_length=64)]
BoundedTraceRef = Annotated[str, Field(min_length=1, max_length=120)]


class AttentionFocus(BaseModel):
    topic: str | None = Field(default=None, max_length=160)
    task: str | None = Field(default=None, max_length=160)
    status: AttentionStatus = "active"


class RuntimeState(BaseModel):
    runtime_state_id: str
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(default="unknown", max_length=64)
    active_scene: str | None = Field(default=None, max_length=64)
    interaction_mode: str | None = Field(default=None, max_length=64)
    attention_focus: AttentionFocus | None = None
    temporary_constraints: list[BoundedLabel] = Field(default_factory=list, max_length=8)
    reset_after_turn: bool = False
    trace_refs: list[BoundedTraceRef] = Field(default_factory=list, max_length=16)
    created_at: str
    updated_at: str


class RuntimeStateResolveRequest(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(default="unknown", max_length=64)


class RuntimeStateUpdate(BaseModel):
    active_scene: str | None = Field(default=None, max_length=64)
    interaction_mode: str | None = Field(default=None, max_length=64)
    attention_focus: AttentionFocus | None = None
    temporary_constraints: list[BoundedLabel] | None = Field(default=None, max_length=8)
    reset_after_turn: bool | None = None
    trace_refs: list[BoundedTraceRef] | None = Field(default=None, max_length=16)


class RuntimeStateUpdateRequest(RuntimeStateResolveRequest):
    updates: RuntimeStateUpdate


class RuntimeStateResetRequest(RuntimeStateResolveRequest):
    reason: str | None = Field(default=None, max_length=80)


class RuntimeStateResponse(BaseModel):
    runtime_state: RuntimeState


class RuntimeOverlay(BaseModel):
    overlay_id: str
    runtime_state_id: str
    overlay_type: Literal["runtime_state"] = "runtime_state"
    priority: Literal["after_profile_before_retrieval"] = "after_profile_before_retrieval"
    role: Literal["system"] = "system"
    content: str
    source_fields: list[BoundedLabel] = Field(default_factory=list)


class RuntimeOverlayResponse(BaseModel):
    runtime_state: RuntimeState
    overlay: RuntimeOverlay | None = None
    omitted: bool
    omission_reason: str | None = None


class RuntimeStateResetResponse(BaseModel):
    runtime_state: RuntimeState
    reset: bool
