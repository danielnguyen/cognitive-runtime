from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

AttentionStatus = Literal["active", "paused", "resolved"]
BoundedLabel = Annotated[str, Field(min_length=1, max_length=64)]
BoundedTraceRef = Annotated[str, Field(min_length=1, max_length=120)]
BoundedCompanionContent = Annotated[str, Field(min_length=1, max_length=1200)]
BoundedScene = Annotated[str, Field(min_length=1, max_length=64)]
BoundedRule = Annotated[str, Field(min_length=1, max_length=240)]


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


class InteractionContract(BaseModel):
    contract_id: str = Field(max_length=120)
    contract_version: int
    owner_id: str = Field(max_length=120)
    scope: Literal["global_default", "owner_default"] = "global_default"
    source: Literal["default_static", "persisted"] = "default_static"
    trust_rules: list[BoundedRule] = Field(default_factory=list, max_length=8)
    interaction_boundaries: list[BoundedRule] = Field(default_factory=list, max_length=8)
    repair_rules: list[BoundedRule] = Field(default_factory=list, max_length=8)
    memory_or_recall_boundaries: list[BoundedRule] = Field(
        default_factory=list,
        max_length=8,
    )
    autonomy_rules: list[BoundedRule] = Field(default_factory=list, max_length=8)
    tone_constraints: list[BoundedRule] = Field(default_factory=list, max_length=8)
    allowed_intervention_styles: list[BoundedLabel] = Field(
        default_factory=list,
        max_length=8,
    )
    disallowed_intervention_styles: list[BoundedLabel] = Field(
        default_factory=list,
        max_length=8,
    )
    defer_conditions: list[BoundedRule] = Field(default_factory=list, max_length=8)


class InteractionContractTrace(BaseModel):
    contract_id: str = Field(max_length=120)
    contract_version: int
    source: Literal["default_static", "persisted"]
    scope: Literal["global_default", "owner_default"]
    selected_rule_groups: list[BoundedLabel] = Field(default_factory=list, max_length=12)
    selected_boundary_rules: list[BoundedRule] = Field(default_factory=list, max_length=8)
    selected_repair_rules: list[BoundedRule] = Field(default_factory=list, max_length=8)
    warnings: list[BoundedLabel] = Field(default_factory=list, max_length=8)


class CompanionPolicyCompileRequest(RuntimeStateResolveRequest):
    requested_scene: BoundedScene | None = None


class CompanionPolicyOverlay(BaseModel):
    overlay_id: str
    overlay_type: Literal[
        "interaction_contract",
        "companion_profile",
        "scene_policy",
    ]
    priority: Literal["companion_policy"] = "companion_policy"
    role: Literal["system"] = "system"
    content: BoundedCompanionContent


class CompanionPolicyCompileResponse(BaseModel):
    profile_id: str
    profile_version: int
    contract_id: str
    contract_version: int
    scene_id: str
    scene_confidence: float = Field(ge=0.0, le=1.0)
    scene_source: Literal[
        "requested_scene",
        "runtime_state",
        "general",
        "fallback_general",
    ]
    warnings: list[BoundedLabel] = Field(default_factory=list, max_length=8)
    interaction_contract: InteractionContract
    contract_trace: InteractionContractTrace
    runtime_state: RuntimeState
    overlays: list[CompanionPolicyOverlay] = Field(min_length=3, max_length=3)

