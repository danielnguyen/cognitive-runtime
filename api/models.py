from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

AttentionStatus = Literal["active", "paused", "resolved"]
BoundedLabel = Annotated[str, Field(min_length=1, max_length=64)]
BoundedTraceRef = Annotated[str, Field(min_length=1, max_length=120)]
BoundedCompanionContent = Annotated[str, Field(min_length=1, max_length=1200)]
BoundedScene = Annotated[str, Field(min_length=1, max_length=64)]
BoundedRule = Annotated[str, Field(min_length=1, max_length=240)]
BoundedText = Annotated[str, Field(min_length=1, max_length=2000)]
InterruptTriggerClass = Literal[
    "repetitive_branching",
    "speculative_simulation_with_weak_evidence",
    "avoidance_disguised_as_analysis",
    "complexity_expansion_beyond_task_value",
    "rising_agitation_with_shrinking_informational_gain",
    "mismatch_between_context_and_answer_depth",
    "known_recurring_trap_pattern",
]
InterruptStyle = Literal[
    "soft_redirect",
    "crisp_callout",
    "constraint_reset",
    "next_step_forcing",
    "evidence_anchor",
    "scene_aware_simplification",
]


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
    source: Literal["default_compiled", "default_static", "persisted"] = "default_compiled"
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
    source: Literal["default_compiled", "default_static", "persisted"]
    scope: Literal["global_default", "owner_default"]
    selected_rule_groups: list[BoundedLabel] = Field(default_factory=list, max_length=12)
    selected_boundary_rules: list[BoundedRule] = Field(default_factory=list, max_length=8)
    selected_repair_rules: list[BoundedRule] = Field(default_factory=list, max_length=8)
    warnings: list[BoundedLabel] = Field(default_factory=list, max_length=8)


class CompanionPolicyCompileRequest(RuntimeStateResolveRequest):
    requested_scene: BoundedScene | None = None


class CompanionProfileActiveResponse(BaseModel):
    profile_id: str = Field(max_length=120)
    profile_version: int
    name: str = Field(max_length=120)
    scope: Literal["global_default", "owner_default"] = "global_default"
    source: str = Field(max_length=64)
    status: str = Field(max_length=64)
    role_label: str = Field(max_length=120)
    core_traits_json: dict[str, Any] = Field(default_factory=dict)
    behavioral_laws_json: list[BoundedRule] = Field(default_factory=list, max_length=16)
    style_constraints_json: dict[str, Any] = Field(default_factory=dict)
    surface_overrides_json: dict[str, Any] = Field(default_factory=dict)


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


class InterruptMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str = Field(max_length=2000)


class InterruptEvaluateRequest(RuntimeStateResolveRequest):
    requested_scene: BoundedScene | None = None
    current_user_text: BoundedText | None = None
    recent_messages: list[InterruptMessage] = Field(default_factory=list, max_length=12)
    runtime_state: RuntimeState | None = None
    interaction_contract: InteractionContract | None = None
    contract_trace: InteractionContractTrace | None = None


class InterruptDebug(BaseModel):
    detector_signals: dict[str, Any] = Field(default_factory=dict)
    advisory_text: str | None = Field(default=None, max_length=240)
    user_visible_suppressed: bool = True
    degraded: bool = False


class InterruptEvaluateResponse(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    requested_scene: str | None = Field(default=None, max_length=64)
    runtime_state: RuntimeState | None = None
    interaction_contract: InteractionContract | None = None
    contract_trace: InteractionContractTrace | None = None
    trigger_class: InterruptTriggerClass | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    style_selected: InterruptStyle | None = None
    should_interrupt: bool
    should_defer: bool
    reason_json: dict[str, Any] = Field(default_factory=dict)
    contract_constraints_applied: dict[str, Any] = Field(default_factory=dict)
    warnings: list[BoundedLabel] = Field(default_factory=list, max_length=12)
    debug: InterruptDebug
