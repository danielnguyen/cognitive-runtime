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


RuntimeSessionStatus = Literal[
    "opening",
    "active",
    "paused",
    "idle",
    "closing",
    "closed",
]
RuntimeTurnStatus = Literal[
    "received",
    "retrieving",
    "responding",
    "completed",
    "abandoned",
]
RuntimeEventType = Literal[
    "session_resolved",
    "turn_started",
    "turn_updated",
    "turn_completed",
    "identity_resolved",
]


class RuntimeSession(BaseModel):
    runtime_session_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    surface_session_id: str | None = Field(default=None, max_length=120)
    status: RuntimeSessionStatus = "active"
    active_mode: str | None = Field(default=None, max_length=64)
    attention_state: str | None = Field(default=None, max_length=64)
    started_at: str
    last_activity_at: str
    closed_at: str | None = None


class RuntimeTurn(BaseModel):
    runtime_turn_id: str = Field(max_length=120)
    runtime_session_id: str = Field(max_length=120)
    input_message_id: str | None = Field(default=None, max_length=120)
    turn_status: RuntimeTurnStatus
    intent_class: str | None = Field(default=None, max_length=64)
    timing_policy: str | None = Field(default=None, max_length=64)
    restraint_policy: str | None = Field(default=None, max_length=64)
    continuation_state: str | None = Field(default=None, max_length=64)
    created_at: str
    updated_at: str
    completed_at: str | None = None


class RuntimeEvent(BaseModel):
    event_id: str = Field(max_length=120)
    runtime_session_id: str = Field(max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    event_type: RuntimeEventType
    event_payload_json: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class RuntimeSessionResolveRequest(RuntimeStateResolveRequest):
    surface_session_id: str | None = Field(default=None, max_length=120)
    active_mode: str | None = Field(default=None, max_length=64)


class RuntimeSessionResponse(BaseModel):
    runtime_session: RuntimeSession


class RuntimeTurnStartRequest(RuntimeSessionResolveRequest):
    input_message_id: str | None = Field(default=None, max_length=120)
    intent_class: str | None = Field(default=None, max_length=64)
    timing_policy: str | None = Field(default=None, max_length=64)
    restraint_policy: str | None = Field(default=None, max_length=64)
    continuation_state: str | None = Field(default=None, max_length=64)


class RuntimeTurnUpdateRequest(BaseModel):
    request_id: str = Field(max_length=120)
    runtime_session_id: str = Field(max_length=120)
    runtime_turn_id: str = Field(max_length=120)
    turn_status: Literal["retrieving", "responding"]
    timing_policy: str | None = Field(default=None, max_length=64)
    restraint_policy: str | None = Field(default=None, max_length=64)
    continuation_state: str | None = Field(default=None, max_length=64)


class RuntimeTurnCompleteRequest(BaseModel):
    request_id: str = Field(max_length=120)
    runtime_session_id: str = Field(max_length=120)
    runtime_turn_id: str = Field(max_length=120)
    turn_status: Literal["completed", "abandoned"] = "completed"
    continuation_state: str | None = Field(default=None, max_length=64)


class RuntimeTurnResponse(BaseModel):
    runtime_session: RuntimeSession
    runtime_turn: RuntimeTurn
    event: RuntimeEvent


class RuntimeSessionDiagnosticsResponse(BaseModel):
    runtime_session: RuntimeSession
    active_turn: RuntimeTurn | None = None
    latest_turn: RuntimeTurn | None = None
    events: list[RuntimeEvent] = Field(default_factory=list)


class PersonaProfile(BaseModel):
    persona_id: str = Field(max_length=120)
    display_name: str = Field(max_length=120)
    capability_domain: str = Field(max_length=120)
    description: str = Field(max_length=240)
    communication_policy_summary: list[BoundedRule] = Field(default_factory=list, max_length=8)
    runtime_policy_summary: list[BoundedRule] = Field(default_factory=list, max_length=8)
    advisory_memory_scope_summary: list[BoundedLabel] = Field(default_factory=list, max_length=12)
    advisory_tool_permission_summary: list[BoundedLabel] = Field(default_factory=list, max_length=12)
    persona_owns_durable_memory: Literal[False] = False


class SurfaceBinding(BaseModel):
    surface_id: str = Field(max_length=120)
    surface_type: str = Field(max_length=80)
    surface_display_name: str = Field(max_length=120)
    default_persona_id: str = Field(max_length=120)
    allow_user_persona_override: bool = False
    response_length: str | None = Field(default=None, max_length=64)
    default_mode: str | None = Field(default=None, max_length=64)


class RuntimeIdentityContext(BaseModel):
    active_persona_id: str = Field(max_length=120)
    surface_id: str = Field(max_length=120)
    surface_type: str = Field(max_length=80)
    surface_display_name: str = Field(max_length=120)
    capability_domain: str = Field(max_length=120)
    communication_policy_summary: list[BoundedRule] = Field(default_factory=list, max_length=8)
    runtime_policy_summary: list[BoundedRule] = Field(default_factory=list, max_length=8)
    advisory_memory_scope_summary: list[BoundedLabel] = Field(default_factory=list, max_length=12)
    advisory_tool_permission_summary: list[BoundedLabel] = Field(default_factory=list, max_length=12)
    persona_owns_durable_memory: Literal[False] = False
    content: BoundedCompanionContent


class RuntimeIdentityTrace(BaseModel):
    runtime_session_id: str = Field(max_length=120)
    active_persona_id: str = Field(max_length=120)
    persona_resolution_reason: Literal[
        "requested_persona_id",
        "surface_binding",
        "default_fallback",
    ]
    persona_override_source: Literal["internal_test", "surface_binding", "none"]
    surface_id: str = Field(max_length=120)
    surface_type: str = Field(max_length=80)
    surface_display_name: str = Field(max_length=120)
    persona_owns_durable_memory: Literal[False] = False
    advisory_memory_scope_summary: list[BoundedLabel] = Field(default_factory=list, max_length=12)
    advisory_tool_permission_summary: list[BoundedLabel] = Field(default_factory=list, max_length=12)


class RuntimeIdentityResolveRequest(RuntimeSessionResolveRequest):
    runtime_session_id: str | None = Field(default=None, max_length=120)
    requested_persona_id: str | None = Field(default=None, max_length=120)
    allow_requested_persona_bypass: bool = False


class RuntimeIdentityResolveResponse(BaseModel):
    runtime_session: RuntimeSession
    surface_binding: SurfaceBinding
    persona: PersonaProfile
    runtime_identity: RuntimeIdentityContext
    trace: RuntimeIdentityTrace


WorldStateFreshnessState = Literal[
    "fresh",
    "aging",
    "stale",
    "expired",
    "unknown",
    "superseded",
    "conflicted",
]
WorldStateAuthority = Literal[
    "observed_user_report",
    "verified_tool_output",
    "trusted_integration_event",
    "derived_from_multiple_sources",
    "model_inferred",
    "unverified_assumption",
]
WorldStateSourceType = Literal[
    "user_report",
    "tool_output",
    "integration_event",
    "sensor_update",
    "repository_inspection",
    "calendar_event",
    "automation_workflow",
    "model_inference",
    "artifact_metadata",
    "runtime_update",
]
WorldStateConfirmationPolicy = Literal[
    "none",
    "confirm_before_action",
    "confirm_before_high_impact_action",
    "reverify_before_use",
]
WorldStateSensitivity = Literal["low", "medium", "high", "restricted"]
WorldStateTransitionType = Literal[
    "created",
    "updated",
    "superseded",
    "conflicted",
    "resolved",
]


class WorldStateClaimInput(BaseModel):
    entity_id: str = Field(max_length=120)
    entity_type: str = Field(max_length=80)
    domain: BoundedLabel
    attribute: BoundedLabel
    value_json: Any
    source_type: WorldStateSourceType
    source_ref: str = Field(max_length=240)
    confidence: float = Field(ge=0.0, le=1.0)
    freshness_state: WorldStateFreshnessState
    state_authority: WorldStateAuthority
    observed_at: str
    last_verified_at: str | None = None
    expires_at: str | None = None
    ttl_seconds: int | None = Field(default=None, ge=1)
    revalidation_interval_seconds: int | None = Field(default=None, ge=1)
    confirmation_policy: WorldStateConfirmationPolicy
    sensitivity: WorldStateSensitivity = "medium"
    scope_labels: list[BoundedLabel] = Field(default_factory=list, max_length=12)
    supersede_existing_claim_id: str | None = Field(default=None, max_length=120)


class WorldStateClaimUpsertRequest(RuntimeStateResolveRequest):
    claim: WorldStateClaimInput


class WorldStateClaimView(BaseModel):
    world_state_claim_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    entity_id: str = Field(max_length=120)
    entity_type: str = Field(max_length=80)
    domain: BoundedLabel
    attribute: BoundedLabel
    value_json: Any | None = None
    value_redacted: bool = False
    source_type: WorldStateSourceType
    source_ref: str = Field(max_length=240)
    confidence: float
    freshness_state: WorldStateFreshnessState
    effective_freshness_state: WorldStateFreshnessState
    state_authority: WorldStateAuthority
    observed_at: str
    last_verified_at: str | None = None
    expires_at: str | None = None
    ttl_seconds: int | None = None
    revalidation_interval_seconds: int | None = None
    confirmation_policy: WorldStateConfirmationPolicy
    sensitivity: WorldStateSensitivity
    scope_labels: list[BoundedLabel] = Field(default_factory=list, max_length=12)
    created_at: str
    updated_at: str
    superseded_by_claim_id: str | None = None
    conflict_claim_ids: list[str] = Field(default_factory=list, max_length=16)


class WorldStateClaimSummary(BaseModel):
    world_state_claim_id: str = Field(max_length=120)
    entity_id: str = Field(max_length=120)
    attribute: BoundedLabel
    domain: BoundedLabel
    freshness_state: WorldStateFreshnessState
    effective_freshness_state: WorldStateFreshnessState
    sensitivity: WorldStateSensitivity
    reason: BoundedLabel
    superseded_by_claim_id: str | None = None
    conflict_claim_ids: list[str] = Field(default_factory=list, max_length=16)


class WorldStateTransition(BaseModel):
    transition_id: str = Field(max_length=120)
    world_state_claim_id: str = Field(max_length=120)
    transition_type: WorldStateTransitionType
    created_at: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class WorldStateClaimResponse(BaseModel):
    claim: WorldStateClaimView
    transitions: list[WorldStateTransition] = Field(default_factory=list)


class WorldStateDiagnosticsRequest(RuntimeStateResolveRequest):
    include_sensitive_values: bool = False


class WorldStateDiagnosticsResponse(BaseModel):
    claims: list[WorldStateClaimView] = Field(default_factory=list)
    excluded_claims: list[WorldStateClaimSummary] = Field(default_factory=list)
    transitions: list[WorldStateTransition] = Field(default_factory=list)


class InteractionContract(BaseModel):
    contract_id: str = Field(max_length=120)
    contract_version: int
    owner_id: str = Field(max_length=120)
    scope: Literal["global_default", "owner_default"] = "global_default"
    source: Literal["default_compiled", "persisted"] = "default_compiled"
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
    source: Literal["default_compiled", "persisted"]
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

DiagnosticResult = Literal["pass", "warn", "fail"]
DiagnosticSeverity = Literal["none", "low", "medium", "high"]
SceneSource = Literal["requested_scene", "runtime_state", "general", "fallback_general"]


class ScenePolicyDetail(BaseModel):
    scene_id: str = Field(max_length=64)
    scene_version: int
    aliases: list[BoundedScene] = Field(default_factory=list, max_length=8)
    content: BoundedCompanionContent
    active: bool
    status: str = Field(max_length=64)
    constraints_json: dict[str, Any] = Field(default_factory=dict)
    initiative_policy_json: dict[str, Any] = Field(default_factory=dict)
    interrupt_policy_json: dict[str, Any] = Field(default_factory=dict)
    recall_policy_json: dict[str, Any] = Field(default_factory=dict)
    format_policy_json: dict[str, Any] = Field(default_factory=dict)


class SceneResolveRequest(RuntimeStateResolveRequest):
    requested_scene: BoundedScene | None = None


class SceneResolveResponse(BaseModel):
    scene_id: str = Field(max_length=64)
    scene_version: int
    scene_confidence: float = Field(ge=0.0, le=1.0)
    scene_source: SceneSource
    warnings: list[BoundedLabel] = Field(default_factory=list, max_length=8)
    signals_json: dict[str, Any] = Field(default_factory=dict)
    used_fallback: bool
    used_default_scene: bool
    policy: ScenePolicyDetail


class DiagnosticFinding(BaseModel):
    finding_type: BoundedLabel
    severity: DiagnosticSeverity
    message: str = Field(max_length=240)


class InteractionContractValidateRequest(RuntimeStateResolveRequest):
    text: BoundedText
    requested_scene: BoundedScene | None = None
    interaction_contract: InteractionContract | None = None


class InteractionContractValidateResponse(BaseModel):
    diagnostic_only: Literal[True] = True
    result: DiagnosticResult
    severity: DiagnosticSeverity
    warnings: list[BoundedLabel] = Field(default_factory=list, max_length=12)
    findings: list[DiagnosticFinding] = Field(default_factory=list, max_length=12)
    contract_id: str = Field(max_length=120)
    contract_version: int
    reason_json: dict[str, Any] = Field(default_factory=dict)


class RepairSimulateRequest(RuntimeStateResolveRequest):
    miss_description: BoundedText
    corrected_substance: BoundedText
    requested_scene: BoundedScene | None = None


class RepairSimulateResponse(BaseModel):
    diagnostic_only: Literal[True] = True
    repair_text: BoundedText
    result: DiagnosticResult
    severity: DiagnosticSeverity
    warnings: list[BoundedLabel] = Field(default_factory=list, max_length=12)
    contract_id: str = Field(max_length=120)
    contract_version: int
    reason_json: dict[str, Any] = Field(default_factory=dict)
