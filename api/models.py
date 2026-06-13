from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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


class WorldStateResolveRequest(RuntimeStateResolveRequest):
    runtime_session_id: str | None = Field(default=None, max_length=120)
    active_persona_id: str | None = Field(default=None, max_length=120)
    requested_domains: list[BoundedLabel] = Field(default_factory=list, max_length=12)


class WorldStateResolveTrace(BaseModel):
    active_persona_id: str = Field(max_length=120)
    allowed_domains: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    included_claim_count: int = 0
    excluded_claim_count: int = 0
    stale_count: int = 0
    aging_count: int = 0
    expired_count: int = 0
    conflicted_count: int = 0
    confirmation_required: bool = False


class WorldStateResolveResponse(BaseModel):
    included_claims: list[WorldStateClaimView] = Field(default_factory=list)
    excluded_claim_summaries: list[WorldStateClaimSummary] = Field(default_factory=list)
    prompt_content: str | None = None
    trace: WorldStateResolveTrace


RelationshipStatus = Literal[
    "active",
    "provisional",
    "inferred",
    "needs_confirmation",
    "superseded",
    "revoked",
    "expired",
    "conflicted",
]
RelationshipMentionability = Literal[
    "mentionable",
    "use_for_routing_only",
    "use_for_filtering_only",
    "confirm_before_mentioning",
    "suppress_by_default",
    "restricted",
]
RelationshipType = Literal[
    "owns",
    "uses",
    "works_on",
    "maintains",
    "manages",
    "belongs_to",
    "member_of",
    "contains",
    "depends_on",
    "blocks",
    "documents",
    "references",
    "authored_by",
    "created_by",
    "governs",
    "responsible_for",
    "defaults_to",
    "bound_to",
    "allowed_for",
    "related_to",
    "colleague_of",
    "collaborates_with",
]
RelationshipSourceType = Literal[
    "explicit_user_confirmation",
    "trusted_config",
    "trusted_integration_metadata",
    "trusted_import_metadata",
    "tool_output",
    "repository_inspection",
    "artifact_metadata",
    "model_inference",
    "system_default",
]
RelationshipEvidenceType = Literal[
    "user_confirmation",
    "config_reference",
    "integration_metadata",
    "tool_output",
    "artifact_reference",
    "model_rationale",
]
SocialContextType = Literal[
    "communication_preference",
    "known_boundary",
    "recurring_topic",
    "relationship_reference",
    "interaction_style_signal",
    "support_preference",
    "sensitivity_marker",
    "do_not_surface_marker",
]
SocialContextUsageType = Literal[
    "diagnostic_review",
    "selection_trace",
    "suppression_trace",
]


class RelationshipEntityInput(BaseModel):
    entity_id: str = Field(max_length=120)
    entity_type: str = Field(max_length=80)
    canonical_label: str = Field(max_length=240)
    display_label: str | None = Field(default=None, max_length=240)
    domain: BoundedLabel
    sensitivity_level: WorldStateSensitivity = "medium"
    source_type: RelationshipSourceType
    source_ref: str = Field(max_length=240)
    canonical_memory_ref: str | None = Field(default=None, max_length=240)
    artifact_ref: str | None = Field(default=None, max_length=240)
    status: Literal["active", "archived"] = "active"
    archived_at: str | None = None


class RelationshipEntityUpsertRequest(RuntimeStateResolveRequest):
    entity: RelationshipEntityInput


class RelationshipEntityView(BaseModel):
    entity_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    entity_type: str = Field(max_length=80)
    canonical_label: str = Field(max_length=240)
    display_label: str | None = Field(default=None, max_length=240)
    domain: BoundedLabel
    sensitivity_level: WorldStateSensitivity
    source_type: RelationshipSourceType
    source_ref: str = Field(max_length=240)
    canonical_memory_ref: str | None = Field(default=None, max_length=240)
    artifact_ref: str | None = Field(default=None, max_length=240)
    status: Literal["active", "archived"]
    created_at: str
    updated_at: str
    archived_at: str | None = None


class RelationshipEntityResponse(BaseModel):
    entity: RelationshipEntityView


class RelationshipEdgeEvidenceInput(BaseModel):
    evidence_type: RelationshipEvidenceType
    source_ref: str = Field(max_length=240)
    summary: str | None = Field(default=None, max_length=400)
    confidence_delta: float = Field(ge=-1.0, le=1.0)


class RelationshipEdgeInput(BaseModel):
    relationship_id: str | None = Field(default=None, max_length=120)
    subject_entity_id: str = Field(max_length=120)
    relationship_type: RelationshipType
    object_entity_id: str = Field(max_length=120)
    relationship_scope: BoundedLabel
    source_type: RelationshipSourceType
    source_refs_json: list[str] = Field(default_factory=list, max_length=16)
    confidence: float = Field(ge=0.0, le=1.0)
    status: RelationshipStatus
    sensitivity_level: WorldStateSensitivity = "medium"
    mentionability: RelationshipMentionability = "mentionable"
    allowed_persona_scopes_json: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    blocked_persona_scopes_json: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    valid_from: str | None = None
    valid_until: str | None = None
    supersede_existing_relationship_id: str | None = Field(default=None, max_length=120)
    superseded_by_relationship_id: str | None = Field(default=None, max_length=120)
    revoked_at: str | None = None


class RelationshipEdgeUpsertRequest(RuntimeStateResolveRequest):
    edge: RelationshipEdgeInput
    evidence: list[RelationshipEdgeEvidenceInput] = Field(default_factory=list, max_length=8)


class RelationshipEdgeView(BaseModel):
    relationship_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    subject_entity_id: str = Field(max_length=120)
    relationship_type: RelationshipType
    object_entity_id: str = Field(max_length=120)
    relationship_scope: BoundedLabel
    source_type: RelationshipSourceType
    source_refs_json: list[str] = Field(default_factory=list, max_length=16)
    source_refs_redacted: bool = False
    confidence: float
    status: RelationshipStatus
    sensitivity_level: WorldStateSensitivity
    mentionability: RelationshipMentionability
    allowed_persona_scopes_json: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    blocked_persona_scopes_json: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    valid_from: str | None = None
    valid_until: str | None = None
    superseded_by_relationship_id: str | None = None
    created_at: str
    updated_at: str
    revoked_at: str | None = None


class RelationshipEdgeEvidenceView(BaseModel):
    evidence_id: str = Field(max_length=120)
    relationship_id: str = Field(max_length=120)
    evidence_type: RelationshipEvidenceType
    source_ref: str = Field(max_length=240)
    summary: str | None = Field(default=None, max_length=400)
    summary_redacted: bool = False
    confidence_delta: float
    created_at: str


class RelationshipEdgeResponse(BaseModel):
    relationship: RelationshipEdgeView
    evidence: list[RelationshipEdgeEvidenceView] = Field(default_factory=list)


class RelationshipEdgeConfirmRequest(RuntimeStateResolveRequest):
    relationship_id: str = Field(max_length=120)
    evidence: RelationshipEdgeEvidenceInput | None = None


class RelationshipEdgeRevokeRequest(RuntimeStateResolveRequest):
    relationship_id: str = Field(max_length=120)
    evidence: RelationshipEdgeEvidenceInput | None = None


class RelationshipGraphDiagnosticsRequest(RuntimeStateResolveRequest):
    include_restricted_details: bool = False


class RelationshipDiagnosticsResponse(BaseModel):
    entities: list[RelationshipEntityView] = Field(default_factory=list)
    relationships: list[RelationshipEdgeView] = Field(default_factory=list)
    evidence: list[RelationshipEdgeEvidenceView] = Field(default_factory=list)


class RelationshipExcludedSummary(BaseModel):
    relationship_id: str = Field(max_length=120)
    subject_entity_id: str = Field(max_length=120)
    relationship_type: RelationshipType
    object_entity_id: str = Field(max_length=120)
    relationship_scope: BoundedLabel
    status: RelationshipStatus
    mentionability: RelationshipMentionability
    sensitivity_level: WorldStateSensitivity
    confidence: float
    reason: BoundedLabel


class RelationshipSelectRequest(RuntimeStateResolveRequest):
    runtime_session_id: str | None = Field(default=None, max_length=120)
    active_persona_id: str | None = Field(default=None, max_length=120)
    requested_scopes: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    entity_ids: list[str] = Field(default_factory=list, max_length=16)
    relationship_types: list[RelationshipType] = Field(default_factory=list, max_length=16)


class RelationshipSelectTrace(BaseModel):
    relationship_edges_used: list[str] = Field(default_factory=list, max_length=64)
    relationship_edges_excluded: list[str] = Field(default_factory=list, max_length=64)
    relationship_exclusion_reasons: dict[str, str] = Field(default_factory=dict)
    relationship_context_overlay_applied: bool = False
    relationship_conflicts: list[str] = Field(default_factory=list, max_length=64)
    relationship_confirmation_required: bool = False
    selected_relationship_count: int = 0
    excluded_relationship_count: int = 0
    active_persona_id: str = Field(max_length=120)
    allowed_relationship_scopes: list[BoundedLabel] = Field(default_factory=list, max_length=16)


class RelationshipSelectResponse(BaseModel):
    selected_entities: list[RelationshipEntityView] = Field(default_factory=list)
    selected_relationships: list[RelationshipEdgeView] = Field(default_factory=list)
    excluded_relationship_summaries: list[RelationshipExcludedSummary] = Field(default_factory=list)
    prompt_content: str | None = None
    trace: RelationshipSelectTrace


class SocialContextItemInput(BaseModel):
    social_context_id: str | None = Field(default=None, max_length=120)
    context_type: SocialContextType
    summary: str = Field(max_length=400)
    source_refs_json: list[str] = Field(default_factory=list, max_length=16)
    relationship_edge_refs_json: list[str] = Field(default_factory=list, max_length=16)
    confidence: float = Field(ge=0.0, le=1.0)
    freshness: BoundedLabel
    mentionability: RelationshipMentionability = "mentionable"


class SocialContextItemUpsertRequest(RuntimeStateResolveRequest):
    item: SocialContextItemInput


class SocialContextItemView(BaseModel):
    social_context_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    context_type: SocialContextType
    summary: str = Field(max_length=400)
    source_refs_json: list[str] = Field(default_factory=list, max_length=16)
    relationship_edge_refs_json: list[str] = Field(default_factory=list, max_length=16)
    confidence: float
    freshness: BoundedLabel
    mentionability: RelationshipMentionability
    suppressed_by_default: bool = False
    suppression_reasons: list[BoundedLabel] = Field(default_factory=list, max_length=8)
    created_at: str
    updated_at: str


class SocialContextItemResponse(BaseModel):
    item: SocialContextItemView


class SocialContextUsageEventInput(BaseModel):
    event_id: str | None = Field(default=None, max_length=120)
    social_context_id: str = Field(max_length=120)
    relationship_edge_refs_json: list[str] = Field(default_factory=list, max_length=16)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    usage_type: SocialContextUsageType
    policy_decision: BoundedLabel


class SocialContextUsageEventRecordRequest(RuntimeStateResolveRequest):
    event: SocialContextUsageEventInput


class SocialContextUsageEventView(BaseModel):
    event_id: str = Field(max_length=120)
    social_context_id: str = Field(max_length=120)
    relationship_edge_refs_json: list[str] = Field(default_factory=list, max_length=16)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    usage_type: SocialContextUsageType
    policy_decision: BoundedLabel
    created_at: str


class SocialContextUsageEventResponse(BaseModel):
    event: SocialContextUsageEventView


class SocialContextDiagnosticsRequest(RuntimeStateResolveRequest):
    pass


class SocialContextDiagnosticsResponse(BaseModel):
    items: list[SocialContextItemView] = Field(default_factory=list)
    usage_events: list[SocialContextUsageEventView] = Field(default_factory=list)


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


HumanCompatibilityRiskLevel = Literal["low", "medium", "high"]
HumanCompatibilityReviewSurface = Literal[
    "persona_or_identity_selection",
    "proactive_nudges",
    "social_memory",
    "affect_and_pacing",
    "native_voice_presence",
    "idle_state_behavior",
    "emotionally_salient_responses",
    "repeated_personalization",
]
InteractionRiskType = Literal[
    "attachment_pressure",
    "dependency_framing",
    "reciprocity_claim",
    "personhood_implication",
    "hidden_influence",
    "intensity_escalation",
    "agency_erosion",
    "over_personalization",
]
InteractionRiskSeverity = Literal["low", "medium", "high"]
HumanCompatibilityReviewResult = Literal[
    "approved",
    "mitigations_required",
    "requires_human_review",
    "rejected",
]

_OPERATOR_RISK_NOTE_LABELS = {
    "attachment_pressure",
    "dependency_framing",
    "reciprocity_claim",
    "personhood_implication",
}
_OPERATOR_RISK_NOTE_PATTERN = re.compile(
    r"\b(?:attachment_pressure|dependency_framing|reciprocity_claim|personhood_implication)\b"
)


class HumanCompatibilityRiskFlagInput(BaseModel):
    flag_id: str | None = Field(default=None, max_length=120)
    risk_type: InteractionRiskType
    severity: InteractionRiskSeverity
    triggering_policy: str = Field(min_length=1, max_length=240)

    @field_validator("triggering_policy")
    @classmethod
    def _triggering_policy_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("triggering_policy_required")
        return stripped


class HumanCompatibilityReviewRequest(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    review_id: str | None = Field(default=None, max_length=120)
    feature_ref: str = Field(min_length=1, max_length=240)
    spec_ref: str = Field(default="human_compatibility_v1", min_length=1, max_length=64)
    review_surfaces: list[HumanCompatibilityReviewSurface] = Field(min_length=1, max_length=8)
    proposed_behavior_summary: BoundedText
    risk_level: HumanCompatibilityRiskLevel
    review_notes: str | None = Field(default=None, max_length=2000)
    mitigations_json: Any = None
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    interaction_risk_flags: list[HumanCompatibilityRiskFlagInput] = Field(
        default_factory=list,
        max_length=16,
    )

    @field_validator("feature_ref", "spec_ref", "proposed_behavior_summary")
    @classmethod
    def _non_blank_required_fields(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field_must_not_be_blank")
        return stripped

    @field_validator("review_notes")
    @classmethod
    def _normalize_review_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _validate_review_shape(self) -> "HumanCompatibilityReviewRequest":
        if self.risk_level == "high" and not _has_non_empty_value(self.mitigations_json):
            raise ValueError("human_compatibility_mitigations_required_for_high_risk")
        if self._requires_review_notes() and not self.review_notes:
            raise ValueError("human_compatibility_review_notes_required")
        return self

    def _requires_review_notes(self) -> bool:
        if any(flag.risk_type in _OPERATOR_RISK_NOTE_LABELS for flag in self.interaction_risk_flags):
            return True
        if self.review_notes and _OPERATOR_RISK_NOTE_PATTERN.search(self.review_notes):
            return True
        return False


class HumanCompatibilityReviewView(BaseModel):
    review_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    request_id: str = Field(max_length=120)
    feature_ref: str = Field(max_length=240)
    spec_ref: str = Field(max_length=64)
    review_surfaces: list[HumanCompatibilityReviewSurface] = Field(default_factory=list, max_length=8)
    proposed_behavior_summary: BoundedText
    risk_level: HumanCompatibilityRiskLevel
    review_result: HumanCompatibilityReviewResult
    review_notes: str | None = Field(default=None, max_length=2000)
    mitigations_json: Any = None
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    principles_checked: list[BoundedLabel] = Field(default_factory=list, min_length=6, max_length=6)
    mitigations_required: bool = False
    created_at: str


class HumanCompatibilityRiskFlagView(BaseModel):
    flag_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    risk_type: InteractionRiskType
    severity: InteractionRiskSeverity
    triggering_policy: str = Field(max_length=240)
    created_at: str


class HumanCompatibilityReviewResponse(BaseModel):
    review_id: str = Field(max_length=120)
    review_result: HumanCompatibilityReviewResult
    principles_checked: list[BoundedLabel] = Field(default_factory=list, min_length=6, max_length=6)
    mitigations_required: bool
    flags_recorded: int
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class HumanCompatibilityDiagnosticsRequest(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    feature_ref: str | None = Field(default=None, max_length=240)
    runtime_turn_id: str | None = Field(default=None, max_length=120)

    @field_validator("feature_ref")
    @classmethod
    def _normalize_feature_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class HumanCompatibilityDiagnosticsResponse(BaseModel):
    reviews: list[HumanCompatibilityReviewView] = Field(default_factory=list)
    interaction_risk_flags: list[HumanCompatibilityRiskFlagView] = Field(default_factory=list)


def _has_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict | list | tuple | set):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return True
