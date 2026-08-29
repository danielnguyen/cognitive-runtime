from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

AttentionStatus = Literal["active", "paused", "resolved"]
BoundedLabel = Annotated[str, Field(min_length=1, max_length=64)]
BoundedSurface = Annotated[str, Field(max_length=64)]
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
    surface: str = Field(default="unknown", min_length=1, max_length=64)


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
    "interaction_governance_evaluated",
    "persona_containment_evaluated",
    "restraint_evaluated",
    "situated_presence_evaluated",
    "memory_hygiene_evaluated",
    "privacy_context_evaluated",
    "world_state_verification_evaluated",
    "capability_authorization_evaluated",
    "confirmation_challenge_evaluated",
    "action_summary_recorded",
    "claim_calibration_evaluated",
    "claim_support_evaluated",
    "evidence_shape_derived",
    "evidence_plan_compiled",
    "evidence_sufficiency_evaluated",
    "evidence_next_step_selected",
]
RuntimeThreadState = Literal["idle", "active", "contended", "unavailable"]


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
    model_config = ConfigDict(extra="forbid")

    input_message_id: str | None = Field(default=None, max_length=120)
    intent_class: str | None = Field(default=None, max_length=64)
    timing_policy: str | None = Field(default=None, max_length=64)
    restraint_policy: str | None = Field(default=None, max_length=64)
    continuation_state: str | None = Field(default=None, max_length=64)
    expected_thread_revision: int | None = Field(default=None, ge=0, strict=True)


class RuntimeThreadResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)


class RuntimeThreadProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    state: RuntimeThreadState
    revision: int = Field(ge=0)
    active_runtime_session_id: str | None = Field(default=None, max_length=120)
    active_runtime_turn_id: str | None = Field(default=None, max_length=120)
    active_surface: BoundedSurface | None = None
    participating_surfaces: list[BoundedSurface] = Field(default_factory=list, max_length=32)
    participating_session_count: int = Field(ge=0)
    last_activity_at: str
    created_at: str
    updated_at: str


ContinuationCandidateLifecycle = Literal["open", "closed", "superseded"]
ContinuationSelectionOutcome = Literal[
    "resume",
    "create_new",
    "clarify",
    "wait",
    "decline",
]
ContinuationTimingPolicy = Literal[
    "answer_now",
    "ask_clarifying_question",
    "pause_or_wait",
    "resume_previous_thread",
    "close_turn",
]
ContinuationSelectionReason = Literal[
    "candidate_set_incomplete",
    "no_candidates",
    "one_eligible_candidate",
    "multiple_eligible_candidates",
    "active_thread_present",
    "contended_thread_present",
    "unavailable_thread_present",
    "runtime_state_missing",
    "runtime_state_inconsistent",
    "runtime_session_missing",
    "candidate_stale",
    "candidate_not_open",
    "no_eligible_candidates",
]


class ContinuationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    conversation_id: str = Field(min_length=1, max_length=120)
    lifecycle_state: ContinuationCandidateLifecycle
    durable_updated_at: datetime

    @field_validator("durable_updated_at", mode="before")
    @classmethod
    def validate_datetime_input(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("durable_updated_at_invalid") from exc
        raise ValueError("durable_updated_at_invalid")

    @field_validator("durable_updated_at")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("durable_updated_at_timezone_required")
        return value


class ContinuationSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    request_id: str = Field(min_length=1, max_length=120)
    owner_id: str = Field(min_length=1, max_length=120)
    surface: str = Field(min_length=1, max_length=64)
    candidate_set_complete: bool = Field(strict=True)
    stale_after_seconds: int = Field(ge=60, le=86400, strict=True)
    candidates: list[ContinuationCandidate] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_unique_candidates(self) -> "ContinuationSelectionRequest":
        conversation_ids = [candidate.conversation_id for candidate in self.candidates]
        if len(conversation_ids) != len(set(conversation_ids)):
            raise ValueError("duplicate_conversation_id")
        return self


class ContinuationSelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    outcome: ContinuationSelectionOutcome
    timing_policy: ContinuationTimingPolicy
    selected_conversation_id: str | None = Field(default=None, min_length=1, max_length=120)
    selected_thread_revision: int | None = Field(default=None, ge=0, strict=True)
    candidate_count: int = Field(ge=0, strict=True)
    eligible_candidate_count: int = Field(ge=0, strict=True)
    reason_codes: list[ContinuationSelectionReason] = Field(min_length=1, max_length=8)
    policy_version: Literal["continuation-selection.v1"] = "continuation-selection.v1"

    @model_validator(mode="after")
    def validate_coherence(self) -> "ContinuationSelectionResult":
        expected_timing = {
            "resume": "resume_previous_thread",
            "create_new": "answer_now",
            "clarify": "ask_clarifying_question",
            "wait": "pause_or_wait",
            "decline": "close_turn",
        }
        if self.timing_policy != expected_timing[self.outcome]:
            raise ValueError("continuation_timing_policy_inconsistent")
        if self.eligible_candidate_count > self.candidate_count:
            raise ValueError("continuation_candidate_counts_inconsistent")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("continuation_reason_codes_duplicate")
        if self.outcome == "resume":
            if (
                self.selected_conversation_id is None
                or self.selected_thread_revision is None
                or self.eligible_candidate_count != 1
                or self.reason_codes != ["one_eligible_candidate"]
            ):
                raise ValueError("continuation_resume_result_inconsistent")
        elif (
            self.selected_conversation_id is not None
            or self.selected_thread_revision is not None
        ):
            raise ValueError("continuation_nonresume_selection_forbidden")
        return self


class ContinuationSelectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["runtime-continuation-selection.v1"] = (
        "runtime-continuation-selection.v1"
    )
    request_id: str = Field(min_length=1, max_length=120)
    owner_id: str = Field(min_length=1, max_length=120)
    surface: str = Field(min_length=1, max_length=64)
    result: ContinuationSelectionResult


RetirementReservationOutcome = Literal["reserved", "wait", "decline"]
RetirementReservationReason = Literal[
    "safe_idle_retirement_reserved",
    "existing_retirement_reservation",
    "candidate_not_open",
    "durable_activity_not_over_horizon",
    "runtime_activity_not_over_horizon",
    "runtime_state_missing",
    "runtime_state_inconsistent",
    "runtime_thread_active",
    "runtime_thread_contended",
    "runtime_thread_unavailable",
]


def _parse_retirement_datetime(value: Any) -> Any:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("retirement_timestamp_invalid") from exc
    raise ValueError("retirement_timestamp_invalid")


def _require_retirement_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("retirement_timestamp_timezone_required")
    return value


class RetirementReservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    request_id: str = Field(min_length=1, max_length=120)
    owner_id: str = Field(min_length=1, max_length=120)
    conversation_id: str = Field(min_length=1, max_length=120)
    lifecycle_state: ContinuationCandidateLifecycle
    durable_updated_at: datetime
    retirement_before: datetime

    @field_validator("durable_updated_at", "retirement_before", mode="before")
    @classmethod
    def validate_datetime_input(cls, value: Any) -> Any:
        return _parse_retirement_datetime(value)

    @field_validator("durable_updated_at", "retirement_before")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        return _require_retirement_timezone(value)


class RetirementReservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    outcome: RetirementReservationOutcome
    reservation_id: str | None = Field(default=None, min_length=1, max_length=120)
    reserved_thread_revision: int | None = Field(default=None, ge=0, strict=True)
    reserved_durable_updated_at: datetime | None = None
    reason_codes: list[RetirementReservationReason] = Field(min_length=1, max_length=1)
    policy_version: Literal["conversation-retirement-safety.v1"] = (
        "conversation-retirement-safety.v1"
    )

    @field_validator("reserved_durable_updated_at", mode="before")
    @classmethod
    def validate_datetime_input(cls, value: Any) -> Any:
        if value is None:
            return value
        return _parse_retirement_datetime(value)

    @field_validator("reserved_durable_updated_at")
    @classmethod
    def validate_timezone(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        return _require_retirement_timezone(value)

    @model_validator(mode="after")
    def validate_coherence(self) -> "RetirementReservationResult":
        reserved_reasons = {
            "safe_idle_retirement_reserved",
            "existing_retirement_reservation",
        }
        reason = self.reason_codes[0]
        if self.outcome == "reserved":
            if (
                self.reservation_id is None
                or self.reserved_thread_revision is None
                or self.reserved_durable_updated_at is None
                or reason not in reserved_reasons
            ):
                raise ValueError("retirement_reservation_result_inconsistent")
        elif (
            self.reservation_id is not None
            or self.reserved_thread_revision is not None
            or self.reserved_durable_updated_at is not None
            or reason in reserved_reasons
        ):
            raise ValueError("retirement_nonreserved_result_inconsistent")
        if self.outcome == "wait" and reason != "runtime_thread_active":
            raise ValueError("retirement_wait_result_inconsistent")
        if self.outcome == "decline" and reason == "runtime_thread_active":
            raise ValueError("retirement_decline_result_inconsistent")
        return self


class RetirementReservationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["runtime-retirement-reservation.v1"] = (
        "runtime-retirement-reservation.v1"
    )
    request_id: str = Field(min_length=1, max_length=120)
    owner_id: str = Field(min_length=1, max_length=120)
    conversation_id: str = Field(min_length=1, max_length=120)
    result: RetirementReservationResult


class _RetirementReservationMutationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    request_id: str = Field(min_length=1, max_length=120)
    owner_id: str = Field(min_length=1, max_length=120)
    conversation_id: str = Field(min_length=1, max_length=120)
    reservation_id: str = Field(min_length=1, max_length=120)
    reserved_thread_revision: int = Field(ge=0, strict=True)


class RetirementReservationCancelRequest(_RetirementReservationMutationRequest):
    pass


class RetirementReservationFinalizeRequest(_RetirementReservationMutationRequest):
    pass


class RetirementReservationCancelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["runtime-retirement-cancellation.v1"] = (
        "runtime-retirement-cancellation.v1"
    )
    request_id: str = Field(min_length=1, max_length=120)
    owner_id: str = Field(min_length=1, max_length=120)
    conversation_id: str = Field(min_length=1, max_length=120)
    reservation_id: str = Field(min_length=1, max_length=120)
    thread_revision: int = Field(ge=0, strict=True)
    outcome: Literal["cancelled"] = "cancelled"


class RetirementReservationFinalizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["runtime-retirement-finalization.v1"] = (
        "runtime-retirement-finalization.v1"
    )
    request_id: str = Field(min_length=1, max_length=120)
    owner_id: str = Field(min_length=1, max_length=120)
    conversation_id: str = Field(min_length=1, max_length=120)
    reservation_id: str = Field(min_length=1, max_length=120)
    previous_thread_revision: int = Field(ge=0, strict=True)
    fenced_thread_revision: int = Field(ge=1, strict=True)
    outcome: Literal["finalized"] = "finalized"

    @model_validator(mode="after")
    def validate_revision_fence(self) -> "RetirementReservationFinalizeResponse":
        if self.fenced_thread_revision != self.previous_thread_revision + 1:
            raise ValueError("retirement_finalization_revision_inconsistent")
        return self


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


InteractionGovernanceKind = Literal[
    "command",
    "question",
    "brainstorm",
    "joke_or_playful",
    "vent_or_expression",
    "mistake_or_failure_report",
    "tense_debugging",
    "high_impact_decision",
    "ambiguous",
]
InteractionGovernanceTension = Literal["low", "medium", "high"]
InteractionGovernancePrivacySensitivity = Literal["normal", "private", "sensitive"]
InteractionGovernanceResponsePosture = Literal[
    "direct",
    "supportive",
    "tactical",
    "brief",
    "reflective",
    "playful",
    "silent_or_minimal",
]
HistoryFollowupCandidateSource = Literal["deterministic", "classifier"]
HistoryFollowupIntent = Literal[
    "not_history_followup",
    "support_explanation",
    "acquisition_checked",
    "acquisition_coverage",
    "acquisition_gaps",
    "new_verification_request",
    "ambiguous_history_followup",
]
HistoryFollowupTargetMode = Literal["immediate_previous", "explicit_reference"]
HistoryFollowupPolicyStatus = Literal[
    "not_applicable",
    "accepted",
    "clarification_required",
    "rejected",
    "explicit_reference",
]
HistoryFollowupExplanationKind = Literal["support", "acquisition"]
HistoryFollowupAcquisitionQuestion = Literal["checked", "coverage", "gaps"]
HistoryFollowupConfidenceBand = Literal[
    "not_applicable",
    "low",
    "medium",
    "high",
]
HistoryFollowupReasonCode = Literal[
    "no_candidate",
    "not_history_candidate",
    "ambiguous_candidate",
    "deterministic_candidate_accepted",
    "classifier_candidate_accepted",
    "classifier_confidence_requires_clarification",
    "classifier_confidence_rejected",
    "explicit_reference_routed",
]
MemoryHygieneFreshnessState = Literal[
    "active",
    "parked",
    "stale",
    "superseded",
    "corrected",
    "forgotten_or_demoted",
    "unknown_freshness",
]
MemoryHygieneItemRefType = Literal["message", "derived_text"]
MemoryHygieneFraming = Literal[
    "current",
    "parked_or_historical",
    "stale_or_unverified",
    "corrected_replacement",
    "omit",
    "unknown_or_unverified",
]


class MemoryHygieneItemRef(BaseModel):
    ref_type: MemoryHygieneItemRefType
    ref_id: str = Field(max_length=120)


class MemoryHygieneItemInput(BaseModel):
    item_ref: MemoryHygieneItemRef
    memory_id: str | None = Field(default=None, max_length=120)
    freshness_state: MemoryHygieneFreshnessState = "unknown_freshness"
    last_verified_at: str | None = Field(default=None, max_length=64)
    source_kind: BoundedLabel | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    supersedes: str | None = Field(default=None, max_length=120)
    superseded_by: str | None = Field(default=None, max_length=120)

    @field_validator("freshness_state", mode="before")
    @classmethod
    def normalize_freshness_state(cls, value: Any) -> str:
        allowed = {
            "active",
            "parked",
            "stale",
            "superseded",
            "corrected",
            "forgotten_or_demoted",
            "unknown_freshness",
        }
        if not isinstance(value, str):
            return "unknown_freshness"
        normalized = value.strip().lower()
        return normalized if normalized in allowed else "unknown_freshness"


class MemoryHygieneDecision(BaseModel):
    item_ref: MemoryHygieneItemRef
    freshness_state: MemoryHygieneFreshnessState
    use_allowed: bool
    mention_as_current_allowed: bool
    framing: MemoryHygieneFraming
    reason_codes: list[BoundedLabel] = Field(default_factory=list, max_length=8)


class MemoryHygieneAggregate(BaseModel):
    evaluated_item_count: int = Field(ge=0)
    usable_item_count: int = Field(ge=0)
    current_mention_allowed_count: int = Field(ge=0)
    restricted_or_omitted_count: int = Field(ge=0)
    counts_by_freshness_state: dict[MemoryHygieneFreshnessState, int] = Field(default_factory=dict)
    reason_codes: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    supersession_handling_applied: bool = False


class MemoryHygieneResult(BaseModel):
    decisions: list[MemoryHygieneDecision] = Field(default_factory=list, max_length=64)
    aggregate: MemoryHygieneAggregate


class MemoryHygieneEvaluateRequest(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    runtime_session_id: str | None = Field(default=None, max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    items: list[MemoryHygieneItemInput] = Field(default_factory=list, max_length=64)


class MemoryHygieneEvaluateResponse(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    runtime_session_id: str = Field(max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    result: MemoryHygieneResult


ClaimCalibrationIdentifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
ClaimEvidenceRefType = Literal[
    "message",
    "derived_text",
    "artifact",
    "external_source",
    "world_state_claim",
    "tool_output",
    "integration_event",
]
ClaimEvidenceSupportKind = Literal[
    "direct",
    "corroborating",
    "contextual",
    "contradictory",
]
ClaimEvidenceAuthority = Literal[
    "peer_reviewed_evidence",
    "clinical_guidance",
    "manufacturer_guidance",
    "tool_output",
    "trusted_integration",
    "user_report",
    "runtime_inference",
    "speculation",
    "unknown",
]
ClaimEvidenceFreshnessState = Literal[
    "active",
    "stale",
    "superseded",
    "corrected",
    "unknown_freshness",
    "not_applicable",
]
ClaimClass = Literal[
    "verified_fact",
    "source_backed_fact",
    "manufacturer_guidance",
    "expert_consensus",
    "runtime_inference",
    "speculation",
    "unknown",
]
ClaimCalibrationStatus = Literal["supported", "limited", "unsupported"]
ClaimEvidenceStrength = Literal["strong", "moderate", "weak", "none"]
ClaimConfidence = Literal["high", "medium", "low", "unknown"]
ClaimFreshnessSummary = Literal["current", "mixed", "stale", "unknown", "not_applicable"]
ClaimLimitationCode = Literal[
    "no_supporting_evidence",
    "context_only",
    "low_authority_evidence",
    "stale_evidence",
    "unknown_freshness",
    "superseded_or_corrected_evidence",
    "contradictory_evidence",
    "single_source",
    "inference_dominant",
    "speculation_only",
]


class ClaimEvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref_type: ClaimEvidenceRefType
    ref_id: ClaimCalibrationIdentifier
    owner_id: ClaimCalibrationIdentifier
    conversation_id: ClaimCalibrationIdentifier | None = None
    support_kind: ClaimEvidenceSupportKind
    authority: ClaimEvidenceAuthority
    freshness_state: ClaimEvidenceFreshnessState


class ClaimCalibrationEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: ClaimCalibrationIdentifier
    owner_id: ClaimCalibrationIdentifier
    conversation_id: ClaimCalibrationIdentifier
    surface: Annotated[str, Field(min_length=1, max_length=64)]
    runtime_session_id: ClaimCalibrationIdentifier
    runtime_turn_id: ClaimCalibrationIdentifier
    claim_anchor: Annotated[str, Field(min_length=1, max_length=500)]
    evidence_references: list[ClaimEvidenceReference] = Field(
        default_factory=list,
        max_length=16,
    )

    @field_validator("claim_anchor", mode="before")
    @classmethod
    def normalize_claim_anchor(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return " ".join(value.split())

    @model_validator(mode="after")
    def validate_evidence_scope(self) -> ClaimCalibrationEvaluateRequest:
        seen: set[tuple[str, str]] = set()
        for reference in self.evidence_references:
            if reference.owner_id != self.owner_id:
                raise ValueError("evidence_owner_mismatch")
            if (
                reference.conversation_id is not None
                and reference.conversation_id != self.conversation_id
            ):
                raise ValueError("evidence_conversation_mismatch")
            identity = (reference.ref_type, reference.ref_id)
            if identity in seen:
                raise ValueError("duplicate_evidence_reference")
            seen.add(identity)
        return self


class ClaimCalibrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: ClaimCalibrationIdentifier
    claim_anchor: Annotated[str, Field(min_length=1, max_length=500)]
    claim_anchor_digest: Annotated[str, Field(min_length=71, max_length=71)]
    claim_class: ClaimClass
    calibration_status: ClaimCalibrationStatus
    evidence_strength: ClaimEvidenceStrength
    confidence: ClaimConfidence
    strongest_authority: ClaimEvidenceAuthority
    freshness_summary: ClaimFreshnessSummary
    uncertainty_disclosure_required: bool
    validated_evidence_references: list[ClaimEvidenceReference] = Field(
        default_factory=list,
        max_length=16,
    )
    limitation_codes: list[ClaimLimitationCode] = Field(default_factory=list, max_length=10)
    user_safe_summary: Annotated[str, Field(min_length=1, max_length=500)]


class ClaimCalibrationEvaluateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: ClaimCalibrationIdentifier
    owner_id: ClaimCalibrationIdentifier
    conversation_id: ClaimCalibrationIdentifier
    surface: Annotated[str, Field(min_length=1, max_length=64)]
    runtime_session_id: ClaimCalibrationIdentifier
    runtime_turn_id: ClaimCalibrationIdentifier
    result: ClaimCalibrationResult


ClaimSupportSourceAuthority = Literal["established", "limited", "unknown"]
ClaimSupportFreshness = Literal["current", "stale", "unknown", "not_applicable"]
ClaimSupportMaterialRole = Literal["support", "counterevidence", "neutral"]
ClaimSupportInputBasis = Literal["system_established", "model_interpreted"]
ClaimSupportScopeBasis = Literal["declared_scope", "supplied_evidence"]
ClaimSupportCalibrationStatus = Literal["supported", "limited", "unsupported"]
ClaimSupportConclusionDisposition = Literal["allowed", "qualified", "withheld"]
ClaimSupportLimitationCode = Literal[
    "no_supporting_evidence",
    "limited_source_authority",
    "unknown_source_authority",
    "stale_evidence",
    "unknown_freshness",
    "complete_scope_not_established",
    "material_acquisition_limited",
    "material_evidence_omitted",
    "material_counterevidence_present",
    "material_counterevidence_excluded",
    "material_counterevidence_misclassified",
    "material_support_excluded",
    "interpretation_dependent_derivation",
    "privacy_constraint",
    "consequence_constraint",
    "declared_counterevidence",
    "material_exclusion",
]
ClaimSupportCanonicalNumber = Annotated[str, Field(min_length=1, max_length=128)]
ClaimSupportDigest = Annotated[
    str,
    Field(pattern=r"^sha256:[0-9a-f]{64}$", min_length=71, max_length=71),
]


class ClaimSupportEvidenceAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref_id: ClaimCalibrationIdentifier
    owner_id: ClaimCalibrationIdentifier
    conversation_id: ClaimCalibrationIdentifier
    source_authority: ClaimSupportSourceAuthority
    freshness: ClaimSupportFreshness
    material_disclosure_required: bool = False
    material_role: ClaimSupportMaterialRole = "neutral"


class ClaimSupportDerivationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    derivation_id: ClaimCalibrationIdentifier
    owner_id: ClaimCalibrationIdentifier
    conversation_id: ClaimCalibrationIdentifier
    runtime_session_id: ClaimCalibrationIdentifier
    runtime_turn_id: ClaimCalibrationIdentifier
    operation: Literal["divide", "mean"]
    canonical_inputs: list[ClaimSupportCanonicalNumber] = Field(
        min_length=1,
        max_length=32,
    )
    canonical_result: ClaimSupportCanonicalNumber
    execution_status: Literal["executed"]
    execution_digest: ClaimSupportDigest
    executor_version: ClaimCalibrationIdentifier
    supporting_evidence_ref_ids: list[ClaimCalibrationIdentifier] = Field(
        default_factory=list,
        max_length=16,
    )
    input_basis: ClaimSupportInputBasis

    @model_validator(mode="after")
    def validate_supporting_references(self) -> ClaimSupportDerivationRecord:
        if len(self.supporting_evidence_ref_ids) != len(
            set(self.supporting_evidence_ref_ids)
        ):
            raise ValueError("duplicate_derivation_evidence_reference")
        if self.input_basis == "model_interpreted" and not self.supporting_evidence_ref_ids:
            raise ValueError("interpreted_derivation_evidence_required")
        return self


class ClaimSupportAuthorityContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_id: ClaimCalibrationIdentifier
    conversation_id: ClaimCalibrationIdentifier
    surface: Annotated[str, Field(min_length=1, max_length=64)]
    runtime_session_id: ClaimCalibrationIdentifier
    runtime_turn_id: ClaimCalibrationIdentifier
    evidence_references: list[ClaimSupportEvidenceAuthority] = Field(
        default_factory=list,
        max_length=16,
    )
    claim_scope_basis: ClaimSupportScopeBasis = "declared_scope"
    complete_declared_scope_required: bool = False
    complete_declared_scope_established: bool | None = None
    material_acquisition_limited: bool = False
    privacy_policy_allows_claim: bool = True
    consequence_policy_allows_claim: bool = True
    executed_derivations: list[ClaimSupportDerivationRecord] = Field(
        default_factory=list,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_authority_associations(self) -> ClaimSupportAuthorityContext:
        evidence_ids: set[str] = set()
        for reference in self.evidence_references:
            if reference.owner_id != self.owner_id:
                raise ValueError("evidence_owner_mismatch")
            if reference.conversation_id != self.conversation_id:
                raise ValueError("evidence_conversation_mismatch")
            if reference.ref_id in evidence_ids:
                raise ValueError("duplicate_evidence_reference")
            evidence_ids.add(reference.ref_id)

        derivation_ids: set[str] = set()
        for derivation in self.executed_derivations:
            if derivation.owner_id != self.owner_id:
                raise ValueError("derivation_owner_mismatch")
            if derivation.conversation_id != self.conversation_id:
                raise ValueError("derivation_conversation_mismatch")
            if derivation.runtime_session_id != self.runtime_session_id:
                raise ValueError("derivation_session_mismatch")
            if derivation.runtime_turn_id != self.runtime_turn_id:
                raise ValueError("derivation_turn_mismatch")
            if derivation.derivation_id in derivation_ids:
                raise ValueError("duplicate_derivation_reference")
            derivation_ids.add(derivation.derivation_id)
            if not set(derivation.supporting_evidence_ref_ids) <= evidence_ids:
                raise ValueError("derivation_evidence_reference_not_authorized")
        return self


class ClaimSupportMaterialExclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_ref_id: ClaimCalibrationIdentifier
    reason: Annotated[str, Field(min_length=1, max_length=240)]

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return " ".join(value.split())


class ClaimSupportProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposed_claim: Annotated[str, Field(min_length=1, max_length=1000)]
    supporting_evidence_ref_ids: list[ClaimCalibrationIdentifier] = Field(
        default_factory=list,
        max_length=16,
    )
    counterevidence_ref_ids: list[ClaimCalibrationIdentifier] = Field(
        default_factory=list,
        max_length=16,
    )
    material_exclusions: list[ClaimSupportMaterialExclusion] = Field(
        default_factory=list,
        max_length=16,
    )
    executed_derivation_ref_ids: list[ClaimCalibrationIdentifier] = Field(
        default_factory=list,
        max_length=16,
    )

    @field_validator("proposed_claim", mode="before")
    @classmethod
    def normalize_proposed_claim(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return " ".join(value.split())

    @model_validator(mode="after")
    def validate_reference_uniqueness(self) -> ClaimSupportProposal:
        collections = (
            self.supporting_evidence_ref_ids,
            self.counterevidence_ref_ids,
            self.executed_derivation_ref_ids,
        )
        if any(len(items) != len(set(items)) for items in collections):
            raise ValueError("duplicate_proposal_reference")
        exclusion_ids = [item.evidence_ref_id for item in self.material_exclusions]
        if len(exclusion_ids) != len(set(exclusion_ids)):
            raise ValueError("duplicate_material_exclusion")
        support_ids = set(self.supporting_evidence_ref_ids)
        counter_ids = set(self.counterevidence_ref_ids)
        exclusion_id_set = set(exclusion_ids)
        if support_ids & counter_ids or counter_ids & exclusion_id_set:
            raise ValueError("conflicting_proposal_evidence_role")
        return self


class ClaimSupportEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: ClaimCalibrationIdentifier
    authority_context: ClaimSupportAuthorityContext
    proposal: ClaimSupportProposal

    @model_validator(mode="after")
    def validate_proposal_references(self) -> ClaimSupportEvaluateRequest:
        evidence_ids = {
            item.ref_id for item in self.authority_context.evidence_references
        }
        proposal_evidence_ids = (
            set(self.proposal.supporting_evidence_ref_ids)
            | set(self.proposal.counterevidence_ref_ids)
            | {item.evidence_ref_id for item in self.proposal.material_exclusions}
        )
        if not proposal_evidence_ids <= evidence_ids:
            raise ValueError("proposal_evidence_reference_not_authorized")
        derivation_ids = {
            item.derivation_id for item in self.authority_context.executed_derivations
        }
        if not set(self.proposal.executed_derivation_ref_ids) <= derivation_ids:
            raise ValueError("proposal_derivation_reference_not_executed")
        return self


class ClaimSupportEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: ClaimCalibrationIdentifier
    claim_digest: ClaimSupportDigest
    calibration_status: ClaimSupportCalibrationStatus
    conclusion_disposition: ClaimSupportConclusionDisposition
    qualification_required: bool
    limitation_codes: list[ClaimSupportLimitationCode] = Field(
        default_factory=list,
        max_length=17,
    )
    validated_supporting_evidence_ref_ids: list[ClaimCalibrationIdentifier] = Field(
        default_factory=list,
        max_length=16,
    )
    validated_counterevidence_ref_ids: list[ClaimCalibrationIdentifier] = Field(
        default_factory=list,
        max_length=16,
    )
    validated_material_exclusions: list[ClaimSupportMaterialExclusion] = Field(
        default_factory=list,
        max_length=16,
    )
    validated_executed_derivation_ref_ids: list[ClaimCalibrationIdentifier] = Field(
        default_factory=list,
        max_length=16,
    )
    user_safe_summary: Annotated[str, Field(min_length=1, max_length=500)]


class ClaimSupportEvaluateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: ClaimCalibrationIdentifier
    owner_id: ClaimCalibrationIdentifier
    conversation_id: ClaimCalibrationIdentifier
    surface: Annotated[str, Field(min_length=1, max_length=64)]
    runtime_session_id: ClaimCalibrationIdentifier
    runtime_turn_id: ClaimCalibrationIdentifier
    result: ClaimSupportEvaluationResult


EvidenceSufficiencyIdentifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
EvidenceSufficiencySurface = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
EvidenceTaskShape = Literal[
    "targeted_lookup",
    "aggregate",
    "bounded_exhaustive_review",
    "cross_source_comparison",
    "contradiction_review",
    "absence_or_coverage_check",
    "historical_reconstruction",
    "recommendation_or_decision_support",
]
EvidenceInputKind = Literal[
    "memory",
    "artifact",
    "external_source",
    "tool_output",
    "integration_record",
    "world_state",
]
EvidenceShapeDerivationStatus = Literal["derived", "not_applicable", "ambiguous"]
EvidenceShapeReasonCode = Literal[
    "source_context_present",
    "external_verification_required",
    "freshness_sensitive",
    "high_stakes_accuracy_required",
    "explicit_evidence_language",
    "targeted_lookup_derived",
    "exhaustive_scope_requested",
    "comparison_requested",
    "contradiction_requested",
    "absence_scope_requested",
    "historical_reconstruction_requested",
    "decision_support_requested",
    "prior_shape_inherited",
    "ordinary_chat_without_material_evidence_scope",
    "non_evidence_interaction",
    "ambiguous_interaction_without_shape_signal",
    "multiple_incompatible_shapes",
    "semantic_operation_hint",
    "semantic_operation_unsupported",
]
SourceInventoryStatus = Literal["complete", "partial", "unknown", "unavailable"]
SourceCapability = Literal["profile", "search", "fetch", "context"]
SourceAvailability = Literal["available", "unavailable", "disabled", "unknown"]
SourceAuthorityRole = Literal["authoritative", "supplemental", "unknown"]
SourceMatchStatus = Literal[
    "matched",
    "no_match",
    "ambiguous",
    "inventory_unavailable",
]
SourceMatchReasonCode = Literal[
    "source_id_match",
    "display_name_match",
    "domain_tag_match",
    "scope_reference_match",
    "multiple_explicit_source_matches",
    "multiple_possible_source_matches",
    "no_source_specific_match",
    "generic_source_signal_rejected",
    "inventory_partial",
    "inventory_unknown",
    "inventory_unavailable",
    "semantic_candidate_validated",
    "semantic_candidates_ambiguous",
    "semantic_no_match",
]
SemanticInterpretationStatus = Literal["resolved", "ambiguous", "no_match"]
SemanticOperationHint = Literal[
    "lookup",
    "latest",
    "comparison",
    "exhaustive_review",
    "contradiction_review",
    "absence_check",
    "historical_reconstruction",
    "decision_support",
    "aggregate",
    "unknown",
]
AggregateFunction = Literal[
    "median",
    "mean",
    "count",
    "sum",
    "minimum",
    "maximum",
]
DiscoverableContentField = Annotated[
    str,
    Field(strict=True, min_length=1, max_length=120),
]
AggregateFieldName = Annotated[
    str,
    Field(strict=True, min_length=1, max_length=120),
]


def _validate_discoverable_content_field(value: str) -> str:
    if not value.strip():
        raise ValueError("discoverable_content_field_blank")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("discoverable_content_field_has_control_character")
    return value


def _validate_aggregate_field_name(value: str) -> str:
    if value != value.strip():
        raise ValueError("aggregate_field_name_has_outer_whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("aggregate_field_name_has_control_character")
    return value


class AggregateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: AggregateFunction
    field_name: AggregateFieldName

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, value: AggregateFieldName) -> AggregateFieldName:
        return _validate_aggregate_field_name(value)


class SourceDiscoveryScopeReferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: EvidenceSufficiencyIdentifier | None = None
    version: EvidenceSufficiencyIdentifier | None = None
    domain: EvidenceSufficiencyIdentifier | None = None
    project: EvidenceSufficiencyIdentifier | None = None

    @model_validator(mode="after")
    def validate_supplied_values(self) -> SourceDiscoveryScopeReferences:
        if not self.model_fields_set:
            raise ValueError("source_discovery_scope_refs_empty")
        if any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("source_discovery_scope_ref_null")
        return self


class SourceDiscoveryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: EvidenceSufficiencyIdentifier
    display_name: Annotated[str, Field(min_length=1, max_length=240)]
    connector: EvidenceSufficiencyIdentifier
    domain_tags: list[EvidenceSufficiencyIdentifier] = Field(max_length=8)
    scope_refs: SourceDiscoveryScopeReferences | None = None
    content_fields: list[DiscoverableContentField] | None = Field(
        default=None,
        max_length=24,
    )
    capabilities: list[SourceCapability] = Field(max_length=4)
    availability: SourceAvailability
    authority_role: SourceAuthorityRole

    @model_serializer(mode="wrap")
    def omit_absent_content_fields(self, handler):
        serialized = handler(self)
        if self.content_fields is None:
            serialized.pop("content_fields", None)
        return serialized

    @field_validator("content_fields")
    @classmethod
    def validate_content_fields(
        cls,
        value: list[DiscoverableContentField] | None,
    ) -> list[DiscoverableContentField] | None:
        if value is None:
            return value
        validated = [_validate_discoverable_content_field(field) for field in value]
        if len(set(validated)) != len(validated):
            raise ValueError("duplicate_source_discovery_content_field")
        if validated != sorted(validated):
            raise ValueError("source_discovery_content_fields_not_sorted")
        return validated

    @model_validator(mode="after")
    def validate_unique_values(self) -> SourceDiscoveryEntry:
        if len(set(self.domain_tags)) != len(self.domain_tags):
            raise ValueError("duplicate_source_discovery_domain_tag")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("duplicate_source_discovery_capability")
        if "scope_refs" in self.model_fields_set and self.scope_refs is None:
            raise ValueError("source_discovery_scope_refs_null")
        if "content_fields" in self.model_fields_set and self.content_fields is None:
            raise ValueError("source_discovery_content_fields_null")
        return self


class SourceDiscoveryContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inventory_status: SourceInventoryStatus
    sources: list[SourceDiscoveryEntry] = Field(max_length=32)

    @model_validator(mode="after")
    def validate_unique_sources(self) -> SourceDiscoveryContext:
        source_ids = [source.source_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("duplicate_source_discovery_source_id")
        return self


class SemanticEvidenceAdvisory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interpretation_status: SemanticInterpretationStatus
    operation_hint: SemanticOperationHint
    candidate_source_ids: list[EvidenceSufficiencyIdentifier] = Field(max_length=3)
    aggregate_function: AggregateFunction | None = None
    aggregate_field_name: AggregateFieldName | None = None

    @model_serializer(mode="wrap")
    def omit_absent_aggregate_fields(self, handler):
        serialized = handler(self)
        if self.aggregate_function is None:
            serialized.pop("aggregate_function", None)
        if self.aggregate_field_name is None:
            serialized.pop("aggregate_field_name", None)
        return serialized

    @field_validator("aggregate_field_name")
    @classmethod
    def validate_aggregate_field_name(
        cls,
        value: AggregateFieldName | None,
    ) -> AggregateFieldName | None:
        if value is None:
            return value
        return _validate_aggregate_field_name(value)

    @model_validator(mode="after")
    def validate_candidate_set(self) -> SemanticEvidenceAdvisory:
        function_supplied = "aggregate_function" in self.model_fields_set
        field_supplied = "aggregate_field_name" in self.model_fields_set
        if function_supplied and self.aggregate_function is None:
            raise ValueError("aggregate_function_null")
        if field_supplied and self.aggregate_field_name is None:
            raise ValueError("aggregate_field_name_null")
        if self.operation_hint != "aggregate":
            if function_supplied or field_supplied:
                raise ValueError("aggregate_details_require_aggregate_operation")
        elif function_supplied != field_supplied:
            raise ValueError("aggregate_details_must_be_supplied_together")

        candidate_count = len(self.candidate_source_ids)
        if len(set(self.candidate_source_ids)) != candidate_count:
            raise ValueError("duplicate_semantic_candidate_source_id")
        if self.interpretation_status == "resolved" and candidate_count != 1:
            raise ValueError("resolved_semantic_candidate_required")
        if self.interpretation_status == "ambiguous" and candidate_count not in {
            2,
            3,
        }:
            raise ValueError("ambiguous_semantic_candidates_required")
        if self.interpretation_status == "no_match" and candidate_count != 0:
            raise ValueError("semantic_no_match_candidates_not_allowed")
        return self


class SourceMatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: SourceMatchStatus
    matched_source_ids: list[EvidenceSufficiencyIdentifier] = Field(max_length=32)
    probe_source_ids: list[EvidenceSufficiencyIdentifier] = Field(
        default_factory=list,
        max_length=3,
    )
    reason_codes: list[SourceMatchReasonCode] = Field(min_length=1, max_length=11)

    @model_serializer(mode="wrap")
    def omit_absent_probe_source_ids(self, handler):
        serialized = handler(self)
        if not self.probe_source_ids:
            serialized.pop("probe_source_ids", None)
        return serialized

    @model_validator(mode="after")
    def validate_match_outcome(self) -> SourceMatchResult:
        if self.matched_source_ids != sorted(set(self.matched_source_ids)):
            raise ValueError("source_match_ids_not_sorted_unique")
        if self.probe_source_ids != sorted(set(self.probe_source_ids)):
            raise ValueError("probe_source_ids_not_sorted_unique")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("duplicate_source_match_reason_code")
        if self.status == "matched":
            if not self.matched_source_ids:
                raise ValueError("matched_source_id_required")
        elif self.matched_source_ids:
            raise ValueError("matched_source_ids_not_allowed")
        if self.probe_source_ids:
            if len(self.probe_source_ids) not in {2, 3}:
                raise ValueError("probe_source_ids_must_be_bounded_ambiguity")
            if self.status != "ambiguous":
                raise ValueError("probe_source_ids_require_ambiguous_status")
            if "semantic_candidates_ambiguous" not in self.reason_codes:
                raise ValueError("probe_source_ids_require_semantic_ambiguity")
        return self


class EvidenceShapeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_input_kinds: list[EvidenceInputKind] = Field(max_length=6)
    external_verification_required: bool
    freshness_sensitive: bool
    high_stakes_accuracy_required: bool
    continuation_of_prior_evidence_task: bool
    prior_task_shape: EvidenceTaskShape | None = None
    source_discovery: SourceDiscoveryContext | None = None
    semantic_advisory: SemanticEvidenceAdvisory | None = None

    @model_validator(mode="after")
    def validate_continuation_context(self) -> EvidenceShapeContext:
        if len(set(self.evidence_input_kinds)) != len(self.evidence_input_kinds):
            raise ValueError("duplicate_evidence_input_kind")
        if self.continuation_of_prior_evidence_task and self.prior_task_shape is None:
            raise ValueError("prior_task_shape_required")
        if not self.continuation_of_prior_evidence_task and self.prior_task_shape is not None:
            raise ValueError("prior_task_shape_not_allowed")
        if self.semantic_advisory is not None:
            if self.source_discovery is None:
                raise ValueError("semantic_advisory_source_discovery_required")
            inventory_source_ids = {
                source.source_id for source in self.source_discovery.sources
            }
            if not set(self.semantic_advisory.candidate_source_ids).issubset(
                inventory_source_ids
            ):
                raise ValueError("semantic_candidate_source_not_in_inventory")
            if (
                self.semantic_advisory.operation_hint == "aggregate"
                and self.semantic_advisory.aggregate_field_name is not None
                and self.semantic_advisory.candidate_source_ids
            ):
                sources_by_id = {
                    source.source_id: source for source in self.source_discovery.sources
                }
                for source_id in self.semantic_advisory.candidate_source_ids:
                    content_fields = sources_by_id[source_id].content_fields
                    if content_fields is None:
                        raise ValueError("aggregate_candidate_content_fields_required")
                    if self.semantic_advisory.aggregate_field_name not in content_fields:
                        raise ValueError("aggregate_field_not_configured_for_candidate")
        return self


class EvidenceShapeDeriveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: EvidenceSufficiencyIdentifier
    owner_id: EvidenceSufficiencyIdentifier
    conversation_id: EvidenceSufficiencyIdentifier
    surface: EvidenceSufficiencySurface
    runtime_session_id: EvidenceSufficiencyIdentifier
    runtime_turn_id: EvidenceSufficiencyIdentifier
    task_text: Annotated[str, Field(min_length=1, max_length=500)]
    interaction_kind: InteractionGovernanceKind
    task_context: EvidenceShapeContext

    @field_validator("task_text", mode="before")
    @classmethod
    def normalize_task_text(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return " ".join(value.split())


class EvidenceShapeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    derivation_id: EvidenceSufficiencyIdentifier
    question_anchor: Annotated[str, Field(min_length=1, max_length=500)]
    question_anchor_digest: Annotated[
        str,
        Field(pattern=r"^sha256:[0-9a-f]{64}$", min_length=71, max_length=71),
    ]
    derivation_status: EvidenceShapeDerivationStatus
    task_shape: EvidenceTaskShape | None = None
    candidate_task_shapes: list[EvidenceTaskShape] = Field(max_length=7)
    evidence_scope_material: bool
    clarification_required: bool
    reason_codes: list[EvidenceShapeReasonCode] = Field(max_length=17)
    user_safe_summary: Annotated[str, Field(min_length=1, max_length=500)]
    source_match: SourceMatchResult | None = None
    aggregate_spec: AggregateSpec | None = None

    @model_serializer(mode="wrap")
    def omit_absent_optional_fields(self, handler):
        serialized = handler(self)
        if self.source_match is None:
            serialized.pop("source_match", None)
        if self.aggregate_spec is None:
            serialized.pop("aggregate_spec", None)
        return serialized

    @model_validator(mode="after")
    def validate_derivation_outcome(self) -> EvidenceShapeResult:
        aggregate_spec_supplied = "aggregate_spec" in self.model_fields_set
        if self.derivation_status == "derived":
            if self.task_shape is None or self.candidate_task_shapes != [self.task_shape]:
                raise ValueError("invalid_derived_shape_outcome")
            if not self.evidence_scope_material or self.clarification_required:
                raise ValueError("invalid_derived_shape_flags")
        elif self.derivation_status == "not_applicable":
            if self.task_shape is not None or self.candidate_task_shapes:
                raise ValueError("invalid_not_applicable_shape_outcome")
            if self.evidence_scope_material or self.clarification_required:
                raise ValueError("invalid_not_applicable_shape_flags")
        else:
            if self.task_shape is not None:
                raise ValueError("invalid_ambiguous_shape_outcome")
            if not self.evidence_scope_material or not self.clarification_required:
                raise ValueError("invalid_ambiguous_shape_flags")
        if self.derivation_status == "derived" and self.task_shape == "aggregate":
            if self.aggregate_spec is None:
                raise ValueError("aggregate_shape_requires_spec")
        elif aggregate_spec_supplied:
            raise ValueError("aggregate_spec_requires_derived_aggregate_shape")
        return self


class EvidenceShapeDeriveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: EvidenceSufficiencyIdentifier
    owner_id: EvidenceSufficiencyIdentifier
    conversation_id: EvidenceSufficiencyIdentifier
    surface: EvidenceSufficiencySurface
    runtime_session_id: EvidenceSufficiencyIdentifier
    runtime_turn_id: EvidenceSufficiencyIdentifier
    result: EvidenceShapeResult


EvidenceRequirementCriticality = Literal["material", "optional"]
EvidenceAcquisitionOutcome = Literal[
    "satisfied",
    "partial",
    "not_attempted",
    "unavailable",
    "unsupported",
    "failed",
    "excluded",
    "filtered",
    "truncated",
    "unresolved_contradiction",
    "unknown",
]
EvidenceRequirementEffectiveOutcome = Literal[
    "satisfied",
    "partial",
    "not_attempted",
    "unavailable",
    "unsupported",
    "failed",
    "excluded",
    "filtered",
    "truncated",
    "unresolved_contradiction",
    "unknown",
    "missing",
]
EvidenceSufficiencyStatus = Literal[
    "sufficient_for_declared_scope",
    "sufficient_with_limitations",
    "insufficient",
    "unknown",
]
EvidenceAnswerConstraint = Literal[
    "qualify_conclusion",
    "disclose_limitations",
    "identify_unexamined_scope",
    "additional_acquisition_or_clarification_required",
    "withhold_unqualified_conclusion",
    "withhold_exhaustive_conclusion",
    "withhold_absence_conclusion",
    "withhold_contradiction_sensitive_conclusion",
]
EvidenceSufficiencyReasonCode = Literal[
    "all_declared_requirements_satisfied",
    "optional_requirement_incomplete",
    "material_requirement_not_satisfied",
    "material_requirement_unknown",
    "material_requirement_missing",
    "unresolved_material_contradiction",
    "exhaustive_scope_incomplete",
    "absence_scope_unproven",
    "contradiction_sensitive_scope_unresolved",
]


class EvidenceRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: EvidenceSufficiencyIdentifier
    requirement_kind: EvidenceSufficiencyIdentifier
    criticality: EvidenceRequirementCriticality


class EvidenceAcquisitionFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: EvidenceSufficiencyIdentifier
    outcome: EvidenceAcquisitionOutcome


class EvidenceRequirementEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: EvidenceSufficiencyIdentifier
    requirement_kind: EvidenceSufficiencyIdentifier
    criticality: EvidenceRequirementCriticality
    effective_outcome: EvidenceRequirementEffectiveOutcome


class EvidenceSufficiencyEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: EvidenceSufficiencyIdentifier
    owner_id: EvidenceSufficiencyIdentifier
    conversation_id: EvidenceSufficiencyIdentifier
    surface: EvidenceSufficiencySurface
    runtime_session_id: EvidenceSufficiencyIdentifier
    runtime_turn_id: EvidenceSufficiencyIdentifier
    evidence_plan_id: EvidenceSufficiencyIdentifier
    acquisition_manifest_id: EvidenceSufficiencyIdentifier
    task_shape: EvidenceTaskShape
    declared_requirements: list[EvidenceRequirement] = Field(min_length=1, max_length=32)
    acquisition_facts: list[EvidenceAcquisitionFact] = Field(
        default_factory=list,
        max_length=32,
    )

    @model_validator(mode="after")
    def validate_requirement_facts(self) -> EvidenceSufficiencyEvaluateRequest:
        requirement_ids: set[str] = set()
        for requirement in self.declared_requirements:
            if requirement.requirement_id in requirement_ids:
                raise ValueError("duplicate_evidence_requirement")
            requirement_ids.add(requirement.requirement_id)

        fact_ids: set[str] = set()
        for fact in self.acquisition_facts:
            if fact.requirement_id in fact_ids:
                raise ValueError("duplicate_acquisition_fact")
            if fact.requirement_id not in requirement_ids:
                raise ValueError("undeclared_evidence_requirement")
            fact_ids.add(fact.requirement_id)
        return self


class EvidenceSufficiencyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_id: EvidenceSufficiencyIdentifier
    task_shape: EvidenceTaskShape
    sufficiency_status: EvidenceSufficiencyStatus
    evaluated_requirements: list[EvidenceRequirementEvaluation] = Field(max_length=32)
    reason_codes: list[EvidenceSufficiencyReasonCode] = Field(max_length=9)
    answer_constraints: list[EvidenceAnswerConstraint] = Field(max_length=8)
    qualification_required: bool
    additional_acquisition_required: bool
    user_safe_summary: Annotated[str, Field(min_length=1, max_length=500)]


class EvidenceSufficiencyEvaluateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: EvidenceSufficiencyIdentifier
    owner_id: EvidenceSufficiencyIdentifier
    conversation_id: EvidenceSufficiencyIdentifier
    surface: EvidenceSufficiencySurface
    runtime_session_id: EvidenceSufficiencyIdentifier
    runtime_turn_id: EvidenceSufficiencyIdentifier
    evidence_plan_id: EvidenceSufficiencyIdentifier
    acquisition_manifest_id: EvidenceSufficiencyIdentifier
    result: EvidenceSufficiencyResult


EvidenceInventoryStatus = Literal[
    "complete_for_declared_scope",
    "partial",
    "unknown",
    "unavailable",
]
EvidenceSourceCapability = Literal[
    "targeted_retrieval",
    "exact_fetch",
    "bounded_full_context",
    "structured_query",
    "context_expansion",
]
EvidenceSourceAvailability = Literal[
    "available",
    "unavailable",
    "disabled",
    "unknown",
]
EvidenceSourceAuthorityRole = Literal["authoritative", "supplemental", "unknown"]
EvidenceAcquisitionStrategy = Literal[
    "targeted_retrieval",
    "exact_fetch",
    "bounded_full_context",
    "structured_query",
    "hybrid",
    "structured_field_values",
]
EvidencePlanStatus = Literal["ready", "ready_with_limitations", "unsupported"]
EvidenceCompletenessExpectation = Literal[
    "targeted_scope",
    "complete_for_declared_scope",
    "complete_for_selected_sources",
    "complete_for_time_window",
    "bounded_decision_support",
]
EvidencePlanLimitationCode = Literal[
    "declared_source_missing_from_inventory",
    "declared_category_not_available",
    "source_inventory_partial",
    "source_inventory_unknown",
    "source_inventory_unavailable",
    "authoritative_source_missing",
    "authoritative_source_unavailable",
    "required_capability_unavailable",
    "targeted_only_not_exhaustive",
    "absence_scope_not_enumerable",
    "insufficient_comparison_scope",
    "contradiction_search_not_supported",
    "historical_time_scope_missing",
    "historical_sequence_not_supported",
    "decision_support_scope_insufficient",
    "optional_source_unavailable",
    "aggregate_field_unavailable",
]


class EvidenceExactSourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ref: Annotated[str, Field(min_length=1, max_length=240)]
    source_id: EvidenceSufficiencyIdentifier

    @field_validator("source_ref")
    @classmethod
    def validate_source_ref(cls, value: str) -> str:
        if re.search(r"\s", value):
            raise ValueError("exact_source_ref_contains_whitespace")
        if "://" in value or "?" in value:
            raise ValueError("unsafe_exact_source_ref")
        return value


class EvidenceDeclaredScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ids: list[EvidenceSufficiencyIdentifier] = Field(
        default_factory=list,
        max_length=32,
    )
    source_categories: list[EvidenceSufficiencyIdentifier] = Field(
        default_factory=list,
        max_length=16,
    )
    exact_source_refs: list[EvidenceExactSourceReference] = Field(
        default_factory=list,
        max_length=16,
    )
    inventory_status: EvidenceInventoryStatus
    time_scope_ref: EvidenceSufficiencyIdentifier | None = None
    version_scope_ref: EvidenceSufficiencyIdentifier | None = None
    domain_scope_ref: EvidenceSufficiencyIdentifier | None = None
    project_scope_ref: EvidenceSufficiencyIdentifier | None = None

    @model_validator(mode="after")
    def validate_unique_scope_values(self) -> EvidenceDeclaredScope:
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("duplicate_declared_source_id")
        if len(set(self.source_categories)) != len(self.source_categories):
            raise ValueError("duplicate_declared_source_category")
        source_refs = [reference.source_ref for reference in self.exact_source_refs]
        if len(set(source_refs)) != len(source_refs):
            raise ValueError("duplicate_exact_source_ref")
        if self.source_ids and any(
            reference.source_id not in self.source_ids
            for reference in self.exact_source_refs
        ):
            raise ValueError("exact_source_ref_outside_declared_source_ids")
        return self


class EvidenceSourceDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: EvidenceSufficiencyIdentifier
    source_categories: list[EvidenceSufficiencyIdentifier] = Field(max_length=8)
    capabilities: list[EvidenceSourceCapability] = Field(max_length=5)
    availability: EvidenceSourceAvailability
    authority_role: EvidenceSourceAuthorityRole
    content_fields: list[DiscoverableContentField] | None = Field(
        default=None,
        max_length=24,
    )

    @model_serializer(mode="wrap")
    def omit_absent_content_fields(self, handler):
        serialized = handler(self)
        if self.content_fields is None:
            serialized.pop("content_fields", None)
        return serialized

    @field_validator("content_fields")
    @classmethod
    def validate_content_fields(
        cls,
        value: list[DiscoverableContentField] | None,
    ) -> list[DiscoverableContentField] | None:
        if value is None:
            return value
        validated = [_validate_discoverable_content_field(field) for field in value]
        if len(set(validated)) != len(validated):
            raise ValueError("duplicate_source_content_field")
        if validated != sorted(validated):
            raise ValueError("source_content_fields_not_sorted")
        return validated

    @model_validator(mode="after")
    def validate_unique_source_values(self) -> EvidenceSourceDescriptor:
        if len(set(self.source_categories)) != len(self.source_categories):
            raise ValueError("duplicate_source_category")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("duplicate_source_capability")
        if "content_fields" in self.model_fields_set and self.content_fields is None:
            raise ValueError("source_content_fields_null")
        return self


class EvidencePlanCompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: EvidenceSufficiencyIdentifier
    owner_id: EvidenceSufficiencyIdentifier
    conversation_id: EvidenceSufficiencyIdentifier
    surface: EvidenceSufficiencySurface
    runtime_session_id: EvidenceSufficiencyIdentifier
    runtime_turn_id: EvidenceSufficiencyIdentifier
    question_anchor: Annotated[str, Field(min_length=1, max_length=500)]
    task_shape: EvidenceTaskShape
    declared_scope: EvidenceDeclaredScope
    source_inventory: list[EvidenceSourceDescriptor] = Field(max_length=32)
    aggregate_spec: AggregateSpec | None = None

    @model_serializer(mode="wrap")
    def omit_absent_aggregate_spec(self, handler):
        serialized = handler(self)
        if self.aggregate_spec is None:
            serialized.pop("aggregate_spec", None)
        return serialized

    @field_validator("question_anchor", mode="before")
    @classmethod
    def normalize_question_anchor(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return " ".join(value.split())

    @model_validator(mode="after")
    def validate_unique_source_ids(self) -> EvidencePlanCompileRequest:
        aggregate_spec_supplied = "aggregate_spec" in self.model_fields_set
        source_ids = [source.source_id for source in self.source_inventory]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("duplicate_source_descriptor")
        if self.task_shape == "aggregate":
            if self.aggregate_spec is None:
                raise ValueError("aggregate_plan_request_requires_spec")
        elif aggregate_spec_supplied:
            raise ValueError("aggregate_spec_requires_aggregate_plan_request")
        return self


class EvidencePlanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: EvidenceSufficiencyIdentifier
    question_anchor: Annotated[str, Field(min_length=1, max_length=500)]
    question_anchor_digest: Annotated[
        str,
        Field(pattern=r"^sha256:[0-9a-f]{64}$", min_length=71, max_length=71),
    ]
    task_shape: EvidenceTaskShape
    plan_status: EvidencePlanStatus
    completeness_expectation: EvidenceCompletenessExpectation
    contradiction_search_required: bool
    eligible_source_ids: list[EvidenceSufficiencyIdentifier] = Field(max_length=32)
    authoritative_source_ids: list[EvidenceSufficiencyIdentifier] = Field(max_length=32)
    selected_strategies: list[EvidenceAcquisitionStrategy] = Field(max_length=5)
    declared_requirements: list[EvidenceRequirement] = Field(min_length=1, max_length=32)
    limitation_codes: list[EvidencePlanLimitationCode] = Field(max_length=16)
    user_safe_summary: Annotated[str, Field(min_length=1, max_length=500)]
    aggregate_spec: AggregateSpec | None = None

    @model_serializer(mode="wrap")
    def omit_absent_aggregate_spec(self, handler):
        serialized = handler(self)
        if self.aggregate_spec is None:
            serialized.pop("aggregate_spec", None)
        return serialized

    @model_validator(mode="after")
    def validate_aggregate_spec(self) -> EvidencePlanResult:
        aggregate_spec_supplied = "aggregate_spec" in self.model_fields_set
        if self.task_shape == "aggregate":
            if self.aggregate_spec is None:
                raise ValueError("aggregate_plan_requires_spec")
        elif aggregate_spec_supplied:
            raise ValueError("aggregate_spec_requires_aggregate_plan")
        return self


class EvidencePlanCompileResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: EvidenceSufficiencyIdentifier
    owner_id: EvidenceSufficiencyIdentifier
    conversation_id: EvidenceSufficiencyIdentifier
    surface: EvidenceSufficiencySurface
    runtime_session_id: EvidenceSufficiencyIdentifier
    runtime_turn_id: EvidenceSufficiencyIdentifier
    result: EvidencePlanResult


EvidenceNextStep = Literal[
    "answer_within_declared_scope",
    "provide_qualified_partial_answer",
    "perform_additional_acquisition",
    "ask_narrow_clarification",
    "disclose_unexamined_scope",
    "withhold_unsupported_conclusion",
]
EvidenceConclusionDisposition = Literal[
    "bounded_conclusion_allowed",
    "qualified_partial_only",
    "requested_conclusion_withheld",
]
EvidenceProviderDisposition = Literal["allowed", "blocked"]
EvidenceReacquisitionGuard = Literal[
    "not_applicable",
    "changed_premise_allowed",
    "unchanged_premise_blocked",
    "premise_already_attempted",
]
EvidenceClarificationTarget = Literal[
    "question_scope",
    "source_scope",
    "exact_reference",
    "time_scope",
    "version_scope",
    "domain_scope",
    "project_scope",
]
EvidenceNextStepReasonCode = Literal[
    "declared_scope_sufficient",
    "optional_limitations_remain",
    "material_uncertainty_requires_clarification",
    "changed_acquisition_premise_available",
    "unchanged_acquisition_premise",
    "acquisition_premise_already_selected",
    "substantive_partial_evidence_available",
    "unexamined_material_scope",
    "unsupported_conclusion_withheld",
]


class EvidenceAcquisitionPremise(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_anchor_digest: Annotated[
        str,
        Field(pattern=r"^sha256:[0-9a-f]{64}$", min_length=71, max_length=71),
    ]
    task_shape: EvidenceTaskShape
    declared_scope: EvidenceDeclaredScope
    source_inventory: list[EvidenceSourceDescriptor] = Field(max_length=32)
    selected_strategies: list[EvidenceAcquisitionStrategy] = Field(max_length=5)
    aggregate_spec: AggregateSpec | None = None

    @model_serializer(mode="wrap")
    def omit_absent_aggregate_spec(self, handler):
        serialized = handler(self)
        if self.aggregate_spec is None:
            serialized.pop("aggregate_spec", None)
        return serialized

    @model_validator(mode="after")
    def validate_unique_premise_values(self) -> EvidenceAcquisitionPremise:
        aggregate_spec_supplied = "aggregate_spec" in self.model_fields_set
        source_ids = [source.source_id for source in self.source_inventory]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("duplicate_source_descriptor")
        if len(set(self.selected_strategies)) != len(self.selected_strategies):
            raise ValueError("duplicate_acquisition_strategy")
        if self.task_shape == "aggregate":
            if self.aggregate_spec is None:
                raise ValueError("aggregate_premise_requires_spec")
        elif aggregate_spec_supplied:
            raise ValueError("aggregate_spec_requires_aggregate_premise")
        return self


class EvidenceNextStepSelectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: EvidenceSufficiencyIdentifier
    owner_id: EvidenceSufficiencyIdentifier
    conversation_id: EvidenceSufficiencyIdentifier
    surface: EvidenceSufficiencySurface
    runtime_session_id: EvidenceSufficiencyIdentifier
    runtime_turn_id: EvidenceSufficiencyIdentifier
    evaluation_id: EvidenceSufficiencyIdentifier
    evidence_plan_id: EvidenceSufficiencyIdentifier
    acquisition_manifest_id: EvidenceSufficiencyIdentifier
    evaluated_requirements: list[EvidenceRequirementEvaluation] = Field(
        min_length=1,
        max_length=32,
    )
    current_premise: EvidenceAcquisitionPremise
    proposed_acquisition_premise: EvidenceAcquisitionPremise | None = None
    clarification_target: EvidenceClarificationTarget | None = None

    @model_validator(mode="after")
    def validate_next_step_inputs(self) -> EvidenceNextStepSelectRequest:
        requirement_ids = [
            requirement.requirement_id for requirement in self.evaluated_requirements
        ]
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("duplicate_evidence_requirement")
        if (
            self.proposed_acquisition_premise is not None
            and self.clarification_target is not None
        ):
            raise ValueError("conflicting_next_step_inputs")
        return self


class EvidenceNextStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection_id: EvidenceSufficiencyIdentifier
    evaluation_id: EvidenceSufficiencyIdentifier
    evidence_plan_id: EvidenceSufficiencyIdentifier
    acquisition_manifest_id: EvidenceSufficiencyIdentifier
    task_shape: EvidenceTaskShape
    sufficiency_status: EvidenceSufficiencyStatus
    selected_next_step: EvidenceNextStep
    conclusion_disposition: EvidenceConclusionDisposition
    provider_disposition: EvidenceProviderDisposition
    current_premise_digest: Annotated[
        str,
        Field(pattern=r"^sha256:[0-9a-f]{64}$", min_length=71, max_length=71),
    ]
    proposed_premise_digest: Annotated[
        str,
        Field(pattern=r"^sha256:[0-9a-f]{64}$", min_length=71, max_length=71),
    ] | None = None
    reacquisition_guard: EvidenceReacquisitionGuard
    clarification_target: EvidenceClarificationTarget | None = None
    unresolved_material_requirement_ids: list[EvidenceSufficiencyIdentifier] = Field(
        max_length=32
    )
    reason_codes: list[EvidenceNextStepReasonCode] = Field(max_length=4)
    user_safe_summary: Annotated[str, Field(min_length=1, max_length=500)]

    @model_validator(mode="after")
    def validate_deterministic_result(self) -> EvidenceNextStepResult:
        if self.unresolved_material_requirement_ids != sorted(
            set(self.unresolved_material_requirement_ids)
        ):
            raise ValueError("unordered_unresolved_material_requirements")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("duplicate_next_step_reason_code")

        if self.sufficiency_status == "sufficient_for_declared_scope":
            if (
                self.selected_next_step != "answer_within_declared_scope"
                or self.conclusion_disposition != "bounded_conclusion_allowed"
                or self.provider_disposition != "allowed"
                or self.unresolved_material_requirement_ids
            ):
                raise ValueError("incoherent_sufficient_next_step")
        elif self.sufficiency_status == "sufficient_with_limitations":
            if (
                self.selected_next_step != "provide_qualified_partial_answer"
                or self.conclusion_disposition != "qualified_partial_only"
                or self.provider_disposition != "allowed"
                or self.unresolved_material_requirement_ids
            ):
                raise ValueError("incoherent_limited_next_step")
        elif self.conclusion_disposition == "bounded_conclusion_allowed":
            raise ValueError("unsupported_bounded_conclusion")

        if self.selected_next_step == "provide_qualified_partial_answer":
            if (
                self.conclusion_disposition != "qualified_partial_only"
                or self.provider_disposition != "allowed"
            ):
                raise ValueError("incoherent_qualified_partial_next_step")
            required_reason = (
                "optional_limitations_remain"
                if self.sufficiency_status == "sufficient_with_limitations"
                else "substantive_partial_evidence_available"
            )
            if required_reason not in self.reason_codes:
                raise ValueError("qualified_partial_reason_required")
        elif self.selected_next_step == "withhold_unsupported_conclusion":
            if self.conclusion_disposition != "requested_conclusion_withheld":
                raise ValueError("incoherent_withheld_conclusion")
            if "unsupported_conclusion_withheld" not in self.reason_codes:
                raise ValueError("withheld_conclusion_reason_required")
            if self.provider_disposition == "allowed" and (
                self.task_shape != "targeted_lookup"
                or self.sufficiency_status not in {"insufficient", "unknown"}
            ):
                raise ValueError("incoherent_advisory_provider_permission")
        elif self.selected_next_step != "answer_within_declared_scope" and (
            self.conclusion_disposition != "requested_conclusion_withheld"
            or self.provider_disposition != "blocked"
        ):
            raise ValueError("incoherent_blocked_next_step")

        required_step_reason = {
            "answer_within_declared_scope": "declared_scope_sufficient",
            "perform_additional_acquisition": (
                "changed_acquisition_premise_available"
            ),
            "ask_narrow_clarification": (
                "material_uncertainty_requires_clarification"
            ),
            "disclose_unexamined_scope": "unexamined_material_scope",
        }.get(self.selected_next_step)
        if (
            required_step_reason is not None
            and required_step_reason not in self.reason_codes
        ):
            raise ValueError("selected_next_step_reason_required")

        if self.selected_next_step == "ask_narrow_clarification":
            if self.clarification_target is None:
                raise ValueError("clarification_target_required")
        elif self.clarification_target is not None:
            raise ValueError("clarification_target_not_allowed")

        if self.reacquisition_guard == "not_applicable":
            if self.proposed_premise_digest is not None:
                raise ValueError("proposed_premise_not_allowed")
        elif self.proposed_premise_digest is None:
            raise ValueError("proposed_premise_required")
        elif self.reacquisition_guard == "changed_premise_allowed":
            if (
                self.selected_next_step != "perform_additional_acquisition"
                or self.proposed_premise_digest == self.current_premise_digest
            ):
                raise ValueError("incoherent_changed_premise")
        elif self.reacquisition_guard == "unchanged_premise_blocked" and (
            self.proposed_premise_digest != self.current_premise_digest
        ):
            raise ValueError("incoherent_unchanged_premise")

        if (
            self.selected_next_step == "perform_additional_acquisition"
            and self.reacquisition_guard != "changed_premise_allowed"
        ):
            raise ValueError("changed_premise_guard_required")
        return self


class EvidenceNextStepSelectResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: EvidenceSufficiencyIdentifier
    owner_id: EvidenceSufficiencyIdentifier
    conversation_id: EvidenceSufficiencyIdentifier
    surface: EvidenceSufficiencySurface
    runtime_session_id: EvidenceSufficiencyIdentifier
    runtime_turn_id: EvidenceSufficiencyIdentifier
    result: EvidenceNextStepResult


PrivacySurfaceCategory = Literal[
    "desktop_private",
    "mobile_private",
    "telegram_private",
    "voice_private",
    "car_voice_possible_passenger",
    "glasses_public_or_semi_public",
    "notification_preview",
    "unknown_surface",
]
PrivacyZone = Literal[
    "private",
    "shared_or_uncertain",
    "public_or_semi_public",
    "preview_limited",
    "unknown",
]
PrivacySensitivityLevel = Literal[
    "normal",
    "sensitive",
    "highly_sensitive",
    "unknown",
]
PrivacySensitivityDomain = Literal["personal", "health", "financial", "work"]


class PrivacyContextEvaluateRequest(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    runtime_session_id: str | None = Field(default=None, max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    surface_category: PrivacySurfaceCategory | None = None
    sensitivity_level: PrivacySensitivityLevel = "unknown"
    sensitivity_domains: list[PrivacySensitivityDomain] = Field(default_factory=list, max_length=4)

    @field_validator("surface_category", mode="before")
    @classmethod
    def normalize_surface_category(cls, value: Any) -> str | None:
        allowed = {
            "desktop_private",
            "mobile_private",
            "telegram_private",
            "voice_private",
            "car_voice_possible_passenger",
            "glasses_public_or_semi_public",
            "notification_preview",
            "unknown_surface",
        }
        if value is None:
            return None
        if not isinstance(value, str):
            return "unknown_surface"
        normalized = value.strip().lower()
        return normalized if normalized in allowed else "unknown_surface"

    @field_validator("sensitivity_level", mode="before")
    @classmethod
    def normalize_sensitivity_level(cls, value: Any) -> str:
        allowed = {
            "normal",
            "sensitive",
            "highly_sensitive",
            "unknown",
        }
        if not isinstance(value, str):
            return "unknown"
        normalized = value.strip().lower()
        return normalized if normalized in allowed else "unknown"


class PrivacyContextResult(BaseModel):
    privacy_zone: PrivacyZone
    surface_type: PrivacySurfaceCategory
    sensitivity_level: PrivacySensitivityLevel
    sensitive_detail_allowed: bool
    notification_detail_allowed: bool
    voice_detail_allowed: bool
    screen_detail_allowed: bool
    redaction_required: bool
    safe_summary_required: bool
    reason_codes: list[BoundedLabel] = Field(default_factory=list, max_length=8)


class PrivacyContextEvaluateResponse(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    runtime_session_id: str = Field(max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    result: PrivacyContextResult


class HistoryFollowupCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source: HistoryFollowupCandidateSource
    intent: HistoryFollowupIntent
    confidence: float = Field(ge=0.0, le=1.0)
    target_mode: HistoryFollowupTargetMode
    new_verification_requested: bool

    @model_validator(mode="after")
    def validate_candidate_consistency(self):
        if self.source == "deterministic" and self.confidence != 1.0:
            raise ValueError("deterministic_candidate_requires_full_confidence")
        if (
            self.intent == "new_verification_request"
            and not self.new_verification_requested
        ):
            raise ValueError("new_verification_intent_requires_explicit_request")
        if (
            self.intent
            in {"not_history_followup", "ambiguous_history_followup"}
            and self.new_verification_requested
        ):
            raise ValueError("non_actionable_history_intent_forbids_verification")
        return self


class InteractionGovernanceEvaluateRequest(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    runtime_session_id: str | None = Field(default=None, max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    surface_session_id: str | None = Field(default=None, max_length=120)
    active_mode: str | None = Field(default=None, max_length=64)
    current_user_text: BoundedText | None = None
    recent_messages: list["InterruptMessage"] = Field(default_factory=list, max_length=12)
    surface_metadata_json: dict[str, Any] = Field(default_factory=dict)
    history_followup_candidate: HistoryFollowupCandidate | None = None

    @model_validator(mode="after")
    def validate_history_candidate_has_current_user_turn(self):
        if self.history_followup_candidate is None:
            return self
        if self.current_user_text is not None and self.current_user_text.strip():
            return self
        if any(
            message.role == "user" and message.content.strip()
            for message in reversed(self.recent_messages)
        ):
            return self
        raise ValueError("history_followup_candidate_requires_current_user_turn")


class HistoryFollowupPolicyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: HistoryFollowupPolicyStatus
    intent: HistoryFollowupIntent | None = None
    candidate_source: HistoryFollowupCandidateSource | None = None
    target_mode: HistoryFollowupTargetMode | None = None
    explanation_kind: HistoryFollowupExplanationKind | None = None
    acquisition_question: HistoryFollowupAcquisitionQuestion | None = None
    history_lookup_allowed: bool
    new_verification_requested: bool
    new_verification_allowed_after_history_resolution: bool
    clarification_required: bool
    confidence_band: HistoryFollowupConfidenceBand
    reason_codes: list[HistoryFollowupReasonCode] = Field(min_length=1, max_length=4)


class InteractionGovernanceResult(BaseModel):
    interaction_kind: InteractionGovernanceKind
    tension_level: InteractionGovernanceTension
    literal_command_confidence: float = Field(ge=0.0, le=1.0)
    commentary_allowed: bool
    humor_allowed: bool
    clarifying_question_allowed: bool
    action_allowed: bool
    requires_confirmation: bool
    persona_scope_hint: str | None = Field(default=None, max_length=64)
    privacy_sensitivity_hint: InteractionGovernancePrivacySensitivity
    response_posture: InteractionGovernanceResponsePosture
    confidence: float = Field(ge=0.0, le=1.0)
    reason_summary: list[BoundedLabel] = Field(default_factory=list, max_length=8)
    history_followup_policy: HistoryFollowupPolicyResult


class InteractionGovernanceEvaluateResponse(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    runtime_session_id: str = Field(max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    result: InteractionGovernanceResult


class PersonaContainmentEvaluateRequest(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    runtime_session_id: str | None = Field(default=None, max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    active_persona_id: str | None = Field(default=None, max_length=120)
    requested_persona_id: str | None = Field(default=None, max_length=120)
    persona_scope_hint: str | None = Field(default=None, max_length=64)
    interaction_kind: InteractionGovernanceKind | None = None
    current_user_text: BoundedText | None = None
    recent_messages: list["InterruptMessage"] = Field(default_factory=list, max_length=12)
    surface_metadata_json: dict[str, Any] = Field(default_factory=dict)


ArtifactContentClass = Literal[
    "document",
    "code",
    "image",
    "screenshot",
    "audio",
    "video",
    "other",
]


class ArtifactAccessPolicy(BaseModel):
    enforcement_mode: Literal["mandatory"] = "mandatory"
    allowed_content_classes: list[ArtifactContentClass] = Field(default_factory=list, max_length=8)
    allowed_domains: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    maximum_sensitivity: WorldStateSensitivity
    surface_content_capabilities: list[ArtifactContentClass] = Field(
        default_factory=list,
        max_length=8,
    )
    reason_codes: list[BoundedLabel] = Field(default_factory=list, max_length=8)


class PersonaContainmentResult(BaseModel):
    active_persona_id: str = Field(max_length=120)
    capability_domain: BoundedLabel
    allowed_memory_domains: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    blocked_memory_domains: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    allowed_world_state_domains: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    allowed_relationship_domains: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    allowed_tool_domains: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    cross_scope_access_allowed: bool = False
    cross_scope_reason: BoundedLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason_summary: list[BoundedLabel] = Field(default_factory=list, max_length=8)
    artifact_access_policy: ArtifactAccessPolicy


class PersonaContainmentEvaluateResponse(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    runtime_session_id: str = Field(max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    result: PersonaContainmentResult


RestraintPolicy = Literal[
    "answer_normally",
    "short_answer",
    "defer_expansion",
    "ask_clarifying_question",
    "do_not_retrieve",
    "do_not_personalize",
    "suppress_proactive_output",
]


class RestraintEvaluateRequest(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    runtime_session_id: str | None = Field(default=None, max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    interaction_kind: InteractionGovernanceKind | None = None
    response_posture: InteractionGovernanceResponsePosture | None = None
    active_persona_id: str | None = Field(default=None, max_length=120)
    capability_domain: str | None = Field(default=None, max_length=64)
    current_user_text: BoundedText | None = None
    recent_messages: list["InterruptMessage"] = Field(default_factory=list, max_length=12)
    surface_metadata_json: dict[str, Any] = Field(default_factory=dict)


class RestraintResult(BaseModel):
    restraint_policy: RestraintPolicy
    domains: list[BoundedLabel] = Field(default_factory=list, max_length=8)
    reason: BoundedLabel
    prompt_overlay: str = Field(default="", max_length=240)
    trace_ref: BoundedTraceRef
    confidence: float = Field(ge=0.0, le=1.0)
    reason_summary: list[BoundedLabel] = Field(default_factory=list, max_length=8)
    retrieval_suppressed: bool = False
    personalization_suppressed: bool = False
    proactive_output_suppressed: bool = False
    brevity_preferred: bool = False
    clarification_preferred: bool = False


class RestraintEvaluateResponse(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    runtime_session_id: str = Field(max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    result: RestraintResult


SituatedPresenceVisibility = Literal["private", "shared", "public", "unknown"]
SituatedPresenceConstraint = Literal["normal", "constrained", "unknown"]
SituatedPresenceEmotionalAttunement = Literal["none", "minimal", "brief"]
SituatedPresenceChallenge = Literal["none", "low", "medium"]
SituatedPresenceReason = Literal[
    "upstream_confidence_insufficient",
    "tense_context",
    "tactical_response_required",
    "high_impact_context",
    "brief_steadying_allowed",
    "light_commentary_allowed",
    "low_risk_commentary_allowed",
    "ambiguous_context",
    "surface_public",
    "surface_shared",
    "surface_visibility_unknown",
    "surface_constrained",
    "surface_constraint_unknown",
    "privacy_sensitive",
    "proactive_output_suppressed",
    "personalization_suppressed",
    "confirmation_required",
    "upstream_commentary_suppressed",
    "upstream_humor_suppressed",
    "brevity_preferred",
    "clarification_preferred",
]
SITUATED_PRESENCE_REASON_ORDER: tuple[SituatedPresenceReason, ...] = (
    "upstream_confidence_insufficient",
    "tense_context",
    "tactical_response_required",
    "high_impact_context",
    "brief_steadying_allowed",
    "light_commentary_allowed",
    "low_risk_commentary_allowed",
    "ambiguous_context",
    "surface_public",
    "surface_shared",
    "surface_visibility_unknown",
    "surface_constrained",
    "surface_constraint_unknown",
    "privacy_sensitive",
    "proactive_output_suppressed",
    "personalization_suppressed",
    "confirmation_required",
    "upstream_commentary_suppressed",
    "upstream_humor_suppressed",
    "brevity_preferred",
    "clarification_preferred",
)


class SituatedPresenceSurfaceContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    visibility: SituatedPresenceVisibility
    constraint: SituatedPresenceConstraint


class SituatedPresenceInteractionGovernanceProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    interaction_kind: InteractionGovernanceKind
    tension_level: InteractionGovernanceTension
    commentary_allowed: bool = Field(strict=True)
    humor_allowed: bool = Field(strict=True)
    action_allowed: bool = Field(strict=True)
    requires_confirmation: bool = Field(strict=True)
    privacy_sensitivity_hint: InteractionGovernancePrivacySensitivity
    response_posture: InteractionGovernanceResponsePosture
    confidence: float = Field(ge=0.0, le=1.0, strict=True)

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_strict_confidence(cls, value: Any) -> Any:
        if not isinstance(value, float):
            raise ValueError("situated_presence_confidence_float_required")
        return value


class SituatedPresenceRestraintProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    restraint_policy: RestraintPolicy
    proactive_output_suppressed: bool = Field(strict=True)
    personalization_suppressed: bool = Field(strict=True)
    brevity_preferred: bool = Field(strict=True)
    clarification_preferred: bool = Field(strict=True)
    confidence: float = Field(ge=0.0, le=1.0, strict=True)

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_strict_confidence(cls, value: Any) -> Any:
        if not isinstance(value, float):
            raise ValueError("situated_presence_confidence_float_required")
        return value


class SituatedPresenceEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    request_id: str = Field(min_length=1, max_length=120)
    owner_id: str = Field(min_length=1, max_length=120)
    conversation_id: str = Field(min_length=1, max_length=120)
    surface: str = Field(min_length=1, max_length=64)
    runtime_session_id: str = Field(min_length=1, max_length=120)
    runtime_turn_id: str = Field(min_length=1, max_length=120)
    surface_context: SituatedPresenceSurfaceContext
    interaction_governance: SituatedPresenceInteractionGovernanceProjection
    restraint: SituatedPresenceRestraintProjection

    @field_validator(
        "request_id",
        "owner_id",
        "conversation_id",
        "surface",
        "runtime_session_id",
        "runtime_turn_id",
    )
    @classmethod
    def validate_nonblank_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("situated_presence_identifier_blank")
        return value


class SituatedPresenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    commentary_allowed: bool = Field(strict=True)
    humor_allowed: bool = Field(strict=True)
    emotional_attunement_allowed: SituatedPresenceEmotionalAttunement
    challenge_allowed: SituatedPresenceChallenge
    silence_preferred: bool = Field(strict=True)
    surface_allows_commentary: bool = Field(strict=True)
    response_posture: InteractionGovernanceResponsePosture
    action_implication_allowed: Literal[False] = False
    reason_summary: list[SituatedPresenceReason] = Field(min_length=1, max_length=8)
    policy_version: Literal["situated-presence.v1"] = "situated-presence.v1"

    @field_validator("action_implication_allowed", mode="before")
    @classmethod
    def validate_action_implication_false(cls, value: Any) -> Any:
        if value is not False:
            raise ValueError("situated_presence_action_implication_forbidden")
        return value

    @model_validator(mode="after")
    def validate_coherence(self) -> "SituatedPresenceResult":
        if self.commentary_allowed and not self.surface_allows_commentary:
            raise ValueError("situated_presence_commentary_surface_inconsistent")
        if self.humor_allowed and (
            not self.commentary_allowed or not self.surface_allows_commentary
        ):
            raise ValueError("situated_presence_humor_inconsistent")
        if self.silence_preferred and (
            self.commentary_allowed or self.humor_allowed
        ):
            raise ValueError("situated_presence_silence_inconsistent")
        if len(self.reason_summary) != len(set(self.reason_summary)):
            raise ValueError("situated_presence_reasons_duplicate")
        reason_positions = {
            reason: index
            for index, reason in enumerate(SITUATED_PRESENCE_REASON_ORDER)
        }
        if self.reason_summary != sorted(
            self.reason_summary,
            key=reason_positions.__getitem__,
        ):
            raise ValueError("situated_presence_reasons_out_of_order")
        return self


class SituatedPresenceEvaluateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["situated-presence.v1"] = "situated-presence.v1"
    request_id: str = Field(min_length=1, max_length=120)
    owner_id: str = Field(min_length=1, max_length=120)
    conversation_id: str = Field(min_length=1, max_length=120)
    surface: str = Field(min_length=1, max_length=64)
    runtime_session_id: str = Field(min_length=1, max_length=120)
    runtime_turn_id: str = Field(min_length=1, max_length=120)
    result: SituatedPresenceResult


class PersonaProfile(BaseModel):
    persona_id: str = Field(max_length=120)
    display_name: str = Field(max_length=120)
    capability_domain: str = Field(max_length=120)
    description: str = Field(max_length=240)
    communication_policy_summary: list[BoundedRule] = Field(default_factory=list, max_length=8)
    runtime_policy_summary: list[BoundedRule] = Field(default_factory=list, max_length=8)
    advisory_memory_scope_summary: list[BoundedLabel] = Field(default_factory=list, max_length=12)
    advisory_tool_permission_summary: list[BoundedLabel] = Field(
        default_factory=list,
        max_length=12,
    )
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
    advisory_tool_permission_summary: list[BoundedLabel] = Field(
        default_factory=list,
        max_length=12,
    )
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
    advisory_tool_permission_summary: list[BoundedLabel] = Field(
        default_factory=list,
        max_length=12,
    )


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
    "verified",
    "superseded",
    "conflicted",
    "resolved",
]

TrustedWorldStateVerificationSourceType = Literal[
    "tool_output",
    "integration_event",
    "sensor_update",
    "repository_inspection",
    "calendar_event",
    "automation_workflow",
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


class WorldStateClaimVerifyRequest(RuntimeStateResolveRequest):
    runtime_session_id: str | None = Field(default=None, max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    world_state_claim_id: str = Field(max_length=120)
    expected_value_digest: str = Field(max_length=120)
    verifier_id: BoundedLabel | None = None
    verification_source_type: BoundedLabel
    verification_source_ref: str = Field(max_length=240)
    observed_at: str
    verified_at: str
    resulting_authority: WorldStateAuthority
    resulting_confidence: float = Field(ge=0.0, le=1.0)
    resulting_freshness_state: WorldStateFreshnessState
    resulting_expires_at: str | None = None
    resulting_ttl_seconds: int | None = Field(default=None, ge=1)
    resulting_revalidation_interval_seconds: int | None = Field(default=None, ge=1)


class WorldStateClaimView(BaseModel):
    world_state_claim_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    entity_id: str = Field(max_length=120)
    entity_type: str = Field(max_length=80)
    domain: BoundedLabel
    attribute: BoundedLabel
    value_json: Any | None = None
    value_redacted: bool = False
    value_digest: str = Field(max_length=120)
    source_type: WorldStateSourceType
    source_ref: str = Field(max_length=240)
    verification_verifier_id: BoundedLabel | None = None
    verification_source_type: BoundedLabel | None = None
    verification_source_ref: str | None = Field(default=None, max_length=240)
    confidence: float
    freshness_state: WorldStateFreshnessState
    effective_freshness_state: WorldStateFreshnessState
    state_authority: WorldStateAuthority
    observed_at: str
    last_verified_at: str | None = None
    last_verified_runtime_session_id: str | None = Field(default=None, max_length=120)
    last_verified_runtime_turn_id: str | None = Field(default=None, max_length=120)
    last_verification_request_id: str | None = Field(default=None, max_length=120)
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


CapabilityAuthorizationPhase = Literal["exposure", "selection", "dispatch"]
CapabilityOperationKind = Literal[
    "read_only",
    "state_change",
    "restart",
    "notification",
    "draft_or_prepare",
    "blocked_external_action",
]
CapabilityOperationClass = Literal[
    "read",
    "draft",
    "external_write",
    "destructive",
    "high_impact",
]
CapabilityDecisionCode = Literal[
    "allowed",
    "authorization_denied",
    "confirmation_required",
    "confirmation_accepted",
    "confirmation_rejected",
    "revalidation_required",
]
CapabilityReasonCode = Literal[
    "allowed",
    "registered_capability_domain_mismatch",
    "registered_operation_class_mismatch",
    "registered_supported_surfaces_mismatch",
    "registered_surface_not_allowed",
    "registered_persona_not_allowed",
    "runtime_session_mismatch",
    "runtime_turn_mismatch",
    "unknown_persona",
    "unknown_surface",
    "persona_mismatch",
    "surface_mismatch",
    "capability_domain_denied",
    "surface_unsupported",
    "argument_digest_required",
    "argument_digest_mismatch",
    "relationship_required",
    "relationship_not_selected",
    "relationship_not_authorized",
    "world_state_required",
    "world_state_not_selected",
    "world_state_not_authorized",
    "world_state_revalidation_required",
    "world_state_revalidator_required",
    "world_state_revalidator_not_configured",
    "world_state_revalidator_not_authorized",
    "world_state_revalidation_inadequate",
    "world_state_revalidator_conflict",
    "confirmation_required",
    "originating_turn_required",
    "confirmation_turn_not_current",
    "dispatch_turn_required",
    "challenge_missing",
    "challenge_expired",
    "challenge_mismatch",
    "challenge_turn_mismatch",
    "challenge_rejected",
    "challenge_consumed",
    "challenge_not_confirmed",
]
CapabilityConfirmationState = Literal[
    "not_required",
    "required",
    "issued",
    "accepted",
    "rejected",
]
CapabilityMatchReasonCode = Literal[
    "matched",
    "no_registered_capability",
    "registry_unavailable",
    "surface_not_allowed",
    "persona_not_allowed",
    "raw_capability_name_ignored",
]
ActionRiskLevel = Literal[
    "read_only",
    "low_reversible",
    "medium_requires_confirmation",
    "high_requires_confirmation",
    "blocked",
]
ActionAuthorityLevel = Literal[
    "answer_only",
    "suggest_only",
    "prepare_only",
    "execute_low_risk",
    "execute_after_confirmation",
    "blocked",
]
ActionTargetResolutionState = Literal["resolved", "ambiguous", "missing"]
ActionWorldStateFreshness = Literal["fresh", "stale", "unknown"]
ActionUserAuthorizationSignal = Literal["explicit", "vague", "none"]
ActionFlowIntent = Literal[
    "preview_requested",
    "execution_requested",
    "confirmation_received",
    "confirmation_cancelled",
    "confirmation_expired",
]
VerificationMethod = Literal["capability_verification"]
ActionExecutionStatus = Literal[
    "not_attempted",
    "blocked_by_policy",
    "cancelled_by_user",
    "executed",
    "partially_executed",
    "failed",
    "unknown",
]
ActionVerificationStatus = Literal[
    "not_supported",
    "not_required",
    "passed",
    "failed",
    "unknown",
]
ActionConfirmationStatus = Literal[
    "not_required",
    "required_pending",
    "accepted",
    "rejected",
    "expired",
    "cancelled",
    "unknown",
]
ActionRequestedBy = Literal["conversation_participant"]
ActionSummaryIdentifier = Annotated[
    str,
    Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
]
ActionSummaryCode = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"),
]


class ActionConsequenceFlags(BaseModel):
    external_consequence: bool = False
    destructive: bool = False
    data_loss: bool = False
    security: bool = False
    financial: bool = False
    message: bool = False


class CapabilityRegistryRecord(BaseModel):
    capability_id: str = Field(max_length=120)
    display_name: str = Field(max_length=120)
    domain: BoundedLabel
    description: str = Field(max_length=300)
    operation_kind: CapabilityOperationKind
    risk_level: BoundedLabel
    requires_confirmation: bool
    allowed_surfaces: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    allowed_personas: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    reversible: bool
    dry_run_supported: bool
    verification_supported: bool
    audit_required: bool


class CapabilityMatchRequest(RuntimeStateResolveRequest):
    active_persona_id: str = Field(max_length=120)
    current_user_text: BoundedText
    registry_enabled: bool = True


class CapabilityMatchResult(BaseModel):
    capability_matched: bool
    action_taken: Literal[False] = False
    reason_codes: list[CapabilityMatchReasonCode] = Field(default_factory=list, max_length=8)
    capability: CapabilityRegistryRecord | None = None


class CapabilityMatchResponse(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    active_persona_id: str = Field(max_length=120)
    result: CapabilityMatchResult


class CapabilityDiscoveryRequest(RuntimeStateResolveRequest):
    active_persona_id: str = Field(max_length=120)
    registry_enabled: bool = True


class CapabilityDiscoveryExample(BaseModel):
    capability_id: str = Field(max_length=120)
    display_name: str = Field(max_length=120)
    description: str = Field(max_length=300)
    operation_kind: CapabilityOperationKind
    risk_level: BoundedLabel
    reason_codes: list[CapabilityMatchReasonCode] = Field(default_factory=list, max_length=8)


class CapabilityDiscoveryResult(BaseModel):
    registry_available: bool
    action_taken: Literal[False] = False
    allowed_examples: list[CapabilityDiscoveryExample] = Field(default_factory=list, max_length=16)
    blocked_examples: list[CapabilityDiscoveryExample] = Field(default_factory=list, max_length=16)


class CapabilityDiscoveryResponse(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    active_persona_id: str = Field(max_length=120)
    result: CapabilityDiscoveryResult


class ActionAuthorityDecisionRequest(RuntimeStateResolveRequest):
    active_persona_id: str = Field(max_length=120)
    capability_id: str = Field(max_length=120)
    target_resolution_state: ActionTargetResolutionState = "resolved"
    world_state_freshness: ActionWorldStateFreshness = "fresh"
    consequence_flags: ActionConsequenceFlags = Field(
        default_factory=ActionConsequenceFlags
    )
    interaction_governance_kind: InteractionGovernanceKind | None = None
    interaction_governance_tension: InteractionGovernanceTension | None = None
    user_authorization_signal: ActionUserAuthorizationSignal = "explicit"
    registry_enabled: bool = True


class ActionAuthorityDecision(BaseModel):
    capability_id: str = Field(max_length=120)
    risk_level: ActionRiskLevel
    authority_level: ActionAuthorityLevel
    requires_confirmation: bool
    allowed: bool
    reason_summary: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    action_taken: Literal[False] = False


class ActionAuthorityDecisionResponse(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    active_persona_id: str = Field(max_length=120)
    result: ActionAuthorityDecision


class ActionFlowDecisionRequest(RuntimeStateResolveRequest):
    active_persona_id: str = Field(max_length=120)
    capability_id: str = Field(max_length=120)
    flow_intent: ActionFlowIntent = "execution_requested"
    target_resolution_state: ActionTargetResolutionState = "resolved"
    target_label: str | None = Field(default=None, max_length=120)
    world_state_freshness: ActionWorldStateFreshness = "fresh"
    affects_multiple_systems: bool = False
    consequence_flags: ActionConsequenceFlags = Field(
        default_factory=ActionConsequenceFlags
    )
    interaction_governance_kind: InteractionGovernanceKind | None = None
    interaction_governance_tension: InteractionGovernanceTension | None = None
    user_authorization_signal: ActionUserAuthorizationSignal = "explicit"
    registry_enabled: bool = True


class DryRunEffect(BaseModel):
    capability_id: str = Field(max_length=120)
    display_name: str = Field(max_length=120)
    operation_kind: CapabilityOperationKind
    target_label: str | None = Field(default=None, max_length=120)
    intended_effect: str = Field(max_length=600)
    reversible: bool
    consequence_summary: list[BoundedLabel] = Field(default_factory=list, max_length=8)


class ActionFlowDecision(BaseModel):
    capability_id: str = Field(max_length=120)
    dry_run_required: bool
    dry_run_supported: bool
    dry_run_effects: list[DryRunEffect] = Field(default_factory=list, max_length=4)
    confirmation_required: bool
    confirmation_text: str | None = Field(default=None, max_length=500)
    execution_allowed: bool
    verification_required: bool
    verification_supported: bool
    verification_method: VerificationMethod | None = None
    reason_summary: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    action_taken: Literal[False] = False


class ActionFlowDecisionResponse(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    surface: str = Field(max_length=64)
    active_persona_id: str = Field(max_length=120)
    result: ActionFlowDecision


class ActionSummaryRequest(RuntimeStateResolveRequest):
    model_config = ConfigDict(extra="forbid")

    request_id: ActionSummaryIdentifier
    owner_id: ActionSummaryIdentifier
    conversation_id: ActionSummaryIdentifier
    surface: ActionSummaryIdentifier
    runtime_session_id: ActionSummaryIdentifier
    runtime_turn_id: ActionSummaryIdentifier | None = None
    capability_id: ActionSummaryIdentifier
    active_persona_id: ActionSummaryIdentifier
    risk_level: ActionRiskLevel
    authority_level: ActionAuthorityLevel
    confirmation_status: ActionConfirmationStatus
    policy_reason_codes: list[ActionSummaryCode] = Field(
        default_factory=list, max_length=16
    )
    execution_status: ActionExecutionStatus
    execution_reason_code: ActionSummaryCode | None = None
    verification_status: ActionVerificationStatus
    verification_reason_code: ActionSummaryCode | None = None
    degradation_reason: ActionSummaryCode | None = None

    @model_validator(mode="after")
    def validate_action_outcomes(self) -> ActionSummaryRequest:
        attempted = self.execution_status in {
            "executed",
            "partially_executed",
            "failed",
        }
        if self.verification_status == "passed" and self.execution_status not in {
            "executed",
            "partially_executed",
        }:
            raise ValueError("verification_passed_without_execution")
        if self.execution_status == "not_attempted" and self.verification_status not in {
            "not_required",
            "not_supported",
            "unknown",
        }:
            raise ValueError("verification_invalid_without_attempt")
        if attempted and self.authority_level in {
            "suggest_only",
            "prepare_only",
            "blocked",
        }:
            raise ValueError("execution_not_permitted_by_authority")
        if attempted and self.risk_level == "blocked":
            raise ValueError("execution_not_permitted_by_risk")
        if (
            attempted
            and self.authority_level == "answer_only"
            and self.risk_level != "read_only"
        ):
            raise ValueError("execution_not_permitted_by_authority")
        if attempted and self.confirmation_status in {
            "required_pending",
            "rejected",
            "expired",
            "cancelled",
            "unknown",
        }:
            raise ValueError("execution_without_resolved_confirmation")
        if (
            attempted
            and self.authority_level == "execute_after_confirmation"
            and self.confirmation_status != "accepted"
        ):
            raise ValueError("execution_without_accepted_confirmation")
        if (
            attempted
            and self.risk_level
            in {"medium_requires_confirmation", "high_requires_confirmation"}
            and self.confirmation_status != "accepted"
        ):
            raise ValueError("execution_without_accepted_confirmation")
        return self


class ActionSummary(BaseModel):
    action_id: str = Field(max_length=64)
    capability_id: ActionSummaryIdentifier
    requested_by: ActionRequestedBy
    surface_type: ActionSummaryIdentifier
    active_persona_id: ActionSummaryIdentifier
    risk_level: ActionRiskLevel
    authority_level: ActionAuthorityLevel
    confirmation_status: ActionConfirmationStatus
    execution_status: ActionExecutionStatus
    verification_status: ActionVerificationStatus
    degradation_reason: ActionSummaryCode | None = None
    policy_reason_codes: list[ActionSummaryCode] = Field(
        default_factory=list, max_length=16
    )
    execution_reason_code: ActionSummaryCode | None = None
    verification_reason_code: ActionSummaryCode | None = None
    user_visible_summary: str = Field(max_length=500)


class ActionSummaryResponse(BaseModel):
    request_id: ActionSummaryIdentifier
    owner_id: ActionSummaryIdentifier
    conversation_id: ActionSummaryIdentifier
    runtime_session_id: ActionSummaryIdentifier
    runtime_turn_id: ActionSummaryIdentifier | None = None
    result: ActionSummary


class CapabilityRelationshipRequirement(BaseModel):
    relationship_scope: BoundedLabel | None = None
    relationship_type: BoundedLabel | None = None
    selector_ref: str | None = Field(default=None, max_length=120)
    authorization_required: bool = False


class CapabilityWorldStateRequirement(BaseModel):
    domain: BoundedLabel | None = None
    attribute: BoundedLabel | None = None
    entity_id: str | None = Field(default=None, max_length=120)
    selector_ref: str | None = Field(default=None, max_length=120)
    min_authority: WorldStateAuthority | None = None
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    max_freshness_state: WorldStateFreshnessState | None = None
    revalidator_id: str | None = Field(default=None, max_length=120)


class CapabilityAuthorizationRequest(RuntimeStateResolveRequest):
    runtime_session_id: str = Field(max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    active_persona_id: str = Field(max_length=120)
    authorization_phase: CapabilityAuthorizationPhase
    capability_id: str = Field(max_length=120)
    capability_domain: BoundedLabel
    operation_class: CapabilityOperationClass
    argument_digest: str | None = Field(default=None, max_length=120)
    supported_surfaces: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    relationship_requirements: list[CapabilityRelationshipRequirement] = Field(
        default_factory=list,
        max_length=16,
    )
    selected_relationship_ids: list[str] = Field(default_factory=list, max_length=64)
    world_state_requirements: list[CapabilityWorldStateRequirement] = Field(
        default_factory=list,
        max_length=16,
    )
    selected_world_state_claim_ids: list[str] = Field(default_factory=list, max_length=64)
    confirmation_challenge_ref: str | None = Field(default=None, max_length=120)


class CapabilityRevalidationSelector(BaseModel):
    world_state_claim_ids: list[str] = Field(default_factory=list, max_length=64)
    revalidator_id: str = Field(max_length=120)


class CapabilityAuthorizationResult(BaseModel):
    phase: CapabilityAuthorizationPhase
    allowed: bool
    decision_code: CapabilityDecisionCode
    reason_codes: list[CapabilityReasonCode] = Field(default_factory=list, max_length=16)
    confirmation_state: CapabilityConfirmationState
    challenge_ref: str | None = Field(default=None, max_length=120)
    challenge_expires_at: str | None = Field(default=None, max_length=64)
    revalidation_required: bool = False
    revalidation_selector: CapabilityRevalidationSelector | None = None
    relationship_ids_used: list[str] = Field(default_factory=list, max_length=64)
    world_state_claim_ids_used: list[str] = Field(default_factory=list, max_length=64)


class CapabilityAuthorizationResponse(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    runtime_session_id: str = Field(max_length=120)
    runtime_turn_id: str | None = Field(default=None, max_length=120)
    capability_id: str = Field(max_length=120)
    result: CapabilityAuthorizationResult


class CapabilityConfirmationRequest(RuntimeStateResolveRequest):
    runtime_session_id: str = Field(max_length=120)
    runtime_turn_id: str = Field(max_length=120)
    confirmation_challenge_ref: str = Field(max_length=120)
    capability_id: str = Field(max_length=120)
    operation_class: CapabilityOperationClass
    argument_digest: str = Field(max_length=120)
    confirmed: bool = True


class CapabilityConfirmationResponse(BaseModel):
    request_id: str = Field(max_length=120)
    owner_id: str = Field(max_length=120)
    conversation_id: str = Field(max_length=120)
    runtime_session_id: str = Field(max_length=120)
    runtime_turn_id: str = Field(max_length=120)
    confirmation_challenge_ref: str = Field(max_length=120)
    confirmation_state: CapabilityConfirmationState


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


class RelationshipRetrievalScopeProjection(BaseModel):
    applied: bool = False
    relationship_ids: list[str] = Field(default_factory=list, max_length=64)
    entity_ids: list[str] = Field(default_factory=list, max_length=64)
    relationship_scopes: list[BoundedLabel] = Field(default_factory=list, max_length=16)
    reason_codes: list[BoundedLabel] = Field(default_factory=list, max_length=8)


class RelationshipSelectResponse(BaseModel):
    selected_entities: list[RelationshipEntityView] = Field(default_factory=list)
    selected_relationships: list[RelationshipEdgeView] = Field(default_factory=list)
    excluded_relationship_summaries: list[RelationshipExcludedSummary] = Field(default_factory=list)
    prompt_content: str | None = None
    trace: RelationshipSelectTrace
    retrieval_scope_projection: RelationshipRetrievalScopeProjection


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
        if any(
            flag.risk_type in _OPERATOR_RISK_NOTE_LABELS
            for flag in self.interaction_risk_flags
        ):
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
    review_surfaces: list[HumanCompatibilityReviewSurface] = Field(
        default_factory=list,
        max_length=8,
    )
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
