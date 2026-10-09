from __future__ import annotations

import json
from hashlib import sha256
from typing import get_args

from models import (
    InteractionGovernanceKind,
    PersonaProfile,
    PersonaSelectionDecision,
    RuntimeIdentityContext,
    RuntimeIdentityResolveRequest,
    RuntimeIdentityResolveResponse,
    RuntimeIdentityTrace,
    SurfaceBinding,
)
from services.companion_contracts import companion_contracts_repository
from services.interaction_governance import persona_scope_hint_for_kind
from services.runtime_state import (
    resolve_runtime_session,
    runtime_session_by_id,
    runtime_state_repository,
)

_PERSONA_SCOPE_HINTS = {
    "general_assistant": "general_assistant",
    "technical_operator": "technical_architect",
    "supportive_listener": "personal_companion",
    "careful_decider": "general_assistant",
}


def persona_from_scope_hint(hint: str) -> str | None:
    return _PERSONA_SCOPE_HINTS.get(hint)


def _selection_ref(payload: dict) -> str:
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"psel_{sha256(material.encode()).hexdigest()[:32]}"


def _selection_scope(body) -> dict:
    return {key: getattr(body, key) for key in (
        "request_id", "owner_id", "conversation_id", "surface", "runtime_session_id",
        "runtime_turn_id", "expected_thread_revision",
    )}


def _strict_bound_persona(surface: str):
    repository = companion_contracts_repository()
    try:
        record = _surface_binding(surface)
        binding = SurfaceBinding.model_validate(record.__dict__, strict=True)
        if any(not getattr(binding, key).strip() for key in (
            "surface_id", "surface_type", "default_persona_id",
        )):
            raise ValueError("persona_surface_binding_invalid")
        if binding.surface_id not in {surface, "unknown"}:
            raise ValueError("persona_surface_binding_invalid")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("persona_surface_binding_invalid") from exc
    except RuntimeError as exc:
        if str(exc) == "default_surface_binding_missing":
            raise ValueError("persona_surface_binding_unavailable") from exc
        raise
    unknown = binding.surface_id != surface or binding.surface_id == "unknown"
    persona = repository.persona_profile(binding.default_persona_id)
    if persona is None:
        persona = repository.default_persona_profile()
        return (
            binding, PersonaProfile.model_validate(persona.__dict__, strict=True),
            "conservative_fallback", "bound_persona_unavailable",
        )
    persona = PersonaProfile.model_validate(persona.__dict__, strict=True)
    return binding, persona, (
        "conservative_fallback" if unknown else "surface_binding"
    ), ("unknown_surface_default" if unknown else "surface_default")


def _contextual_proposal(events, request_id: str):
    governance = [
        event for event in events if event.event_type == "interaction_governance_evaluated"
    ]
    if not governance:
        raise ValueError("persona_selection_governance_missing")
    event = governance[-1]
    payload = event.event_payload_json
    if payload.get("request_id") != request_id:
        raise ValueError("persona_selection_governance_mismatch")
    kind = payload.get("interaction_kind")
    if type(kind) is not str or kind not in get_args(InteractionGovernanceKind):
        raise ValueError("persona_selection_governance_invalid")
    hint = persona_scope_hint_for_kind(kind)
    if hint is None:
        return event, None, "none", "none", "no_contextual_proposal"
    proposed = persona_from_scope_hint(hint)
    if proposed is None or companion_contracts_repository().persona_profile(proposed) is None:
        return event, proposed, "interaction_governance", "rejected", "proposal_unmapped"
    return (
        event, proposed, "interaction_governance", "advisory", "contextual_activation_not_enabled",
    )


def resolve_persona_selection(body: RuntimeIdentityResolveRequest):
    """Only binding authority activates a persona; classification stays advisory."""
    scope = _selection_scope(body)
    repository = runtime_state_repository()
    session, revision, events = repository.persona_selection_events(**scope)
    governance, proposed, proposal_source, status, reason = _contextual_proposal(
        events, body.request_id,
    )
    binding, persona, source, selection_reason = _strict_bound_persona(body.surface)
    payload = {
        **{key: value for key, value in scope.items() if key != "expected_thread_revision"},
        "thread_revision": revision, "governance_event_ref": governance.event_id,
        "active_persona_id": persona.persona_id, "selection_source": source,
        "selection_reason": selection_reason, "proposed_persona_id": proposed,
        "proposal_source": proposal_source, "proposal_status": status, "proposal_reason": reason,
        "requested_selection_status": (
            "unverified_rejected" if body.requested_persona_id is not None else "not_requested"
        ),
        "contextual_activation": False, "explicit_selection_verified": False,
    }
    selection = PersonaSelectionDecision(selection_ref=_selection_ref(payload), **payload)
    repository.persona_selection_events(**scope, selection=selection)
    return session, binding, persona, selection


def consume_persona_selection_evidence(body, *, include_containment: bool = False):
    scope = _selection_scope(body)
    session, revision, events = runtime_state_repository().persona_selection_events(
        **scope, include_containment=include_containment,
    )
    candidates = [event for event in events if event.event_type == "persona_selection_resolved"]
    if len(candidates) != 1:
        raise ValueError("persona_selection_not_found")
    try:
        selection = PersonaSelectionDecision.model_validate(candidates[0].event_payload_json)
    except (ValueError, TypeError) as exc:
        raise ValueError("persona_selection_authority_invalid") from exc
    material = selection.model_dump(exclude={"selection_ref"})
    if selection.selection_ref != body.persona_selection_ref or (
        _selection_ref(material) != selection.selection_ref
    ):
        raise ValueError("persona_selection_reference_mismatch")
    if any(getattr(selection, key) != value for key, value in scope.items() if (
        key != "expected_thread_revision"
    )) or selection.thread_revision != revision:
        raise ValueError("persona_selection_binding_mismatch")
    governance, proposed, proposal_source, status, reason = _contextual_proposal(
        events, body.request_id,
    )
    binding, persona, source, selection_reason = _strict_bound_persona(body.surface)
    if (
        selection.governance_event_ref, selection.proposed_persona_id, selection.proposal_source,
        selection.proposal_status, selection.proposal_reason, selection.active_persona_id,
        selection.selection_source, selection.selection_reason,
    ) != (
        governance.event_id, proposed, proposal_source, status, reason, persona.persona_id,
        source, selection_reason,
    ):
        raise ValueError("persona_selection_authority_changed")
    if body.active_persona_id is not None and body.active_persona_id != selection.active_persona_id:
        raise ValueError("persona_selection_persona_mismatch")
    if getattr(body, "requested_persona_id", None) is not None:
        raise ValueError("persona_selection_override_unverified")
    if getattr(body, "persona_scope_hint", None) is not None and (
        persona_from_scope_hint(body.persona_scope_hint) != selection.proposed_persona_id
        or selection.proposal_status != "advisory"
    ):
        raise ValueError("persona_selection_hint_mismatch")
    return session, selection, events


def consume_persona_selection(body):
    session, selection, _ = consume_persona_selection_evidence(body)
    return session, selection


def _surface_binding(surface: str):
    repository = companion_contracts_repository()
    binding = repository.surface_binding(surface)
    if binding is not None:
        return binding
    fallback = repository.surface_binding("unknown")
    if fallback is None:
        raise RuntimeError("default_surface_binding_missing")
    return fallback


def _persona_record(
    *,
    requested_persona_id: str | None,
    binding: SurfaceBinding,
    allow_requested_persona_bypass: bool,
):
    repository = companion_contracts_repository()
    if requested_persona_id is not None:
        persona = repository.persona_profile(requested_persona_id)
        if persona is not None:
            if allow_requested_persona_bypass:
                return persona, "requested_persona_id", "internal_test"
            if binding.allow_user_persona_override:
                return persona, "requested_persona_id", "surface_binding"
    persona = repository.persona_profile(binding.default_persona_id)
    if persona is not None:
        return persona, "surface_binding", "none"
    return repository.default_persona_profile(), "default_fallback", "none"


def _resolve_identity_session(body: RuntimeIdentityResolveRequest):
    if body.runtime_session_id:
        session = runtime_session_by_id(body.runtime_session_id)
        if (
            session is not None
            and session.owner_id == body.owner_id
            and session.conversation_id == body.conversation_id
            and session.surface == body.surface
        ):
            return session
    return resolve_runtime_session(
        request_id=body.request_id,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        surface=body.surface,
        surface_session_id=body.surface_session_id,
        active_mode=body.active_mode,
    )


def _identity_content(*, persona: PersonaProfile, binding: SurfaceBinding) -> str:
    memory_scope = ",".join(persona.advisory_memory_scope_summary[:3]) or "none"
    tools = ",".join(persona.advisory_tool_permission_summary[:3]) or "none"
    return (
        "Runtime identity: "
        f"persona={persona.persona_id}; "
        f"surface={binding.surface_id}; "
        f"capability_domain={persona.capability_domain}; "
        f"advisory_memory_scope={memory_scope}; "
        f"advisory_tools={tools}; "
        "persona_owns_durable_memory=false."
    )


def resolve_runtime_identity(
    body: RuntimeIdentityResolveRequest,
) -> RuntimeIdentityResolveResponse:
    selection = None
    if body.persona_selection_mode == "strict":
        session, binding, persona_record, selection = resolve_persona_selection(body)
        resolution_reason = (
            "surface_binding" if selection.selection_source == "surface_binding"
            else "default_fallback"
        )
        override_source = "none"
    else:
        session = _resolve_identity_session(body)
        binding_record = _surface_binding(body.surface)
        binding = SurfaceBinding(
            surface_id=binding_record.surface_id,
            surface_type=binding_record.surface_type,
            surface_display_name=binding_record.surface_display_name,
            default_persona_id=binding_record.default_persona_id,
            allow_user_persona_override=binding_record.allow_user_persona_override,
            response_length=binding_record.response_length,
            default_mode=binding_record.default_mode,
        )
        persona_record, resolution_reason, override_source = _persona_record(
            requested_persona_id=body.requested_persona_id,
            binding=binding,
            allow_requested_persona_bypass=body.allow_requested_persona_bypass,
        )
    persona = PersonaProfile(
        persona_id=persona_record.persona_id,
        display_name=persona_record.display_name,
        capability_domain=persona_record.capability_domain,
        description=persona_record.description,
        communication_policy_summary=persona_record.communication_policy_summary,
        runtime_policy_summary=persona_record.runtime_policy_summary,
        advisory_memory_scope_summary=persona_record.advisory_memory_scope_summary,
        advisory_tool_permission_summary=persona_record.advisory_tool_permission_summary,
        persona_owns_durable_memory=False,
    )
    identity = RuntimeIdentityContext(
        active_persona_id=persona.persona_id,
        surface_id=binding.surface_id,
        surface_type=binding.surface_type,
        surface_display_name=binding.surface_display_name,
        capability_domain=persona.capability_domain,
        communication_policy_summary=persona.communication_policy_summary,
        runtime_policy_summary=persona.runtime_policy_summary,
        advisory_memory_scope_summary=persona.advisory_memory_scope_summary,
        advisory_tool_permission_summary=persona.advisory_tool_permission_summary,
        persona_owns_durable_memory=False,
        content=_identity_content(persona=persona, binding=binding),
    )
    trace = RuntimeIdentityTrace(
        runtime_session_id=session.runtime_session_id,
        active_persona_id=persona.persona_id,
        persona_resolution_reason=resolution_reason,
        persona_override_source=override_source,
        surface_id=binding.surface_id,
        surface_type=binding.surface_type,
        surface_display_name=binding.surface_display_name,
        persona_owns_durable_memory=False,
        advisory_memory_scope_summary=persona.advisory_memory_scope_summary,
        advisory_tool_permission_summary=persona.advisory_tool_permission_summary,
    )
    return RuntimeIdentityResolveResponse(
        runtime_session=session,
        surface_binding=binding,
        persona=persona,
        runtime_identity=identity,
        trace=trace,
        selection_contract="strict_turn" if selection else "legacy_unbound",
        persona_selection=selection,
    )
