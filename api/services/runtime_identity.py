from __future__ import annotations

from models import (
    PersonaProfile,
    RuntimeIdentityContext,
    RuntimeIdentityResolveRequest,
    RuntimeIdentityResolveResponse,
    RuntimeIdentityTrace,
    SurfaceBinding,
)
from services.companion_contracts import companion_contracts_repository
from services.runtime_state import resolve_runtime_session, runtime_session_by_id


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
    repository = companion_contracts_repository()
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
    )
