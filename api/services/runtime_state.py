from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

from models import RuntimeOverlay, RuntimeState, RuntimeStateUpdate

StateKey = tuple[str, str, str]

_STATE: dict[StateKey, RuntimeState] = {}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _state_id(owner_id: str, conversation_id: str, surface: str) -> str:
    digest = sha256(f"{owner_id}:{conversation_id}:{surface}".encode("utf-8")).hexdigest()[:16]
    return f"rtstate_{digest}"


def _overlay_id(state: RuntimeState, source_fields: list[str]) -> str:
    material = "|".join(
        [
            state.runtime_state_id,
            state.active_scene or "",
            state.interaction_mode or "",
            ",".join(state.temporary_constraints),
            ",".join(source_fields),
        ]
    )
    digest = sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"rtoverlay_{digest}"


def _key(owner_id: str, conversation_id: str, surface: str) -> StateKey:
    return (owner_id, conversation_id, surface)


def resolve_state(*, owner_id: str, conversation_id: str, surface: str) -> RuntimeState:
    key = _key(owner_id, conversation_id, surface)
    if key not in _STATE:
        now = _now()
        _STATE[key] = RuntimeState(
            runtime_state_id=_state_id(owner_id, conversation_id, surface),
            owner_id=owner_id,
            conversation_id=conversation_id,
            surface=surface,
            created_at=now,
            updated_at=now,
        )
    return _STATE[key]


def update_state(
    *,
    owner_id: str,
    conversation_id: str,
    surface: str,
    updates: RuntimeStateUpdate,
) -> RuntimeState:
    state = resolve_state(owner_id=owner_id, conversation_id=conversation_id, surface=surface)
    update_payload = updates.model_dump(exclude_unset=True)
    next_state = RuntimeState(
        **{
            **state.model_dump(),
            **update_payload,
            "updated_at": _now(),
        }
    )
    _STATE[_key(owner_id, conversation_id, surface)] = next_state
    return next_state


def reset_state(*, owner_id: str, conversation_id: str, surface: str) -> RuntimeState:
    state = resolve_state(owner_id=owner_id, conversation_id=conversation_id, surface=surface)
    next_state = state.model_copy(
        update={
            "active_scene": None,
            "interaction_mode": None,
            "attention_focus": None,
            "temporary_constraints": [],
            "reset_after_turn": False,
            "trace_refs": [],
            "updated_at": _now(),
        }
    )
    _STATE[_key(owner_id, conversation_id, surface)] = next_state
    return next_state


def build_overlay(state: RuntimeState) -> tuple[RuntimeOverlay | None, str | None]:
    source_fields: list[str] = []
    parts: list[str] = []
    if state.active_scene:
        parts.append(f"scene={state.active_scene}")
        source_fields.append("active_scene")
    if state.interaction_mode:
        parts.append(f"interaction_mode={state.interaction_mode}")
        source_fields.append("interaction_mode")
    if state.temporary_constraints:
        parts.append(f"constraints={','.join(state.temporary_constraints)}")
        source_fields.append("temporary_constraints")

    if not parts:
        return None, "empty_runtime_state"

    return (
        RuntimeOverlay(
            overlay_id=_overlay_id(state, source_fields),
            runtime_state_id=state.runtime_state_id,
            content=f"Runtime context: {'; '.join(parts)}.",
            source_fields=source_fields,
        ),
        None,
    )


def clear_states_for_tests() -> None:
    _STATE.clear()
