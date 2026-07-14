# API and runtime behavior

Cognitive Runtime exposes bounded state, policy, relationship, capability, and
companion operations through FastAPI. The endpoints are grouped below by the
responsibility they serve.

## Runtime lifecycle

Runtime state is scoped by owner, conversation, and surface:

| Operation | Endpoint |
| --- | --- |
| Resolve state | `POST /v1/runtime/state/resolve` |
| Update state | `POST /v1/runtime/state/update` |
| Reset state | `POST /v1/runtime/state/reset` |
| Build a bounded overlay | `POST /v1/runtime/overlay` |

Session and turn operations provide a durable local runtime lifecycle:

| Operation | Endpoint |
| --- | --- |
| Resolve a session | `POST /v1/runtime/sessions/resolve` |
| Read session diagnostics | `GET /v1/runtime/sessions/{runtime_session_id}` |
| Start a turn | `POST /v1/runtime/turns/start` |
| Update a turn | `POST /v1/runtime/turns/update` |
| Complete a turn | `POST /v1/runtime/turns/complete` |

Session diagnostics contain bounded state, turn, and event summaries. Runtime
state and overlays are operational context, not canonical durable memory.

## Runtime policy evaluation

| Responsibility | Endpoint |
| --- | --- |
| Interaction governance | `POST /v1/runtime/interaction-governance/evaluate` |
| Persona containment | `POST /v1/runtime/persona-containment/evaluate` |
| Restraint | `POST /v1/runtime/restraint/evaluate` |
| Privacy context | `POST /v1/runtime/privacy-context/evaluate` |
| Memory hygiene | `POST /v1/runtime/memory-hygiene/evaluate` |
| Runtime identity | `POST /v1/runtime/identity/resolve` |

These evaluators return typed, bounded decisions for the current context. When
an active runtime session is supplied, supported evaluators may also record a
summarized runtime event.

## Relationships and context

Relationship entities and edges use explicit owner scope and bounded
provenance:

- `POST /v1/relationships/entities/upsert`
- `GET /v1/relationships/entities/{entity_id}?owner_id={owner_id}`
- `POST /v1/relationships/edges/upsert`
- `POST /v1/relationships/edges/confirm`
- `POST /v1/relationships/edges/revoke`
- `GET /v1/relationships/edges/{relationship_id}?owner_id={owner_id}`
- `POST /v1/relationships/select`
- `POST /v1/relationships/diagnostics`

Social-context operations are:

- `POST /v1/social-context/items/upsert`
- `POST /v1/social-context/usage-events/record`
- `POST /v1/social-context/diagnostics`

Expected relationship and social-context policy failures use bounded
`{"detail":"<reason_code>"}` responses. Missing or invalid provenance is a
`400`, disallowed provenance is a `403`, missing records are a `404`, and
invalid state transitions or dependencies are a `409`. These responses omit
labels, source references, storage details, and unrelated record identifiers.

World-state operations are:

- `POST /v1/world-state/claims/upsert`
- `POST /v1/world-state/claims/verify`
- `POST /v1/world-state/resolve`
- `POST /v1/world-state/diagnostics`

Verification uses the configured trusted-verifier registry and fails closed
when no valid verifier policy applies. Diagnostics exclude sensitive claim
values.

## Capability governance

| Operation | Endpoint |
| --- | --- |
| Authorize a capability | `POST /v1/capabilities/authorize` |
| Match a capability | `POST /v1/capabilities/match` |
| Discover capabilities | `POST /v1/capabilities/discover` |
| Decide authority | `POST /v1/capabilities/authority` |
| Decide action flow | `POST /v1/capabilities/flow` |
| Record confirmation | `POST /v1/capabilities/confirm` |
| Compose an action summary | `POST /v1/capabilities/action-summary` |

Cognitive Runtime owns canonical capability policy, authority, confirmation,
dispatch eligibility, verification policy, and bounded action summaries. It
does not call an external action connector or perform the consequential effect.

## Companion and diagnostics

| Operation | Endpoint |
| --- | --- |
| Read the active companion profile | `GET /v1/companion/profile/active` |
| Resolve a scene | `POST /v1/companion/scene/resolve` |
| Read a scene policy | `GET /v1/companion/scene/{scene_id}` |
| Validate an interaction contract | `POST /v1/companion/interaction-contract/validate` |
| Simulate repair wording | `POST /v1/companion/repair/simulate` |
| Compile a companion profile | `POST /v1/companion/profile/compile` |
| Compile companion policy | `POST /v1/companion/policy/compile` |
| Submit human-compatibility review | `POST /v1/runtime/human-compatibility/review` |
| Read human-compatibility diagnostics | `POST /v1/runtime/human-compatibility/diagnostics` |
| Evaluate an interrupt | `POST /v1/interrupt/evaluate` |

Companion compilation resolves current profile, scene, and interaction-policy
inputs into a bounded contract for downstream orchestration. Repair simulation
and diagnostics do not execute an interrupt or external action.

## General API behavior

`GET /healthz` returns the service name and health status. FastAPI validation
errors and recognized domain failures are bounded; recognized failures expose
stable reason codes rather than internal exception details.

Runtime event payloads must remain summarized. They must not expose raw user
text, private memory, hidden reasoning, credentials, stack traces, raw
exception output, or unrestricted dependency responses.
