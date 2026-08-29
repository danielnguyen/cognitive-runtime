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
| Resolve a shared conversation projection | `POST /v1/runtime/threads/resolve` |
| Select a bounded continuation candidate | `POST /v1/runtime/continuations/select` |
| Reserve safe retirement coordination | `POST /v1/runtime/retirements/reserve` |
| Cancel retirement coordination | `POST /v1/runtime/retirements/cancel` |
| Finalize retirement coordination | `POST /v1/runtime/retirements/finalize` |
| Resolve a session | `POST /v1/runtime/sessions/resolve` |
| Read session diagnostics | `GET /v1/runtime/sessions/{runtime_session_id}` |
| Start a turn | `POST /v1/runtime/turns/start` |
| Update a turn | `POST /v1/runtime/turns/update` |
| Complete a turn | `POST /v1/runtime/turns/complete` |

One shared projection is persisted for each owner and conversation. Surface
sessions remain separate, so each surface retains its own runtime session,
surface session identifier, and operational state while participating in the
same projection. Resolving a projection does not assert that the durable
conversation exists, belongs to the caller, or may be resumed. Exact durable
authorization remains an upstream responsibility.

`POST /v1/runtime/threads/resolve` accepts `request_id`, `owner_id`, and
`conversation_id`. Its bounded response contains the owner and conversation,
thread state, non-negative revision, optional active runtime session, turn, and
surface, sorted participating surfaces, participating session count, and
timestamps. Thread state is one of `idle`, `active`, `contended`, or
`unavailable`. The response contains no message content, provider output,
semantic memory, or durable lifecycle decision.

Turn admission is atomic across all surface sessions for the owner and
conversation. `POST /v1/runtime/turns/start` accepts the optional non-negative
`expected_thread_revision`. A matching revision admits an idle thread; a stale
revision returns `runtime_thread_revision_conflict`. Callers may omit the field
for compatibility. While a turn is active, another request or surface receives
`runtime_thread_contended`. An exact retry from the same surface and request
returns the existing admitted turn without advancing the revision.

Admission advances the shared revision once. Ordinary turn updates do not
advance it. Completion or abandonment releases the active admission and
advances it once; an exact terminal retry does neither again. Active admission,
terminal release, and revisions persist across repository and service restart.
Preexisting state is reconstructed conservatively: no active turn becomes
`idle`, one becomes `active`, and multiple active turns become `contended`.
Inconsistent stored associations are exposed as `unavailable` without choosing
or deleting a turn.

Expected runtime-state failures use bounded reason codes. Contention, revision
conflicts, and a non-current turn return `409`; session or turn absence returns
`404`; a session/turn mismatch is a bounded client error; unavailable or failed
persistence returns `503`. Responses do not expose storage paths, statements,
exception details, identifiers from another surface, or turn content.

Session diagnostics contain bounded state, turn, and event summaries. Runtime
state, projections, and overlays are operational context, not canonical durable
memory. Shared admission coordinates active work; it does not itself decide
whether a durable conversation is eligible to resume.

`POST /v1/runtime/continuations/select` evaluates a caller-supplied bounded set
of owner-authorized durable conversation facts against existing runtime-thread
state. The strict request contains `request_id`, `owner_id`, the current
`surface`, an explicit `candidate_set_complete` boolean, a bounded
`stale_after_seconds` interval, and at most eight unique candidates. Each
candidate contains only `conversation_id`, durable lifecycle (`open`, `closed`,
or `superseded`), and a timezone-aware durable update timestamp. The endpoint
does not accept titles, content, summaries, client identifiers, message counts,
semantic scores, embeddings, provider output, or adapter state.

The response uses schema `runtime-continuation-selection.v1` and returns one
policy result: `resume`, `create_new`, `clarify`, `wait`, or `decline`. A
complete empty candidate set authorizes `create_new`. Exactly one fresh open
candidate may return `resume` only when an existing, internally consistent idle
thread has at least one participating runtime session. Multiple eligible
candidates clarify; an active candidate waits; contended, unavailable, or
inconsistent state declines. Missing runtime state, missing participating
sessions, stale activity, and non-open durable lifecycle cannot resume a
candidate. Durable recency alone never establishes eligibility.

Only `resume` returns a conversation identifier and the exact inspected thread
revision. The revision is read-only advisory input for normal atomic turn
admission: it is not incremented, reserved, or guaranteed to remain current.
Non-resume outcomes disclose no candidate identifiers, runtime session or turn
identifiers, timestamps, active surfaces, or participating surfaces.

Selection uses one read-only SQLite transaction and never creates or updates a
conversation, runtime thread, runtime session, turn, event, revision, or
activity timestamp. Cognitive Runtime does not call a durable store, provider,
semantic retrieval system, model, or adapter while evaluating candidates. The
current surface is response context only; it is not treated as a permission,
presence, attention, or preferred-device signal.

`POST /v1/runtime/retirements/reserve` accepts only `request_id`, `owner_id`,
`conversation_id`, durable lifecycle, `durable_updated_at`, and an absolute
`retirement_before` cutoff. Both timestamps must carry a timezone or UTC
offset. The durable lifecycle must be `open`, `closed`, or `superseded`. A new
reservation is possible only for an `open` candidate whose durable activity
and existing runtime thread activity are both strictly earlier than the
cutoff. Equality is not over the horizon. Missing, inconsistent, contended, or
unavailable runtime state declines; an active runtime thread waits. The
request accepts no content, client, surface, provider, model, retrieval, or
adapter fields.

A successful reserve response uses schema
`runtime-retirement-reservation.v1` and policy
`conversation-retirement-safety.v1`. It returns an internal reservation ID,
the exact inspected thread revision, and the durable activity instant captured
for later durable compare-and-set enforcement. A reservation is persistent,
unique per owner and conversation, and has no TTL, expiry, background cleanup,
or autonomous release. A valid existing reservation is returned without being
replaced, even when a later request carries a different durable activity fact.
Wait and decline results expose none of the reservation ID, captured revision,
or captured durable activity fields.
The durable store remains responsible for deciding whether that captured fact
still permits a lifecycle transition.

Reservation acquisition, turn admission, and session resolution serialize at
the same SQLite writer boundary. While a reservation exists,
`POST /v1/runtime/turns/start` and
`POST /v1/runtime/sessions/resolve` fail with
`runtime_thread_retirement_reserved` before creating or changing a session,
turn, event, thread revision, or thread activity.

`POST /v1/runtime/retirements/cancel` requires the exact live reservation ID
and captured thread revision for the owner and conversation. It removes the
coordination fence without changing thread state, revision, or activity.
`POST /v1/runtime/retirements/finalize` also requires the exact live
reservation and a still-consistent idle thread with no nonterminal turn. In
one transaction it advances the thread revision exactly once, leaves
`last_activity_at` unchanged, and removes the reservation. That revision
advance invalidates admission decisions bound to the pre-retirement revision.

After cancellation or successful finalization removes the reservation, a
repeated cancel or finalize returns a bounded missing-reservation error and
makes no inference from the current thread revision. There is no finalization
receipt or permanent runtime retirement marker. An ambiguous finalize
transport outcome remains safe: either the live reservation continues to
block admission, or the successful revision fence rejects an admission bound
to the earlier revision.

These endpoints coordinate runtime admission only. Cognitive Runtime neither
calls the durable store nor marks a conversation closed. The durable store
remains lifecycle authority, and the reservation ID is not a client- or
adapter-visible conversation identity.

## Runtime policy evaluation

| Responsibility | Endpoint |
| --- | --- |
| Interaction governance | `POST /v1/runtime/interaction-governance/evaluate` |
| Persona containment | `POST /v1/runtime/persona-containment/evaluate` |
| Restraint | `POST /v1/runtime/restraint/evaluate` |
| Situated presence | `POST /v1/runtime/situated-presence/evaluate` |
| Privacy context | `POST /v1/runtime/privacy-context/evaluate` |
| Memory hygiene | `POST /v1/runtime/memory-hygiene/evaluate` |
| Claim calibration | `POST /v1/runtime/claim-calibration/evaluate` |
| Generic claim support evaluation | `POST /v1/runtime/claim-support/evaluate` |
| Evidence shape derivation | `POST /v1/runtime/evidence-shapes/derive` |
| Evidence planning | `POST /v1/runtime/evidence-plans/compile` |
| Evidence sufficiency | `POST /v1/runtime/evidence-sufficiency/evaluate` |
| Evidence next-step selection | `POST /v1/runtime/evidence-next-steps/select` |
| Runtime identity | `POST /v1/runtime/identity/resolve` |

These evaluators return typed, bounded decisions for the current context. When
an active runtime session is supplied, supported evaluators may also record a
summarized runtime event.

`POST /v1/runtime/situated-presence/evaluate` requires an existing admitted
runtime session and turn. Its strict request binds request, owner,
conversation, surface, session, and turn identifiers to explicit surface
visibility and constraint facts. It consumes compact projections from the
existing interaction-governance and restraint decisions. Those projections
contain bounded decision fields only; the request accepts no user text, recent
messages, prompt overlay, title, content, provider output, or arbitrary
metadata.

The response uses schema and policy version `situated-presence.v1`. It returns
only commentary and humor permission, bounded emotional-attunement and
challenge levels, a silence preference, the surface commentary gate, response
posture, an always-false action-implication gate, and neutral reason codes. The
result is a response-shaping envelope, not final wording. It contains no joke,
sympathy line, tactical phrase, or response template.

A private, normal, low-risk playful context may permit light commentary and
humor when both upstream decisions allow it. Tense debugging suppresses humor
and commentary while retaining tactical posture. High-impact context suppresses
casual commentary and humor. Private mistake or expression context may permit
brief bounded attunement without labeling or inferring a feeling. Shared,
public, constrained, or unknown surface facts suppress commentary
conservatively while required direct or tactical posture remains available.

Interaction-governance commentary, humor, privacy, and confirmation boundaries
cannot be loosened. Restraint suppression, personalization, brevity, and
clarification inputs can only preserve or tighten the envelope. The endpoint
cannot authorize or imply an action and does not replace confirmation.
Insufficient upstream confidence returns a silent-or-minimal envelope.

Successful evaluation records one `situated_presence_evaluated` event associated
with the existing session and turn. Its payload contains the final gates,
posture, policy version, and reason codes only. Evaluation does not create a
session or turn or change thread, session, turn, revision, activity, intent,
restraint, or status state. Missing or mismatched scope records no event.

The evaluator is deterministic and calls no provider, model, semantic retrieval
system, adapter, or external service. Persistent presence and idle transitions,
a complete timing decision, behavior tuning, watch delivery, and proactive
silence scheduling are not part of this endpoint.

Claim calibration evaluates one caller-selected factual claim against only the
bounded evidence references attributed to that claim. It derives a deterministic
claim classification, evidence strength, confidence, freshness summary,
uncertainty requirement, and safe explanation summary. The evaluator does not
select evidence from a retrieval bundle, verify that referenced records physically
exist, persist a durable claim record, call a model provider, or execute an
external action.

Generic claim support evaluation is an additive provider-free authority path.
The request keeps system-established authority context separate from a shallow
claim proposal. The authority context supplies the exact owner, conversation,
surface, runtime session and turn, authorized evidence identities, claim-relevant
scope and acquisition facts, and actual deterministic derivation records. The
proposal supplies only a bounded claim, supporting or counterevidence identities,
material exclusions, and references to those executed derivations.
The system-owned `claim_scope_basis` authority fact distinguishes a claim about
the full `declared_scope` from a claim already bounded by the trusted caller to
`supplied_evidence`. `declared_scope` is the compatibility default and retains
strict complete-scope enforcement. `supplied_evidence` does not assert that the
broader declared scope was complete; incomplete broader acquisition therefore
requires qualification rather than automatic withholding when no independent
hard blocker exists. Universal and absence conclusions still require the
appropriate complete-scope authority. The claim proposal cannot select this
field, and the evaluation response adds no corresponding field.
One bounded evidence unit may both support a claim and carry a disclosed material
exclusion when only part of that unit was usable. That overlap necessarily
qualifies the result. Supporting and counterevidence roles remain mutually
exclusive, and counterevidence cannot be relabelled as an exclusion.

The evaluator rejects references outside that authority context and returns a
CR-owned `supported`, `limited`, or `unsupported` calibration together with an
`allowed`, `qualified`, or `withheld` conclusion disposition. Complete-scope and
material-disclosure constraints apply only when the system context marks them as
claim-relevant. Arithmetic execution does not upgrade a model-interpreted input
premise into an independently established fact. The endpoint records only a
bounded structural event; it does not log claim prose, source content, prompts,
or hidden reasoning. It does not change the existing claim-calibration,
evidence-planning, sufficiency, or next-step paths.

Evidence shape derivation consumes the existing interaction classification, a
bounded current task statement, and structural evidence-materiality context. It
derives only a broad evidence acquisition shape, or reports that evidence-scope
planning is not applicable or remains ambiguous. It does not create a
product-specific intent catalog, call a model provider, inspect source
inventories, compile an evidence plan, retrieve evidence, persist a plan or
acquisition manifest, or generate or enforce the final answer.

The derivation may identify a bounded targeted lookup from task semantics when
the requested conclusion depends on external verification of a specific
compatibility relationship, exact current implementation or deployment state,
or whether material review or validation was performed. This admission does not
make evidence planning universal. Ordinary casual, creative, explanatory,
writing, brainstorming, and playful requests remain outside the path unless
they contain a distinct material verification request.

Evidence planning accepts an already-resolved broad task shape, a bounded
declared scope, and a normalized governed source inventory. It declares bounded
evidence requirements, derives completeness and contradiction expectations, and
selects a deterministic capability-aware acquisition strategy. The compiler does
not treat a source registry ID as an individual record: source IDs narrow the
eligible source registries, while an exact fetch requires explicit opaque exact
source references associated with those registries. Exact references are bounded
data, not URLs or source content, and source IDs alone never select exact fetch.
Source capability is necessary but not sufficient for a plan to be `ready`.
Readiness means the declared source scope satisfies the planning constraints and
the selected acquisition strategy is structurally supported by the eligible
sources' capabilities and reference boundary. It is not a promise that every
caller implements every strategy. Callers remain responsible for executing only
strategies they implement while preserving the compiled plan and scope
association. An `unsupported` plan still retains its task-shape requirements and
may retain a deterministic candidate strategy that failed an intrinsic
prerequisite. Actual acquisition facts and evidence-sufficiency evaluation
determine whether the declared requirements were satisfied. Cognitive Runtime
does not execute acquisition itself.
The compiler does not derive task shape from user text, call Basic Memory Store
or Data Source Aggregator, execute the planned retrieval or fetch, persist a plan
or acquisition manifest, assess actual acquired evidence, call a model provider,
or generate or enforce the final answer.

Evidence sufficiency compares caller-declared evidence requirements with
caller-supplied acquisition facts. It deterministically reports whether those
facts satisfy the declared scope and returns bounded answer constraints when
the scope is limited, insufficient, or unknown. It does not derive the evidence
plan, perform retrieval, verify source records, persist an acquisition manifest,
call a model provider, generate the final answer, or enforce the constraints in
the final answer path.

Evidence next-step selection is a separate policy operation over a retained
sufficiency evaluation. The caller supplies the exact evaluated requirements,
and the runtime associates their canonical digest with the retained evaluation
event before using that event's status, task shape, and answer constraints. The
selector does not accept caller-selected sufficiency status, provider permission,
conclusion disposition, or next step.

An acquisition premise consists only of the normalized question-anchor digest,
task shape, declared evidence scope, source inventory, and selected strategies.
Request, session, turn, plan, manifest, and evaluation identifiers are excluded,
so equivalent premises remain stable across turns while material question,
scope, inventory, availability, authority, capability, or strategy changes
produce a different digest. Evidence planning retains this privacy-safe premise
digest on the compiled-plan event without adding it to the public planning
response. Next-step selection requires the caller's current premise to match
that retained digest, so callers cannot redefine the premise already attempted.
Only a proposed premise that differs from the actual retained plan premise can
authorize more acquisition.

Sufficient evidence permits an answer within the declared scope. Optional
limitations permit a qualified partial answer without requiring more
acquisition. For insufficient or unknown evidence, a narrow clarification is
selected only for missing or unknown material facts. Additional acquisition is
selected only for a structurally changed premise: an unchanged premise is
blocked immediately, and a changed premise already selected in the same runtime
session is not selected again. A qualified partial answer requires delivered
context plus substantive satisfied or partial evidence; administrative
conditions alone are insufficient. Remaining scope gaps select bounded
unexamined-scope disclosure, while other unsupported conclusions are withheld.
After clarification and changed-premise acquisition opportunities are exhausted
or unavailable, an insufficient or unknown low-risk targeted lookup may withhold
the requested conclusion while permitting the provider to offer bounded,
non-authoritative guidance. Provider permission is not conclusion permission:
the factual conclusion remains unsupported and may not be presented as
verified. Missing or mismatched associated shape evidence remains conservative,
as do high-impact, high-stakes, exhaustive, absence-sensitive,
contradiction-sensitive, historical-completeness, decision-support, and
clarification-dependent branches. Cognitive Runtime returns only the policy
disposition and a bounded summary; it does not generate advisory prose.

Selections are deterministic and idempotent. Their retained runtime events
contain association identifiers, premise digests, bounded dispositions, counts,
guards, and reason codes, but not source identifiers, exact references, premise
contents, provider text, or evidence content. Cognitive Runtime selects policy;
it does not execute acquisition, call a provider, or enforce the result in the
answer path. The public evidence-planning and evidence-sufficiency response
shapes remain unchanged; only privacy-safe association digests are added to
retained events.

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
