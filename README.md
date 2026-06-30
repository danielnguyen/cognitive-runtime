# cognitive-runtime

`cognitive-runtime` owns runtime overlays, interaction governance evaluation, companion contract compilation, interrupt evaluation, and runtime diagnostic surfaces.

This repo defines runtime-state and companion-policy boundaries that remain separate from canonical memory in `basic-memory-store` and prompt assembly in `chat-orchestrator`.

## Current Responsibilities

- compile companion profile and policy inputs
- serve runtime overlay APIs used by downstream orchestration
- evaluate interaction governance for the current user turn
- evaluate persona containment for the current user turn
- evaluate restraint for the current user turn
- evaluate interrupt-related runtime signals exposed by the current service
- expose health and diagnostic surfaces for operators

## Local run

1. Install requirements from `api/requirements.txt`
2. Copy `api/.env.example` to `api/.env` if local environment overrides are needed
3. Run `make dev-start` from repo root, or `uvicorn main:app --host 0.0.0.0 --port 4371 --reload` from `api/`

For local host-run, `api/.env` is reserved as the canonical app config location.

### Companion contract storage

`cognitive-runtime` persists companion contract data in SQLite.

- Config key: `COMPANION_CONTRACTS_DB_PATH`
- Local default: `./data/companion_contracts.sqlite3`
- Image default: `/data/companion_contracts.sqlite3`
- The app creates the parent directory automatically if it does not exist.
- The configured DB path must match a mounted writable path in the deployed container.
- If the container mounts persistent storage at `/app/data`, set `COMPANION_CONTRACTS_DB_PATH=/app/data/companion_contracts.sqlite3`.
- If you keep `COMPANION_CONTRACTS_DB_PATH=/data/companion_contracts.sqlite3`, mount the persistent volume at `/data`.
- A common deployment failure is `sqlite3.OperationalError: unable to open database file` when the configured DB path and mounted volume path do not match.

### Operator checks

Normal request flow:

`surface/client -> chat-orchestrator POST /v1/chat -> basic-memory-store/cognitive-runtime/LiteLLM as downstream services`

Ownership summary:
- `chat-orchestrator` owns normal chat request handling.
- `basic-memory-store` owns durable conversation, retrieval, and trace persistence.
- `cognitive-runtime` owns runtime overlay, interaction governance evaluation, companion contract compilation, interrupt evaluation, and diagnostic surfaces.

Post-deploy smoke checklist:
- `GET /healthz` returns success from `cognitive-runtime`.
- `POST /v1/runtime/interaction-governance/evaluate` returns a typed governance result for a normal question and a tense debugging report.
- `POST /v1/runtime/persona-containment/evaluate` returns a typed containment result with active persona, seeded domain summaries, and summarized cross-scope diagnostics.
- `POST /v1/runtime/restraint/evaluate` returns a typed restraint result with primary policy, affected domains, and summarized suppression signals.
- `chat-orchestrator` `POST /v1/chat` returns a valid response for a normal request.
- `POST /v1/companion/profile/compile` succeeds with the deployed `COMPANION_CONTRACTS_DB_PATH`.
- `POST /v1/runtime/overlay` is reachable when runtime overlay integration is enabled.
- The request trace is visible through `basic-memory-store` `GET /v1/traces/{request_id}`.

If answers behave oddly, check:
- whether `COMPANION_CONTRACTS_DB_PATH` points to a writable mounted path
- whether interaction governance is classifying the current turn and recording only summarized diagnostics
- whether companion compile is failing and being traced as an omitted companion layer
- whether runtime overlay calls are unavailable
- the corresponding request trace in `basic-memory-store`

## Health check

- `GET /healthz`
- Returns:
  - `status`
  - `service`

## Interaction Governance Endpoint

- `POST /v1/runtime/interaction-governance/evaluate`
- Returns a typed governance result for the current user turn, including interaction kind, posture, non-actioning flags, and summarized reasons.
- When a runtime session and turn already exist, the evaluation also updates runtime-turn diagnostics and records a summarized governance event.

## Persona Containment Endpoint

- `POST /v1/runtime/persona-containment/evaluate`
- Returns a typed containment result for the current user turn, including active persona, seeded containment domain summaries, and explicit-only cross-scope allowance signals.
- When a runtime session is available, the evaluation records a summarized persona containment event without storing raw user text.

## Restraint Endpoint

- `POST /v1/runtime/restraint/evaluate`
- Returns a typed restraint result for the current user turn, including primary restraint policy, affected domains, a short operational prompt overlay, and summarized suppression signals.
- When a runtime session is available, the evaluation records a summarized restraint event without storing raw user text.

## Local validation

From repo root:

```bash
make dev-lint
make dev-test
make dev-check
make replay-test
```

`make replay-test` runs the versioned runtime replay corpus in
`api/replay/runtime/v1`. Each scenario uses a clean disposable SQLite database,
executes the real runtime service boundary, normalizes generated identifiers and
timestamps, and compares the structural result with its persisted expected
snapshot. It requires no deployed stack, external provider, network service, or
credentials.

`make dev-start` serves the API on `APP_PORT`, defaulting to `4371`.

For a lightweight operator smoke check against an already-running local service:

```bash
make smoke
```

`make smoke` uses `curl` and `jq`. It checks `/healthz`, exercises the interaction governance, persona containment, and restraint endpoints across representative inputs, and verifies a simple runtime turn/session integration path. Deeper diagnostics review remains a manual operator check.

Manual operator validation for persona containment and restraint diagnostics:
- runtime event payloads for persona containment and restraint must contain only allowed summary fields
- no raw user text, raw private memory, hidden reasoning, raw exception text, or implementation-planning identifiers inside those payloads

## Relationship bootstrap

Trusted project and system relationships can be bootstrapped from a seed file without using runtime mutation curl calls.

Example command:

```bash
python scripts/bootstrap_relationships.py config/relationships.seed.example.yaml --owner-id <owner>
```

Local development:

```bash
cd api && ./.venv/bin/python scripts/bootstrap_relationships.py config/relationships.seed.example.yaml --owner-id <owner>
```

Dry-run validation:

```bash
cd api && ./.venv/bin/python scripts/bootstrap_relationships.py config/relationships.seed.example.yaml --owner-id <owner> --dry-run
```

The bootstrap script is idempotent for seeded entities, relationships, and duplicate evidence records. It is bootstrap tooling only and does not perform runtime inference, retrieval filtering, or social-context creation.

## Relationship and social-context mutation errors

The relationship and social-context mutation endpoints return bounded error bodies for expected policy failures:

- `400`: `relationship_source_refs_required`, `model_inference_cannot_create_active_relationship`, `model_inference_socialish_relationship_requires_confirmation`, `relationship_confirmation_evidence_required`, `social_context_source_refs_required`
- `403`: `trusted_provenance_required_for_active_socialish_relationship`
- `404`: `relationship_entity_not_found`, `relationship_edge_not_found`, `social_context_item_not_found`
- `409`: `relationship_edge_status_not_confirmable`, `social_context_requires_approved_relationship_edge`

Error responses use `{"detail": "<reason_code>"}` and do not include entity labels, evidence summaries, source refs, storage details, stack traces, or unrelated record IDs. Unknown runtime failures are not converted into these public reason codes.
