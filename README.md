# cognitive-runtime

`cognitive-runtime` owns the R40 runtime-state boundary scaffold and the current Cluster 8 companion policy scaffold. The companion policy compiler includes a practical R19 static/default interaction contract substrate so downstream R24 work can inspect boundary and repair constraints without hardcoding them into prompt text.

The runtime-state boundary may eventually own temporary, inspectable interaction state such as active scene, interaction mode, temporary task state, reset semantics, and trace references. That state must remain separate from canonical memory in `basic-memory-store` and from prompt assembly in `chat-orchestrator`. The R19 contract substrate remains static/default in this pass: it is not personalized, persisted, or full R17/R18/R19 completion.

Out of scope for Cluster 7.5:

- service runtime or API server
- state machine implementation
- worker or persistence layer
- full conversational runtime engine
- Phase 3 R41/R42 live-state, turn negotiation, timing, pause, backchannel, or interruption behavior
- R24 interrupt detection, grounding execution, trigger evaluation, or interrupt event persistence

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
- `cognitive-runtime` owns runtime overlay, companion contract compilation, interrupt evaluation, and diagnostic surfaces.

Post-deploy smoke checklist:
- `GET /healthz` returns success from `cognitive-runtime`.
- `chat-orchestrator` `POST /v1/chat` returns a valid response for a normal request.
- `POST /v1/companion/profile/compile` succeeds with the deployed `COMPANION_CONTRACTS_DB_PATH`.
- `POST /v1/runtime/overlay` is reachable when runtime overlay integration is enabled.
- The request trace is visible through `basic-memory-store` `GET /v1/traces/{request_id}`.

If answers behave oddly, check:
- whether `COMPANION_CONTRACTS_DB_PATH` points to a writable mounted path
- whether companion compile is failing and being traced as an omitted companion layer
- whether runtime overlay calls are unavailable
- the corresponding request trace in `basic-memory-store`

## Health check

- `GET /healthz`
- Returns:
  - `status`
  - `service`

## Local validation

From repo root:

```bash
make dev-lint
make dev-test
make dev-check
```

`make dev-start` serves the API on `APP_PORT`, defaulting to `4371`.
