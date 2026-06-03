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
- Docker path: `/data/companion_contracts.sqlite3`
- The app creates the parent directory automatically if it does not exist.
- Docker deployments should mount `/data` as a persistent volume to retain seeded/default contract data across container restarts.

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
