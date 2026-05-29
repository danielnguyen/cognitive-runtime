# cognitive-runtime

Cluster 7.5 keeps this repo as an R40 runtime-state boundary scaffold only.

The runtime-state boundary may eventually own temporary, inspectable interaction state such as active scene, interaction mode, temporary task state, reset semantics, and trace references. That state must remain separate from canonical memory in `basic-memory-store` and from prompt assembly in `chat-orchestrator`.

Out of scope for Cluster 7.5:

- service runtime or API server
- state machine implementation
- worker or persistence layer
- full conversational runtime engine
- Phase 3 R41/R42 live-state, turn negotiation, timing, pause, backchannel, or interruption behavior

## Local run

1. Install requirements from `api/requirements.txt`
2. Copy `api/.env.example` to `api/.env` if local environment overrides are needed
3. Run `make dev-start` from repo root, or `uvicorn main:app --host 0.0.0.0 --port 4371 --reload` from `api/`

For local host-run, `api/.env` is reserved as the canonical app config location. This scaffold currently has no required environment variables.

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
