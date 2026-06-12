# cognitive-runtime

`cognitive-runtime` owns runtime overlays, companion contract compilation, interrupt evaluation, and runtime diagnostic surfaces.

This repo defines runtime-state and companion-policy boundaries that remain separate from canonical memory in `basic-memory-store` and prompt assembly in `chat-orchestrator`.

## Current Responsibilities

- compile companion profile and policy inputs
- serve runtime overlay APIs used by downstream orchestration
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

## Relationship bootstrap

Trusted project and system relationships can be bootstrapped from a seed file without using runtime mutation curl calls.

Example command:

```bash
cd api && ./.venv/bin/python scripts/bootstrap_relationships.py ../config/relationships.seed.example.yaml --owner-id <owner>
```

Dry-run validation:

```bash
cd api && ./.venv/bin/python scripts/bootstrap_relationships.py ../config/relationships.seed.example.yaml --owner-id <owner> --dry-run
```

The bootstrap script is idempotent for seeded entities, relationships, and duplicate evidence records. It is bootstrap tooling only and does not perform runtime inference, retrieval filtering, or social-context creation.
