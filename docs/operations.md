# Operations

This guide covers local startup, SQLite persistence, trusted world-state
verifiers, relationship bootstrap, and durable troubleshooting. Use
[`api/.env.example`](../api/.env.example) as the configuration reference.

## Local configuration and startup

Local configuration lives in `api/.env`. The service reads settings from its
process environment, so export the file before starting it:

```bash
python3 -m venv api/.venv
make dev-install
cp api/.env.example api/.env
set -a
. api/.env
set +a
make dev-start
```

`make dev-start` runs the API at `http://127.0.0.1:4371` by default. The
container image listens on port `8000`. Use `GET /healthz` to verify the
service process.

## Companion-contract storage

Companion profiles, scenes, policies, and related contracts are stored in
SQLite. `COMPANION_CONTRACTS_DB_PATH` selects the database file.

- Local default, relative to the `api/` working directory:
  `./data/companion_contracts.sqlite3`.
- Image default: `/data/companion_contracts.sqlite3`.
- The parent directory is created automatically, but the service process must
  be able to write to the directory and database file.

The deployed mount and `COMPANION_CONTRACTS_DB_PATH` must describe the same
location. If the path is `/data/companion_contracts.sqlite3`, mount persistent
storage at `/data`. A mismatch or read-only mount commonly appears as
`sqlite3.OperationalError: unable to open database file` during companion
repository initialization or compilation.

## Trusted world-state verifiers

`TRUSTED_WORLD_STATE_VERIFIERS_PATH` points to the YAML registry used for
world-state verification and revalidation. The current example is
[`api/config/trusted_world_state_verifiers.example.yaml`](../api/config/trusted_world_state_verifiers.example.yaml).
The example is not activated unless the environment variable points to it.

The YAML document contains exactly one top-level `verifiers` list. Each entry
identifies a verifier and its bounded policy:

- verifier ID and verification source type;
- allowed source references and domains;
- optional allowed attributes and entity IDs;
- maximum authority, confidence, and freshness;
- maximum TTL and revalidation interval.

If the environment variable is absent, the registry is empty and trusted
verification cannot proceed. A missing, malformed, duplicated, or invalid
registry fails closed with the bounded `trusted_verifier_registry_invalid`
reason on relevant service paths. The registry describes trust policy only;
it must never contain credentials, access tokens, endpoints with embedded
secrets, or private payloads.

## Relationship bootstrap

The packaged example seed is
[`api/config/relationships.seed.example.yaml`](../api/config/relationships.seed.example.yaml).
From the repository root, run:

```bash
api/.venv/bin/python \
  api/scripts/bootstrap_relationships.py \
  api/config/relationships.seed.example.yaml \
  --owner-id <owner>
```

The equivalent command from `api/` is:

```bash
cd api
./.venv/bin/python \
  scripts/bootstrap_relationships.py \
  config/relationships.seed.example.yaml \
  --owner-id <owner>
```

Add `--dry-run` to validate and simulate without writing. Applying the same
trusted seed is idempotent for its entities, relationships, and duplicate
evidence records.

Bootstrap tooling seeds explicitly trusted records only. It does not perform
runtime inference, alter retrieval filtering, or create social-context items
implicitly.

## Troubleshooting

1. Confirm `GET /healthz` returns `status: ok` and the expected service name.
2. Confirm the companion SQLite path resolves inside the service environment
   and its parent directory is writable by the service user.
3. If trusted world-state verification is enabled, confirm the configured YAML
   exists, is readable, and matches the current bounded schema.
4. Call the affected policy endpoint directly to distinguish local policy
   failure from orchestration or provider degradation.
5. Verify `POST /v1/companion/profile/compile` against the deployed companion
   database when a companion layer is missing.
6. When Chat Orchestrator consumes Cognitive Runtime, inspect the corresponding
   bounded request trace through Basic Memory Store at
   `GET /v1/traces/{request_id}`.

Optional integration failures must remain bounded. Missing verifier policy
fails closed; unavailable optional orchestration inputs may be omitted or
degraded only according to their existing service contract.
