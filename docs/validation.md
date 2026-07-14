# Validation

Run validation from the repository root with the Python environment at
`api/.venv`. Use repository-local or disposable data only; never point these
commands at production state or credentials.

## Supported commands

| Command | Purpose |
| --- | --- |
| `make dev-lint` | Runs Ruff against the API source and tests. |
| `make dev-test` | Runs the complete pytest suite in the local environment. |
| `make dev-check` | Runs lint followed by the complete pytest suite. |
| `make replay-test` | Runs deterministic runtime scenarios against disposable SQLite databases. |
| `make process-naming-check` | Checks added source text for reserved internal naming. |
| `make smoke` | Exercises an already-running service through its local HTTP API. |

## Deterministic replay

`make replay-test` loads the repository-local scenarios in
[`api/replay/runtime/v1`](../api/replay/runtime/v1). Each scenario uses clean
SQLite files under a temporary directory, normalizes generated identifiers and
timestamps, and compares the result with its stored structural snapshot.

Replay requires no deployed stack, external provider, network service, or
credentials. Temporary databases are removed when the run finishes.

## Local smoke validation

`make smoke` expects Cognitive Runtime to be running. It uses
`http://127.0.0.1:4371` by default; override the base URL with `CR_BASE`:

```bash
CR_BASE=http://127.0.0.1:4371 make smoke
```

The smoke command checks health, interaction governance, persona containment,
restraint, and a runtime turn/session integration path. It uses fixed local
payloads and requires `curl` and `jq`; it does not start the service.

## Operator checks

- Check `GET /healthz` before evaluating deeper behavior.
- Verify the companion SQLite path and mounted directory are writable.
- Exercise each enabled policy endpoint directly with bounded non-production
  inputs.
- Verify `POST /v1/companion/profile/compile` when companion composition is in
  use.
- When Cognitive Runtime is consumed by Chat Orchestrator, inspect the related
  bounded request trace through Basic Memory Store at
  `GET /v1/traces/{request_id}`.

## Privacy and failure expectations

Persisted runtime event payloads must not contain raw user text, private
memory, hidden reasoning, credentials, raw exception text, or internal
delivery identifiers. Validation must not use production data, databases, or
credentials.

Optional integration failures must remain bounded and must fail closed or
degrade according to their documented contract. Validation output should not
include unrestricted dependency responses or stack traces.
