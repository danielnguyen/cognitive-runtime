# Cognitive Runtime

Cognitive Runtime is CCP's bounded runtime-policy service. It manages active
runtime state, evaluates interaction and action policy, compiles companion
contracts, and returns diagnostics suitable for service composition.

## Service boundaries

- Cognitive Runtime owns runtime state, policy decisions, capability
  governance, companion contracts, and bounded runtime diagnostics.
- Chat Orchestrator owns the normal `POST /v1/chat` request and prompt-assembly
  path.
- Basic Memory Store owns durable conversations, retrieval, artifacts, and
  traces.

Cognitive Runtime decides policy for permissioned actions; it does not execute
external connector effects.

## Capabilities

The service provides APIs for:

- runtime sessions, turns, state, and overlays;
- interaction governance, persona containment, restraint, privacy, and memory
  hygiene;
- relationships, social context, world state, and runtime identity;
- capability discovery, authority, confirmation, flow, and action summaries;
- companion profiles, scenes, policy, diagnostics, and interrupt evaluation.

See [API and runtime behavior](docs/api.md) for the current routes.

## Run locally

Create a Python virtual environment and start the service from the repository
root:

```bash
python3 -m venv api/.venv
make dev-install
cp api/.env.example api/.env
set -a
. api/.env
set +a
make dev-start
```

Local configuration lives in `api/.env`. The service defaults to
`http://127.0.0.1:4371`; check it with `GET /healthz`.

## Validation

Primary validation commands are:

```bash
make dev-check
make replay-test
make process-naming-check
make smoke
```

`make smoke` expects an already-running local service. See
[Validation](docs/validation.md) for suite details.

## Documentation

- [API and runtime behavior](docs/api.md)
- [Operations](docs/operations.md)
- [Validation](docs/validation.md)
