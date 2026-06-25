from __future__ import annotations

import argparse
import copy
import difflib
import json
import tempfile
from pathlib import Path
from typing import Any

from models import (
    RuntimeStateResetRequest,
    RuntimeStateResolveRequest,
    RuntimeStateUpdateRequest,
    RuntimeTurnCompleteRequest,
    RuntimeTurnStartRequest,
    RuntimeTurnUpdateRequest,
)
from pydantic import ValidationError
from services.runtime_state import (
    build_overlay,
    clear_states_for_tests,
    complete_turn,
    get_runtime_session,
    reset_state,
    resolve_state,
    start_turn,
    update_state,
    update_turn,
)

SCHEMA_VERSION = "runtime-replay-v1"
DEFAULT_CORPUS_PATH = Path(__file__).parents[1] / "replay" / "runtime" / "v1"
REQUIRED_CATEGORIES = {
    "empty_normal",
    "actionable_context",
    "reflective_context",
    "reset_behavior",
    "durable_turn_lifecycle",
    "rejected_input",
    "overlay_omission",
}
_ID_PREFIXES = ("rtstate_", "rtsession_", "rtturn_", "rtevent_", "rtoverlay_")
_TIMESTAMP_FIELDS = {
    "closed_at",
    "completed_at",
    "created_at",
    "last_activity_at",
    "started_at",
    "updated_at",
}


class ReplayMismatch(AssertionError):
    pass


class _Normalizer:
    def __init__(self) -> None:
        self._ids: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def normalize(self, value: Any, *, field: str | None = None) -> Any:
        if field in _TIMESTAMP_FIELDS and value is not None:
            return "<timestamp>"
        if isinstance(value, dict):
            return {key: self.normalize(item, field=key) for key, item in value.items()}
        if isinstance(value, list):
            return [self.normalize(item, field=field) for item in value]
        if isinstance(value, str):
            for prefix in _ID_PREFIXES:
                if value.startswith(prefix):
                    if value not in self._ids:
                        self._counters[prefix] = self._counters.get(prefix, 0) + 1
                        self._ids[value] = f"{prefix}{self._counters[prefix]}"
                    return self._ids[value]
        return value


def load_corpus(corpus_path: Path = DEFAULT_CORPUS_PATH) -> list[dict[str, Any]]:
    fixtures = []
    for path in sorted(corpus_path.glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        if fixture.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported runtime replay schema in {path}")
        fixture["_path"] = path
        fixtures.append(fixture)
    if not fixtures:
        raise ValueError(f"runtime replay corpus is empty: {corpus_path}")
    return fixtures


def _model_dict(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json")


def _execute_step(
    step: dict[str, Any],
    *,
    db_path: Path,
    context: dict[str, Any],
) -> Any:
    operation = step["operation"]
    payload = copy.deepcopy(step.get("request", {}))

    for key, context_key in step.get("from_context", {}).items():
        payload[key] = context[context_key]

    if operation == "resolve_state":
        body = RuntimeStateResolveRequest.model_validate(payload)
        result = resolve_state(
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            surface=body.surface,
        )
        return {"runtime_state": _model_dict(result)}

    if operation == "update_state":
        body = RuntimeStateUpdateRequest.model_validate(payload)
        result = update_state(
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            surface=body.surface,
            updates=body.updates,
        )
        return {"runtime_state": _model_dict(result)}

    if operation == "overlay":
        body = RuntimeStateResolveRequest.model_validate(payload)
        state = resolve_state(
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            surface=body.surface,
        )
        overlay, omission_reason = build_overlay(state)
        return {
            "runtime_state": _model_dict(state),
            "overlay": _model_dict(overlay) if overlay is not None else None,
            "omitted": overlay is None,
            "omission_reason": omission_reason,
        }

    if operation == "reset_state":
        body = RuntimeStateResetRequest.model_validate(payload)
        state = reset_state(
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            surface=body.surface,
        )
        return {"runtime_state": _model_dict(state), "reset": True}

    if operation == "start_turn":
        body = RuntimeTurnStartRequest.model_validate(payload)
        session, turn, event = start_turn(
            request_id=body.request_id,
            owner_id=body.owner_id,
            conversation_id=body.conversation_id,
            surface=body.surface,
            surface_session_id=body.surface_session_id,
            active_mode=body.active_mode,
            input_message_id=body.input_message_id,
            intent_class=body.intent_class,
            timing_policy=body.timing_policy,
            restraint_policy=body.restraint_policy,
            continuation_state=body.continuation_state,
        )
        context["runtime_session_id"] = session.runtime_session_id
        context["runtime_turn_id"] = turn.runtime_turn_id
        return {
            "runtime_session": _model_dict(session),
            "runtime_turn": _model_dict(turn),
            "event": _model_dict(event),
        }

    if operation == "update_turn":
        body = RuntimeTurnUpdateRequest.model_validate(payload)
        session, turn, event = update_turn(
            request_id=body.request_id,
            runtime_session_id=body.runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
            turn_status=body.turn_status,
            timing_policy=body.timing_policy,
            restraint_policy=body.restraint_policy,
            continuation_state=body.continuation_state,
        )
        return {
            "runtime_session": _model_dict(session),
            "runtime_turn": _model_dict(turn),
            "event": _model_dict(event),
        }

    if operation == "complete_turn":
        body = RuntimeTurnCompleteRequest.model_validate(payload)
        session, turn, event = complete_turn(
            request_id=body.request_id,
            runtime_session_id=body.runtime_session_id,
            runtime_turn_id=body.runtime_turn_id,
            turn_status=body.turn_status,
            continuation_state=body.continuation_state,
        )
        return {
            "runtime_session": _model_dict(session),
            "runtime_turn": _model_dict(turn),
            "event": _model_dict(event),
        }

    if operation == "reopen_repository":
        clear_states_for_tests(db_path=db_path)
        return {"reopened": True}

    if operation == "diagnostics":
        runtime_session_id = context["runtime_session_id"]
        return _model_dict(get_runtime_session(runtime_session_id))

    if operation == "rejected_update":
        before_body = RuntimeStateResolveRequest.model_validate(payload)
        before = resolve_state(
            owner_id=before_body.owner_id,
            conversation_id=before_body.conversation_id,
            surface=before_body.surface,
        )
        rejected_request = copy.deepcopy(step["rejected_request"])
        try:
            RuntimeStateUpdateRequest.model_validate(rejected_request)
        except ValidationError as exc:
            rejection = {
                "accepted": False,
                "error_type": "validation_error",
                "error_locations": [list(error["loc"]) for error in exc.errors()],
            }
        else:
            raise AssertionError("rejected_update fixture unexpectedly passed validation")
        after = resolve_state(
            owner_id=before_body.owner_id,
            conversation_id=before_body.conversation_id,
            surface=before_body.surface,
        )
        return {
            "rejection": rejection,
            "state_unchanged": _model_dict(before) == _model_dict(after),
            "runtime_state": _model_dict(after),
        }

    raise ValueError(f"unsupported replay operation: {operation}")


def run_scenario(fixture: dict[str, Any], db_path: Path) -> dict[str, Any]:
    if db_path.exists():
        raise ValueError(f"replay database must not already exist: {db_path}")
    clear_states_for_tests(db_path=db_path)
    normalizer = _Normalizer()
    context: dict[str, Any] = {}
    captures: dict[str, Any] = {}

    for step in fixture["steps"]:
        result = _execute_step(step, db_path=db_path, context=context)
        if capture := step.get("capture"):
            captures[capture] = result

    snapshot = {
        "schema_version": fixture["schema_version"],
        "scenario": fixture["scenario"],
        "category": fixture["category"],
        "request_ids": fixture["request_ids"],
        "captures": captures,
    }
    return normalizer.normalize(snapshot)


def structural_diff(expected: Any, actual: Any) -> str:
    expected_text = json.dumps(expected, indent=2, sort_keys=True).splitlines()
    actual_text = json.dumps(actual, indent=2, sort_keys=True).splitlines()
    return "\n".join(
        difflib.unified_diff(
            expected_text,
            actual_text,
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
    )


def compare_snapshot(expected: Any, actual: Any, scenario: str) -> None:
    if expected != actual:
        raise ReplayMismatch(f"{scenario} replay mismatch:\n{structural_diff(expected, actual)}")


def assert_snapshot_privacy_safe(snapshot: Any) -> None:
    serialized = json.dumps(snapshot, sort_keys=True).lower()
    forbidden = (
        "api_key",
        "authorization",
        "bearer ",
        "credential",
        "password",
        "private_key",
        "secret",
    )
    found = [marker for marker in forbidden if marker in serialized]
    if found:
        raise AssertionError(f"privacy-unsafe replay snapshot markers: {found}")

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"raw_payload", "raw_user_text", "prompt", "response_body"}:
                    raise AssertionError(f"privacy-unsafe replay snapshot field: {key}")
                _walk(item)
        elif isinstance(value, list):
            for item in value:
                _walk(item)
        elif isinstance(value, str) and len(value) > 320:
            raise AssertionError("replay snapshot contains an unrestricted string payload")

    _walk(snapshot)


def run_corpus(corpus_path: Path = DEFAULT_CORPUS_PATH) -> list[dict[str, Any]]:
    fixtures = load_corpus(corpus_path)
    categories = {fixture["category"] for fixture in fixtures}
    missing = REQUIRED_CATEGORIES - categories
    if missing:
        raise AssertionError(f"runtime replay corpus missing categories: {sorted(missing)}")

    results = []
    with tempfile.TemporaryDirectory(prefix="cognitive-runtime-replay-") as temp_dir:
        root = Path(temp_dir)
        for index, fixture in enumerate(fixtures):
            actual = run_scenario(fixture, root / f"scenario-{index}-first.sqlite3")
            repeated = run_scenario(fixture, root / f"scenario-{index}-second.sqlite3")
            assert_snapshot_privacy_safe(actual)
            compare_snapshot(actual, repeated, f"{fixture['scenario']} clean-database repeat")
            compare_snapshot(fixture["expected"], actual, fixture["scenario"])
            results.append(actual)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic Cognitive Runtime replay corpus"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--print-actual", action="store_true")
    args = parser.parse_args()

    if args.print_actual:
        with tempfile.TemporaryDirectory(prefix="cognitive-runtime-replay-") as temp_dir:
            for index, fixture in enumerate(load_corpus(args.corpus)):
                actual = run_scenario(fixture, Path(temp_dir) / f"scenario-{index}.sqlite3")
                print(json.dumps(actual, indent=2, sort_keys=True))
        return 0

    results = run_corpus(args.corpus)
    print(f"Runtime replay corpus passed: {len(results)} scenarios.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
