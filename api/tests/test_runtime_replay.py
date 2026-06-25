from __future__ import annotations

import copy
from pathlib import Path

import pytest
from services.runtime_replay import (
    DEFAULT_CORPUS_PATH,
    REQUIRED_CATEGORIES,
    ReplayMismatch,
    assert_snapshot_privacy_safe,
    compare_snapshot,
    load_corpus,
    run_corpus,
    run_scenario,
)


def test_complete_persisted_runtime_replay_corpus_passes():
    results = run_corpus()

    assert len(results) == 7


def test_corpus_is_equivalent_across_two_clean_database_runs(tmp_path: Path):
    for index, fixture in enumerate(load_corpus()):
        first = run_scenario(fixture, tmp_path / "first" / f"{index}.sqlite3")
        second = run_scenario(fixture, tmp_path / "second" / f"{index}.sqlite3")

        assert first == second


def test_modified_expected_value_produces_readable_structural_diff(tmp_path: Path):
    fixture = load_corpus()[0]
    actual = run_scenario(fixture, tmp_path / "runtime.sqlite3")
    modified = copy.deepcopy(actual)
    modified["captures"]["overlay"]["omission_reason"] = "changed-reason"

    with pytest.raises(ReplayMismatch) as exc_info:
        compare_snapshot(modified, actual, fixture["scenario"])

    message = str(exc_info.value)
    assert "--- expected" in message
    assert "+++ actual" in message
    assert "changed-reason" in message
    assert "empty_runtime_state" in message


def test_required_runtime_replay_categories_are_present():
    categories = {fixture["category"] for fixture in load_corpus()}

    assert categories == REQUIRED_CATEGORIES


def test_expected_runtime_replay_snapshots_are_privacy_safe():
    for fixture in load_corpus():
        assert_snapshot_privacy_safe(fixture["expected"])


def test_rejected_input_does_not_mutate_valid_runtime_state(tmp_path: Path):
    fixture = next(fixture for fixture in load_corpus() if fixture["category"] == "rejected_input")
    snapshot = run_scenario(fixture, tmp_path / "runtime.sqlite3")

    assert snapshot["captures"]["rejected"]["rejection"]["accepted"] is False
    assert snapshot["captures"]["rejected"]["state_unchanged"] is True
    assert (
        snapshot["captures"]["valid_state"]["runtime_state"]
        == snapshot["captures"]["rejected"]["runtime_state"]
    )


def test_durable_turn_lifecycle_survives_repository_reinitialization(tmp_path: Path):
    fixture = next(
        fixture
        for fixture in load_corpus(DEFAULT_CORPUS_PATH)
        if fixture["category"] == "durable_turn_lifecycle"
    )
    snapshot = run_scenario(fixture, tmp_path / "runtime.sqlite3")
    diagnostics = snapshot["captures"]["diagnostics_after_reopen"]

    assert diagnostics["active_turn"] is None
    assert diagnostics["latest_turn"]["turn_status"] == "completed"
    assert [event["event_type"] for event in diagnostics["events"]] == [
        "session_resolved",
        "turn_started",
        "turn_updated",
        "turn_completed",
    ]
    assert all(
        event["runtime_session_id"] == diagnostics["runtime_session"]["runtime_session_id"]
        for event in diagnostics["events"]
    )
    assert all(
        event["runtime_turn_id"] in {None, diagnostics["latest_turn"]["runtime_turn_id"]}
        for event in diagnostics["events"]
    )
