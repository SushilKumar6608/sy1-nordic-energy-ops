"""Pytest wrapper for trajectory evaluation - runs offline, no API key needed."""

import pytest

from src.policy.engine import PermissionTier
from src.evals.trajectory_eval import (
    KILL_SWITCH_TRAJECTORY,
    STANDARD_TRAJECTORY,
    TrajectoryStep,
    run_trajectory,
)


def test_standard_trajectory_all_pass():
    results = run_trajectory(STANDARD_TRAJECTORY)
    failures = [r for r in results if not r["passed"]]
    assert failures == [], f"Trajectory failures: {failures}"


def test_kill_switch_trajectory_all_pass():
    results = run_trajectory(KILL_SWITCH_TRAJECTORY)
    failures = [r for r in results if not r["passed"]]
    assert failures == [], f"Kill switch trajectory failures: {failures}"


def test_read_step_allowed():
    steps = [
        TrajectoryStep(
            tool_name="get_demand_forecast",
            inputs={},
            expect_allowed=True,
            expect_tier=PermissionTier.READ,
        )
    ]
    results = run_trajectory(steps)
    assert results[0]["passed"] is True


def test_irreversible_step_blocked():
    steps = [
        TrajectoryStep(
            tool_name="dispatch_maintenance_crew",
            inputs={},
            expect_allowed=False,
            expect_tier=PermissionTier.IRREVERSIBLE,
            expect_requires_approval=True,
        )
    ]
    results = run_trajectory(steps)
    assert results[0]["passed"] is True


def test_unknown_tool_step_fails():
    steps = [
        TrajectoryStep(
            tool_name="nonexistent_tool",
            inputs={},
            expect_allowed=False,
            expect_tier=PermissionTier.READ,
        )
    ]
    results = run_trajectory(steps)
    assert results[0]["error"] is not None


def test_trajectory_result_has_required_keys():
    results = run_trajectory(STANDARD_TRAJECTORY[:1])
    r = results[0]
    for key in ["step", "tool_name", "passed", "allowed", "tier", "requires_approval"]:
        assert key in r


@pytest.mark.live_api
def test_judge_requires_api_key_when_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from src.evals.judge import is_live_api_available

    assert is_live_api_available() is False


def test_all_registered_read_tools_pass_trajectory():
    tools = [
        "get_demand_forecast",
        "get_anomaly_summary",
        "get_grid_topology",
        "get_node_status",
        "query_fault_ontology",
        "retrieve_rag_context",
    ]
    steps = [
        TrajectoryStep(
            tool_name=t,
            inputs={},
            expect_allowed=True,
            expect_tier=PermissionTier.READ,
        )
        for t in tools
    ]
    results = run_trajectory(steps)
    failures = [r for r in results if not r["passed"]]
    assert failures == []
