"""
Offline trajectory evaluation harness for SY1-Gov.

Runs scripted tool-call sequences through PolicyEngine and asserts
the expected policy outcome at each step. No API keys required.
"""

from dataclasses import dataclass
from typing import Any

from src.policy.engine import PermissionTier, PolicyEngine


@dataclass
class TrajectoryStep:
    tool_name: str
    inputs: dict[str, Any]
    expect_allowed: bool
    expect_tier: PermissionTier
    expect_requires_approval: bool = False
    kill_switch: bool = False


def run_trajectory(steps: list[TrajectoryStep]) -> list[dict[str, Any]]:
    """
    Execute a list of trajectory steps and return pass/fail results.
    Each step gets a fresh PolicyEngine unless kill_switch is shared.
    """
    results = []
    for i, step in enumerate(steps):
        engine = PolicyEngine(kill_switch_active=step.kill_switch)
        try:
            decision = engine.evaluate(step.tool_name, step.inputs)
            passed = (
                decision.allowed == step.expect_allowed
                and decision.tier == step.expect_tier
                and decision.requires_approval == step.expect_requires_approval
            )
            results.append(
                {
                    "step": i,
                    "tool_name": step.tool_name,
                    "passed": passed,
                    "allowed": decision.allowed,
                    "tier": decision.tier.value,
                    "requires_approval": decision.requires_approval,
                    "reason": decision.reason,
                    "error": None,
                }
            )
        except ValueError as exc:
            results.append(
                {
                    "step": i,
                    "tool_name": step.tool_name,
                    "passed": False,
                    "allowed": None,
                    "tier": None,
                    "requires_approval": None,
                    "reason": None,
                    "error": str(exc),
                }
            )
    return results


STANDARD_TRAJECTORY: list[TrajectoryStep] = [
    TrajectoryStep(
        tool_name="get_demand_forecast",
        inputs={"node_id": 3, "horizon_hours": 24},
        expect_allowed=True,
        expect_tier=PermissionTier.READ,
    ),
    TrajectoryStep(
        tool_name="get_anomaly_summary",
        inputs={"node_id": 3},
        expect_allowed=True,
        expect_tier=PermissionTier.READ,
    ),
    TrajectoryStep(
        tool_name="get_grid_topology",
        inputs={},
        expect_allowed=True,
        expect_tier=PermissionTier.READ,
    ),
    TrajectoryStep(
        tool_name="set_alert_threshold",
        inputs={"node_id": 5, "threshold_mw": 150},
        expect_allowed=True,
        expect_tier=PermissionTier.REVERSIBLE,
        expect_requires_approval=False,
    ),
    TrajectoryStep(
        tool_name="dispatch_maintenance_crew",
        inputs={"crew_id": "C-04", "location": "node_7"},
        expect_allowed=False,
        expect_tier=PermissionTier.IRREVERSIBLE,
        expect_requires_approval=True,
    ),
    TrajectoryStep(
        tool_name="isolate_grid_segment",
        inputs={"segment_id": "S-02"},
        expect_allowed=False,
        expect_tier=PermissionTier.IRREVERSIBLE,
        expect_requires_approval=True,
    ),
]

KILL_SWITCH_TRAJECTORY: list[TrajectoryStep] = [
    TrajectoryStep(
        tool_name="get_demand_forecast",
        inputs={},
        expect_allowed=False,
        expect_tier=PermissionTier.READ,
        kill_switch=True,
    ),
    TrajectoryStep(
        tool_name="set_alert_threshold",
        inputs={},
        expect_allowed=False,
        expect_tier=PermissionTier.READ,
        kill_switch=True,
    ),
]

if __name__ == "__main__":
    for name, traj in [
        ("standard", STANDARD_TRAJECTORY),
        ("kill_switch", KILL_SWITCH_TRAJECTORY),
    ]:
        results = run_trajectory(traj)
        passed = sum(1 for r in results if r["passed"])
        print(f"{name}: {passed}/{len(results)} steps passed")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status}] {r['tool_name']}")
