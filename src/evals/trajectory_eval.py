"""
Trajectory evaluator for SY1-Gov.

A trajectory is a sequence of tool calls made by the supervisor agent
during one operational scenario. This module:

1. Defines TrajectoryStep and ExpectedStep dataclasses
2. Runs a scenario (list of tool calls) through the PolicyEngine and
   ApprovalGateNode, recording the actual outcome at each step
3. Compares actual vs expected outcomes and returns a TrajectoryResult

This is a pure-Python offline evaluator - it does NOT call the LLM.
It tests that the governance layer enforces the right policy decisions
given a scripted sequence of tool calls.

Usage:
    from src.evals.trajectory_eval import TrajectoryEvaluator, TrajectoryScenario, ExpectedStep

    scenario = TrajectoryScenario(
        name="Routine demand check then emergency dispatch",
        steps=[
            ExpectedStep(
                tool_name="get_demand_forecast",
                inputs={"city": "Stockholm", "target": "heating_demand_mw"},
                expected_allowed=True,
                expected_tier="read",
                expected_requires_approval=False,
            ),
            ExpectedStep(
                tool_name="dispatch_maintenance_crew",
                inputs={
                    "node_id": 5,
                    "fault_type": "pipe_burst",
                    "priority": "high",
                    "justification": "GNN confidence 0.94. Pressure drop at node 5.",
                    "idempotency_key": "uuid-traj-001",
                },
                expected_allowed=False,
                expected_tier="irreversible",
                expected_requires_approval=True,
            ),
        ]
    )

    evaluator = TrajectoryEvaluator()
    result = evaluator.run(scenario)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


from src.policy.engine import PolicyEngine, PolicyDecision

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExpectedStep:
    """One expected step in a trajectory."""

    tool_name: str
    inputs: dict[str, Any]
    expected_allowed: bool
    expected_tier: str  # "read", "reversible", "irreversible"
    expected_requires_approval: bool
    description: str = ""  # Optional human label for the step


@dataclass
class ActualStep:
    """The actual outcome recorded when a step was evaluated."""

    tool_name: str
    inputs: dict[str, Any]
    actual_allowed: bool
    actual_tier: str
    actual_requires_approval: bool
    actual_reason: str
    error: Optional[str] = None  # Set if PolicyEngine raised


@dataclass
class StepResult:
    """Comparison of one expected vs actual step."""

    step_index: int
    expected: ExpectedStep
    actual: ActualStep
    passed: bool
    failures: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        label = self.expected.description or self.expected.tool_name
        if self.passed:
            return f"  [{status}] step {self.step_index}: {label}"
        return f"  [{status}] step {self.step_index}: {label} -- {'; '.join(self.failures)}"


@dataclass
class TrajectoryScenario:
    """A named sequence of expected tool call outcomes."""

    name: str
    steps: list[ExpectedStep]
    kill_switch_active: bool = False  # Activate kill switch for this scenario


@dataclass
class TrajectoryResult:
    """Full result of running one trajectory scenario."""

    scenario_name: str
    step_results: list[StepResult]
    passed: bool  # True if ALL steps passed

    def summary(self) -> str:
        lines = [
            f"Trajectory: {self.scenario_name}",
            f"Result: {'PASS' if self.passed else 'FAIL'} "
            f"({sum(1 for s in self.step_results if s.passed)}"
            f"/{len(self.step_results)} steps passed)",
        ]
        for sr in self.step_results:
            lines.append(sr.summary_line())
        return "\n".join(lines)

    def failed_steps(self) -> list[StepResult]:
        return [s for s in self.step_results if not s.passed]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class TrajectoryEvaluator:
    """
    Runs a TrajectoryScenario through the PolicyEngine and compares
    each step's actual outcome against the expected outcome.
    """

    def __init__(self, policy_engine: Optional[PolicyEngine] = None):
        self._policy = policy_engine

    def run(self, scenario: TrajectoryScenario) -> TrajectoryResult:
        """
        Run all steps in the scenario and return a TrajectoryResult.

        A fresh PolicyEngine is created per scenario so kill-switch state
        does not leak between scenarios.
        """
        engine = PolicyEngine(kill_switch_active=scenario.kill_switch_active)
        step_results: list[StepResult] = []

        for i, expected in enumerate(scenario.steps):
            actual = self._evaluate_step(engine, expected)
            result = self._compare(i, expected, actual)
            step_results.append(result)

        passed = all(r.passed for r in step_results)
        return TrajectoryResult(
            scenario_name=scenario.name,
            step_results=step_results,
            passed=passed,
        )

    def _evaluate_step(self, engine: PolicyEngine, step: ExpectedStep) -> ActualStep:
        try:
            decision: PolicyDecision = engine.evaluate(step.tool_name, step.inputs)
            return ActualStep(
                tool_name=step.tool_name,
                inputs=step.inputs,
                actual_allowed=decision.allowed,
                actual_tier=decision.tier.value,
                actual_requires_approval=decision.requires_approval,
                actual_reason=decision.reason,
            )
        except ValueError as e:
            # Unknown tool - fail-closed
            return ActualStep(
                tool_name=step.tool_name,
                inputs=step.inputs,
                actual_allowed=False,
                actual_tier="unknown",
                actual_requires_approval=False,
                actual_reason=str(e),
                error=str(e),
            )

    def _compare(
        self, index: int, expected: ExpectedStep, actual: ActualStep
    ) -> StepResult:
        failures: list[str] = []

        if actual.actual_allowed != expected.expected_allowed:
            failures.append(
                f"allowed: expected={expected.expected_allowed}, "
                f"actual={actual.actual_allowed}"
            )
        if actual.actual_tier != expected.expected_tier:
            failures.append(
                f"tier: expected={expected.expected_tier}, "
                f"actual={actual.actual_tier}"
            )
        if actual.actual_requires_approval != expected.expected_requires_approval:
            failures.append(
                f"requires_approval: expected={expected.expected_requires_approval}, "
                f"actual={actual.actual_requires_approval}"
            )

        return StepResult(
            step_index=index,
            expected=expected,
            actual=actual,
            passed=len(failures) == 0,
            failures=failures,
        )
