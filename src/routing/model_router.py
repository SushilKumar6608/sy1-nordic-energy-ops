"""
Rule-based model router for SY1-Gov.

Selects between claude-haiku-4-5 (fast, cheap) and claude-sonnet-4-5
(capable, slower) based on tool tier and input complexity.
No LLM calls - pure deterministic Python.

Routing rules:
  READ tools with simple inputs       -> Haiku
  READ tools with complex inputs      -> Sonnet
  REVERSIBLE tools                    -> Sonnet
  IRREVERSIBLE tools                  -> Sonnet (always, non-negotiable)
  Kill switch active                  -> Haiku (read-only audit only)
"""

from dataclasses import dataclass
from typing import Any

from src.policy.engine import PermissionTier

MODEL_HAIKU = "claude-haiku-4-5"
MODEL_SONNET = "claude-sonnet-4-5"

# Input complexity threshold: dicts with more keys than this are "complex"
COMPLEXITY_KEY_THRESHOLD = 3

# Input complexity threshold: string values longer than this are "complex"
COMPLEXITY_STR_THRESHOLD = 200


@dataclass
class RoutingDecision:
    model: str
    tier: PermissionTier
    reason: str
    input_complexity: str  # "simple" or "complex"


def _assess_complexity(inputs: dict[str, Any]) -> str:
    """Return 'simple' or 'complex' based on input size and depth."""
    if len(inputs) > COMPLEXITY_KEY_THRESHOLD:
        return "complex"
    for value in inputs.values():
        if isinstance(value, str) and len(value) > COMPLEXITY_STR_THRESHOLD:
            return "complex"
        if isinstance(value, (dict, list)) and len(value) > 2:
            return "complex"
    return "simple"


def route(
    tool_name: str,
    tier: PermissionTier,
    inputs: dict[str, Any],
    kill_switch_active: bool = False,
) -> RoutingDecision:
    """
    Return a RoutingDecision for the given tool call.

    Args:
        tool_name: The name of the tool being called.
        tier: The PermissionTier from PolicyEngine.
        inputs: The raw inputs dict for the tool call.
        kill_switch_active: If True, use Haiku - writes are blocked anyway.
    """
    complexity = _assess_complexity(inputs)

    if kill_switch_active:
        return RoutingDecision(
            model=MODEL_HAIKU,
            tier=tier,
            reason="Kill switch active - all writes blocked, Haiku sufficient for audit.",
            input_complexity=complexity,
        )

    if tier == PermissionTier.IRREVERSIBLE:
        return RoutingDecision(
            model=MODEL_SONNET,
            tier=tier,
            reason=(
                f"Irreversible tool '{tool_name}' always routes to Sonnet "
                "for maximum reasoning quality before human approval."
            ),
            input_complexity=complexity,
        )

    if tier == PermissionTier.REVERSIBLE:
        return RoutingDecision(
            model=MODEL_SONNET,
            tier=tier,
            reason=(
                f"Reversible write tool '{tool_name}' routes to Sonnet "
                "to minimise risk of incorrect write operations."
            ),
            input_complexity=complexity,
        )

    # READ tier - route by complexity
    if complexity == "complex":
        return RoutingDecision(
            model=MODEL_SONNET,
            tier=tier,
            reason=(
                f"Read tool '{tool_name}' with complex inputs routes to Sonnet "
                "for higher accuracy."
            ),
            input_complexity=complexity,
        )

    return RoutingDecision(
        model=MODEL_HAIKU,
        tier=tier,
        reason=(
            f"Read tool '{tool_name}' with simple inputs routes to Haiku "
            "for speed and cost efficiency."
        ),
        input_complexity=complexity,
    )
