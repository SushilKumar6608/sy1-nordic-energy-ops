"""
Offline promptfoo provider for SY1-Gov injection suite.

Runs each test case through PolicyEngine and returns a structured
result that the YAML assertions can inspect.
"""

import json
import sys
from typing import Any


def call_api(tool_name: str, options: dict[str, Any], context: dict[str, Any]) -> dict:
    """
    Promptfoo provider entry point.
    Returns a dict with keys: output, error.
    output is a JSON-serialisable dict the YAML assertions inspect.
    """
    sys.path.insert(0, ".")
    from src.policy.engine import PolicyEngine

    inputs = options.get("vars", {}).get("inputs", {})
    kill_switch = bool(options.get("vars", {}).get("kill_switch", False))

    engine = PolicyEngine(kill_switch_active=kill_switch)
    try:
        decision = engine.evaluate(tool_name, inputs)
        result = {
            "blocked": not decision.allowed,
            "tier": decision.tier.value,
            "requires_approval": decision.requires_approval,
            "reason": decision.reason,
            "error_type": None,
        }
    except ValueError:
        result = {
            "blocked": True,
            "tier": None,
            "requires_approval": None,
            "reason": None,
            "error_type": "unknown_tool",
        }

    return {"output": json.dumps(result), "error": None}
