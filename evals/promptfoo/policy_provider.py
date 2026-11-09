"""
Promptfoo Python provider for SY1-Gov policy engine.

Promptfoo calls this module's `call_api` function for each test case.
The provider evaluates the tool call against the PolicyEngine and returns
a structured output dict that the YAML assertions can inspect.

This runs entirely offline - no LLM API key needed.

Promptfoo provider protocol:
    call_api(prompt: str, options: dict, context: dict) -> dict
        Returns: {"output": <any>, "error": <str|None>}
"""

from __future__ import annotations

import json
import sys
import os

# Allow running from the project root (where pyproject.toml lives)
# When called by promptfoo the cwd may vary, so we add the project root
# to sys.path using this file's location as an anchor.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.policy.engine import PolicyEngine


def call_api(prompt: str, options: dict, context: dict) -> dict:
    """
    Called by promptfoo for each test case.

    vars injected by promptfoo are in context["vars"].
    Expected keys:
        tool_name (str): The tool to evaluate
        inputs (dict): The tool inputs
        kill_switch (bool, optional): Activate kill switch before eval
        injected_instruction (str, optional): Logged but ignored by engine
    """
    try:
        vars_ = context.get("vars", {})
        tool_name = vars_.get("tool_name", "")
        inputs = vars_.get("inputs", {})
        kill_switch = vars_.get("kill_switch", False)

        engine = PolicyEngine(kill_switch_active=bool(kill_switch))

        try:
            decision = engine.evaluate(tool_name, inputs)
            output = decision.to_dict()
        except ValueError as e:
            # Unknown tool - this is fail-closed, not a provider error
            output = {
                "tool_name": tool_name,
                "allowed": False,
                "requires_approval": False,
                "tier": "unknown",
                "reason": str(e),
                "error": str(e),
            }

        return {"output": output}

    except Exception as e:
        return {
            "output": None,
            "error": f"Provider error: {type(e).__name__}: {e}",
        }
