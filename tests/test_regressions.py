"""Regression tests for bugs found when restoring the healthy files."""

import pytest

from src.policy.engine import PolicyEngine
from src.tools.executor import ToolExecutionError


def test_kill_switch_active_property_reflects_state():
    engine = PolicyEngine()
    assert engine.kill_switch_active is False
    engine.activate_kill_switch()
    assert engine.kill_switch_active is True
    engine.deactivate_kill_switch()
    assert engine.kill_switch_active is False


def test_kill_switch_active_set_at_construction():
    assert PolicyEngine(kill_switch_active=True).kill_switch_active is True


def test_tool_execution_error_is_raisable_and_carries_payload():
    with pytest.raises(ToolExecutionError) as exc_info:
        raise ToolExecutionError.build(
            tool_name="get_demand_forecast",
            error_code="FAILED_AFTER_RETRIES",
            message="upstream timeout",
            retryable=True,
            idempotency_key="abc",
        )
    err = exc_info.value.error
    assert err.tool_name == "get_demand_forecast"
    assert err.retryable is True
    assert err.idempotency_key == "abc"
    assert "upstream timeout" in str(exc_info.value)
