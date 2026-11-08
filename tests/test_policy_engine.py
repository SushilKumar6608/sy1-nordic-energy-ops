import pytest
from src.policy.engine import PolicyEngine, PermissionTier


def test_read_only_tier_exists():
    assert PermissionTier.READ_ONLY


def test_advisory_tier_exists():
    assert PermissionTier.ADVISORY


def test_irreversible_tier_exists():
    assert PermissionTier.IRREVERSIBLE


def test_policy_engine_instantiates():
    engine = PolicyEngine()
    assert engine is not None


def test_known_read_tool_evaluates():
    engine = PolicyEngine()
    decision = engine.evaluate("get_sensor_reading", {})
    assert decision is not None
    assert decision.tier in (PermissionTier.READ_ONLY, PermissionTier.ADVISORY, PermissionTier.IRREVERSIBLE)


def test_decision_has_tier_and_reason():
    engine = PolicyEngine()
    decision = engine.evaluate("get_sensor_reading", {})
    assert hasattr(decision, "tier")
    assert hasattr(decision, "reason")
    assert isinstance(decision.reason, str)


def test_decision_to_dict():
    engine = PolicyEngine()
    decision = engine.evaluate("get_sensor_reading", {})
    d = decision.to_dict()
    assert isinstance(d, dict)
    assert "tier" in d
