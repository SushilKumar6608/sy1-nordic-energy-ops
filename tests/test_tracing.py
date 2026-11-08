from src.policy.engine import PolicyEngine, PermissionTier


def test_read_only_tier_value():
    assert PermissionTier.READ_ONLY.value is not None


def test_advisory_tier_value():
    assert PermissionTier.ADVISORY.value is not None


def test_irreversible_tier_value():
    assert PermissionTier.IRREVERSIBLE.value is not None


def test_engine_evaluate_with_inputs():
    engine = PolicyEngine()
    decision = engine.evaluate("get_sensor_reading", {"sensor_id": "S01"})
    assert decision.tier == PermissionTier.READ_ONLY
