import pytest
from src.evals.budget_guard import BudgetGuard, BudgetExceeded


def test_budget_guard_default_instantiation():
    guard = BudgetGuard()
    assert guard.max_tokens > 0
    assert guard.max_cost_usd > 0


def test_budget_guard_respects_token_limit():
    guard = BudgetGuard(max_tokens=50, max_cost_usd=99.0)
    guard.record_usage(input_tokens=30, output_tokens=30)
    with pytest.raises(BudgetExceeded) as exc_info:
        guard.check_budget()
    assert exc_info.value.current_tokens == 60


def test_budget_exceeded_has_attributes():
    guard = BudgetGuard(max_tokens=10, max_cost_usd=99.0)
    guard.record_usage(input_tokens=10, output_tokens=5)
    try:
        guard.check_budget()
        pytest.fail("Expected BudgetExceeded")
    except BudgetExceeded as e:
        assert hasattr(e, "reason")
        assert hasattr(e, "current_tokens")
        assert hasattr(e, "current_cost")
