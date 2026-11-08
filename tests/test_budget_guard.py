import pytest
from src.evals.budget_guard import BudgetGuard, BudgetExceeded


def test_fresh_guard_allows_check():
    guard = BudgetGuard(max_tokens=10_000, max_cost_usd=1.0)
    guard.check_budget()


def test_token_cap_raises_budget_exceeded():
    guard = BudgetGuard(max_tokens=100, max_cost_usd=10.0)
    guard.record_usage(input_tokens=60, output_tokens=60)
    with pytest.raises(BudgetExceeded):
        guard.check_budget()


def test_cost_cap_raises_budget_exceeded():
    guard = BudgetGuard(max_tokens=1_000_000, max_cost_usd=0.001)
    guard.record_usage(input_tokens=1000, output_tokens=1000, model="claude-opus-5-5")
    with pytest.raises(BudgetExceeded):
        guard.check_budget()


def test_record_usage_accumulates_tokens():
    guard = BudgetGuard(max_tokens=10_000, max_cost_usd=1.0)
    guard.record_usage(input_tokens=100, output_tokens=50)
    guard.record_usage(input_tokens=200, output_tokens=100)
    assert guard.total_tokens == 450


def test_reset_clears_all_counters():
    guard = BudgetGuard(max_tokens=10_000, max_cost_usd=1.0)
    guard.record_usage(input_tokens=500, output_tokens=500)
    guard.reset()
    assert guard.total_tokens == 0
    assert guard.total_cost_usd == 0.0


def test_remaining_tokens_decreases_after_usage():
    guard = BudgetGuard(max_tokens=1000, max_cost_usd=10.0)
    before = guard.remaining_tokens()
    guard.record_usage(input_tokens=100, output_tokens=100)
    assert guard.remaining_tokens() < before


def test_summary_returns_string_with_budgetguard():
    guard = BudgetGuard(max_tokens=10_000, max_cost_usd=1.0)
    s = guard.summary()
    assert isinstance(s, str)
    assert "BudgetGuard" in s


def test_usage_records_stored():
    guard = BudgetGuard(max_tokens=10_000, max_cost_usd=1.0)
    guard.record_usage(input_tokens=10, output_tokens=5, model="claude-haiku-5-5")
    records = guard.usage_records()
    assert len(records) == 1
    assert records[0].model == "claude-haiku-5-5"
