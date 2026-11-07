"""
Tests for src/evals/budget_guard.py

All tests are fully offline. No API keys needed.
Covers:
- BudgetGuard initialisation from constructor args and env vars
- check_budget does not raise when under limits
- check_budget raises BudgetExceeded when token cap is reached
- check_budget raises BudgetExceeded when cost cap is reached
- record_usage accumulates tokens and cost correctly
- Thread safety: concurrent record_usage calls
- remaining_tokens and remaining_cost_usd
- reset() zeroes all counters
- summary() returns a non-empty string
- _lookup_rate returns correct rates for known models
- _lookup_rate falls back for unknown models
- BudgetExceeded carries correct attributes
"""

import os
import threading
import pytest

from src.evals.budget_guard import (
    BudgetGuard,
    BudgetExceeded,
    UsageRecord,
    _lookup_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guard(**kwargs) -> BudgetGuard:
    """Create a BudgetGuard with small defaults for easy testing."""
    return BudgetGuard(
        max_tokens=kwargs.pop("max_tokens", 1_000),
        max_cost_usd=kwargs.pop("max_cost_usd", 0.10),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestBudgetGuardInit:
    def test_defaults_are_set(self):
        guard = BudgetGuard()
        assert guard.max_tokens > 0
        assert guard.max_cost_usd > 0

    def test_explicit_args_override_defaults(self):
        guard = BudgetGuard(max_tokens=5_000, max_cost_usd=0.25)
        assert guard.max_tokens == 5_000
        assert guard.max_cost_usd == 0.25

    def test_env_var_max_tokens(self, monkeypatch):
        monkeypatch.setenv("SY1_MAX_TOKENS", "7777")
        monkeypatch.delenv("SY1_MAX_COST_USD", raising=False)
        guard = BudgetGuard()
        assert guard.max_tokens == 7777

    def test_env_var_max_cost(self, monkeypatch):
        monkeypatch.delenv("SY1_MAX_TOKENS", raising=False)
        monkeypatch.setenv("SY1_MAX_COST_USD", "2.50")
        guard = BudgetGuard()
        assert guard.max_cost_usd == 2.50

    def test_constructor_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("SY1_MAX_TOKENS", "9999")
        guard = BudgetGuard(max_tokens=100)
        assert guard.max_tokens == 100


# ---------------------------------------------------------------------------
# check_budget - under limits
# ---------------------------------------------------------------------------

class TestCheckBudgetUnderLimit:
    def test_no_raise_when_zero_usage(self):
        guard = _guard()
        guard.check_budget()   # Must not raise

    def test_no_raise_just_below_token_cap(self):
        guard = _guard(max_tokens=1_000)
        guard.record_usage(input_tokens=400, output_tokens=599, model="claude-haiku-5-5")
        guard.check_budget()   # 999 < 1000 - must not raise

    def test_no_raise_just_below_cost_cap(self):
        guard = _guard(max_tokens=100_000, max_cost_usd=1.00)
        # Record a tiny cost - well below $1.00
        guard.record_usage(input_tokens=1, output_tokens=1, model="claude-haiku-5-5")
        guard.check_budget()


# ---------------------------------------------------------------------------
# check_budget - over limits
# ---------------------------------------------------------------------------

class TestCheckBudgetOverLimit:
    def test_raises_when_token_cap_reached(self):
        guard = _guard(max_tokens=100)
        guard.record_usage(input_tokens=60, output_tokens=40, model="claude-haiku-5-5")
        with pytest.raises(BudgetExceeded) as exc_info:
            guard.check_budget()
        assert exc_info.value.current_tokens == 100

    def test_raises_when_token_cap_exceeded(self):
        guard = _guard(max_tokens=50)
        guard.record_usage(input_tokens=30, output_tokens=30, model="claude-haiku-5-5")
        with pytest.raises(BudgetExceeded):
            guard.check_budget()

    def test_raises_when_cost_cap_reached(self):
        guard = _guard(max_tokens=1_000_000, max_cost_usd=0.001)
        # claude-opus rate is $0.015/1k input = $0.015 for 1000 input tokens
        guard.record_usage(input_tokens=1_000, output_tokens=0, model="claude-opus-5-5")
        with pytest.raises(BudgetExceeded):
            guard.check_budget()

    def test_budget_exceeded_has_tokens(self):
        guard = _guard(max_tokens=10)
        guard.record_usage(input_tokens=5, output_tokens=5, model="claude-haiku-5-5")
        with pytest.raises(BudgetExceeded) as exc_info:
            guard.check_budget()
        exc = exc_info.value
        assert exc.current_tokens == 10

    def test_budget_exceeded_has_cost(self):
        guard = _guard(max_tokens=10)
        guard.record_usage(input_tokens=5, output_tokens=5, model="claude-haiku-5-5")
        with pytest.raises(BudgetExceeded) as exc_info:
            guard.check_budget()
        exc = exc_info.value
        assert exc.current_cost >= 0.0

    def test_reason_mentions_token_cap(self):
        guard = _guard(max_tokens=5)
        guard.record_usage(input_tokens=3, output_tokens=2, model="claude-haiku-5-5")
        with pytest.raises(BudgetExceeded) as exc_info:
            guard.check_budget()
        assert "token" in exc_info.value.reason.lower()


# ---------------------------------------------------------------------------
# record_usage accumulation
# ---------------------------------------------------------------------------

class TestRecordUsage:
    def test_tokens_accumulate(self):
        guard = _guard()
        guard.record_usage(input_tokens=100, output_tokens=50, model="claude-haiku-5-5")
        guard.record_usage(input_tokens=200, output_tokens=75, model="claude-haiku-5-5")
        assert guard.total_input_tokens == 300
        assert guard.total_output_tokens == 125
        assert guard.total_tokens == 425

    def test_cost_accumulates(self):
        guard = _guard()
        guard.record_usage(input_tokens=1_000, output_tokens=0, model="claude-haiku-5-5")
        guard.record_usage(input_tokens=1_000, output_tokens=0, model="claude-haiku-5-5")
        # haiku input rate is $0.00025/1k so 2x1000 = $0.00050
        assert guard.total_cost_usd == pytest.approx(0.00050, rel=1e-3)

    def test_returns_usage_record(self):
        guard = _guard()
        rec = guard.record_usage(input_tokens=10, output_tokens=5, model="claude-haiku-5-5")
        assert isinstance(rec, UsageRecord)
        assert rec.input_tokens == 10
        assert rec.output_tokens == 5
        assert rec.model == "claude-haiku-5-5"
        assert rec.cost_usd >= 0

    def test_usage_records_list(self):
        guard = _guard()
        guard.record_usage(input_tokens=10, output_tokens=5, model="claude-haiku-5-5")
        guard.record_usage(input_tokens=20, output_tokens=10, model="claude-sonnet-4-6")
        records = guard.usage_records()
        assert len(records) == 2
        assert records[0].model == "claude-haiku-5-5"
        assert records[1].model == "claude-sonnet-4-6"

    def test_usage_records_is_copy(self):
        guard = _guard()
        guard.record_usage(input_tokens=10, output_tokens=5, model="claude-haiku-5-5")
        r1 = guard.usage_records()
        r1.append("extra")
        r2 = guard.usage_records()
        assert len(r2) == 1   # Original list unmodified


# ---------------------------------------------------------------------------
# remaining_tokens and remaining_cost_usd
# ---------------------------------------------------------------------------

class TestRemainingBudget:
    def test_remaining_tokens_full_at_start(self):
        guard = _guard(max_tokens=1_000)
        assert guard.remaining_tokens() == 1_000

    def test_remaining_tokens_decreases(self):
        guard = _guard(max_tokens=1_000)
        guard.record_usage(input_tokens=300, output_tokens=100, model="claude-haiku-5-5")
        assert guard.remaining_tokens() == 600

    def test_remaining_tokens_clamps_at_zero(self):
        guard = _guard(max_tokens=5)
        guard.record_usage(input_tokens=10, output_tokens=0, model="claude-haiku-5-5")
        assert guard.remaining_tokens() == 0

    def test_remaining_cost_full_at_start(self):
        guard = _guard(max_cost_usd=1.00)
        assert guard.remaining_cost_usd() == pytest.approx(1.00)

    def test_remaining_cost_clamps_at_zero(self):
        guard = _guard(max_cost_usd=0.001)
        guard.record_usage(input_tokens=1_000, output_tokens=0, model="claude-opus-5-5")
        assert guard.remaining_cost_usd() == 0.0


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_zeroes_tokens(self):
        guard = _guard()
        guard.record_usage(input_tokens=500, output_tokens=200, model="claude-haiku-5-5")
        guard.reset()
        assert guard.total_tokens == 0

    def test_reset_zeroes_cost(self):
        guard = _guard()
        guard.record_usage(input_tokens=500, output_tokens=200, model="claude-haiku-5-5")
        guard.reset()
        assert guard.total_cost_usd == 0.0

    def test_reset_clears_records(self):
        guard = _guard()
        guard.record_usage(input_tokens=10, output_tokens=5, model="claude-haiku-5-5")
        guard.reset()
        assert guard.usage_records() == []

    def test_check_budget_passes_after_reset(self):
        guard = _guard(max_tokens=10)
        guard.record_usage(input_tokens=5, output_tokens=5, model="claude-haiku-5-5")
        guard.reset()
        guard.check_budget()   # Must not raise


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_is_string(self):
        guard = _guard()
        assert isinstance(guard.summary(), str)

    def test_summary_contains_tokens(self):
        guard = _guard()
        assert "token" in guard.summary().lower()

    def test_summary_contains_percentages(self):
        guard = _guard(max_tokens=1_000)
        guard.record_usage(input_tokens=500, output_tokens=0, model="claude-haiku-5-5")
        summary = guard.summary()
        assert "%" in summary


# ---------------------------------------------------------------------------
# _lookup_rate
# ---------------------------------------------------------------------------

class TestLookupRate:
    def test_haiku_rate(self):
        rate = _lookup_rate("claude-haiku-5-5")
        input_per_1k, output_per_1k = rate
        assert input_per_1k < output_per_1k  # Output always more expensive
        assert input_per_1k > 0

    def test_opus_rate_higher_than_haiku(self):
        haiku_in, _ = _lookup_rate("claude-haiku-5-5")
        opus_in, _ = _lookup_rate("claude-opus-5-5")
        assert opus_in > haiku_in

    def test_unknown_model_falls_back(self):
        rate = _lookup_rate("some-unknown-model-xyz")
        assert rate is not None
        assert len(rate) == 2
        assert rate[0] > 0

    def test_case_insensitive(self):
        lower = _lookup_rate("claude-haiku-5-5")
        upper = _lookup_rate("CLAUDE-HAIKU-5-5")
        assert lower == upper


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_record_usage(self):
        guard = _guard(max_tokens=1_000_000, max_cost_usd=100.0)
        errors = []

        def worker():
            try:
                for _ in range(100):
                    guard.record_usage(input_tokens=1, output_tokens=1, model="claude-haiku-5-5")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        # 10 threads x 100 calls x (1 input + 1 output) = 2000 tokens
        assert guard.total_tokens == 2_000
