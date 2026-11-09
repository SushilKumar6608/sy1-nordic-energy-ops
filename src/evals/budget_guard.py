"""
Budget guard for SY1-Gov.

Enforces per-session token and cost caps before each LLM call. The
supervisor agent calls check_budget() before invoking the LLM. If either
cap is exceeded, BudgetExceeded is raised and the call must not proceed.

Design goals:
- Pure Python, no external dependencies.
- Thread-safe token accounting.
- Configurable via constructor or environment variables.
- Budgets reset per-session (each BudgetGuard instance is one session).

Environment variables (all optional, used as defaults):
  SY1_MAX_TOKENS     - Maximum input + output tokens per session (int)
  SY1_MAX_COST_USD   - Maximum spend per session in USD (float)

Token-to-cost mapping:
  Uses conservative per-model rates. If the model is not recognised,
  falls back to a safe default of $0.015 per 1k tokens (input) and
  $0.075 per 1k tokens (output).

Usage:
    from src.evals.budget_guard import BudgetGuard, BudgetExceeded

    guard = BudgetGuard(max_tokens=50_000, max_cost_usd=0.50)

    # Before each LLM call:
    guard.check_budget()   # raises BudgetExceeded if over limit

    # After the call returns:
    guard.record_usage(input_tokens=512, output_tokens=128, model="claude-haiku-5-5")

    print(guard.summary())
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BudgetExceeded(Exception):
    """Raised when a session has exceeded its token or cost budget."""

    def __init__(self, reason: str, current_tokens: int, current_cost: float):
        self.reason = reason
        self.current_tokens = current_tokens
        self.current_cost = current_cost
        super().__init__(
            f"Budget exceeded: {reason} "
            f"(tokens used: {current_tokens}, cost used: ${current_cost:.4f})"
        )


# ---------------------------------------------------------------------------
# Per-model cost rates (USD per 1k tokens)
# Source: Anthropic public pricing (approximate)
# ---------------------------------------------------------------------------

# Format: model_prefix -> (input_per_1k, output_per_1k)
_MODEL_RATES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (0.015, 0.075),
    "claude-sonnet-5": (0.003, 0.015),
    "claude-haiku-5": (0.00025, 0.00125),
    "claude-sonnet-4": (0.003, 0.015),
    "claude-haiku-4": (0.00025, 0.00125),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-5-haiku": (0.0008, 0.004),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-haiku": (0.00025, 0.00125),
}

_DEFAULT_RATE = (0.015, 0.075)  # Conservative fallback


def _lookup_rate(model: str) -> tuple[float, float]:
    """Return (input_per_1k, output_per_1k) for the given model string."""
    model_lower = model.lower()
    for prefix, rate in _MODEL_RATES.items():
        if model_lower.startswith(prefix):
            return rate
    return _DEFAULT_RATE


# ---------------------------------------------------------------------------
# Usage record
# ---------------------------------------------------------------------------


@dataclass
class UsageRecord:
    """One LLM call's token usage."""

    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------


class BudgetGuard:
    """
    Per-session token and cost budget enforcer.

    Thread-safe: multiple threads can record usage concurrently.
    """

    _DEFAULT_MAX_TOKENS = 100_000
    _DEFAULT_MAX_COST = 1.00  # USD

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
    ):
        """
        Args:
            max_tokens: Maximum total tokens (input + output) for this session.
                        Defaults to SY1_MAX_TOKENS env var, then 100 000.
            max_cost_usd: Maximum spend in USD for this session.
                          Defaults to SY1_MAX_COST_USD env var, then $1.00.
        """
        if max_tokens is None:
            env_val = os.environ.get("SY1_MAX_TOKENS")
            max_tokens = int(env_val) if env_val else self._DEFAULT_MAX_TOKENS

        if max_cost_usd is None:
            env_val = os.environ.get("SY1_MAX_COST_USD")
            max_cost_usd = float(env_val) if env_val else self._DEFAULT_MAX_COST

        self.max_tokens: int = max_tokens
        self.max_cost_usd: float = max_cost_usd

        self._lock = threading.Lock()
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._records: list[UsageRecord] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    def check_budget(self) -> None:
        """
        Raise BudgetExceeded if either cap has been reached.

        Call this BEFORE making each LLM call.
        """
        with self._lock:
            tokens = self._total_input_tokens + self._total_output_tokens
            cost = self._total_cost_usd

        if tokens >= self.max_tokens:
            raise BudgetExceeded(
                reason=f"token cap reached ({tokens} >= {self.max_tokens})",
                current_tokens=tokens,
                current_cost=cost,
            )

        if cost >= self.max_cost_usd:
            raise BudgetExceeded(
                reason=f"cost cap reached (${cost:.4f} >= ${self.max_cost_usd:.4f})",
                current_tokens=tokens,
                current_cost=cost,
            )

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "claude-sonnet-4-6",
    ) -> UsageRecord:
        """
        Record token usage from one LLM call.

        Call this AFTER the call returns.

        Returns the UsageRecord for this call.
        """
        input_rate, output_rate = _lookup_rate(model)
        call_cost = (input_tokens / 1000.0) * input_rate + (
            output_tokens / 1000.0
        ) * output_rate

        record = UsageRecord(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            cost_usd=call_cost,
        )

        with self._lock:
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cost_usd += call_cost
            self._records.append(record)

        return record

    def remaining_tokens(self) -> int:
        """Tokens remaining before the cap is hit."""
        return max(0, self.max_tokens - self.total_tokens)

    def remaining_cost_usd(self) -> float:
        """USD remaining before the cost cap is hit."""
        return max(0.0, self.max_cost_usd - self._total_cost_usd)

    def usage_records(self) -> list[UsageRecord]:
        """Return a copy of all usage records (oldest first)."""
        with self._lock:
            return list(self._records)

    def reset(self) -> None:
        """Reset all counters. Use between test cases or new sessions."""
        with self._lock:
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_cost_usd = 0.0
            self._records = []

    def summary(self) -> str:
        """Return a human-readable budget status string."""
        tokens = self.total_tokens
        cost = self._total_cost_usd
        pct_tokens = 100.0 * tokens / self.max_tokens if self.max_tokens else 0
        pct_cost = 100.0 * cost / self.max_cost_usd if self.max_cost_usd else 0
        return (
            f"BudgetGuard: "
            f"{tokens}/{self.max_tokens} tokens ({pct_tokens:.1f}%), "
            f"${cost:.4f}/${self.max_cost_usd:.4f} ({pct_cost:.1f}%)"
        )
