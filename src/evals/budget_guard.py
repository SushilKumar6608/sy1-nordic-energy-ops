"""
Budget guard for eval runs.
Tracks token usage and enforces per-run cost ceilings.
"""

from dataclasses import dataclass, field


@dataclass
class BudgetGuard:
    max_tokens: int = 100_000
    _used: int = field(default=0, init=False, repr=False)

    def record_usage(self, tokens: int) -> None:
        """Add tokens to the running total."""
        self._used += tokens

    def is_over_budget(self) -> bool:
        """Return True if cumulative usage exceeds the ceiling."""
        return self._used > self.max_tokens

    def remaining(self) -> int:
        """Tokens remaining before the ceiling is hit."""
        return max(0, self.max_tokens - self._used)

    def reset(self) -> None:
        """Reset usage counter (test helper)."""
        self._used = 0
