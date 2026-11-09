"""
Tests for src/tracing/langfuse_tracer.py

All tests are fully offline. No Langfuse credentials needed.
The tests verify:
- Tracer is a no-op when LANGFUSE_SECRET_KEY is absent
- Tracer singleton is returned consistently
- policy_span context manager does not raise
- set_outcome and set_error are safe to call on no-op span
- trace context manager yields None when disabled
- flush is a safe no-op when disabled
- reset_tracer allows re-initialisation in tests
- enabled flag reflects credential presence
"""

import pytest

from src.tracing.langfuse_tracer import (
    SY1Tracer,
    PolicySpanContext,
    get_tracer,
    reset_tracer,
    _tracing_enabled,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure each test starts with a fresh tracer singleton."""
    reset_tracer()
    yield
    reset_tracer()


@pytest.fixture()
def no_langfuse_env(monkeypatch):
    """Ensure Langfuse env vars are absent."""
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)


# ---------------------------------------------------------------------------
# _tracing_enabled helper
# ---------------------------------------------------------------------------


class TestTracingEnabled:
    def test_disabled_when_no_keys(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        assert _tracing_enabled() is False

    def test_disabled_when_only_secret(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        assert _tracing_enabled() is False

    def test_disabled_when_only_public(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        assert _tracing_enabled() is False

    def test_enabled_when_both_keys(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        assert _tracing_enabled() is True


# ---------------------------------------------------------------------------
# get_tracer singleton
# ---------------------------------------------------------------------------


class TestGetTracerSingleton:
    def test_returns_sy1tracer_instance(self, no_langfuse_env):
        tracer = get_tracer()
        assert isinstance(tracer, SY1Tracer)

    def test_same_instance_on_repeated_calls(self, no_langfuse_env):
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_disabled_when_no_credentials(self, no_langfuse_env):
        tracer = get_tracer()
        assert tracer.enabled is False

    def test_reset_tracer_clears_singleton(self, no_langfuse_env):
        t1 = get_tracer()
        reset_tracer()
        t2 = get_tracer()
        assert t1 is not t2


# ---------------------------------------------------------------------------
# No-op tracer behaviour (no credentials)
# ---------------------------------------------------------------------------


class TestNoOpTracer:
    def test_policy_span_does_not_raise(self, no_langfuse_env):
        tracer = get_tracer()
        # Should complete without any exception
        with tracer.policy_span(
            "get_demand_forecast", {"city": "Stockholm", "target": "heating_demand_mw"}
        ):
            pass

    def test_policy_span_yields_context(self, no_langfuse_env):
        tracer = get_tracer()
        with tracer.policy_span(
            "get_demand_forecast", {"city": "Oslo", "target": "heating_demand_mw"}
        ) as ctx:
            assert isinstance(ctx, PolicySpanContext)
            assert ctx.tool_name == "get_demand_forecast"

    def test_set_outcome_is_safe_noop(self, no_langfuse_env):
        tracer = get_tracer()

        class FakeDecision:
            allowed = True

            class tier:
                value = "read"

            requires_approval = False
            reason = "ok"

        with tracer.policy_span("get_node_status", {"node_id": 1}) as ctx:
            ctx.set_outcome(FakeDecision())  # Must not raise

    def test_set_error_is_safe_noop(self, no_langfuse_env):
        tracer = get_tracer()
        with tracer.policy_span("get_demand_forecast", {}) as ctx:
            ctx.set_error(ValueError("test error"))  # Must not raise

    def test_policy_span_reraises_exceptions(self, no_langfuse_env):
        tracer = get_tracer()
        with pytest.raises(RuntimeError, match="test failure"):
            with tracer.policy_span("get_demand_forecast", {}):
                raise RuntimeError("test failure")

    def test_trace_context_yields_none(self, no_langfuse_env):
        tracer = get_tracer()
        with tracer.trace("agent_turn", metadata={"session": "s1"}) as t:
            assert t is None

    def test_flush_is_safe_noop(self, no_langfuse_env):
        tracer = get_tracer()
        tracer.flush()  # Must not raise

    def test_enabled_property_is_false(self, no_langfuse_env):
        tracer = get_tracer()
        assert tracer.enabled is False


# ---------------------------------------------------------------------------
# Tracer degrades gracefully when langfuse package not installed
# ---------------------------------------------------------------------------


class TestLangfuseImportFailure:
    def test_degrades_when_langfuse_missing(self, monkeypatch):
        """
        Even when credentials are set, if langfuse package is not installed
        the tracer should fall back to a no-op rather than crashing.
        """
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")

        # Patch the import inside langfuse_tracer so it raises ImportError
        import src.tracing.langfuse_tracer as tracer_mod

        tracer_mod._tracer_instance

        # Monkeypatch builtins.__import__ to fail for 'langfuse'
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "langfuse":
                raise ImportError("langfuse not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Reset and re-get the tracer; should degrade to disabled
        reset_tracer()
        tracer = get_tracer()

        # Whether disabled or not, policy_span must work
        with tracer.policy_span(
            "get_demand_forecast", {"city": "Stockholm", "target": "heating_demand_mw"}
        ):
            pass

        # Restore
        monkeypatch.setattr(builtins, "__import__", real_import)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)


# ---------------------------------------------------------------------------
# PolicySpanContext directly
# ---------------------------------------------------------------------------


class TestPolicySpanContextDirectly:
    def test_disabled_span_set_outcome_noop(self):
        ctx = PolicySpanContext(
            tool_name="t",
            inputs={},
            _span=None,
            _enabled=False,
        )
        ctx.set_outcome(object())  # Must not raise
        ctx.set_error(ValueError())  # Must not raise
        ctx._close()  # Must not raise

    def test_enabled_span_with_bad_span_object(self):
        """When _span is an unexpected object, set_outcome must not crash."""

        class BadSpan:
            def update(self, **kwargs):
                raise RuntimeError("bad span")

            def end(self):
                raise RuntimeError("bad end")

        ctx = PolicySpanContext(
            tool_name="t",
            inputs={},
            _span=BadSpan(),
            _enabled=True,
        )
        ctx.set_outcome(object())  # Must not raise
        ctx.set_error(ValueError())  # Must not raise
        ctx._close()  # Must not raise
