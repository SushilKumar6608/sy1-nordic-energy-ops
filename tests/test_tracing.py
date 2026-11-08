from src.tracing.langfuse_tracer import LangfuseTracer


def test_tracer_instantiates_without_keys():
    tracer = LangfuseTracer(public_key=None, secret_key=None)
    assert tracer is not None


def test_tracer_disabled_without_keys():
    tracer = LangfuseTracer(public_key=None, secret_key=None)
    assert tracer.enabled is False


def test_start_trace_returns_none_when_disabled():
    tracer = LangfuseTracer(public_key=None, secret_key=None)
    result = tracer.start_trace(name="test", input_data={})
    assert result is None


def test_end_trace_does_not_raise_when_disabled():
    tracer = LangfuseTracer(public_key=None, secret_key=None)
    tracer.end_trace(trace_id=None, output_data={})
