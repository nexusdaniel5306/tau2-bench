"""Opt-in OpenInference spans exported through the standard OTLP HTTP exporter."""

import json
import os
import secrets
import threading
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any, Iterator

_provider = None
_lock = threading.Lock()
_trace_attributes: ContextVar[dict[str, str]] = ContextVar(
    "tau2_trace_attributes", default={}
)


class _IndependentIdGenerator:
    """Avoid ID reuse when tau2 seeds Python's random generator for a trial."""

    def generate_span_id(self) -> int:
        return secrets.randbits(64) or 1

    def generate_trace_id(self) -> int:
        return secrets.randbits(128) or 1


def _tracer():
    if os.environ.get("TAU2_OTEL_ENABLED") != "1":
        return None
    if not os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        raise ValueError(
            "TAU2_OTEL_ENABLED requires OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        )
    if not os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS"):
        raise ValueError("TAU2_OTEL_ENABLED requires OTEL_EXPORTER_OTLP_TRACES_HEADERS")

    from opentelemetry import trace

    global _provider
    with _lock:
        if _provider is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider(
                resource=Resource.create({"service.name": "tau2-bench"}),
                id_generator=_IndependentIdGenerator(),
            )
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            trace.set_tracer_provider(provider)
            _provider = provider
    return trace.get_tracer("tau2-bench")


def as_json(value: Any) -> str:
    """Serialize trace inputs and outputs as OpenInference JSON values."""
    return json.dumps(value, default=str, ensure_ascii=False)


@contextmanager
def start_span(name: str, kind: str, *, input_value: Any = None) -> Iterator[Any]:
    """Start an OpenInference span when OTLP tracing is enabled."""
    tracer = _tracer()
    if tracer is None:
        with nullcontext() as span:
            yield span
        return
    with tracer.start_as_current_span(name) as span:
        span.set_attribute("openinference.span.kind", kind)
        span.set_attribute(
            "langfuse.observation.type",
            {"LLM": "generation", "AGENT": "agent"}.get(kind, kind.lower()),
        )
        for key, value in _trace_attributes.get().items():
            span.set_attribute(key, value)
        if input_value is not None:
            span.set_attribute("input.value", as_json(input_value))
            span.set_attribute("input.mime_type", "application/json")
        yield span


@contextmanager
def simulation_span(
    run_id: str, simulation_id: str, task_id: str, domain: str
) -> Iterator[Any]:
    """Keep tau2 identity on every observation for Langfuse filtering."""
    attrs = {
        "langfuse.trace.name": f"tau2.{domain}",
        "session.id": run_id,
        "langfuse.trace.metadata.run_id": run_id,
        "langfuse.trace.metadata.simulation_id": simulation_id,
        "langfuse.trace.metadata.task_id": task_id,
        "tau2.run_id": run_id,
        "tau2.simulation_id": simulation_id,
        "tau2.task_id": task_id,
        "tau2.domain": domain,
    }
    token = _trace_attributes.set(attrs)
    try:
        with start_span(
            "tau2.simulation",
            "AGENT",
            input_value={
                "run_id": run_id,
                "simulation_id": simulation_id,
                "task_id": task_id,
            },
        ) as span:
            yield span
    finally:
        _trace_attributes.reset(token)


def set_output(span: Any, value: Any) -> None:
    """Attach a JSON output to an active span."""
    if span is not None:
        span.set_attribute("output.value", as_json(value))
        span.set_attribute("output.mime_type", "application/json")


def trace_id(span: Any) -> str | None:
    """Return the OTel trace ID in Langfuse's 32-digit hex format."""
    if span is None:
        return None
    return format(span.get_span_context().trace_id, "032x")
