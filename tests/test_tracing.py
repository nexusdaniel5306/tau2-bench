"""Offline check of the trace hierarchy emitted by a text simulation."""

import json
import random
from typing import Callable

import pytest
from litellm import ModelResponse
from opentelemetry import trace

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import UserMessage
from tau2.data_model.simulation import RewardInfo
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner.simulation import run_simulation
from tau2.utils import llm_utils, tracing


def test_trace_ids_do_not_repeat_when_tau2_reseeds_random():
    generator = tracing._IndependentIdGenerator()
    random.seed(42)
    first = (generator.generate_trace_id(), generator.generate_span_id())
    random.seed(42)
    second = (generator.generate_trace_id(), generator.generate_span_id())
    assert first != second


def test_tracer_reused_after_initialization(monkeypatch):
    monkeypatch.setenv("TAU2_OTEL_ENABLED", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost/otel")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "Authorization=Basic test")
    monkeypatch.setattr(tracing, "_provider", object())
    sentinel = object()
    monkeypatch.setattr(trace, "get_tracer", lambda name: sentinel)
    assert tracing._tracer() is sentinel


class ScriptedUser:
    def __init__(self):
        self.turn = 0

    def get_init_state(self, message_history=None):
        return None

    def generate_next_message(self, message, state):
        self.turn += 1
        content = "Create a task" if self.turn == 1 else "###STOP###"
        return UserMessage(role="user", content=content, cost=0.0), state

    @classmethod
    def is_stop(cls, message):
        return message.content == "###STOP###"

    def stop(self, message=None, state=None):
        pass

    def set_seed(self, seed):
        pass


def _response(content=None, tool_calls=None):
    return ModelResponse(
        model="gpt-3.5-turbo",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def test_text_simulation_emits_correlated_openinference_tree(
    monkeypatch,
    get_environment: Callable[[], Environment],
    base_task: Task,
):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_tracer", lambda: provider.get_tracer("test"))
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.01)
    responses = iter(
        [
            _response(
                tool_calls=[
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "create_task",
                            "arguments": json.dumps(
                                {"user_id": "user_1", "title": "test"}
                            ),
                        },
                    }
                ]
            ),
            _response(content="Done"),
        ]
    )
    monkeypatch.setattr(llm_utils, "completion", lambda **kwargs: next(responses))
    monkeypatch.setattr(
        "tau2.runner.simulation.evaluate_simulation",
        lambda **kwargs: RewardInfo(reward=1.0),
    )

    environment = get_environment()
    agent = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm="gpt-3.5-turbo",
    )
    orchestrator = Orchestrator(
        domain="mock",
        agent=agent,
        user=ScriptedUser(),
        environment=environment,
        task=base_task,
        simulation_id="sim-test",
    )
    result = run_simulation(orchestrator, run_id="batch-test")

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans if span.name != "tau2.step"}
    root = by_name["tau2.simulation"]
    agent_span = by_name["tau2.agent_step"]
    tool = by_name["create_task"]
    llm_spans = [span for span in spans if span.name == "agent_response"]
    assert len(llm_spans) == 2
    assert result.info["otel_trace_id"] == format(root.context.trace_id, "032x")
    assert root.attributes["tau2.simulation_id"] == "sim-test"
    assert all(span.attributes["session.id"] == "batch-test" for span in spans)
    assert agent_span.parent.span_id in {
        span.context.span_id for span in spans if span.name == "tau2.step"
    }
    assert llm_spans[0].parent.span_id != root.context.span_id
    assert tool.parent.span_id in {
        span.context.span_id for span in spans if span.name == "tau2.step"
    }
    assert tool.attributes["tool.name"] == "create_task"
    assert json.loads(tool.attributes["output.value"])["error"] is False
    assert llm_spans[0].attributes["openinference.span.kind"] == "LLM"
    assert llm_spans[0].attributes["llm.token_count.prompt"] == 10
