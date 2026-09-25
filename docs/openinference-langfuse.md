# Trace a Telecom simulation in Langfuse

Text simulations emit OpenInference-compatible OpenTelemetry spans when
`TAU2_OTEL_ENABLED=1`. Install the optional exporter with
`uv sync --extra dev --extra tracing`. Normal tau2 runs leave tracing off.

## Configuration

Start the capstone Langfuse stack and create a project API key pair in its UI.
The capstone Compose file routes Langfuse through Caddy at
`http://localhost:82`. Set these values in the shell that starts tau2 (or in
the untracked `.env` file). Keep the key pair and encoded authorization header
out of Git:

```sh
export TAU2_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:82/api/public/otel/v1/traces
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Basic $(printf '%s' "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" | base64 | tr -d '\n'),x-langfuse-ingestion-version=4"
uv run --extra tracing tau2 run --domain telecom --task-split-name small \
  --task-ids '[mobile_data_issue]user_abroad_roaming_disabled_on[PERSONA:Easy]' \
  --num-trials 1 --max-concurrency 1 --max-steps 40 --timeout 360 \
  --agent-llm openrouter/stealth/space-bunny-alpha \
  --user-llm openrouter/stealth/space-bunny-alpha \
  --agent-llm-args '{"temperature":0,"max_tokens":512}' \
  --user-llm-args '{"temperature":0,"max_tokens":512}' \
  --save-to w2-s2-telecom-space-bunny-verified-20260925
```

The endpoint uses OTLP over HTTP/protobuf. Langfuse v4's ingestion header
makes the new observations visible promptly. The model and user simulator
still need their normal provider credentials (`OPENROUTER_API_KEY` for this
example). The run's result is saved at
`data/simulations/w2-s2-telecom-space-bunny-verified-20260925/results.json` and is inspectable with
`tau2 view`.

## Span boundary and correlation

`run_simulation` creates one root `tau2.simulation` span around the
orchestrator and evaluation. Each orchestrator step is a `CHAIN` child.
The text agent's response has an `AGENT` child with its triggering message and
response. `llm_utils.generate` creates an `LLM` child around each LiteLLM
completion, including the user simulator's calls. Each environment tool call
creates a `TOOL` child with its arguments, result, and error flag. Tool calls
are children of the orchestration step that executes them; they are not
children of the earlier step that selected them.

Every span carries `tau2.run_id`, `tau2.simulation_id`, `tau2.task_id`,
`tau2.domain`, Langfuse trace metadata, and `session.id` for the saved batch.
The root also records reward, termination reason, runtime, and the agent and
user costs reported by tau2. The exact Langfuse/OTel trace ID is saved in
each `SimulationRun.info.otel_trace_id`; compare it with the trace ID in the
Langfuse UI. Search Langfuse by the run name or simulation ID to find it.

The model span currently records model name, raw request/response JSON,
prompt/completion token totals, and LiteLLM's estimated total cost. It does
not emit flattened OpenInference `llm.input_messages.*` and
`llm.output_messages.*` fields, provider-specific identity, or separate
prompt/completion costs. Full-duplex voice internals have not been instrumented
at their model-call boundary; this setup targets text Telecom runs.

## Verified Space Bunny run

The [redacted span tree](evidence/w2-s2-space-bunny-20260925.json) records the
September 25, 2026 run. Its simulation ID is
`b02438d8-9a48-498f-a6c7-d039c5c7cb74`, and the Langfuse trace ID saved in
`results.json` is `a64a398e09551f1134a7c4cb7cb1a59b`. Langfuse returned
95 unique observations: 1 root and 14 agent spans, 40 step spans, 29 model
generations, and 11 tool calls. Every observation has the same trace and
simulation IDs, a resolving parent link where applicable, and nonempty input
and output. The run lasted 44.75 seconds, reported $0.00 model cost, and
reached the 40-step cap with reward 0. The span tree confirms tracing and tool
execution; this model run did not complete the Telecom task.
