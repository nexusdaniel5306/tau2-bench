"""
Layer 1: Simulation execution.

Runs a pre-built orchestrator and evaluates the result.
No registry dependency, no config parsing, no side effects.
"""

from typing import Optional, Union

from loguru import logger

from tau2.data_model.simulation import SimulationRun
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.modes import CommunicationMode
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.utils.tracing import set_output, simulation_span, trace_id


def run_simulation(
    orchestrator: Union[Orchestrator, FullDuplexOrchestrator],
    *,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    env_kwargs: Optional[dict] = None,
    run_id: Optional[str] = None,
) -> SimulationRun:
    """Run a simulation and evaluate the result.

    Takes a fully constructed orchestrator (with agent, user, environment, and task
    already wired in), runs the simulation, evaluates it, and returns the result
    with reward_info attached.

    This is the lowest-level entry point. It has no dependency on the registry
    or RunConfig -- everything is already encapsulated in the orchestrator.

    Args:
        orchestrator: A fully constructed Orchestrator (half-duplex) or
            FullDuplexOrchestrator (full-duplex/voice). Must have agent, user,
            environment, and task set.
        evaluation_type: The type of evaluation to perform. Defaults to ALL.
        env_kwargs: Additional kwargs passed to the evaluator's environment
            constructor (e.g., retrieval_variant for banking_knowledge).

    Returns:
        SimulationRun with reward_info attached.

    Example:
        # Build your own instances (no registry needed):
        env = MyEnvironment()
        agent = MyAgent(tools=env.get_tools(), domain_policy=env.get_policy())
        user = UserSimulator(llm="gpt-4.1", instructions=task.user_scenario,
                             tools=env.get_user_tools())
        orchestrator = Orchestrator(
            domain="airline", agent=agent, user=user,
            environment=env, task=task, max_steps=100,
        )
        result = run_simulation(orchestrator)
        print(result.reward_info.reward)
    """
    domain = orchestrator.environment.get_domain_name()
    task = orchestrator.task
    with simulation_span(
        run_id or orchestrator.simulation_id,
        orchestrator.simulation_id,
        task.id,
        domain,
    ) as span:
        simulation = orchestrator.run()
        simulation.policy = orchestrator.environment.get_policy()

        mode = (
            CommunicationMode.FULL_DUPLEX
            if isinstance(orchestrator, FullDuplexOrchestrator)
            else CommunicationMode.HALF_DUPLEX
        )
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=evaluation_type,
            solo_mode=getattr(orchestrator, "solo_mode", False),
            domain=domain,
            mode=mode,
            env_kwargs=env_kwargs,
        )
        simulation.reward_info = reward_info
        if span is not None:
            span.set_attribute("tau2.reward", reward_info.reward)
            span.set_attribute(
                "tau2.termination_reason", simulation.termination_reason.value
            )
            span.set_attribute("tau2.duration_seconds", simulation.duration)
            if simulation.agent_cost is not None:
                span.set_attribute("tau2.agent_cost_usd", simulation.agent_cost)
            if simulation.user_cost is not None:
                span.set_attribute("tau2.user_cost_usd", simulation.user_cost)
            simulation.info = {
                **(simulation.info or {}),
                "otel_trace_id": trace_id(span),
            }
        set_output(
            span,
            {
                "reward": reward_info.reward,
                "termination_reason": simulation.termination_reason.value,
                "agent_cost_usd": simulation.agent_cost,
                "user_cost_usd": simulation.user_cost,
            },
        )

    logger.info(
        f"Simulation complete: domain={domain}, task={task.id}, "
        f"reward={reward_info.reward}"
    )

    return simulation
