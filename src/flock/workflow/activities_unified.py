"""Unified Temporal activities for running a chain of agents with the new component architecture."""

from datetime import datetime

from opentelemetry import trace
from temporalio import activity

from flock.core.context.context import FlockContext
from flock.core.context.context_vars import FLOCK_CURRENT_AGENT, FLOCK_MODEL
from flock.core.flock_agent_unified import FlockAgentUnified
from flock.core.flock_registry import get_registry
# HandOffRequest removed - using agent.next_agent directly
from flock.core.logging.logging import get_logger
from flock.core.util.input_resolver import resolve_inputs

logger = get_logger("activities.unified")
tracer = trace.get_tracer(__name__)


@activity.defn
async def run_agent_unified(context: FlockContext) -> dict:
    """Runs a chain of agents using the unified component architecture.

    Key changes from original:
    - Uses FlockAgentUnified with components list
    - Routing decisions come from agent.next_handoff (set during evaluation)
    - Simplified workflow - no separate routing step
    
    The context contains state, history, and agent definitions.
    After each agent run, its output is merged into the context.
    """
    # Start a top-level span for the entire run_agent activity.
    with tracer.start_as_current_span("run_agent_unified") as span:
        registry = get_registry()

        previous_agent_name = ""
        if isinstance(context, dict):
            context = FlockContext.from_dict(context)
        current_agent_name = context.get_variable(FLOCK_CURRENT_AGENT)
        span.set_attribute("initial.agent", current_agent_name)
        logger.info("Starting unified agent chain", initial_agent=current_agent_name)

        agent = registry.get_agent(current_agent_name)
        if not agent:
            logger.error("Agent not found", agent=current_agent_name)
            span.record_exception(
                Exception(f"Agent '{current_agent_name}' not found")
            )
            return {"error": f"Agent '{current_agent_name}' not found."}

        # Set model if not configured
        if agent.model is None:
            model = context.get_variable(FLOCK_MODEL)
            if hasattr(agent, 'evaluator') and agent.evaluator and hasattr(agent.evaluator, 'config'):
                agent.evaluator.config.model = model
            agent.model = model
            
        agent.resolve_callables(context=context)

        # Loop over agents in the chain.
        while agent:
            # Create a nested span for this iteration.
            with tracer.start_as_current_span("agent_iteration_unified") as iter_span:
                iter_span.set_attribute("agent.name", agent.name)
                agent.context = context
                
                # Resolve inputs for the agent.
                agent_inputs = resolve_inputs(
                    agent.input, context, previous_agent_name
                )
                iter_span.add_event(
                    "resolved inputs", attributes={"inputs": str(agent_inputs)}
                )

                # Execute the agent with its own span.
                # NOTE: In unified architecture, routing happens DURING execution
                with tracer.start_as_current_span("execute_agent_unified") as exec_span:
                    logger.info("Executing unified agent", agent=agent.name)
                    try:
                        # This will set agent.next_handoff during evaluation
                        result = await agent.run_async(agent_inputs)
                        exec_span.set_attribute("result", str(result))
                        logger.debug(
                            "Unified agent execution completed", agent=agent.name
                        )
                    except Exception as e:
                        logger.error(
                            "Unified agent execution failed",
                            agent=agent.name,
                            error=str(e),
                        )
                        exec_span.record_exception(e)
                        raise

                # Get routing decision from agent.next_handoff (set during evaluation)
                handoff_data = agent.next_handoff
                
                if handoff_data is None:
                    # No routing component or router decided to end workflow
                    logger.info(
                        "No handoff data found, completing chain",
                        agent=agent.name,
                    )
                    context.record(
                        agent.name,
                        result,
                        timestamp=datetime.now().isoformat(),
                        hand_off=None,
                        called_from=previous_agent_name,
                    )
                    iter_span.add_event("chain completed - no handoff")
                    return result

                # Process the handoff data
                logger.info(
                    f"Processing handoff to: {handoff_data.next_agent}",
                    agent=agent.name,
                )

                # Handle callable handoff functions (if still needed for backward compatibility)
                if callable(handoff_data):
                    logger.debug("Executing handoff function", agent=agent.name)
                    try:
                        handoff_data = handoff_data(context, result)
                        if isinstance(handoff_data.next_agent, FlockAgentUnified):
                            handoff_data.next_agent = handoff_data.next_agent.name
                    except Exception as e:
                        logger.error(
                            "Handoff function error",
                            agent=agent.name,
                            error=str(e),
                        )
                        iter_span.record_exception(e)
                        return {"error": f"Handoff function error: {e}"}
                elif isinstance(handoff_data.next_agent, FlockAgentUnified):
                    handoff_data.next_agent = handoff_data.next_agent.name

                # Check if we should end the workflow
                if not handoff_data.next_agent:
                    logger.info(
                        "Router found no suitable next agent",
                        agent=agent.name,
                    )
                    context.record(
                        agent.name,
                        result,
                        timestamp=datetime.now().isoformat(),
                        hand_off=None,
                        called_from=previous_agent_name,
                    )
                    logger.info("Completing chain", agent=agent.name)
                    iter_span.add_event("chain completed - no next agent")
                    return result

                # Record the agent run in the context.
                context.record(
                    agent.name,
                    result,
                    timestamp=datetime.now().isoformat(),
                    hand_off=handoff_data.model_dump(),
                    called_from=previous_agent_name,
                )
                previous_agent_name = agent.name
                previous_agent_output = agent.output
                
                if handoff_data.override_context:
                    context.update(handoff_data.override_context)

                # Prepare the next agent.
                try:
                    agent = registry.get_agent(handoff_data.next_agent)
                    if not agent:
                        logger.error(
                            "Next agent not found",
                            agent=handoff_data.next_agent,
                        )
                        iter_span.record_exception(
                            Exception(
                                f"Next agent '{handoff_data.next_agent}' not found"
                            )
                        )
                        return {
                            "error": f"Next agent '{handoff_data.next_agent}' not found."
                        }

                    # Apply handoff modifications to the next agent
                    if handoff_data.output_to_input_merge_strategy == "add":
                        agent.input = previous_agent_output + ", " + agent.input

                    if handoff_data.add_input_fields:
                        for field in handoff_data.add_input_fields:
                            agent.input = field + ", " + agent.input

                    if handoff_data.add_output_fields:
                        for field in handoff_data.add_output_fields:
                            agent.output = field + ", " + agent.output

                    if handoff_data.add_description:
                        if agent.description:
                            agent.description = (
                                agent.description
                                + "\n"
                                + handoff_data.add_description
                            )
                        else:
                            agent.description = handoff_data.add_description

                    agent.resolve_callables(context=context)
                    context.set_variable(FLOCK_CURRENT_AGENT, agent.name)

                    logger.info("Handing off to next agent", next=agent.name)
                    iter_span.set_attribute("next.agent", agent.name)
                    
                except Exception as e:
                    logger.error("Error during handoff", error=str(e))
                    iter_span.record_exception(e)
                    return {"error": f"Error during handoff: {e}"}

        # If the loop exits unexpectedly, return the initial input.
        return context.get_variable("init_input")


# Backward compatibility wrapper
@activity.defn  
async def run_agent(context: FlockContext) -> dict:
    """Backward compatibility wrapper for run_agent_unified."""
    logger.warning(
        "Using backward compatibility wrapper. Consider migrating to run_agent_unified."
    )
    return await run_agent_unified(context)
