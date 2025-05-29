# src/flock/core/flock_agent.py
"""FlockAgent with unified component architecture."""

import uuid
from abc import ABC
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from flock.core.agent.flock_agent_execution import FlockAgentExecution
from flock.core.agent.flock_agent_integration import FlockAgentIntegration
from flock.core.agent.flock_agent_serialization import FlockAgentSerialization
from flock.core.component.agent_component_base import AgentComponent
from flock.core.component.evaluation_component_base import EvaluationComponentBase
from flock.core.component.routing_component_base import RoutingComponentBase
from flock.core.config.flock_agent_config import FlockAgentConfig
from flock.core.context.context import FlockContext

from flock.core.mcp.flock_mcp_server import FlockMCPServerBase
from flock.workflow.temporal_config import TemporalActivityConfig

from pydantic import BaseModel, Field

# Mixins and Serialization components
from flock.core.mixin.dspy_integration import DSPyIntegrationMixin
from flock.core.serialization.serializable import Serializable
from flock.core.logging.logging import get_logger

logger = get_logger("agent.unified")

T = TypeVar("T", bound="FlockAgent")

SignatureType = (
    str
    | Callable[..., str]
    | type[BaseModel]
    | Callable[..., type[BaseModel]]
    | None
)


class FlockAgent(BaseModel, Serializable, DSPyIntegrationMixin, ABC):
    """Unified FlockAgent using the new component architecture.
    
    This is the next-generation FlockAgent that uses a single components list
    instead of separate evaluator, router, and modules. All agent functionality
    is now provided through AgentComponent instances.
    
    Key changes:
    - components: list[AgentComponent] - unified component list
    - next_agent: str | None - explicit workflow state
    - evaluator/router properties - convenience access to primary components
    """

    agent_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Internal, Unique UUID4 for this agent instance.",
    )

    name: str = Field(..., description="Unique identifier for the agent.")

    model: str | None = Field(
        None,
        description="The model identifier to use (e.g., 'openai/gpt-4o'). If None, uses Flock's default.",
    )
    description: str | Callable[..., str] | None = Field(
        "",
        description="A human-readable description or a callable returning one.",
    )
    input: SignatureType = Field(
        None,
        description="Signature for input keys. Supports type hints (:) and descriptions (|).",
    )
    output: SignatureType = Field(
        None,
        description="Signature for output keys. Supports type hints (:) and descriptions (|).",
    )
    tools: list[Callable[..., Any]] | None = Field(
        default=None,
        description="List of callable tools the agent can use. These must be registered.",
    )
    servers: list[str | FlockMCPServerBase] | None = Field(
        default=None,
        description="List of MCP Servers the agent can use to enhance its capabilities.",
    )

    # --- UNIFIED COMPONENT SYSTEM ---
    components: list[AgentComponent] = Field(
        default_factory=list,
        description="List of all agent components (evaluators, routers, modules).",
    )
    
    # --- EXPLICIT WORKFLOW STATE ---
    next_agent: str | Callable[..., str] | None = Field(
        default=None,
        exclude=True,  # Runtime state, don't serialize
        description="Next agent in workflow - set by user or routing components.",
    )

    config: FlockAgentConfig = Field(
        default_factory=lambda: FlockAgentConfig(),
        description="Configuration for this agent.",
    )

    temporal_activity_config: TemporalActivityConfig | None = Field(
        default=None,
        description="Optional Temporal settings specific to this agent.",
    )

    # --- Runtime State (Excluded from Serialization) ---
    context: FlockContext | None = Field(
        default=None,
        exclude=True,
        description="Runtime context associated with the flock execution.",
    )

    def __init__(
        self,
        name: str,
        model: str | None = None,
        description: str | Callable[..., str] | None = "",
        input: SignatureType = None,
        output: SignatureType = None,
        tools: list[Callable[..., Any]] | None = None,
        servers: list[str | FlockMCPServerBase] | None = None,
        components: list[AgentComponent] | None = None,
        write_to_file: bool = False,
        wait_for_input: bool = False,
        temporal_activity_config: TemporalActivityConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            model=model,
            description=description,
            input=input,
            output=output,
            tools=tools,
            servers=servers,
            components=components if components is not None else [],
            config=FlockAgentConfig(
                write_to_file=write_to_file,
                wait_for_input=wait_for_input,
            ),
            temporal_activity_config=temporal_activity_config,
            **kwargs,
        )

        # Initialize helper systems (reuse existing logic)
        self._execution = FlockAgentExecution(self)
        self._integration = FlockAgentIntegration(self)
        self._serialization = FlockAgentSerialization(self)

    # --- CONVENIENCE PROPERTIES ---
    # These provide familiar access patterns while using the unified model
    
    @property
    def evaluator(self) -> EvaluationComponentBase | None:
        """Get the primary evaluation component for this agent."""
        return self.components_helper.get_primary_evaluator()
    
    @property
    def router(self) -> RoutingComponentBase | None:
        """Get the primary routing component for this agent."""
        return self.components_helper.get_primary_router()
    
    @property 
    def modules(self) -> list[AgentComponent]:
        """Get all components (for backward compatibility with module-style access)."""
        return self.components.copy()
    
    @property
    def components_helper(self):
        """Get the component management helper."""
        if not hasattr(self, '_components_helper'):
            from flock.core.agent.flock_agent_components import FlockAgentComponents
            self._components_helper = FlockAgentComponents(self)
        return self._components_helper
    
    # Component management delegated to components_helper
    def add_component(self, component: AgentComponent) -> None:
        """Add a component to this agent."""
        self.components_helper.add_component(component)

    def remove_component(self, component_name: str) -> None:
        """Remove a component from this agent."""
        self.components_helper.remove_component(component_name)

    def get_component(self, component_name: str) -> AgentComponent | None:
        """Get a component by name."""
        return self.components_helper.get_component(component_name)

    # --- BACKWARD COMPATIBILITY METHODS ---
    # These maintain the old API while using the new architecture
    
    def add_module(self, module: AgentComponent) -> None:
        """Add a module (backward compatibility)."""
        self.add_component(module)

    def remove_module(self, module_name: str) -> None:
        """Remove a module (backward compatibility)."""
        self.remove_component(module_name)

    def get_module(self, module_name: str) -> AgentComponent | None:
        """Get a module (backward compatibility)."""
        return self.get_component(module_name)

    def get_enabled_modules(self) -> list[AgentComponent]:
        """Get enabled modules (backward compatibility)."""
        return self.get_enabled_components()

    # --- UNIFIED LIFECYCLE EXECUTION ---
    
    async def initialize(self, inputs: dict[str, Any]) -> None:
        """Initialize agent and run component initializers."""
        logger.debug(f"Initializing unified agent '{self.name}'")
        
        for component in self.get_enabled_components():
            try:
                await component.on_initialize(self, inputs, self.context)
            except Exception as e:
                logger.error(f"Error initializing component '{component.name}': {e}")

    async def evaluate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Core evaluation logic using unified component system."""
        logger.debug(f"Evaluating unified agent '{self.name}'")
        
        current_inputs = inputs
        
        # 1. Pre-evaluate hooks (all components)
        for component in self.get_enabled_components():
            try:
                current_inputs = await component.on_pre_evaluate(self, current_inputs, self.context)
            except Exception as e:
                logger.error(f"Error in pre-evaluate for component '{component.name}': {e}")

        # 2. Core evaluation (primary evaluator component)
        result = current_inputs  # Default if no evaluator
        
        evaluator = self.evaluator
        if evaluator:
            try:
                # Get tools through integration system
                registered_tools = self.tools or []
                mcp_tools = await self._integration.get_mcp_tools() if self.servers else []
                
                result = await evaluator.evaluate_core(
                    self, current_inputs, self.context, registered_tools, mcp_tools
                )
            except Exception as e:
                logger.error(f"Error in core evaluation: {e}")
                raise
        else:
            logger.warning(f"Agent '{self.name}' has no evaluation component")

        # 3. Post-evaluate hooks (all components)
        current_result = result
        for component in self.get_enabled_components():
            try:
                tmp_result = await component.on_post_evaluate(
                    self, current_inputs, self.context, current_result
                )
                if tmp_result is not None:
                    current_result = tmp_result
            except Exception as e:
                logger.error(f"Error in post-evaluate for component '{component.name}': {e}")

        # 4. Determine next step (routing components)
        self.next_agent = None  # Reset
        
        router = self.router
        if router:
            try:
                self.next_agent = await router.determine_next_step(
                    self, current_result, self.context
                )
            except Exception as e:
                logger.error(f"Error in routing: {e}")

        return current_result

    async def terminate(self, inputs: dict[str, Any], result: dict[str, Any]) -> None:
        """Terminate agent and run component terminators."""
        logger.debug(f"Terminating unified agent '{self.name}'")
        
        current_result = result
        
        for component in self.get_enabled_components():
            try:
                tmp_result = await component.on_terminate(self, inputs, self.context, current_result)
                if tmp_result is not None:
                    current_result = tmp_result
            except Exception as e:
                logger.error(f"Error in terminate for component '{component.name}': {e}")

        # Handle output file writing
        if self.config.write_to_file:
            self._serialization._save_output(self.name, current_result)

        if self.config.wait_for_input:
            input("Press Enter to continue...")

    async def on_error(self, error: Exception, inputs: dict[str, Any]) -> None:
        """Handle errors and run component error handlers."""
        logger.error(f"Error occurred in unified agent '{self.name}': {error}")
        
        for component in self.get_enabled_components():
            try:
                await component.on_error(self, inputs, self.context, error)
            except Exception as e:
                logger.error(f"Error in error handler for component '{component.name}': {e}")

    # --- EXECUTION METHODS ---
    # Delegate to the execution system
    
    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Synchronous wrapper for run_async."""
        return self._execution.run(inputs)

    async def run_async(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Asynchronous execution logic with unified lifecycle."""
        try:
            await self.initialize(inputs)
            result = await self.evaluate(inputs)
            await self.terminate(inputs, result)
            logger.info("Unified agent run completed", agent=self.name)
            return result
        except Exception as run_error:
            logger.error(f"Error running unified agent: {run_error}")
            await self.on_error(run_error, inputs)
            raise

    # --- SERIALIZATION ---
    # Delegate to the serialization system
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary using unified component serialization."""
        from flock.core.flock_registry import get_registry
        from flock.core.serialization.serialization_utils import serialize_item

        FlockRegistry = get_registry()
        
        # Basic agent data (exclude components and runtime state)
        exclude = ["components", "context", "next_agent", "tools", "servers"]
        
        # Handle callable fields
        if callable(self.description):
            exclude.append("description")
        if callable(self.input):
            exclude.append("input")
        if callable(self.output):
            exclude.append("output")

        data = self.model_dump(
            exclude=exclude,
            mode="json",
            exclude_none=True,
        )
        
        # Serialize components list
        if self.components:
            serialized_components = []
            for component in self.components:
                try:
                    comp_type = type(component)
                    type_name = FlockRegistry.get_component_type_name(comp_type)
                    if type_name:
                        component_data = serialize_item(component)
                        if isinstance(component_data, dict):
                            component_data["type"] = type_name
                            serialized_components.append(component_data)
                        else:
                            logger.warning(f"Component {component.name} serialization failed")
                    else:
                        logger.warning(f"Component {component.name} type not registered")
                except Exception as e:
                    logger.error(f"Failed to serialize component {component.name}: {e}")
            
            if serialized_components:
                data["components"] = serialized_components

        # Handle other serializable fields (tools, servers, callables)
        if self.tools:
            serialized_tools = []
            for tool in self.tools:
                if callable(tool):
                    path_str = FlockRegistry.get_callable_path_string(tool)
                    if path_str:
                        func_name = path_str.split(".")[-1]
                        serialized_tools.append(func_name)
            if serialized_tools:
                data["tools"] = serialized_tools

        if self.servers:
            serialized_servers = []
            for server in self.servers:
                if isinstance(server, str):
                    serialized_servers.append(server)
                elif hasattr(server, 'config') and hasattr(server.config, 'name'):
                    serialized_servers.append(server.config.name)
            if serialized_servers:
                data["mcp_servers"] = serialized_servers

        return data

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Deserialize from dictionary using unified component deserialization."""
        from flock.core.flock_registry import get_registry
        from flock.core.serialization.serialization_utils import deserialize_component

        registry = get_registry()
        
        # Separate component data from agent data
        components_data = data.pop("components", [])
        tools_data = data.pop("tools", [])
        servers_data = data.pop("mcp_servers", [])
        
        # Create base agent
        agent = cls(**data)
        
        # Deserialize components
        if components_data:
            for component_data in components_data:
                try:
                    # Use the existing deserialize_component function
                    component = deserialize_component(component_data, AgentComponent)
                    if component:
                        agent.add_component(component)
                except Exception as e:
                    logger.error(f"Failed to deserialize component: {e}")

        # Deserialize tools
        if tools_data:
            agent.tools = []
            for tool_name in tools_data:
                try:
                    tool = registry.get_callable(tool_name)
                    if tool:
                        agent.tools.append(tool)
                except Exception as e:
                    logger.warning(f"Could not resolve tool '{tool_name}': {e}")

        # Deserialize servers
        if servers_data:
            agent.servers = servers_data  # Store as names, resolve at runtime

        return agent

    def set_model(self, model: str):
        """Set the model for the agent and its evaluator.
        
        This method updates both the agent's model property and propagates
        the model to the evaluator component if it has a config with a model field.
        """
        self.model = model
        if self.evaluator and hasattr(self.evaluator, "config"):
            self.evaluator.config.model = model
            logger.info(
                f"Set model to '{model}' for agent '{self.name}' and its evaluator."
            )
        elif self.evaluator:
            logger.warning(
                f"Evaluator for agent '{self.name}' does not have a standard config to set model."
            )
        else:
            logger.warning(
                f"Agent '{self.name}' has no evaluator to set model for."
            )

    def resolve_callables(self, context: FlockContext | None = None) -> None:
        """Resolves callable fields (description, input, output) using context."""
        if callable(self.description):
            self.description = self.description(
                context
            )  # Pass context if needed by callable
        if callable(self.input):
            self.input = self.input(context)
        if callable(self.output):
            self.output = self.output(context)

    @property
    def resolved_description(self) -> str | None:
        """Returns the resolved agent description.
        If the description is a callable, it attempts to call it.
        Returns None if the description is None or a callable that fails.
        """
        if callable(self.description):
            try:
                # Attempt to call without context first.
                return self.description()
            except TypeError:
                # Log a warning that context might be needed
                logger.warning(
                    f"Callable description for agent '{self.name}' could not be resolved "
                    f"without context via the simple 'resolved_description' property. "
                    f"Consider calling 'agent.resolve_callables(context)' beforehand if context is required."
                )
                return None  # Or a placeholder like "[Callable Description]"
            except Exception as e:
                logger.error(
                    f"Error resolving callable description for agent '{self.name}': {e}"
                )
                return None
        elif isinstance(self.description, str):
            return self.description
        return None

    def _save_output(self, agent_name: str, result: dict[str, Any]) -> None:
        """Save output to file if configured."""
        if not self.config.write_to_file:
            return

        from datetime import datetime
        import os
        import json
        from flock.core.serialization.json_encoder import FlockJSONEncoder

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{agent_name}_output_{timestamp}.json"
        filepath = os.path.join(".flock/output/", filename)
        os.makedirs(".flock/output/", exist_ok=True)

        output_data = {
            "agent": agent_name,
            "timestamp": timestamp,
            "output": result,
        }

        try:
            with open(filepath, "w") as f:
                json.dump(output_data, f, indent=2, cls=FlockJSONEncoder)
        except Exception as e:
            logger.warning(f"Failed to save output to file: {e}")

    def add_legacy_component(
        self,
        config_instance: Any,
        component_name: str | None = None,
    ) -> "FlockAgent":
        """Adds or replaces a component based on its configuration object.
        
        This method provides backward compatibility with the old component system
        while working with the new unified components architecture.

        Args:
            config_instance: An instance of a config class for FlockModule, FlockRouter, or FlockEvaluator.
            component_name: Explicit name for the component.

        Returns:
            self for potential chaining.
        """
        from flock.core.flock_registry import get_registry

        config_type = type(config_instance)
        registry = get_registry()
        logger.debug(
            f"Attempting to add component via config: {config_type.__name__}"
        )

        # Find Component Class using Registry Map
        ComponentClass = registry.get_component_class_for_config(config_type)

        if not ComponentClass:
            logger.error(
                f"No component class registered for config type {config_type.__name__}. Use @flock_component(config_class=...) on the component."
            )
            raise TypeError(
                f"Cannot find component class for config {config_type.__name__}"
            )

        component_class_name = ComponentClass.__name__
        logger.debug(
            f"Found component class '{component_class_name}' mapped to config '{config_type.__name__}'"
        )

        # Determine instance name
        instance_name = component_name
        if not instance_name:
            instance_name = getattr(
                config_instance, "name", component_class_name.lower()
            )

        # Instantiate the Component
        try:
            init_args = {"config": config_instance, "name": instance_name}
            component_instance = ComponentClass(**init_args)
        except Exception as e:
            logger.error(
                f"Failed to instantiate {ComponentClass.__name__} with config {config_type.__name__}: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Component instantiation failed: {e}") from e

        # Add to unified components list
        self.add_component(component_instance)
        logger.info(
            f"Added component '{instance_name}' (type: {ComponentClass.__name__}) to agent '{self.name}'"
        )

        return self

    # --- Pydantic v2 Configuration ---
    model_config = {"arbitrary_types_allowed": True}
