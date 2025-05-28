# src/flock/core/agent/flock_agent_components.py
"""Component management functionality for FlockAgent."""

from typing import TYPE_CHECKING, Any

from flock.core.flock_evaluator import FlockEvaluator, FlockEvaluatorConfig
from flock.core.flock_module import FlockModule, FlockModuleConfig
from flock.core.flock_router import FlockRouter, FlockRouterConfig
from flock.core.logging.logging import get_logger

if TYPE_CHECKING:
    from flock.core.flock_agent import FlockAgent

logger = get_logger("agent.components")


class FlockAgentComponents:
    """Handles component management for FlockAgent including modules, evaluators, and routers."""

    def __init__(self, agent: "FlockAgent"):
        self.agent = agent

    def add_module(self, module: FlockModule) -> None:
        """Add a module to this agent."""
        if not module.name:
            logger.error("Module must have a name to be added.")
            return
        if module.name in self.agent.modules:
            logger.warning(f"Overwriting existing module: {module.name}")
        self.agent.modules[module.name] = module
        logger.debug(f"Added module '{module.name}' to agent '{self.agent.name}'")

    def remove_module(self, module_name: str) -> None:
        """Remove a module from this agent."""
        if module_name in self.agent.modules:
            del self.agent.modules[module_name]
            logger.debug(
                f"Removed module '{module_name}' from agent '{self.agent.name}'"
            )
        else:
            logger.warning(
                f"Module '{module_name}' not found on agent '{self.agent.name}'."
            )

    def get_module(self, module_name: str) -> FlockModule | None:
        """Get a module by name."""
        return self.agent.modules.get(module_name)

    def get_enabled_modules(self) -> list[FlockModule]:
        """Get a list of currently enabled modules attached to this agent."""
        return [m for m in self.agent.modules.values() if m.config.enabled]

    def add_component(
        self,
        config_instance: FlockModuleConfig
        | FlockRouterConfig
        | FlockEvaluatorConfig,
        component_name: str | None = None,
    ) -> "FlockAgent":
        """Adds or replaces a component (Evaluator, Router, Module) based on its configuration object.

        Args:
            config_instance: An instance of a config class inheriting from
                             FlockModuleConfig, FlockRouterConfig, or FlockEvaluatorConfig.
            component_name: Explicit name for the component (required for Modules if not in config).

        Returns:
            self.agent for potential chaining.
        """
        from flock.core.flock_registry import get_registry

        config_type = type(config_instance)
        registry = get_registry()  # Get registry instance
        logger.debug(
            f"Attempting to add component via config: {config_type.__name__}"
        )

        # --- 1. Find Component Class using Registry Map ---
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

        # --- 2. Determine Assignment Target and Name ---
        instance_name = component_name
        attribute_name: str = ""

        if issubclass(ComponentClass, FlockEvaluator):
            attribute_name = "evaluator"
            if not instance_name:
                instance_name = getattr(
                    config_instance, "name", component_class_name.lower()
                )

        elif issubclass(ComponentClass, FlockRouter):
            attribute_name = "handoff_router"
            if not instance_name:
                instance_name = getattr(
                    config_instance, "name", component_class_name.lower()
                )

        elif issubclass(ComponentClass, FlockModule):
            attribute_name = "modules"
            if not instance_name:
                instance_name = getattr(
                    config_instance, "name", component_class_name.lower()
                )
            if not instance_name:
                raise ValueError(
                    "Module name must be provided either in config or as component_name argument."
                )
            # Ensure config has name if module expects it
            if hasattr(config_instance, "name") and not getattr(
                config_instance, "name", None
            ):
                setattr(config_instance, "name", instance_name)

        else:  # Should be caught by registry map logic ideally
            raise TypeError(
                f"Class '{component_class_name}' mapped from config is not a valid Flock component."
            )

        # --- 3. Instantiate the Component ---
        try:
            init_args = {"config": config_instance, "name": instance_name}
            component_instance = ComponentClass(**init_args)
        except Exception as e:
            logger.error(
                f"Failed to instantiate {ComponentClass.__name__} with config {config_type.__name__}: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Component instantiation failed: {e}") from e

        # --- 4. Assign to the Agent ---
        if attribute_name == "modules":
            if not isinstance(self.agent.modules, dict):
                self.agent.modules = {}
            self.agent.modules[instance_name] = component_instance
            logger.info(
                f"Added/Updated module '{instance_name}' (type: {ComponentClass.__name__}) to agent '{self.agent.name}'"
            )
        else:
            setattr(self.agent, attribute_name, component_instance)
            logger.info(
                f"Set {attribute_name} to {ComponentClass.__name__} (instance name: '{instance_name}') for agent '{self.agent.name}'"
            )

        return self.agent

    def set_model(self, model: str):
        """Set the model for the agent and its evaluator."""
        self.agent.model = model
        if self.agent.evaluator and hasattr(self.agent.evaluator, "config"):
            self.agent.evaluator.config.model = model
            logger.info(
                f"Set model to '{model}' for agent '{self.agent.name}' and its evaluator."
            )
        elif self.agent.evaluator:
            logger.warning(
                f"Evaluator for agent '{self.agent.name}' does not have a standard config to set model."
            )
        else:
            logger.warning(
                f"Agent '{self.agent.name}' has no evaluator to set model for."
            )
