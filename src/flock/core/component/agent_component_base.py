# src/flock/core/component/agent_component_base.py
"""Base classes for the unified Flock component system."""

from abc import ABC
from typing import Any, TypeVar

from pydantic import BaseModel, Field, create_model

from flock.core.context.context import FlockContext
from flock.core.flock_router import HandOffRequest

T = TypeVar("T", bound="AgentComponentConfig")


class AgentComponentConfig(BaseModel):
    """Base configuration class for all Flock agent components.
    
    This unified config class replaces FlockModuleConfig, FlockEvaluatorConfig, 
    and FlockRouterConfig, providing common functionality for all component types.
    """
    
    enabled: bool = Field(
        default=True, 
        description="Whether this component is currently enabled"
    )
    
    model: str | None = Field(
        default=None, 
        description="Model to use for this component (if applicable)"
    )
    
    @classmethod
    def with_fields(cls: type[T], **field_definitions) -> type[T]:
        """Create a new config class with additional fields.
        
        This allows dynamic config creation for components with custom configuration needs.
        
        Example:
            CustomConfig = AgentComponentConfig.with_fields(
                temperature=Field(default=0.7, description="LLM temperature"),
                max_tokens=Field(default=1000, description="Max tokens to generate")
            )
        """
        return create_model(
            f"Dynamic{cls.__name__}", 
            __base__=cls, 
            **field_definitions
        )


class AgentComponent(BaseModel, ABC):
    """Base class for all Flock agent components.
    
    This unified base class replaces the separate FlockModule, FlockEvaluator, 
    and FlockRouter base classes. All agent extensions now inherit from this 
    single base class and use the unified lifecycle hooks.
    
    Components can specialize by:
    - EvaluationComponentBase: Implements evaluate_core() for agent intelligence
    - RoutingModuleBase: Implements determine_next_step() for workflow routing  
    - UtilityModuleBase: Uses standard lifecycle hooks for cross-cutting concerns
    """
    
    name: str = Field(
        ..., 
        description="Unique identifier for this component"
    )
    
    config: AgentComponentConfig = Field(
        default_factory=AgentComponentConfig,
        description="Component configuration"
    )
    
    def __init__(self, **data):
        super().__init__(**data)
    
    # --- Standard Lifecycle Hooks ---
    # These are called for ALL components during agent execution
    
    async def on_initialize(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
    ) -> None:
        """Called when the agent starts running.
        
        Use this for component initialization, resource setup, etc.
        """
        pass

    async def on_pre_evaluate(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
    ) -> dict[str, Any]:
        """Called before agent evaluation, can modify inputs.
        
        Args:
            agent: The agent being executed
            inputs: Current input data
            context: Execution context
            
        Returns:
            Modified input data (or original if no changes)
        """
        return inputs

    async def on_post_evaluate(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Called after agent evaluation, can modify results.
        
        Args:
            agent: The agent that was executed  
            inputs: Original input data
            context: Execution context
            result: Evaluation result
            
        Returns:
            Modified result data (or original if no changes)
        """
        return result

    async def on_terminate(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Called when the agent finishes running.
        
        Use this for cleanup, final result processing, etc.
        """
        return result

    async def on_error(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
        error: Exception | None = None,
    ) -> None:
        """Called when an error occurs during agent execution.
        
        Use this for error handling, logging, recovery, etc.
        """
        pass

    # --- Specialized Hooks ---
    # These are overridden by specialized component types
    
    async def evaluate_core(
        self,
        agent: Any,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
        tools: list[Any] | None = None,
        mcp_tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Core evaluation logic - override in EvaluationComponentBase.
        
        This is where the main "intelligence" of the agent happens.
        Only one component per agent should implement this meaningfully.
        
        Args:
            agent: The agent being executed
            inputs: Input data for evaluation
            context: Execution context  
            tools: Available tools for the agent
            mcp_tools: Available MCP tools
            
        Returns:
            Evaluation result
        """
        # Default implementation is pass-through
        return inputs

    async def determine_next_step(
        self,
        agent: Any,
        result: dict[str, Any],
        context: FlockContext | None = None,
    ) -> HandOffRequest | None:
        """Determine the next step in the workflow - override in RoutingModuleBase.
        
        This is where routing decisions are made. The result is stored in
        agent.next_handoff for the orchestrator to use.
        
        Args:
            agent: The agent that just completed
            result: Result from the agent's evaluation
            context: Execution context
            
        Returns:
            HandOffRequest for next step, or None if no routing needed
        """
        # Default implementation provides no routing
        return None

    # --- MCP Server Lifecycle Hooks ---
    # For components that interact directly with MCP servers
    
    async def on_pre_server_init(self, server: Any) -> None:
        """Called before a server initializes."""
        pass

    async def on_post_server_init(self, server: Any) -> None:
        """Called after a server initializes."""
        pass

    async def on_pre_server_terminate(self, server: Any) -> None:
        """Called before a server terminates."""
        pass

    async def on_post_server_terminate(self, server: Any) -> None:
        """Called after a server terminates."""
        pass
