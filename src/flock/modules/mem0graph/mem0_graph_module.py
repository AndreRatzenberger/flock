from typing import Any

from pydantic import Field

from flock.core.context.context import FlockContext
from flock.core.flock_agent import FlockAgent
from flock.core.flock_module import FlockModule, FlockModuleConfig
from flock.core.flock_registry import flock_component
from flock.core.logging.logging import get_logger

logger = get_logger("module.mem0")


class Mem0GraphModuleConfig(FlockModuleConfig):
    qdrant_host: str = Field(default="localhost", description="Qdrant hostname")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    collection: str = Field(default="flock_memories", description="Vector collection")
    embedder_provider: str = Field(default="openai", description="'openai' or 'local'")
    embedder_model: str = Field(default="text-embedding-ada-002",
                                description="Model name for embeddings")
    # Optional: allow separate LLM for reflection/summarisation
    llm_provider: str | None = Field(default=None)
    llm_model: str | None = Field(default=None)
    top_k: int = Field(default=5, description="Number of memories to retrieve")



@flock_component(config_class=Mem0GraphModuleConfig)
class Mem0GraphModule(FlockModule):


    name: str = "mem0"
    config: Mem0GraphModuleConfig = Mem0GraphModuleConfig()
    session_id: str | None = None
    user_id: str | None = None

    def __init__(self, name, config: Mem0GraphModuleConfig) -> None:
        """Initialize Mem0 module."""
        super().__init__(name=name, config=config)
        logger.debug("Initializing Mem0 module")


    async def on_post_evaluate(
        self,
        agent: FlockAgent,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        return result

    async def on_pre_evaluate(
        self,
        agent: FlockAgent,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
    ) -> dict[str, Any]:




        return inputs
