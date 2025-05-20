from typing import Any

from mem0 import AsyncMemory, AsyncMemoryClient
from pydantic import Field

from flock.core.context.context import FlockContext
from flock.core.flock_agent import FlockAgent
from flock.core.flock_module import FlockModule, FlockModuleConfig
from flock.core.flock_registry import flock_component
from flock.core.logging.logging import get_logger

logger = get_logger("module.mem0")


config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "flock_memory",
            "path": ".flock/memory",
        }
    }
}


class Mem0ModuleConfig(FlockModuleConfig):
    top_k: int = Field(default=10, description="Number of memories to retrieve")
    user_id: str = Field(default="flock", description="User ID the memories will be associated with")
    memory_input_key: str | None = Field(default=None, description="Input key to use for memory, if none the description of the agent will be used")
    api_key: str | None = Field(default=None, description="API key for mem0 Platform")
    config: dict[str, Any] = Field(default=config, description="Configuration for mem0")


@flock_component(config_class=Mem0ModuleConfig)
class Mem0Module(FlockModule):

    name: str = "mem0"
    config: Mem0ModuleConfig = Mem0ModuleConfig()


    def __init__(self, name, config: Mem0ModuleConfig) -> None:
        global memory
        """Initialize Mem0 module."""
        super().__init__(name=name, config=config)
        logger.debug("Initializing Mem0 module")



    def dict_to_str_repr(self,d: dict) -> str:
        return repr(d)


    async def on_post_evaluate(
        self,
        agent: FlockAgent,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.config.api_key:
            memory = AsyncMemoryClient(api_key=self.config.api_key)
        else:
            memory = await AsyncMemory.from_config(config_dict=self.config.config)


        to_remember = " - ".join(str(value) for value in result.values())

        messages = [ {"role": "user", "content": "Please remember the following information: " + to_remember}]

        await memory.add(messages, user_id=self.config.user_id)


        return result

    async def on_pre_evaluate(
        self,
        agent: FlockAgent,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
    ) -> dict[str, Any]:
        if self.config.api_key:
            memory = AsyncMemoryClient(api_key=self.config.api_key)
        else:
            memory = await AsyncMemory.from_config(config_dict=self.config.config)
        message = self.dict_to_str_repr(inputs)


        relevant_memories = await memory.search(query=message, user_id=self.config.user_id, limit=self.config.top_k)
        logger.info(f"Relevant memories: {relevant_memories}")
        if relevant_memories:
            if "results" in relevant_memories:
                memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
            else:
                memories_str = "\n".join(f"- {entry}" for entry in relevant_memories)

            if memories_str:
                if self.config.memory_input_key:
                    inputs[self.config.memory_input_key] = memories_str
                else:
                    agent.description = agent.description + "\n\n Memories:" + memories_str


        return inputs
