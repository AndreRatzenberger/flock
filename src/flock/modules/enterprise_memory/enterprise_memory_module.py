from __future__ import annotations

"""Enterprise-grade memory module for Flock.

This module persists:
• vector embeddings in a Chroma collection (or any collection that
  implements the same API)
• a concept graph in Neo4j/Memgraph (Cypher-compatible)

It follows the same life-cycle callbacks as the standard MemoryModule but
is designed for large-scale, concurrent deployments.
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Literal

from neo4j import AsyncGraphDatabase
from opentelemetry import trace
from pydantic import Field
from sentence_transformers import SentenceTransformer

from flock.core.context.context import FlockContext
from flock.core.flock_agent import FlockAgent
from flock.core.flock_module import FlockModule, FlockModuleConfig
from flock.core.flock_registry import flock_component
from flock.core.logging.logging import get_logger

logger = get_logger("enterprise_memory")
tracer = trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class EnterpriseMemoryModuleConfig(FlockModuleConfig):
    """Configuration for EnterpriseMemoryModule."""

    # ---------------------
    # Vector store settings
    # ---------------------

    vector_backend: Literal["chroma", "pinecone", "azure"] = Field(
        default="chroma",
        description="Which vector backend to use (chroma | pinecone | azure)",
    )

    # --- Chroma ---
    chroma_path: str | None = Field(
        default="./vector_store",
        description="Disk path for Chroma persistent storage (if running embedded).",
    )
    chroma_collection: str = Field(
        default="flock_memories", description="Chroma collection name"
    )
    chroma_host: str | None = Field(
        default=None,
        description="If provided, connect to a remote Chroma HTTP server at this host",
    )
    chroma_port: int = Field(default=8000, description="Remote Chroma HTTP port")

    # --- Pinecone ---
    pinecone_api_key: str | None = Field(default=None, description="Pinecone API key")
    pinecone_env: str | None = Field(default=None, description="Pinecone environment")
    pinecone_index: str | None = Field(default=None, description="Pinecone index name")

    # --- Azure Cognitive Search ---
    azure_search_endpoint: str | None = Field(default=None, description="Azure search endpoint (https://<service>.search.windows.net)")
    azure_search_key: str | None = Field(default=None, description="Azure search admin/key")
    azure_search_index_name: str | None = Field(default=None, description="Azure search index name")

    # Graph DB (Neo4j / Memgraph) settings
    cypher_uri: str = Field(
        default="bolt://localhost:7687", description="Bolt URI for the graph DB"
    )
    cypher_username: str = Field(default="neo4j", description="Username for DB")
    cypher_password: str = Field(default="password", description="Password for DB")

    similarity_threshold: float = Field(
        default=0.5, description="Cosine-similarity threshold for retrieval"
    )
    max_results: int = Field(default=10, description="Maximum retrieved memories")
    number_of_concepts_to_extract: int = Field(
        default=3, description="Number of concepts extracted per chunk"
    )
    save_interval: int = Field(
        default=10,
        description="Persist to disk after this many new chunks (0 disables auto-save)",
    )

    export_graph_image: bool = Field(
        default=False,
        description="If true, exports a PNG image of the concept graph each time it is updated.",
    )
    graph_image_dir: str = Field(
        default="./concept_graphs",
        description="Directory where graph images will be stored when export_graph_image is true.",
    )


# ---------------------------------------------------------------------------
# Storage Abstraction
# ---------------------------------------------------------------------------
class EnterpriseMemoryStore:
    """Persistence layer that wraps Chroma + Cypher graph."""

    def __init__(self, cfg: EnterpriseMemoryModuleConfig):
        self.cfg = cfg
        # Lazy initialise expensive resources
        self._embedding_model: SentenceTransformer | None = None
        self._vector_store = None  # could be chroma collection, pinecone index, azure search client
        self._driver = None  # Neo4j driver
        self._pending_writes: list[tuple[str, dict[str, Any]]] = []
        self._write_lock = asyncio.Lock()
        self._concept_cache: set[str] | None = None  # names of known concepts

    # ---------------------------------------------------------------------
    # Connections
    # ---------------------------------------------------------------------
    def _ensure_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.debug("Loading embedding model 'all-MiniLM-L6-v2'")
            with tracer.start_as_current_span("memory.load_embedding_model") as span:
                try:
                    self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                    span.set_attribute("model", "all-MiniLM-L6-v2")
                except Exception as e:
                    span.record_exception(e)
                    raise
        return self._embedding_model

    def _ensure_vector_store(self):
        if self._vector_store is not None:
            return self._vector_store

        backend = self.cfg.vector_backend

        if backend == "chroma":
            self._vector_store = self._init_chroma()
        elif backend == "pinecone":
            self._vector_store = self._init_pinecone()
        elif backend == "azure":
            self._vector_store = self._init_azure()
        else:
            raise ValueError(f"Unsupported vector backend: {backend}")

        return self._vector_store

    def _init_chroma(self):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            raise RuntimeError("chromadb package is required for Chroma backend") from e

        if self.cfg.chroma_host:
            client = chromadb.HttpClient(
                host=self.cfg.chroma_host, port=self.cfg.chroma_port
            )
        else:
            path = Path(self.cfg.chroma_path)
            path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(settings=Settings(path=str(path)))

        return client.get_or_create_collection(self.cfg.chroma_collection)

    def _init_pinecone(self):
        try:
            import pinecone
        except ImportError as e:
            raise RuntimeError("pinecone-client package required for Pinecone backend") from e

        if not (self.cfg.pinecone_api_key and self.cfg.pinecone_env and self.cfg.pinecone_index):
            raise ValueError("pinecone_api_key, pinecone_env and pinecone_index must be set for Pinecone backend")

        pinecone.init(api_key=self.cfg.pinecone_api_key, environment=self.cfg.pinecone_env)
        return pinecone.Index(self.cfg.pinecone_index)

    def _init_azure(self):
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
        except ImportError as e:
            raise RuntimeError("azure-search-documents package required for Azure backend") from e

        if not (self.cfg.azure_search_endpoint and self.cfg.azure_search_key and self.cfg.azure_search_index_name):
            raise ValueError("Azure search endpoint, key and index name must be provided for Azure backend")

        return SearchClient(
            endpoint=self.cfg.azure_search_endpoint,
            index_name=self.cfg.azure_search_index_name,
            credential=AzureKeyCredential(self.cfg.azure_search_key),
        )

    def _ensure_graph_driver(self):
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.cfg.cypher_uri,
                auth=(self.cfg.cypher_username, self.cfg.cypher_password),
                encrypted=False,
            )
        return self._driver

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def add_entry(
        self,
        content: str,
        concepts: set[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a chunk in both vector store and graph DB and return its id."""
        with tracer.start_as_current_span("memory.add_entry") as span:
            span.set_attribute("entry_id", str(uuid.uuid4()))

            # Embed
            embedding = self._ensure_embedding_model().encode(content).tolist()
            span.set_attribute("embedding_length", len(embedding))

            # Vector store write
            store = self._ensure_vector_store()
            backend = self.cfg.vector_backend
            span.set_attribute("vector_backend", backend)

            try:
                if backend == "chroma":
                    store.add(ids=[span.get_attribute("entry_id")], documents=[content], embeddings=[embedding], metadatas=[metadata or {}])
                elif backend == "pinecone":
                    store.upsert(vectors=[(span.get_attribute("entry_id"), embedding, metadata or {})])
                elif backend == "azure":
                    document = {
                        "id": span.get_attribute("entry_id"),
                        "content": content,
                        "embedding": embedding,
                        **(metadata or {}),
                    }
                    store.upload_documents(documents=[document])
                else:
                    raise ValueError(f"Unsupported vector backend: {backend}")
            except Exception as e:
                span.record_exception(e)
                raise

        # Schedule graph writes (batched)
        async with self._write_lock:
            self._pending_writes.append((span.get_attribute("entry_id"), {"concepts": concepts}))
            if self.cfg.save_interval and len(self._pending_writes) >= self.cfg.save_interval:
                await self._flush_pending_graph_writes()
        return span.get_attribute("entry_id")

    async def search(
        self, query_text: str, threshold: float, k: int
    ) -> list[dict[str, Any]]:
        """Vector similarity search followed by graph enrichment."""
        with tracer.start_as_current_span("memory.search") as span:
            span.set_attribute("vector_backend", self.cfg.vector_backend)
            embedding = (
                self._ensure_embedding_model().encode(query_text).tolist()
            )
            span.set_attribute("embedding_length", len(embedding))
            store = self._ensure_vector_store()
            backend = self.cfg.vector_backend
            results: list[dict[str, Any]] = []

            if backend == "chroma":
                res = store.query(
                    query_embeddings=[embedding],
                    n_results=k,
                    include=["documents", "metadatas", "distances", "ids"],
                )
                if res and res["ids"]:
                    for idx in range(len(res["ids"][0])):
                        distance = res["distances"][0][idx]
                        score = 1 - distance
                        if score < threshold:
                            continue
                        results.append(
                            {
                                "id": res["ids"][0][idx],
                                "content": res["documents"][0][idx],
                                "metadata": res["metadatas"][0][idx],
                                "score": score,
                            }
                        )

            elif backend == "pinecone":
                query_res = store.query(vector=embedding, top_k=k, include_values=False, include_metadata=True)
                for match in query_res.matches:
                    if match.score < threshold:
                        continue
                    results.append(
                        {
                            "id": match.id,
                            "content": None,  # pinecone doesn't store doc content directly
                            "metadata": match.metadata,
                            "score": match.score,
                        }
                    )

            elif backend == "azure":
                # Azure vector search – assuming index has 'embedding' vector field
                search_res = store.search(
                    search_text=None,
                    vector=embedding,
                    k=k,
                    vector_fields="embedding",
                )
                for doc in search_res:
                    score = doc["@search.score"]
                    if score < threshold:
                        continue
                    results.append(
                        {
                            "id": doc["id"],
                            "content": doc.get("content"),
                            "metadata": {k: v for k, v in doc.items() if k not in ("id", "content", "embedding", "@search.score")},
                            "score": score,
                        }
                    )

            else:
                raise ValueError(f"Unsupported backend: {backend}")

            span.set_attribute("results_count", len(results))
            return results

    # ------------------------------------------------------------------
    # Graph persistence helpers
    # ------------------------------------------------------------------
    async def _flush_pending_graph_writes(self):
        """Commit queued node/edge creations to the Cypher store."""
        if not self._pending_writes:
            return
        driver = self._ensure_graph_driver()
        async with driver.session() as session:
            tx_commands: list[str] = []
            params: dict[str, Any] = {}
            # Build Cypher in one transaction
            for idx, (entry_id, extra) in enumerate(self._pending_writes):
                concept_param = f"concepts_{idx}"
                tx_commands.append(
                    f"MERGE (e:Memory {{id: '{entry_id}'}}) "
                    f"SET e.created = datetime() "
                )
                if extra.get("concepts"):
                    tx_commands.append(
                        f"WITH e UNWIND ${concept_param} AS c "
                        "MERGE (co:Concept {name: c}) "
                        "MERGE (e)-[:MENTIONS]->(co)"
                    )
                    params[concept_param] = list(extra["concepts"])
            cypher = "\n".join(tx_commands)
            await session.run(cypher, params)
            # Export graph image if requested
            if self.cfg.export_graph_image:
                await self._export_graph_image(session)
        self._pending_writes.clear()

    async def _export_graph_image(self, session):
        """Generate and save a PNG of the concept graph."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx

            records = await session.run(
                "MATCH (c1:Concept)<-[:MENTIONS]-(:Memory)-[:MENTIONS]->(c2:Concept) "
                "RETURN DISTINCT c1.name AS source, c2.name AS target"
            )
            edges = [(r["source"], r["target"]) for r in await records.values("source", "target")]
            if not edges:
                return

            G = nx.Graph()
            G.add_edges_from(edges)

            pos = nx.spring_layout(G, k=0.4)
            plt.figure(figsize=(12, 9), dpi=100)
            nx.draw_networkx_nodes(G, pos, node_color="#8fa8d6", node_size=500, edgecolors="white")
            nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)
            nx.draw_networkx_labels(G, pos, font_size=8)
            plt.axis("off")

            img_dir = Path(self.cfg.graph_image_dir)
            img_dir.mkdir(parents=True, exist_ok=True)
            filename = img_dir / f"concept_graph_{uuid.uuid4().hex[:8]}.png"
            plt.savefig(filename, bbox_inches="tight", facecolor="white")
            plt.close()
            logger.info("Concept graph image exported to %s", filename)
        except Exception as e:
            logger.warning("Failed to export concept graph image: %s", e)

    async def close(self):
        if self._pending_writes:
            await self._flush_pending_graph_writes()
        if self._driver:
            await self._driver.close()


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
@flock_component(config_class=EnterpriseMemoryModuleConfig)
class EnterpriseMemoryModule(FlockModule):
    """Enterprise-ready memory module using real datastores."""

    name: str = "enterprise_memory"
    config: EnterpriseMemoryModuleConfig = Field(default_factory=EnterpriseMemoryModuleConfig)

    _store: EnterpriseMemoryStore | None = None

    # ----------------------------------------------------------
    # Life-cycle hooks
    # ----------------------------------------------------------
    async def on_initialize(
        self,
        agent: FlockAgent,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
    ) -> None:
        self._store = EnterpriseMemoryStore(self.config)
        logger.info("EnterpriseMemoryModule initialised", agent=agent.name)

    async def on_pre_evaluate(
        self,
        agent: FlockAgent,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
    ) -> dict[str, Any]:
        if not self._store:
            return inputs
        try:
            query_str = json.dumps(inputs)
            matches = await self._store.search(
                query_str,
                threshold=self.config.similarity_threshold,
                k=self.config.max_results,
            )
            if matches:
                inputs = {**inputs, "context": matches}
                # Advertise new input key to DSPy signature if needed
                if isinstance(agent.input, str) and "context:" not in agent.input:
                    agent.input += ", context: list | retrieved memories"
        except Exception as e:
            logger.warning("Enterprise memory retrieval failed: %s", e, agent=agent.name)
        return inputs

    async def on_post_evaluate(
        self,
        agent: FlockAgent,
        inputs: dict[str, Any],
        context: FlockContext | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self._store:
            return result
        try:
            full_text = json.dumps(inputs) + (json.dumps(result) if result else "")
            concepts = await self._extract_concepts(agent, full_text)
            if self._store:
                concepts = await self._store._deduplicate_concepts(concepts)
            await self._store.add_entry(full_text, concepts)
        except Exception as e:
            logger.warning("Enterprise memory store failed: %s", e, agent=agent.name)
        return result

    async def on_terminate(
        self,
        agent: FlockAgent,
        inputs: dict[str, Any],
        result: dict[str, Any],
        context: FlockContext | None = None,
    ) -> None:
        if self._store:
            await self._store.close()

    # ----------------------------------------------------------
    # Helpers (mostly copied from original module but simplified)
    # ----------------------------------------------------------
    async def _extract_concepts(self, agent: FlockAgent, text: str) -> set[str]:
        """Use the LLM to extract concept tokens."""
        concept_signature = agent.create_dspy_signature_class(
            f"{agent.name}_concept_extractor_enterprise",
            "Extract key concepts from text",
            "text: str | Input text -> concepts: list[str] | key concepts lower case",
        )
        agent._configure_language_model(agent.model, True, 0.0, 8192)
        predictor = agent._select_task(concept_signature, "Completion")
        res = predictor(text=text)
        return set(getattr(res, "concepts", []))

    # --------------------------------------------------------------
    # Concept helpers
    # --------------------------------------------------------------
    async def _ensure_concept_cache(self):
        if self._concept_cache is not None:
            return
        driver = self._ensure_graph_driver()
        async with driver.session() as session:
            records = await session.run("MATCH (c:Concept) RETURN c.name AS name")
            self._concept_cache = {r["name"] for r in await records.values("name")}

    async def _deduplicate_concepts(self, new_concepts: set[str]) -> set[str]:
        """Return a set of concept names that merges with existing ones to avoid duplicates.

        Strategy: case-insensitive equality first, then fuzzy match via difflib with cutoff 0.85.
        """
        await self._ensure_concept_cache()
        assert self._concept_cache is not None

        from difflib import get_close_matches

        unified: set[str] = set()
        for concept in new_concepts:
            # Exact (case-insensitive) match
            lower = concept.lower()
            exact = next((c for c in self._concept_cache if c.lower() == lower), None)
            if exact:
                unified.add(exact)
                continue

            # Fuzzy match (>=0.85 similarity)
            close = get_close_matches(concept, list(self._concept_cache), n=1, cutoff=0.85)
            if close:
                unified.add(close[0])
                continue

            # No match – treat as new
            unified.add(concept)
            self._concept_cache.add(concept)
        return unified
