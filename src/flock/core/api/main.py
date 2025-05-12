# src/flock/core/api/main.py
"""Main Flock API server class and setup."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# Flock core imports
from flock.core.api.models import FlockBatchRequest
from flock.core.logging.logging import get_logger

if TYPE_CHECKING:
    # These imports are only for type hints
    from flock.core.flock import Flock

logger = get_logger("api.main")

from .endpoints import create_api_router

# Import components from the api package
from .run_store import RunStore

# Conditionally import for the new UI integration
NEW_UI_SERVICE_AVAILABLE = False
WEBAPP_FASTAPI_APP = None
try:
    from flock.webapp.app.main import (
        app as webapp_fastapi_app,  # Import the FastAPI app instance
    )
    from flock.webapp.app.services.flock_service import (
        set_current_flock_instance_programmatically,
    )

    WEBAPP_FASTAPI_APP = webapp_fastapi_app
    NEW_UI_SERVICE_AVAILABLE = True
except ImportError:
    logger.warning(
        "New webapp components (flock.webapp.app.main:app or flock.webapp.app.services.flock_service) not found. "
        "UI mode will fall back to old FastHTML UI if available."
    )
    # Fallback: Import old UI components if new one isn't available and create_ui is True
    try:
        from .ui.routes import FASTHTML_AVAILABLE, create_ui_app

        if FASTHTML_AVAILABLE:  # Only import utils if fasthtml is there
            from .ui.utils import format_result_to_html, parse_input_spec
        else:
            # Define placeholders if fasthtml itself is not available
            def parse_input_spec(*args, **kwargs):
                return []

            def format_result_to_html(*args, **kwargs):
                return ""

            FASTHTML_AVAILABLE = False  # Ensure it's false if import failed

    except ImportError:
        FASTHTML_AVAILABLE = False  # Ensure it's defined as false

        # Define placeholders if utils can't be imported
        def parse_input_spec(*args, **kwargs):
            return []

        def format_result_to_html(*args, **kwargs):
            return ""

from flock.core.api.custom_endpoint import FlockEndpoint


class FlockAPI:
    """Coordinates the Flock API server, including endpoints and UI.

    A user can provide custom FastAPI-style routes via the ``custom_endpoints`` dict.
    Each key is a tuple of ``(<path:str>, <methods:list[str] | None>)`` and the
    value is a callback ``Callable``.  ``methods`` can be ``None`` or an empty
    list to default to ``["GET"]``.  The callback can be synchronous or
    ``async``.  At execution time we provide the following keyword arguments and
    filter them to the callback's signature:

    • ``body``   – request json/plain payload (for POST/PUT/PATCH)
    • ``query``  – dict of query parameters
    • ``flock``  – current :class:`Flock` instance
    • any path parameters extracted from the route pattern
    """

    def __init__(
        self,
        flock: "Flock",
        custom_endpoints: Sequence[FlockEndpoint] | dict[tuple[str, list[str] | None], Callable[..., Any]] | None = None,
    ):
        self.flock = flock
        # Normalize into list[FlockEndpoint]
        self.custom_endpoints: list[FlockEndpoint] = []
        if custom_endpoints:
            merged: list[FlockEndpoint] = []
            if isinstance(custom_endpoints, dict):
                for (path, methods), cb in custom_endpoints.items():
                    merged.append(
                        FlockEndpoint(path=path, methods=list(methods) if methods else ["GET"], callback=cb)
                    )
            else:
                merged.extend(list(custom_endpoints))

            pending_endpoints = merged
        else:
            pending_endpoints = []

        # FastAPI app instance
        self.app = FastAPI(title="Flock API")

        # Store run information
        self.run_store = RunStore()

        # Register any pending custom endpoints collected before app creation
        if pending_endpoints:
            self.custom_endpoints.extend(pending_endpoints)

        self._setup_routes()

    def _setup_routes(self):
        """Includes API routers."""
        # Create and include the API router, passing self
        api_router = create_api_router(self)
        self.app.include_router(api_router)

        # Root redirect (if UI is enabled later) will be added in start()

        # --- Register user-supplied custom endpoints ---------------------
        if self.custom_endpoints:
            import inspect

            from fastapi import Body, Depends, Request

            # Register any endpoints collected during __init__ (self.custom_endpoints)
            if self.custom_endpoints:
                def _create_handler_factory(callback: Callable[..., Any], req_model: type[BaseModel] | None, query_model: type[BaseModel] | None):
                    async def _invoke(request: Request, body, query):
                        payload: dict[str, Any] = {"flock": self.flock}
                        if request:
                            payload.update(request.path_params)
                            if query is None:
                                payload["query"] = dict(request.query_params)
                            else:
                                payload["query"] = query
                        else:
                            payload["query"] = query or {}
                        if body is not None:
                            payload["body"] = body
                        elif request and request.method in {"POST", "PUT", "PATCH"} and req_model is None:
                            try:
                                payload["body"] = await request.json()
                            except Exception:
                                payload["body"] = await request.body()

                        sig = inspect.signature(callback)
                        filtered_kwargs = {k: v for k, v in payload.items() if k in sig.parameters}
                        if inspect.iscoroutinefunction(callback):
                            return await callback(**filtered_kwargs)
                        return callback(**filtered_kwargs)

                    # Dynamically build wrapper with appropriate signature so FastAPI can document it
                    params: list[str] = []
                    if req_model is not None:
                        params.append("body")
                    if query_model is not None:
                        params.append("query")

                    # Build wrapper function based on which params are present
                    if req_model and query_model:
                        async def _route_handler(
                            request: Request,
                            body: req_model = Body(...),  # type: ignore[arg-type,valid-type]
                            query: query_model = Depends(),  # type: ignore[arg-type,valid-type]
                        ):
                            return await _invoke(request, body, query)

                    elif req_model and not query_model:
                        async def _route_handler(
                            request: Request,
                            body: req_model = Body(...),  # type: ignore[arg-type,valid-type]
                        ):
                            return await _invoke(request, body, None)

                    elif query_model and not req_model:
                        async def _route_handler(
                            request: Request,
                            query: query_model = Depends(),  # type: ignore[arg-type,valid-type]
                        ):
                            return await _invoke(request, None, query)

                    else:
                        async def _route_handler(request: Request):
                            return await _invoke(request, None, None)

                    return _route_handler

                for ep in self.custom_endpoints:
                    self.app.add_api_route(
                        ep.path,
                        _create_handler_factory(ep.callback, ep.request_model, ep.query_model),
                        methods=ep.methods or ["GET"],
                        name=ep.name or f"custom:{ep.path}",
                        include_in_schema=ep.include_in_schema,
                        response_model=ep.response_model,
                        summary=ep.summary,
                        description=ep.description,
                        dependencies=ep.dependencies,
                    )

    # --- Core Execution Helper Methods ---
    # These remain here as they need access to self.flock and self.run_store

    async def _run_agent(
        self, run_id: str, agent_name: str, inputs: dict[str, Any]
    ):
        """Executes a single agent run (internal helper)."""
        try:
            if agent_name not in self.flock.agents:
                raise ValueError(f"Agent '{agent_name}' not found")
            agent = self.flock.agents[agent_name]
            # Type conversion (remains important)
            typed_inputs = self._type_convert_inputs(agent_name, inputs)

            logger.debug(
                f"Executing single agent '{agent_name}' (run_id: {run_id})",
                inputs=typed_inputs,
            )
            result = await agent.run_async(typed_inputs)
            logger.info(
                f"Single agent '{agent_name}' completed (run_id: {run_id})"
            )

            # Use RunStore to update
            self.run_store.update_run_result(run_id, result)

        except Exception as e:
            logger.error(
                f"Error in single agent run {run_id} ('{agent_name}'): {e!s}",
                exc_info=True,
            )
            # Update store status
            self.run_store.update_run_status(run_id, "failed", str(e))
            raise  # Re-raise for the endpoint handler

    async def _run_flock(
        self, run_id: str, agent_name: str, inputs: dict[str, Any]
    ):
        """Executes a flock workflow run (internal helper)."""
        try:
            if agent_name not in self.flock.agents:
                raise ValueError(f"Starting agent '{agent_name}' not found")

            # Type conversion
            typed_inputs = self._type_convert_inputs(agent_name, inputs)

            logger.debug(
                f"Executing flock workflow starting with '{agent_name}' (run_id: {run_id})",
                inputs=typed_inputs,
            )
            result = await self.flock.run_async(
                start_agent=agent_name, input=typed_inputs
            )
            # Result is potentially a Box object

            # Use RunStore to update
            self.run_store.update_run_result(run_id, result)

            # Log using the local result variable
            final_agent_name = (
                result.get("agent_name", "N/A") if result is not None else "N/A"
            )
            logger.info(
                f"Flock workflow completed (run_id: {run_id})",
                final_agent=final_agent_name,
            )

        except Exception as e:
            logger.error(
                f"Error in flock run {run_id} (started with '{agent_name}'): {e!s}",
                exc_info=True,
            )
            # Update store status
            self.run_store.update_run_status(run_id, "failed", str(e))
            raise  # Re-raise for the endpoint handler

    async def _run_batch(self, batch_id: str, request: "FlockBatchRequest"):
        """Executes a batch of runs (internal helper)."""
        try:
            if request.agent_name not in self.flock.agents:
                raise ValueError(f"Agent '{request.agent_name}' not found")

            logger.debug(
                f"Executing batch run starting with '{request.agent_name}' (batch_id: {batch_id})",
                batch_size=len(request.batch_inputs)
                if isinstance(request.batch_inputs, list)
                else "CSV",
            )

            # Import the thread pool executor here to avoid circular imports
            import asyncio
            import threading
            from concurrent.futures import ThreadPoolExecutor

            # Define a synchronous function to run the batch processing
            def run_batch_sync():
                # Use a new event loop for the batch processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Set the total number of batch items if possible
                    batch_size = (
                        len(request.batch_inputs)
                        if isinstance(request.batch_inputs, list)
                        else 0
                    )
                    if batch_size > 0:
                        # Directly call the store method - no need for asyncio here
                        # since we're already in a separate thread
                        self.run_store.set_batch_total_items(
                            batch_id, batch_size
                        )

                    # Custom progress tracking wrapper
                    class ProgressTracker:
                        def __init__(self, store, batch_id, total_size):
                            self.store = store
                            self.batch_id = batch_id
                            self.current_count = 0
                            self.total_size = total_size
                            self._lock = threading.Lock()
                            self.partial_results = []

                        def increment(self, result=None):
                            with self._lock:
                                self.current_count += 1
                                if result is not None:
                                    # Store partial result
                                    self.partial_results.append(result)

                                # Directly call the store method - no need for asyncio here
                                # since we're already in a separate thread
                                try:
                                    self.store.update_batch_progress(
                                        self.batch_id,
                                        self.current_count,
                                        self.partial_results,
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error updating progress: {e}"
                                    )
                                return self.current_count

                    # Create a progress tracker
                    progress_tracker = ProgressTracker(
                        self.run_store, batch_id, batch_size
                    )

                    # Define a custom worker that reports progress
                    async def progress_aware_worker(index, item_inputs):
                        try:
                            result = await self.flock.run_async(
                                start_agent=request.agent_name,
                                input=item_inputs,
                                box_result=request.box_results,
                            )
                            # Report progress after each item
                            progress_tracker.increment(result)
                            return result
                        except Exception as e:
                            logger.error(
                                f"Error processing batch item {index}: {e}"
                            )
                            progress_tracker.increment(
                                e if request.return_errors else None
                            )
                            if request.return_errors:
                                return e
                            return None

                    # Process the batch items with progress tracking
                    batch_inputs = request.batch_inputs
                    if isinstance(batch_inputs, list):
                        # Process list of inputs with progress tracking
                        tasks = []
                        for i, item_inputs in enumerate(batch_inputs):
                            # Combine with static inputs if provided
                            full_inputs = {
                                **(request.static_inputs or {}),
                                **item_inputs,
                            }
                            tasks.append(progress_aware_worker(i, full_inputs))

                        # Run all tasks
                        if request.parallel and request.max_workers > 1:
                            # Run in parallel with semaphore for max_workers
                            semaphore = asyncio.Semaphore(request.max_workers)

                            async def bounded_worker(i, inputs):
                                async with semaphore:
                                    return await progress_aware_worker(
                                        i, inputs
                                    )

                            bounded_tasks = []
                            for i, item_inputs in enumerate(batch_inputs):
                                full_inputs = {
                                    **(request.static_inputs or {}),
                                    **item_inputs,
                                }
                                bounded_tasks.append(
                                    bounded_worker(i, full_inputs)
                                )

                            results = loop.run_until_complete(
                                asyncio.gather(*bounded_tasks)
                            )
                        else:
                            # Run sequentially
                            results = []
                            for i, item_inputs in enumerate(batch_inputs):
                                full_inputs = {
                                    **(request.static_inputs or {}),
                                    **item_inputs,
                                }
                                result = loop.run_until_complete(
                                    progress_aware_worker(i, full_inputs)
                                )
                                results.append(result)
                    else:
                        # Let the original run_batch_async handle DataFrame or CSV
                        results = loop.run_until_complete(
                            self.flock.run_batch_async(
                                start_agent=request.agent_name,
                                batch_inputs=request.batch_inputs,
                                input_mapping=request.input_mapping,
                                static_inputs=request.static_inputs,
                                parallel=request.parallel,
                                max_workers=request.max_workers,
                                use_temporal=request.use_temporal,
                                box_results=request.box_results,
                                return_errors=request.return_errors,
                                silent_mode=request.silent_mode,
                                write_to_csv=request.write_to_csv,
                            )
                        )

                    # Update progress one last time with final count
                    if results:
                        progress_tracker.current_count = len(results)
                        self.run_store.update_batch_progress(
                            batch_id,
                            len(results),
                            results,  # Include all results as partial results
                        )

                    # Update store with results from this thread
                    self.run_store.update_batch_result(batch_id, results)

                    logger.info(
                        f"Batch run completed (batch_id: {batch_id})",
                        num_results=len(results),
                    )
                    return results
                except Exception as e:
                    logger.error(
                        f"Error in batch run {batch_id} (started with '{request.agent_name}'): {e!s}",
                        exc_info=True,
                    )
                    # Update store status
                    self.run_store.update_batch_status(
                        batch_id, "failed", str(e)
                    )
                    return None
                finally:
                    loop.close()

            # Run the batch processing in a thread pool
            try:
                loop = asyncio.get_running_loop()
                with ThreadPoolExecutor() as pool:
                    await loop.run_in_executor(pool, run_batch_sync)
            except Exception as e:
                error_msg = f"Error running batch in thread pool: {e!s}"
                logger.error(error_msg, exc_info=True)
                self.run_store.update_batch_status(
                    batch_id, "failed", error_msg
                )

        except Exception as e:
            logger.error(
                f"Error setting up batch run {batch_id} (started with '{request.agent_name}'): {e!s}",
                exc_info=True,
            )
            # Update store status
            self.run_store.update_batch_status(batch_id, "failed", str(e))
            raise  # Re-raise for the endpoint handler

    # --- UI Helper Methods (kept here as they are called by endpoints via self) ---

    def _parse_input_spec(self, input_spec: str) -> list[dict[str, str]]:
        """Parses an agent input string into a list of field definitions."""
        # Use the implementation moved to ui.utils
        return parse_input_spec(input_spec)

    def _format_result_to_html(self, data: Any) -> str:
        """Recursively formats a Python object into an HTML string."""
        # Use the implementation moved to ui.utils
        return format_result_to_html(data)

    def _type_convert_inputs(
        self, agent_name: str, inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Converts input values (esp. from forms) to expected Python types."""
        typed_inputs = {}
        agent_def = self.flock.agents.get(agent_name)
        if not agent_def or not agent_def.input:
            return inputs  # Return original if no spec

        parsed_fields = self._parse_input_spec(agent_def.input)
        field_types = {f["name"]: f["type"] for f in parsed_fields}

        for k, v in inputs.items():
            target_type = field_types.get(k)
            if target_type and target_type.startswith("bool"):
                typed_inputs[k] = (
                    str(v).lower() in ["true", "on", "1", "yes"]
                    if isinstance(v, str)
                    else bool(v)
                )
            elif target_type and target_type.startswith("int"):
                try:
                    typed_inputs[k] = int(v)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert input '{k}' value '{v}' to int for agent '{agent_name}'"
                    )
                    typed_inputs[k] = v
            elif target_type and target_type.startswith("float"):
                try:
                    typed_inputs[k] = float(v)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert input '{k}' value '{v}' to float for agent '{agent_name}'"
                    )
                    typed_inputs[k] = v
            # TODO: Add list/dict parsing (e.g., json.loads) if needed
            else:
                typed_inputs[k] = v  # Assume string or already correct type
        return typed_inputs

    # --- Server Start/Stop ---

    def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8344,
        server_name: str = "Flock API",
        create_ui: bool = False,
        #custom_endpoints: Sequence[FlockEndpoint] | dict[tuple[str, list[str] | None], Callable[..., Any]] | None = None,
    ):
        """Start the API server. If create_ui is True, it mounts the new webapp or the old FastHTML UI at the root."""
        if create_ui:
            if NEW_UI_SERVICE_AVAILABLE and WEBAPP_FASTAPI_APP:
                logger.info(
                    f"Preparing to mount new Scoped Web UI at root for Flock: {self.flock.name}"
                )
                try:
                    # Set the flock instance for the webapp
                    set_current_flock_instance_programmatically(
                        self.flock,
                        f"{self.flock.name.replace(' ', '_').lower()}_api_scoped.flock",
                    )
                    logger.info(
                        f"Flock '{self.flock.name}' set for the new web UI (now part of the main app)."
                    )

                    # Mount the new web UI app at the root of self.app
                    # The WEBAPP_FASTAPI_APP handles its own routes including '/', static files etc.
                    # It will need to be started with ui_mode=scoped, which should be handled by
                    # the client accessing /?ui_mode=scoped initially.
                    self.app.mount(
                        "/", WEBAPP_FASTAPI_APP, name="flock_ui_root"
                    )
                    logger.info(
                        f"New Web UI (scoped mode) mounted at root. Access at http://{host}:{port}/?ui_mode=scoped"
                    )
                    # No explicit root redirect needed from self.app to WEBAPP_FASTAPI_APP's root,
                    # as WEBAPP_FASTAPI_APP now *is* the handler for "/".
                    # The API's own routes (e.g. /api/...) will still be served by self.app if they don't conflict.

                    logger.info(
                        f"API server '{server_name}' (with integrated UI) starting on http://{host}:{port}"
                    )
                    logger.info(
                        f"Access the Scoped UI for '{self.flock.name}' at http://{host}:{port}/?ui_mode=scoped"
                    )

                except Exception as e:
                    logger.error(
                        f"Error setting up or mounting new scoped UI at root: {e}. "
                        "API will start, UI might be impacted.",
                        exc_info=True,
                    )
            elif FASTHTML_AVAILABLE:  # Fallback to old FastHTML UI
                logger.warning(
                    "New webapp not available or WEBAPP_FASTAPI_APP is None. Falling back to old FastHTML UI (mounted at /ui)."
                )
                try:
                    from .ui.routes import create_ui_app
                except ImportError:
                    logger.error(
                        "Failed to import create_ui_app for old UI. API running without UI."
                    )
                    FASTHTML_AVAILABLE = False

                if FASTHTML_AVAILABLE:
                    logger.info(
                        "Attempting to create and mount old FastHTML UI at /ui"
                    )  # Old UI stays at /ui
                    try:
                        fh_app = create_ui_app(
                            self,
                            api_host=host,
                            api_port=port,
                            server_name=server_name,
                        )
                        self.app.mount(
                            "/ui", fh_app, name="old_flock_ui"
                        )  # Old UI still at /ui
                        logger.info(
                            "Old FastHTML UI mounted successfully at /ui."
                        )

                        @self.app.get(
                            "/",
                            include_in_schema=False,
                            response_class=RedirectResponse,
                        )
                        async def root_redirect_to_old_ui():  # Redirect / to /ui/ for old UI
                            logger.debug("Redirecting / to /ui/ (old UI)")
                            return RedirectResponse(url="/ui/", status_code=303)

                        logger.info(
                            f"Old FastHTML UI available at http://{host}:{port}/ui/"
                        )
                    except Exception as e:
                        logger.error(
                            f"An error occurred setting up the old FastHTML UI: {e}. Running API only.",
                            exc_info=True,
                        )
            else:
                logger.error(
                    "No UI components available. API running without UI."
                )

        if not create_ui:
            logger.info(
                f"API server '{server_name}' starting on http://{host}:{port} (UI not requested)."
            )
        elif (
            not (NEW_UI_SERVICE_AVAILABLE and WEBAPP_FASTAPI_APP)
            and not FASTHTML_AVAILABLE
        ):
            logger.info(
                f"API server '{server_name}' starting on http://{host}:{port}. UI was requested but no components found."
            )

        uvicorn.run(self.app, host=host, port=port)

    async def stop(self):
        """Stop the API server."""
        logger.info("Stopping API server (cleanup if necessary)")
        pass


# --- End of file ---
