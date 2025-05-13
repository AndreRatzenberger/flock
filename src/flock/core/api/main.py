# src/flock/core/api/main.py
"""This module defines the FlockAPI class, which is now primarily responsible for
managing and adding user-defined custom API endpoints to a main FastAPI application.
It no longer handles core API endpoints or server startup.
"""

import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from fastapi import Body, Depends, FastAPI, Request as FastAPIRequest
from pydantic import BaseModel

from flock.core.logging.logging import get_logger

from .custom_endpoint import (
    FlockEndpoint,  # Assuming custom_endpoint.py is in the same directory
)

if TYPE_CHECKING:
    from flock.core.flock import Flock  # For type hinting

logger = get_logger("core.api.main") # Changed logger name for clarity


class FlockAPI:
    """Manages the addition of custom API endpoints related to a Flock instance
    to an existing FastAPI application.
    """

    def __init__(
        self,
        flock: "Flock",
        custom_endpoints: Sequence[FlockEndpoint] | dict[tuple[str, list[str] | None], Callable[..., Any]] | None = None,
    ):
        self.flock = flock

        # Normalize custom_endpoints into a list[FlockEndpoint]
        self.custom_endpoints: list[FlockEndpoint] = []
        if custom_endpoints:
            if isinstance(custom_endpoints, dict):
                # This path is for the older dict-based custom_endpoints format
                for (path, methods), cb in custom_endpoints.items():
                    self.custom_endpoints.append(
                        FlockEndpoint(path=path, methods=list(methods) if methods else ["GET"], callback=cb)
                    )
                    logger.debug(f"Converted dict custom endpoint: {path} {methods or ['GET']}")
            elif isinstance(custom_endpoints, Sequence): # Check if it's a sequence
                for ep in custom_endpoints:
                    if isinstance(ep, FlockEndpoint):
                        self.custom_endpoints.append(ep)
                    else:
                        logger.warning(f"Skipping non-FlockEndpoint item in custom_endpoints sequence: {type(ep)}")
            else:
                logger.warning(f"Unsupported type for custom_endpoints: {type(custom_endpoints)}")


        logger.info(f"FlockAPI helper initialized for Flock: '{self.flock.name}'. Prepared {len(self.custom_endpoints)} custom endpoints for potential registration.")


    def add_custom_routes_to_app(self, app: FastAPI):
        """Adds the custom endpoints (prepared during __init__) to the provided FastAPI app.
        This method is intended to be called by the main application's setup/lifespan logic.
        """
        if not self.custom_endpoints:
            logger.debug("No custom endpoints to add to the FastAPI app.")
            return

        logger.info(f"Adding {len(self.custom_endpoints)} custom endpoints to the FastAPI app instance.")

        for ep in self.custom_endpoints:
            # Factory to create the actual route handler with correct DI for body/query
            def _create_handler_factory(
                callback: Callable[..., Any],
                req_model: type[BaseModel] | None,
                query_model: type[BaseModel] | None
            ):
                # This inner async function will be the actual route handler
                async def _invoke_callback_with_di(
                    fastapi_request_di: FastAPIRequest, # Injected by FastAPI, renamed to avoid clash
                    # Using Depends() for query_model and Body() for req_model allows FastAPI
                    # to handle validation and parsing automatically.
                    # Type ignore as Pydantic models are assigned dynamically to these type hints.
                    body: req_model = Body(None) if req_model else None, # type: ignore
                    query: query_model = Depends(query_model) if query_model else None # type: ignore
                ):
                    # Arguments to potentially pass to the user's callback
                    payload_to_callback: dict[str, Any] = {"flock": self.flock} # Always provide Flock instance

                    # Add path parameters from the request
                    payload_to_callback.update(fastapi_request_di.path_params)

                    # Add query parameters
                    if query is not None: # query_model was defined and parsed by FastAPI
                        payload_to_callback["query"] = query
                    else:
                        payload_to_callback["query"] = None
                    # elif fastapi_request_di.query_params: # No query_model defined, use raw query_params
                    #     payload_to_callback["query"] = dict(fastapi_request_di.query_params)
                    # If query_model was defined but 'query' is None (e.g., optional fields not provided),
                    # 'query' will be an instance of query_model with default values or None for optional fields.

                    # Add body
                    if body is not None: # req_model was defined and parsed by FastAPI
                        payload_to_callback["body"] = body
                    elif req_model is None and fastapi_request_di.method in {"POST", "PUT", "PATCH"}:
                        # No req_model, try to get raw JSON or body bytes
                        try:
                            payload_to_callback["body"] = await fastapi_request_di.json()
                        except Exception:
                            payload_to_callback["body"] = await fastapi_request_di.body()
                    # else: body is None because no req_model or not a relevant HTTP method

                    # Filter kwargs to match the user callback's signature
                    sig = inspect.signature(callback)
                    filtered_kwargs = {
                        k: v for k, v in payload_to_callback.items() if k in sig.parameters
                    }

                    if inspect.iscoroutinefunction(callback):
                        return await callback(**filtered_kwargs)
                    else:
                        # For sync callbacks, FastAPI runs them in a threadpool
                        return callback(**filtered_kwargs)

                return _invoke_callback_with_di

            # Create the specific handler for this endpoint
            route_handler = _create_handler_factory(
                ep.callback, ep.request_model, ep.query_model
            )

            # Add the route to the main FastAPI app instance passed in
            app.add_api_route(
                ep.path,
                route_handler,
                methods=ep.methods or ["GET"],
                name=ep.name or f"custom:{ep.path.replace('/', '_').lstrip('_')}", # Ensure name is valid
                include_in_schema=ep.include_in_schema,
                response_model=ep.response_model,
                summary=ep.summary,
                description=ep.description,
                dependencies=ep.dependencies, # List of FastAPI Depends
            )
            logger.debug(f"Added custom route to app: {ep.methods} {ep.path}")

    # Core execution helper methods (_run_agent, _run_flock, _run_batch, _type_convert_inputs)
    # have been removed from this class. They are now either:
    # 1. Part of the `Flock` class methods (e.g., `Flock.run_async`, `Flock.run_batch_async`).
    # 2. Implemented as inline helper tasks within `src/flock/core/api/endpoints.py`
    #    which directly call the `Flock` instance's methods.
    # Type conversion for inputs from web forms (`_type_convert_inputs`) would typically
    # be handled within the API endpoint in `endpoints.py` before calling `Flock.run_async`.
