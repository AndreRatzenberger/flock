# src/flock/core/api/main.py
"""This module defines the FlockAPI class, which is now primarily responsible for
managing and adding user-defined custom API endpoints to a main FastAPI application.
It no longer handles core API endpoints (like /run, /batch) or server startup.
"""

import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from fastapi import Body, Depends, FastAPI, Request as FastAPIRequest

from flock.core.logging.logging import get_logger

from .custom_endpoint import (
    FlockEndpoint,  # Assuming custom_endpoint.py is in the same directory
)

if TYPE_CHECKING:
    from flock.core.flock import Flock  # For type hinting

logger = get_logger("core.api.custom_setup") # More specific logger name


class FlockAPI:
    """A helper class to manage the addition of user-defined custom API endpoints
    to an existing FastAPI application, in the context of a Flock instance.
    """

    def __init__(
        self,
        flock_instance: "Flock", # The Flock instance this API helper is associated with
        custom_endpoints: Sequence[FlockEndpoint] | None = None,
    ):
        """Initializes the FlockAPI helper.

        Args:
            flock_instance: The active Flock instance that custom endpoints might need access to.
            custom_endpoints: A sequence of FlockEndpoint definitions provided by the user.
                              This can also be the older dict format for backward compatibility,
                              which will be normalized.
        """
        self.flock = flock_instance

        # Normalize custom_endpoints into a list[FlockEndpoint]
        self.processed_custom_endpoints: list[FlockEndpoint] = []
        if custom_endpoints:
            if isinstance(custom_endpoints, dict):
                # Handle older dict format if necessary, though ideally user passes Sequence[FlockEndpoint]
                logger.warning("Received custom_endpoints as dict, converting. Prefer Sequence[FlockEndpoint].")
                for (path, methods), cb in custom_endpoints.items():
                    # Basic conversion attempt, assuming simple structure
                    self.processed_custom_endpoints.append(
                        FlockEndpoint(path=path, methods=list(methods) if methods else ["GET"], callback=cb)
                    )
            elif isinstance(custom_endpoints, Sequence):
                for ep in custom_endpoints:
                    if isinstance(ep, FlockEndpoint):
                        self.processed_custom_endpoints.append(ep)
                    else:
                        logger.warning(f"Skipping non-FlockEndpoint item in custom_endpoints sequence: {type(ep)}")
            else:
                logger.warning(f"Unsupported type for custom_endpoints: {type(custom_endpoints)}")

        logger.info(
            f"FlockAPI helper initialized for Flock: '{self.flock.name}'. "
            f"Prepared {len(self.processed_custom_endpoints)} custom endpoints for potential registration."
        )

    # Inside src/flock/core/api/main.py - class FlockAPI

    def add_custom_routes_to_app(self, app: FastAPI):
        if not self.processed_custom_endpoints:
            logger.debug("No custom endpoints to add to the FastAPI app.")
            return

        logger.info(f"Adding {len(self.processed_custom_endpoints)} custom endpoints to the FastAPI app instance.")

        for ep in self.processed_custom_endpoints: # ep is a FlockEndpoint instance

            # --- Dynamically choose and define the route handler signature ---

            # --- Inner function that performs the actual work ---
            async def _execute_user_callback(
                fastapi_request_obj_di: FastAPIRequest,
                # These will be populated based on which outer handler is chosen
                body_param: Any = None,
                query_param: Any = None
            ):
                payload_for_user_callback: dict[str, Any] = {"flock": self.flock}
                payload_for_user_callback.update(fastapi_request_obj_di.path_params)

                if ep.query_model and query_param is not None:
                    payload_for_user_callback["query"] = query_param
                elif 'query' in inspect.signature(ep.callback).parameters and not ep.query_model:
                    if fastapi_request_obj_di.query_params:
                         payload_for_user_callback["query"] = dict(fastapi_request_obj_di.query_params)

                if ep.request_model and body_param is not None:
                    payload_for_user_callback["body"] = body_param
                elif 'body' in inspect.signature(ep.callback).parameters and \
                     not ep.request_model and \
                     fastapi_request_obj_di.method in {"POST", "PUT", "PATCH"}:
                    try: payload_for_user_callback["body"] = await fastapi_request_obj_di.json()
                    except Exception: payload_for_user_callback["body"] = await fastapi_request_obj_di.body()

                if 'request' in inspect.signature(ep.callback).parameters:
                    payload_for_user_callback['request'] = fastapi_request_obj_di

                user_callback_sig = inspect.signature(ep.callback)
                final_kwargs_for_user_callback = {
                    k: v for k, v in payload_for_user_callback.items() if k in user_callback_sig.parameters
                }

                if inspect.iscoroutinefunction(ep.callback):
                    return await ep.callback(**final_kwargs_for_user_callback)
                else:
                    return ep.callback(**final_kwargs_for_user_callback)
            # --- End of inner execution function ---


            # --- Define handler variants based on ep.request_model and ep.query_model ---
            # This is the key to making OpenAPI generation precise.
            if ep.request_model and ep.query_model:
                async def handler_with_body_and_query(
                    fastapi_request_obj_di: FastAPIRequest,
                    body: ep.request_model = Body(...), # type: ignore
                    query: ep.query_model = Depends(ep.query_model) # type: ignore
                ):
                    return await _execute_user_callback(fastapi_request_obj_di, body_param=body, query_param=query)
                selected_handler = handler_with_body_and_query
            elif ep.request_model and not ep.query_model:
                async def handler_with_body_only(
                    fastapi_request_obj_di: FastAPIRequest,
                    body: ep.request_model = Body(...) # type: ignore
                ):
                    return await _execute_user_callback(fastapi_request_obj_di, body_param=body, query_param=None)
                selected_handler = handler_with_body_only
            elif not ep.request_model and ep.query_model:
                async def handler_with_query_only(
                    fastapi_request_obj_di: FastAPIRequest,
                    query: ep.query_model = Depends(ep.query_model) # type: ignore
                ):
                    return await _execute_user_callback(fastapi_request_obj_di, body_param=None, query_param=query)
                selected_handler = handler_with_query_only
            else: # No request_model and no query_model
                async def handler_with_request_only(
                    fastapi_request_obj_di: FastAPIRequest
                ):
                    return await _execute_user_callback(fastapi_request_obj_di, body_param=None, query_param=None)
                selected_handler = handler_with_request_only

            # Set a more descriptive name for the chosen handler for debugging/tracing if needed
            selected_handler.__name__ = f"handler_for_{ep.path.replace('/', '_').lstrip('_')}"

            app.add_api_route(
                ep.path,
                selected_handler, # Use the specifically chosen handler
                methods=ep.methods or ["GET"],
                name=ep.name or f"custom:{ep.path.replace('/', '_').lstrip('_')}",
                include_in_schema=ep.include_in_schema,
                response_model=ep.response_model,
                summary=ep.summary,
                description=ep.description,
                dependencies=ep.dependencies,
            )
            logger.debug(f"Added custom route to app: {ep.methods} {ep.path} (Handler: {selected_handler.__name__}, Summary: {ep.summary})")
