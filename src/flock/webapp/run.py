# src/flock/webapp/run.py
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn

# Import core Flock components
if TYPE_CHECKING:
    from flock.core.api.custom_endpoint import FlockEndpoint
    from flock.core.flock import Flock

# --- Ensure src is in path for imports ---
current_file_path = Path(__file__).resolve()
flock_webapp_dir = current_file_path.parent
flock_dir = flock_webapp_dir.parent
src_dir = flock_dir.parent # Assuming `flock` is a package within `src`

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# --- Main Server Startup Function ---
def start_unified_server(
    flock_instance: "Flock",
    host: str,
    port: int,
    server_title: str,
    enable_ui_routes: bool, # Currently, UI routes are always part of the app. This could be used later.
    ui_theme: str | None = None,
    custom_endpoints: Sequence["FlockEndpoint"] | dict[tuple[str, list[str] | None], Callable[..., Any]] | None = None,
):
    """Starts the unified FastAPI server for Flock.
    - Initializes the web application (imported from webapp.app.main).
    - Sets the provided Flock instance and a RunStore for dependency injection
      and makes them available via app.state.
    - Configures the UI theme.
    - Stores custom API endpoints for registration during app lifespan startup.
    - Runs Uvicorn.
    """
    print(f"Attempting to start unified server for Flock '{flock_instance.name}' on http://{host}:{port}")
    print(f"UI Routes Enabled: {enable_ui_routes}, Theme: {ui_theme or 'Default'}")

    try:
        # Import necessary webapp components HERE, after path setup.
        from flock.core.api.run_store import RunStore
        from flock.core.logging.logging import get_logger  # For logging
        from flock.webapp.app.config import (  # For logging resolved theme
            get_current_theme_name,
            set_current_theme_name,
        )
        from flock.webapp.app.dependencies import (
            add_pending_custom_endpoints,
            set_global_flock_services,
        )
        from flock.webapp.app.main import (
            app as fastapi_app,  # The single FastAPI app instance
        )

        logger = get_logger("webapp.run") # Use a logger

        # 1. Set UI Theme globally for the webapp
        set_current_theme_name(ui_theme)
        logger.info(f"Unified server configured to use theme: {get_current_theme_name()}")

        # 2. Create RunStore & Set Global Services for Dependency Injection
        run_store_instance = RunStore()
        set_global_flock_services(flock_instance, run_store_instance)
        logger.info("Global Flock instance and RunStore set for dependency injection.")

        # 3. Make Flock instance and filename available on app.state
        fastapi_app.state.flock_instance = flock_instance
        source_file_attr = "_source_file_path" # Attribute where Flock might store its load path
        fastapi_app.state.flock_filename = getattr(flock_instance, source_file_attr, None) or \
                                           f"{flock_instance.name.replace(' ', '_').lower()}.flock.yaml"
        fastapi_app.state.run_store = run_store_instance

        logger.info(f"Flock '{flock_instance.name}' (from '{fastapi_app.state.flock_filename}') made available via app.state.")

        # 4. Store Custom Endpoints for registration by the lifespan manager in app.main
        processed_custom_endpoints = []
        if custom_endpoints:
            from flock.core.api.custom_endpoint import (
                FlockEndpoint,  # Ensure it's imported
            )
            if isinstance(custom_endpoints, dict):
                for (path_val, methods_val), cb_val in custom_endpoints.items():
                    processed_custom_endpoints.append(
                        FlockEndpoint(path=path_val, methods=list(methods_val) if methods_val else ["GET"], callback=cb_val)
                    )
            else: # Assumed Sequence[FlockEndpoint]
                processed_custom_endpoints.extend(list(custom_endpoints))

        if processed_custom_endpoints:
            add_pending_custom_endpoints(processed_custom_endpoints)
            logger.info(f"{len(processed_custom_endpoints)} custom endpoints stored for registration by app lifespan.")

        # 5. Update FastAPI app title (FastAPI app instance is now imported from main)
        fastapi_app.title = server_title

        # 6. Run Uvicorn
        logger.info(f"Running Uvicorn with application: flock.webapp.app.main:app")
        uvicorn.run(
            "flock.webapp.app.main:app",
            host=host,
            port=port,
            reload=False # Critical for programmatically set state like flock_instance
        )

    except ImportError as e:
        # More specific error logging
        print(f"CRITICAL: Error importing components for unified server: {e}", file=sys.stderr)
        print(f"Module not found: {e.name}", file=sys.stderr)
        print("This usually means a problem with sys.path or missing dependencies.", file=sys.stderr)
        print(f"Current sys.path: {sys.path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL: Error starting unified server: {e}", file=sys.stderr)
        # Consider logging the full traceback for easier debugging
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


# --- Standalone Webapp Runner (for `flock --web` or direct execution `python -m flock.webapp.run`) ---
def main():
    """Runs the Flock web application in standalone mode.
    In this mode, no specific Flock is pre-loaded by the startup script;
    the user will load or create one via the UI.
    The FastAPI app (`webapp.app.main:app`) will initialize with DI services
    set to None for Flock, and a new RunStore.
    """
    print("Starting Flock web application in standalone mode...")

    from flock.core.api.run_store import RunStore
    from flock.webapp.app.config import (
        get_current_theme_name,  # To log the theme being used
    )
    from flock.webapp.app.dependencies import set_global_flock_services

    # In true standalone, there's no flock instance initially.
    # The UI handles this "no flock loaded" state.
    # We still need a RunStore for API calls if they are made.
    # Pass None for flock_instance to set_global_flock_services.
    standalone_run_store = RunStore()
    set_global_flock_services(None, standalone_run_store)
    print(f"Standalone mode: Initialized global services. Flock: None, RunStore: {type(standalone_run_store)}")
    print(f"Standalone webapp using theme: {get_current_theme_name()}")


    host = os.environ.get("FLOCK_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("FLOCK_WEB_PORT", "8344"))
    webapp_reload = os.environ.get("FLOCK_WEB_RELOAD", "true").lower() == "true"

    app_import_string = "flock.webapp.app.main:app" # Uvicorn will import this
    print(f"Running Uvicorn: app='{app_import_string}', host='{host}', port={port}, reload={webapp_reload}")

    uvicorn.run(
        app_import_string,
        host=host,
        port=port,
        reload=webapp_reload,
    )

if __name__ == "__main__":
    # This allows running `python -m flock.webapp.run` for standalone UI testing
    main()
