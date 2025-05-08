import os  # For environment variable
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn

if TYPE_CHECKING:
    from flock.core import Flock

# --- Integrated Server Function ---


def start_integrated_server(
    flock_instance: "Flock",
    host: str,
    port: int,
    server_name: str,  # Currently unused as UI sets its own title
    theme_name: str | None = None,
):
    """Starts the webapp, preloads flock & theme, includes API routes (TODO)."""
    print(
        f"Starting integrated server for Flock '{flock_instance.name}' on {host}:{port}"
    )
    try:
        # Ensure src is in path (important if called from core)
        src_dir = Path(__file__).resolve().parent.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        # Import necessary webapp components *after* path setup
        from flock.core.api.run_store import (
            RunStore,  # Needed for API routes later
        )
        from flock.webapp.app.config import (
            get_current_theme_name,
            set_current_theme_name,
        )
        from flock.webapp.app.main import app as webapp_fastapi_app
        from flock.webapp.app.services.flock_service import (
            set_current_flock_instance_programmatically,
        )
        # from flock.core.api.endpoints import create_api_router # Need to adapt this later

        # 1. Set Theme (use provided or default)
        set_current_theme_name(
            theme_name
        )  # Uses default from config if theme_name is None
        print(f"Integrated server using theme: {get_current_theme_name()}")

        # 2. Set Flock Instance
        set_current_flock_instance_programmatically(
            flock_instance,
            f"{flock_instance.name.replace(' ', '_').lower()}_integrated.flock",
        )
        print(f"Flock '{flock_instance.name}' preloaded.")

        # 3. TODO: Adapt and Include API Routes
        # run_store = RunStore()
        # api_router = create_api_router(flock_instance, run_store) # Assuming refactored signature
        # webapp_fastapi_app.include_router(api_router, prefix="/api")
        # print("API routes included.")

        # 4. Run Uvicorn - STILL PASSING INSTANCE here because we need to modify it (add routes)
        #    and set state BEFORE running. Reload won't work well here.
        uvicorn.run(
            webapp_fastapi_app, host=host, port=port, reload=False
        )  # Ensure reload=False

    except ImportError as e:
        print(
            f"Error importing components for integrated server: {e}",
            file=sys.stderr,
        )
        print(
            "Ensure all dependencies are installed and paths are correct.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error starting integrated server: {e}", file=sys.stderr)
        sys.exit(1)


# --- Standalone Webapp Runner (for `flock --web`) ---


def main():
    """Run the Flock web application standalone."""
    # Theme is now set via environment variable read by config.py on import
    # No need to explicitly set it here anymore.
    print(f"Starting standalone webapp...")
    # Ensure src is in path
    src_dir = Path(__file__).resolve().parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        # Determine host, port, and reload settings
        host = os.environ.get("FLOCK_WEB_HOST", "127.0.0.1")
        port = int(os.environ.get("FLOCK_WEB_PORT", "8344"))
        reload = os.environ.get("FLOCK_WEB_RELOAD", "true").lower() == "true"
        # Use import string for app path
        app_import_string = "flock.webapp.app.main:app"

        # No need to import app instance here anymore for standalone mode
        # from flock.webapp.app.main import app as webapp_fastapi_app
        # from flock.webapp.app.config import get_current_theme_name
        # print(f"Standalone webapp using theme: {get_current_theme_name()}") # Config now logs this on load

        uvicorn.run(
            app_import_string,  # Use import string for reload capability
            host=host,
            port=port,
            reload=reload,
        )
    except ImportError as e:
        # Catch potential import error during uvicorn startup if path is wrong
        print(f"Error loading webapp modules via Uvicorn: {e}", file=sys.stderr)
        print(
            "Make sure all required packages are installed and src path is correct.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error starting standalone webapp: {e}", file=sys.stderr)
        sys.exit(1)


# Note: The `if __name__ == "__main__":` block is removed as this module
# is now primarily meant to be called via `main()` or `start_integrated_server()`
# The CLI entry point will call `main()` after potentially calling `set_initial_theme()`.
