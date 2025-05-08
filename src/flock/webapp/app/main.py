# ... (keep existing imports and app setup) ...
import json
import shutil
import sys  # For path
import urllib.parse
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from flock.webapp.app.api import (
    agent_management,
    execution,
    flock_management,
    registry_viewer,
)

# Import config functions
from flock.webapp.app.config import (
    DEFAULT_THEME_NAME,  # Import default for fallback
    FLOCK_FILES_DIR,
    get_current_theme_name,
)
from flock.webapp.app.services.flock_service import (
    clear_current_flock,
    create_new_flock_service,
    get_available_flock_files,
    get_current_flock_filename,
    get_current_flock_instance,
    get_flock_preview_service,
    load_flock_from_file_service,
)

# Helper for theme loading

# Find the 'src/flock' directory
flock_base_dir = (
    Path(__file__).resolve().parent.parent.parent
)  # src/flock/webapp/app -> src/flock

# Calculate themes directory relative to the flock base dir
themes_dir = flock_base_dir / "themes"

# Ensure the parent ('src') is in the path for core imports
src_dir = flock_base_dir.parent  # src/
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from flock.core.logging.formatters.themed_formatter import (
        load_theme_from_file,
    )

    THEME_LOADER_AVAILABLE = True
    # themes_dir calculated above
except ImportError:
    print(
        "Warning: Could not import flock.core theme loading utilities.",
        file=sys.stderr,
    )
    THEME_LOADER_AVAILABLE = False
    themes_dir = None  # Keep it None if loader failed

app = FastAPI(title="Flock UI")

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount(
    "/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static"
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.include_router(
    flock_management.router, prefix="/api/flocks", tags=["Flock Management API"]
)
app.include_router(
    agent_management.router, prefix="/api/flocks", tags=["Agent Management API"]
)
# Ensure execution router is imported and included BEFORE it's referenced by the renamed route
app.include_router(
    execution.router, prefix="/api/flocks", tags=["Execution API"]
)
app.include_router(
    registry_viewer.router, prefix="/api/registry", tags=["Registry API"]
)


def generate_theme_css(theme_name: str | None) -> str:
    """Loads a theme TOML and generates CSS variable overrides."""
    if not THEME_LOADER_AVAILABLE or themes_dir is None:
        return ""  # Return empty if theme loading isn't possible

    active_theme_name = theme_name or DEFAULT_THEME_NAME
    theme_filename = f"{active_theme_name}.toml"
    theme_path = themes_dir / theme_filename

    if not theme_path.exists():
        print(
            f"Warning: Theme file not found: {theme_path}. Using default theme.",
            file=sys.stderr,
        )
        # Optionally load the default theme file if the requested one isn't found
        theme_filename = f"{DEFAULT_THEME_NAME}.toml"
        theme_path = themes_dir / theme_filename
        if not theme_path.exists():
            print(
                f"Warning: Default theme file not found: {theme_path}. No theme CSS generated.",
                file=sys.stderr,
            )
            return ""  # Return empty if even default isn't found
        active_theme_name = (
            DEFAULT_THEME_NAME  # Update active name if defaulted
        )

    try:
        theme_dict = load_theme_from_file(str(theme_path))
    except Exception as e:
        print(f"Error loading theme file {theme_path}: {e}", file=sys.stderr)
        return ""  # Return empty on error

    # --- Define TOML Color -> CSS Variable Mapping ---
    # This mapping is crucial and may need adjustment based on theme intent & Pico usage
    css_vars = {}
    try:
        css_vars["--pico-background-color"] = theme_dict["colors"][
            "primary"
        ].get("background", "#161719")
        css_vars["--pico-color"] = theme_dict["colors"]["primary"].get(
            "foreground", "#c5c8c6"
        )

        # Use bright blue for primary interactive elements? Requires theme to have it.
        css_vars["--pico-primary"] = theme_dict["colors"]["bright"].get(
            "blue", "#007bff"
        )  # Fallback blue
        css_vars["--pico-primary-hover"] = theme_dict["colors"]["normal"].get(
            "blue", "#0056b3"
        )  # Fallback darker blue

        # Contrast background for things like buttons using primary color
        css_vars["--pico-contrast"] = theme_dict["colors"]["primary"][
            "background"
        ]
        # Background of cards, menus, etc.
        css_vars["--pico-card-background-color"] = theme_dict["colors"][
            "selection"
        ].get("background", "#ffffff")
        css_vars["--pico-card-border-color"] = theme_dict["colors"][
            "normal"
        ].get("black", "#e1e1e1")
        css_vars["--pico-muted-border-color"] = theme_dict["colors"][
            "normal"
        ].get("black", "#e1e1e1")
        css_vars["--pico-form-element-background-color"] = theme_dict["colors"][
            "primary"
        ].get("background", "#ffffff")  # Or slightly different?
        css_vars["--pico-form-element-border-color"] = theme_dict["colors"][
            "bright"
        ].get("black", "#cccccc")
        css_vars["--pico-form-element-focus-color"] = theme_dict["colors"][
            "bright"
        ].get("blue", "#007bff")  # Same as primary?

        # Custom vars (examples)
        css_vars["--flock-sidebar-background"] = theme_dict["colors"][
            "selection"
        ].get("background", "#161719")  # Same as main bg for now
        css_vars["--flock-header-background"] = theme_dict["colors"][
            "primary"
        ].get("background", "#161719")
        css_vars["--flock-error-color"] = theme_dict["colors"]["bright"].get(
            "red", "#dc3545"
        )
        css_vars["--flock-success-color"] = theme_dict["colors"]["bright"].get(
            "green", "#28a745"
        )

    except KeyError as e:
        print(
            f"Warning: Missing expected key in theme '{active_theme_name}': {e}. CSS may be incomplete.",
            file=sys.stderr,
        )

    if not css_vars:
        return ""  # Return empty if no variables were mapped

    css_rules = [f"    {name}: {value};" for name, value in css_vars.items()]

    # Apply overrides within the currently active theme selector for better specificity
    # We could get the theme name passed in, or maybe check the <html> tag attribute?
    # For simplicity, let's assume we want to override Pico's dark theme vars when a theme is loaded.
    # A better approach might involve removing data-theme="dark" and applying theme to :root
    # or having specific data-theme selectors for each flock theme.
    # Let's try applying to [data-theme="dark"] first.
    selector = '[data-theme="dark"]'
    css_string = ":root {\n" + "\n".join(css_rules) + "\n}"

    print(
        f"--- Generated CSS for theme '{active_theme_name}' ---"
    )  # Debugging print
    print(css_string)  # Debugging print
    print(
        "----------------------------------------------------"
    )  # Debugging print
    return css_string


def get_base_context(
    request: Request,
    error: str = None,
    success: str = None,
    ui_mode: str = "standalone",
) -> dict:
    theme_name = get_current_theme_name()  # Get theme from config
    theme_css = generate_theme_css(theme_name)
    return {
        "request": request,
        "current_flock": get_current_flock_instance(),
        "current_filename": get_current_flock_filename(),
        "error_message": error,
        "success_message": success,
        "ui_mode": ui_mode,
        "theme_css": theme_css,  # Add generated CSS to context
    }


# --- Main Page Routes ---
@app.get("/", response_class=HTMLResponse)
async def page_dashboard(
    request: Request,
    error: str = None,
    success: str = None,
    ui_mode: str = Query(
        None
    ),  # Default to None to detect if it was explicitly passed
):
    # Determine effective ui_mode
    effective_ui_mode = ui_mode
    flock_is_preloaded = get_current_flock_instance() is not None

    if effective_ui_mode is None:  # ui_mode not in query parameters
        if flock_is_preloaded:
            # If a flock is preloaded (likely by API server) and no mode specified,
            # default to scoped and redirect to make the URL explicit.
            return RedirectResponse(url="/?ui_mode=scoped", status_code=307)
        else:
            effective_ui_mode = "standalone"  # True standalone launch
    elif effective_ui_mode == "scoped" and not flock_is_preloaded:
        # If explicitly asked for scoped mode but no flock is loaded (e.g. user bookmarked URL after server restart)
        # It will show the "scoped-no-flock-view". We could also redirect to standalone.
        # For now, let it show the "no flock loaded in scoped mode" message.
        pass

    # Conditional flock clearing based on the *effective* ui_mode
    if effective_ui_mode != "scoped":
        # If we are about to enter standalone mode, and a flock might have been
        # preloaded (e.g. user navigated from /?ui_mode=scoped to /?ui_mode=standalone),
        # ensure it's cleared for a true standalone experience.
        if flock_is_preloaded:  # Clear only if one was there
            clear_current_flock()

    context = get_base_context(request, error, success, effective_ui_mode)

    if effective_ui_mode == "scoped":
        if get_current_flock_instance():  # Re-check, as clear_current_flock might have run if user switched modes
            context["initial_content_url"] = (
                "/api/flocks/htmx/flock-properties-form"
            )
        else:
            context["initial_content_url"] = "/ui/htmx/scoped-no-flock-view"
    else:  # Standalone mode
        context["initial_content_url"] = "/ui/htmx/load-flock-view"

    return templates.TemplateResponse("base.html", context)


@app.get("/ui/editor/properties", response_class=HTMLResponse)
async def page_editor_properties(
    request: Request,
    success: str = None,
    error: str = None,
    ui_mode: str = Query("standalone"),
):
    # ... (same as before) ...
    flock = get_current_flock_instance()
    if not flock:
        err_msg = "No flock loaded. Please load or create a flock first."
        # Preserve ui_mode on redirect if it was passed
        redirect_url = f"/?error={urllib.parse.quote(err_msg)}"
        if ui_mode == "scoped":
            redirect_url += "&ui_mode=scoped"
        return RedirectResponse(url=redirect_url, status_code=303)
    context = get_base_context(request, error, success, ui_mode)
    context["initial_content_url"] = "/api/flocks/htmx/flock-properties-form"
    return templates.TemplateResponse("base.html", context)


@app.get("/ui/editor/agents", response_class=HTMLResponse)
async def page_editor_agents(
    request: Request,
    success: str = None,
    error: str = None,
    ui_mode: str = Query("standalone"),
):
    # ... (same as before) ...
    flock = get_current_flock_instance()
    if not flock:
        # Preserve ui_mode on redirect
        redirect_url = (
            f"/?error={urllib.parse.quote('No flock loaded for agent view.')}"
        )
        if ui_mode == "scoped":
            redirect_url += "&ui_mode=scoped"
        return RedirectResponse(url=redirect_url, status_code=303)
    context = get_base_context(request, error, success, ui_mode)
    context["initial_content_url"] = "/ui/htmx/agent-manager-view"
    return templates.TemplateResponse("base.html", context)


@app.get("/ui/editor/execute", response_class=HTMLResponse)
async def page_editor_execute(
    request: Request,
    success: str = None,
    error: str = None,
    ui_mode: str = Query("standalone"),
):
    flock = get_current_flock_instance()
    if not flock:
        # Preserve ui_mode on redirect
        redirect_url = (
            f"/?error={urllib.parse.quote('No flock loaded to execute.')}"
        )
        if ui_mode == "scoped":
            redirect_url += "&ui_mode=scoped"
        return RedirectResponse(url=redirect_url, status_code=303)
    context = get_base_context(request, error, success, ui_mode)
    # UPDATED initial_content_url
    context["initial_content_url"] = "/ui/htmx/execution-view-container"
    return templates.TemplateResponse("base.html", context)


# ... (registry and create page routes remain the same) ...
@app.get("/ui/registry", response_class=HTMLResponse)
async def page_registry(
    request: Request,
    error: str = None,
    success: str = None,
    ui_mode: str = Query("standalone"),
):
    context = get_base_context(request, error, success, ui_mode)
    context["initial_content_url"] = "/ui/htmx/registry-viewer"
    return templates.TemplateResponse("base.html", context)


@app.get("/ui/create", response_class=HTMLResponse)
async def page_create(
    request: Request,
    error: str = None,
    success: str = None,
    ui_mode: str = Query("standalone"),
):
    clear_current_flock()
    # Create page should arguably not be accessible in scoped mode directly via URL,
    # as the sidebar link will be hidden. If accessed, treat as standalone.
    context = get_base_context(
        request, error, success, "standalone"
    )  # Force standalone for direct access
    context["initial_content_url"] = "/ui/htmx/create-flock-form"
    return templates.TemplateResponse("base.html", context)


# --- HTMX Content Routes ---
@app.get("/ui/htmx/sidebar", response_class=HTMLResponse)
async def htmx_get_sidebar(
    request: Request, ui_mode: str = Query("standalone")
):
    # ... (same as before) ...
    return templates.TemplateResponse(
        "partials/_sidebar.html",
        {
            "request": request,
            "current_flock": get_current_flock_instance(),
            "ui_mode": ui_mode,
        },
    )


@app.get("/ui/htmx/header-flock-status", response_class=HTMLResponse)
async def htmx_get_header_flock_status(
    request: Request, ui_mode: str = Query("standalone")
):
    # ui_mode isn't strictly needed for this partial's content, but good to accept if passed by hx-get
    return templates.TemplateResponse(
        "partials/_header_flock_status.html",
        {
            "request": request,
            "current_flock": get_current_flock_instance(),
            "current_filename": get_current_flock_filename(),
        },
    )


@app.get("/ui/htmx/load-flock-view", response_class=HTMLResponse)
async def htmx_get_load_flock_view(
    request: Request,
    error: str = None,
    success: str = None,
    ui_mode: str = Query("standalone"),
):
    # ... (same as before) ...
    # This view is part of the "standalone" functionality.
    # If somehow accessed in scoped mode, it might be confusing, but let it render.
    return templates.TemplateResponse(
        "partials/_load_manage_view.html",
        {
            "request": request,
            "error_message": error,
            "success_message": success,
            "ui_mode": ui_mode,  # Pass for consistency, though not directly used in this partial
        },
    )


@app.get("/ui/htmx/dashboard-flock-file-list", response_class=HTMLResponse)
async def htmx_get_dashboard_flock_file_list_partial(request: Request):
    # ... (same as before) ...
    return templates.TemplateResponse(
        "partials/_dashboard_flock_file_list.html",
        {"request": request, "flock_files": get_available_flock_files()},
    )


@app.get("/ui/htmx/dashboard-default-action-pane", response_class=HTMLResponse)
async def htmx_get_dashboard_default_action_pane(request: Request):
    # ... (same as before) ...
    return HTMLResponse("""
        <article style="text-align:center; margin-top: 2rem; border: none; background: transparent;">
            <p>Select a Flock from the list to view its details and load it into the editor.</p>
            <hr>
            <p>Or, create a new Flock or upload an existing one using the "Create New Flock" option in the sidebar.</p>
        </article>
    """)


@app.get(
    "/ui/htmx/dashboard-flock-properties-preview/{filename}",
    response_class=HTMLResponse,
)
async def htmx_get_dashboard_flock_properties_preview(
    request: Request, filename: str
):
    # ... (same as before) ...
    preview_flock_data = get_flock_preview_service(filename)
    return templates.TemplateResponse(
        "partials/_dashboard_flock_properties_preview.html",
        {
            "request": request,
            "selected_filename": filename,
            "preview_flock": preview_flock_data,
        },
    )


@app.get("/ui/htmx/create-flock-form", response_class=HTMLResponse)
async def htmx_get_create_flock_form(
    request: Request,
    error: str = None,
    success: str = None,
    ui_mode: str = Query("standalone"),
):
    # ... (same as before) ...
    # This view is part of the "standalone" functionality.
    return templates.TemplateResponse(
        "partials/_create_flock_form.html",
        {
            "request": request,
            "error_message": error,
            "success_message": success,
            "ui_mode": ui_mode,  # Pass for consistency
        },
    )


@app.get("/ui/htmx/agent-manager-view", response_class=HTMLResponse)
async def htmx_get_agent_manager_view(request: Request):
    # ... (same as before) ...
    flock = get_current_flock_instance()
    if not flock:
        return HTMLResponse(
            "<article class='error'><p>No flock loaded. Cannot manage agents.</p></article>"
        )
    return templates.TemplateResponse(
        "partials/_agent_manager_view.html",
        {"request": request, "flock": flock},
    )


@app.get("/ui/htmx/registry-viewer", response_class=HTMLResponse)
async def htmx_get_registry_viewer(request: Request):
    # ... (same as before) ...
    return templates.TemplateResponse(
        "partials/_registry_viewer_content.html", {"request": request}
    )


# --- NEW HTMX ROUTE FOR THE EXECUTION VIEW CONTAINER ---
@app.get("/ui/htmx/execution-view-container", response_class=HTMLResponse)
async def htmx_get_execution_view_container(request: Request):
    flock = get_current_flock_instance()
    if not flock:
        return HTMLResponse(
            "<article class='error'><p>No Flock loaded. Cannot execute.</p></article>"
        )
    return templates.TemplateResponse(
        "partials/_execution_view_container.html", {"request": request}
    )


# A new HTMX route for scoped mode when no flock is initially loaded (should ideally not happen)
@app.get("/ui/htmx/scoped-no-flock-view", response_class=HTMLResponse)
async def htmx_scoped_no_flock_view(request: Request):
    return HTMLResponse("""
        <article style="text-align:center; margin-top: 2rem; border: none; background: transparent;">
            <hgroup>
                <h2>Scoped Flock Mode</h2>
                <h3>No Flock Loaded</h3>
            </hgroup>
            <p>This UI is in a scoped mode, expecting a Flock to be pre-loaded.</p>
            <p>Please ensure the calling application provides a Flock instance.</p>
        </article>
    """)


# Endpoint to launch the UI in scoped mode with a preloaded flock
@app.post("/ui/launch-scoped", response_class=RedirectResponse)
async def launch_scoped_ui(
    request: Request,
    flock_data: dict,  # This would be the flock's JSON data
    # Potentially also receive filename if it's from a saved file
):
    # Here, you would parse flock_data, create a Flock instance,
    # and set it as the current flock using your flock_service methods.
    # For now, let's assume flock_service has a method like:
    # set_current_flock_from_data(data) -> bool (returns True if successful)

    # This is a placeholder for actual flock loading logic
    # from flock.core.entities.flock import Flock # Assuming Flock can be instantiated from dict
    # from flock.webapp.app.services.flock_service import set_current_flock_instance, set_current_flock_filename

    # try:
    #     # Assuming flock_data is a dict that can initialize a Flock object
    #     # You might need a more robust way to deserialize, e.g., using Pydantic models
    #     loaded_flock = Flock(**flock_data) # This is a simplistic example
    #     set_current_flock_instance(loaded_flock)
    #     # If the flock has a name or identifier, you might set it as well
    #     # set_current_flock_filename(flock_data.get("name", "scoped_flock")) # Example
    #
    #     # Redirect to the agent editor or properties page in scoped mode
    #     # The page_dashboard will handle ui_mode=scoped and redirect/set initial content appropriately
    #     return RedirectResponse(url="/?ui_mode=scoped", status_code=303)
    # except Exception as e:
    #     # Log error e
    #     # Redirect to an error page or the standalone dashboard with an error message
    #     error_msg = f"Failed to load flock for scoped view: {e}"
    #     return RedirectResponse(url=f"/?error={urllib.parse.quote(error_msg)}&ui_mode=standalone", status_code=303)

    # For now, since we don't have the flock loading logic here,
    # we'll just redirect. The calling service (`src/flock/core/api`)
    # will need to ensure the flock is loaded into the webapp's session/state
    # *before* redirecting to this UI.

    # A more direct way if `load_flock_from_data_service` exists and sets it globally for the session:
    # success = load_flock_from_data_service(flock_data, "scoped_runtime_flock") # example filename
    # if success:
    #    return RedirectResponse(url="/ui/editor/agents?ui_mode=scoped", status_code=303) # or properties
    # else:
    #    return RedirectResponse(url="/?error=Failed+to+load+scoped+flock&ui_mode=standalone", status_code=303)

    # Given the current structure, the simplest way for an external service to "preload" a flock
    # is to use the existing `load_flock_from_file_service` if the flock can be temporarily saved,
    # or by enhancing `flock_service` to allow setting a Flock instance directly.
    # Let's assume the flock is already loaded into the session by the calling API for now.
    # The calling API will be responsible for calling a service function within the webapp's context.

    # This endpoint's primary job is now to redirect to the UI in the correct mode.
    # The actual loading of the flock should happen *before* this redirect,
    # by the API server calling a service function within the webapp's context.

    # For demonstration, let's imagine the calling API has already used a service
    # to set the flock. We just redirect.
    if get_current_flock_instance():
        return RedirectResponse(
            url="/ui/editor/agents?ui_mode=scoped", status_code=303
        )
    else:
        # If no flock is loaded, go to the main page in scoped mode, which will show the "no flock" message.
        return RedirectResponse(url="/?ui_mode=scoped", status_code=303)


# --- Action Routes ...
# The `load-flock-action/*` and `create-flock` POST routes should remain the same as they already
# correctly target `#main-content-area` and trigger `flockLoaded`.
# ... (rest of action routes: load-flock-action/by-name, by-upload, create-flock)
@app.post("/ui/load-flock-action/by-name", response_class=HTMLResponse)
async def ui_load_flock_by_name_action(
    request: Request, selected_flock_filename: str = Form(...)
):
    loaded_flock = load_flock_from_file_service(selected_flock_filename)
    response_headers = {}
    if loaded_flock:
        success_message = f"Flock '{loaded_flock.name}' loaded from '{selected_flock_filename}'."
        response_headers["HX-Push-Url"] = "/ui/editor/properties"
        response_headers["HX-Trigger"] = json.dumps(
            {
                "flockLoaded": None,
                "notify": {"type": "success", "message": success_message},
            }
        )
        return templates.TemplateResponse(
            "partials/_flock_properties_form.html",
            {
                "request": request,
                "flock": loaded_flock,
                "current_filename": get_current_flock_filename(),
            },
            headers=response_headers,
        )
    else:
        error_message = (
            f"Failed to load flock file '{selected_flock_filename}'."
        )
        response_headers["HX-Trigger"] = json.dumps(
            {"notify": {"type": "error", "message": error_message}}
        )
        return templates.TemplateResponse(
            "partials/_load_manage_view.html",
            {"request": request, "error_message_inline": error_message},
            headers=response_headers,
        )


@app.post("/ui/load-flock-action/by-upload", response_class=HTMLResponse)
async def ui_load_flock_by_upload_action(
    request: Request, flock_file_upload: UploadFile = File(...)
):
    error_message = None
    filename_to_load = None
    response_headers = {}
    if flock_file_upload and flock_file_upload.filename:
        if not flock_file_upload.filename.endswith((".yaml", ".yml", ".flock")):
            error_message = "Invalid file type."
        else:
            upload_path = FLOCK_FILES_DIR / flock_file_upload.filename
            try:
                with upload_path.open("wb") as buffer:
                    shutil.copyfileobj(flock_file_upload.file, buffer)
                filename_to_load = flock_file_upload.filename
            except Exception as e:
                error_message = f"Upload failed: {e}"
            finally:
                await flock_file_upload.close()
    else:
        error_message = "No file uploaded."

    if filename_to_load and not error_message:
        loaded_flock = load_flock_from_file_service(filename_to_load)
        if loaded_flock:
            success_message = (
                f"Flock '{loaded_flock.name}' loaded from '{filename_to_load}'."
            )
            response_headers["HX-Push-Url"] = "/ui/editor/properties"
            response_headers["HX-Trigger"] = json.dumps(
                {
                    "flockLoaded": None,
                    "flockFileListChanged": None,
                    "notify": {"type": "success", "message": success_message},
                }
            )
            return templates.TemplateResponse(
                "partials/_flock_properties_form.html",
                {
                    "request": request,
                    "flock": loaded_flock,
                    "current_filename": get_current_flock_filename(),
                },
                headers=response_headers,
            )
        else:
            error_message = f"Failed to process uploaded '{filename_to_load}'."

    response_headers["HX-Trigger"] = json.dumps(
        {
            "notify": {
                "type": "error",
                "message": error_message or "Upload failed.",
            }
        }
    )
    return templates.TemplateResponse(
        "partials/_create_flock_form.html",
        {  # Changed target to create form on upload error
            "request": request,
            "error_message": error_message or "Upload action failed.",
        },
        headers=response_headers,
    )


@app.post("/ui/create-flock", response_class=HTMLResponse)
async def ui_create_flock_action(
    request: Request,
    flock_name: str = Form(...),
    default_model: str = Form(None),
    description: str = Form(None),
):
    if not flock_name.strip():
        return templates.TemplateResponse(
            "partials/_create_flock_form.html",
            {
                "request": request,
                "error_message": "Flock name cannot be empty.",
            },
        )
    new_flock = create_new_flock_service(flock_name, default_model, description)
    success_msg = (
        f"New flock '{new_flock.name}' created. Configure properties and save."
    )
    response_headers = {
        "HX-Push-Url": "/ui/editor/properties",
        "HX-Trigger": json.dumps(
            {
                "flockLoaded": None,
                "notify": {"type": "success", "message": success_msg},
            }
        ),
    }
    return templates.TemplateResponse(
        "partials/_flock_properties_form.html",
        {
            "request": request,
            "flock": new_flock,
            "current_filename": get_current_flock_filename(),
        },
        headers=response_headers,
    )
