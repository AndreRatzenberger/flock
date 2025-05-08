# ... (keep existing imports and app setup) ...
import json
import shutil
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
from flock.webapp.app.config import FLOCK_FILES_DIR
from flock.webapp.app.services.flock_service import (
    clear_current_flock,
    create_new_flock_service,
    get_available_flock_files,
    get_current_flock_filename,
    get_current_flock_instance,
    get_flock_preview_service,
    load_flock_from_file_service,
)

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


def get_base_context(
    request: Request,
    error: str = None,
    success: str = None,
    ui_mode: str = "standalone",
) -> dict:
    return {
        "request": request,
        "current_flock": get_current_flock_instance(),
        "current_filename": get_current_flock_filename(),
        "error_message": error,
        "success_message": success,
        "ui_mode": ui_mode,
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
