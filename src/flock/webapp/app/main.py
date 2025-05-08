# ... (keep existing imports and app setup) ...
import json
import shutil
import urllib.parse
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
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
    request: Request, error: str = None, success: str = None
) -> dict:
    return {
        "request": request,
        "current_flock": get_current_flock_instance(),
        "current_filename": get_current_flock_filename(),
        "error_message": error,
        "success_message": success,
    }


# --- Main Page Routes ---
@app.get("/", response_class=HTMLResponse)
async def page_dashboard(
    request: Request, error: str = None, success: str = None
):
    clear_current_flock()
    context = get_base_context(request, error, success)
    context["initial_content_url"] = "/ui/htmx/load-flock-view"
    return templates.TemplateResponse("base.html", context)


@app.get("/ui/editor/properties", response_class=HTMLResponse)
async def page_editor_properties(
    request: Request, success: str = None, error: str = None
):
    # ... (same as before) ...
    flock = get_current_flock_instance()
    if not flock:
        err_msg = "No flock loaded. Please load or create a flock first."
        return RedirectResponse(
            url=f"/?error={urllib.parse.quote(err_msg)}", status_code=303
        )
    context = get_base_context(request, error, success)
    context["initial_content_url"] = "/api/flocks/htmx/flock-properties-form"
    return templates.TemplateResponse("base.html", context)


@app.get("/ui/editor/agents", response_class=HTMLResponse)
async def page_editor_agents(
    request: Request, success: str = None, error: str = None
):
    # ... (same as before) ...
    flock = get_current_flock_instance()
    if not flock:
        return RedirectResponse(
            url=f"/?error={urllib.parse.quote('No flock loaded for agent view.')}",
            status_code=303,
        )
    context = get_base_context(request, error, success)
    context["initial_content_url"] = "/ui/htmx/agent-manager-view"
    return templates.TemplateResponse("base.html", context)


@app.get("/ui/editor/execute", response_class=HTMLResponse)
async def page_editor_execute(
    request: Request, success: str = None, error: str = None
):
    flock = get_current_flock_instance()
    if not flock:
        return RedirectResponse(
            url=f"/?error={urllib.parse.quote('No flock loaded to execute.')}",
            status_code=303,
        )
    context = get_base_context(request, error, success)
    # UPDATED initial_content_url
    context["initial_content_url"] = "/ui/htmx/execution-view-container"
    return templates.TemplateResponse("base.html", context)


# ... (registry and create page routes remain the same) ...
@app.get("/ui/registry", response_class=HTMLResponse)
async def page_registry(
    request: Request, error: str = None, success: str = None
):
    context = get_base_context(request, error, success)
    context["initial_content_url"] = "/ui/htmx/registry-viewer"
    return templates.TemplateResponse("base.html", context)


@app.get("/ui/create", response_class=HTMLResponse)
async def page_create(request: Request, error: str = None, success: str = None):
    clear_current_flock()
    context = get_base_context(request, error, success)
    context["initial_content_url"] = "/ui/htmx/create-flock-form"
    return templates.TemplateResponse("base.html", context)


# --- HTMX Content Routes ---
@app.get("/ui/htmx/sidebar", response_class=HTMLResponse)
async def htmx_get_sidebar(request: Request):
    # ... (same as before) ...
    return templates.TemplateResponse(
        "partials/_sidebar.html",
        {"request": request, "current_flock": get_current_flock_instance()},
    )


@app.get("/ui/htmx/load-flock-view", response_class=HTMLResponse)
async def htmx_get_load_flock_view(
    request: Request, error: str = None, success: str = None
):
    # ... (same as before) ...
    return templates.TemplateResponse(
        "partials/_load_manage_view.html",
        {
            "request": request,
            "error_message": error,
            "success_message": success,
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
    request: Request, error: str = None, success: str = None
):
    # ... (same as before) ...
    return templates.TemplateResponse(
        "partials/_create_flock_form.html",
        {
            "request": request,
            "error_message": error,
            "success_message": success,
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
