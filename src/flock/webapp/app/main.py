# src/flock/webapp/app/main.py
import json
import shutil
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from flock.core.api.endpoints import create_api_router
from flock.core.api.run_store import RunStore

# Import core Flock components and API related modules
from flock.core.flock import Flock  # For type hinting
from flock.core.logging.logging import get_logger  # For logging

# Import UI-specific routers
from flock.webapp.app.api import (
    agent_management,
    execution,
    flock_management,
    registry_viewer,
)
from flock.webapp.app.config import (
    DEFAULT_THEME_NAME,
    FLOCK_FILES_DIR,
    THEMES_DIR,
    get_current_theme_name,
)

# Import dependency management and config
from flock.webapp.app.dependencies import (
    get_pending_custom_endpoints_and_clear,
    set_global_flock_services,
)

# Import service functions (which now expect app_state)
from flock.webapp.app.services.flock_service import (
    clear_current_flock_service,
    create_new_flock_service,
    get_available_flock_files,
    get_flock_preview_service,
    load_flock_from_file_service,
    # Note: get_current_flock_instance/filename are removed from service,
    # as main.py will use request.app.state for this.
)
from flock.webapp.app.theme_mapper import alacritty_to_pico

logger = get_logger("webapp.main")


try:
    from flock.core.logging.formatters.themed_formatter import (
        load_theme_from_file,
    )
    THEME_LOADER_AVAILABLE = True
except ImportError:
    logger.warning("Could not import flock.core theme loading utilities.")
    THEME_LOADER_AVAILABLE = False

# --- .env helpers (copied from original main.py for self-containment) ---
ENV_FILE_PATH = Path(".env") #Path(os.getenv("FLOCK_WEB_ENV_FILE", Path.home() / ".flock" / ".env"))
#ENV_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
SHOW_SECRETS_KEY = "SHOW_SECRETS"

def load_env_file_web() -> dict[str, str]:
    env_vars: dict[str, str] = {}
    if not ENV_FILE_PATH.exists(): return env_vars
    with open(ENV_FILE_PATH) as f: lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line: env_vars[""] = ""; continue
        if line.startswith("#"): env_vars[line] = ""; continue
        if "=" in line: k, v = line.split("=", 1); env_vars[k] = v
        else: env_vars[line] = ""
    return env_vars

def save_env_file_web(env_vars: dict[str, str]):
    try:
        with open(ENV_FILE_PATH, "w") as f:
            for k, v in env_vars.items():
                if k.startswith("#"): f.write(f"{k}\n")
                elif not k: f.write("\n")
                else: f.write(f"{k}={v}\n")
    except Exception as e: logger.error(f"[Settings] Failed to save .env: {e}")

def is_sensitive_web(key: str) -> bool:
    patterns = ["key", "token", "secret", "password", "api", "pat"]; low = key.lower()
    return any(p in low for p in patterns)

def mask_sensitive_value_web(value: str) -> str:
    if not value: return value
    if len(value) <= 4: return "••••"
    return value[:2] + "•" * (len(value) - 4) + value[-2:]

def get_show_secrets_setting_web(env_vars: dict[str, str]) -> bool:
    return env_vars.get(SHOW_SECRETS_KEY, "false").lower() == "true"

def set_show_secrets_setting_web(show: bool):
    env_vars = load_env_file_web()
    env_vars[SHOW_SECRETS_KEY] = str(show)
    save_env_file_web(env_vars)
# --- End .env helpers ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application starting up...")
    # Flock instance and RunStore are expected to be set on app.state
    # by `start_unified_server` in `webapp/run.py` *before* uvicorn starts the app.
    # The call to `set_global_flock_services` also happens there.

    # Add custom routes if any were passed during server startup
    # These are retrieved from the dependency module where `start_unified_server` stored them.
    pending_endpoints = get_pending_custom_endpoints_and_clear()
    if pending_endpoints:
        flock_instance_from_state: Flock | None = getattr(app.state, "flock_instance", None)
        if flock_instance_from_state:
            from flock.core.api.main import (
                FlockAPI,  # Local import for this specific task
            )
            # Create a temporary FlockAPI service object just for adding routes
            temp_flock_api_service = FlockAPI(
                flock_instance_from_state,
                custom_endpoints=pending_endpoints
            )
            temp_flock_api_service.add_custom_routes_to_app(app)
            logger.info(f"Lifespan: Added {len(pending_endpoints)} custom API routes to main app.")
        else:
            logger.warning("Lifespan: Pending custom endpoints found, but no Flock instance in app.state. Cannot add custom routes.")
    yield
    logger.info("FastAPI application shutting down...")

app = FastAPI(title="Flock Web UI & API", lifespan=lifespan)

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

core_api_router = create_api_router()
app.include_router(core_api_router, prefix="/api", tags=["Flock API Core"])
app.include_router(flock_management.router, prefix="/ui/api/flock", tags=["UI Flock Management"])
app.include_router(agent_management.router, prefix="/ui/api/flock", tags=["UI Agent Management"])
app.include_router(execution.router, prefix="/ui/api/flock", tags=["UI Execution"])
app.include_router(registry_viewer.router, prefix="/ui/api/registry", tags=["UI Registry"])

def generate_theme_css_web(theme_name: str | None) -> str:
    if not THEME_LOADER_AVAILABLE or THEMES_DIR is None: return ""
    active_theme_name = theme_name or get_current_theme_name() or DEFAULT_THEME_NAME
    theme_filename = f"{active_theme_name}.toml"
    theme_path = THEMES_DIR / theme_filename
    if not theme_path.exists():
        logger.warning(f"Theme file not found: {theme_path}. Using default: {DEFAULT_THEME_NAME}.toml")
        theme_path = THEMES_DIR / f"{DEFAULT_THEME_NAME}.toml"
        active_theme_name = DEFAULT_THEME_NAME
        if not theme_path.exists():
            logger.warning(f"Default theme file not found: {theme_path}. No theme CSS.")
            return ""
    try: theme_dict = load_theme_from_file(str(theme_path))
    except Exception as e: logger.error(f"Error loading theme {theme_path}: {e}"); return ""

    pico_vars = alacritty_to_pico(theme_dict)
    if not pico_vars: return ""
    css_rules = [f"    {name}: {value};" for name, value in pico_vars.items()]
    css_string = ":root {\n" + "\n".join(css_rules) + "\n}"
    return css_string

def get_base_context_web(
    request: Request, error: str = None, success: str = None, ui_mode: str = "standalone"
) -> dict:
    flock_instance_from_state: Flock | None = getattr(request.app.state, "flock_instance", None)
    current_flock_filename_from_state: str | None = getattr(request.app.state, "flock_filename", None)
    theme_name = get_current_theme_name()
    theme_css = generate_theme_css_web(theme_name)
    return {
        "request": request,
        "current_flock": flock_instance_from_state,
        "current_filename": current_flock_filename_from_state,
        "error_message": error,
        "success_message": success,
        "ui_mode": ui_mode,
        "theme_css": theme_css,
        "active_theme_name": theme_name,
        "chat_enabled": getattr(request.app.state, "chat_enabled", False),
    }

@app.get("/", response_class=HTMLResponse, tags=["UI Pages"])
async def page_dashboard(
    request: Request, error: str = None, success: str = None, ui_mode: str = Query(None)
):
    effective_ui_mode = ui_mode
    flock_is_preloaded = hasattr(request.app.state, "flock_instance") and request.app.state.flock_instance is not None

    if effective_ui_mode is None:
        effective_ui_mode = "scoped" if flock_is_preloaded else "standalone"
        if effective_ui_mode == "scoped":
             return RedirectResponse(url=f"/?ui_mode=scoped&initial_load=true", status_code=307)

    if effective_ui_mode == "standalone" and flock_is_preloaded:
        clear_current_flock_service(request.app.state) # Pass app.state
        logger.info("Switched to standalone mode, cleared preloaded Flock instance from app.state.")

    context = get_base_context_web(request, error, success, effective_ui_mode)
    flock_in_state = hasattr(request.app.state, "flock_instance") and request.app.state.flock_instance is not None

    if effective_ui_mode == "scoped":
        context["initial_content_url"] = "/ui/htmx/execution-view-container" if flock_in_state else "/ui/htmx/scoped-no-flock-view"
    else:
        context["initial_content_url"] = "/ui/htmx/load-flock-view"
    return templates.TemplateResponse("base.html", context)

@app.get("/ui/editor/{section:path}", response_class=HTMLResponse, tags=["UI Pages"])
async def page_editor_section(
    request: Request, section: str, success: str = None, error: str = None, ui_mode: str = Query("standalone")
):
    flock_instance_from_state: Flock | None = getattr(request.app.state, "flock_instance", None)
    if not flock_instance_from_state:
        err_msg = "No flock loaded. Please load or create a flock first."
        redirect_url = f"/?error={urllib.parse.quote(err_msg)}"
        if ui_mode == "scoped": redirect_url += "&ui_mode=scoped"
        return RedirectResponse(url=redirect_url, status_code=303)

    context = get_base_context_web(request, error, success, ui_mode)
    content_map = {
        "properties": "/ui/api/flock/htmx/flock-properties-form",
        "agents": "/ui/htmx/agent-manager-view",
        "execute": "/ui/htmx/execution-view-container"
    }
    context["initial_content_url"] = content_map.get(section, "/ui/htmx/load-flock-view")
    if section not in content_map: context["error_message"] = "Invalid editor section."
    return templates.TemplateResponse("base.html", context)

@app.get("/ui/registry", response_class=HTMLResponse, tags=["UI Pages"])
async def page_registry(request: Request, error: str = None, success: str = None, ui_mode: str = Query("standalone")):
    context = get_base_context_web(request, error, success, ui_mode)
    context["initial_content_url"] = "/ui/htmx/registry-viewer"
    return templates.TemplateResponse("base.html", context)

@app.get("/ui/create", response_class=HTMLResponse, tags=["UI Pages"])
async def page_create(request: Request, error: str = None, success: str = None, ui_mode: str = Query("standalone")):
    clear_current_flock_service(request.app.state) # Pass app.state
    context = get_base_context_web(request, error, success, "standalone")
    context["initial_content_url"] = "/ui/htmx/create-flock-form"
    return templates.TemplateResponse("base.html", context)

@app.get("/ui/htmx/sidebar", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_get_sidebar(request: Request, ui_mode: str = Query("standalone")):
    return templates.TemplateResponse("partials/_sidebar.html", get_base_context_web(request, ui_mode=ui_mode))

@app.get("/ui/htmx/header-flock-status", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_get_header_flock_status(request: Request, ui_mode: str = Query("standalone")):
    return templates.TemplateResponse("partials/_header_flock_status.html", get_base_context_web(request, ui_mode=ui_mode))

@app.get("/ui/htmx/load-flock-view", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_get_load_flock_view(request: Request, error: str = None, success: str = None, ui_mode: str = Query("standalone")):
    return templates.TemplateResponse("partials/_load_manager_view.html", get_base_context_web(request, error, success, ui_mode))

@app.get("/ui/htmx/dashboard-flock-file-list", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_get_dashboard_flock_file_list_partial(request: Request):
    return templates.TemplateResponse("partials/_dashboard_flock_file_list.html", {"request": request, "flock_files": get_available_flock_files()})

@app.get("/ui/htmx/dashboard-default-action-pane", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_get_dashboard_default_action_pane(request: Request):
    return HTMLResponse("""<article style="text-align:center; margin-top: 2rem; border: none; background: transparent;"><p>Select a Flock from the list to view its details and load it into the editor.</p><hr><p>Or, create a new Flock or upload an existing one using the "Create New Flock" option in the sidebar.</p></article>""")

@app.get("/ui/htmx/dashboard-flock-properties-preview/{filename}", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_get_dashboard_flock_properties_preview(request: Request, filename: str):
    preview_flock_data = get_flock_preview_service(filename)
    return templates.TemplateResponse("partials/_dashboard_flock_properties_preview.html", {"request": request, "selected_filename": filename, "preview_flock": preview_flock_data})

@app.get("/ui/htmx/create-flock-form", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_get_create_flock_form(request: Request, error: str = None, success: str = None, ui_mode: str = Query("standalone")):
    return templates.TemplateResponse("partials/_create_flock_form.html", get_base_context_web(request, error, success, ui_mode))

@app.get("/ui/htmx/agent-manager-view", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_get_agent_manager_view(request: Request):
    context = get_base_context_web(request) # This gets flock from app.state
    if not context.get("current_flock"): # Check if flock exists in the context
        return HTMLResponse("<article class='error'><p>No flock loaded. Cannot manage agents.</p></article>")
    # Pass the 'current_flock' from the context to the template as 'flock'
    return templates.TemplateResponse(
        "partials/_agent_manager_view.html",
        {"request": request, "flock": context.get("current_flock")}
    )

@app.get("/ui/htmx/registry-viewer", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_get_registry_viewer(request: Request):
    return templates.TemplateResponse("partials/_registry_viewer_content.html", get_base_context_web(request))

@app.get("/ui/htmx/execution-view-container", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_get_execution_view_container(request: Request):
    context = get_base_context_web(request)
    if not context.get("current_flock"): return HTMLResponse("<article class='error'><p>No Flock loaded. Cannot execute.</p></article>")
    return templates.TemplateResponse("partials/_execution_view_container.html", context)

@app.get("/ui/htmx/scoped-no-flock-view", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_scoped_no_flock_view(request: Request):
    return HTMLResponse("""<article style="text-align:center; margin-top: 2rem; border: none; background: transparent;"><hgroup><h2>Scoped Flock Mode</h2><h3>No Flock Loaded</h3></hgroup><p>This UI is in a scoped mode, expecting a Flock to be pre-loaded.</p><p>Please ensure the calling application provides a Flock instance.</p></article>""")

# --- Action Routes (POST requests for UI interactions) ---
@app.post("/ui/load-flock-action/by-name", response_class=HTMLResponse, tags=["UI Actions"])
async def ui_load_flock_by_name_action(request: Request, selected_flock_filename: str = Form(...)):
    loaded_flock = load_flock_from_file_service(selected_flock_filename, request.app.state)
    response_headers = {}
    ui_mode_query = request.query_params.get("ui_mode", "standalone")
    if loaded_flock:
        success_message_text = f"Flock '{loaded_flock.name}' loaded from '{selected_flock_filename}'."
        response_headers["HX-Push-Url"] = "/ui/editor/properties?ui_mode=" + request.query_params.get("ui_mode", "standalone")
        response_headers["HX-Trigger"] = json.dumps({"flockLoaded": None, "notify": {"type": "success", "message": success_message_text}})
        # Use get_base_context_web to ensure all necessary context vars are present for the partial
        context = get_base_context_web(request, success=success_message_text, ui_mode=ui_mode_query)
        return templates.TemplateResponse("partials/_flock_properties_form.html", context, headers=response_headers)
    else:
        error_message_text = f"Failed to load flock file '{selected_flock_filename}'."
        response_headers["HX-Trigger"] = json.dumps({"notify": {"type": "error", "message": error_message_text}})
        context = get_base_context_web(request, error=error_message_text, ui_mode=ui_mode_query)
        context["error_message_inline"] = error_message_text # For direct display in partial
        return templates.TemplateResponse("partials/_load_manager_view.html", context, headers=response_headers)

@app.post("/ui/load-flock-action/by-upload", response_class=HTMLResponse, tags=["UI Actions"])
async def ui_load_flock_by_upload_action(request: Request, flock_file_upload: UploadFile = File(...)):
    error_message_text, filename_to_load, response_headers = None, None, {}
    ui_mode_query = request.query_params.get("ui_mode", "standalone")

    if flock_file_upload and flock_file_upload.filename:
        if not flock_file_upload.filename.endswith((".yaml", ".yml", ".flock")): error_message_text = "Invalid file type."
        else:
            upload_path = FLOCK_FILES_DIR / flock_file_upload.filename
            try:
                with upload_path.open("wb") as buffer: shutil.copyfileobj(flock_file_upload.file, buffer)
                filename_to_load = flock_file_upload.filename
            except Exception as e: error_message_text = f"Upload failed: {e}"
            finally: await flock_file_upload.close()
    else: error_message_text = "No file uploaded."

    if filename_to_load and not error_message_text:
        loaded_flock = load_flock_from_file_service(filename_to_load, request.app.state)
        if loaded_flock:
            success_message_text = f"Flock '{loaded_flock.name}' loaded from '{filename_to_load}'."
            response_headers["HX-Push-Url"] = f"/ui/editor/properties?ui_mode={ui_mode_query}"
            response_headers["HX-Trigger"] = json.dumps({"flockLoaded": None, "flockFileListChanged": None, "notify": {"type": "success", "message": success_message_text}})
            # CORRECTED CALL:
            context = get_base_context_web(request, success=success_message_text, ui_mode=ui_mode_query)
            return templates.TemplateResponse("partials/_flock_properties_form.html", context, headers=response_headers)
        else: error_message_text = f"Failed to process uploaded '{filename_to_load}'."

    final_error_msg = error_message_text or "Upload failed."
    response_headers["HX-Trigger"] = json.dumps({"notify": {"type": "error", "message": final_error_msg}})
    # CORRECTED CALL:
    context = get_base_context_web(request, error=final_error_msg, ui_mode=ui_mode_query)
    return templates.TemplateResponse("partials/_create_flock_form.html", context, headers=response_headers)

@app.post("/ui/create-flock", response_class=HTMLResponse, tags=["UI Actions"])
async def ui_create_flock_action(request: Request, flock_name: str = Form(...), default_model: str = Form(None), description: str = Form(None)):
    ui_mode_query = request.query_params.get("ui_mode", "standalone")
    if not flock_name.strip():
        # CORRECTED CALL:
        context = get_base_context_web(request, error="Flock name cannot be empty.", ui_mode=ui_mode_query)
        return templates.TemplateResponse("partials/_create_flock_form.html", context)

    new_flock = create_new_flock_service(flock_name, default_model, description, request.app.state)
    success_msg_text = f"New flock '{new_flock.name}' created. Configure properties and save."
    response_headers = {"HX-Push-Url": f"/ui/editor/properties?ui_mode={ui_mode_query}", "HX-Trigger": json.dumps({"flockLoaded": None, "notify": {"type": "success", "message": success_msg_text}})}
    # CORRECTED CALL:
    context = get_base_context_web(request, success=success_msg_text, ui_mode=ui_mode_query)
    return templates.TemplateResponse("partials/_flock_properties_form.html", context, headers=response_headers)

# --- Settings Page & Endpoints ---
@app.get("/ui/settings", response_class=HTMLResponse, tags=["UI Pages"])
async def page_settings(request: Request, error: str = None, success: str = None, ui_mode: str = Query("standalone")):
    context = get_base_context_web(request, error, success, ui_mode)
    context["initial_content_url"] = "/ui/htmx/settings-view"
    return templates.TemplateResponse("base.html", context)

def _prepare_env_vars_for_template_web():
    env_vars_raw = load_env_file_web(); show_secrets = get_show_secrets_setting_web(env_vars_raw)
    env_vars_list = []
    for name, value in env_vars_raw.items():
        if name.startswith("#") or name == "": continue
        display_value = value if (not is_sensitive_web(name) or show_secrets) else mask_sensitive_value_web(value)
        env_vars_list.append({"name": name, "value": display_value})
    return env_vars_list, show_secrets

@app.get("/ui/htmx/settings-view", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_get_settings_view(request: Request):
    env_vars_list, show_secrets = _prepare_env_vars_for_template_web()
    theme_name = get_current_theme_name()
    themes_available = [p.stem for p in THEMES_DIR.glob("*.toml")] if THEMES_DIR and THEMES_DIR.exists() else []
    return templates.TemplateResponse("partials/_settings_view.html", {"request": request, "env_vars": env_vars_list, "show_secrets": show_secrets, "themes": themes_available, "current_theme": theme_name})

@app.post("/ui/htmx/toggle-show-secrets", response_class=HTMLResponse, tags=["UI Actions"])
async def htmx_toggle_show_secrets(request: Request):
    env_vars_raw = load_env_file_web(); current = get_show_secrets_setting_web(env_vars_raw)
    set_show_secrets_setting_web(not current)
    env_vars_list, show_secrets = _prepare_env_vars_for_template_web()
    return templates.TemplateResponse("partials/_env_vars_table.html", {"request": request, "env_vars": env_vars_list, "show_secrets": show_secrets})

@app.post("/ui/htmx/env-delete", response_class=HTMLResponse, tags=["UI Actions"])
async def htmx_env_delete(request: Request, var_name: str = Form(...)):
    env_vars_raw = load_env_file_web()
    if var_name in env_vars_raw: del env_vars_raw[var_name]; save_env_file_web(env_vars_raw)
    env_vars_list, show_secrets = _prepare_env_vars_for_template_web()
    return templates.TemplateResponse("partials/_env_vars_table.html", {"request": request, "env_vars": env_vars_list, "show_secrets": show_secrets})

@app.post("/ui/htmx/env-edit", response_class=HTMLResponse, tags=["UI Actions"])
async def htmx_env_edit(request: Request, var_name: str = Form(...)):
    new_value = request.headers.get("HX-Prompt")
    env_vars_list, show_secrets = _prepare_env_vars_for_template_web()
    if new_value is not None:
        env_vars_raw = load_env_file_web()
        env_vars_raw[var_name] = new_value
        save_env_file_web(env_vars_raw)
        env_vars_list, show_secrets = _prepare_env_vars_for_template_web()
    return templates.TemplateResponse("partials/_env_vars_table.html", {"request": request, "env_vars": env_vars_list, "show_secrets": show_secrets})

@app.get("/ui/htmx/env-add-form", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_env_add_form(request: Request):
    return HTMLResponse("""<form hx-post='/ui/htmx/env-add' hx-target='#env-vars-container' hx-swap='outerHTML' style='display:flex; gap:0.5rem; margin-bottom:0.5rem;'><input name='var_name' placeholder='NAME' required style='flex:2;'><input name='var_value' placeholder='VALUE' style='flex:3;'><button type='submit'>Add</button></form>""")

@app.post("/ui/htmx/env-add", response_class=HTMLResponse, tags=["UI Actions"])
async def htmx_env_add(request: Request, var_name: str = Form(...), var_value: str = Form("")):
    env_vars_raw = load_env_file_web()
    env_vars_raw[var_name] = var_value; save_env_file_web(env_vars_raw)
    env_vars_list, show_secrets = _prepare_env_vars_for_template_web()
    return templates.TemplateResponse("partials/_env_vars_table.html", {"request": request, "env_vars": env_vars_list, "show_secrets": show_secrets})

@app.get("/ui/htmx/theme-preview", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_theme_preview(request: Request, theme: str = Query(None)):
    theme_name = theme or get_current_theme_name() or DEFAULT_THEME_NAME
    try:
        theme_path = THEMES_DIR / f"{theme_name}.toml" if THEMES_DIR else None
        if not (theme_path and theme_path.exists()): return HTMLResponse("<p>Theme not found.</p>")
        theme_data = load_theme_from_file(str(theme_path))
    except Exception as e: return HTMLResponse(f"<p>Error loading theme: {e}</p>")
    css_vars = alacritty_to_pico(theme_data)
    css_vars_str = ":root {\n" + "\n".join([f"  {k}: {v};" for k, v in css_vars.items()]) + "\n}"
    main_colors = [("Background", css_vars.get("--pico-background-color")), ("Text", css_vars.get("--pico-color")), ("Primary", css_vars.get("--pico-primary")), ("Secondary", css_vars.get("--pico-secondary")), ("Muted", css_vars.get("--pico-muted-color"))]
    return templates.TemplateResponse("partials/_theme_preview.html", {"request": request, "theme_name": theme_name, "css_vars_str": css_vars_str, "main_colors": main_colors})

@app.post("/ui/apply-theme", tags=["UI Actions"])
async def apply_theme(request: Request, theme: str = Form(...)):
    try:
        from flock.webapp.app.config import set_current_theme_name
        set_current_theme_name(theme)
        headers = {"HX-Refresh": "true"}
        return HTMLResponse("", headers=headers)
    except Exception as e: return HTMLResponse(f"Failed to apply theme: {e}", status_code=500)

@app.get("/ui/htmx/settings/env-vars", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_settings_env_vars(request: Request):
    env_vars_list, show_secrets = _prepare_env_vars_for_template_web()
    return templates.TemplateResponse("partials/_settings_env_content.html", {"request": request, "env_vars": env_vars_list, "show_secrets": show_secrets})

@app.get("/ui/htmx/settings/theme", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_settings_theme(request: Request):
    theme_name = get_current_theme_name()
    themes_available = [p.stem for p in THEMES_DIR.glob("*.toml")] if THEMES_DIR and THEMES_DIR.exists() else []
    return templates.TemplateResponse("partials/_settings_theme_content.html", {"request": request, "themes": themes_available, "current_theme": theme_name})

@app.get("/ui/chat", response_class=HTMLResponse, tags=["UI Pages"])
async def page_chat(request: Request, ui_mode: str = Query("standalone")):
    context = get_base_context_web(request, ui_mode=ui_mode)
    context["initial_content_url"] = "/ui/htmx/chat-view"
    return templates.TemplateResponse("base.html", context)

@app.get("/ui/htmx/chat-view", response_class=HTMLResponse, tags=["UI HTMX Partials"])
async def htmx_get_chat_view(request: Request):
    # Render container partial; session handled in chat router
    return templates.TemplateResponse("partials/_chat_container.html", get_base_context_web(request))

if __name__ == "__main__":
    import uvicorn
    # Ensure the dependency injection system is initialized for standalone run
    temp_run_store = RunStore()
    # Create a default/dummy Flock instance for standalone UI testing
    # This allows the UI to function without being started by `Flock.start_api()`
    dev_flock_instance = Flock(name="DevStandaloneFlock", model="test/dummy", enable_logging=True, show_flock_banner=False)

    set_global_flock_services(dev_flock_instance, temp_run_store)
    app.state.flock_instance = dev_flock_instance
    app.state.run_store = temp_run_store
    app.state.flock_filename = "development_standalone.flock.yaml"

    logger.info("Running webapp.app.main directly for development with a dummy Flock instance.")
    uvicorn.run(app, host="127.0.0.1", port=8344, reload=True)
