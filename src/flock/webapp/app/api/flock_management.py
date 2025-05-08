from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from flock.webapp.app.services.flock_service import (
    get_current_flock_filename,
    get_current_flock_instance,
    save_current_flock_to_file_service,
    update_flock_properties_service,
)

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Points to flock-ui/
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/htmx/flock-properties-form", response_class=HTMLResponse)
async def htmx_get_flock_properties_form(
    request: Request, update_message: str = None, success: bool = None
):
    flock = get_current_flock_instance()
    if not flock:
        # This case should ideally not be reached if editor page properly redirects
        return HTMLResponse(
            "<div class='error'>Error: No flock loaded. Please load or create one first.</div>"
        )
    return templates.TemplateResponse(
        "partials/_flock_properties_form.html",
        {
            "request": request,
            "flock": flock,
            "current_filename": get_current_flock_filename(),
            "update_message": update_message,
            "success": success,
        },
    )


@router.post("/htmx/flock-properties", response_class=HTMLResponse)
async def htmx_update_flock_properties(
    request: Request,
    flock_name: str = Form(...),
    default_model: str = Form(...),
    description: str = Form(""),
):
    success_update = update_flock_properties_service(
        flock_name, default_model, description
    )
    flock = get_current_flock_instance()  # Get updated instance
    # Re-render the form with a message
    return templates.TemplateResponse(
        "partials/_flock_properties_form.html",
        {
            "request": request,
            "flock": flock,
            "current_filename": get_current_flock_filename(),
            "update_message": "Flock properties updated!"
            if success_update
            else "Failed to update properties.",
            "success": success_update,
        },
    )


@router.post("/htmx/save-flock", response_class=HTMLResponse)
async def htmx_save_flock(request: Request, save_filename: str = Form(...)):
    if not save_filename.strip():  # Basic validation
        flock = get_current_flock_instance()
        return templates.TemplateResponse(
            "partials/_flock_properties_form.html",
            {
                "request": request,
                "flock": flock,
                "current_filename": get_current_flock_filename(),
                "save_message": "Filename cannot be empty.",
                "success": False,
            },
        )

    if not (
        save_filename.endswith(".yaml")
        or save_filename.endswith(".yml")
        or save_filename.endswith(".flock")
    ):
        save_filename += ".flock.yaml"  # Add default extension

    success, message = save_current_flock_to_file_service(save_filename)
    flock = get_current_flock_instance()
    return templates.TemplateResponse(
        "partials/_flock_properties_form.html",
        {
            "request": request,
            "flock": flock,
            "current_filename": get_current_flock_filename()
            if success
            else get_current_flock_filename(),  # Update filename if save was successful
            "save_message": message,
            "success": success,
        },
    )
