# src/flock/webapp/app/api/execution.py
import json
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request  # Added Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

if TYPE_CHECKING:
    from flock.core.flock import Flock


from flock.core.util.spliter import parse_schema

# Import the dependency to get the current Flock instance
from flock.webapp.app.dependencies import (
    get_flock_instance,
    get_optional_flock_instance,
)

# Service function now takes app_state
from flock.webapp.app.services.flock_service import (
    run_current_flock_service,
    # get_current_flock_instance IS NO LONGER IMPORTED
)
from flock.webapp.app.utils import pydantic_to_dict

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/htmx/execution-form-content", response_class=HTMLResponse)
async def htmx_get_execution_form_content(
    request: Request,
    current_flock: "Flock | None" = Depends(get_optional_flock_instance) # Use optional if form can show 'no flock'
):
    # flock instance is injected
    return templates.TemplateResponse(
        "partials/_execution_form.html",
        {
            "request": request,
            "flock": current_flock, # Pass the injected flock instance
            "input_fields": [],
            "selected_agent_name": None, # Form starts with no agent selected
        },
    )


@router.get("/htmx/agents/{agent_name}/input-form", response_class=HTMLResponse)
async def htmx_get_agent_input_form(
    request: Request,
    agent_name: str,
    current_flock: "Flock" = Depends(get_flock_instance) # Expect flock to be loaded
):
    # flock instance is injected
    agent = current_flock.agents.get(agent_name)
    if not agent:
        return HTMLResponse(
            f"<p class='error'>Agent '{agent_name}' not found in the current Flock.</p>"
        )

    input_fields = []
    if agent.input and isinstance(agent.input, str):
        try:
            parsed_spec = parse_schema(agent.input)
            for name, type_str, description in parsed_spec:
                field_info = {
                    "name": name,
                    "type": type_str.lower(),
                    "description": description or "",
                }
                if "bool" in field_info["type"]: field_info["html_type"] = "checkbox"
                elif "int" in field_info["type"] or "float" in field_info["type"]: field_info["html_type"] = "number"
                elif "list" in field_info["type"] or "dict" in field_info["type"]:
                    field_info["html_type"] = "textarea"
                    field_info["placeholder"] = f"Enter JSON for {field_info['type']}"
                else: field_info["html_type"] = "text"
                input_fields.append(field_info)
        except Exception as e:
            return HTMLResponse(
                f"<p class='error'>Error parsing input signature for {agent_name}: {e}</p>"
            )
    return templates.TemplateResponse(
        "partials/_dynamic_input_form_content.html",
        {"request": request, "input_fields": input_fields},
    )


@router.post("/htmx/run", response_class=HTMLResponse)
async def htmx_run_flock(
    request: Request,
    # current_flock: Flock = Depends(get_flock_instance) # Service will use app_state
):
    # The service function run_current_flock_service now takes app_state
    # We retrieve current_flock from app_state inside the service or before calling if needed for validation here

    # It's better to get flock from app_state here to validate before calling service
    current_flock: Flock | None = getattr(request.app.state, 'flock_instance', None)

    if not current_flock:
        return HTMLResponse("<p class='error'>No Flock loaded to run.</p>")

    form_data = await request.form()
    start_agent_name = form_data.get("start_agent_name")

    if not start_agent_name:
        return HTMLResponse("<p class='error'>Starting agent not selected.</p>")

    agent = current_flock.agents.get(start_agent_name)
    if not agent:
        return HTMLResponse(
            f"<p class='error'>Agent '{start_agent_name}' not found in the current Flock.</p>"
        )

    inputs = {}
    if agent.input and isinstance(agent.input, str):
        try:
            parsed_spec = parse_schema(agent.input)
            for name, type_str, _ in parsed_spec:
                form_field_name = f"agent_input_{name}" # Matches the name in _dynamic_input_form_content.html
                raw_value = form_data.get(form_field_name)

                if raw_value is None and "bool" in type_str.lower(): inputs[name] = False; continue
                if raw_value is None: inputs[name] = None; continue

                if "int" in type_str.lower():
                    try: inputs[name] = int(raw_value)
                    except ValueError: return HTMLResponse(f"<p class='error'>Invalid integer for '{name}'.</p>")
                elif "float" in type_str.lower():
                    try: inputs[name] = float(raw_value)
                    except ValueError: return HTMLResponse(f"<p class='error'>Invalid float for '{name}'.</p>")
                elif "bool" in type_str.lower():
                    inputs[name] = raw_value.lower() in ["true", "on", "1", "yes"]
                elif "list" in type_str.lower() or "dict" in type_str.lower():
                    try: inputs[name] = json.loads(raw_value)
                    except json.JSONDecodeError: return HTMLResponse(f"<p class='error'>Invalid JSON for '{name}'.</p>")
                else: inputs[name] = raw_value
        except Exception as e:
            return HTMLResponse(f"<p class='error'>Error processing inputs for {start_agent_name}: {e}</p>")

    try:
        # Pass request.app.state to the service function
        result_data = await run_current_flock_service(start_agent_name, inputs, request.app.state)

        try:
            result_data = pydantic_to_dict(result_data)
            try: json.dumps(result_data)
            except (TypeError, ValueError) as ser_e:
                result_data = f"Error: Result contains non-serializable data: {ser_e!s}\nOriginal result: {result_data!s}"
        except Exception as proc_e:
            result_data = f"Error: Failed to process result data: {proc_e!s}"

        return templates.TemplateResponse(
            "partials/_results_display.html",
            {"request": request, "result_data": result_data},
        )
    except Exception as e:
        error_message = f"Error during execution: {e!s}"
        return templates.TemplateResponse(
            "partials/_results_display.html",
            {"request": request, "result_data": error_message}, # Display error in the same partial
        )
