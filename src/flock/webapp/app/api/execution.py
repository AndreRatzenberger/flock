import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from flock.core.util.spliter import parse_schema
from flock.webapp.app.services.flock_service import (
    get_current_flock_instance,
    run_current_flock_service,
)
from flock.webapp.app.utils import pydantic_to_dict

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# RENAMED this endpoint to avoid clash and for clarity
@router.get("/htmx/execution-form-content", response_class=HTMLResponse)
async def htmx_get_execution_form_content(request: Request):  # Renamed function
    flock = get_current_flock_instance()
    return templates.TemplateResponse(
        "partials/_execution_form.html",
        {
            "request": request,
            "flock": flock,
            "input_fields": [],
            "selected_agent_name": None,
        },
    )


@router.get("/htmx/agents/{agent_name}/input-form", response_class=HTMLResponse)
async def htmx_get_agent_input_form(request: Request, agent_name: str):
    # ... (same as before) ...
    flock = get_current_flock_instance()
    if not flock:
        return HTMLResponse("")
    agent = flock.agents.get(agent_name)
    if not agent:
        return HTMLResponse(
            f"<p class='error'>Agent '{agent_name}' not found.</p>"
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
                if "bool" in field_info["type"]:
                    field_info["html_type"] = "checkbox"
                elif (
                    "int" in field_info["type"] or "float" in field_info["type"]
                ):
                    field_info["html_type"] = "number"
                elif (
                    "list" in field_info["type"] or "dict" in field_info["type"]
                ):
                    field_info["html_type"] = "textarea"
                    field_info["placeholder"] = (
                        f"Enter JSON for {field_info['type']}"
                    )
                else:
                    field_info["html_type"] = "text"
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
async def htmx_run_flock(request: Request):
    # ... (same as before, ensure it uses the correct _results_display.html) ...
    flock = get_current_flock_instance()
    if not flock:
        return HTMLResponse("<p class='error'>No Flock loaded to run.</p>")
    form_data = await request.form()
    start_agent_name = form_data.get("start_agent_name")
    if not start_agent_name:
        return HTMLResponse("<p class='error'>Starting agent not selected.</p>")
    agent = flock.agents.get(start_agent_name)
    if not agent:
        return HTMLResponse(
            f"<p class='error'>Agent '{start_agent_name}' not found.</p>"
        )
    inputs = {}
    if agent.input and isinstance(agent.input, str):
        try:
            parsed_spec = parse_schema(agent.input)
            for name, type_str, _ in parsed_spec:
                form_field_name = f"agent_input_{name}"
                raw_value = form_data.get(form_field_name)
                if raw_value is None and "bool" in type_str.lower():
                    inputs[name] = False
                    continue
                if raw_value is None:
                    inputs[name] = None
                    continue
                if "int" in type_str.lower():
                    try:
                        inputs[name] = int(raw_value)
                    except ValueError:
                        return HTMLResponse(
                            f"<p class='error'>Invalid integer for '{name}'.</p>"
                        )
                elif "float" in type_str.lower():
                    try:
                        inputs[name] = float(raw_value)
                    except ValueError:
                        return HTMLResponse(
                            f"<p class='error'>Invalid float for '{name}'.</p>"
                        )
                elif "bool" in type_str.lower():
                    inputs[name] = raw_value.lower() in [
                        "true",
                        "on",
                        "1",
                        "yes",
                    ]
                elif "list" in type_str.lower() or "dict" in type_str.lower():
                    try:
                        inputs[name] = json.loads(raw_value)
                    except json.JSONDecodeError:
                        return HTMLResponse(
                            f"<p class='error'>Invalid JSON for '{name}'.</p>"
                        )
                else:
                    inputs[name] = raw_value
        except Exception as e:
            return HTMLResponse(
                f"<p class='error'>Error processing inputs for {start_agent_name}: {e}</p>"
            )

    try:
        # Run the flock service and get the result
        result_data = await run_current_flock_service(start_agent_name, inputs)

        # Detect Pydantic models and convert to dictionaries
        try:
            # Convert Pydantic models to dictionaries for JSON serialization
            result_data = pydantic_to_dict(result_data)

            # Test JSON serialization to catch any remaining issues
            try:
                json.dumps(result_data)
            except (TypeError, ValueError) as e:
                # If JSON serialization fails, convert to a string representation
                result_data = f"Error: Result contains non-serializable data: {e!s}\nOriginal result: {result_data!s}"

        except Exception as e:
            result_data = f"Error: Failed to process result data: {e!s}"

        return templates.TemplateResponse(
            "partials/_results_display.html",
            {"request": request, "result_data": result_data},
        )
    except Exception as e:
        error_message = f"Error during execution: {e!s}"
        return templates.TemplateResponse(
            "partials/_results_display.html",
            {"request": request, "result_data": error_message},
        )
