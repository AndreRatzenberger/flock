# src/flock/webapp/app/api/execution.py
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import (  # Ensure Form and HTTPException are imported
    APIRouter,
    Depends,
    Form,
    Request,
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

if TYPE_CHECKING:
    from flock.core.flock import Flock


from flock.core.flock import (
    Flock as ConcreteFlock,  # For creating Flock instance
)
from flock.core.logging.logging import (
    get_logger as get_flock_logger,  # For logging within the new endpoint
)
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
    current_flock_from_state: Flock | None = getattr(request.app.state, 'flock_instance', None)
    logger = get_flock_logger("webapp.execution.regular_run") # Standard logger

    if not current_flock_from_state:
        logger.error("HTMX Run (Regular): No Flock loaded in app_state.")
        return HTMLResponse("<p class='error'>No Flock loaded to run.</p>")

    form_data = await request.form()
    start_agent_name = form_data.get("start_agent_name")

    if not start_agent_name:
        logger.warning("HTMX Run (Regular): Starting agent not selected.")
        return HTMLResponse("<p class='error'>Starting agent not selected.</p>")

    agent = current_flock_from_state.agents.get(start_agent_name)
    if not agent:
        logger.error(f"HTMX Run (Regular): Agent '{start_agent_name}' not found in Flock '{current_flock_from_state.name}'.")
        return HTMLResponse(
            f"<p class='error'>Agent '{start_agent_name}' not found in the current Flock.</p>"
        )

    inputs = {}
    if agent.input and isinstance(agent.input, str):
        try:
            parsed_spec = parse_schema(agent.input)
            for name, type_str, _ in parsed_spec:
                form_field_name = f"agent_input_{name}"
                raw_value = form_data.get(form_field_name)
                if raw_value is None and "bool" in type_str.lower(): inputs[name] = False; continue
                if raw_value is None: inputs[name] = None; continue
                if "int" in type_str.lower(): inputs[name] = int(raw_value)
                elif "float" in type_str.lower(): inputs[name] = float(raw_value)
                elif "bool" in type_str.lower(): inputs[name] = raw_value.lower() in ["true", "on", "1", "yes"]
                elif "list" in type_str.lower() or "dict" in type_str.lower(): inputs[name] = json.loads(raw_value)
                else: inputs[name] = raw_value
        except ValueError as ve:
            logger.error(f"HTMX Run (Regular): Input parsing error for agent '{start_agent_name}': {ve}", exc_info=True)
            return HTMLResponse(f"<p class='error'>Invalid input format: {ve!s}</p>")
        except Exception as e_parse:
            logger.error(f"HTMX Run (Regular): Error processing inputs for '{start_agent_name}': {e_parse}", exc_info=True)
            return HTMLResponse(f"<p class='error'>Error processing inputs for {start_agent_name}: {e_parse}</p>")

    try:
        logger.info(f"HTMX Run (Regular): Executing agent '{start_agent_name}' from Flock '{current_flock_from_state.name}' via service.")
        result_data = await run_current_flock_service(start_agent_name, inputs, request.app.state)
    except Exception as e_run:
        logger.error(f"HTMX Run (Regular): Error during service execution for '{start_agent_name}': {e_run}", exc_info=True)
        return templates.TemplateResponse(
            "partials/_results_display.html",
            {"request": request, "result_data": {"error": f"Error during execution: {e_run!s}"}},
        )

    # Process and serialize result for template
    try:
        processed_result = pydantic_to_dict(result_data)
        try: json.dumps(processed_result)
        except (TypeError, ValueError) as ser_e:
            processed_result = f"Error: Result contains non-serializable data: {ser_e!s}\nOriginal result preview: {str(result_data)[:500]}..."
            logger.warning(f"HTMX Run (Regular): Serialization issue: {processed_result}")
    except Exception as proc_e:
        processed_result = f"Error: Failed to process result data: {proc_e!s}"
        logger.error(f"HTMX Run (Regular): Result processing error: {processed_result}", exc_info=True)

    return templates.TemplateResponse(
        "partials/_results_display.html",
        {"request": request, "result_data": processed_result},
    )


# --- NEW ENDPOINT FOR SHARED RUNS ---
@router.post("/htmx/run-shared", response_class=HTMLResponse)
async def htmx_run_shared_flock(
    request: Request,
    share_id: str = Form(...), # Expect share_id from the form
    # flock_definition_str: str = Form(...), # No longer needed
    # start_agent_name from form is still required implicitly via form_data.get()
    # No DI for current_flock, as we are using the one from app.state via share_id
):
    shared_logger = get_flock_logger("webapp.execution.shared_run_stateful")
    form_data = await request.form()
    start_agent_name = form_data.get("start_agent_name")

    if not start_agent_name:
        shared_logger.warning("HTMX Run Shared (Stateful): Starting agent name not provided in form.")
        return HTMLResponse("<p class='error'>Critical error: Starting agent name missing from shared run form.</p>")

    shared_logger.info(f"HTMX Run Shared (Stateful): Attempting to execute agent '{start_agent_name}' using pre-loaded Flock for share_id '{share_id}'.")

    inputs = {}
    result_data: Any = None
    temp_flock: ConcreteFlock | None = None

    try:
        # Retrieve the pre-loaded Flock instance from app.state
        shared_flocks_store = getattr(request.app.state, 'shared_flocks', {})
        temp_flock = shared_flocks_store.get(share_id)

        if not temp_flock:
            shared_logger.error(f"HTMX Run Shared: Flock instance for share_id '{share_id}' not found in app.state.")
            return HTMLResponse(f"<p class='error'>Shared session not found or expired. Please try accessing the shared link again.</p>")

        shared_logger.info(f"HTMX Run Shared: Successfully retrieved pre-loaded Flock '{temp_flock.name}' for agent '{start_agent_name}' (share_id: {share_id}).")

        agent = temp_flock.agents.get(start_agent_name)
        if not agent:
            shared_logger.error(f"HTMX Run Shared: Agent '{start_agent_name}' not found in shared Flock '{temp_flock.name}'.")
            return HTMLResponse(f"<p class='error'>Agent '{start_agent_name}' not found in the provided shared Flock definition.</p>")

        # Parse inputs for the agent from the temporary flock
        if agent.input and isinstance(agent.input, str):
            parsed_spec = parse_schema(agent.input)
            for name, type_str, _ in parsed_spec:
                form_field_name = f"agent_input_{name}" # Name used in shared_run_page.html
                raw_value = form_data.get(form_field_name)

                # Input type conversion (consistent with the other endpoint)
                if raw_value is None and "bool" in type_str.lower(): inputs[name] = False; continue
                if raw_value is None: inputs[name] = None; continue
                if "int" in type_str.lower(): inputs[name] = int(raw_value)
                elif "float" in type_str.lower(): inputs[name] = float(raw_value)
                elif "bool" in type_str.lower(): inputs[name] = raw_value.lower() in ["true", "on", "1", "yes"]
                elif "list" in type_str.lower() or "dict" in type_str.lower(): inputs[name] = json.loads(raw_value)
                else: inputs[name] = raw_value

        shared_logger.info(f"HTMX Run Shared: Executing agent '{start_agent_name}' in pre-loaded Flock '{temp_flock.name}'. Inputs: {list(inputs.keys())}")
        result_data = await temp_flock.run_async(start_agent=start_agent_name, input=inputs, box_result=False)
        shared_logger.info(f"HTMX Run Shared: Agent '{start_agent_name}' executed. Result keys: {list(result_data.keys()) if isinstance(result_data, dict) else 'N/A'}")

        # Process and serialize result for template (same logic as other endpoint)
        try:
            processed_result = pydantic_to_dict(result_data)
            try: json.dumps(processed_result)
            except (TypeError, ValueError) as ser_e:
                processed_result = f"Error: Result contains non-serializable data: {ser_e!s}\nOriginal result preview: {str(result_data)[:500]}..."
                shared_logger.warning(f"HTMX Run Shared: Serialization issue: {processed_result}")
        except Exception as proc_e:
            processed_result = f"Error: Failed to process result data: {proc_e!s}"
            shared_logger.error(f"HTMX Run Shared: Result processing error: {processed_result}", exc_info=True)

        return templates.TemplateResponse(
            "partials/_results_display.html",
            {"request": request, "result_data": processed_result},
        )

    except ValueError as ve: # Catch specific input parsing errors
        shared_logger.error(f"HTMX Run Shared: Input parsing error for agent '{start_agent_name}': {ve}", exc_info=True)
        return HTMLResponse(f"<p class='error'>Invalid input format: {ve!s}</p>")
    except Exception as e_general:
        error_message = f"An unexpected error occurred during shared execution: {e_general!s}"
        shared_logger.error(f"HTMX Run Shared: General error for agent '{start_agent_name}': {e_general}", exc_info=True)
        return templates.TemplateResponse(
            "partials/_results_display.html",
            {"request": request, "result_data": {"error": error_message } },
        )
