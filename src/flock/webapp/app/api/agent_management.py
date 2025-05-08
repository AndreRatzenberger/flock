from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from flock.webapp.app.services.flock_service import (
    add_agent_to_current_flock_service,
    get_current_flock_instance,
    get_registered_items_service,
    remove_agent_from_current_flock_service,
    update_agent_in_current_flock_service,
)

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/htmx/agent-list", response_class=HTMLResponse)
async def htmx_get_agent_list(
    request: Request, message: str = None, success: bool = None
):
    flock = get_current_flock_instance()
    if not flock:
        return HTMLResponse("<p class='error'>No Flock loaded.</p>")
    return templates.TemplateResponse(
        "partials/_agent_list.html",
        {
            "request": request,
            "flock": flock,
            "message": message,
            "success": success,
        },
    )


@router.get(
    "/htmx/agents/{agent_name}/details-form", response_class=HTMLResponse
)
async def htmx_get_agent_details_form(request: Request, agent_name: str):
    flock = get_current_flock_instance()
    if not flock:
        return HTMLResponse("<p class='error'>No Flock loaded.</p>")
    agent = flock.agents.get(agent_name)
    if not agent:
        return HTMLResponse(
            f"<p class='error'>Agent '{agent_name}' not found.</p>"
        )

    registered_tools = get_registered_items_service("tool")
    current_tools = (
        [tool.__name__ for tool in agent.tools] if agent.tools else []
    )

    return templates.TemplateResponse(
        "partials/_agent_detail_form.html",
        {
            "request": request,
            "agent": agent,
            "is_new": False,
            "registered_tools": registered_tools,
            "current_tools": current_tools,
        },
    )


@router.get("/htmx/agents/new-agent-form", response_class=HTMLResponse)
async def htmx_get_new_agent_form(request: Request):
    flock = get_current_flock_instance()
    if not flock:
        return HTMLResponse("<p class='error'>No Flock loaded.</p>")
    registered_tools = get_registered_items_service("tool")
    return templates.TemplateResponse(
        "partials/_agent_detail_form.html",
        {
            "request": request,
            "agent": None,  # For new agent
            "is_new": True,
            "registered_tools": registered_tools,
            "current_tools": [],
        },
    )


@router.post(
    "/htmx/agents", response_class=HTMLResponse
)  # For creating new agent
async def htmx_create_agent(
    request: Request,
    agent_name: str = Form(...),
    agent_description: str = Form(""),
    agent_model: str = Form(None),  # Can be empty to use Flock default
    input_signature: str = Form(...),
    output_signature: str = Form(...),
    tools: list[str] = Form([]),
):  # FastAPI handles list from multiple form fields with same name
    flock = get_current_flock_instance()
    if not flock:
        return HTMLResponse("<p class='error'>No Flock loaded.</p>")

    if (
        not agent_name.strip()
        or not input_signature.strip()
        or not output_signature.strip()
    ):
        # Render form again with error (or use a different target for error message)
        registered_tools = get_registered_items_service("tool")
        return templates.TemplateResponse(
            "partials/_agent_detail_form.html",
            {
                "request": request,
                "agent": None,
                "is_new": True,
                "error_message": "Name, Input Signature, and Output Signature are required.",
                "registered_tools": registered_tools,
                "current_tools": tools,  # Pass back selected tools
            },
        )

    agent_config = {
        "name": agent_name,
        "description": agent_description,
        "model": agent_model
        if agent_model
        else None,  # Pass None if empty string for FlockFactory
        "input": input_signature,
        "output": output_signature,
        "tools_names": tools,  # Pass tool names
    }
    success = add_agent_to_current_flock_service(agent_config)

    # After action, re-render the agent list and clear the detail form
    # Set headers for HTMX to trigger multiple target updates
    response_headers = {}
    if success:
        response_headers["HX-Trigger"] = (
            "agentListChanged"  # Custom event to refresh list
        )

    # Render an empty detail form or a success message for the detail panel
    empty_detail_form = templates.TemplateResponse(
        "partials/_agent_detail_form.html",
        {
            "request": request,
            "agent": None,
            "is_new": True,
            "registered_tools": get_registered_items_service("tool"),
            "form_message": "Agent created successfully!"
            if success
            else "Failed to create agent.",
            "success": success,
        },
    ).body.decode()

    return HTMLResponse(content=empty_detail_form, headers=response_headers)


@router.put(
    "/htmx/agents/{original_agent_name}", response_class=HTMLResponse
)  # For updating existing agent
async def htmx_update_agent(
    request: Request,
    original_agent_name: str,
    agent_name: str = Form(...),
    agent_description: str = Form(""),
    agent_model: str = Form(None),
    input_signature: str = Form(...),
    output_signature: str = Form(...),
    tools: list[str] = Form([]),
):
    flock = get_current_flock_instance()
    if not flock:
        return HTMLResponse("<p class='error'>No Flock loaded.</p>")

    agent_config = {
        "name": agent_name,
        "description": agent_description,
        "model": agent_model if agent_model else None,
        "input": input_signature,
        "output": output_signature,
        "tools_names": tools,
    }
    success = update_agent_in_current_flock_service(
        original_agent_name, agent_config
    )

    response_headers = {}
    if success:
        response_headers["HX-Trigger"] = "agentListChanged"

    # Re-render the form with update message
    updated_agent = flock.agents.get(
        agent_name
    )  # Get the potentially renamed agent
    registered_tools = get_registered_items_service("tool")
    current_tools = (
        [tool.__name__ for tool in updated_agent.tools]
        if updated_agent and updated_agent.tools
        else []
    )

    updated_form = templates.TemplateResponse(
        "partials/_agent_detail_form.html",
        {
            "request": request,
            "agent": updated_agent,  # Pass the updated agent
            "is_new": False,
            "form_message": "Agent updated successfully!"
            if success
            else "Failed to update agent.",
            "success": success,
            "registered_tools": registered_tools,
            "current_tools": current_tools,
        },
    ).body.decode()
    return HTMLResponse(content=updated_form, headers=response_headers)


@router.delete("/htmx/agents/{agent_name}", response_class=HTMLResponse)
async def htmx_delete_agent(request: Request, agent_name: str):
    flock = get_current_flock_instance()
    if not flock:
        return HTMLResponse("")  # Return empty to clear detail view

    success = remove_agent_from_current_flock_service(agent_name)

    response_headers = {}
    if success:
        response_headers["HX-Trigger"] = "agentListChanged"
        # Return an empty agent detail form to clear the panel
        # Also, the agent list will re-render due to HX-Trigger
        return HTMLResponse(
            templates.TemplateResponse(
                "partials/_agent_detail_form.html",
                {
                    "request": request,
                    "agent": None,
                    "is_new": True,
                    "form_message": f"Agent '{agent_name}' removed.",
                    "success": True,
                    "registered_tools": get_registered_items_service("tool"),
                },
            ).body.decode(),
            headers=response_headers,
        )
    else:
        # If deletion fails, re-render the agent detail form with an error
        # This scenario should be rare unless the agent was already removed
        agent = flock.agents.get(
            agent_name
        )  # Should still exist if delete failed
        registered_tools = get_registered_items_service("tool")
        current_tools = (
            [tool.__name__ for tool in agent.tools]
            if agent and agent.tools
            else []
        )
        return templates.TemplateResponse(
            "partials/_agent_detail_form.html",
            {
                "request": request,
                "agent": agent,
                "is_new": False,
                "form_message": f"Failed to remove agent '{agent_name}'.",
                "success": False,
                "registered_tools": registered_tools,
                "current_tools": current_tools,
            },
        )
