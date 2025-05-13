from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from flock.webapp.app.main import get_base_context_web, templates

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory session store (cookie-based). Not suitable for production scale.
# ---------------------------------------------------------------------------
_chat_sessions: dict[str, list[dict[str, str]]] = {}

COOKIE_NAME = "chat_sid"


def _ensure_session(request: Request):
    """Returns (sid, history_list) tuple and guarantees cookie presence."""
    sid: str | None = request.cookies.get(COOKIE_NAME)
    if not sid:
        sid = uuid4().hex
    if sid not in _chat_sessions:
        _chat_sessions[sid] = []
    return sid, _chat_sessions[sid]


# ---------------------------------------------------------------------------
# Chat configuration (per app instance)
# ---------------------------------------------------------------------------


class ChatConfig(BaseModel):
    agent_name: str | None = None  # Name of the Flock agent to chat with
    message_key: str = "message"
    history_key: str = "history"
    response_key: str = "response"


# Store a single global chat config on the FastAPI app state
def get_chat_config(request: Request) -> ChatConfig:
    if not hasattr(request.app.state, "chat_config"):
        request.app.state.chat_config = ChatConfig()
    return request.app.state.chat_config


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/chat", response_class=HTMLResponse, tags=["Chat"])
async def chat_page(request: Request):
    """Full-page chat UI (works even when the main UI is disabled)."""
    sid, history = _ensure_session(request)
    cfg = get_chat_config(request)
    context = get_base_context_web(request, ui_mode="standalone")
    context.update({"history": history, "chat_cfg": cfg, "chat_subtitle": f"Agent: {cfg.agent_name}" if cfg.agent_name else "Echo demo"})
    response = templates.TemplateResponse("chat.html", context)
    # Set cookie if not already present
    if COOKIE_NAME not in request.cookies:
        response.set_cookie(COOKIE_NAME, sid, max_age=60 * 60 * 24 * 7)
    return response


@router.get("/chat/messages", response_class=HTMLResponse, tags=["Chat"], include_in_schema=False)
async def chat_history_partial(request: Request):
    """HTMX endpoint that returns the rendered message list."""
    _, history = _ensure_session(request)
    return templates.TemplateResponse(
        "partials/_chat_messages.html",
        {"request": request, "history": history, "now": datetime.now}
    )


@router.post("/chat/send", response_class=HTMLResponse, tags=["Chat"])
async def chat_send(request: Request, message: str = Form(...)):
    """Echo-back mock implementation. Adds user msg + bot reply to history."""
    _, history = _ensure_session(request)
    current_time = datetime.now().strftime('%H:%M')
    cfg = get_chat_config(request)
    history.append({"role": "user", "text": message, "timestamp": current_time})
    start_time = datetime.now()

    flock_inst = getattr(request.app.state, "flock_instance", None)
    bot_agent = cfg.agent_name if cfg.agent_name else None
    bot_text: str
    if bot_agent and flock_inst and bot_agent in getattr(flock_inst, "agents", {}):
        # Build input according to mapping keys
        run_input: dict = {}
        if cfg.message_key:
            run_input[cfg.message_key] = message
        if cfg.history_key:
            # Provide history without timestamps to keep things small
            run_input[cfg.history_key] = [h["text"] for h in history]

        try:
            result_dict = await flock_inst.run_async(start_agent=bot_agent, input=run_input, box_result=False)
        except Exception as e:
            bot_text = f"Error: {e}"
        else:
            if cfg.response_key:
                bot_text = str(result_dict.get(cfg.response_key, result_dict))
            else:
                bot_text = str(result_dict)
    else:
        # Fallback echo behavior
        bot_text = f"Echo: {message}"

    duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
    history.append({"role": "bot", "text": bot_text, "timestamp": current_time, "agent": bot_agent or "echo", "duration_ms": duration_ms})
    # Return updated history partial
    return templates.TemplateResponse(
        "partials/_chat_messages.html",
        {"request": request, "history": history, "now": datetime.now}
    )


@router.get("/ui/htmx/chat-view", response_class=HTMLResponse, tags=["Chat"], include_in_schema=False)
async def chat_container_partial(request: Request):
    _ensure_session(request)
    return templates.TemplateResponse("partials/_chat_container.html", {"request": request})


# ---------------------------------------------------------------------------
# Chat settings management
# ---------------------------------------------------------------------------


@router.get("/ui/htmx/chat-settings-form", response_class=HTMLResponse, include_in_schema=False)
async def chat_settings_form(request: Request):
    """Returns the form for configuring chat behaviour (HTMX partial)."""
    cfg = get_chat_config(request)
    flock_inst = getattr(request.app.state, "flock_instance", None)
    input_fields, output_fields = [], []
    if cfg.agent_name and flock_inst and cfg.agent_name in flock_inst.agents:
        agent_obj = flock_inst.agents[cfg.agent_name]
        # Expect signatures like "field: type | desc, ..." or "field: type" etc.
        def _extract(sig: str):
            fields = []
            for seg in sig.split(','):
                parts = seg.strip().split(':')
                if parts:
                    fields.append(parts[0].strip())
            return [f for f in fields if f]
        input_fields = _extract(agent_obj.input) if getattr(agent_obj, 'input', '') else []
        output_fields = _extract(agent_obj.output) if getattr(agent_obj, 'output', '') else []

    context = get_base_context_web(request)
    context.update({
        "chat_cfg": cfg,
        "current_flock": flock_inst,
        "input_fields": input_fields,
        "output_fields": output_fields,
    })
    return templates.TemplateResponse("partials/_chat_settings_form.html", context)


@router.post("/chat/settings", response_class=HTMLResponse, include_in_schema=False)
async def chat_settings_submit(
    request: Request,
    agent_name: str | None = Form(default=None),
    message_key: str = Form("message"),
    history_key: str = Form("history"),
    response_key: str = Form("response"),
):
    """Apply submitted chat config, then re-render the form with a success message."""
    cfg = get_chat_config(request)
    cfg.agent_name = agent_name or None
    cfg.message_key = message_key
    cfg.history_key = history_key
    cfg.response_key = response_key

    headers = {
        "HX-Trigger": json.dumps({"notify": {"type": "success", "message": "Chat settings saved"}}),
        "HX-Redirect": "/chat"
    }
    # Response body empty; HTMX will redirect
    return HTMLResponse("", headers=headers)


# --- Stand-alone Chat HTML page access to settings --------------------------


@router.get("/chat/settings-standalone", response_class=HTMLResponse, tags=["Chat"], include_in_schema=False)
async def chat_settings_standalone(request: Request):
    """Standalone page to render chat settings (used by full-page chat HTML)."""
    cfg = get_chat_config(request)
    context = get_base_context_web(request, ui_mode="standalone")
    context.update({
        "chat_cfg": cfg,
        "current_flock": getattr(request.app.state, "flock_instance", None),
    })
    return templates.TemplateResponse("chat_settings.html", context)


# ---------------------------------------------------------------------------
# Stand-alone HTMX partials (chat view & settings) for in-page swapping
# ---------------------------------------------------------------------------


@router.get("/chat/htmx/chat-view", response_class=HTMLResponse, include_in_schema=False)
async def htmx_chat_view(request: Request):
    """Return chat container partial for standalone page reload via HTMX."""
    _ensure_session(request)
    return templates.TemplateResponse("partials/_chat_container.html", {"request": request})


@router.get("/chat/htmx/settings-form", response_class=HTMLResponse, include_in_schema=False)
async def htmx_chat_settings_partial(request: Request):
    cfg = get_chat_config(request)
    # Allow temporarily selecting agent via query param without saving
    agent_override = request.query_params.get("agent_name")
    if agent_override is not None:
        cfg = cfg.copy()
        cfg.agent_name = agent_override or None

    flock_inst = getattr(request.app.state, "flock_instance", None)
    input_fields, output_fields = [], []
    if cfg.agent_name and flock_inst and cfg.agent_name in flock_inst.agents:
        agent_obj = flock_inst.agents[cfg.agent_name]
        def _extract(sig: str):
            return [seg.strip().split(':')[0].strip() for seg in sig.split(',') if seg.strip()]
        input_fields = _extract(agent_obj.input) if getattr(agent_obj, 'input', '') else []
        output_fields = _extract(agent_obj.output) if getattr(agent_obj, 'output', '') else []

    context = {"request": request, "chat_cfg": cfg, "current_flock": flock_inst, "input_fields": input_fields, "output_fields": output_fields}
    return templates.TemplateResponse("partials/_chat_settings_form.html", context)
