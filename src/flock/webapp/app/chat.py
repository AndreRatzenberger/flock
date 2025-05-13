from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

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
# Routes
# ---------------------------------------------------------------------------


@router.get("/chat", response_class=HTMLResponse, tags=["Chat"])
async def chat_page(request: Request):
    """Full-page chat UI (works even when the main UI is disabled)."""
    sid, history = _ensure_session(request)
    context = get_base_context_web(request, ui_mode="standalone")
    context["history"] = history
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
    history.append({"role": "user", "text": message, "timestamp": current_time})
    history.append({"role": "bot", "text": f"Echo: {message}", "timestamp": current_time})
    # Return updated history partial
    return templates.TemplateResponse(
        "partials/_chat_messages.html", 
        {"request": request, "history": history, "now": datetime.now}
    )


@router.get("/ui/htmx/chat-view", response_class=HTMLResponse, tags=["Chat"], include_in_schema=False)
async def chat_container_partial(request: Request):
    _ensure_session(request)
    return templates.TemplateResponse("partials/_chat_container.html", {"request": request})
