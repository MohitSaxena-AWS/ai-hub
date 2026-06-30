"""Conversational REST endpoints (assignment sections 2.1 and 2.2).

* ``POST /sessions``               — start a conversation, get the opening message.
* ``POST /sessions/{id}/messages`` — send a user turn, get the assistant's reply.
* ``GET  /sessions/{id}``          — inspect session state (useful for demos/tests).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.auth import Principal, authorize_session_owner, get_principal
from app.api.dependencies import get_engine, get_sessions_repo
from app.config import get_settings
from app.core.conversation_engine import ConversationEngine
from app.db.sessions_repo import ConcurrentUpdateError, SessionGoneError, SessionsRepository
from app.models.session import (
    CreateSessionResponse,
    PostMessageRequest,
    PostMessageResponse,
    Role,
    Session,
    SessionStatus,
    SessionView,
)

router = APIRouter(tags=["conversation"])


@router.post("/sessions", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    repo: SessionsRepository = Depends(get_sessions_repo),
    engine: ConversationEngine = Depends(get_engine),
    principal: Principal | None = Depends(get_principal),
) -> CreateSessionResponse:
    """Create a new conversation session and return the assistant's first message."""

    session = Session(owner=principal.subject if principal else None)
    await engine.open(session)
    await repo.create(session)
    return CreateSessionResponse(
        session_id=session.id,
        status=session.status,
        message=session.messages[-1].content,
    )


@router.post("/sessions/{session_id}/messages", response_model=PostMessageResponse)
async def post_message(
    session_id: str,
    body: PostMessageRequest,
    repo: SessionsRepository = Depends(get_sessions_repo),
    engine: ConversationEngine = Depends(get_engine),
    principal: Principal | None = Depends(get_principal),
) -> PostMessageResponse:
    """Send a user message into an ongoing session and get the assistant's reply."""

    settings = get_settings()

    # Bound the input size before any work (LLM cost / abuse guard).
    if len(body.message) > settings.max_user_message_chars:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Message is too long.",
        )

    session = await repo.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    authorize_session_owner(session.owner, principal)

    if session.status == SessionStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session is already completed",
        )

    # Cap the number of user turns: every turn is an LLM call, so an unbounded
    # conversation is a cost/abuse vector. Reject once the cap is reached.
    user_turns = sum(1 for m in session.messages if m.role == Role.USER)
    if user_turns >= settings.max_session_turns:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session has too many turns; please start a new one.",
        )

    actor = principal.subject if principal else None
    reply = await engine.handle_message(session, body.message, actor=actor)
    # The engine mutates the session in place; persist the full updated state.
    # A version-checked write rejects concurrent turns on the same session.
    try:
        await repo.save(session)
    except SessionGoneError:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Session has expired; please start a new one.",
        ) from None
    except ConcurrentUpdateError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session was updated concurrently; please retry.",
        ) from None

    return PostMessageResponse(
        session_id=session.id,
        status=session.status,
        message=reply,
        request_id=session.request_id,
    )


@router.get("/sessions/{session_id}", response_model=SessionView)
async def get_session(
    session_id: str,
    repo: SessionsRepository = Depends(get_sessions_repo),
    principal: Principal | None = Depends(get_principal),
) -> SessionView:
    session = await repo.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    authorize_session_owner(session.owner, principal)
    return SessionView.from_session(session)
