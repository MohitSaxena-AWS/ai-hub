"""Domain models for a conversation session and the API request/response shapes.

A *session* holds the full state of one conversation: its status, the message
history exchanged with the requester, and the fields collected so far. It lives
for the entire duration of the conversation (assignment section 2.2).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionStatus(str, Enum):
    """Lifecycle of a conversation."""

    COLLECTING = "collecting"
    # All fields gathered, a possible duplicate was found, awaiting the
    # requester's confirmation on whether to update the existing request.
    AWAITING_DUPLICATE_CONFIRMATION = "awaiting_duplicate_confirmation"
    COMPLETED = "completed"


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str
    created_at: datetime = Field(default_factory=_now)


class Session(BaseModel):
    """Persisted conversation state."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    # Principal (JWT ``sub``) that created the session; only they may access it.
    # None when authentication is disabled.
    owner: str | None = None
    status: SessionStatus = SessionStatus.COLLECTING
    messages: list[Message] = Field(default_factory=list)
    # Fields collected so far, keyed by field name from the prompt config.
    collected_fields: dict[str, Any] = Field(default_factory=dict)
    # Fingerprint of the prompt config that drove this conversation.
    prompt_fingerprint: str = ""
    # Set once the request has been persisted; links back to the stored row.
    request_id: str | None = None
    # While awaiting duplicate confirmation, the id of the existing request the
    # new one might update.
    pending_duplicate_id: str | None = None
    # How many times we've re-asked the yes/no duplicate question after an
    # ambiguous answer. Bounded so the conversation can't get stuck looping on
    # an unparseable reply (see ConversationEngine._handle_confirmation).
    duplicate_clarify_count: int = 0
    # Optimistic-concurrency token: bumped on every save and checked against the
    # stored document, so two messages racing on the same session can't silently
    # clobber each other (see SessionsRepository.save).
    version: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def to_mongo(self) -> dict[str, Any]:
        doc = self.model_dump(mode="python")
        doc["_id"] = doc.pop("id")
        return doc

    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "Session":
        data = dict(doc)
        data["id"] = data.pop("_id")
        return cls(**data)


# ----- API request / response models -------------------------------------------------


class CreateSessionResponse(BaseModel):
    session_id: str
    status: SessionStatus
    # The assistant's opening message.
    message: str


class PostMessageRequest(BaseModel):
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """Reject whitespace-only messages (``min_length`` alone admits "   ").

        A blank turn carries no information yet would still cost an LLM call, so
        it is rejected at the boundary with a 422 like any other invalid input.
        The trimmed value is stored so leading/trailing noise never reaches the
        model or the transcript.
        """

        trimmed = value.strip()
        if not trimmed:
            raise ValueError("message must not be blank")
        return trimmed


class PostMessageResponse(BaseModel):
    session_id: str
    status: SessionStatus
    # The assistant's reply.
    message: str
    # Populated once the conversation completes and a request is stored.
    request_id: str | None = None


class SessionView(BaseModel):
    """Read-only projection returned by GET /sessions/{id}."""

    session_id: str
    status: SessionStatus
    messages: list[Message]
    collected_fields: dict[str, Any]
    request_id: str | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_session(cls, s: Session) -> "SessionView":
        return cls(
            session_id=s.id,
            status=s.status,
            messages=s.messages,
            collected_fields=s.collected_fields,
            request_id=s.request_id,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
