"""Conversation engine.

Given a session and a new user message, the engine drives the conversation
through the configured ``LLMClient``, merges the structured fields the model
extracts into the session, and — once everything is collected — runs
privacy-preserving duplicate detection before persisting the request.

Flow once all fields are gathered:
    collected -> duplicate check
        no duplicate            -> persist new request, close session
        duplicate found         -> ask "update existing request?" and wait
            user says yes        -> update the existing request, close session
            user says no         -> persist a new request, close session

The public contract (``open`` / ``handle_message``) is stable; the API layer is
unaffected by this internal flow.
"""

from __future__ import annotations

import logging

from app.config import PromptConfig
from app.core.audit import record_persisted
from app.core.duplicate_service import DuplicateService
from app.core.llm_client import LLMClient, LLMError
from app.db.requests_repo import RequestsRepository
from app.models.request import RequestRecord
from app.models.session import Message, Role, Session, SessionStatus

logger = logging.getLogger(__name__)

_AFFIRMATIVE = {"yes", "y", "yeah", "yep", "update", "confirm", "ok", "okay", "sure"}
_NEGATIVE = {"no", "n", "nope", "new", "separate", "keep", "different"}

# Fallback shown to the requester when the LLM backend is temporarily
# unavailable. The session stays open and no state is mutated, so the requester
# can simply resend their message once the backend recovers.
_LLM_UNAVAILABLE_REPLY = (
    "Sorry, I'm having trouble processing that right now. "
    "Please send your message again in a moment."
)

# Default number of most-recent conversation messages forwarded to the LLM each
# turn. The full transcript is always persisted; only the prompt window is
# bounded, which caps token cost (and latency) on long conversations without
# losing slot-filling context — already-collected fields are re-sent separately.
_DEFAULT_HISTORY_WINDOW = 20

# How many times the assistant re-asks the yes/no duplicate question on an
# ambiguous answer before giving up and taking the safe, non-destructive default
# (create a new request rather than risk overwriting the existing one). This
# guarantees the conversation always terminates instead of looping forever.
_MAX_DUPLICATE_CLARIFY = 3


def _question_for(field) -> str:
    """The question to ask for a field when the engine prompts for it directly.

    Prefers the config-authored ``question`` (so a different-domain prompt reads
    naturally); otherwise falls back to a phrase generated from the field's name /
    enum / description, so an older prompt without ``question`` still works.
    """

    if field.question:
        return field.question
    if field.enum:
        return f"To start, what is the {field.name}? One of: {', '.join(field.enum)}."
    return f"To start, {field.description.lower()}"


def parse_confirmation(text: str) -> bool | None:
    """Interpret a yes/no confirmation. Returns None if absent or ambiguous.

    If both affirmative and negative words appear (e.g. "no, don't update"), the
    answer is treated as ambiguous rather than guessing, so we re-ask instead of
    risking the wrong action.
    """

    tokens = set(text.strip().lower().replace(",", " ").replace(".", " ").split())
    has_yes = bool(tokens & _AFFIRMATIVE)
    has_no = bool(tokens & _NEGATIVE)
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None


class ConversationEngine:
    def __init__(
        self,
        prompt_config: PromptConfig,
        llm: LLMClient,
        requests_repo: RequestsRepository,
        duplicate_service: DuplicateService,
        history_window: int = _DEFAULT_HISTORY_WINDOW,
    ) -> None:
        self._config = prompt_config
        self._llm = llm
        self._requests = requests_repo
        self._dedup = duplicate_service
        self._history_window = history_window

    def opening_message(self) -> str:
        """The assistant's first message when a session is created.

        The greeting itself is configurable (``opening_message`` in the prompt
        config) so the experience follows the prompt's domain; the engine only
        appends a question for the first field, derived from the field definition.
        """

        first = self._config.fields[0] if self._config.fields else None
        if first is None:
            return self._config.opening_message
        return self._config.opening_message + " " + _question_for(first)

    async def open(self, session: Session) -> None:
        """Initialise a freshly created session with the opening message."""

        session.prompt_fingerprint = self._config.fingerprint
        session.messages.append(Message(role=Role.ASSISTANT, content=self.opening_message()))

    async def handle_message(
        self, session: Session, user_text: str, actor: str | None = None
    ) -> str:
        """Process a user message and return the assistant's reply.

        ``actor`` is the authenticated principal driving the conversation; it is
        used only for the PII-free audit trail when a request is persisted, never
        for business logic. Completed sessions are rejected at the API boundary
        (HTTP 409), so this method only ever sees active or awaiting-confirmation
        sessions.
        """

        # The requester is answering the "update existing request?" question.
        if session.status == SessionStatus.AWAITING_DUPLICATE_CONFIRMATION:
            return await self._handle_confirmation(session, user_text, actor)

        session.messages.append(Message(role=Role.USER, content=user_text))

        # Only forward the most recent slice of the transcript to bound token
        # cost; the full history stays in the session for the audit trail.
        recent = session.messages[-self._history_window :]
        history = [(m.role.value, m.content) for m in recent]
        try:
            turn = await self._llm.run_turn(
                base_system_prompt=self._config.system_prompt,
                history=history,
                fields=self._config.fields,
                collected=session.collected_fields,
            )
        except LLMError:
            # Leave the session untouched (status, collected fields) so the
            # requester can simply retry once the backend recovers.
            logger.exception("LLM turn failed for session %s", session.id)
            session.messages.append(
                Message(role=Role.ASSISTANT, content=_LLM_UNAVAILABLE_REPLY)
            )
            return _LLM_UNAVAILABLE_REPLY
        session.collected_fields.update(self._sanitize(turn.collected_fields))

        # Completion is gated on BOTH the model's ``is_complete`` and our own
        # check that every required field is present. The server-side check
        # guards against the model declaring completion prematurely; conversely
        # we still wait for ``is_complete`` so the assistant can run its final
        # "summarise and confirm" turn (per the prompt) before the session is
        # closed, rather than terminating the instant the last slot is filled.
        if turn.is_complete and self._all_required_present(session.collected_fields):
            reply = await self._on_all_collected(session, actor)
        else:
            reply = turn.assistant_message

        session.messages.append(Message(role=Role.ASSISTANT, content=reply))
        return reply

    def _sanitize(self, collected: dict) -> dict:
        """Drop enum values that are not in the configured allowed set.

        The Anthropic tool schema already constrains enum fields at the API
        level, but this is a defensive server-side net: an out-of-vocabulary
        value is ignored (the field stays "missing" and gets re-asked) so it can
        never be persisted, regardless of which LLM backend produced it.
        """

        choices = self._config.enum_choices
        return {
            name: value
            for name, value in collected.items()
            if name not in choices or value in choices[name]
        }

    def _all_required_present(self, collected: dict) -> bool:
        """Guard against the model claiming completion prematurely."""

        return all(collected.get(name) for name in self._config.required_field_names)

    async def _on_all_collected(self, session: Session, actor: str | None) -> str:
        """Run duplicate detection, then either ask to confirm or persist."""

        duplicate = await self._dedup.find_duplicate(session.collected_fields)
        if duplicate is not None:
            session.pending_duplicate_id = duplicate.id
            session.status = SessionStatus.AWAITING_DUPLICATE_CONFIRMATION
            # Privacy: the configured text must not reveal anything about the
            # other request/requester (assignment section 2.3).
            return self._config.duplicate_prompt_message
        return await self._persist_new(session, actor)

    async def _handle_confirmation(
        self, session: Session, user_text: str, actor: str | None
    ) -> str:
        """Resolve the duplicate-confirmation question."""

        session.messages.append(Message(role=Role.USER, content=user_text))
        decision = parse_confirmation(user_text)

        if decision is None:
            session.duplicate_clarify_count += 1
            if session.duplicate_clarify_count >= _MAX_DUPLICATE_CLARIFY:
                # Repeated unparseable answers: stop looping and take the safe,
                # non-destructive default — create a new request rather than risk
                # overwriting the existing one — so the session always terminates.
                reply = await self._persist_new(session, actor)
            else:
                reply = self._config.duplicate_clarify_message
        elif decision:
            reply = await self._update_existing(session, actor)
        else:
            reply = await self._persist_new(session, actor)

        session.messages.append(Message(role=Role.ASSISTANT, content=reply))
        return reply

    async def _persist_new(self, session: Session, actor: str | None) -> str:
        """Store a new request and close the session."""

        record = RequestRecord.from_session(session, self._config)
        await self._requests.create(record)
        session.request_id = record.id
        session.pending_duplicate_id = None
        session.status = SessionStatus.COMPLETED
        record_persisted(
            "created",
            request_id=record.id,
            actor=actor,
            session_id=session.id,
            schema_version=record.schema_version,
            prompt_fingerprint=record.prompt_fingerprint,
        )
        return self._config.completion_message

    async def _update_existing(self, session: Session, actor: str | None) -> str:
        """Update the existing duplicate request and close the session."""

        record = RequestRecord.from_session(session, self._config)
        await self._requests.update_data(
            session.pending_duplicate_id,
            data=record.data,
            pii=record.pii,
            business_hash=record.business_hash,
            # Stamp the updated record with the prompt that produced the new
            # payload, so provenance always matches the stored data shape.
            schema_version=record.schema_version,
            prompt_fingerprint=record.prompt_fingerprint,
        )
        updated_id = session.pending_duplicate_id
        session.request_id = updated_id
        session.pending_duplicate_id = None
        session.status = SessionStatus.COMPLETED
        record_persisted(
            "updated",
            request_id=updated_id,
            actor=actor,
            session_id=session.id,
            schema_version=record.schema_version,
            prompt_fingerprint=record.prompt_fingerprint,
        )
        return self._config.duplicate_updated_message + " " + self._config.completion_message
