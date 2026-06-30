"""Config-driven conversational text (assignment section 3).

Proves the user-facing wording follows ``prompt.yaml`` rather than being
hard-coded in the engine: swapping the prompt for a different domain changes the
greeting and the duplicate-flow messages, not just the fields collected.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.config import FieldDef, PromptConfig
from app.core.conversation_engine import ConversationEngine
from app.core.mock_llm import MockLLMClient
from app.models.session import Session, SessionStatus


def _hr_config() -> PromptConfig:
    """A prompt for a completely different domain than the engineering desk."""

    return PromptConfig(
        system_prompt="You are an HR onboarding assistant.",
        fields=[
            FieldDef(name="department", description="Target department.", business=True),
        ],
        opening_message="Welcome to HR onboarding! Let's get you set up.",
        duplicate_prompt_message="A matching onboarding already exists. Update it? (yes/no)",
        duplicate_clarify_message="Reply 'yes' to update or 'no' to start fresh.",
        duplicate_updated_message="Your onboarding has been updated.",
        completion_message="Onboarding recorded. Goodbye.",
    )


def test_opening_message_comes_from_config():
    engine = ConversationEngine(_hr_config(), MockLLMClient(), None, None)
    opening = engine.opening_message()

    # The configured greeting is used verbatim, with a question for the first
    # field appended by the engine — no "engineering service desk" wording.
    assert opening.startswith("Welcome to HR onboarding! Let's get you set up.")
    assert "engineering" not in opening.lower()
    assert "department" in opening.lower()


@dataclass
class _StubDuplicate:
    id: str = "existing-123"


class _AlwaysDuplicateService:
    """Stub dedup service that reports every completed request as a duplicate."""

    async def find_duplicate(self, collected: dict):
        return _StubDuplicate()


@pytest.mark.asyncio
async def test_duplicate_flow_messages_come_from_config():
    config = _hr_config()
    engine = ConversationEngine(
        config, MockLLMClient(), requests_repo=None, duplicate_service=_AlwaysDuplicateService()
    )
    session = Session()

    # One message fills the single field; the engine then runs dedup, finds the
    # stubbed duplicate, and must ask using the configured wording.
    reply = await engine.handle_message(session, "payments")
    assert session.status == SessionStatus.AWAITING_DUPLICATE_CONFIRMATION
    assert reply == config.duplicate_prompt_message

    # An ambiguous confirmation answer re-asks using the configured clarify text.
    clarify = await engine.handle_message(session, "maybe")
    assert clarify == config.duplicate_clarify_message
