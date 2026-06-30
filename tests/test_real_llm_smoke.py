"""Opt-in smoke test against the *real* Claude API.

Everything else in the suite runs offline against deterministic stubs, which is
the right default. But the production assistant is a real LLM, and the one thing
no stub can validate is that the configured prompt + forced tool schema actually
make a real model extract the right fields and converge. This test does — against
the live Anthropic API.

It is **skipped by default**. To run it (e.g. before submitting, with your key):

    RUN_LIVE_LLM=1 ANTHROPIC_API_KEY=sk-... pytest tests/test_real_llm_smoke.py -v -m live

It costs a few cheap Haiku calls. Because a real model is non-deterministic, the
assertions check robust invariants (valid enum value extracted, conversation
converges to completion, a request is persisted) rather than exact wording.
"""

from __future__ import annotations

import os
import pathlib

import pytest
from mongomock_motor import AsyncMongoMockClient

from app.config import get_settings, load_prompt_config
from app.core.conversation_engine import ConversationEngine
from app.core.duplicate_service import DuplicateService
from app.core.llm_client import AnthropicLLMClient
from app.core.similarity import LexicalSimilarity
from app.db.requests_repo import RequestsRepository
from app.models.session import Session, SessionStatus

_RUN = os.environ.get("RUN_LIVE_LLM") == "1" and bool(os.environ.get("ANTHROPIC_API_KEY"))

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _RUN,
        reason="set RUN_LIVE_LLM=1 and ANTHROPIC_API_KEY to run the live Claude smoke test",
    ),
]


def _config():
    return load_prompt_config(pathlib.Path("config/prompt.yaml"))


def _live_client() -> AnthropicLLMClient:
    settings = get_settings()
    return AnthropicLLMClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=settings.anthropic_model,
        max_tokens=settings.llm_max_tokens,
    )


@pytest.mark.asyncio
async def test_real_model_extracts_multiple_fields_in_one_turn():
    """A single natural sentence should populate request_type and environment."""

    config = _config()
    turn = await _live_client().run_turn(
        base_system_prompt=config.system_prompt,
        history=[("user", "I need infrastructure provisioning in the production environment")],
        fields=config.fields,
        collected={},
    )

    # The model must report a *valid* request_type and environment — the forced
    # tool schema constrains enums, and we re-check against the config anyway.
    assert turn.collected_fields.get("request_type") == "infrastructure-provisioning"
    assert turn.collected_fields.get("environment") == "production"
    assert turn.assistant_message  # a non-empty natural-language reply was produced


@pytest.mark.asyncio
async def test_real_model_drives_a_conversation_to_completion():
    """End-to-end with the real model: collect every field, persist one request."""

    config = _config()
    db = AsyncMongoMockClient()["live_smoke"]
    repo = RequestsRepository(db)
    engine = ConversationEngine(
        config,
        _live_client(),
        repo,
        DuplicateService(repo, LexicalSimilarity(), config),
    )

    session = Session()
    await engine.open(session)

    # Answer each topic; the model decides when it has everything and completes.
    for message in [
        "I'd like to submit an access-grant request",
        "staging environment",
        "The data team needs read access to the reporting bucket for Q3 analytics",
        "My name is Alice Engineer",
        "Employee ID E12345",
        "Yes, that all looks correct, please file it",
    ]:
        if session.status == SessionStatus.COMPLETED:
            break
        await engine.handle_message(session, message)

    assert session.status == SessionStatus.COMPLETED, (
        f"conversation did not converge; last collected={session.collected_fields}"
    )
    assert session.request_id is not None
    assert await repo.count() == 1

    stored = await repo.get(session.request_id)
    assert stored.data["request_type"] == "access-grant"
    assert stored.data["environment"] == "staging"
    # PII captured but stored apart from the business data.
    assert "requester_name" not in stored.data
    assert stored.pii  # name / employee id landed in the PII bucket
