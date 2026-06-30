"""Error-resilience tests for the LLM layer.

Cover the two failure paths added for robustness:

* the engine degrades gracefully (no state mutation) when a turn raises
  ``LLMError``, so an upstream outage never corrupts a conversation; and
* the Anthropic client retries transient failures with backoff and finally
  surfaces ``LLMError`` rather than leaking the raw SDK exception.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import httpx
import pytest
from anthropic import APIConnectionError

from app.config import load_prompt_config
from app.core.conversation_engine import _LLM_UNAVAILABLE_REPLY, ConversationEngine
from app.core.llm_client import (
    RESPOND_TOOL,
    AnthropicLLMClient,
    EngineTurn,
    LLMClient,
    LLMError,
    _EMPTY_REPLY_FALLBACK,
    build_respond_tool,
)
from app.models.session import Session, SessionStatus


def _config():
    return load_prompt_config(pathlib.Path("config/prompt.yaml"))


def _tool_use_response(tool_input: dict):
    """A fake Anthropic response carrying a single forced ``respond`` tool call."""

    block = SimpleNamespace(type="tool_use", name=RESPOND_TOOL, input=tool_input)
    return SimpleNamespace(content=[block])


class _FailingLLM(LLMClient):
    async def run_turn(self, *args, **kwargs) -> EngineTurn:  # noqa: D102
        raise LLMError("backend down")


@pytest.mark.asyncio
async def test_engine_degrades_gracefully_on_llm_error():
    engine = ConversationEngine(
        _config(), _FailingLLM(), requests_repo=None, duplicate_service=None
    )
    session = Session()

    reply = await engine.handle_message(session, "infrastructure-provisioning")

    # The requester gets a friendly retry message and the session is untouched.
    assert reply == _LLM_UNAVAILABLE_REPLY
    assert session.status == SessionStatus.COLLECTING
    assert session.collected_fields == {}
    # The user turn plus the fallback reply are recorded for the audit trail.
    assert session.messages[-1].content == _LLM_UNAVAILABLE_REPLY


class _AlwaysFailingMessages:
    """Stand-in for ``client.messages`` that always raises a transient error."""

    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        raise APIConnectionError(request=httpx.Request("POST", "http://test"))


@pytest.mark.asyncio
async def test_anthropic_client_retries_then_raises_llm_error():
    client = AnthropicLLMClient(
        api_key="x", model="m", max_retries=2, backoff_base_seconds=0
    )
    fake = _AlwaysFailingMessages()
    client._client.messages = fake  # type: ignore[attr-defined]

    with pytest.raises(LLMError):
        await client.run_turn(
            base_system_prompt="sys",
            history=[("user", "hi")],
            fields=_config().fields,
            collected={},
        )

    # Initial attempt + 2 retries.
    assert fake.calls == 3


# ----- tool schema & response parsing -------------------------------------------------


def test_build_respond_tool_injects_enums_and_required_fields():
    tool = build_respond_tool(_config().fields)
    props = tool["input_schema"]["properties"]

    # The two control fields are always present and required.
    assert tool["input_schema"]["required"] == ["assistant_message", "is_complete"]
    assert props["assistant_message"]["type"] == "string"
    assert props["is_complete"]["type"] == "boolean"

    # Enum fields from the config constrain the model to the allowed values...
    assert props["request_type"]["enum"] == [
        "infrastructure-provisioning",
        "service-deployment",
        "access-grant",
        "pipeline-change",
        "incident-fix",
    ]
    # ...while free-text fields carry no enum constraint.
    assert "enum" not in props["business_justification"]


def test_parse_extracts_message_fields_and_completion():
    turn = AnthropicLLMClient._parse(
        _tool_use_response({
            "assistant_message": "Got it, what environment?",
            "is_complete": False,
            "request_type": "access-grant",
            "environment": "",  # empty -> dropped
            "not_a_field": "ignored",  # unknown -> dropped
        }),
        _config().fields,
    )

    assert turn.assistant_message == "Got it, what environment?"
    assert turn.is_complete is False
    assert turn.collected_fields == {"request_type": "access-grant"}


def test_parse_falls_back_when_message_missing():
    turn = AnthropicLLMClient._parse(_tool_use_response({"is_complete": True}), _config().fields)
    assert turn.assistant_message == _EMPTY_REPLY_FALLBACK
    assert turn.is_complete is True


def test_parse_handles_response_without_tool_call():
    """Defensive: no tool_use block still yields a usable turn, not a crash."""

    empty = SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])
    turn = AnthropicLLMClient._parse(empty, _config().fields)
    assert turn.assistant_message == _EMPTY_REPLY_FALLBACK
    assert turn.collected_fields == {}
    assert turn.is_complete is False


class _CapturingMessages:
    """Stand-in for ``client.messages`` that records the call and returns a reply."""

    def __init__(self, response) -> None:
        self.response = response
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


@pytest.mark.asyncio
async def test_run_turn_forces_tool_call_and_starts_with_user_message():
    client = AnthropicLLMClient(api_key="x", model="m")
    fake = _CapturingMessages(
        _tool_use_response({"assistant_message": "Welcome", "is_complete": False})
    )
    client._client.messages = fake  # type: ignore[attr-defined]

    turn = await client.run_turn(
        base_system_prompt="sys",
        # The leading assistant opening must be dropped: the API requires the
        # message list to start with a user turn.
        history=[("assistant", "opening"), ("user", "hello")],
        fields=_config().fields,
        collected={},
    )

    assert turn.assistant_message == "Welcome"
    assert fake.kwargs["tool_choice"] == {"type": "tool", "name": RESPOND_TOOL}
    assert fake.kwargs["messages"][0]["role"] == "user"
    assert fake.kwargs["messages"] == [{"role": "user", "content": "hello"}]
