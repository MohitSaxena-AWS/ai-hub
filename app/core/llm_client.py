"""LLM client abstraction for the conversation engine.

The engine never talks to Anthropic directly; it depends on the small
``LLMClient`` interface defined here. This keeps Anthropic-specific details in
one place and lets tests inject a deterministic ``MockLLMClient`` (no network,
no API key).

Each turn the model is asked to do two things at once via a single forced
tool call:

* produce the natural-language ``assistant_message`` shown to the requester, and
* report the structured ``collected_fields`` it has gathered so far plus an
  ``is_complete`` flag.

Forcing a tool call gives us reliable structured output alongside the chat
reply, which is exactly what a guided, slot-filling conversation needs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    RateLimitError,
)

from app.config import FieldDef

logger = logging.getLogger(__name__)

# Name of the tool the model is forced to call each turn.
RESPOND_TOOL = "respond"

# Shown if a turn yields no natural-language reply (e.g. the model omitted
# ``assistant_message`` or, defensively, returned no tool call), so the requester
# never receives an empty message and can simply restate their input.
_EMPTY_REPLY_FALLBACK = "Sorry, I didn't catch that — could you say it again?"


class LLMError(RuntimeError):
    """Raised when the LLM turn cannot be completed (transient or fatal).

    The conversation engine catches this and returns a graceful retry message to
    the requester instead of crashing the session, so an upstream outage or rate
    limit never corrupts conversational state.
    """


@dataclass
class EngineTurn:
    """The structured result of one assistant turn."""

    assistant_message: str
    collected_fields: dict[str, Any] = field(default_factory=dict)
    is_complete: bool = False


class LLMClient(ABC):
    """Interface the conversation engine depends on."""

    @abstractmethod
    async def run_turn(
        self,
        base_system_prompt: str,
        history: list[tuple[str, str]],
        fields: list[FieldDef],
        collected: dict[str, Any],
    ) -> EngineTurn:
        """Run one assistant turn given the conversation so far.

        ``history`` is an ordered list of ``(role, content)`` tuples where role
        is ``"user"`` or ``"assistant"``. ``collected`` is the state gathered in
        previous turns. Implementations return the assistant's reply plus the
        updated structured view.
        """


def build_respond_tool(fields: list[FieldDef]) -> dict[str, Any]:
    """Build the Anthropic tool schema for reporting collected fields.

    Every field is optional because information arrives incrementally over the
    conversation. Enum fields constrain the model to the allowed values.
    """

    properties: dict[str, Any] = {
        "assistant_message": {
            "type": "string",
            "description": "The natural-language reply to show the requester.",
        },
        "is_complete": {
            "type": "boolean",
            "description": (
                "True only when every required field has been collected and "
                "confirmed with the requester."
            ),
        },
    }
    for f in fields:
        prop: dict[str, Any] = {"type": "string", "description": f.description}
        if f.enum:
            prop["enum"] = f.enum
        properties[f.name] = prop

    return {
        "name": RESPOND_TOOL,
        "description": (
            "Reply to the requester and report all request fields collected so "
            "far. Always call this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": ["assistant_message", "is_complete"],
        },
    }


def _build_system_prompt(
    base_system_prompt: str,
    fields: list[FieldDef],
    collected: dict[str, Any],
) -> str:
    """Augment the configured prompt with mechanics the model needs each turn."""

    lines = [base_system_prompt, "", "Fields to collect (ask for any that are still missing):"]
    for f in fields:
        constraint = f" (one of: {', '.join(f.enum)})" if f.enum else ""
        lines.append(f"- {f.name}: {f.description}{constraint}")
    lines += [
        "",
        "Already collected so far (do not ask again for these):",
        json.dumps(collected, ensure_ascii=False) if collected else "(nothing yet)",
        "",
        f"Always respond by calling the `{RESPOND_TOOL}` tool. Include every "
        "field value you currently know (carry forward previously collected "
        "ones) and set is_complete to true only once all fields are gathered.",
    ]
    return "\n".join(lines)


class AnthropicLLMClient(LLMClient):
    """Real client backed by the Anthropic Claude API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_tokens: int = 1024,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.5,
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._backoff_base = backoff_base_seconds

    async def run_turn(
        self,
        base_system_prompt: str,
        history: list[tuple[str, str]],
        fields: list[FieldDef],
        collected: dict[str, Any],
    ) -> EngineTurn:
        system = _build_system_prompt(base_system_prompt, fields, collected)
        tool = build_respond_tool(fields)

        # The Anthropic API requires the message list to start with a user turn,
        # so drop the assistant's opening message(s) — the system prompt already
        # carries the conversational intent.
        messages: list[dict[str, str]] = []
        for role, content in history:
            if not messages and role != "user":
                continue
            messages.append({"role": role, "content": content})

        response = await self._create_with_retries(system, messages, tool)
        return self._parse(response, fields)

    async def _create_with_retries(
        self,
        system: str,
        messages: list[dict[str, str]],
        tool: dict[str, Any],
    ) -> Any:
        """Call the Anthropic API, retrying transient failures with backoff.

        Transient errors (connection drops, rate limits, 5xx) are retried with
        exponential backoff. Client errors (4xx other than 429) are not retried —
        they indicate a request problem that a retry won't fix. Either way, a
        failure is surfaced as ``LLMError`` so the engine can degrade gracefully.
        """

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system,
                    messages=messages,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": RESPOND_TOOL},
                )
            except (APIConnectionError, RateLimitError) as exc:
                last_exc = exc
            except APIStatusError as exc:
                # Don't retry deterministic client errors (bad request, auth);
                # only server-side (5xx) failures are worth another attempt.
                if exc.status_code < 500:
                    raise LLMError(f"LLM request rejected ({exc.status_code})") from exc
                last_exc = exc

            if attempt < self._max_retries:
                delay = self._backoff_base * (2**attempt)
                logger.warning(
                    "Anthropic call failed (attempt %d/%d); retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    delay,
                    last_exc,
                )
                await asyncio.sleep(delay)

        raise LLMError("LLM request failed after retries") from last_exc

    @staticmethod
    def _parse(response: Any, fields: list[FieldDef]) -> EngineTurn:
        """Extract the forced tool call into an ``EngineTurn``."""

        tool_input: dict[str, Any] = {}
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == RESPOND_TOOL:
                tool_input = block.input
                break

        field_names = {f.name for f in fields}
        collected = {
            k: v for k, v in tool_input.items() if k in field_names and v not in (None, "")
        }
        return EngineTurn(
            assistant_message=tool_input.get("assistant_message") or _EMPTY_REPLY_FALLBACK,
            collected_fields=collected,
            is_complete=bool(tool_input.get("is_complete", False)),
        )
