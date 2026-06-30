"""Deterministic mock LLM for tests and offline builds.

It emulates a competent slot-filling assistant without any network calls: on
each turn it assigns the requester's latest message to the next still-missing
field (in configured order) and then asks for the following one. When every
field has a value it reports completion. This makes the full conversation flow
exercisable end-to-end in a reproducible way.
"""

from __future__ import annotations

from typing import Any

from app.config import FieldDef
from app.core.llm_client import EngineTurn, LLMClient


class MockLLMClient(LLMClient):
    async def run_turn(
        self,
        base_system_prompt: str,
        history: list[tuple[str, str]],
        fields: list[FieldDef],
        collected: dict[str, Any],
    ) -> EngineTurn:
        updated = dict(collected)

        # The most recent user message answers the field the assistant last
        # asked for, i.e. the first field that is currently missing.
        last_user_msg = next(
            (content for role, content in reversed(history) if role == "user"),
            None,
        )
        if last_user_msg is not None:
            next_missing = self._first_missing(fields, updated)
            if next_missing is not None:
                updated[next_missing.name] = last_user_msg.strip()

        # Decide what to ask next, or whether we're done.
        still_missing = self._first_missing(fields, updated)
        if still_missing is None:
            return EngineTurn(
                assistant_message="Thank you, I have everything I need.",
                collected_fields=updated,
                is_complete=True,
            )

        return EngineTurn(
            assistant_message=self._prompt_for(still_missing),
            collected_fields=updated,
            is_complete=False,
        )

    @staticmethod
    def _first_missing(fields: list[FieldDef], collected: dict[str, Any]) -> FieldDef | None:
        for f in fields:
            if not collected.get(f.name):
                return f
        return None

    @staticmethod
    def _prompt_for(field: FieldDef) -> str:
        if field.question:
            return field.question
        if field.enum:
            return f"What is the {field.name}? One of: {', '.join(field.enum)}."
        return f"What is the {field.name}? ({field.description})"
