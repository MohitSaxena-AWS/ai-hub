"""Conversation-engine tests: slot filling and completion via the mock LLM."""

from __future__ import annotations

import pytest

# The mock assigns each user message to the next missing field, in config order:
# request_type -> environment -> business_justification -> requester_name -> employee_id.
CONVERSATION = [
    "infrastructure-provisioning",
    "development",
    "Need extra capacity for the Q3 launch",
    "Alice Engineer",
    "E12345",
]


async def _send(client, sid: str, text: str) -> dict:
    resp = await client.post(f"/sessions/{sid}/messages", json={"message": text})
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.asyncio
async def test_full_conversation_collects_all_fields_and_completes(client):
    sid = (await client.post("/sessions")).json()["session_id"]

    last = None
    for text in CONVERSATION:
        last = await _send(client, sid, text)

    # After the final field the engine finalizes and closes the session.
    assert last["status"] == "completed"
    assert last["message"]  # configured completion message

    state = (await client.get(f"/sessions/{sid}")).json()
    assert state["status"] == "completed"
    assert state["collected_fields"] == {
        "request_type": "infrastructure-provisioning",
        "environment": "development",
        "business_justification": "Need extra capacity for the Q3 launch",
        "requester_name": "Alice Engineer",
        "employee_id": "E12345",
    }


@pytest.mark.asyncio
async def test_session_stays_open_until_all_fields_collected(client):
    sid = (await client.post("/sessions")).json()["session_id"]

    # Provide only the first two fields; session must remain collecting.
    await _send(client, sid, "service-deployment")
    mid = await _send(client, sid, "staging")
    assert mid["status"] == "collecting"
    assert mid["request_id"] is None

    state = (await client.get(f"/sessions/{sid}")).json()
    assert state["collected_fields"]["request_type"] == "service-deployment"
    assert state["collected_fields"]["environment"] == "staging"
    assert "requester_name" not in state["collected_fields"]


@pytest.mark.asyncio
async def test_invalid_enum_value_is_dropped_not_persisted(client):
    """Out-of-vocabulary enum values are rejected server-side, not stored."""

    sid = (await client.post("/sessions")).json()["session_id"]
    reply = await _send(client, sid, "totally-not-a-valid-request-type")

    # The bogus value never lands in the collected fields, and the session keeps
    # collecting instead of advancing on garbage.
    state = (await client.get(f"/sessions/{sid}")).json()
    assert "request_type" not in state["collected_fields"]
    assert reply["status"] == "collecting"


@pytest.mark.asyncio
async def test_completed_session_rejects_further_messages(client):
    sid = (await client.post("/sessions")).json()["session_id"]
    for text in CONVERSATION:
        await _send(client, sid, text)

    # Once completed, the API guards against further messages.
    resp = await client.post(f"/sessions/{sid}/messages", json={"message": "hello again"})
    assert resp.status_code == 409
