"""End-to-end test: drives the packaged stack over real HTTP.

Unlike the in-process unit tests, this runs against the actual Docker image and
a real MongoDB brought up by docker-compose, hitting the service over HTTP at
``E2E_BASE_URL`` (default http://localhost:8000). It is invoked by the build
script after the stack is healthy. The LLM and similarity backends are pinned to
their deterministic offline variants (mock LLM + lexical) via compose, so the
run is fully reproducible.

One thorough scenario covers nearly every requirement in a single flow:
config-driven prompt -> session -> field collection -> completion -> persistence
-> privacy-preserving duplicate detection -> "update existing request?" flow.
"""

from __future__ import annotations

import os

import httpx
import pytest

BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")

# A full conversation answering each field the sample prompt asks for.
CONVERSATION = [
    "infrastructure-provisioning",
    "production",
    "Provision a new Kafka cluster for the payments platform",
    "Alice Engineer",
    "E12345",
]


async def _complete(client: httpx.AsyncClient) -> dict:
    """Run a full conversation to completion and return the final response."""

    created = (await client.post("/sessions")).json()
    sid = created["session_id"]
    last = None
    for text in CONVERSATION:
        resp = await client.post(f"/sessions/{sid}/messages", json={"message": text})
        assert resp.status_code == 200, resp.text
        last = resp.json()
    return {"session_id": sid, "last": last}


@pytest.mark.asyncio
async def test_full_flow_with_duplicate_update():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
        # Sanity: the service is up and the database is reachable.
        health = (await client.get("/healthz")).json()
        assert health["status"] == "ok"
        assert health["database"] == "ok"

        # 1) Happy path: collect all fields, persist a request, close the session.
        first = await _complete(client)
        assert first["last"]["status"] == "completed"
        first_request_id = first["last"]["request_id"]
        assert first_request_id

        state = (await client.get(f"/sessions/{first['session_id']}")).json()
        assert state["status"] == "completed"
        assert state["collected_fields"]["request_type"] == "infrastructure-provisioning"
        assert state["collected_fields"]["environment"] == "production"

        # 2) Identical request -> privacy-preserving duplicate detection kicks in
        #    and asks to confirm, without leaking the other requester's identity.
        dup = await _complete(client)
        assert dup["last"]["status"] == "awaiting_duplicate_confirmation"
        assert "alice" not in dup["last"]["message"].lower()
        assert "e12345" not in dup["last"]["message"].lower()

        # 3) Confirm update -> the existing request is updated, not duplicated.
        confirm = await client.post(
            f"/sessions/{dup['session_id']}/messages",
            json={"message": "yes, update it"},
        )
        body = confirm.json()
        assert body["status"] == "completed"
        assert body["request_id"] == first_request_id

        # 4) Decline the duplicate -> a *separate* request is created, so the
        #    requester is never forced to overwrite an existing one.
        dup2 = await _complete(client)
        assert dup2["last"]["status"] == "awaiting_duplicate_confirmation"
        decline = await client.post(
            f"/sessions/{dup2['session_id']}/messages",
            json={"message": "no, create a new one"},
        )
        declined = decline.json()
        assert declined["status"] == "completed"
        assert declined["request_id"]
        assert declined["request_id"] != first_request_id  # a new row, not the old one


@pytest.mark.asyncio
async def test_rejects_blank_and_oversized_messages():
    """Input-validation guards are enforced over real HTTP (assignment §7)."""

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
        sid = (await client.post("/sessions")).json()["session_id"]

        # Whitespace-only message -> 422 (rejected before any LLM call).
        blank = await client.post(f"/sessions/{sid}/messages", json={"message": "   "})
        assert blank.status_code == 422

        # Oversized message -> 413 (abuse / runaway-cost guard).
        huge = await client.post(
            f"/sessions/{sid}/messages", json={"message": "x" * 5000}
        )
        assert huge.status_code == 413
