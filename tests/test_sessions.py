"""Smoke tests for the session lifecycle and REST surface (stages 1-2)."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    # The public probe must not leak configuration detail (prompt fingerprint,
    # schema version, effective backend) — those are logged at startup instead.
    assert "prompt_fingerprint" not in body
    assert "schema_version" not in body
    assert "mock_llm" not in body


@pytest.mark.asyncio
async def test_healthz_echoes_correlation_id(client):
    resp = await client.get("/healthz", headers={"X-Request-ID": "trace-123"})
    assert resp.headers["X-Request-ID"] == "trace-123"


@pytest.mark.asyncio
async def test_blank_message_is_rejected(client):
    created = (await client.post("/sessions")).json()
    sid = created["session_id"]
    resp = await client.post(f"/sessions/{sid}/messages", json={"message": "   "})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_session_returns_opening_message(client):
    resp = await client.post("/sessions")
    assert resp.status_code == 201
    body = resp.json()
    assert body["session_id"]
    assert body["status"] == "collecting"
    assert body["message"]


@pytest.mark.asyncio
async def test_post_message_roundtrip(client):
    created = (await client.post("/sessions")).json()
    sid = created["session_id"]

    resp = await client.post(f"/sessions/{sid}/messages", json={"message": "infrastructure-provisioning"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == sid
    assert body["status"] == "collecting"
    assert body["message"]


@pytest.mark.asyncio
async def test_get_session_includes_history(client):
    sid = (await client.post("/sessions")).json()["session_id"]
    await client.post(f"/sessions/{sid}/messages", json={"message": "hello"})

    resp = await client.get(f"/sessions/{sid}")
    assert resp.status_code == 200
    body = resp.json()
    # opening (assistant) + user turn + assistant reply = 3 messages
    assert len(body["messages"]) == 3
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["assistant", "user", "assistant"]


@pytest.mark.asyncio
async def test_unknown_session_returns_404(client):
    resp = await client.post("/sessions/does-not-exist/messages", json={"message": "hi"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_empty_message_rejected(client):
    sid = (await client.post("/sessions")).json()["session_id"]
    resp = await client.post(f"/sessions/{sid}/messages", json={"message": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_oversized_message_rejected(client):
    """Cost/abuse guard: over-long messages are rejected before reaching the LLM."""

    from app.config import get_settings

    sid = (await client.post("/sessions")).json()["session_id"]
    too_long = "x" * (get_settings().max_user_message_chars + 1)
    resp = await client.post(f"/sessions/{sid}/messages", json={"message": too_long})
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_turn_limit_rejects_further_messages(client, monkeypatch):
    """Cost/abuse guard: a session can't accept unbounded LLM-backed turns."""

    from app.config import get_settings

    # Shrink the cap so the test stays fast (the setting is a cached singleton).
    monkeypatch.setattr(get_settings(), "max_session_turns", 2)

    sid = (await client.post("/sessions")).json()["session_id"]

    assert (
        await client.post(f"/sessions/{sid}/messages", json={"message": "one"})
    ).status_code == 200
    assert (
        await client.post(f"/sessions/{sid}/messages", json={"message": "two"})
    ).status_code == 200

    # The third user turn would exceed the cap of 2 -> rejected.
    resp = await client.post(f"/sessions/{sid}/messages", json={"message": "three"})
    assert resp.status_code == 409
    assert "too many turns" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_stale_session_write_is_rejected(client):
    """Optimistic concurrency: a write against a stale version is rejected."""

    from app.db import mongo
    from app.db.sessions_repo import ConcurrentUpdateError, SessionsRepository
    from app.models.session import Session

    repo = SessionsRepository(mongo.get_db())
    await repo.create(Session(id="race"))

    # Two handlers each read the same version of the session.
    first = await repo.get("race")
    second = await repo.get("race")

    await repo.save(first)  # wins, bumps the stored version
    with pytest.raises(ConcurrentUpdateError):
        await repo.save(second)  # stale version -> no match -> rejected


@pytest.mark.asyncio
async def test_expired_session_write_raises_gone(client):
    """A vanished session (e.g. TTL-reaped) is reported distinctly from a conflict."""

    from app.db import mongo
    from app.db.sessions_repo import SessionGoneError, SessionsRepository
    from app.models.session import Session

    repo = SessionsRepository(mongo.get_db())
    await repo.create(Session(id="ttl"))
    loaded = await repo.get("ttl")

    # Simulate the TTL index reaping the session between read and save.
    await mongo.get_db()["sessions"].delete_one({"_id": "ttl"})

    with pytest.raises(SessionGoneError):
        await repo.save(loaded)
