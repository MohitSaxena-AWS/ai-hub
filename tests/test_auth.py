"""Authentication and per-session ownership tests.

The shared ``client`` fixture is authenticated as the default principal
(``alice``); these tests additionally exercise the unauthenticated path and a
second principal (``bob``) to prove sessions are isolated per owner.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from tests.conftest import auth_headers


def _anon_client() -> AsyncClient:
    """A client with no Authorization header (Mongo is set by the ``client`` fixture)."""

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_healthz_is_public(client):
    async with _anon_client() as anon:
        resp = await anon.get("/healthz")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_session_endpoints_require_a_token(client):
    # Create a session as the authenticated default principal first.
    sid = (await client.post("/sessions")).json()["session_id"]

    async with _anon_client() as anon:
        assert (await anon.post("/sessions")).status_code == 401
        assert (await anon.get(f"/sessions/{sid}")).status_code == 401
        msg = await anon.post(f"/sessions/{sid}/messages", json={"message": "hi"})
        assert msg.status_code == 401


@pytest.mark.asyncio
async def test_invalid_token_is_rejected(client):
    bad = {"Authorization": "Bearer not-a-real-jwt"}
    assert (await client.post("/sessions", headers=bad)).status_code == 401


@pytest.mark.asyncio
async def test_token_without_subject_is_rejected(client):
    import jwt

    from tests.conftest import TEST_AUTH_SECRET

    no_sub = jwt.encode({"role": "x"}, TEST_AUTH_SECRET, algorithm="HS256")
    resp = await client.post("/sessions", headers={"Authorization": f"Bearer {no_sub}"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_owner_can_access_own_session(client):
    sid = (await client.post("/sessions")).json()["session_id"]
    # Same principal (default token on the fixture) can read and post.
    assert (await client.get(f"/sessions/{sid}")).status_code == 200
    posted = await client.post(f"/sessions/{sid}/messages", json={"message": "hi"})
    assert posted.status_code == 200


@pytest.mark.asyncio
async def test_other_principal_cannot_access_session(client):
    # alice (fixture default) creates the session...
    sid = (await client.post("/sessions")).json()["session_id"]

    # ...bob must not be able to read it or post to it.
    bob = auth_headers("bob")
    assert (await client.get(f"/sessions/{sid}", headers=bob)).status_code == 403
    msg = await client.post(
        f"/sessions/{sid}/messages", json={"message": "hi"}, headers=bob
    )
    assert msg.status_code == 403
