"""Shared test fixtures.

Tests run fully offline: an in-memory Mongo (``mongomock-motor``) is injected in
place of the real database, and the mock LLM is forced on so no Anthropic API
key or network access is required.
"""

from __future__ import annotations

import os

import jwt
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from mongomock_motor import AsyncMongoMockClient

os.environ.setdefault("USE_MOCK_LLM", "true")
# Use the dependency-free lexical similarity backend so tests run offline
# without the embeddings (torch) extra installed.
os.environ.setdefault("SIMILARITY_BACKEND", "lexical")
# Authentication is on by default (secure by default); pin a known secret so
# tests can mint valid bearer tokens.
TEST_AUTH_SECRET = "test-secret-at-least-32-bytes-long!!"
os.environ.setdefault("AUTH_SECRET", TEST_AUTH_SECRET)

from app.db import mongo  # noqa: E402
from app.main import app  # noqa: E402

# Default principal used by the shared ``client`` fixture.
DEFAULT_SUBJECT = "alice"


def make_token(subject: str = DEFAULT_SUBJECT) -> str:
    """Mint a valid HS256 bearer token for ``subject``."""

    return jwt.encode({"sub": subject}, TEST_AUTH_SECRET, algorithm="HS256")


def auth_headers(subject: str = DEFAULT_SUBJECT) -> dict[str, str]:
    return {"Authorization": f"Bearer {make_token(subject)}"}


@pytest.fixture
def token():
    """Factory for bearer tokens, e.g. ``token('bob')``."""

    return make_token


@pytest_asyncio.fixture
async def client():
    """HTTP client bound to the app, backed by in-memory Mongo, authenticated.

    Carries a valid bearer token for the default principal so existing tests
    exercise the happy path; per-request ``headers=`` overrides it where a test
    needs a different (or no) identity.
    """

    test_db = AsyncMongoMockClient()["ai_hub_test"]
    mongo.set_db(test_db)

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers=auth_headers()
    ) as ac:
        yield ac

    mongo.close()
