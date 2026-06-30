"""MongoDB connection lifecycle.

A single ``AsyncIOMotorClient`` is created at application startup and shared
across requests. Keeping the client behind a small accessor lets tests swap in
``mongomock-motor`` without touching the rest of the code.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import get_settings

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def connect() -> AsyncIOMotorDatabase:
    """Open the shared Mongo client and return the application database."""

    global _client, _db
    if _db is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
        )
        _db = _client[settings.mongo_db]
    return _db


def get_db() -> AsyncIOMotorDatabase:
    """Return the active database, raising if the app hasn't connected yet."""

    if _db is None:
        raise RuntimeError("Database not initialised; call connect() at startup.")
    return _db


def set_db(db: AsyncIOMotorDatabase) -> None:
    """Inject a database instance (used by tests with mongomock-motor)."""

    global _db
    _db = db


def close() -> None:
    """Close the Mongo client on application shutdown."""

    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None
