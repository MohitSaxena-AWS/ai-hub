"""Persistence for conversation sessions.

Sessions are stored in their own collection and kept for the full duration of a
conversation. The repository is intentionally thin: it maps between ``Session``
domain objects and Mongo documents and exposes only the operations the API
needs.
"""

from __future__ import annotations

from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.session import Session

COLLECTION = "sessions"


class ConcurrentUpdateError(RuntimeError):
    """Raised when a session was modified by another request since it was read."""


class SessionGoneError(RuntimeError):
    """Raised when the session no longer exists at save time.

    Distinct from ``ConcurrentUpdateError``: the document is gone entirely —
    typically reaped by the TTL index after expiring mid-conversation — rather
    than merely changed by a concurrent turn. The API maps this to HTTP 410 Gone
    so the caller gets an accurate "expired", not a misleading "try again".
    """


class SessionsRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[COLLECTION]

    async def ensure_indexes(self, ttl_seconds: int) -> None:
        """Create a TTL index so stale sessions are reaped automatically.

        Conversation sessions are ephemeral working state; MongoDB deletes any
        whose ``updated_at`` is older than ``ttl_seconds``, bounding the size of
        the collection without a separate cleanup job. Finalized requests live in
        their own collection and are unaffected. Index creation is idempotent.
        """

        await self._col.create_index("updated_at", expireAfterSeconds=ttl_seconds)

    async def create(self, session: Session) -> Session:
        await self._col.insert_one(session.to_mongo())
        return session

    async def get(self, session_id: str) -> Session | None:
        doc = await self._col.find_one({"_id": session_id})
        return Session.from_mongo(doc) if doc else None

    async def save(self, session: Session) -> None:
        """Persist the full session document with optimistic concurrency control.

        The write only matches if the stored ``version`` still equals the one we
        read, then bumps it. A non-matching write has two distinct causes, which
        we disambiguate so the API can return the right status:

        * the document still exists but its version moved on — a concurrent turn
          on the same session (``ConcurrentUpdateError`` -> HTTP 409);
        * the document is gone — e.g. the TTL index expired it mid-conversation
          (``SessionGoneError`` -> HTTP 410), not a retryable conflict.
        """

        expected_version = session.version
        session.version = expected_version + 1
        session.updated_at = datetime.now(timezone.utc)
        result = await self._col.replace_one(
            {"_id": session.id, "version": expected_version}, session.to_mongo()
        )
        if result.matched_count == 0:
            session.version = expected_version  # roll back the in-memory bump
            still_exists = await self._col.find_one({"_id": session.id}, {"_id": 1})
            if still_exists is None:
                raise SessionGoneError(session.id)
            raise ConcurrentUpdateError(session.id)
