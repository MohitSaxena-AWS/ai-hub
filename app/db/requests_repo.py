"""Persistence for finalized engineering requests.

Records are appended, never destructively migrated: the collection is
schema-less and every document carries its own ``schema_version`` /
``prompt_fingerprint``, so requests written under an earlier prompt remain
intact and queryable after the prompt (and thus the data shape) changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.request import RequestRecord

COLLECTION = "requests"


class RequestsRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db[COLLECTION]

    async def ensure_indexes(self, categorical_field_names: list[str]) -> None:
        """Create the indexes that back duplicate detection.

        Both duplicate-detection queries run on every completed conversation, so
        they are indexed rather than left to collection scans:

        * ``business_hash`` — the exact-match (layer 1) lookup.
        * the categorical business fields (e.g. request_type + environment) — the
          candidate filter that narrows the semantic (layer 2) comparison.

        Index creation is idempotent, so this is safe to call on every startup.
        The set of categorical fields comes from the prompt config, so the index
        stays correct when the prompt — and thus the data shape — changes.
        """

        await self._col.create_index("business_hash")
        if categorical_field_names:
            await self._col.create_index(
                [(f"data.{name}", 1) for name in categorical_field_names]
            )

    async def create(self, record: RequestRecord) -> RequestRecord:
        await self._col.insert_one(record.to_mongo())
        return record

    async def get(self, request_id: str) -> RequestRecord | None:
        doc = await self._col.find_one({"_id": request_id})
        return RequestRecord.from_mongo(doc) if doc else None

    async def find_by_business_hash(self, business_hash: str) -> list[RequestRecord]:
        """Exact-match lookup on the privacy-preserving business key (stage 5)."""

        cursor = self._col.find({"business_hash": business_hash})
        return [RequestRecord.from_mongo(doc) async for doc in cursor]

    async def find_by_data_match(self, criteria: dict[str, Any]) -> list[RequestRecord]:
        """Find requests whose ``data`` fields all equal the given values.

        Used to narrow duplicate candidates to those sharing the categorical
        business fields (e.g. same request_type and environment) before the
        semantic comparison runs.
        """

        query = {f"data.{key}": value for key, value in criteria.items()}
        cursor = self._col.find(query)
        return [RequestRecord.from_mongo(doc) async for doc in cursor]

    async def update_data(
        self,
        request_id: str,
        *,
        data: dict[str, Any],
        pii: dict[str, Any],
        business_hash: str,
        schema_version: int,
        prompt_fingerprint: str,
    ) -> None:
        """Overwrite the business/PII payload of an existing request.

        Used by the "update existing request?" duplicate-confirmation flow
        (stage 5). The provenance stamps (``schema_version`` /
        ``prompt_fingerprint``) are updated to the prompt that produced the *new*
        payload, so the record's stamp always describes the data it now holds —
        if the prompt (and thus the data shape) changed between the original
        request and the update, the record would otherwise carry new-shape data
        under the old prompt's stamp, breaking the audit trail. ``created_at`` is
        preserved so the record's age is unchanged.
        """

        await self._col.update_one(
            {"_id": request_id},
            {
                "$set": {
                    "data": data,
                    "pii": pii,
                    "business_hash": business_hash,
                    "schema_version": schema_version,
                    "prompt_fingerprint": prompt_fingerprint,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    async def count(self) -> int:
        return await self._col.count_documents({})
