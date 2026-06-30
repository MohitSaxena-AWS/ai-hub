"""Persisted engineering request ("one row in a database", assignment section 3).

Because the fields are defined entirely by the prompt, different prompts produce
different data shapes. To keep historical data intact across prompt changes
(section 4) each record is self-describing: it stamps the ``schema_version`` and
``prompt_fingerprint`` of the prompt that produced it. Storage is schema-less
(Mongo), so older records are never migrated or overwritten — they remain
queryable exactly as written.

PII is stored separately from the business ``data`` so that duplicate detection
can operate on the non-PII part alone (section 2.3).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.core.hashing import compute_business_hash

if TYPE_CHECKING:
    from app.config import PromptConfig
    from app.models.session import Session


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RequestRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str | None = None

    # Self-describing provenance — the basis for historical compatibility.
    schema_version: int
    prompt_fingerprint: str

    # Privacy-preserving duplicate key (hash of non-PII business fields only).
    business_hash: str

    # The non-PII request details. Shape varies with the prompt.
    data: dict[str, Any] = Field(default_factory=dict)
    # Personal information, kept apart from `data` and out of `business_hash`.
    pii: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def to_mongo(self) -> dict[str, Any]:
        doc = self.model_dump(mode="python")
        doc["_id"] = doc.pop("id")
        return doc

    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "RequestRecord":
        data = dict(doc)
        data["id"] = data.pop("_id")
        return cls(**data)

    @classmethod
    def from_session(cls, session: "Session", config: "PromptConfig") -> "RequestRecord":
        """Build a record from a completed session's collected fields.

        Splits collected values into non-PII ``data`` and ``pii`` using the
        prompt's field definitions, and computes the privacy-preserving hash
        from the business fields only.
        """

        collected = session.collected_fields
        pii_names = {f.name for f in config.pii_fields}

        data = {
            name: value
            for name, value in collected.items()
            if name not in pii_names
        }
        pii = {name: collected[name] for name in pii_names if name in collected}
        business_hash = compute_business_hash(
            collected, [f.name for f in config.business_fields]
        )

        return cls(
            session_id=session.id,
            schema_version=config.schema_version,
            prompt_fingerprint=config.fingerprint,
            business_hash=business_hash,
            data=data,
            pii=pii,
        )
