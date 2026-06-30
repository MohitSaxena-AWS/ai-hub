"""Persistence and historical-compatibility tests (stage 4)."""

from __future__ import annotations

from datetime import timezone

import pytest
from mongomock_motor import AsyncMongoMockClient

from app.config import load_prompt_config
from app.core.hashing import compute_business_hash
from app.db.requests_repo import RequestsRepository
from app.models.request import RequestRecord

CONVERSATION = [
    "infrastructure-provisioning",
    "development",
    "Need extra capacity for the Q3 launch",
    "Alice Engineer",
    "E12345",
]


async def _complete_conversation(client) -> str:
    sid = (await client.post("/sessions")).json()["session_id"]
    for text in CONVERSATION:
        await client.post(f"/sessions/{sid}/messages", json={"message": text})
    return sid


@pytest.mark.asyncio
async def test_completed_request_is_persisted_with_provenance(client):
    sid = await _complete_conversation(client)

    state = (await client.get(f"/sessions/{sid}")).json()
    assert state["status"] == "completed"
    request_id = state["request_id"]
    assert request_id

    # Read the stored record straight from the (injected) database.
    from app.db import mongo

    doc = await mongo.get_db()["requests"].find_one({"_id": request_id})
    assert doc is not None
    assert doc["schema_version"] == 1
    assert doc["prompt_fingerprint"]
    assert doc["business_hash"]


@pytest.mark.asyncio
async def test_persisting_a_request_emits_a_pii_free_audit_event(client, caplog):
    """A request create emits a structured audit event with the actor, no PII."""

    import logging

    with caplog.at_level(logging.INFO, logger="ai_hub.audit"):
        sid = await _complete_conversation(client)

    request_id = (await client.get(f"/sessions/{sid}")).json()["request_id"]

    created = [r for r in caplog.records if getattr(r, "event", None) == "request.created"]
    assert len(created) == 1
    event = created[0]
    # The default test principal owns the session, so it is the recorded actor.
    assert event.actor == "alice"
    assert event.request_id == request_id
    assert event.schema_version == 1
    # PII must never appear in the audit trail.
    for value in (CONVERSATION[3], CONVERSATION[4]):  # requester name, employee id
        assert value not in event.getMessage()
        assert value not in str(event.__dict__)


@pytest.mark.asyncio
async def test_pii_is_separated_from_business_data(client):
    sid = await _complete_conversation(client)
    request_id = (await client.get(f"/sessions/{sid}")).json()["request_id"]

    from app.db import mongo

    doc = await mongo.get_db()["requests"].find_one({"_id": request_id})

    # Business data carries the request details but no personal information.
    assert doc["data"]["request_type"] == "infrastructure-provisioning"
    assert doc["data"]["environment"] == "development"
    assert "requester_name" not in doc["data"]
    assert "employee_id" not in doc["data"]

    # PII is stored apart.
    assert doc["pii"] == {"requester_name": "Alice Engineer", "employee_id": "E12345"}


@pytest.mark.asyncio
async def test_business_hash_excludes_pii_and_is_stable(client):
    """The hash depends only on business fields, not on the requester."""

    config = load_prompt_config(__import__("pathlib").Path("config/prompt.yaml"))
    business_names = [f.name for f in config.business_fields]

    base = {
        "request_type": "access-grant",
        "environment": "production",
        "business_justification": "Grant read access",
    }
    h1 = compute_business_hash({**base, "requester_name": "Alice"}, business_names)
    h2 = compute_business_hash({**base, "requester_name": "Bob"}, business_names)
    assert h1 == h2  # different people, same request -> same hash

    h3 = compute_business_hash({**base, "environment": "staging"}, business_names)
    assert h3 != h1  # different business detail -> different hash


@pytest.mark.asyncio
async def test_update_data_refreshes_provenance_stamps():
    """Updating a record restamps schema_version/fingerprint to match new data.

    If the prompt changed between the original request and the update, the stored
    payload now has the new shape; its provenance stamp must move with it so the
    audit trail never describes new-shape data under the old prompt's stamp.
    """

    db = AsyncMongoMockClient()["update_prov_test"]
    repo = RequestsRepository(db)

    original = RequestRecord(
        schema_version=1,
        prompt_fingerprint="oldfp",
        business_hash="h1",
        data={"request_type": "incident-fix", "environment": "production"},
        pii={"requester_name": "Carol"},
    )
    await repo.create(original)

    await repo.update_data(
        original.id,
        data={"request_type": "access-grant", "team": "payments"},
        pii={"requester_name": "Dan"},
        business_hash="h2",
        schema_version=2,
        prompt_fingerprint="newfp",
    )

    updated = await repo.get(original.id)
    assert updated.schema_version == 2
    assert updated.prompt_fingerprint == "newfp"
    assert updated.business_hash == "h2"
    assert updated.data == {"request_type": "access-grant", "team": "payments"}
    # created_at is preserved (not in the $set; record age unchanged). Compared at
    # second resolution because the mock store drops tz/sub-ms precision on save.
    assert abs(
        (updated.created_at.replace(tzinfo=timezone.utc) - original.created_at).total_seconds()
    ) < 1


@pytest.mark.asyncio
async def test_historical_records_survive_schema_change():
    """Records written under different schema versions remain intact together."""

    db = AsyncMongoMockClient()["hist_test"]
    repo = RequestsRepository(db)

    old = RequestRecord(
        schema_version=1,
        prompt_fingerprint="oldfp",
        business_hash="h1",
        data={"request_type": "incident-fix", "environment": "production"},
        pii={"requester_name": "Carol"},
    )
    # A later prompt produces a different shape (new field, no environment).
    new = RequestRecord(
        schema_version=2,
        prompt_fingerprint="newfp",
        business_hash="h2",
        data={"request_type": "access-grant", "team": "payments", "ttl_days": 30},
        pii={"requester_name": "Dan"},
    )
    await repo.create(old)
    await repo.create(new)

    fetched_old = await repo.get(old.id)
    fetched_new = await repo.get(new.id)

    # Both versions coexist, each readable in its original shape.
    assert fetched_old.schema_version == 1
    assert "environment" in fetched_old.data
    assert fetched_new.schema_version == 2
    assert fetched_new.data["team"] == "payments"
    assert "environment" not in fetched_new.data
    assert await repo.count() == 2
