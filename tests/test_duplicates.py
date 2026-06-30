"""Privacy-preserving duplicate-detection tests (stage 5)."""

from __future__ import annotations

import pathlib

import pytest

from app.config import load_prompt_config
from app.core.conversation_engine import parse_confirmation
from app.core.duplicate_service import DuplicateService
from app.core.similarity import LexicalSimilarity, SimilarityProvider, cosine_similarity
from app.db import mongo
from app.db.requests_repo import RequestsRepository
from app.models.request import RequestRecord

BASE = [
    "infrastructure-provisioning",
    "development",
    "Need extra capacity for the Q3 launch",
    "Alice Engineer",
    "E12345",
]


async def _run(client, messages: list[str]) -> dict:
    sid = (await client.post("/sessions")).json()["session_id"]
    last = None
    for text in messages:
        last = (await client.post(f"/sessions/{sid}/messages", json={"message": text})).json()
    return {"session_id": sid, "last": last}


# ----- end-to-end confirmation flow (exact duplicate via business_hash) ---------------


@pytest.mark.asyncio
async def test_exact_duplicate_prompts_for_confirmation(client):
    await _run(client, BASE)  # first request persisted
    result = await _run(client, BASE)  # identical second request

    last = result["last"]
    assert last["status"] == "awaiting_duplicate_confirmation"
    assert "similar request already exists" in last["message"].lower()
    assert last["request_id"] is None
    # Privacy: the prompt must not leak the other requester's identity.
    assert "alice" not in last["message"].lower()
    assert "e12345" not in last["message"].lower()


@pytest.mark.asyncio
async def test_confirm_update_modifies_existing_record(client):
    first = await _run(client, BASE)
    first_request_id = (await client.get(f"/sessions/{first['session_id']}")).json()["request_id"]

    dup = await _run(client, BASE)
    resp = await client.post(
        f"/sessions/{dup['session_id']}/messages", json={"message": "yes, update it"}
    )
    body = resp.json()
    assert body["status"] == "completed"
    assert body["request_id"] == first_request_id  # updated, not newly created

    assert await RequestsRepository(mongo.get_db()).count() == 1


@pytest.mark.asyncio
async def test_decline_update_creates_new_record(client):
    await _run(client, BASE)
    dup = await _run(client, BASE)

    resp = await client.post(
        f"/sessions/{dup['session_id']}/messages", json={"message": "no, create a new one"}
    )
    body = resp.json()
    assert body["status"] == "completed"
    assert body["request_id"]

    assert await RequestsRepository(mongo.get_db()).count() == 2


@pytest.mark.asyncio
async def test_repeated_ambiguous_confirmation_falls_back_to_new_record(client):
    """The confirmation loop terminates: repeated unparseable answers default to
    the safe, non-destructive choice (create a new request) instead of looping."""

    await _run(client, BASE)
    dup = await _run(client, BASE)
    sid = dup["session_id"]
    assert dup["last"]["status"] == "awaiting_duplicate_confirmation"

    # Two ambiguous answers (no yes/no keywords) keep re-asking without resolving.
    for _ in range(2):
        body = (
            await client.post(f"/sessions/{sid}/messages", json={"message": "maybe later"})
        ).json()
        assert body["status"] == "awaiting_duplicate_confirmation"

    # The third ambiguous answer hits the cap and the session terminates safely
    # by creating a separate record rather than risking an unwanted overwrite.
    body = (
        await client.post(f"/sessions/{sid}/messages", json={"message": "i cannot decide"})
    ).json()
    assert body["status"] == "completed"
    assert body["request_id"]
    assert await RequestsRepository(mongo.get_db()).count() == 2  # original + new


@pytest.mark.asyncio
async def test_different_request_is_not_a_duplicate(client):
    await _run(client, BASE)
    # Different environment -> different categorical key -> not a duplicate.
    other = [
        "infrastructure-provisioning",
        "production",
        "Provision a brand new cluster",
        "Bob Builder",
        "E99999",
    ]
    result = await _run(client, other)
    assert result["last"]["status"] == "completed"
    assert await RequestsRepository(mongo.get_db()).count() == 2


# ----- unit tests for the service and helpers -----------------------------------------


class _StubSimilarity(SimilarityProvider):
    """Returns a fixed score so the semantic branch is deterministic."""

    def __init__(self, score: float) -> None:
        self._score = score

    @property
    def default_threshold(self) -> float:
        return 0.8

    def similarity(self, a: str, b: str) -> float:
        return self._score


def _config():
    return load_prompt_config(pathlib.Path("config/prompt.yaml"))


@pytest.mark.asyncio
async def test_semantic_match_above_threshold_is_duplicate(client):
    config = _config()
    repo = RequestsRepository(mongo.get_db())
    await repo.create(
        RequestRecord.from_session(
            _session({
                "request_type": "service-deployment",
                "environment": "staging",
                "business_justification": "Deploy the payments service",
                "requester_name": "X",
                "employee_id": "1",
            }),
            config,
        )
    )

    service = DuplicateService(repo, _StubSimilarity(0.95), config, threshold=0.8)
    dup = await service.find_duplicate({
        "request_type": "service-deployment",
        "environment": "staging",
        "business_justification": "Roll out the payment microservice",  # paraphrase
    })
    assert dup is not None


@pytest.mark.asyncio
async def test_semantic_match_below_threshold_is_not_duplicate(client):
    config = _config()
    repo = RequestsRepository(mongo.get_db())
    await repo.create(
        RequestRecord.from_session(
            _session({
                "request_type": "service-deployment",
                "environment": "staging",
                "business_justification": "Deploy the payments service",
                "requester_name": "X",
                "employee_id": "1",
            }),
            config,
        )
    )

    service = DuplicateService(repo, _StubSimilarity(0.5), config, threshold=0.8)
    dup = await service.find_duplicate({
        "request_type": "service-deployment",
        "environment": "staging",
        "business_justification": "Completely unrelated request",
    })
    assert dup is None


@pytest.mark.asyncio
async def test_lexical_backend_catches_reworded_request(client):
    """The real (un-stubbed) lexical backend catches a re-worded justification.

    Exercises the semantic layer end to end with the dependency-free backend and
    its own tuned default threshold — the path that the default offline build
    relies on (the stubbed tests above only check the service plumbing).
    """

    config = _config()
    repo = RequestsRepository(mongo.get_db())
    await repo.create(
        RequestRecord.from_session(
            _session({
                "request_type": "incident-fix",
                "environment": "production",
                "business_justification": "Need extra capacity for the Q3 launch",
                "requester_name": "X",
                "employee_id": "1",
            }),
            config,
        )
    )
    service = DuplicateService(repo, LexicalSimilarity(), config)  # default threshold

    reworded = await service.find_duplicate({
        "request_type": "incident-fix",
        "environment": "production",
        "business_justification": "Require additional capacity for Q3 launch",
    })
    assert reworded is not None  # overlapping vocabulary -> duplicate

    unrelated = await service.find_duplicate({
        "request_type": "incident-fix",
        "environment": "production",
        "business_justification": "Rotate the expired TLS certificates",
    })
    assert unrelated is None  # same category, unrelated text -> not a duplicate


def test_cosine_similarity_scoring():
    """The embedding backend's scoring math, validated without loading a model."""

    pytest.importorskip("numpy")  # ships with the embeddings extra
    assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero-vector guard
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0  # negative clamps to 0


def test_parse_confirmation():
    assert parse_confirmation("yes, update it") is True
    assert parse_confirmation("no, make a new one") is False
    assert parse_confirmation("maybe later") is None
    # Conflicting signals are ambiguous, not a guess.
    assert parse_confirmation("no, don't update") is None


def _session(collected: dict):
    from app.models.session import Session

    return Session(collected_fields=collected)
