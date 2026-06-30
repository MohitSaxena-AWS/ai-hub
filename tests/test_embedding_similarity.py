"""Integration tests for the production-default semantic backend.

The rest of the suite exercises duplicate detection through the dependency-free
*lexical* backend (and a stubbed cosine), which keeps the inner loop fast and
offline. But the backend that actually ships by default — ``embedding`` with a
locally-loaded sentence-transformers model — is otherwise only covered by the
pure cosine math. These tests close that gap: they load the *real* configured
model and validate both the scoring behaviour and the end-to-end
``DuplicateService`` decision at the backend's own tuned threshold (~0.90,
calibrated in ``eval/``).

They are marked ``slow`` and skipped automatically when the ``embeddings`` extra
(torch + sentence-transformers) or the model weights are unavailable, so the
default lean dev/test run is unaffected. They DO run inside the embedding Docker
image (weights baked in) and locally after ``pip install -e ".[embeddings]"``.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("sentence_transformers", reason="requires the 'embeddings' extra")
pytest.importorskip("numpy", reason="requires the 'embeddings' extra")

from app.config import load_prompt_config  # noqa: E402
from app.core.duplicate_service import DuplicateService  # noqa: E402
from app.core.similarity import LocalEmbeddingSimilarity  # noqa: E402
from app.db.requests_repo import RequestsRepository  # noqa: E402
from app.models.request import RequestRecord  # noqa: E402
from app.models.session import Session  # noqa: E402

pytestmark = pytest.mark.slow

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@pytest.fixture(scope="module")
def provider() -> LocalEmbeddingSimilarity:
    """Load the real embedding model once for the module.

    Skips (rather than errors) if the weights cannot be loaded — e.g. no network
    on first download and nothing cached — so the test never turns into a flaky
    failure outside the baked image.
    """

    try:
        return LocalEmbeddingSimilarity(EMBEDDING_MODEL)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"embedding model unavailable: {exc}")


def _config():
    return load_prompt_config(pathlib.Path("config/prompt.yaml"))


def test_embedding_scores_separate_paraphrase_from_unrelated(provider):
    """Sanity-check the model's score distribution and the chosen threshold.

    A paraphrase must score above the backend's default cut-off and an unrelated
    sentence clearly below it, with the paraphrase always ranked higher — the
    property duplicate detection relies on.
    """

    base = "Provision a new Kafka cluster for the payments platform"
    paraphrase = "Set up a fresh Kafka messaging cluster for the payments team"
    unrelated = "Rotate the expired TLS certificates on the API gateway"

    assert provider.similarity(base, base) == pytest.approx(1.0, abs=1e-3)

    para_score = provider.similarity(base, paraphrase)
    unrel_score = provider.similarity(base, unrelated)

    assert para_score > unrel_score  # ranking is the core invariant
    assert para_score >= provider.default_threshold  # paraphrase clears the cut-off
    assert unrel_score < provider.default_threshold  # unrelated stays below it


def test_similarity_to_many_embeds_query_once(provider):
    """The batched hot path returns one score per candidate, query embedded once."""

    scores = provider.similarity_to_many(
        "Deploy the payments microservice to production",
        [
            "Roll out the payment service to prod",  # close
            "Grant database read access to an analyst",  # far
        ],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]


@pytest.mark.asyncio
async def test_duplicate_service_with_real_embeddings(client, provider):
    """End-to-end duplicate detection through the real embedding backend.

    Uses the ``client`` fixture only for its in-memory Mongo wiring; the LLM is
    irrelevant here. A stored request is matched against a paraphrase (same
    categorical fields, re-worded justification) and against an unrelated request.
    Runs at the backend's own default threshold — the exact production path.
    """

    from app.db import mongo

    config = _config()
    repo = RequestsRepository(mongo.get_db())
    await repo.create(
        RequestRecord.from_session(
            Session(
                collected_fields={
                    "request_type": "service-deployment",
                    "environment": "production",
                    "business_justification": "Provision a new Kafka cluster for the payments platform",
                    "requester_name": "X",
                    "employee_id": "1",
                }
            ),
            config,
        )
    )

    service = DuplicateService(repo, provider, config)  # backend's default threshold

    duplicate = await service.find_duplicate({
        "request_type": "service-deployment",
        "environment": "production",
        "business_justification": "Set up a fresh Kafka messaging cluster for the payments team",
    })
    assert duplicate is not None  # paraphrase of an existing request -> duplicate

    not_duplicate = await service.find_duplicate({
        "request_type": "service-deployment",
        "environment": "production",
        "business_justification": "Rotate the expired TLS certificates on the API gateway",
    })
    assert not_duplicate is None  # same category, unrelated text -> not a duplicate
