"""Duplicate-detection *quality* gates, measured on a labelled set.

The rest of the suite proves the dedup *plumbing* (the right query runs, the
service returns the right shape). This module proves the dedup actually
*discriminates* — duplicates score above non-duplicates — on the hard,
same-category pairs in ``eval/dataset.py``, and pins the operating point so a
regression (or a careless threshold change) is caught in CI instead of in
production.

Two backends, two very different expectations:

* ``embedding`` (production default) must clear a real quality bar at its tuned
  threshold. Marked ``slow`` and skipped automatically when the ``embeddings``
  extra is absent, so the lean dev loop stays fast.
* ``lexical`` (offline fallback) is *characterised*, not gated for precision: on
  same-category boilerplate it cannot discriminate, and we assert exactly that so
  nobody promotes it to the default thinking it is good enough. These run
  offline with no ML dependency.
"""

from __future__ import annotations

import importlib.util

import pytest

from app.core.hashing import normalize_value
from app.core.similarity import LexicalSimilarity
from eval.calibrate import metrics_at, score_pairs, separation
from eval.dataset import PAIRS

_HAS_EMBEDDINGS = (
    importlib.util.find_spec("sentence_transformers") is not None
    and importlib.util.find_spec("numpy") is not None
)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# --- offline: dataset + lexical fallback characterisation ---------------------------


def test_dataset_is_balanced_and_nontrivial():
    """Guard the dataset itself: balanced labels, no duplicate rows."""

    pos = sum(1 for *_, dup in PAIRS if dup)
    neg = len(PAIRS) - pos
    assert pos >= 10 and neg >= 10
    # No accidental identical pairs (would make scoring trivial).
    assert len({(a, b) for a, b, _ in PAIRS}) == len(PAIRS)


def test_lexical_backend_cannot_discriminate_same_category_pairs():
    """Characterisation: the lexical fallback has *no useful* separation here.

    Shared boilerplate ("provision a Kafka cluster for the X platform") dominates
    Jaccard overlap, so token similarity ranks distinct-but-similar requests as
    high as true paraphrases. This is *why* embedding is the default and lexical
    is only an offline fallback; the assertion locks that rationale in.
    """

    scored = score_pairs(LexicalSimilarity())
    mean_pos, mean_neg = separation(scored)
    assert mean_pos <= mean_neg + 0.05  # no useful separation (in fact negative)


# --- production-default embedding backend (slow, needs the embeddings extra) ---------

embedding_only = pytest.mark.skipif(
    not _HAS_EMBEDDINGS, reason="requires the 'embeddings' extra"
)


@pytest.fixture(scope="module")
def embedding_backend():
    if not _HAS_EMBEDDINGS:  # pragma: no cover - guarded by the marker below too
        pytest.skip("requires the 'embeddings' extra")
    from app.core.similarity import LocalEmbeddingSimilarity

    try:
        return LocalEmbeddingSimilarity(EMBEDDING_MODEL)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"embedding model unavailable: {exc}")


@pytest.mark.slow
@embedding_only
def test_embedding_separates_duplicates_from_non_duplicates(embedding_backend):
    """Duplicates must score clearly higher than non-duplicates on average."""

    scored = score_pairs(embedding_backend)
    mean_pos, mean_neg = separation(scored)
    assert mean_pos > mean_neg  # the core invariant duplicate detection relies on


@pytest.mark.slow
@embedding_only
def test_embedding_meets_quality_bar_at_default_threshold(embedding_backend):
    """The shipped operating point clears a measured precision/recall bar.

    These floors are below the seed set's actual numbers (P≈0.78, R≈1.00 at the
    tuned 0.90 cut-off) to leave slack for model/version noise, while still
    failing loudly if the default regresses to the old, over-eager 0.80 (which
    scored only ~0.54 precision here).
    """

    scored = score_pairs(embedding_backend)
    m = metrics_at(scored, embedding_backend.default_threshold)
    assert m.recall >= 0.90
    assert m.precision >= 0.70
    assert m.f1 >= 0.80


@pytest.mark.slow
@embedding_only
def test_normalisation_is_idempotent_for_scoring(embedding_backend):
    """The eval scores text exactly as production does (via normalize_value)."""

    a, b = "Provision a NEW   Kafka cluster", "provision a new kafka cluster"
    assert embedding_backend.similarity(normalize_value(a), normalize_value(b)) == pytest.approx(
        1.0, abs=1e-3
    )
