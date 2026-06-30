"""Text-similarity providers for semantic duplicate detection.

All providers run *inside the application's trust boundary* — no request text is
ever sent to an external service. This is a deliberate, security-driven choice
for the banking context: the free-text justification may contain sensitive
business information, so embeddings are computed locally.

Two implementations behind one interface:

* ``LocalEmbeddingSimilarity`` — dense sentence embeddings (sentence-transformers)
  compared by cosine similarity. Captures paraphrases/synonyms; this is the
  accuracy-oriented default for production.
* ``LexicalSimilarity`` — Jaccard token overlap. No ML dependency, fully
  deterministic; used for offline builds/tests and as a fallback.

Keeping the choice behind ``SimilarityProvider`` makes it swappable and easy to
mock, so the duplicate service never depends on a concrete backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class SimilarityProvider(ABC):
    @abstractmethod
    def similarity(self, a: str, b: str) -> float:
        """Return a similarity score in [0.0, 1.0] for two texts."""

    def similarity_to_many(self, query: str, candidates: Sequence[str]) -> list[float]:
        """Score ``query`` against several ``candidates`` in one call.

        The duplicate service compares one new request against a set of existing
        ones, so this is the hot path. The default implementation simply loops
        over ``similarity``; backends that can amortise per-query work (e.g.
        embedding the query once and batching the candidates) override it.
        """

        return [self.similarity(query, c) for c in candidates]

    @property
    @abstractmethod
    def default_threshold(self) -> float:
        """The duplicate cut-off tuned for *this* backend's score distribution.

        Lexical overlap and dense-embedding cosine live on different scales, so a
        single shared threshold cannot fit both. Each backend therefore declares
        its own tuned default; the duplicate service uses it unless an explicit
        override is configured.
        """


class LexicalSimilarity(SimilarityProvider):
    """Deterministic Jaccard overlap over whitespace tokens.

    Catches re-worded requests that still share vocabulary, but not pure
    synonym/paraphrase. Useful offline and as a dependency-free fallback.
    """

    # Token overlap is sparse, so the cut-off is far lower than a cosine one: at
    # ~0.35 a re-worded justification that shares roughly half its words with an
    # existing one (already narrowed to the same request_type + environment) is
    # treated as a duplicate, while unrelated text stays well below.
    @property
    def default_threshold(self) -> float:
        return 0.35

    def similarity(self, a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)


def cosine_similarity(va: Sequence[float], vb: Sequence[float]) -> float:
    """Cosine of two vectors, clamped to [0, 1].

    Extracted as a pure function so the scoring is unit-testable without loading
    an embedding model. Sentence-embedding cosines are almost always
    non-negative, and conventional thresholds (~0.8) apply directly.
    """

    import numpy as np

    a = np.asarray(va, dtype=float)
    b = np.asarray(vb, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / denom)))


class LocalEmbeddingSimilarity(SimilarityProvider):
    """Dense-embedding cosine similarity via a locally-loaded model.

    The model weights ship inside the container (baked at image-build time), so
    inference is in-process and no text leaves the trust boundary.
    ``sentence-transformers`` is imported lazily so the rest of the app (and the
    lexical fallback) does not require it.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    # This is a *symmetric* comparison (justification vs justification), so no
    # asymmetric query instruction prefix (as bge uses for retrieval) is applied.
    #
    # Threshold tuned against the labelled set in eval/dataset.py (run
    # `python -m eval.calibrate --backend embedding`). bge-small cosines cluster
    # high for short same-domain English text: on *same-category* pairs the naive
    # 0.8 cut-off gives only ~0.54 precision (it flags ~half of legitimately
    # distinct requests as duplicates), whereas 0.90 maximises F1 (P≈0.78,
    # R≈1.00) — a much better point for a banking desk, where a false "you
    # already requested this" can block real work. Re-calibrate on a larger,
    # domain-sampled set before production; override per-deployment via
    # DUPLICATE_SIMILARITY_THRESHOLD.
    @property
    def default_threshold(self) -> float:
        return 0.9

    def similarity(self, a: str, b: str) -> float:
        va, vb = self._model.encode([a, b])
        return cosine_similarity(va, vb)

    def similarity_to_many(self, query: str, candidates: Sequence[str]) -> list[float]:
        """Embed the query once and the candidates in a single batch.

        The base implementation would re-encode ``query`` for every candidate;
        here all texts are encoded in one ``encode`` call, so the query is
        embedded exactly once regardless of how many candidates are compared.
        """

        if not candidates:
            return []
        vectors = self._model.encode([query, *candidates])
        query_vec = vectors[0]
        return [cosine_similarity(query_vec, vectors[i + 1]) for i in range(len(candidates))]
