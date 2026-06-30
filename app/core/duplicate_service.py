"""Privacy-preserving duplicate detection (assignment section 2.3).

Detect a possible duplicate request *without exposing any personal information*.
Two layers, neither of which uses PII:

1. Exact match — compare the privacy-preserving ``business_hash`` (a SHA-256 of
   the non-PII business fields). Catches identical requests instantly.
2. Semantic match — among existing requests that share the same *categorical*
   business fields (e.g. request_type + environment), compare the *free-text*
   business fields (e.g. justification) with a local similarity provider. This
   catches the same request worded differently.

Only business fields participate; the requester's name, employee ID and contact
details are never read here.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import PromptConfig
from app.core.hashing import compute_business_hash, normalize_value
from app.core.similarity import SimilarityProvider
from app.db.requests_repo import RequestsRepository
from app.models.request import RequestRecord

logger = logging.getLogger(__name__)


class DuplicateService:
    def __init__(
        self,
        repo: RequestsRepository,
        similarity: SimilarityProvider,
        config: PromptConfig,
        threshold: float | None = None,
    ) -> None:
        self._repo = repo
        self._similarity = similarity
        self._config = config
        # Fall back to the backend's own tuned cut-off when no explicit override
        # is configured, since lexical and embedding scores live on different
        # scales (see SimilarityProvider.default_threshold).
        self._threshold = threshold if threshold is not None else similarity.default_threshold

    async def find_duplicate(self, collected: dict) -> RequestRecord | None:
        """Return an existing request that duplicates ``collected``, or None."""

        # Layer 1: exact match on the privacy-preserving hash. The lookup returns
        # a list because several records can legitimately share a hash (a
        # requester who declined the "update existing?" prompt deliberately
        # created a second identical request). Any one of them is enough to flag
        # a duplicate, so we surface the first.
        business_names = [f.name for f in self._config.business_fields]
        exact = await self._repo.find_by_business_hash(
            compute_business_hash(collected, business_names)
        )
        if exact:
            return exact[0]

        # Layer 2: semantic match among same-category requests.
        freetext_fields = self._config.freetext_business_fields
        if not freetext_fields:
            return None

        criteria = {
            f.name: collected.get(f.name)
            for f in self._config.categorical_business_fields
            if collected.get(f.name) is not None
        }
        # Require a categorical narrowing key before the semantic scan. Without
        # one, ``find_by_data_match({})`` would match the entire collection and
        # compare the free text against every request regardless of category —
        # both wrong (cross-category matches) and unscalable. A prompt with
        # free-text but no categorical business fields therefore skips layer 2.
        if not criteria:
            return None
        candidates = await self._repo.find_by_data_match(criteria)
        if not candidates:
            return None

        new_text = self._freetext_blob(collected, freetext_fields)
        candidate_texts = [self._freetext_blob(c.data, freetext_fields) for c in candidates]
        # Score the new request against every candidate in one call so backends
        # that embed text can encode the query once (see similarity_to_many).
        # Embedding inference is CPU-bound and synchronous, so run it off the
        # event loop to avoid blocking other requests during model inference.
        # The exact-hash layer above already ran; if scoring itself fails (e.g.
        # the embedding model errors at runtime) the semantic layer degrades to
        # "no duplicate" — the safe, non-destructive outcome — rather than
        # failing the whole turn.
        try:
            scores = await asyncio.to_thread(
                self._similarity.similarity_to_many, new_text, candidate_texts
            )
        except Exception:
            logger.warning("semantic duplicate scoring failed; skipping it", exc_info=True)
            return None

        best: RequestRecord | None = None
        best_score = 0.0
        for candidate, score in zip(candidates, scores):
            if score > best_score:
                best_score = score
                best = candidate

        return best if best_score >= self._threshold else None

    @staticmethod
    def _freetext_blob(source: dict, fields) -> str:
        """Concatenate normalised free-text business values for comparison."""

        return " ".join(normalize_value(source.get(f.name, "")) for f in fields)
