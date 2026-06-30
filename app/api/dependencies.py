"""FastAPI dependency providers.

Centralising these makes the wiring explicit and easy to override in tests
(e.g. swapping the database or forcing the mock LLM).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import Settings, get_prompt_config, get_settings
from app.core.conversation_engine import ConversationEngine
from app.core.duplicate_service import DuplicateService
from app.core.llm_client import AnthropicLLMClient, LLMClient
from app.core.mock_llm import MockLLMClient
from app.core.similarity import LexicalSimilarity, LocalEmbeddingSimilarity, SimilarityProvider
from app.db import mongo
from app.db.requests_repo import RequestsRepository
from app.db.sessions_repo import SessionsRepository

logger = logging.getLogger(__name__)


def get_sessions_repo() -> SessionsRepository:
    return SessionsRepository(mongo.get_db())


def get_requests_repo() -> RequestsRepository:
    return RequestsRepository(mongo.get_db())


@lru_cache
def _build_llm_client() -> LLMClient:
    """Select the real Claude client or the deterministic mock.

    The mock is used when explicitly requested or when no API key is available,
    so the service stays runnable (and testable) offline.
    """

    settings: Settings = get_settings()
    if settings.use_mock_llm or not settings.anthropic_api_key:
        return MockLLMClient()
    return AnthropicLLMClient(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        max_tokens=settings.llm_max_tokens,
    )


@lru_cache
def build_similarity_provider() -> SimilarityProvider:
    """Select the semantic (local embedding) or lexical similarity backend.

    When ``embedding`` is requested but its dependencies/model are unavailable,
    we fall back to the dependency-free lexical provider so the service stays
    runnable — but log a warning, because that silently degrades paraphrase
    detection and is almost always a misconfiguration rather than intent.
    """

    settings: Settings = get_settings()
    if settings.similarity_backend == "lexical":
        return LexicalSimilarity()
    try:
        return LocalEmbeddingSimilarity(settings.embedding_model)
    except Exception:  # pragma: no cover - exercised only without the ML extra
        logger.warning(
            "similarity_backend='embedding' requested but the embedding model "
            "could not be loaded (is the 'embeddings' extra installed?); falling "
            "back to the lexical backend. Semantic paraphrase detection is "
            "disabled until this is fixed.",
        )
        return LexicalSimilarity()


def get_duplicate_service() -> DuplicateService:
    settings = get_settings()
    return DuplicateService(
        get_requests_repo(),
        build_similarity_provider(),
        get_prompt_config(),
        # None -> the backend's own tuned default threshold is used.
        threshold=settings.duplicate_similarity_threshold,
    )


def get_engine() -> ConversationEngine:
    return ConversationEngine(
        get_prompt_config(),
        _build_llm_client(),
        get_requests_repo(),
        get_duplicate_service(),
        history_window=get_settings().llm_history_window,
    )
