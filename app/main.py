"""FastAPI application entry point.

Wires together configuration, the Mongo connection lifecycle, and the
conversational REST API. Run locally with:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response, status

from app.api import conversation
from app.config import get_prompt_config, get_settings
from app.db import mongo
from app.db.requests_repo import RequestsRepository
from app.db.sessions_repo import SessionsRepository
from app.logging_config import configure_logging, request_id_var

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Configure structured logging before anything else so startup is captured.
    configure_logging(settings.log_level, settings.log_format)
    # Validate the prompt config at startup so misconfiguration fails fast.
    config = get_prompt_config()
    # Don't boot a "secured" service with no way to verify tokens.
    if settings.auth_enabled and not settings.auth_secret:
        raise RuntimeError(
            "AUTH_ENABLED is true but AUTH_SECRET is not set. Provide a secret, "
            "or set AUTH_ENABLED=false for an offline demo."
        )
    db = mongo.connect()
    # Back the duplicate-detection queries with indexes (idempotent).
    categorical = [f.name for f in config.categorical_business_fields]
    await RequestsRepository(db).ensure_indexes(categorical)
    # TTL index that auto-expires stale conversation sessions.
    await SessionsRepository(db).ensure_indexes(settings.session_ttl_seconds)
    # Load the similarity backend now so the (potentially heavy) embedding model
    # is ready before the first request — and the container isn't reported
    # healthy until it is. Off the event loop so a slow load can't stall startup.
    from app.api.dependencies import build_similarity_provider

    await asyncio.to_thread(build_similarity_provider)
    # Surface the effective configuration in the logs (not on the public probe).
    effective_mock = settings.use_mock_llm or not settings.anthropic_api_key
    logger.info(
        "startup",
        extra={
            "event": "startup",
            "prompt_fingerprint": config.fingerprint,
            "schema_version": config.schema_version,
            "similarity_backend": settings.similarity_backend,
            "mock_llm": effective_mock,
            "auth_enabled": settings.auth_enabled,
        },
    )
    try:
        yield
    finally:
        mongo.close()


app = FastAPI(
    title="AI Hub — Engineering Service Desk",
    version="0.1.0",
    description="Configuration-driven AI chatbot backend.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a correlation id to every request for traceable logs.

    Honours an inbound ``X-Request-ID`` (set by an upstream gateway) or mints
    one, exposes it to the JSON formatter via ``request_id_var`` so every log
    line in this request is correlated, and echoes it back in the response.
    """

    correlation_id = request.headers.get("X-Request-ID") or uuid4().hex
    token = request_id_var.set(correlation_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = correlation_id
    return response


app.include_router(conversation.router)


@app.get("/healthz", tags=["health"])
async def healthz(response: Response) -> dict:
    """Liveness/readiness probe used by docker-compose and load balancers.

    Pings MongoDB so the probe reflects real readiness: if the database is
    unreachable the endpoint reports ``"degraded"`` with HTTP 503, letting an
    orchestrator stop routing traffic instead of sending it to a broken backend.

    The probe is public, so it returns only liveness/readiness — no
    configuration detail (prompt fingerprint, schema version, effective backend).
    Those are logged at startup instead, keeping them out of an unauthenticated
    response.
    """

    db_ok = True
    try:
        await mongo.get_db().command("ping")
    except Exception:
        db_ok = False
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "unreachable",
    }
