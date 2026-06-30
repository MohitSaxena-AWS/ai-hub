"""Audit logging for persisted engineering requests.

A bank needs an answer to "who changed what request, when". Every time a request
is created or updated, an audit event is emitted on a dedicated ``ai_hub.audit``
logger. Events are deliberately **PII-free**: they carry the actor (the
authenticated principal, or ``"anonymous"`` when auth is disabled), the action,
the request id, and the provenance stamps — never the requester's name, employee
id, or the free-text justification. The correlation id is attached automatically
by the JSON formatter (see ``app/logging_config.py``).

Kept separate from the request repository so persistence stays a thin data layer
and the audit trail is driven by the domain action in the conversation engine.
"""

from __future__ import annotations

import logging

# Dedicated logger so audit events can be routed/retained independently of
# application logs (e.g. shipped to a tamper-evident store) in production.
audit_logger = logging.getLogger("ai_hub.audit")
# Pin the audit logger to INFO regardless of the global LOG_LEVEL: an audit trail
# must not silently vanish because the app was started at WARNING. The record
# still propagates to the root handler (which emits everything it receives), so
# raising LOG_LEVEL quietens app logs without ever dropping audit events.
audit_logger.setLevel(logging.INFO)


def record_persisted(
    action: str,
    *,
    request_id: str,
    actor: str | None,
    session_id: str,
    schema_version: int,
    prompt_fingerprint: str,
) -> None:
    """Emit a structured, PII-free audit event for a request create/update.

    ``action`` is ``"created"`` or ``"updated"``. ``actor`` is the principal that
    drove the conversation, or ``None`` when authentication is disabled (recorded
    as ``"anonymous"``).
    """

    audit_logger.info(
        "request.%s",
        action,
        extra={
            "event": f"request.{action}",
            "actor": actor or "anonymous",
            "request_id": request_id,
            "session_id": session_id,
            "schema_version": schema_version,
            "prompt_fingerprint": prompt_fingerprint,
        },
    )
