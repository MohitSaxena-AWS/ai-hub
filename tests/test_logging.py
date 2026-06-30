"""Unit tests for the structured JSON log formatter."""

from __future__ import annotations

import json
import logging

from app.logging_config import JsonFormatter, request_id_var


def _format(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


def test_extra_fields_are_promoted_to_top_level():
    record = logging.LogRecord(
        "ai_hub.audit", logging.INFO, __file__, 1, "request.%s", ("created",), None
    )
    record.event = "request.created"
    record.actor = "alice"

    payload = _format(record)
    assert payload["message"] == "request.created"
    assert payload["event"] == "request.created"
    assert payload["actor"] == "alice"
    assert payload["level"] == "INFO"


def test_correlation_id_and_business_request_id_do_not_collide():
    """The HTTP trace id and the audit's request id must stay distinct keys."""

    token = request_id_var.set("trace-abc")
    try:
        record = logging.LogRecord(
            "ai_hub.audit", logging.INFO, __file__, 1, "request.created", (), None
        )
        record.request_id = "biz-123"  # business engineering-request id (audit extra)
        payload = _format(record)
    finally:
        request_id_var.reset(token)

    assert payload["correlation_id"] == "trace-abc"
    assert payload["request_id"] == "biz-123"


def test_audit_events_survive_a_higher_log_level():
    """An audit trail must not vanish when the app runs at LOG_LEVEL=WARNING."""

    from app.core.audit import audit_logger, record_persisted
    from app.logging_config import configure_logging

    # Configure the app at WARNING — app logs quietened, audit must not be.
    configure_logging("WARNING", "json")
    try:
        # The pin makes the audit logger emit INFO regardless of the root level.
        assert audit_logger.isEnabledFor(logging.INFO)

        captured: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = captured.append  # type: ignore[method-assign]
        audit_logger.addHandler(handler)
        try:
            record_persisted(
                "created",
                request_id="r1",
                actor="alice",
                session_id="s1",
                schema_version=1,
                prompt_fingerprint="fp",
            )
        finally:
            audit_logger.removeHandler(handler)
    finally:
        configure_logging("INFO", "json")  # restore for other tests

    assert [r.event for r in captured] == ["request.created"]  # type: ignore[attr-defined]


def test_exception_info_is_serialised():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "app", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
        )
    payload = _format(record)
    assert "ValueError: boom" in payload["exc_info"]
