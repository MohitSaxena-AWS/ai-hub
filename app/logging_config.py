"""Structured (JSON) logging with per-request correlation ids.

Banking back-ends need machine-parseable logs that can be traced across a single
request. Two pieces live here:

* ``JsonFormatter`` — renders every log record as one JSON line, automatically
  including any non-standard attributes passed via ``logger.info(..., extra=...)``
  (this is how audit events carry their structured fields — see
  ``app/core/audit.py``).
* ``request_id_var`` — a context variable holding the current request's
  correlation id. The HTTP middleware in ``app/main.py`` sets it per request and
  echoes it back in the ``X-Request-ID`` response header; the formatter stamps it
  onto every log line emitted while handling that request.

Logging is configured once at startup (``configure_logging``) from the
``LOG_LEVEL`` / ``LOG_FORMAT`` settings, so the format is oper. controllable
without code changes.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

# Correlation id for the in-flight request; ``None`` outside request handling
# (e.g. startup logs). Set by the request-context middleware in app/main.py.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes that every ``LogRecord`` carries. Anything outside this set is
# treated as a caller-supplied ``extra`` field and emitted into the JSON payload,
# which is what lets audit events ship structured key/values with no PII.
_STANDARD_RECORD_ATTRS = frozenset(
    vars(logging.makeLogRecord({})).keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # The HTTP trace id is logged under ``correlation_id`` — deliberately not
        # ``request_id``, which already means the business engineering-request id
        # everywhere in this codebase (and is emitted as an ``extra`` field on
        # audit events). Keeping the two keys distinct avoids one overwriting the
        # other on an audit log line.
        correlation_id = request_id_var.get()
        if correlation_id:
            payload["correlation_id"] = correlation_id

        # Promote any caller-supplied ``extra=`` fields (e.g. audit attributes)
        # to top-level keys so they are queryable in a log pipeline.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install a single stdout handler on the root logger.

    ``fmt="json"`` (default) emits structured logs suitable for a banking log
    pipeline; ``fmt="text"`` keeps a human-readable format for local debugging.
    Replaces existing handlers so it is idempotent across reloads/tests.
    """

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "text":
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    else:
        handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
