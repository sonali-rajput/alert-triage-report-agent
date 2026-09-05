"""Structured JSON logging. Cloud Run ingests JSON lines on stdout into
Cloud Logging with severity mapping; locally the same format keeps runs
grep-able."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

_SEVERITY = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "severity": _SEVERITY.get(record.levelno, "DEFAULT"),
            "time": datetime.now(UTC).isoformat(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            entry.update(extra)
        return json.dumps(entry, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


def log_with(logger: logging.Logger, level: int, message: str, **fields) -> None:
    """Log a message with structured extra fields."""
    logger.log(level, message, extra={"extra_fields": fields})
