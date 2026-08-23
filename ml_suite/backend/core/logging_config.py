"""Logging setup: human-readable on the console, one JSON line per record in
the log file so the same output can be shipped to a collector unchanged."""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler

from backend.core.config import settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_CONSOLE_FMT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:  # idempotent under uvicorn's reloader
        return
    root.setLevel(settings.LOG_LEVEL)

    console = logging.StreamHandler(sys.stdout)
    console.addFilter(RequestIdFilter())
    console.setFormatter(
        JsonFormatter() if settings.LOG_JSON
        else logging.Formatter(_CONSOLE_FMT, datefmt="%H:%M:%S")
    )
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        settings.LOG_DIR / "mlsuite.log", maxBytes=5_000_000, backupCount=3
    )
    file_handler.addFilter(RequestIdFilter())
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
