"""Shared logging helpers for the LLM log analyzer apps."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from elasticsearch import Elasticsearch


_FILE_HANDLER_FLAG = "_llm_log_file_handler"
_ES_HANDLER_FLAG = "_llm_es_log_handler"


class ElasticsearchLogHandler(logging.Handler):
    """Logging handler that indexes log records into Elasticsearch."""

    def __init__(self, es: Elasticsearch, index: str) -> None:
        super().__init__()
        self._es = es
        self.index = index

    def set_client(self, es: Elasticsearch) -> None:
        self._es = es

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - network
        try:
            if not self._es:
                return

            formatter = self.formatter or logging.Formatter("%(message)s")
            message = formatter.format(record)

            document = {
                "@timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": message,
            }
            if record.exc_info:
                document["exception"] = self.formatException(record.exc_info)

            self._es.index(index=self.index, document=document)
        except Exception:  # pragma: no cover - logging error path
            self.handleError(record)


def ensure_file_logger(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    """Ensure the shared application logger writes to a file."""

    logger = logging.getLogger("llm_log_analyzer")
    logger.setLevel(level)
    logger.propagate = False

    if not any(getattr(handler, _FILE_HANDLER_FLAG, False) for handler in logger.handlers):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        setattr(handler, _FILE_HANDLER_FLAG, True)
        logger.addHandler(handler)

    return logger


def attach_elasticsearch_handler(
    logger: logging.Logger, es: Elasticsearch, index: str, level: int = logging.INFO
) -> ElasticsearchLogHandler:
    """Attach (or update) an Elasticsearch logging handler."""

    existing: Optional[ElasticsearchLogHandler] = None
    for handler in logger.handlers:
        if getattr(handler, _ES_HANDLER_FLAG, False) and isinstance(handler, ElasticsearchLogHandler):
            if handler.index == index:
                existing = handler
                break

    if existing:
        existing.set_client(es)
        existing.setLevel(level)
        return existing

    handler = ElasticsearchLogHandler(es, index)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    setattr(handler, _ES_HANDLER_FLAG, True)
    logger.addHandler(handler)
    return handler


def detach_elasticsearch_handler(
    logger: logging.Logger, handler: Optional[ElasticsearchLogHandler]
) -> None:
    """Remove a previously attached Elasticsearch handler."""

    if handler and handler in logger.handlers:
        logger.removeHandler(handler)
