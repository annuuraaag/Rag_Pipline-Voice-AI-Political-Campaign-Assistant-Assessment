"""Structured (one JSON object per line) request logging."""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

_EVENT_LOGGER = logging.getLogger("campaign_rag.events")


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if not any(getattr(h, "_campaign_rag", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
        handler._campaign_rag = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level.upper())


def log_event(event: str, **fields: Any) -> None:
    """Emit one machine-parseable telemetry line (request_id, timings, decisions, source ids)."""
    _EVENT_LOGGER.info(json.dumps({"event": event, **fields}, default=str, ensure_ascii=False))
