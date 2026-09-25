"""Structured request logging for the FastAPI service."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("energy_forecasting.api.request")


def new_request_id() -> str:
    return str(uuid.uuid4())


def log_request(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    model_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code,
    }
    if model_version is not None:
        payload["model_version"] = model_version
    if extra:
        payload.update(extra)
    logger.info(json.dumps(payload, default=str))
