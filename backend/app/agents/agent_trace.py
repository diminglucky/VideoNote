"""Small, append-only trace sink for agent decisions and observations."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SECRET_KEY = re.compile(r"(api[_-]?key|token|secret|authorization|password)", re.I)


def _redact(value: Any, key: str = "") -> Any:
    if _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item, key) for item in value[:50]]
    if isinstance(value, tuple):
        return [_redact(item, key) for item in value[:50]]
    if isinstance(value, str):
        return value[:500]
    return value


class JsonlTraceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = _redact(dict(event))
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
