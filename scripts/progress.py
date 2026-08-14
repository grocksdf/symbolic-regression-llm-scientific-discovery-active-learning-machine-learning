"""Crash-durable JSONL and human-readable progress reporting for formal runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any


class ProgressReporter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started

    def emit(self, event: str, message: str, **payload: Any) -> None:
        record = {
            "event": event,
            "message": message,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": self.elapsed_seconds,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        print(f"[{record['elapsed_seconds']:9.1f}s] {message}", flush=True)
