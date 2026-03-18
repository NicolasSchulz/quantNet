from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TrainingProgressTracker:
    def __init__(self, path: str, symbol: str, feature_version: str, optimize: bool) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol.upper()
        self.feature_version = feature_version
        self.optimize = bool(optimize)
        self._write(
            {
                "status": "running",
                "symbol": self.symbol,
                "feature_version": self.feature_version,
                "optimize": self.optimize,
                "phase": "starting",
                "message": "Training gestartet",
                "percent": 0.0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _write(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def update(self, **kwargs: Any) -> None:
        current = self.read()
        current.update(kwargs)
        self._write(current)

    def complete(self, model_id: str | None = None) -> None:
        current = self.read()
        current.update(
            {
                "status": "completed",
                "phase": "done",
                "message": "Training abgeschlossen",
                "percent": 100.0,
                "model_id": model_id,
            }
        )
        self._write(current)

    def fail(self, error: str) -> None:
        current = self.read()
        current.update(
            {
                "status": "failed",
                "phase": "failed",
                "message": error,
            }
        )
        self._write(current)

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))
