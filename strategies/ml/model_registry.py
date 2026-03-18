from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from strategies.ml.models.base_model import BaseModel

LOGGER = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """Raised when a requested model does not exist in registry."""


@dataclass
class ModelRegistry:
    registry_path: str = "./models/registry.json"

    def __post_init__(self) -> None:
        self.path = Path(self.registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"models": {}, "default": {}})

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def register(
        self,
        model_id: str,
        model_path: str,
        symbol: str,
        feature_version: str,
        walk_forward_metrics: dict[str, Any],
        trained_at: str,
        analysis: dict[str, Any] | None = None,
        loss_curves_path: str | None = None,
    ) -> None:
        payload = self._read()
        metrics = {
            "sharpe": float(walk_forward_metrics.get("strategy_sharpe", walk_forward_metrics.get("sharpe_ratio", 0.0))),
            "cagr": float(walk_forward_metrics.get("strategy_cagr", walk_forward_metrics.get("cagr", 0.0))),
            "max_drawdown": float(walk_forward_metrics.get("strategy_max_drawdown", walk_forward_metrics.get("max_drawdown", 0.0))),
        }
        payload["models"][model_id] = {
            "model_path": model_path,
            "symbol": symbol.upper(),
            "feature_version": feature_version,
            "trained_at": trained_at,
            "metrics": metrics,
            "analysis": analysis or {},
            "loss_curves_path": loss_curves_path,
            "status": "active",
        }
        self._write(payload)

    def get_model(self, model_id: str) -> BaseModel:
        payload = self._read()
        info = payload.get("models", {}).get(model_id)
        if info is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")

        model_path = Path(info["model_path"])
        if not model_path.exists():
            raise ModelNotFoundError(f"Model file missing: {model_path}")

        import joblib

        model = joblib.load(model_path)
        if not isinstance(model, BaseModel):
            # runtime fallback for subclasses loaded from joblib
            if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
                raise TypeError(f"Loaded object from {model_path} is not a BaseModel-compatible instance")
        return model

    def get_default_model(self, symbol: str) -> tuple[BaseModel, str]:
        payload = self._read()
        model_id = payload.get("default", {}).get(symbol.upper())
        if model_id is None:
            raise ModelNotFoundError(f"No default model set for symbol '{symbol.upper()}'")
        model = self.get_model(model_id)
        info = payload["models"][model_id]
        return model, str(info["feature_version"])

    def set_default(self, symbol: str, model_id: str) -> None:
        payload = self._read()
        if model_id not in payload.get("models", {}):
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")
        payload.setdefault("default", {})[symbol.upper()] = model_id
        self._write(payload)

    def list_models(self, symbol: str | None = None) -> pd.DataFrame:
        payload = self._read()
        rows: list[dict[str, Any]] = []
        for model_id, info in payload.get("models", {}).items():
            if symbol is not None and info.get("symbol") != symbol.upper():
                continue
            rows.append(
                {
                    "model_id": model_id,
                    "symbol": info.get("symbol"),
                    "trained_at": info.get("trained_at"),
                    "sharpe": float(info.get("metrics", {}).get("sharpe", 0.0)),
                    "status": info.get("status", "unknown"),
                }
            )
        return pd.DataFrame(rows, columns=["model_id", "symbol", "trained_at", "sharpe", "status"])

    def archive(self, model_id: str) -> None:
        payload = self._read()
        info = payload.get("models", {}).get(model_id)
        if info is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")
        info["status"] = "archived"
        self._write(payload)
