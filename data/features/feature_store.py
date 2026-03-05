from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from strategies.ml.labeler import TripleBarrierLabeler


class FeatureStore:
    def __init__(self, base_path: str = "./data/cache/features") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _version_path(self, symbol: str, feature_version: str) -> Path:
        return self.base_path / symbol.upper() / feature_version

    def save(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        symbol: str,
        feature_version: str,
    ) -> None:
        version_path = self._version_path(symbol, feature_version)
        version_path.mkdir(parents=True, exist_ok=True)

        features_path = version_path / "features.parquet"
        labels_path = version_path / "labels.parquet"
        metadata_path = version_path / "metadata.json"

        features.to_parquet(features_path, index=True)
        labels.to_frame(name="label").to_parquet(labels_path, index=True)

        labeler = TripleBarrierLabeler(
            take_profit=float(labels.attrs.get("take_profit", 0.02)),
            stop_loss=float(labels.attrs.get("stop_loss", 0.01)),
            max_holding=int(labels.attrs.get("max_holding", 10)),
            min_return=float(labels.attrs.get("min_return", 0.001)),
        )
        distribution = labeler.label_distribution(labels)

        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": int(len(features)),
            "feature_names": list(features.columns),
            "label_distribution": distribution,
            "take_profit": float(labels.attrs.get("take_profit", 0.02)),
            "stop_loss": float(labels.attrs.get("stop_loss", 0.01)),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load(
        self,
        symbol: str,
        feature_version: str,
        start: str | None = None,
        end: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        version_path = self._version_path(symbol, feature_version)
        features_path = version_path / "features.parquet"
        labels_path = version_path / "labels.parquet"

        if not features_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Feature version not found for {symbol.upper()}::{feature_version}"
            )

        features = pd.read_parquet(features_path)
        labels = pd.read_parquet(labels_path)["label"]

        features.index = pd.to_datetime(features.index, utc=True)
        labels.index = pd.to_datetime(labels.index, utc=True)

        if start is not None:
            start_ts = pd.Timestamp(start)
            start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
            features = features[features.index >= start_ts]
            labels = labels[labels.index >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end)
            end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
            features = features[features.index <= end_ts]
            labels = labels[labels.index <= end_ts]

        return features.sort_index(), labels.sort_index()

    def exists(self, symbol: str, feature_version: str) -> bool:
        version_path = self._version_path(symbol, feature_version)
        return (version_path / "features.parquet").exists() and (version_path / "labels.parquet").exists()

    def list_versions(self, symbol: str) -> list[str]:
        symbol_path = self.base_path / symbol.upper()
        if not symbol_path.exists():
            return []
        return sorted([p.name for p in symbol_path.iterdir() if p.is_dir()])

    def invalidate(self, symbol: str, feature_version: str) -> None:
        version_path = self._version_path(symbol, feature_version)
        if version_path.exists():
            shutil.rmtree(version_path)
