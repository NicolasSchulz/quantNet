from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler

from strategies.ml.labeler import TripleBarrierLabeler

LOGGER = logging.getLogger(__name__)


class FeatureStore:
    def __init__(self, base_path: str = "./data/cache/features") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _version_path(self, symbol: str, feature_version: str) -> Path:
        return self.base_path / symbol.upper() / feature_version

    def save(
        self,
        raw_features: pd.DataFrame,
        labels: pd.Series,
        symbol: str,
        feature_version: str,
        scaler: RobustScaler | None = None,
        metadata_extra: dict[str, object] | None = None,
    ) -> None:
        version_path = self._version_path(symbol, feature_version)
        version_path.mkdir(parents=True, exist_ok=True)

        features_path = version_path / "features_raw.parquet"
        labels_path = version_path / "labels.parquet"
        metadata_path = version_path / "metadata.json"

        raw_features.to_parquet(features_path, index=True)
        labels.to_frame(name="label").to_parquet(labels_path, index=True)
        if scaler is not None:
            joblib.dump(scaler, version_path / "scaler_global.joblib")
            LOGGER.warning("Global scaler ist deprecated – nur fuer Legacy")

        labeler = TripleBarrierLabeler(
            take_profit=float(labels.attrs.get("take_profit", 0.02)),
            stop_loss=float(labels.attrs.get("stop_loss", 0.01)),
            max_holding=int(labels.attrs.get("max_holding", 10)),
            min_return=float(labels.attrs.get("min_return", 0.001)),
        )
        distribution = labeler.label_distribution(labels)

        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": int(len(raw_features)),
            "feature_names": list(raw_features.columns),
            "label_distribution": distribution,
            "entry_strategy": str(labels.attrs.get("entry_strategy", "all_candles")),
            "take_profit": float(labels.attrs.get("take_profit", 0.02)),
            "stop_loss": float(labels.attrs.get("stop_loss", 0.01)),
            "scaled": False,
        }
        if metadata_extra:
            metadata.update(metadata_extra)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def metadata(self, symbol: str, feature_version: str) -> dict[str, object]:
        metadata_path = self._version_path(symbol, feature_version) / "metadata.json"
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def load(
        self,
        symbol: str,
        feature_version: str,
        start: str | None = None,
        end: str | None = None,
        scaled: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series]:
        version_path = self._version_path(symbol, feature_version)
        raw_features_path = version_path / "features_raw.parquet"
        legacy_features_path = version_path / "features.parquet"
        labels_path = version_path / "labels.parquet"

        if not labels_path.exists():
            raise FileNotFoundError(
                f"Feature version not found for {symbol.upper()}::{feature_version}"
            )

        if raw_features_path.exists():
            features = pd.read_parquet(raw_features_path)
        elif legacy_features_path.exists():
            LOGGER.warning(
                "Legacy scaled feature cache detected for %s/%s. Rebuild via feature_pipeline.py to migrate to raw features.",
                symbol.upper(),
                feature_version,
            )
            features = pd.read_parquet(legacy_features_path)
        else:
            raise FileNotFoundError(
                f"Feature version not found for {symbol.upper()}::{feature_version}"
            )

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

        if scaled:
            LOGGER.warning("load(..., scaled=True) is deprecated. Prefer load_raw() and fold-local scaling.")
            scaler_path = version_path / "scaler_global.joblib"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                scale_cols = [c for c in features.columns if c not in {"rsi_14", "rsi_28", "stoch_k", "stoch_d", "bb_position", "body_size", "upper_wick", "lower_wick"}]
                if scale_cols:
                    features = features.copy()
                    features[scale_cols] = scaler.transform(features[scale_cols])
            else:
                LOGGER.warning("Global scaler not found for %s/%s; returning raw features.", symbol.upper(), feature_version)

        return features.sort_index(), labels.sort_index()

    def load_raw(
        self,
        symbol: str,
        feature_version: str,
        start: str | None = None,
        end: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        return self.load(symbol=symbol, feature_version=feature_version, start=start, end=end, scaled=False)

    def exists(self, symbol: str, feature_version: str) -> bool:
        version_path = self._version_path(symbol, feature_version)
        return (
            ((version_path / "features_raw.parquet").exists() or (version_path / "features.parquet").exists())
            and (version_path / "labels.parquet").exists()
        )

    def list_versions(self, symbol: str) -> list[str]:
        symbol_path = self.base_path / symbol.upper()
        if not symbol_path.exists():
            return []
        return sorted([p.name for p in symbol_path.iterdir() if p.is_dir()])

    def invalidate(self, symbol: str, feature_version: str) -> None:
        version_path = self._version_path(symbol, feature_version)
        if version_path.exists():
            shutil.rmtree(version_path)
