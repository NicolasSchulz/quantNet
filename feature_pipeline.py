from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from data.features.feature_store import FeatureStore
from data.ingestion.feed_factory import FeedFactory
from data.normalizer import normalize_ohlcv
from data.storage.parquet_store import ParquetStore
from strategies.ml.feature_engineer import FeatureEngineer
from strategies.ml.labeler import TripleBarrierLabeler


def load_settings(path: str = "config/settings.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_ohlcv(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    feed,
    store: ParquetStore,
) -> pd.DataFrame:
    if store.exists(symbol, interval):
        cached = store.load(symbol=symbol, interval=interval, start=start, end=end)
        if not cached.empty:
            return cached

    raw = feed.fetch_historical(symbol=symbol, start=start, end=end, interval=interval)
    normalized = normalize_ohlcv(raw, symbol=symbol, asset_class="etf", interval=interval)
    store.save(normalized, symbol=symbol, interval=interval)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature + label dataset for one symbol")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("feature_pipeline")

    settings = load_settings()
    interval = settings["data"]["intervals"]["primary"]
    ml_features_cfg = settings["ml"]["features"]
    ml_label_cfg = settings["ml"]["labeling"]

    symbol = args.symbol.upper()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    feed, bar_store = FeedFactory.create_with_cache(config=settings)

    ohlcv = _load_ohlcv(symbol=symbol, start=start, end=end, interval=interval, feed=feed, store=bar_store)

    engineer = FeatureEngineer(
        config={
            "feature_groups": ml_features_cfg["groups"],
            "normalization": ml_features_cfg["normalization"],
            "lookback_periods": ml_features_cfg.get("lookback_periods", {}),
            "warmup_bars": int(ml_features_cfg.get("lookback_periods", {}).get("warmup_bars", 200)),
        }
    )
    features = engineer.fit_transform(ohlcv)

    labeler = TripleBarrierLabeler(
        take_profit=float(ml_label_cfg["take_profit"]),
        stop_loss=float(ml_label_cfg["stop_loss"]),
        max_holding=int(ml_label_cfg["max_holding"]),
        min_return=float(ml_label_cfg["min_return"]),
        handle_imbalance=str(ml_label_cfg.get("handle_imbalance", "weights")),
    )
    labels = labeler.label(ohlcv)

    aligned_labels = labels.reindex(features.index)
    distribution = labeler.label_distribution(aligned_labels)

    feature_store = FeatureStore(base_path=settings["data"]["storage_path"] + "/features")
    version = str(ml_features_cfg["feature_version"])
    feature_store.save(features=features, labels=aligned_labels, symbol=symbol, feature_version=version)

    out_path = Path(feature_store.base_path) / symbol / version
    valid = aligned_labels.dropna().astype(int)
    total = len(valid)

    logger.info("Label distribution: %s", distribution)

    print(f"Symbol:             {symbol}")
    print(f"Zeitraum:           {args.start} bis {args.end}")
    print(f"Bars nach Warmup:   {len(features)}")
    print(f"Features:           {features.shape[1]}")
    print(f"Label Long  (+1):   {(valid == 1).sum()} ({(((valid == 1).sum() / total) * 100 if total else 0):.1f}%)")
    print(f"Label Short (-1):   {(valid == -1).sum()} ({(((valid == -1).sum() / total) * 100 if total else 0):.1f}%)")
    print(f"Label Flat  ( 0):   {(valid == 0).sum()} ({(((valid == 0).sum() / total) * 100 if total else 0):.1f}%)")
    print(f"Gespeichert unter:  {out_path}/")
    print("Hinweis: Polygon Starter Plan: max. 2 Jahre Historie verfügbar.")
    print("Für längere Historie: Polygon Developer Plan (79$/Monat).")


if __name__ == "__main__":
    main()
