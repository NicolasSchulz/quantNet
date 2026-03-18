from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import os

# ensure the workspace root is on sys.path so that `import data` works
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import yaml

from data.ingestion.feed_factory import FeedFactory
from data.normalizer import normalize_ohlcv
from data.storage.parquet_store import ParquetStore
from data.universes.crypto_universe import CryptoUniverse
from data.universes.etf_universe import EtfUniverse

LOGGER = logging.getLogger(__name__)


def load_settings() -> dict:
    with Path("config/settings.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest asset universes to parquet cache")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--symbol", nargs="*", default=None)
    parser.add_argument("--universe", default="etf", choices=["etf", "crypto", "all"])
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N symbols (default: all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    settings = load_settings()
    interval = settings["data"]["intervals"]["primary"]
    default_years = int(settings["data"].get("lookback_years", 2))
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.utcnow()
    start = pd.Timestamp(args.start) if args.start else (end - pd.DateOffset(years=default_years))

    etf_symbols = EtfUniverse().get_symbols()
    crypto_universe = CryptoUniverse()
    crypto_symbols = crypto_universe.get_symbols()
    if args.symbol:
        symbols = [s.upper() for s in args.symbol]
    elif args.universe == "crypto":
        symbols = crypto_symbols
    elif args.universe == "all":
        symbols = etf_symbols + crypto_symbols
    else:
        symbols = etf_symbols

    if args.limit:
        symbols = symbols[:args.limit]
    store = ParquetStore(storage_path=settings["data"]["storage_path"])

    cached = sum(1 for s in symbols if store.exists(s, interval))
    LOGGER.info("Cache Status: %d/%d Symbole vorhanden", cached, len(symbols))

    failed: list[str] = []
    loaded_rows = 0

    for i, symbol in enumerate(symbols, start=1):
        try:
            if store.exists(symbol, interval):
                continue
            LOGGER.info("[%d/%d] Lade %s", i, len(symbols), symbol)
            asset_class = "crypto" if symbol.endswith("USDT") else "equity"
            history = crypto_universe.get_available_history().get(symbol)
            effective_start = start
            if asset_class == "crypto" and history is not None:
                available_start = pd.Timestamp(history, tz="UTC")
                effective_start = max(start, available_start)
                LOGGER.info("%s: verfuegbar ab %s, lade ab %s", symbol, history, effective_start.date().isoformat())
            feed = FeedFactory.create_for_symbol(symbol, config=settings)
            raw = feed.fetch_historical(
                symbol=symbol,
                start=effective_start.date().isoformat(),
                end=end.date().isoformat(),
                interval=interval,
            )
            normalized = normalize_ohlcv(raw, symbol=symbol, asset_class=asset_class, interval=interval)
            store.save(normalized, symbol=symbol, interval=interval)
            loaded_rows += len(normalized)
        except Exception as exc:
            LOGGER.warning("%s fehlgeschlagen: %s", symbol, exc)
            failed.append(f"{symbol}: {exc}")

    failed_path = Path("data/scripts/failed_symbols.log")
    if failed:
        failed_path.write_text("\n".join(failed), encoding="utf-8")

    size_mb = 0.0
    for symbol in symbols:
        p = Path(settings["data"]["storage_path"]) / symbol / interval / "data.parquet"
        if p.exists():
            size_mb += p.stat().st_size / (1024 * 1024)

    LOGGER.info(
        "Ingestion abgeschlossen:\n"
        "Erfolgreich: %d/%d\n"
        "Fehlgeschlagen: %s\n"
        "Gesamt Bars: %d\n"
        "Speicherplatz: %.1f MB\n"
        "Zeitraum: %s bis %s",
        len(symbols) - len(failed),
        len(symbols),
        ", ".join([f.split(":")[0] for f in failed]) if failed else "None",
        loaded_rows,
        size_mb,
        start.date().isoformat(),
        end.date().isoformat(),
    )


if __name__ == "__main__":
    main()
