from __future__ import annotations

import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from backtesting.cost_model import CostModel
from data.ingestion.feed_factory import FeedFactory
from execution.paper_broker import PaperBroker

LOGGER = logging.getLogger(__name__)


def load_settings(path: str = "config/settings.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_configuration(settings: dict) -> None:
    feed = FeedFactory.create(config=settings)
    if hasattr(feed, "validate_api_key"):
        valid = feed.validate_api_key()
        if not valid:
            raise RuntimeError("Polygon API Key ungültig. Prüfe config/secrets.env")
        LOGGER.info("✓ Polygon API Key validiert")
        if hasattr(feed, "get_market_status"):
            status = feed.get_market_status()
            LOGGER.info("Marktstatus: %s", status.get("market"))

    LOGGER.info("✓ Primärer Feed: %s", type(feed).__name__)
    LOGGER.info("✓ Primäres Intervall: %s", settings["data"]["intervals"]["primary"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    load_dotenv("config/secrets.env")
    settings = load_settings()
    validate_configuration(settings)

    cost_cfg = settings["backtesting"]["cost_model"]
    cost_model = CostModel(
        commission=float(cost_cfg["commission"]),
        slippage_bps=float(cost_cfg["slippage_bps"]),
        spread_bps=float(cost_cfg["spread_bps"]),
    )

    initial_capital = float(settings["backtesting"]["initial_capital"])
    broker = PaperBroker(initial_cash=initial_capital, cost_model=cost_model)

    print(f"Execution mode: {settings['execution']['mode']}")
    print(f"Broker: {settings['execution']['broker']}")
    print(f"Paper broker initialized with cash: {broker.get_cash():.2f}")


if __name__ == "__main__":
    main()
