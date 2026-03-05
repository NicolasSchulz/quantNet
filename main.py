from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

from backtesting.cost_model import CostModel
from execution.paper_broker import PaperBroker


def load_settings(path: str = "config/settings.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    load_dotenv("config/secrets.env")
    settings = load_settings()

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
