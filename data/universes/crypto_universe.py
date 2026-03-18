from __future__ import annotations

import pandas as pd


class CryptoUniverse:
    SYMBOLS = {
        "BTCUSDT": {
            "name": "Bitcoin",
            "base_asset": "BTC",
            "quote_asset": "USDT",
            "category": "large_cap",
            "launch_year": 2009,
        },
        "ETHUSDT": {
            "name": "Ethereum",
            "base_asset": "ETH",
            "quote_asset": "USDT",
            "category": "large_cap",
            "launch_year": 2015,
        },
        "SOLUSDT": {
            "name": "Solana",
            "base_asset": "SOL",
            "quote_asset": "USDT",
            "category": "large_cap",
            "launch_year": 2020,
        },
    }

    def get_symbols(self) -> list[str]:
        return list(self.SYMBOLS.keys())

    def get_metadata(self) -> pd.DataFrame:
        rows = []
        for symbol, meta in self.SYMBOLS.items():
            rows.append(
                {
                    "symbol": symbol,
                    "name": meta["name"],
                    "base_asset": meta["base_asset"],
                    "category": meta["category"],
                    "launch_year": meta["launch_year"],
                }
            )
        return pd.DataFrame(rows, columns=["symbol", "name", "base_asset", "category", "launch_year"])

    def get_available_history(self) -> dict[str, str]:
        return {
            "BTCUSDT": "2017-01-01",
            "ETHUSDT": "2017-01-01",
            "SOLUSDT": "2020-04-01",
        }
