from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def print_metrics(metrics: dict[str, float], n_trades: int) -> None:
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.4f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0.0):.4%}")
    print(f"CAGR: {metrics.get('cagr', 0.0):.4%}")
    print(f"Anzahl Trades: {n_trades}")


def plot_equity_curve(equity_curve: pd.Series, output_path: str = "equity_curve.png") -> Path:
    path = Path(output_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    equity_curve.plot(ax=ax, lw=2)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
