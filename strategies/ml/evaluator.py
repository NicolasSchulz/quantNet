from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from backtesting.metrics import cagr, calmar_ratio, max_drawdown, sharpe_ratio

if TYPE_CHECKING:
    from strategies.ml.walk_forward import WalkForwardResult

LOGGER = logging.getLogger(__name__)


class ModelEvaluator:
    """Computes ML and trading metrics for walk-forward model validation."""

    def compute_classification_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        y_t = y_true.astype(int).to_numpy()
        y_p = np.asarray(y_pred, dtype=int)

        cm = confusion_matrix(y_t, y_p, labels=[-1, 0, 1])
        return {
            "accuracy": float(accuracy_score(y_t, y_p)),
            "f1_macro": float(f1_score(y_t, y_p, labels=[-1, 0, 1], average="macro", zero_division=0)),
            "f1_long": float(f1_score(y_t, y_p, labels=[1], average="macro", zero_division=0)),
            "f1_short": float(f1_score(y_t, y_p, labels=[-1], average="macro", zero_division=0)),
            "precision_long": float(precision_score(y_t, y_p, labels=[1], average="macro", zero_division=0)),
            "recall_long": float(recall_score(y_t, y_p, labels=[1], average="macro", zero_division=0)),
            "confusion_matrix": cm,
            "n_samples": int(len(y_t)),
        }

    def compute_trading_metrics(
        self,
        y_pred: pd.Series,
        returns: pd.Series,
        transaction_cost_bps: float = 7.0,
    ) -> dict[str, float]:
        aligned_returns = returns.reindex(y_pred.index).fillna(0.0).astype(float)
        position = y_pred.astype(float)
        cost = transaction_cost_bps / 10_000.0

        gross = position.shift(1).fillna(0.0) * aligned_returns
        turnover = position.diff().abs().fillna(0.0)
        costs = turnover * cost
        daily_pnl = gross - costs
        equity_curve = (1.0 + daily_pnl).cumprod()

        strategy_sharpe = float(sharpe_ratio(daily_pnl, risk_free_rate=0.0))
        strategy_cagr = float(cagr(equity_curve))
        strategy_mdd = float(max_drawdown(equity_curve))
        strategy_calmar = float(calmar_ratio(equity_curve))

        non_zero = daily_pnl[daily_pnl != 0]
        wins = non_zero[non_zero > 0]
        losses = non_zero[non_zero < 0]

        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        total_cost_drag_bps = float(costs.sum() * 10_000.0)

        return {
            "strategy_cagr": strategy_cagr,
            "strategy_sharpe": strategy_sharpe,
            "strategy_max_drawdown": strategy_mdd,
            "strategy_calmar": strategy_calmar,
            "n_trades": int((turnover > 0).sum()),
            "win_rate": float((non_zero > 0).mean()) if len(non_zero) else 0.0,
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "profit_factor": float(profit_factor),
            "cost_drag_bps": total_cost_drag_bps,
        }

    def plot_results(
        self,
        wf_result: "WalkForwardResult",
        returns: pd.Series,
        output_dir: str = "./outputs/ml/",
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pred = wf_result.predictions.sort_index()
        ret = returns.reindex(pred.index).fillna(0.0)
        strategy_daily = pred.shift(1).fillna(0) * ret
        strategy_equity = (1.0 + strategy_daily).cumprod()
        buy_hold = (1.0 + ret).cumprod()

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax1, ax2, ax3, ax4 = axes.ravel()

        ax1.plot(strategy_equity.index, strategy_equity.values, label="Strategy", lw=1.8)
        ax1.plot(buy_hold.index, buy_hold.values, label="Buy & Hold", lw=1.2)
        ax1.set_title("Equity Curve")
        ax1.grid(alpha=0.3)
        ax1.legend()

        fi_mean = wf_result.feature_importance_mean.sort_values(ascending=False).head(20)
        fi_std = None
        if wf_result.feature_importance_std is not None:
            fi_std = wf_result.feature_importance_std.reindex(fi_mean.index).fillna(0.0)
        ax2.barh(fi_mean.index[::-1], fi_mean.values[::-1], xerr=None if fi_std is None else fi_std.values[::-1])
        ax2.set_title("Top 20 Feature Importance (Gain)")
        ax2.grid(alpha=0.3, axis="x")

        cm = np.asarray(wf_result.aggregate_metrics.get("confusion_matrix", np.zeros((3, 3))), dtype=float)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)
        im = ax3.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax3.set_xticks([0, 1, 2])
        ax3.set_xticklabels(["-1", "0", "+1"])
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(["-1", "0", "+1"])
        ax3.set_title("Confusion Matrix (Normalized)")
        for i in range(3):
            for j in range(3):
                ax3.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        # Only plot per-fold metrics if folds were generated
        if not wf_result.metrics_per_fold.empty and len(wf_result.metrics_per_fold) > 0:
            sharpe_col = "strategy_sharpe" if "strategy_sharpe" in wf_result.metrics_per_fold.columns else "fold_sharpe"
            ax4.bar(range(len(wf_result.metrics_per_fold)), wf_result.metrics_per_fold[sharpe_col])
            ax4.axhline(0.0, color="red", linestyle="--", linewidth=1)
            ax4.set_title("Sharpe per Fold")
            ax4.set_xlabel("Fold")
            ax4.set_xticks(range(len(wf_result.metrics_per_fold)))
            ax4.grid(alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No Walk-Forward Folds Generated\n(Insufficient Time Distribution)", 
                    ha="center", va="center", transform=ax4.transAxes, fontsize=12, color="gray")
            ax4.set_title("Sharpe per Fold")
            ax4.set_xticks([])

        fig.tight_layout()
        fig.savefig(output_path / "walk_forward_results.png", dpi=160)
        plt.close(fig)

    def overfitting_check(self, metrics_per_fold: pd.DataFrame) -> dict[str, float | str]:
        # Handle empty metrics (no walk-forward folds generated)
        if metrics_per_fold.empty or len(metrics_per_fold) == 0:
            return {
                "sharpe_mean": 0.0,
                "sharpe_std": 0.0,
                "sharpe_cv": 0.0,
                "pct_profitable_folds": 0.0,
                "overfitting_risk": "INSUFFICIENT_DATA",
            }
        
        sharpe_col = "strategy_sharpe" if "strategy_sharpe" in metrics_per_fold.columns else "fold_sharpe"
        sharpe = metrics_per_fold[sharpe_col].astype(float)

        sharpe_mean = float(sharpe.mean()) if len(sharpe) else 0.0
        sharpe_std = float(sharpe.std(ddof=0)) if len(sharpe) else 0.0
        sharpe_cv = sharpe_std / max(abs(sharpe_mean), 1e-12)
        pct_profitable = float((sharpe > 0).mean()) if len(sharpe) else 0.0

        if sharpe_cv < 0.5 and pct_profitable > 0.7:
            risk = "LOW"
        elif sharpe_cv < 1.0 and pct_profitable > 0.5:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        return {
            "sharpe_mean": sharpe_mean,
            "sharpe_std": sharpe_std,
            "sharpe_cv": float(sharpe_cv),
            "pct_profitable_folds": pct_profitable,
            "overfitting_risk": risk,
        }

    def generate_report(self, wf_result: "WalkForwardResult", trading_metrics: dict[str, float]) -> str:
        metrics = wf_result.aggregate_metrics
        oc = self.overfitting_check(wf_result.metrics_per_fold)

        if wf_result.predictions.empty:
            start = "N/A"
            end = "N/A"
        else:
            start = wf_result.predictions.index.min().date().isoformat()
            end = wf_result.predictions.index.max().date().isoformat()

        # Handle empty fold metrics
        if wf_result.metrics_per_fold.empty or len(wf_result.metrics_per_fold) == 0:
            worst = None
            best = None
        else:
            sharpe_col = "strategy_sharpe" if "strategy_sharpe" in wf_result.metrics_per_fold.columns else "fold_sharpe"
            worst_idx = wf_result.metrics_per_fold[sharpe_col].idxmin()
            best_idx = wf_result.metrics_per_fold[sharpe_col].idxmax()
            worst = wf_result.metrics_per_fold.loc[worst_idx]
            best = wf_result.metrics_per_fold.loc[best_idx]

        report = f"""
══════════════════════════════════════════
WALK-FORWARD VALIDATION REPORT
══════════════════════════════════════════
Modell:          {metrics.get('model_name', 'LGBMClassifier')}
Zeitraum:        {start} bis {end}
Anzahl Folds:    {len(wf_result.folds)}
Train Window:    {metrics.get('train_window_days', 'N/A')} Tage
Test Window:     {metrics.get('test_window_days', 'N/A')} Tage

CLASSIFICATION METRICS (out-of-sample)
──────────────────────────────────────
Accuracy:        {metrics.get('accuracy', 0.0):.3f}
F1 Macro:        {metrics.get('f1_macro', 0.0):.3f}
F1 Long:         {metrics.get('f1_long', 0.0):.3f}
F1 Short:        {metrics.get('f1_short', 0.0):.3f}

TRADING METRICS (nach Kosten)
──────────────────────────────────────
CAGR:            {trading_metrics.get('strategy_cagr', 0.0):.2%}
Sharpe Ratio:    {trading_metrics.get('strategy_sharpe', 0.0):.2f}
Max Drawdown:    {trading_metrics.get('strategy_max_drawdown', 0.0):.2%}
Calmar Ratio:    {trading_metrics.get('strategy_calmar', 0.0):.2f}
Anzahl Trades:   {int(trading_metrics.get('n_trades', 0))}
Win Rate:        {trading_metrics.get('win_rate', 0.0):.1%}
Profit Factor:   {trading_metrics.get('profit_factor', 0.0):.2f}

OVERFITTING CHECK
──────────────────────────────────────
Fold Sharpe Mean:   {oc['sharpe_mean']:.2f}
Fold Sharpe Std:    {oc['sharpe_std']:.2f}
Schlechtester Fold: {'N/A (Insufficient Folds)' if worst is None else f"{worst.get('strategy_sharpe', worst.get('fold_sharpe', 0.0)):.2f} ({worst['test_start'].date().isoformat()}→{worst['test_end'].date().isoformat()})"}
Bester Fold:        {'N/A (Insufficient Folds)' if best is None else f"{best.get('strategy_sharpe', best.get('fold_sharpe', 0.0)):.2f} ({best['test_start'].date().isoformat()}→{best['test_end'].date().isoformat()})"}
Overfitting Risk:   {oc['overfitting_risk']}
══════════════════════════════════════════
""".strip()
        return report
