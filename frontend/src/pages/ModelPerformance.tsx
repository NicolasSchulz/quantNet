import { useMemo, useState } from "react"
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import { ConfusionMatrix } from "../components/charts/ConfusionMatrix"
import { FoldMetrics } from "../components/charts/FoldMetrics"
import { DataTable, type Column } from "../components/ui/DataTable"
import { ErrorState } from "../components/ui/ErrorState"
import { LoadingSpinner } from "../components/ui/LoadingSpinner"
import { MetricCard } from "../components/ui/MetricCard"
import { MetricCardGrid } from "../components/ui/MetricCardGrid"
import { StatusBadge } from "../components/ui/StatusBadge"
import { useConfusionMatrix, useFeatureImportance, useFoldResults, useModelList, useModelMetrics } from "../hooks/useModelMetrics"
import type { FoldResult } from "../types"
import { cn, formatDecimal, formatPercent, formatUtcToLocal } from "../utils/formatters"
import { colors } from "../utils/colors"

export function ModelPerformance() {
  const modelListQuery = useModelList()
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null)

  const effectiveModelId = selectedModelId ?? modelListQuery.data?.models[0]?.model_id ?? null
  const metricsQuery = useModelMetrics(effectiveModelId)
  const confusionQuery = useConfusionMatrix(effectiveModelId)
  const foldsQuery = useFoldResults(effectiveModelId)
  const featuresQuery = useFeatureImportance(effectiveModelId)
  const sortedFolds = useMemo(() => [...(foldsQuery.data?.folds ?? [])].sort((left, right) => left.sharpe - right.sharpe), [foldsQuery.data?.folds])
  const weakestFoldIds = new Set(sortedFolds.slice(0, 3).map((fold) => fold.fold_id))
  const strongestFoldIds = new Set(sortedFolds.slice(-3).map((fold) => fold.fold_id))

  if (modelListQuery.isLoading || metricsQuery.isLoading || confusionQuery.isLoading || foldsQuery.isLoading || featuresQuery.isLoading) {
    return <LoadingSpinner label="Loading model dashboard..." />
  }

  if (modelListQuery.isError || metricsQuery.isError || confusionQuery.isError || foldsQuery.isError || featuresQuery.isError) {
    return <ErrorState message="Backend nicht erreichbar – starte api/main.py" onRetry={() => void modelListQuery.refetch()} />
  }

  if (!effectiveModelId || !metricsQuery.data || !confusionQuery.data || !foldsQuery.data || !featuresQuery.data) {
    return <div className="rounded-2xl border border-dashed border-border bg-secondary p-10 text-center text-textSecondary">Kein Modell gefunden – fuhre model_training.py aus.</div>
  }

  const metrics = metricsQuery.data
  const foldColumns: Column<FoldResult>[] = [
    { key: "fold_id", header: "Fold", sortable: true, render: (row) => row.fold_id },
    { key: "train", header: "Train Zeitraum", render: (row) => `${formatUtcToLocal(row.train_start, "yyyy-MM-dd")} - ${formatUtcToLocal(row.train_end, "yyyy-MM-dd")}` },
    { key: "test", header: "Test Zeitraum", render: (row) => `${formatUtcToLocal(row.test_start, "yyyy-MM-dd")} - ${formatUtcToLocal(row.test_end, "yyyy-MM-dd")}` },
    { key: "sharpe", header: "Sharpe", sortable: true, render: (row) => formatDecimal(row.sharpe) },
    { key: "accuracy", header: "Accuracy", sortable: true, render: (row) => formatDecimal(row.accuracy, 3) },
    { key: "n_trades", header: "Trades", sortable: true, render: (row) => row.n_trades },
    { key: "cagr", header: "CAGR", sortable: true, render: (row) => formatPercent(row.cagr) },
  ]

  return (
    <div className="space-y-6">
      <div className="grid gap-6 xl:grid-cols-[1.4fr_1fr]">
        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-textMuted">Model Info</p>
              <h2 className="mt-2 text-2xl font-semibold">{metrics.model_id}</h2>
            </div>
            <select className="rounded-xl border border-border bg-primary px-3 py-2 text-sm" value={effectiveModelId} onChange={(event) => setSelectedModelId(event.target.value)}>
              {(modelListQuery.data?.models ?? []).map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.model_id}
                </option>
              ))}
            </select>
          </div>
          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <div><p className="text-xs uppercase tracking-wide text-textMuted">Symbol</p><p className="mt-2">{metrics.symbol}</p></div>
            <div><p className="text-xs uppercase tracking-wide text-textMuted">Trained</p><p className="mt-2">{formatUtcToLocal(metrics.trained_at)}</p></div>
            <div><p className="text-xs uppercase tracking-wide text-textMuted">Feature Version</p><p className="mt-2">{metrics.feature_version}</p></div>
            <div><p className="text-xs uppercase tracking-wide text-textMuted">Walk-Forward Folds</p><p className="mt-2">{metrics.n_folds}</p></div>
          </div>
          <div className="mt-5">
            <StatusBadge label={metrics.overfitting_risk} variant={metrics.overfitting_risk === "LOW" ? "success" : metrics.overfitting_risk === "MEDIUM" ? "warning" : "danger"} />
          </div>
        </section>
        <MetricCardGrid columns={4}>
          <MetricCard title="OOS Sharpe" value={formatDecimal(metrics.oos_sharpe)} tone={metrics.oos_sharpe > 1 ? "positive" : "warning"} />
          <MetricCard title="OOS CAGR" value={formatPercent(metrics.oos_cagr)} tone="positive" />
          <MetricCard title="OOS Max Drawdown" value={formatPercent(metrics.oos_max_drawdown)} tone="negative" />
          <MetricCard title="Win Rate" value={formatPercent(metrics.win_rate)} tone={metrics.win_rate > 0.5 ? "positive" : "negative"} />
        </MetricCardGrid>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <h2 className="text-xl font-semibold">Classification Metrics</h2>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <MetricCard title="Accuracy" value={formatDecimal(metrics.accuracy, 3)} subtitle={metrics.accuracy < 0.55 ? "Knapp uber Random (0.33 bei 3 Klassen)" : "Solider Vorsprung uber Random"} />
            <MetricCard title="F1 Macro" value={formatDecimal(metrics.f1_macro, 3)} subtitle={metrics.f1_macro > 0.5 ? "Stabil uber alle Klassen" : "Noch heterogen zwischen den Klassen"} />
            <MetricCard title="F1 Long" value={formatDecimal(metrics.f1_long, 3)} subtitle="Long-Regime Erkennung" />
            <MetricCard title="F1 Short" value={formatDecimal(metrics.f1_short, 3)} subtitle="Short-Regime Erkennung" />
            <MetricCard title="Precision Long" value={formatDecimal(metrics.precision_long, 3)} subtitle="Wie sauber sind Long-Signale?" />
            <MetricCard title="Recall Long" value={formatDecimal(metrics.recall_long, 3)} subtitle="Wie viele Longs wurden gefunden?" />
          </div>
        </section>
        <ConfusionMatrix data={confusionQuery.data} />
      </div>

      <FoldMetrics folds={foldsQuery.data.folds} />

      <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
        <h3 className="mb-4 text-lg font-semibold">Walk-Forward Tabelle</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="text-xs uppercase tracking-wide text-textSecondary">
              <tr>
                {foldColumns.map((column) => (
                  <th key={column.key} className="px-3 py-2">{column.header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {foldsQuery.data.folds.map((fold) => (
                <tr
                  key={fold.fold_id}
                  className={cn(
                    "border-t border-border",
                    weakestFoldIds.has(fold.fold_id) ? "bg-danger/10" : strongestFoldIds.has(fold.fold_id) ? "bg-success/10" : "bg-primary/20",
                  )}
                >
                  {foldColumns.map((column) => (
                    <td key={column.key} className="px-3 py-2">{column.render(fold)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <h2 className="text-xl font-semibold">Overfitting Check</h2>
          <div className="mt-4 space-y-3 text-sm">
            <div className="flex justify-between"><span className="text-textSecondary">Sharpe Mean</span><span>{formatDecimal(metrics.sharpe_mean)}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Sharpe Std</span><span>{formatDecimal(metrics.sharpe_std)}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Sharpe CV</span><span>{formatDecimal(metrics.sharpe_cv)}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Profitable Folds</span><span>{Math.round(metrics.pct_profitable_folds * metrics.n_folds)}/{metrics.n_folds} ({formatPercent(metrics.pct_profitable_folds)})</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Overfitting Risk</span><span>{metrics.overfitting_risk}</span></div>
          </div>
          <p className="mt-4 text-sm text-textMuted">CV &lt; 0.5 und &gt; 70% profitable Folds = LOW Risk. CV &gt; 1.0 oder &lt; 50% profitable Folds = HIGH Risk.</p>
        </section>

        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <h2 className="text-xl font-semibold">Feature Importance</h2>
          <div className="mt-4 h-[360px]">
            <ResponsiveContainer>
              <BarChart data={[...featuresQuery.data.features].reverse()} layout="vertical" margin={{ left: 24 }}>
                <CartesianGrid stroke={colors.border} horizontal={false} />
                <XAxis type="number" stroke={colors.textSecondary} />
                <YAxis type="category" dataKey="feature" width={150} stroke={colors.textSecondary} />
                <Tooltip contentStyle={{ backgroundColor: colors.bgTertiary, borderColor: colors.border }} />
                <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
                  {featuresQuery.data.features.map((feature) => (
                    <Cell
                      key={feature.feature}
                      fill={
                        feature.feature.startsWith("trend")
                          ? colors.blue
                          : feature.feature.startsWith("momentum")
                            ? colors.yellow
                            : feature.feature.startsWith("volatility")
                              ? colors.purple
                              : colors.cyan
                      }
                    />
                  ))}
                  <ErrorBar dataKey="std" width={4} strokeWidth={1.5} stroke={colors.textPrimary} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <h2 className="text-xl font-semibold">Long Signal</h2>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            <MetricCard title="Precision" value={formatDecimal(metrics.precision_long, 3)} />
            <MetricCard title="Recall" value={formatDecimal(metrics.recall_long, 3)} />
            <MetricCard title="F1" value={formatDecimal(metrics.f1_long, 3)} />
          </div>
          <p className="mt-4 text-sm text-textSecondary">Von 100 Long-Signalen waren rund {Math.round(metrics.precision_long * 100)} tatsachlich profitabel. ROC-AUC: {metrics.roc_auc_long ? formatDecimal(metrics.roc_auc_long, 3) : "n/a"}.</p>
        </section>
        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <h2 className="text-xl font-semibold">Short Signal</h2>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            <MetricCard title="Precision" value={formatDecimal(metrics.precision_short, 3)} />
            <MetricCard title="Recall" value={formatDecimal(metrics.recall_short, 3)} />
            <MetricCard title="F1" value={formatDecimal(metrics.f1_short, 3)} />
          </div>
          <p className="mt-4 text-sm text-textSecondary">Von 100 Short-Signalen waren rund {Math.round(metrics.precision_short * 100)} tatsachlich profitabel. ROC-AUC: {metrics.roc_auc_short ? formatDecimal(metrics.roc_auc_short, 3) : "n/a"}.</p>
        </section>
      </div>
    </div>
  )
}
