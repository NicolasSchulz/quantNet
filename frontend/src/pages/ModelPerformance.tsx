import { useMemo, useState } from "react"
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import { ConfusionMatrix } from "../components/charts/ConfusionMatrix"
import { FoldMetrics } from "../components/charts/FoldMetrics"
import { LossChart } from "../components/charts/LossChart"
import { ThresholdStabilityChart } from "../components/charts/ThresholdStabilityChart"
import { TrainOOSGapChart } from "../components/charts/TrainOOSGapChart"
import { ErrorState } from "../components/ui/ErrorState"
import { LoadingSpinner } from "../components/ui/LoadingSpinner"
import { MetricCard } from "../components/ui/MetricCard"
import { StatusBadge } from "../components/ui/StatusBadge"
import { useConfusionMatrix, useFeatureImportance, useFoldResults, useLossCurves, useModelList, useModelMetrics, useThresholdAnalysis, useTrainOOSComparison, useTrainingProgress } from "../hooks/useModelMetrics"
import type { FoldResult } from "../types"
import { cn, formatDecimal, formatPercent, formatUtcToLocal } from "../utils/formatters"
import { colors } from "../utils/colors"

function verdictVariant(verdict: string) {
  if (verdict === "NONE") return "success"
  if (verdict === "MILD") return "warning"
  if (verdict === "MODERATE") return "warning"
  return "danger"
}

export function ModelPerformance() {
  const modelListQuery = useModelList()
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null)
  const [selectedLossFoldId, setSelectedLossFoldId] = useState<number | "average">("average")
  const trainingProgressQuery = useTrainingProgress()

  const effectiveModelId = selectedModelId ?? modelListQuery.data?.models[0]?.model_id ?? null
  const metricsQuery = useModelMetrics(effectiveModelId)
  const confusionQuery = useConfusionMatrix(effectiveModelId)
  const foldsQuery = useFoldResults(effectiveModelId)
  const comparisonQuery = useTrainOOSComparison(effectiveModelId)
  const thresholdQuery = useThresholdAnalysis(effectiveModelId)
  const lossCurvesQuery = useLossCurves(effectiveModelId)
  const featuresQuery = useFeatureImportance(effectiveModelId)
  const sortedFolds = useMemo(() => [...(foldsQuery.data?.folds ?? [])].sort((left, right) => left.oos_sharpe - right.oos_sharpe), [foldsQuery.data?.folds])
  const weakestFoldIds = new Set(sortedFolds.slice(0, 3).map((fold) => fold.fold_id))
  const strongestFoldIds = new Set(sortedFolds.slice(-3).map((fold) => fold.fold_id))

  if ([modelListQuery, metricsQuery, confusionQuery, foldsQuery, comparisonQuery, thresholdQuery, featuresQuery].some((q) => q.isLoading)) {
    return <LoadingSpinner label="Loading model dashboard..." />
  }

  if ([modelListQuery, metricsQuery, confusionQuery, foldsQuery, comparisonQuery, thresholdQuery, featuresQuery].some((q) => q.isError)) {
    return <ErrorState message="Backend nicht erreichbar – starte api/main.py" onRetry={() => void modelListQuery.refetch()} />
  }

  if (!effectiveModelId || !metricsQuery.data || !confusionQuery.data || !foldsQuery.data || !comparisonQuery.data || !thresholdQuery.data || !featuresQuery.data) {
    return <div className="rounded-2xl border border-dashed border-border bg-secondary p-10 text-center text-textSecondary">Kein Modell gefunden – fuhre model_training.py aus.</div>
  }

  const metrics = metricsQuery.data
  const folds = foldsQuery.data.folds
  const comparison = comparisonQuery.data
  const threshold = thresholdQuery.data
  const lossCurves = lossCurvesQuery.data
  const trainingProgress = trainingProgressQuery.data
  const selectedLossFold = typeof selectedLossFoldId === "number" ? lossCurves?.folds.find((fold) => fold.fold_id === selectedLossFoldId) ?? null : null
  const averageBestIteration = lossCurves?.folds.length ? Math.round(lossCurves.folds.reduce((sum, fold) => sum + fold.best_iteration, 0) / lossCurves.folds.length) : 0
  const averageEarlyStopped = lossCurves?.folds.filter((fold) => fold.early_stopped).length ?? 0
  const averageGap = lossCurves?.folds.length ? lossCurves.folds.reduce((sum, fold) => sum + fold.overfit_gap, 0) / lossCurves.folds.length : 0

  const foldColumns: Array<{ key: string; header: string; render: (row: FoldResult) => string | number }> = [
    { key: "fold_id", header: "Fold", render: (row) => row.fold_id },
    { key: "train", header: "Train", render: (row) => `${formatUtcToLocal(row.train_start, "yyyy-MM-dd")} - ${formatUtcToLocal(row.train_end, "yyyy-MM-dd")}` },
    { key: "test", header: "Test", render: (row) => `${formatUtcToLocal(row.test_start, "yyyy-MM-dd")} - ${formatUtcToLocal(row.test_end, "yyyy-MM-dd")}` },
    { key: "train_sharpe", header: "Train Sharpe", render: (row) => formatDecimal(row.train_sharpe) },
    { key: "oos_sharpe", header: "OOS Sharpe", render: (row) => formatDecimal(row.oos_sharpe) },
    { key: "gap", header: "Gap", render: (row) => formatDecimal(row.sharpe_gap) },
    { key: "flag", header: "Flag", render: (row) => (row.overfitting_flag ? "RED" : "OK") },
    { key: "threshold", header: "Threshold", render: (row) => formatDecimal(row.optimal_threshold, 2) },
  ]

  return (
    <div className="space-y-6">
      {trainingProgress && trainingProgress.status === "running" ? (
        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold">Training Laeuft</h2>
              <p className="mt-1 text-sm text-textMuted">{trainingProgress.message}</p>
            </div>
            <StatusBadge label={`${Math.round(trainingProgress.percent)}%`} variant="info" />
          </div>
          <div className="mt-4 h-3 overflow-hidden rounded-full bg-primary">
            <div
              className="h-full rounded-full bg-blue transition-all duration-500"
              style={{ width: `${Math.max(2, Math.min(100, trainingProgress.percent))}%` }}
            />
          </div>
          <div className="mt-3 grid gap-3 md:grid-cols-3">
            <div className="rounded-xl border border-border bg-primary px-4 py-3 text-xs text-textSecondary">Phase: <span className="ml-2 text-textPrimary">{trainingProgress.phase}</span></div>
            <div className="rounded-xl border border-border bg-primary px-4 py-3 text-xs text-textSecondary">Symbol: <span className="ml-2 text-textPrimary">{trainingProgress.symbol ?? "n/a"}</span></div>
            <div className="rounded-xl border border-border bg-primary px-4 py-3 text-xs text-textSecondary">
              Fortschritt:
              <span className="ml-2 text-textPrimary">
                {trainingProgress.completed_steps ?? 0}/{trainingProgress.total_steps ?? 0}
              </span>
            </div>
          </div>
          {trainingProgress.optimize ? (
            <p className="mt-3 text-xs text-textMuted">
              Optimize-Modus aktiv
              {trainingProgress.outer_fold ? ` · Outer Fold ${trainingProgress.outer_fold}` : ""}
              {trainingProgress.inner_combo_index && trainingProgress.inner_combo_total
                ? ` · Kombination ${trainingProgress.inner_combo_index}/${trainingProgress.inner_combo_total}`
                : ""}
            </p>
          ) : null}
        </section>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(360px,0.95fr)]">
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
          <div className="mt-5 flex flex-wrap gap-3">
            <div title="Misst wie konsistent die Performance ueber verschiedene Marktphasen ist. HIGH = Strategie funktioniert nur in bestimmten Regimes.">
              <StatusBadge label={`STABILITY ${metrics.stability.stability_risk}`} variant={metrics.stability.stability_risk === "LOW" ? "success" : metrics.stability.stability_risk === "MEDIUM" ? "warning" : "danger"} />
            </div>
            <div title="Misst wie viel besser das Modell auf Trainingsdaten als auf ungesehenen Daten ist. SEVERE = Modell hat auswendig gelernt statt zu generalisieren.">
              <StatusBadge label={`OVERFITTING ${metrics.overfitting.overfitting_verdict}`} variant={verdictVariant(metrics.overfitting.overfitting_verdict)} />
            </div>
          </div>
        </section>
        <div className="grid content-start gap-4 sm:grid-cols-2">
          <MetricCard compact className="h-full" title="OOS Sharpe" value={formatDecimal(metrics.oos_sharpe)} tone={metrics.oos_sharpe > 1 ? "positive" : "warning"} />
          <MetricCard compact className="h-full" title="OOS CAGR" value={formatPercent(metrics.oos_cagr)} tone="positive" />
          <MetricCard compact className="h-full" title="OOS Max Drawdown" value={formatPercent(metrics.oos_max_drawdown)} tone="negative" />
          <MetricCard compact className="h-full" title="Win Rate" value={formatPercent(metrics.win_rate)} tone={metrics.win_rate > 0.5 ? "positive" : "negative"} />
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <h2 className="text-xl font-semibold">Train vs. OOS Vergleich</h2>
          <div className="mt-4 overflow-x-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-textSecondary">
                <tr>
                  <th className="px-3 py-2">Metric</th>
                  <th className="px-3 py-2">Train</th>
                  <th className="px-3 py-2">OOS</th>
                  <th className="px-3 py-2">Gap</th>
                  <th className="px-3 py-2">Status</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-border">
                  <td className="px-3 py-2">Sharpe</td>
                  <td className="px-3 py-2">{formatDecimal(metrics.overfitting.train_sharpe_mean)}</td>
                  <td className="px-3 py-2">{formatDecimal(metrics.overfitting.oos_sharpe_mean)}</td>
                  <td className={cn("px-3 py-2", Math.abs(metrics.overfitting.sharpe_gap_pct) < 15 ? "text-success" : Math.abs(metrics.overfitting.sharpe_gap_pct) < 30 ? "text-warning" : "text-danger")}>
                    {formatDecimal(metrics.overfitting.sharpe_gap)} ({formatDecimal(metrics.overfitting.sharpe_gap_pct, 1)}%)
                  </td>
                  <td className="px-3 py-2">{metrics.overfitting.overfitting_verdict}</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2">Accuracy</td>
                  <td className="px-3 py-2">{formatDecimal(metrics.overfitting.train_accuracy_mean, 3)}</td>
                  <td className="px-3 py-2">{formatDecimal(metrics.overfitting.oos_accuracy_mean, 3)}</td>
                  <td className={cn("px-3 py-2", Math.abs(metrics.overfitting.accuracy_gap) < 0.15 ? "text-success" : Math.abs(metrics.overfitting.accuracy_gap) < 0.3 ? "text-warning" : "text-danger")}>
                    {formatDecimal(metrics.overfitting.accuracy_gap, 3)}
                  </td>
                  <td className="px-3 py-2">{Math.abs(metrics.overfitting.accuracy_gap) < 0.15 ? "OK" : "WATCH"}</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2">F1 Macro</td>
                  <td className="px-3 py-2">{formatDecimal(folds.reduce((sum, fold) => sum + fold.train_f1_macro, 0) / Math.max(folds.length, 1), 3)}</td>
                  <td className="px-3 py-2">{formatDecimal(metrics.f1_macro, 3)}</td>
                  <td className="px-3 py-2">{formatDecimal((folds.reduce((sum, fold) => sum + fold.train_f1_macro, 0) / Math.max(folds.length, 1)) - metrics.f1_macro, 3)}</td>
                  <td className="px-3 py-2">OK</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <h2 className="text-xl font-semibold">Gap Visualisierung</h2>
          <div className="mt-4">
            <TrainOOSGapChart folds={comparison.folds} />
          </div>
        </section>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <h2 className="text-xl font-semibold">Classification Metrics</h2>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <MetricCard title="Accuracy" value={formatDecimal(metrics.accuracy, 3)} subtitle={metrics.accuracy < 0.55 ? "Knapp ueber Random (0.33 bei 3 Klassen)" : "Solider Vorsprung ueber Random"} />
            <MetricCard title="F1 Macro" value={formatDecimal(metrics.f1_macro, 3)} subtitle={metrics.f1_macro > 0.5 ? "Stabil ueber alle Klassen" : "Noch heterogen zwischen den Klassen"} />
            <MetricCard title="F1 Long" value={formatDecimal(metrics.f1_long, 3)} subtitle="Long-Regime Erkennung" />
            <MetricCard title="F1 Short" value={formatDecimal(metrics.f1_short, 3)} subtitle="Short-Regime Erkennung" />
            <MetricCard title="Precision Long" value={formatDecimal(metrics.precision_long, 3)} subtitle="Wie sauber sind Long-Signale?" />
            <MetricCard title="Recall Long" value={formatDecimal(metrics.recall_long, 3)} subtitle="Wie viele Longs wurden gefunden?" />
          </div>
        </section>
        <ConfusionMatrix data={confusionQuery.data} />
      </div>

      <FoldMetrics folds={folds} />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.6fr)_minmax(320px,1fr)]">
        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold">Training Dynamics</h2>
              <p className="mt-1 text-sm text-textMuted">Loss ueber Trainingsiterationen pro Fold</p>
            </div>
            {lossCurves && lossCurves.folds.length > 0 ? (
              <select
                className="rounded-xl border border-border bg-tertiary px-3 py-2 text-sm"
                value={selectedLossFoldId}
                onChange={(event) => setSelectedLossFoldId(event.target.value === "average" ? "average" : Number(event.target.value))}
              >
                <option value="average">Ø Alle Folds</option>
                {lossCurves.folds.map((fold) => (
                  <option key={fold.fold_id} value={fold.fold_id}>
                    {`Fold ${fold.fold_id} – ${formatUtcToLocal(fold.test_start, "MMM yyyy")}`}
                  </option>
                ))}
              </select>
            ) : null}
          </div>

          {lossCurvesQuery.isLoading ? (
            <div className="mt-4 h-[280px] animate-pulse rounded-2xl bg-tertiary" />
          ) : lossCurvesQuery.isError ? (
            <div className="mt-4 rounded-2xl border border-border bg-primary p-5 text-sm text-textSecondary">
              <p className="font-medium text-textPrimary">Loss Curves nicht verfuegbar</p>
              <p className="mt-2">Dieses Modell wurde ohne Loss-Tracking trainiert. Modell neu trainieren um diese Visualisierung zu erhalten.</p>
            </div>
          ) : lossCurves && lossCurves.folds.length > 0 ? (
            <>
              <div className="mt-4">
                <LossChart data={lossCurves} selectedFoldId={selectedLossFoldId} height={280} />
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-3">
                {selectedLossFold ? (
                  <>
                    <div className="rounded-xl border border-border bg-primary px-4 py-3 text-xs text-textSecondary">Best Iteration: <span className="ml-2 text-textPrimary">{selectedLossFold.best_iteration}</span></div>
                    <div className="rounded-xl border border-border bg-primary px-4 py-3 text-xs text-textSecondary">Early Stopped: <span className="ml-2 text-textPrimary">{selectedLossFold.early_stopped ? "Ja" : "Nein"}</span></div>
                    <div className={cn("rounded-xl border border-border bg-primary px-4 py-3 text-xs", selectedLossFold.overfit_gap < 0.05 ? "text-success" : selectedLossFold.overfit_gap <= 0.2 ? "text-warning" : "text-danger")}>Final Gap: <span className="ml-2">{formatDecimal(selectedLossFold.overfit_gap, 3)}</span></div>
                  </>
                ) : (
                  <>
                    <div className="rounded-xl border border-border bg-primary px-4 py-3 text-xs text-textSecondary">Ø Best Iteration: <span className="ml-2 text-textPrimary">{averageBestIteration}</span></div>
                    <div className="rounded-xl border border-border bg-primary px-4 py-3 text-xs text-textSecondary">Ø Early Stopped: <span className="ml-2 text-textPrimary">{averageEarlyStopped}/{lossCurves.folds.length}</span></div>
                    <div className={cn("rounded-xl border border-border bg-primary px-4 py-3 text-xs", averageGap < 0.05 ? "text-success" : averageGap <= 0.2 ? "text-warning" : "text-danger")}>Ø Final Gap: <span className="ml-2">{formatDecimal(averageGap, 3)}</span></div>
                  </>
                )}
              </div>
            </>
          ) : (
            <div className="mt-4 rounded-2xl border border-border bg-primary p-5 text-sm text-textSecondary">Keine Fold-Daten verfuegbar.</div>
          )}
        </section>

        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <h2 className="text-xl font-semibold">Fold Overfitting Summary</h2>
          <div className="mt-4 max-h-[320px] overflow-y-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-textSecondary">
                <tr>
                  <th className="px-3 py-2">Fold</th>
                  <th className="px-3 py-2">Test Start</th>
                  <th className="px-3 py-2">Best Iter</th>
                  <th className="px-3 py-2">Gap</th>
                  <th className="px-3 py-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {(lossCurves?.folds ?? []).map((fold) => {
                  const isActive = selectedLossFoldId === fold.fold_id
                  const status = fold.overfit_gap < 0.05 ? "OK" : fold.overfit_gap <= 0.2 ? "Mild" : "High"
                  return (
                    <tr
                      key={fold.fold_id}
                      className={cn("cursor-pointer border-t border-border", isActive ? "bg-tertiary" : "bg-primary/20")}
                      onClick={() => setSelectedLossFoldId(fold.fold_id)}
                    >
                      <td className={cn("px-3 py-2", isActive ? "border-l-2 border-blue" : "")}>{fold.fold_id}</td>
                      <td className="px-3 py-2">{formatUtcToLocal(fold.test_start, "MMM yyyy")}</td>
                      <td className="px-3 py-2">{fold.best_iteration}</td>
                      <td className={cn("px-3 py-2", fold.overfit_gap < 0.05 ? "text-success" : fold.overfit_gap <= 0.2 ? "text-warning" : "text-danger")}>{formatDecimal(fold.overfit_gap, 3)}</td>
                      <td className="px-3 py-2">{status}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <button
            type="button"
            className="mt-4 text-sm text-blue transition hover:text-textPrimary"
            onClick={() => document.getElementById("walk-forward-table")?.scrollIntoView({ behavior: "smooth", block: "start" })}
          >
            → Detaillierte Fold-Analyse
          </button>
        </section>
      </div>

      <section id="walk-forward-table" className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
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
              {folds.map((fold) => (
                <tr
                  key={fold.fold_id}
                  className={cn(
                    "border-t border-border",
                    fold.overfitting_flag ? "bg-danger/10" : weakestFoldIds.has(fold.fold_id) ? "bg-danger/10" : strongestFoldIds.has(fold.fold_id) ? "bg-success/10" : "bg-primary/20",
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

      {threshold.threshold_source === "val_optimized" ? (
        <div className="grid gap-6 xl:grid-cols-2">
          <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
            <h2 className="text-xl font-semibold">Threshold Analyse</h2>
            <div className="mt-4 space-y-3 text-sm">
              <div className="flex justify-between"><span className="text-textSecondary">Optimierter Threshold (Ø)</span><span>{formatDecimal(threshold.mean, 2)}</span></div>
              <div className="flex justify-between"><span className="text-textSecondary">Std ueber Folds</span><span>{formatDecimal(threshold.std, 2)}</span></div>
              <div className="flex justify-between"><span className="text-textSecondary">Stabilitaet</span><span>{threshold.is_stable ? "STABIL" : "INSTABIL"}</span></div>
              <div className="flex justify-between"><span className="text-textSecondary">Min / Max</span><span>{formatDecimal(threshold.min_threshold, 2)} / {formatDecimal(threshold.max_threshold, 2)}</span></div>
            </div>
            <p className="mt-4 text-sm text-textMuted">Ein stabiler Threshold bedeutet, dass die Modell-Konfidenz konsistent kalibriert ist. Starke Schwankungen deuten auf Regime-Abhaengigkeit hin.</p>
          </section>
          <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
            <h2 className="text-xl font-semibold">Threshold ueber Zeit</h2>
            <div className="mt-4">
              <ThresholdStabilityChart analysis={threshold} />
            </div>
          </section>
        </div>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <h2 className="text-xl font-semibold">Risiko-Analyse</h2>
          <div className="mt-4 space-y-3 text-sm">
            <div className="flex justify-between"><span className="text-textSecondary">Sharpe CV</span><span>{formatDecimal(metrics.stability.sharpe_cv)}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Profitable Folds</span><span>{Math.round(metrics.stability.pct_profitable_folds * metrics.n_folds)}/{metrics.n_folds} ({formatPercent(metrics.stability.pct_profitable_folds)})</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Schlechtester Fold</span><span>{formatDecimal(metrics.stability.worst_fold_sharpe)}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Bester Fold</span><span>{formatDecimal(metrics.stability.best_fold_sharpe)}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Stability Risk</span><span>{metrics.stability.stability_risk}</span></div>
          </div>
          <p className="mt-4 text-sm text-textMuted">Misst, ob die Strategie konsistent in verschiedenen Marktphasen funktioniert.</p>
        </section>

        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <h2 className="text-xl font-semibold">Generalisierung</h2>
          <div className="mt-4 space-y-3 text-sm">
            <div className="flex justify-between"><span className="text-textSecondary">Train Sharpe</span><span>{formatDecimal(metrics.overfitting.train_sharpe_mean)}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">OOS Sharpe</span><span>{formatDecimal(metrics.overfitting.oos_sharpe_mean)}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Gap</span><span>{formatDecimal(metrics.overfitting.sharpe_gap)} ({formatDecimal(metrics.overfitting.sharpe_gap_pct, 1)}%)</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Overfitting Verdict</span><span>{metrics.overfitting.overfitting_verdict}</span></div>
          </div>
          <p className="mt-4 text-sm text-textMuted">Misst, ob das Modell auswendig lernt oder echte Muster findet. MODERATE oder SEVERE erfordert weniger Features oder staerkere Regularisierung.</p>
        </section>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
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
    </div>
  )
}
