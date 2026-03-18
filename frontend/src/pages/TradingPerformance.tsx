import { useMemo, useState } from "react"
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import { EquityCurve } from "../components/charts/EquityCurve"
import { AssetClassBadge } from "../components/ui/StatusBadge"
import { DataTable, type Column } from "../components/ui/DataTable"
import { ErrorState } from "../components/ui/ErrorState"
import { LoadingSpinner } from "../components/ui/LoadingSpinner"
import { MetricCard } from "../components/ui/MetricCard"
import { MetricCardGrid } from "../components/ui/MetricCardGrid"
import { useEquityCurve, usePerformanceByExitReason, usePerformanceBySymbol, usePnlDistribution, useTradingStats } from "../hooks/usePerformance"
import type { ExitReasonPerformanceRow, SymbolPerformanceRow } from "../types"
import { formatCurrency, formatDecimal, formatPercent, formatUtcToLocal } from "../utils/formatters"
import { colors } from "../utils/colors"

const periods = [
  { label: "1M", days: 30 },
  { label: "3M", days: 90 },
  { label: "6M", days: 180 },
  { label: "1Y", days: 365 },
  { label: "All", days: null },
]
const assetClassOptions = [
  { label: "Alle Assets", value: undefined },
  { label: "Equities", value: "equity" as const },
  { label: "Crypto", value: "crypto" as const },
]

export function TradingPerformance() {
  const [period, setPeriod] = useState<(typeof periods)[number]>(periods[4])
  const [assetClass, setAssetClass] = useState<"equity" | "crypto" | undefined>(undefined)
  const endDate = new Date()
  const startDate = useMemo(() => (period.days ? new Date(Date.now() - period.days * 24 * 60 * 60 * 1000).toISOString() : undefined), [period.days])

  const statsQuery = useTradingStats({ asset_class: assetClass })
  const equityQuery = useEquityCurve({ start_date: startDate, end_date: period.days ? endDate.toISOString() : undefined, asset_class: assetClass })
  const distributionQuery = usePnlDistribution({ asset_class: assetClass })
  const bySymbolQuery = usePerformanceBySymbol({ asset_class: assetClass })
  const byExitReasonQuery = usePerformanceByExitReason({ asset_class: assetClass })

  if ([statsQuery, equityQuery, distributionQuery, bySymbolQuery, byExitReasonQuery].some((query) => query.isLoading)) {
    return <LoadingSpinner label="Loading performance dashboard..." />
  }

  if ([statsQuery, equityQuery, distributionQuery, bySymbolQuery, byExitReasonQuery].some((query) => query.isError)) {
    return <ErrorState message="Backend nicht erreichbar – starte api/main.py" onRetry={() => void statsQuery.refetch()} />
  }

  const stats = statsQuery.data
  const hasRealPerformanceData =
    (stats?.total_trades ?? 0) > 0 ||
    (equityQuery.data?.points?.length ?? 0) > 0 ||
    (bySymbolQuery.data?.rows?.length ?? 0) > 0
  const symbolColumns: Column<SymbolPerformanceRow>[] = [
    {
      key: "symbol",
      header: "Symbol",
      sortable: true,
      render: (row) => (
        <div className="flex items-center gap-2">
          <span>{row.symbol}</span>
          <AssetClassBadge assetClass={row.asset_class} />
        </div>
      ),
    },
    { key: "trades", header: "Trades", sortable: true, render: (row) => row.trades },
    { key: "win_rate", header: "Win Rate", sortable: true, render: (row) => formatPercent(row.win_rate) },
    { key: "avg_pnl", header: "Avg P&L%", sortable: true, render: (row) => formatPercent(row.avg_pnl) },
    { key: "total_pnl", header: "Total P&L", sortable: true, render: (row) => formatCurrency(row.total_pnl) },
    { key: "sharpe", header: "Sharpe", sortable: true, render: (row) => formatDecimal(row.sharpe) },
  ]
  const exitColumns: Column<ExitReasonPerformanceRow>[] = [
    { key: "exit_reason", header: "Exit Reason", sortable: true, render: (row) => row.exit_reason },
    { key: "trades", header: "Trades", sortable: true, render: (row) => row.trades },
    { key: "win_rate", header: "Win Rate", sortable: true, render: (row) => formatPercent(row.win_rate) },
    { key: "avg_pnl", header: "Avg P&L%", sortable: true, render: (row) => formatPercent(row.avg_pnl) },
    { key: "total_pnl", header: "Total P&L", sortable: true, render: (row) => formatCurrency(row.total_pnl) },
  ]

  return (
    <div className="space-y-6">
      {!hasRealPerformanceData ? (
        <section className="rounded-2xl border border-dashed border-border bg-secondary p-5 text-sm text-textSecondary">
          Keine echten Performance-Daten gefunden. Diese Seite zeigt nur die Struktur, bis echte Backtest- oder Trade-Ergebnisse im Backend verfuegbar sind.
        </section>
      ) : null}
      <MetricCardGrid columns={4}>
        <MetricCard title="Total Trades" value={`${stats?.total_trades ?? 0}`} />
        <MetricCard title="Win Rate" value={formatPercent(stats?.win_rate)} tone={(stats?.win_rate ?? 0) > 0.5 ? "positive" : "negative"} />
        <MetricCard title="Profit Factor" value={formatDecimal(stats?.profit_factor)} tone={(stats?.profit_factor ?? 0) > 1.5 ? "positive" : "warning"} />
        <MetricCard title="Sharpe Ratio" value={formatDecimal(stats?.sharpe_ratio)} tone={(stats?.sharpe_ratio ?? 0) > 1 ? "positive" : (stats?.sharpe_ratio ?? 0) > 0.5 ? "warning" : "negative"} />
        <MetricCard title="CAGR" value={formatPercent(stats?.cagr)} tone="positive" />
        <MetricCard title="Max Drawdown" value={formatPercent(stats?.max_drawdown_pct)} tone="negative" />
        <MetricCard title="Calmar Ratio" value={formatDecimal(stats?.calmar_ratio)} tone="neutral" />
        <MetricCard title="Avg Holding" value={`${formatDecimal(stats?.avg_holding_hours)}h`} />
      </MetricCardGrid>

      <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">Equity Curve</h2>
          <div className="flex flex-wrap gap-2">
            {assetClassOptions.map((item) => (
              <button
                key={item.label}
                type="button"
                className={`rounded-lg border px-3 py-1.5 text-sm ${assetClass === item.value ? "border-warning bg-warning/10 text-warning" : "border-border text-textSecondary"}`}
                onClick={() => setAssetClass(item.value)}
              >
                {item.label}
              </button>
            ))}
            {periods.map((item) => (
              <button
                key={item.label}
                type="button"
                className={`rounded-lg border px-3 py-1.5 text-sm ${period.label === item.label ? "border-success bg-success/10 text-success" : "border-border text-textSecondary"}`}
                onClick={() => setPeriod(item)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
        <div className="mt-4">
          <EquityCurve
            points={
              assetClass === "equity"
                ? equityQuery.data?.equity_points ?? []
                : assetClass === "crypto"
                  ? equityQuery.data?.crypto_points ?? []
                  : equityQuery.data?.combined_points ?? []
            }
            drawdown={equityQuery.data?.points ?? []}
            comparisonSeries={
              assetClass
                ? []
                : [
                    { label: "Equity", color: colors.blue, points: equityQuery.data?.equity_points ?? [] },
                    { label: "Crypto", color: colors.yellow, points: equityQuery.data?.crypto_points ?? [] },
                    { label: "Combined", color: colors.green, points: equityQuery.data?.combined_points ?? [] },
                  ]
            }
          />
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <h2 className="text-xl font-semibold">Win/Loss Analyse</h2>
          <div className="mt-4 space-y-3 text-sm">
            <div className="flex justify-between"><span className="text-textSecondary">Average Win</span><span>{formatPercent(stats?.avg_win_pct)}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Average Loss</span><span>{formatPercent(stats?.avg_loss_pct)}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Max Win</span><span>{formatPercent(stats?.max_win_pct)} {stats?.best_trade ? `(${stats.best_trade.symbol}, ${formatUtcToLocal(stats.best_trade.exit_time)})` : ""}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Max Loss</span><span>{formatPercent(stats?.max_loss_pct)} {stats?.worst_trade ? `(${stats.worst_trade.symbol}, ${formatUtcToLocal(stats.worst_trade.exit_time)})` : ""}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Avg Win / Avg Loss Ratio</span><span>{formatDecimal(Math.abs((stats?.avg_win_pct ?? 0) / ((stats?.avg_loss_pct ?? -1) || -1)))}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Total Commission</span><span>{formatCurrency(stats?.total_commission)}</span></div>
            <div className="flex justify-between"><span className="text-textSecondary">Total Slippage</span><span>{formatCurrency((stats?.total_commission ?? 0) * 0.35)}</span></div>
          </div>
        </section>
        <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
          <h2 className="text-xl font-semibold">P&L Verteilung</h2>
          <div className="mt-4 h-[280px]">
            <ResponsiveContainer>
              <BarChart data={distributionQuery.data?.buckets}>
                <CartesianGrid stroke={colors.border} vertical={false} />
                <XAxis dataKey="range" hide />
                <YAxis stroke={colors.textSecondary} />
                <Tooltip contentStyle={{ backgroundColor: colors.bgTertiary, borderColor: colors.border }} />
                <ReferenceLine x="0" stroke={colors.textMuted} />
                <Bar dataKey="count">
                  {(distributionQuery.data?.buckets ?? []).map((bucket) => (
                    <Cell key={bucket.range} fill={bucket.range.startsWith("-") ? colors.red : colors.green} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <section>
        <h2 className="mb-3 text-xl font-semibold">Performance nach Symbol</h2>
        <DataTable columns={symbolColumns} rows={bySymbolQuery.data?.rows ?? []} emptyState={<p>Keine Symbol-Performance verfugbar.</p>} />
      </section>
      <section>
        <h2 className="mb-3 text-xl font-semibold">Performance nach Exit Reason</h2>
        <DataTable columns={exitColumns} rows={byExitReasonQuery.data?.rows ?? []} emptyState={<p>Keine Exit-Reason-Performance verfugbar.</p>} />
      </section>
    </div>
  )
}
