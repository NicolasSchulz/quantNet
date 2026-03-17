import { useMemo, useState } from "react"

import { DataTable, type Column } from "../components/ui/DataTable"
import { ErrorState } from "../components/ui/ErrorState"
import { LoadingSpinner } from "../components/ui/LoadingSpinner"
import { MetricCard } from "../components/ui/MetricCard"
import { MetricCardGrid } from "../components/ui/MetricCardGrid"
import { StatusBadge } from "../components/ui/StatusBadge"
import { TradeDetailModal } from "../components/ui/TradeDetailModal"
import { useTradeSummary, useTrades } from "../hooks/useTrades"
import type { Trade } from "../types"
import { cn, formatCurrency, formatPercent, formatUtcToLocal } from "../utils/formatters"

type SortKey = keyof Pick<Trade, "symbol" | "direction" | "status" | "entry_time" | "exit_time" | "entry_price" | "exit_price" | "quantity" | "pnl" | "pnl_pct" | "commission" | "exit_reason" | "signal_confidence" | "strategy">

export function TradeBook() {
  const [filters, setFilters] = useState({
    symbol: "",
    direction: "" as "" | "LONG" | "SHORT",
    status: "" as "" | "OPEN" | "CLOSED",
    start_date: "",
    end_date: "",
    strategy: "",
    page: 1,
    page_size: 25,
  })
  const [sortKey, setSortKey] = useState<SortKey>("entry_time")
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc")
  const [selectedTradeId, setSelectedTradeId] = useState<string | null>(null)

  const tradesQuery = useTrades(filters)
  const summaryQuery = useTradeSummary()

  const rows = useMemo(() => {
    const trades = [...(tradesQuery.data?.trades ?? [])]
    trades.sort((left, right) => {
      const leftValue = left[sortKey]
      const rightValue = right[sortKey]
      if (leftValue === rightValue) return 0
      if (leftValue === null) return 1
      if (rightValue === null) return -1
      const comparison = leftValue > rightValue ? 1 : -1
      return sortDirection === "asc" ? comparison : -comparison
    })
    return trades
  }, [sortDirection, sortKey, tradesQuery.data?.trades])

  const symbols = useMemo(() => Array.from(new Set((tradesQuery.data?.trades ?? []).map((trade) => trade.symbol))).sort(), [tradesQuery.data?.trades])
  const strategies = useMemo(() => Array.from(new Set((tradesQuery.data?.trades ?? []).map((trade) => trade.strategy))).sort(), [tradesQuery.data?.trades])

  const columns: Column<Trade>[] = [
    { key: "id", header: "#", render: (row) => row.id.slice(0, 8) },
    { key: "symbol", header: "Symbol", sortable: true, render: (row) => row.symbol },
    {
      key: "direction",
      header: "Direction",
      sortable: true,
      render: (row) => <StatusBadge label={row.direction} variant={row.direction === "LONG" ? "success" : "danger"} />,
    },
    {
      key: "status",
      header: "Status",
      sortable: true,
      render: (row) => <StatusBadge label={row.status} variant={row.status === "OPEN" ? "info" : row.status === "CLOSED" ? "neutral" : "warning"} />,
    },
    { key: "entry_time", header: "Entry Time", sortable: true, render: (row) => formatUtcToLocal(row.entry_time) },
    { key: "exit_time", header: "Exit Time", sortable: true, render: (row) => formatUtcToLocal(row.exit_time) },
    { key: "entry_price", header: "Entry Price", sortable: true, render: (row) => formatCurrency(row.entry_price) },
    { key: "exit_price", header: "Exit Price", sortable: true, render: (row) => formatCurrency(row.exit_price) },
    { key: "quantity", header: "Quantity", sortable: true, render: (row) => row.quantity.toFixed(2) },
    {
      key: "pnl",
      header: "P&L (USD)",
      sortable: true,
      render: (row) => <span className={cn("font-semibold", (row.pnl ?? 0) >= 0 ? "text-success" : "text-danger")}>{formatCurrency(row.pnl)}</span>,
    },
    {
      key: "pnl_pct",
      header: "P&L (%)",
      sortable: true,
      render: (row) => (
        <span className={cn("font-semibold", (row.pnl_pct ?? 0) >= 0 ? "text-success" : "text-danger")}>
          {(row.pnl_pct ?? 0) >= 0 ? "↑ " : "↓ "}
          {formatPercent(row.pnl_pct)}
        </span>
      ),
    },
    { key: "commission", header: "Commission", sortable: true, render: (row) => formatCurrency(row.commission) },
    {
      key: "exit_reason",
      header: "Exit Reason",
      sortable: true,
      render: (row) =>
        row.exit_reason ? (
          <StatusBadge
            label={row.exit_reason}
            variant={row.exit_reason === "TAKE_PROFIT" ? "success" : row.exit_reason === "STOP_LOSS" ? "danger" : row.exit_reason === "TIME_BARRIER" ? "warning" : "info"}
          />
        ) : (
          "—"
        ),
    },
    {
      key: "signal_confidence",
      header: "Confidence",
      sortable: true,
      render: (row) =>
        row.signal_confidence !== null ? (
          <div className="w-24">
            <div className="h-2 rounded-full bg-primary">
              <div className="h-2 rounded-full bg-info" style={{ width: `${row.signal_confidence * 100}%` }} />
            </div>
            <p className="mt-1 text-xs text-textSecondary">{Math.round(row.signal_confidence * 100)}%</p>
          </div>
        ) : (
          "—"
        ),
    },
    { key: "strategy", header: "Strategy", sortable: true, render: (row) => row.strategy },
  ]

  const setFilter = (key: keyof typeof filters, value: string | number) => setFilters((current) => ({ ...current, [key]: value, page: 1 }))

  if (tradesQuery.isLoading || summaryQuery.isLoading) {
    return <LoadingSpinner label="Loading trade book..." />
  }

  if (tradesQuery.isError || summaryQuery.isError) {
    return <ErrorState message="Backend nicht erreichbar – starte api/main.py" onRetry={() => void tradesQuery.refetch()} />
  }

  const summary = summaryQuery.data

  return (
    <div className="space-y-6">
      <MetricCardGrid columns={4}>
        <MetricCard title="Offene Positionen" value={`${summary?.open ?? 0}`} tone="neutral" subtitle="Aktive Trades im Book" />
        <MetricCard title="Heutige P&L" value={formatCurrency(summary?.today_pnl)} tone={(summary?.today_pnl ?? 0) >= 0 ? "positive" : "negative"} subtitle={formatPercent((summary?.today_pnl ?? 0) / 10_000)} />
        <MetricCard title="Gesamte realisierte P&L" value={formatCurrency(summary?.total_pnl)} tone={(summary?.total_pnl ?? 0) >= 0 ? "positive" : "negative"} />
        <MetricCard title="Win Rate" value={formatPercent(summary?.win_rate)} tone={(summary?.win_rate ?? 0) > 0.5 ? "positive" : "negative"} />
      </MetricCardGrid>

      <section className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
        <div className="grid gap-3 xl:grid-cols-6 md:grid-cols-3">
          <select className="rounded-xl border border-border bg-primary px-3 py-2 text-sm" value={filters.symbol} onChange={(event) => setFilter("symbol", event.target.value)}>
            <option value="">Alle Symbole</option>
            {symbols.map((symbol) => (
              <option key={symbol} value={symbol}>
                {symbol}
              </option>
            ))}
          </select>
          <select className="rounded-xl border border-border bg-primary px-3 py-2 text-sm" value={filters.direction} onChange={(event) => setFilter("direction", event.target.value)}>
            <option value="">Alle Richtungen</option>
            <option value="LONG">Long</option>
            <option value="SHORT">Short</option>
          </select>
          <select className="rounded-xl border border-border bg-primary px-3 py-2 text-sm" value={filters.status} onChange={(event) => setFilter("status", event.target.value)}>
            <option value="">Alle Status</option>
            <option value="OPEN">Open</option>
            <option value="CLOSED">Closed</option>
          </select>
          <input className="rounded-xl border border-border bg-primary px-3 py-2 text-sm" type="date" value={filters.start_date} onChange={(event) => setFilter("start_date", event.target.value)} />
          <input className="rounded-xl border border-border bg-primary px-3 py-2 text-sm" type="date" value={filters.end_date} onChange={(event) => setFilter("end_date", event.target.value)} />
          <select className="rounded-xl border border-border bg-primary px-3 py-2 text-sm" value={filters.strategy} onChange={(event) => setFilter("strategy", event.target.value)}>
            <option value="">Alle Strategien</option>
            {strategies.map((strategy) => (
              <option key={strategy} value={strategy}>
                {strategy}
              </option>
            ))}
          </select>
        </div>
        <div className="mt-4 flex items-center justify-between">
          <button
            type="button"
            className="rounded-xl border border-border px-4 py-2 text-sm text-textSecondary hover:bg-primary/50"
            onClick={() =>
              setFilters({
                symbol: "",
                direction: "",
                status: "",
                start_date: "",
                end_date: "",
                strategy: "",
                page: 1,
                page_size: 25,
              })
            }
          >
            Reset Filters
          </button>
          <p className="text-sm text-textSecondary">
            Zeige {(filters.page - 1) * filters.page_size + 1}-{Math.min(filters.page * filters.page_size, tradesQuery.data?.total ?? 0)} von {tradesQuery.data?.total ?? 0} Trades
          </p>
        </div>
      </section>

      <DataTable
        columns={columns}
        rows={rows}
        sortKey={sortKey}
        sortDirection={sortDirection}
        onSort={(key) => {
          if (sortKey === key) {
            setSortDirection((current) => (current === "asc" ? "desc" : "asc"))
          } else {
            setSortKey(key as SortKey)
            setSortDirection("asc")
          }
        }}
        onRowClick={(row) => setSelectedTradeId(row.id)}
        emptyState={<p>Noch keine Trades – starte den Paper Broker.</p>}
      />
      <div className="flex justify-end gap-3">
        <button
          type="button"
          className="rounded-lg border border-border px-3 py-2 text-sm disabled:opacity-40"
          onClick={() => setFilter("page", Math.max(filters.page - 1, 1))}
          disabled={filters.page <= 1}
        >
          Zuruck
        </button>
        <button
          type="button"
          className="rounded-lg border border-border px-3 py-2 text-sm disabled:opacity-40"
          onClick={() => setFilter("page", Math.min(filters.page + 1, tradesQuery.data?.pages ?? 1))}
          disabled={filters.page >= (tradesQuery.data?.pages ?? 1)}
        >
          Weiter
        </button>
      </div>
      <TradeDetailModal tradeId={selectedTradeId} onClose={() => setSelectedTradeId(null)} />
    </div>
  )
}
