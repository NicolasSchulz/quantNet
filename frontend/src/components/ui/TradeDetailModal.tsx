import { PriceChart } from "../charts/PriceChart"
import { useTradeDetail, useTradePriceContext } from "../../hooks/useTrades"
import { formatCurrency, formatPercent, formatUtcToLocal } from "../../utils/formatters"
import { LoadingSpinner } from "./LoadingSpinner"
import { StatusBadge } from "./StatusBadge"

export function TradeDetailModal({ tradeId, onClose }: { tradeId: string | null; onClose: () => void }) {
  const detailQuery = useTradeDetail(tradeId)
  const chartQuery = useTradePriceContext(tradeId)

  if (!tradeId) return null

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 p-8">
      <div className="max-h-[90vh] w-full max-w-5xl overflow-y-auto rounded-3xl border border-border bg-secondary p-6 shadow-card">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-textMuted">Trade Detail</p>
            <h2 className="mt-2 text-2xl font-semibold text-textPrimary">{detailQuery.data?.symbol ?? "Loading..."}</h2>
          </div>
          <button type="button" className="rounded-lg border border-border px-3 py-2 text-sm text-textSecondary hover:bg-tertiary" onClick={onClose}>
            Close
          </button>
        </div>
        {!detailQuery.data || !chartQuery.data ? (
          <LoadingSpinner label="Loading trade details..." />
        ) : (
          <div className="mt-6 space-y-6">
            <div className="grid gap-4 md:grid-cols-4">
              <div>
                <p className="text-xs uppercase tracking-wide text-textMuted">Direction</p>
                <div className="mt-2">
                  <StatusBadge label={detailQuery.data.direction} variant={detailQuery.data.direction === "LONG" ? "success" : "danger"} />
                </div>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wide text-textMuted">Status</p>
                <div className="mt-2">
                  <StatusBadge label={detailQuery.data.status} variant={detailQuery.data.status === "OPEN" ? "info" : "neutral"} />
                </div>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wide text-textMuted">P&L</p>
                <p className="mt-2 text-lg font-semibold">{formatCurrency(detailQuery.data.pnl)}</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wide text-textMuted">P&L %</p>
                <p className="mt-2 text-lg font-semibold">{formatPercent(detailQuery.data.pnl_pct)}</p>
              </div>
            </div>
            <PriceChart data={chartQuery.data} />
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <div>
                <p className="text-xs uppercase tracking-wide text-textMuted">Entry Time</p>
                <p className="mt-2 text-sm text-textPrimary">{formatUtcToLocal(detailQuery.data.entry_time)}</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wide text-textMuted">Exit Time</p>
                <p className="mt-2 text-sm text-textPrimary">{formatUtcToLocal(detailQuery.data.exit_time)}</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wide text-textMuted">Entry Price</p>
                <p className="mt-2 text-sm text-textPrimary">{formatCurrency(detailQuery.data.entry_price)}</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wide text-textMuted">Exit Price</p>
                <p className="mt-2 text-sm text-textPrimary">{formatCurrency(detailQuery.data.exit_price)}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
