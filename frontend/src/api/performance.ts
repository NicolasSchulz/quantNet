import { apiClient, cleanParams } from "./client"
import type {
  EquityCurveResponse,
  ExitReasonPerformanceRow,
  PnlDistributionBucket,
  SymbolPerformanceRow,
  TradingStats,
} from "../types"

export interface PerformanceFilters {
  start_date?: string
  end_date?: string
  symbol?: string
  strategy?: string
}

export const fetchTradingStats = async (filters: PerformanceFilters) => {
  const { data } = await apiClient.get<TradingStats>("/performance/stats", { params: cleanParams(filters) })
  return data
}

export const fetchEquityCurve = async (filters: Pick<PerformanceFilters, "start_date" | "end_date">) => {
  const { data } = await apiClient.get<EquityCurveResponse>("/performance/equity-curve", { params: cleanParams(filters) })
  return data
}

export const fetchPnlDistribution = async () => {
  const { data } = await apiClient.get<{ buckets: PnlDistributionBucket[] }>("/performance/pnl-distribution")
  return data
}

export const fetchPerformanceBySymbol = async () => {
  const { data } = await apiClient.get<{ rows: SymbolPerformanceRow[] }>("/performance/by-symbol")
  return data
}

export const fetchPerformanceByExitReason = async () => {
  const { data } = await apiClient.get<{ rows: ExitReasonPerformanceRow[] }>("/performance/by-exit-reason")
  return data
}
