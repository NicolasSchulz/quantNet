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
  asset_class?: "equity" | "crypto"
}

export const fetchTradingStats = async (filters: PerformanceFilters) => {
  const { data } = await apiClient.get<TradingStats>("/performance/stats", { params: cleanParams(filters) })
  return data
}

export const fetchEquityCurve = async (filters: Pick<PerformanceFilters, "start_date" | "end_date" | "asset_class">) => {
  const { data } = await apiClient.get<EquityCurveResponse>("/performance/equity-curve", { params: cleanParams(filters) })
  return data
}

export const fetchPnlDistribution = async (filters: Pick<PerformanceFilters, "asset_class">) => {
  const { data } = await apiClient.get<{ buckets: PnlDistributionBucket[] }>("/performance/pnl-distribution", { params: cleanParams(filters) })
  return data
}

export const fetchPerformanceBySymbol = async (filters: Pick<PerformanceFilters, "asset_class">) => {
  const { data } = await apiClient.get<{ rows: SymbolPerformanceRow[] }>("/performance/by-symbol", { params: cleanParams(filters) })
  return data
}

export const fetchPerformanceByExitReason = async (filters: Pick<PerformanceFilters, "asset_class">) => {
  const { data } = await apiClient.get<{ rows: ExitReasonPerformanceRow[] }>("/performance/by-exit-reason", { params: cleanParams(filters) })
  return data
}
