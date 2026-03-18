import { useQuery } from "@tanstack/react-query"

import {
  fetchEquityCurve,
  fetchPerformanceByExitReason,
  fetchPerformanceBySymbol,
  fetchPnlDistribution,
  fetchTradingStats,
  type PerformanceFilters,
} from "../api/performance"

export const useTradingStats = (filters: PerformanceFilters) =>
  useQuery({
    queryKey: ["trading-stats", filters],
    queryFn: () => fetchTradingStats(filters),
  })

export const useEquityCurve = (filters: Pick<PerformanceFilters, "start_date" | "end_date" | "asset_class">) =>
  useQuery({
    queryKey: ["equity-curve", filters],
    queryFn: () => fetchEquityCurve(filters),
  })

export const usePnlDistribution = (filters: Pick<PerformanceFilters, "asset_class">) =>
  useQuery({
    queryKey: ["pnl-distribution", filters],
    queryFn: () => fetchPnlDistribution(filters),
  })

export const usePerformanceBySymbol = (filters: Pick<PerformanceFilters, "asset_class">) =>
  useQuery({
    queryKey: ["performance-by-symbol", filters],
    queryFn: () => fetchPerformanceBySymbol(filters),
  })

export const usePerformanceByExitReason = (filters: Pick<PerformanceFilters, "asset_class">) =>
  useQuery({
    queryKey: ["performance-by-exit-reason", filters],
    queryFn: () => fetchPerformanceByExitReason(filters),
  })
