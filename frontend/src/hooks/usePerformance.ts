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

export const useEquityCurve = (filters: Pick<PerformanceFilters, "start_date" | "end_date">) =>
  useQuery({
    queryKey: ["equity-curve", filters],
    queryFn: () => fetchEquityCurve(filters),
  })

export const usePnlDistribution = () =>
  useQuery({
    queryKey: ["pnl-distribution"],
    queryFn: fetchPnlDistribution,
  })

export const usePerformanceBySymbol = () =>
  useQuery({
    queryKey: ["performance-by-symbol"],
    queryFn: fetchPerformanceBySymbol,
  })

export const usePerformanceByExitReason = () =>
  useQuery({
    queryKey: ["performance-by-exit-reason"],
    queryFn: fetchPerformanceByExitReason,
  })
