import { useQuery } from "@tanstack/react-query"

import { fetchTradeDetail, fetchTradePriceContext, fetchTradeSummary, fetchTrades, type TradeFilters } from "../api/trades"

export const useTrades = (filters: TradeFilters) =>
  useQuery({
    queryKey: ["trades", filters],
    queryFn: () => fetchTrades(filters),
  })

export const useTradeSummary = (filters: Pick<TradeFilters, "asset_class"> = {}) =>
  useQuery({
    queryKey: ["trade-summary", filters],
    queryFn: () => fetchTradeSummary(filters),
  })

export const useTradeDetail = (tradeId: string | null) =>
  useQuery({
    queryKey: ["trade-detail", tradeId],
    queryFn: () => fetchTradeDetail(tradeId as string),
    enabled: Boolean(tradeId),
  })

export const useTradePriceContext = (tradeId: string | null) =>
  useQuery({
    queryKey: ["trade-price-context", tradeId],
    queryFn: () => fetchTradePriceContext(tradeId as string),
    enabled: Boolean(tradeId),
  })
