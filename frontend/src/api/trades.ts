import { apiClient, cleanParams } from "./client"
import type { PriceContextResponse, Trade, TradeListResponse, TradesSummary } from "../types"

export interface TradeFilters {
  symbol?: string
  direction?: "LONG" | "SHORT" | ""
  status?: "OPEN" | "CLOSED" | "" | "CANCELLED"
  start_date?: string
  end_date?: string
  strategy?: string
  page?: number
  page_size?: number
}

export const fetchTrades = async (filters: TradeFilters) => {
  const { data } = await apiClient.get<TradeListResponse>("/trades", { params: cleanParams(filters) })
  return data
}

export const fetchTradeSummary = async () => {
  const { data } = await apiClient.get<TradesSummary>("/trades/summary")
  return data
}

export const fetchTradeDetail = async (tradeId: string) => {
  const { data } = await apiClient.get<Trade>(`/trades/${tradeId}`)
  return data
}

export const fetchTradePriceContext = async (tradeId: string) => {
  const { data } = await apiClient.get<PriceContextResponse>(`/trades/price-context/${tradeId}`)
  return data
}
