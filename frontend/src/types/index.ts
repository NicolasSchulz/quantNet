export interface Trade {
  id: string
  symbol: string
  direction: "LONG" | "SHORT"
  status: "OPEN" | "CLOSED" | "CANCELLED"
  entry_time: string
  exit_time: string | null
  entry_price: number
  exit_price: number | null
  quantity: number
  pnl: number | null
  pnl_pct: number | null
  commission: number
  slippage: number
  strategy: string
  signal_confidence: number | null
  exit_reason: "TAKE_PROFIT" | "STOP_LOSS" | "TIME_BARRIER" | "MANUAL" | null
  notes: string | null
}

export interface TradingStats {
  total_trades: number
  open_trades: number
  closed_trades: number
  win_rate: number
  avg_win_pct: number
  avg_loss_pct: number
  max_win_pct: number
  max_loss_pct: number
  profit_factor: number
  sharpe_ratio: number
  max_drawdown_pct: number
  cagr: number
  calmar_ratio: number
  total_pnl: number
  total_commission: number
  avg_holding_hours: number
  best_trade: Trade | null
  worst_trade: Trade | null
}

export interface EquityPoint {
  timestamp: string
  equity: number
  drawdown: number
}

export interface ModelMetrics {
  model_id: string
  symbol: string
  trained_at: string
  feature_version: string
  n_folds: number
  train_window_days: number
  test_window_days: number
  accuracy: number
  f1_macro: number
  f1_long: number
  f1_short: number
  precision_long: number
  recall_long: number
  precision_short: number
  recall_short: number
  oos_sharpe: number
  oos_cagr: number
  oos_max_drawdown: number
  oos_calmar: number
  n_trades: number
  win_rate: number
  sharpe_mean: number
  sharpe_std: number
  sharpe_cv: number
  pct_profitable_folds: number
  overfitting_risk: "LOW" | "MEDIUM" | "HIGH"
  roc_auc_long: number | null
  roc_auc_short: number | null
}

export interface ConfusionMatrixData {
  matrix: number[][]
  labels: string[]
  normalized: number[][]
}

export interface FoldResult {
  fold_id: number
  train_start: string
  train_end: string
  test_start: string
  test_end: string
  sharpe: number
  accuracy: number
  n_trades: number
  cagr: number
}

export interface FeatureImportance {
  feature: string
  importance: number
  std: number
}

export interface TradeListResponse {
  trades: Trade[]
  total: number
  page: number
  pages: number
}

export interface TradesSummary {
  open: number
  closed: number
  today_pnl: number
  total_pnl: number
  win_rate: number
}

export interface PriceBar {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume?: number | null
}

export interface PriceContextResponse {
  bars: PriceBar[]
  entry_time: string
  exit_time: string | null
  entry_price: number
  exit_price: number | null
}

export interface EquityCurveResponse {
  points: EquityPoint[]
  benchmark: EquityPoint[]
}

export interface PnlDistributionBucket {
  range: string
  count: number
  pct: number
}

export interface SymbolPerformanceRow {
  symbol: string
  trades: number
  win_rate: number
  avg_pnl: number
  total_pnl: number
  sharpe: number
}

export interface ExitReasonPerformanceRow {
  exit_reason: string
  trades: number
  win_rate: number
  avg_pnl: number
  total_pnl: number
}

export interface ModelListItem {
  model_id: string
  symbol: string
  trained_at: string
  sharpe: number
  status: string
}

export interface PlatformHealth {
  status: string
  timestamp: string
  platform: {
    trades_loaded?: boolean
    model_registry_loaded?: boolean
    live_mode?: boolean
  }
}
