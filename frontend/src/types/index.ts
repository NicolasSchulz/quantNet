export interface Trade {
  id: string
  symbol: string
  asset_class: "equity" | "crypto"
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
  stability: StabilityAnalysis
  overfitting: OverfittingAnalysis
  threshold_analysis: ThresholdAnalysis
  roc_auc_long: number | null
  roc_auc_short: number | null
}

export interface StabilityAnalysis {
  sharpe_mean: number
  sharpe_std: number
  sharpe_cv: number
  pct_profitable_folds: number
  worst_fold_sharpe: number
  best_fold_sharpe: number
  stability_risk: "LOW" | "MEDIUM" | "HIGH"
}

export interface OverfittingAnalysis {
  train_sharpe_mean: number
  oos_sharpe_mean: number
  sharpe_gap: number
  sharpe_gap_pct: number
  train_accuracy_mean: number
  oos_accuracy_mean: number
  accuracy_gap: number
  overfitting_verdict: "NONE" | "MILD" | "MODERATE" | "SEVERE"
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
  val_start: string
  val_end: string
  test_start: string
  test_end: string
  train_accuracy: number
  train_sharpe: number
  train_f1_macro: number
  oos_accuracy: number
  oos_sharpe: number
  oos_f1_macro: number
  accuracy_gap: number
  sharpe_gap: number
  overfitting_flag: boolean
  optimal_threshold: number
  threshold_source: "fixed" | "val_optimized"
  n_trades: number
  cagr: number
}

export interface ThresholdAnalysis {
  threshold_per_fold: Array<{ fold_id: number; threshold: number }>
  mean: number
  std: number
  min_threshold: number
  max_threshold: number
  is_stable: boolean
  threshold_source: "fixed" | "val_optimized"
}

export interface ParamsStability {
  param_name: string
  values: Record<string, number>
  most_common: string | number
  stability_pct: number
}

export interface TrainOOSComparison {
  folds: FoldResult[]
  summary: OverfittingAnalysis
  chart_data: {
    fold_labels: string[]
    train_sharpes: number[]
    oos_sharpes: number[]
    gaps: number[]
    overfitting_flags: boolean[]
  }
}

export interface FoldLossHistory {
  fold_id: number
  train_start: string
  train_end: string
  test_start: string
  test_end: string
  metric: string
  train: number[]
  validation: number[]
  best_iteration: number
  n_iterations: number
  early_stopped: boolean
  final_train_loss: number
  final_val_loss: number
  min_val_loss: number
  min_val_iteration: number
  overfit_gap: number
}

export interface AggregateLoss {
  metric: string
  train_mean: number[]
  train_std: number[]
  val_mean: number[]
  val_std: number[]
  min_iterations: number
  max_iterations: number
  common_iterations: number
  n_folds_included: number
}

export interface LossCurvesData {
  model_id: string
  n_folds: number
  aggregate: AggregateLoss
  folds: FoldLossHistory[]
}

export interface TrainingProgress {
  status: "idle" | "running" | "completed" | "failed"
  symbol: string | null
  feature_version: string | null
  optimize: boolean
  phase: string
  message: string
  percent: number
  updated_at: string
  outer_fold?: number | null
  inner_combo_index?: number | null
  inner_combo_total?: number | null
  completed_steps?: number | null
  total_steps?: number | null
  model_id?: string | null
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
  equity_points: EquityPoint[]
  crypto_points: EquityPoint[]
  combined_points: EquityPoint[]
}

export interface PnlDistributionBucket {
  range: string
  count: number
  pct: number
}

export interface SymbolPerformanceRow {
  symbol: string
  asset_class: "equity" | "crypto"
  trades: number
  win_rate: number
  avg_pnl: number
  total_pnl: number
  sharpe: number
}

export interface CryptoPosition {
  symbol: string
  base_asset: string
  quantity: number
  quantity_usd: number
  avg_entry_price: number
  current_price: number
  unrealized_pnl: number
  unrealized_pnl_pct: number
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
