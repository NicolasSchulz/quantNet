from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


StabilityRisk = Literal["LOW", "MEDIUM", "HIGH"]
OverfittingVerdict = Literal["NONE", "MILD", "MODERATE", "SEVERE"]
ThresholdSource = Literal["fixed", "val_optimized"]


class ModelListItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    symbol: str
    trained_at: datetime
    sharpe: float
    status: str


class ModelListResponse(BaseModel):
    models: list[ModelListItem]


class StabilityAnalysis(BaseModel):
    sharpe_mean: float
    sharpe_std: float
    sharpe_cv: float
    pct_profitable_folds: float
    worst_fold_sharpe: float
    best_fold_sharpe: float
    stability_risk: StabilityRisk


class OverfittingAnalysis(BaseModel):
    train_sharpe_mean: float
    oos_sharpe_mean: float
    sharpe_gap: float
    sharpe_gap_pct: float
    train_accuracy_mean: float
    oos_accuracy_mean: float
    accuracy_gap: float
    overfitting_verdict: OverfittingVerdict


class ThresholdPoint(BaseModel):
    fold_id: int
    threshold: float


class ThresholdAnalysis(BaseModel):
    threshold_per_fold: list[ThresholdPoint]
    mean: float
    std: float
    min_threshold: float
    max_threshold: float
    is_stable: bool
    threshold_source: ThresholdSource


class ModelMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    symbol: str
    trained_at: datetime
    feature_version: str
    n_folds: int
    train_window_days: int
    test_window_days: int
    accuracy: float
    f1_macro: float
    f1_long: float
    f1_short: float
    precision_long: float
    recall_long: float
    precision_short: float
    recall_short: float
    oos_sharpe: float
    oos_cagr: float
    oos_max_drawdown: float
    oos_calmar: float
    n_trades: int
    win_rate: float
    stability: StabilityAnalysis
    overfitting: OverfittingAnalysis
    threshold_analysis: ThresholdAnalysis
    roc_auc_long: Optional[float] = None
    roc_auc_short: Optional[float] = None


class ConfusionMatrixData(BaseModel):
    matrix: list[list[int]]
    labels: list[str]
    normalized: list[list[float]]


class FoldResult(BaseModel):
    fold_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime
    train_accuracy: float
    train_sharpe: float
    train_f1_macro: float
    oos_accuracy: float
    oos_sharpe: float
    oos_f1_macro: float
    accuracy_gap: float
    sharpe_gap: float
    overfitting_flag: bool
    optimal_threshold: float
    threshold_source: ThresholdSource
    n_trades: int
    cagr: float


class FoldResultsResponse(BaseModel):
    folds: list[FoldResult]


class TrainOOSChartData(BaseModel):
    fold_labels: list[str]
    train_sharpes: list[float]
    oos_sharpes: list[float]
    gaps: list[float]
    overfitting_flags: list[bool]


class TrainOOSComparisonResponse(BaseModel):
    folds: list[FoldResult]
    summary: OverfittingAnalysis
    chart_data: TrainOOSChartData


class FoldLossHistory(BaseModel):
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metric: str
    train: list[float]
    validation: list[float]
    best_iteration: int
    n_iterations: int
    early_stopped: bool
    final_train_loss: float
    final_val_loss: float
    min_val_loss: float
    min_val_iteration: int
    overfit_gap: float


class AggregateLoss(BaseModel):
    metric: str
    train_mean: list[float]
    train_std: list[float]
    val_mean: list[float]
    val_std: list[float]
    min_iterations: int
    max_iterations: int
    common_iterations: int
    n_folds_included: int


class LossCurvesResponse(BaseModel):
    model_id: str
    n_folds: int
    aggregate: AggregateLoss
    folds: list[FoldLossHistory]


class TrainingProgressResponse(BaseModel):
    status: Literal["idle", "running", "completed", "failed"]
    symbol: Optional[str] = None
    feature_version: Optional[str] = None
    optimize: bool = False
    phase: str
    message: str
    percent: float
    updated_at: datetime
    outer_fold: Optional[int] = None
    inner_combo_index: Optional[int] = None
    inner_combo_total: Optional[int] = None
    completed_steps: Optional[int] = None
    total_steps: Optional[int] = None
    model_id: Optional[str] = None


class FeatureImportance(BaseModel):
    feature: str
    importance: float
    std: float


class FeatureImportanceResponse(BaseModel):
    features: list[FeatureImportance]
