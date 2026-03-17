from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


OverfittingRisk = Literal["LOW", "MEDIUM", "HIGH"]


class ModelListItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    symbol: str
    trained_at: datetime
    sharpe: float
    status: str


class ModelListResponse(BaseModel):
    models: list[ModelListItem]


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
    sharpe_mean: float
    sharpe_std: float
    sharpe_cv: float
    pct_profitable_folds: float
    overfitting_risk: OverfittingRisk
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
    test_start: datetime
    test_end: datetime
    sharpe: float
    accuracy: float
    n_trades: int
    cagr: float


class FoldResultsResponse(BaseModel):
    folds: list[FoldResult]


class FeatureImportance(BaseModel):
    feature: str
    importance: float
    std: float


class FeatureImportanceResponse(BaseModel):
    features: list[FeatureImportance]
