from __future__ import annotations

from fastapi import APIRouter, HTTPException

try:
    from api.schemas.model import (
        ConfusionMatrixData,
        FeatureImportanceResponse,
        FoldResultsResponse,
        ModelListResponse,
        ModelMetrics,
    )
    from api.services.model_service import (
        get_confusion_matrix,
        get_default_model,
        get_feature_importance,
        get_fold_results,
        get_model_metrics,
        list_models,
    )
except ImportError:
    from schemas.model import (
        ConfusionMatrixData,
        FeatureImportanceResponse,
        FoldResultsResponse,
        ModelListResponse,
        ModelMetrics,
    )
    from services.model_service import (
        get_confusion_matrix,
        get_default_model,
        get_feature_importance,
        get_fold_results,
        get_model_metrics,
        list_models,
    )

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/list", response_model=ModelListResponse)
def model_list() -> dict[str, object]:
    return list_models()


@router.get("/default/{symbol}", response_model=ModelMetrics)
def default_model(symbol: str) -> dict[str, object]:
    metrics = get_default_model(symbol)
    if metrics is None:
        raise HTTPException(status_code=404, detail="Default model not found")
    return metrics


@router.get("/{model_id}/metrics", response_model=ModelMetrics)
def model_metrics(model_id: str) -> dict[str, object]:
    metrics = get_model_metrics(model_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return metrics


@router.get("/{model_id}/confusion-matrix", response_model=ConfusionMatrixData)
def model_confusion_matrix(model_id: str) -> dict[str, object]:
    matrix = get_confusion_matrix(model_id)
    if matrix is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return matrix


@router.get("/{model_id}/fold-results", response_model=FoldResultsResponse)
def model_fold_results(model_id: str) -> dict[str, object]:
    folds = get_fold_results(model_id)
    if folds is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return folds


@router.get("/{model_id}/feature-importance", response_model=FeatureImportanceResponse)
def model_feature_importance(model_id: str) -> dict[str, object]:
    features = get_feature_importance(model_id)
    if features is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return features
