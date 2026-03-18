import { apiClient } from "./client"
import type {
  ConfusionMatrixData,
  FeatureImportance,
  FoldResult,
  LossCurvesData,
  ModelListItem,
  ModelMetrics,
  ThresholdAnalysis,
  TrainingProgress,
  TrainOOSComparison,
} from "../types"

export const fetchModelList = async () => {
  const { data } = await apiClient.get<{ models: ModelListItem[] }>("/model/list")
  return data
}

export const fetchModelMetrics = async (modelId: string) => {
  const { data } = await apiClient.get<ModelMetrics>(`/model/${modelId}/metrics`)
  return data
}

export const fetchDefaultModelMetrics = async (symbol: string) => {
  const { data } = await apiClient.get<ModelMetrics>(`/model/default/${symbol}`)
  return data
}

export const fetchConfusionMatrix = async (modelId: string) => {
  const { data } = await apiClient.get<ConfusionMatrixData>(`/model/${modelId}/confusion-matrix`)
  return data
}

export const fetchFoldResults = async (modelId: string) => {
  const { data } = await apiClient.get<{ folds: FoldResult[] }>(`/model/${modelId}/fold-results`)
  return data
}

export const fetchFeatureImportance = async (modelId: string) => {
  const { data } = await apiClient.get<{ features: FeatureImportance[] }>(`/model/${modelId}/feature-importance`)
  return data
}

export const fetchTrainOOSComparison = async (modelId: string) => {
  const { data } = await apiClient.get<TrainOOSComparison>(`/model/${modelId}/train-oos-comparison`)
  return data
}

export const fetchThresholdAnalysis = async (modelId: string) => {
  const { data } = await apiClient.get<ThresholdAnalysis>(`/model/${modelId}/threshold-analysis`)
  return data
}

export const fetchLossCurves = async (modelId: string) => {
  const { data } = await apiClient.get<LossCurvesData>(`/model/${modelId}/loss-curves`)
  return data
}

export const fetchTrainingProgress = async () => {
  const { data } = await apiClient.get<TrainingProgress>("/model/training/progress")
  return data
}
