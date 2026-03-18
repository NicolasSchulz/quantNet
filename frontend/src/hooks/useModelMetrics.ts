import { useQuery } from "@tanstack/react-query"

import {
  fetchConfusionMatrix,
  fetchDefaultModelMetrics,
  fetchFeatureImportance,
  fetchFoldResults,
  fetchLossCurves,
  fetchModelList,
  fetchModelMetrics,
  fetchTrainingProgress,
  fetchThresholdAnalysis,
  fetchTrainOOSComparison,
} from "../api/model"

export const useModelList = () =>
  useQuery({
    queryKey: ["model-list"],
    queryFn: fetchModelList,
  })

export const useDefaultModelMetrics = (symbol: string) =>
  useQuery({
    queryKey: ["default-model", symbol],
    queryFn: () => fetchDefaultModelMetrics(symbol),
    enabled: Boolean(symbol),
  })

export const useModelMetrics = (modelId: string | null) =>
  useQuery({
    queryKey: ["model-metrics", modelId],
    queryFn: () => fetchModelMetrics(modelId as string),
    enabled: Boolean(modelId),
  })

export const useConfusionMatrix = (modelId: string | null) =>
  useQuery({
    queryKey: ["confusion-matrix", modelId],
    queryFn: () => fetchConfusionMatrix(modelId as string),
    enabled: Boolean(modelId),
  })

export const useFoldResults = (modelId: string | null) =>
  useQuery({
    queryKey: ["fold-results", modelId],
    queryFn: () => fetchFoldResults(modelId as string),
    enabled: Boolean(modelId),
  })

export const useFeatureImportance = (modelId: string | null) =>
  useQuery({
    queryKey: ["feature-importance", modelId],
    queryFn: () => fetchFeatureImportance(modelId as string),
    enabled: Boolean(modelId),
  })

export const useTrainOOSComparison = (modelId: string | null) =>
  useQuery({
    queryKey: ["train-oos-comparison", modelId],
    queryFn: () => fetchTrainOOSComparison(modelId as string),
    enabled: Boolean(modelId),
  })

export const useThresholdAnalysis = (modelId: string | null) =>
  useQuery({
    queryKey: ["threshold-analysis", modelId],
    queryFn: () => fetchThresholdAnalysis(modelId as string),
    enabled: Boolean(modelId),
  })

export const useLossCurves = (modelId: string | null) =>
  useQuery({
    queryKey: ["loss-curves", modelId],
    queryFn: () => fetchLossCurves(modelId as string),
    enabled: Boolean(modelId),
    staleTime: 300_000,
    retry: 1,
  })

export const useTrainingProgress = () =>
  useQuery({
    queryKey: ["training-progress"],
    queryFn: fetchTrainingProgress,
    refetchInterval: 3000,
  })
