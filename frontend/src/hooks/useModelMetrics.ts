import { useQuery } from "@tanstack/react-query"

import {
  fetchConfusionMatrix,
  fetchDefaultModelMetrics,
  fetchFeatureImportance,
  fetchFoldResults,
  fetchModelList,
  fetchModelMetrics,
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
