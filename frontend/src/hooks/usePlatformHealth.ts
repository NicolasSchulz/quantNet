import { useQuery } from "@tanstack/react-query"

import { healthClient } from "../api/client"
import type { PlatformHealth } from "../types"

export const usePlatformHealth = () =>
  useQuery({
    queryKey: ["platform-health"],
    queryFn: async () => {
      const { data } = await healthClient.get<PlatformHealth>("/health")
      return data
    },
  })
