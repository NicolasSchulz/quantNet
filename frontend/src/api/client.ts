import axios from "axios"

export const apiClient = axios.create({
  baseURL: `${import.meta.env.VITE_API_URL ?? "http://localhost:8000"}/api`,
  timeout: 10_000,
})

apiClient.interceptors.request.use((config) => {
  if (import.meta.env.DEV) {
    console.info(`[api] ${config.method?.toUpperCase()} ${config.url}`)
  }
  return config
})

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (typeof window !== "undefined") {
      window.dispatchEvent(
        new CustomEvent("api-error", {
          detail: error?.message ?? "Network error",
        }),
      )
    }
    return Promise.reject(error)
  },
)

export const healthClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "http://localhost:8000",
  timeout: 10_000,
})

export const cleanParams = <T extends object>(params: T) =>
  Object.fromEntries(
    Object.entries(params as Record<string, unknown>).filter(([, value]) => value !== "" && value !== null && value !== undefined),
  )
