import { useEffect, useState } from "react"

export function ApiErrorToast() {
  const [message, setMessage] = useState<string | null>(null)

  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<string>).detail
      setMessage(detail)
      window.setTimeout(() => setMessage(null), 3000)
    }
    window.addEventListener("api-error", handler)
    return () => window.removeEventListener("api-error", handler)
  }, [])

  if (!message) {
    return null
  }

  return <div className="fixed bottom-6 right-6 z-50 rounded-xl border border-danger/30 bg-secondary px-4 py-3 text-sm text-danger shadow-card">{message}</div>
}
