import type { ReactNode } from "react"

export function MetricCardGrid({ children, columns = 4 }: { children: ReactNode; columns?: 2 | 4 }) {
  return <div className={`grid gap-4 ${columns === 4 ? "xl:grid-cols-4 md:grid-cols-2" : "xl:grid-cols-2"}`}>{children}</div>
}
