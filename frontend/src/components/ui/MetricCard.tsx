import { cn } from "../../utils/formatters"

export function MetricCard({
  title,
  value,
  tone = "neutral",
  subtitle,
}: {
  title: string
  value: string
  tone?: "positive" | "negative" | "neutral" | "warning"
  subtitle?: string
}) {
  const toneClass =
    tone === "positive" ? "text-success" : tone === "negative" ? "text-danger" : tone === "warning" ? "text-warning" : "text-textPrimary"
  return (
    <div className="rounded-2xl border border-border bg-secondary p-5 shadow-card transition duration-150 hover:bg-tertiary">
      <p className="text-sm text-textSecondary">{title}</p>
      <p className={cn("mt-3 text-3xl font-semibold tracking-tight", toneClass)}>{value}</p>
      {subtitle ? <p className="mt-2 text-xs text-textMuted">{subtitle}</p> : null}
    </div>
  )
}
