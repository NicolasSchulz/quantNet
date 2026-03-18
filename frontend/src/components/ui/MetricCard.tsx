import { cn } from "../../utils/formatters"

export function MetricCard({
  title,
  value,
  tone = "neutral",
  subtitle,
  compact = false,
  className,
}: {
  title: string
  value: string
  tone?: "positive" | "negative" | "neutral" | "warning"
  subtitle?: string
  compact?: boolean
  className?: string
}) {
  const toneClass =
    tone === "positive" ? "text-success" : tone === "negative" ? "text-danger" : tone === "warning" ? "text-warning" : "text-textPrimary"
  return (
    <div
      className={cn(
        "rounded-2xl border border-border bg-secondary p-5 shadow-card transition duration-150 hover:bg-tertiary",
        compact ? "flex min-h-[156px] flex-col justify-between" : "",
        className,
      )}
    >
      <div>
        <p className={cn("text-textSecondary", compact ? "text-xs uppercase tracking-wide" : "text-sm")}>{title}</p>
      </div>
      <div className={cn(compact ? "pt-6" : "")}>
        <p className={cn("font-semibold tracking-tight", compact ? "text-[2.15rem] leading-none" : "mt-3 text-3xl", toneClass)}>{value}</p>
        {subtitle ? <p className="mt-2 text-xs text-textMuted">{subtitle}</p> : null}
      </div>
    </div>
  )
}
