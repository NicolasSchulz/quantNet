import { cn } from "../../utils/formatters"

type Variant = "success" | "danger" | "info" | "warning" | "neutral"

const variantClasses: Record<Variant, string> = {
  success: "bg-success/15 text-success border-success/30",
  danger: "bg-danger/15 text-danger border-danger/30",
  info: "bg-info/15 text-info border-info/30",
  warning: "bg-warning/15 text-warning border-warning/30",
  neutral: "bg-white/5 text-textSecondary border-white/10",
}

export function StatusBadge({ label, variant }: { label: string; variant: Variant }) {
  return <span className={cn("inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold uppercase tracking-wide", variantClasses[variant])}>{label}</span>
}
