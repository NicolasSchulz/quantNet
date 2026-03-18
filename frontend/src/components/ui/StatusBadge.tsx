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

export function AssetClassBadge({ assetClass }: { assetClass: "equity" | "crypto" }) {
  const style =
    assetClass === "crypto"
      ? "border-orange-400/30 bg-orange-400/10 text-orange-300"
      : "border-sky-400/30 bg-sky-400/10 text-sky-300"
  const label = assetClass === "crypto" ? "crypto" : "equity"
  const icon = assetClass === "crypto" ? "🔶" : "●"
  return <span className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide", style)}>{icon} {label}</span>
}
