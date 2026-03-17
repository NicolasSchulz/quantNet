import { formatDistanceToNowStrict } from "date-fns"
import { formatInTimeZone } from "date-fns-tz"

export const formatCurrency = (value: number | null | undefined) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value ?? 0)

export const formatPercent = (value: number | null | undefined, decimals = 2) =>
  `${((value ?? 0) * 100).toFixed(decimals)}%`

export const formatDecimal = (value: number | null | undefined, decimals = 2) =>
  (value ?? 0).toFixed(decimals)

export const formatUtcToLocal = (value: string | null | undefined, pattern = "yyyy-MM-dd HH:mm") =>
  value ? formatInTimeZone(new Date(value), Intl.DateTimeFormat().resolvedOptions().timeZone, pattern) : "—"

export const formatRelativeTime = (value: string) =>
  formatDistanceToNowStrict(new Date(value), { addSuffix: true })

export const cn = (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" ")
