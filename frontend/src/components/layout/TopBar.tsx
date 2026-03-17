import { differenceInSeconds } from "date-fns"
import { formatInTimeZone } from "date-fns-tz"
import { useEffect, useMemo, useState } from "react"

import type { PlatformHealth } from "../../types"

function getMarketStatus(now: Date) {
  const dayOfWeek = Number(formatInTimeZone(now, "America/New_York", "i"))
  const time = formatInTimeZone(now, "America/New_York", "HH:mm")
  const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5
  const isOpen = isWeekday && time >= "09:30" && time <= "16:00"
  return { isOpen, label: isOpen ? "Market Open" : "Market Closed" }
}

export function TopBar({ health }: { health?: PlatformHealth }) {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const interval = window.setInterval(() => setNow(new Date()), 1000)
    return () => window.clearInterval(interval)
  }, [])
  const marketStatus = useMemo(() => getMarketStatus(now), [now])
  const updatedSeconds = health ? Math.max(differenceInSeconds(now, new Date(health.timestamp)), 0) : null

  return (
    <header className="flex items-center justify-between border-b border-border bg-primary/70 px-8 py-4 backdrop-blur">
      <div>
        <p className="text-xs uppercase tracking-[0.35em] text-textMuted">AlgoTrader</p>
        <p className="text-lg font-semibold text-textPrimary">Algorithmic Trading Platform</p>
      </div>
      <div className="flex items-center gap-3 rounded-full border border-border bg-secondary px-4 py-2 text-sm text-textPrimary">
        <span className={`h-2.5 w-2.5 rounded-full ${marketStatus.isOpen ? "bg-success" : "bg-danger"}`} />
        <span>{marketStatus.label}</span>
      </div>
      <div className="text-right text-sm text-textSecondary">
        <p>Updated {updatedSeconds !== null ? `${updatedSeconds}s ago` : "—"}</p>
        <p className="text-xs text-textMuted">{health?.platform.live_mode ? "Live backend" : "Mock backend"}</p>
      </div>
    </header>
  )
}
