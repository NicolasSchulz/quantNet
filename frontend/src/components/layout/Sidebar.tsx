import { NavLink } from "react-router-dom"

import { StatusBadge } from "../ui/StatusBadge"

const links = [
  { to: "/trades", label: "Trade Book", icon: "📋" },
  { to: "/performance", label: "Performance", icon: "📈" },
  { to: "/model", label: "Model", icon: "🤖" },
]

export function Sidebar({ liveMode }: { liveMode: boolean }) {
  return (
    <aside className="flex min-h-screen w-[220px] flex-col border-r border-border bg-secondary px-5 py-6">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-textMuted">AlgoTrader</p>
        <h1 className="mt-2 text-2xl font-semibold text-textPrimary">quantNet</h1>
      </div>
      <nav className="mt-8 flex flex-1 flex-col gap-2">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-xl border px-4 py-3 text-sm font-medium transition ${
                isActive
                  ? "border-success/30 bg-tertiary text-textPrimary shadow-card"
                  : "border-transparent text-textSecondary hover:border-border hover:bg-primary/40 hover:text-textPrimary"
              }`
            }
          >
            <span>{link.icon}</span>
            <span>{link.label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="space-y-3 border-t border-border pt-5">
        <StatusBadge label={liveMode ? "LIVE" : "PAPER"} variant={liveMode ? "warning" : "info"} />
        <p className="text-xs text-textMuted">Version 0.1.0</p>
      </div>
    </aside>
  )
}
