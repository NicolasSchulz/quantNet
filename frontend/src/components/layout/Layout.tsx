import { Outlet } from "react-router-dom"

import { usePlatformHealth } from "../../hooks/usePlatformHealth"
import { ApiErrorToast } from "../ui/ApiErrorToast"
import { Sidebar } from "./Sidebar"
import { TopBar } from "./TopBar"

export function Layout() {
  const { data } = usePlatformHealth()

  return (
    <div className="flex min-h-screen bg-primary text-textPrimary">
      <Sidebar liveMode={Boolean(data?.platform.live_mode)} />
      <div className="flex min-h-screen flex-1 flex-col">
        <TopBar health={data} />
        <main className="flex-1 overflow-y-auto px-8 py-8">
          <Outlet />
        </main>
      </div>
      <ApiErrorToast />
    </div>
  )
}
