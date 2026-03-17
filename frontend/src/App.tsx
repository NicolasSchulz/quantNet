import { Navigate, Route, Routes } from "react-router-dom"

import { Layout } from "./components/layout/Layout"
import { ModelPerformance } from "./pages/ModelPerformance"
import { TradeBook } from "./pages/TradeBook"
import { TradingPerformance } from "./pages/TradingPerformance"

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Navigate to="/trades" replace />} />
        <Route path="/trades" element={<TradeBook />} />
        <Route path="/performance" element={<TradingPerformance />} />
        <Route path="/model" element={<ModelPerformance />} />
      </Route>
    </Routes>
  )
}
