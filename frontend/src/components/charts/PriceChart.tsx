import { createChart, type CandlestickData, LineStyle, type Time, type UTCTimestamp } from "lightweight-charts"
import { useEffect, useRef } from "react"

import { colors } from "../../utils/colors"
import type { PriceContextResponse } from "../../types"

export function PriceChart({ data, height = 280 }: { data: PriceContextResponse; height?: number }) {
  const ref = useRef<HTMLDivElement | null>(null)
  const toChartTime = (value: string): Time => Math.floor(new Date(value).getTime() / 1000) as UTCTimestamp

  useEffect(() => {
    if (!ref.current) return
    const chart = createChart(ref.current, {
      height,
      layout: { background: { color: colors.bgSecondary }, textColor: colors.textSecondary },
      grid: { vertLines: { color: colors.border }, horzLines: { color: colors.border } },
      timeScale: { borderColor: colors.border },
      rightPriceScale: { borderColor: colors.border },
    })
    const series = chart.addCandlestickSeries({
      upColor: colors.green,
      downColor: colors.red,
      borderVisible: false,
      wickUpColor: colors.green,
      wickDownColor: colors.red,
    })
    const entryLine = chart.addLineSeries({ color: colors.green, lineStyle: LineStyle.Dashed, lineWidth: 2 })
    const exitLine = chart.addLineSeries({ color: colors.red, lineStyle: LineStyle.Dashed, lineWidth: 2 })
    const resizeObserver = new ResizeObserver(() => {
      if (ref.current) {
        chart.applyOptions({ width: ref.current.clientWidth })
      }
    })
    resizeObserver.observe(ref.current)

    series.setData(
      data.bars.map<CandlestickData>((bar) => ({
        time: toChartTime(bar.time),
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      })),
    )
    entryLine.setData(
      data.bars.map((bar) => ({ time: toChartTime(bar.time), value: data.entry_price })),
    )
    if (data.exit_price !== null) {
      exitLine.setData(
        data.bars.map((bar) => ({ time: toChartTime(bar.time), value: data.exit_price as number })),
      )
    }
    chart.timeScale().fitContent()
    return () => {
      resizeObserver.disconnect()
      chart.remove()
    }
  }, [data, height])

  return <div ref={ref} className="w-full" />
}
