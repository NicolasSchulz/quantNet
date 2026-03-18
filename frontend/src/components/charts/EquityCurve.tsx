import { createChart, type AreaData, type Time, type UTCTimestamp } from "lightweight-charts"
import { useEffect, useRef } from "react"

import type { EquityPoint } from "../../types"
import { colors } from "../../utils/colors"

interface ComparisonSeries {
  label: string
  color: string
  points: EquityPoint[]
}

export function EquityCurve({
  points,
  drawdown,
  height = 300,
  comparisonSeries = [],
}: {
  points: EquityPoint[]
  drawdown: EquityPoint[]
  height?: number
  comparisonSeries?: ComparisonSeries[]
}) {
  const ref = useRef<HTMLDivElement | null>(null)
  const normalizeSeries = (series: EquityPoint[], valueKey: "equity" | "drawdown") => {
    const deduped = new Map<number, number>()
    for (const point of series) {
      deduped.set(Math.floor(new Date(point.timestamp).getTime() / 1000), point[valueKey])
    }
    return [...deduped.entries()]
      .sort(([left], [right]) => left - right)
      .map(([time, value]) => ({ time: time as UTCTimestamp, value: valueKey === "drawdown" ? value * 100 : value }))
  }

  useEffect(() => {
    if (!ref.current) return
    const chart = createChart(ref.current, {
      height,
      layout: { background: { color: colors.bgSecondary }, textColor: colors.textSecondary },
      grid: { vertLines: { color: colors.border }, horzLines: { color: colors.border } },
      timeScale: { borderColor: colors.border },
      rightPriceScale: { borderColor: colors.border },
    })

    const equitySeries = chart.addAreaSeries({
      lineColor: colors.green,
      topColor: "rgba(0,212,170,0.35)",
      bottomColor: "rgba(68,136,255,0.05)",
      lineWidth: 2,
    })

    const drawdownSeries = chart.addLineSeries({
      color: colors.red,
      lineWidth: 2,
      priceScaleId: "left",
    })

    equitySeries.setData(normalizeSeries(points, "equity") as AreaData[])
    drawdownSeries.setData(normalizeSeries(drawdown, "drawdown"))

    for (const series of comparisonSeries) {
      const lineSeries = chart.addLineSeries({
        color: series.color,
        lineWidth: 2,
      })
      lineSeries.setData(normalizeSeries(series.points, "equity"))
    }

    const resizeObserver = new ResizeObserver(() => {
      if (ref.current) {
        chart.applyOptions({ width: ref.current.clientWidth })
      }
    })
    resizeObserver.observe(ref.current)
    chart.timeScale().fitContent()

    return () => {
      resizeObserver.disconnect()
      chart.remove()
    }
  }, [comparisonSeries, drawdown, height, points])

  return <div ref={ref} className="w-full" />
}
