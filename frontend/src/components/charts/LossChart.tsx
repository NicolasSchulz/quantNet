import { Area, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import type { LossCurvesData } from "../../types"
import { colors } from "../../utils/colors"

interface LossChartProps {
  data: LossCurvesData
  selectedFoldId: number | "average"
  height?: number
}

function buildAverageRows(data: LossCurvesData) {
  const n = data.aggregate.common_iterations
  return Array.from({ length: n }, (_, index) => {
    const trainMean = data.aggregate.train_mean[index] ?? 0
    const trainStd = data.aggregate.train_std[index] ?? 0
    const valMean = data.aggregate.val_mean[index] ?? 0
    const valStd = data.aggregate.val_std[index] ?? 0
    return {
      iteration: index,
      train_mean: trainMean,
      train_lower: trainMean - trainStd,
      train_band: trainStd * 2,
      val_mean: valMean,
      val_lower: valMean - valStd,
      val_band: valStd * 2,
    }
  })
}

function buildFoldRows(data: LossCurvesData, foldId: number) {
  const fold = data.folds.find((item) => item.fold_id === foldId)
  if (!fold) return { fold: undefined, rows: [] as Array<Record<string, number>> }
  const avg = buildAverageRows(data)
  const rows = Array.from({ length: fold.n_iterations }, (_, index) => ({
    iteration: index,
    train: fold.train[index] ?? null,
    validation: fold.validation[index] ?? null,
    avg_train: avg[index]?.train_mean ?? null,
    avg_val: avg[index]?.val_mean ?? null,
  }))
  return { fold, rows }
}

export function LossChart({ data, selectedFoldId, height = 280 }: LossChartProps) {
  const averageRows = buildAverageRows(data)
  const { fold, rows: foldRows } = typeof selectedFoldId === "number" ? buildFoldRows(data, selectedFoldId) : { fold: undefined, rows: [] as Array<Record<string, number>> }
  const chartData = selectedFoldId === "average" ? averageRows : foldRows

  return (
    <div style={{ height }}>
      <ResponsiveContainer>
        <ComposedChart data={chartData}>
          <XAxis dataKey="iteration" stroke={colors.textSecondary} />
          <YAxis stroke={colors.textSecondary} domain={["dataMin - 0.03", "dataMax + 0.03"]} />
          <Tooltip
            contentStyle={{ backgroundColor: colors.bgTertiary, borderColor: colors.border, borderRadius: 16 }}
            formatter={(value: number | string, name: string) => [typeof value === "number" ? value.toFixed(4) : value, name]}
            labelFormatter={(value) => `Iteration: ${value}`}
          />
          <Legend />
          {selectedFoldId === "average" ? (
            <>
              <Area type="monotone" dataKey="train_lower" stackId="train" stroke="none" fill="transparent" isAnimationActive={false} />
              <Area type="monotone" dataKey="train_band" stackId="train" stroke="none" fill={colors.blue} fillOpacity={0.12} isAnimationActive={false} name="Train ±1 Std" />
              <Area type="monotone" dataKey="val_lower" stackId="val" stroke="none" fill="transparent" isAnimationActive={false} />
              <Area type="monotone" dataKey="val_band" stackId="val" stroke="none" fill={colors.yellow} fillOpacity={0.12} isAnimationActive={false} name="Validation ±1 Std" />
              <Line type="monotone" dataKey="train_mean" stroke={colors.blue} strokeWidth={2} dot={false} isAnimationActive={false} name="Train Loss (Ø)" />
              <Line type="monotone" dataKey="val_mean" stroke={colors.yellow} strokeWidth={2} dot={false} isAnimationActive={false} name="Validation Loss (Ø)" />
            </>
          ) : (
            <>
              <Line type="monotone" dataKey="train" stroke={colors.blue} strokeWidth={2} dot={false} isAnimationActive={false} name="Train Loss" />
              <Line type="monotone" dataKey="validation" stroke={colors.yellow} strokeWidth={2} dot={false} isAnimationActive={false} name="Validation Loss" />
              <Line type="monotone" dataKey="avg_train" stroke={colors.blue} strokeWidth={1.25} dot={false} strokeDasharray="4 4" isAnimationActive={false} name="Train Ø" />
              <Line type="monotone" dataKey="avg_val" stroke={colors.yellow} strokeWidth={1.25} dot={false} strokeDasharray="4 4" isAnimationActive={false} name="Validation Ø" />
              {fold && fold.early_stopped && fold.n_iterations !== fold.best_iteration ? (
                <ReferenceLine x={fold.best_iteration} stroke={colors.green} strokeDasharray="6 3" label={{ value: `Best: ${fold.best_iteration}`, fill: colors.textSecondary, fontSize: 11, position: "top" }} />
              ) : null}
            </>
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
