import { Bar, BarChart, CartesianGrid, Cell, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import type { FoldResult } from "../../types"
import { colors } from "../../utils/colors"

export function TrainOOSGapChart({ folds }: { folds: FoldResult[] }) {
  const avgOos = folds.reduce((sum, fold) => sum + fold.oos_sharpe, 0) / Math.max(folds.length, 1)

  return (
    <div className="h-[280px]">
      <ResponsiveContainer>
        <BarChart data={folds} barGap={8}>
          <CartesianGrid stroke={colors.border} vertical={false} />
          <XAxis dataKey="fold_id" stroke={colors.textSecondary} />
          <YAxis stroke={colors.textSecondary} />
          <Tooltip contentStyle={{ backgroundColor: colors.bgTertiary, borderColor: colors.border }} />
          <Legend />
          <ReferenceLine y={avgOos} stroke={colors.yellow} strokeDasharray="4 4" />
          <Bar dataKey="train_sharpe" name="Train Sharpe" fill={colors.blue}>
            {folds.map((fold) => (
              <Cell key={`train-${fold.fold_id}`} stroke={fold.overfitting_flag ? colors.red : "transparent"} strokeWidth={fold.overfitting_flag ? 2 : 0} />
            ))}
          </Bar>
          <Bar dataKey="oos_sharpe" name="OOS Sharpe" fill={colors.green}>
            {folds.map((fold) => (
              <Cell key={`oos-${fold.fold_id}`} stroke={fold.overfitting_flag ? colors.red : "transparent"} strokeWidth={fold.overfitting_flag ? 2 : 0} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
