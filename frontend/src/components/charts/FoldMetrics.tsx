import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import type { FoldResult } from "../../types"
import { colors } from "../../utils/colors"

export function FoldMetrics({ folds }: { folds: FoldResult[] }) {
  const averageSharpe = folds.reduce((sum, fold) => sum + fold.sharpe, 0) / Math.max(folds.length, 1)
  return (
    <div className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
      <h3 className="text-lg font-semibold text-textPrimary">Out-of-Sample Performance uber Zeit</h3>
      <div className="mt-4 h-[220px]">
        <ResponsiveContainer>
          <BarChart data={folds}>
            <CartesianGrid stroke={colors.border} vertical={false} />
            <XAxis dataKey="fold_id" stroke={colors.textSecondary} />
            <YAxis stroke={colors.textSecondary} />
            <Tooltip contentStyle={{ backgroundColor: colors.bgTertiary, borderColor: colors.border }} />
            <ReferenceLine y={0} stroke={colors.textMuted} />
            <ReferenceLine y={averageSharpe} stroke={colors.yellow} strokeDasharray="4 4" />
            <Bar dataKey="sharpe" radius={[6, 6, 0, 0]}>
              {folds.map((fold) => (
                <Cell key={fold.fold_id} fill={fold.sharpe >= 0 ? colors.green : colors.red} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
