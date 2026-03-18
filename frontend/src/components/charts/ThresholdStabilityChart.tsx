import { Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import type { ThresholdAnalysis } from "../../types"
import { colors } from "../../utils/colors"

export function ThresholdStabilityChart({ analysis }: { analysis: ThresholdAnalysis }) {
  const lower = analysis.mean - analysis.std
  const upper = analysis.mean + analysis.std

  return (
    <div className="h-[260px]">
      <ResponsiveContainer>
        <LineChart data={analysis.threshold_per_fold}>
          <XAxis dataKey="fold_id" stroke={colors.textSecondary} />
          <YAxis domain={[0.35, 0.65]} stroke={colors.textSecondary} />
          <Tooltip contentStyle={{ backgroundColor: colors.bgTertiary, borderColor: colors.border }} />
          <ReferenceArea y1={lower} y2={upper} fill={colors.blue} fillOpacity={0.08} />
          <ReferenceLine y={analysis.mean} stroke={colors.yellow} strokeDasharray="4 4" />
          <Line type="monotone" dataKey="threshold" stroke={colors.yellow} strokeWidth={2.5} dot={{ r: 3 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
