import { ResponsiveContainer, Tooltip, XAxis, YAxis, ZAxis, ScatterChart, Scatter } from "recharts"
import type { ScatterPointItem } from "recharts/types/cartesian/Scatter"

import type { ConfusionMatrixData } from "../../types"
import { colors } from "../../utils/colors"

export function ConfusionMatrix({ data }: { data: ConfusionMatrixData }) {
  const points = data.matrix.flatMap((row, actualIndex) =>
    row.map((value, predictedIndex) => ({
      x: predictedIndex,
      y: actualIndex,
      z: value,
      normalized: data.normalized[actualIndex][predictedIndex],
      actual: data.labels[actualIndex],
      predicted: data.labels[predictedIndex],
      diagonal: actualIndex === predictedIndex,
    })),
  )

  return (
    <div className="rounded-2xl border border-border bg-secondary p-5 shadow-card">
      <h3 className="text-lg font-semibold text-textPrimary">Confusion Matrix</h3>
      <div className="mt-4 h-[280px]">
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 20, right: 20, left: 20, bottom: 20 }}>
            <XAxis dataKey="x" type="number" domain={[-0.5, 2.5]} tickFormatter={(value) => data.labels[value] ?? ""} stroke={colors.textSecondary} />
            <YAxis dataKey="y" type="number" domain={[-0.5, 2.5]} tickFormatter={(value) => data.labels[value] ?? ""} stroke={colors.textSecondary} />
            <ZAxis dataKey="z" range={[600, 3000]} />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              contentStyle={{ backgroundColor: colors.bgTertiary, borderColor: colors.border }}
              formatter={(value: number, _name, payload) => [`${value} (${(payload?.payload.normalized * 100).toFixed(1)}%)`, "Count"]}
            />
            <Scatter
              data={points}
              shape={(props: ScatterPointItem) => {
                const payload = props.payload as (typeof points)[number]
                const fillOpacity = 0.2 + payload.normalized * 0.8
                return (
                  <g>
                    <rect
                      x={(props.cx ?? 0) - 38}
                      y={(props.cy ?? 0) - 34}
                      width={76}
                      height={68}
                      rx={10}
                      fill={`rgba(0,212,170,${fillOpacity})`}
                      stroke={payload.diagonal ? colors.green : colors.border}
                      strokeWidth={payload.diagonal ? 2 : 1}
                    />
                    <text x={props.cx} y={(props.cy ?? 0) - 4} textAnchor="middle" fill={colors.textPrimary} fontSize={16} fontWeight={700}>
                      {payload.z}
                    </text>
                    <text x={props.cx} y={(props.cy ?? 0) + 16} textAnchor="middle" fill={colors.textSecondary} fontSize={11}>
                      {(payload.normalized * 100).toFixed(1)}%
                    </text>
                  </g>
                )
              }}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-4 text-sm text-textSecondary">Das Modell erkennt Long-Signale am zuverlässigsten und bleibt bei Flat-Zuständen robuster als bei Short-Regimen.</p>
    </div>
  )
}
