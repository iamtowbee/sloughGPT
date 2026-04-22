'use client'

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { cn } from '@/lib/cn'

export interface PersonalityDataPoint {
  step: number
  formality: number
  detail_level: number
  certainty: number
}

const METRICS = {
  formality: { label: 'Formality', color: '#3b82f6' },
  detail_level: { label: 'Detail', color: '#22c55e' },
  certainty: { label: 'Certainty', color: '#a855f7' },
}

interface PersonalityChartProps {
  data: PersonalityDataPoint[]
  className?: string
}

export function PersonalityEvolutionChart({ data, className }: PersonalityChartProps) {
  if (data.length === 0) return null

  return (
    <div className={cn("bg-white dark:bg-gray-900 rounded-lg border p-1.5 sm:p-2", className)}>
      <div className="flex items-center justify-between gap-1 mb-1">
        <h3 className="text-[9px] font-semibold text-foreground">Personality</h3>
        <div className="flex gap-1 text-[8px] text-muted-foreground">
          <span className="flex items-center gap-0.5">
            <span className="w-1 h-1 rounded-full" style={{ backgroundColor: METRICS.formality.color }} />F</span>
          <span className="flex items-center gap-0.5">
            <span className="w-1 h-1 rounded-full" style={{ backgroundColor: METRICS.detail_level.color }} />D</span>
          <span className="flex items-center gap-0.5">
            <span className="w-1 h-1 rounded-full" style={{ backgroundColor: METRICS.certainty.color }} />C</span>
        </div>
      </div>

      <div className="h-20 sm:h-24">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 2, right: 8, left: 0, bottom: 2 }}>
            <XAxis dataKey="step" tick={{ fontSize: 8 }} stroke="var(--muted-foreground)" interval={Math.max(1, Math.floor(data.length / 5))} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 8 }} stroke="var(--muted-foreground)" width={20} />
            <Tooltip contentStyle={{ fontSize: 9, backgroundColor: 'var(--background)', border: '1px solid var(--border)', padding: '2px 4px' }} labelStyle={{ fontSize: 9, marginBottom: 0 }} />
            <Line type="monotone" dataKey="formality" stroke={METRICS.formality.color} strokeWidth={1} dot={false} name="Formality" />
            <Line type="monotone" dataKey="detail_level" stroke={METRICS.detail_level.color} strokeWidth={1} dot={false} name="Detail" />
            <Line type="monotone" dataKey="certainty" stroke={METRICS.certainty.color} strokeWidth={1} dot={false} name="Certainty" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}