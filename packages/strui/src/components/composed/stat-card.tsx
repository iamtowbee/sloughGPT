import type { ReactNode } from 'react'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { cn } from '../../lib/cn'

export interface StatCardProps {
  label: string
  value: ReactNode
  /** Secondary line (e.g. “+12% vs last week”). */
  hint?: ReactNode
  className?: string
}

/** KPI / metric tile for dashboards and training monitors. */
export function StatCard({ label, value, hint, className }: StatCardProps) {
  return (
    <Card className={cn('overflow-hidden', className)}>
      <CardHeader className="space-y-1 pb-2">
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-2xl tabular-nums sm:text-3xl">{value}</CardTitle>
      </CardHeader>
      {hint ? <CardContent className="pt-0 text-xs text-muted-foreground">{hint}</CardContent> : null}
    </Card>
  )
}
