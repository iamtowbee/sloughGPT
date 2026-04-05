import type { ReactNode } from 'react'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { cn } from '../../lib/cn'

export interface EmptyCardProps {
  title: string
  description?: string
  children?: ReactNode
  className?: string
}

/** Compact empty state inside a card — onboarding panels, “no data” tables. */
export function EmptyCard({ title, description, children, className }: EmptyCardProps) {
  return (
    <Card className={cn('border-dashed', className)}>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        {description ? <CardDescription>{description}</CardDescription> : null}
      </CardHeader>
      {children ? <CardContent>{children}</CardContent> : null}
    </Card>
  )
}
