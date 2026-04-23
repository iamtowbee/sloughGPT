'use client'

import { SessionProvider } from 'next-auth/react'
import { ThemeProvider } from '@/components/ThemeProvider'
import { ModelProvider } from '@/contexts/ModelContext'

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider refetchOnWindowFocus={false}>
      <ThemeProvider>
        <ModelProvider>
          {children}
        </ModelProvider>
      </ThemeProvider>
    </SessionProvider>
  )
}
