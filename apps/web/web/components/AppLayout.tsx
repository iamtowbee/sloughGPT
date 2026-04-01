'use client'

import { Sidebar } from '@/components/Sidebar'

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar />
      <main className="flex-1 min-h-screen app-shell-main overflow-auto">
        {children}
      </main>
    </div>
  )
}
