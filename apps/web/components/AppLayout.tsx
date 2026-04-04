'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'

import { IconMenu } from '@/components/icons/NavIcons'
import { Sidebar } from '@/components/Sidebar'

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const [mobileNavOpen, setMobileNavOpen] = useState(false)

  useEffect(() => {
    setMobileNavOpen(false)
  }, [pathname])

  useEffect(() => {
    if (!mobileNavOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMobileNavOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [mobileNavOpen])

  useEffect(() => {
    if (!mobileNavOpen) return
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = prev
    }
  }, [mobileNavOpen])

  const closeMobileNav = () => setMobileNavOpen(false)

  return (
    <div className="flex min-h-dvh flex-col bg-background lg:flex-row">
      <a
        href="#main-content"
        className="sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[200] focus:inline-block focus:rounded-md focus:border focus:border-border focus:bg-card focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-foreground focus:shadow-lg focus:outline-none focus:ring-2 focus:ring-ring"
      >
        Skip to main content
      </a>

      <header className="sticky top-0 z-40 flex min-h-[3.25rem] shrink-0 items-center gap-2 border-b border-border bg-card/90 px-3 pt-[max(0px,env(safe-area-inset-top))] backdrop-blur-md supports-[backdrop-filter]:bg-card/75 lg:hidden">
        <button
          type="button"
          className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-none text-foreground transition-colors hover:bg-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-expanded={mobileNavOpen}
          aria-controls="mobile-navigation-drawer"
          onClick={() => setMobileNavOpen(true)}
        >
          <IconMenu className="h-4 w-4" aria-hidden />
          <span className="sr-only">Open menu</span>
        </button>
        <Link
          href="/"
          className="min-w-0 truncate text-sm font-semibold tracking-tight text-foreground"
        >
          SloughGPT
        </Link>
      </header>

      <div className="hidden h-dvh shrink-0 lg:flex">
        <Sidebar variant="desktop" />
      </div>

      <main
        id="main-content"
        className="sl-shell-main flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden"
        tabIndex={-1}
      >
        <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">{children}</div>
      </main>

      {mobileNavOpen ? (
        <>
          <button
            type="button"
            className="fixed inset-0 z-[100] animate-in fade-in duration-200 bg-black/45 backdrop-blur-[2px] lg:hidden"
            aria-label="Close menu"
            onClick={closeMobileNav}
          />
          <div
            id="mobile-navigation-drawer"
            className="fixed inset-y-0 left-0 z-[110] flex w-[min(var(--sidebar-width),min(18rem,92vw))] max-w-full animate-in slide-in-from-left duration-200 ease-smooth shadow-[4px_0_24px_-4px_rgba(0,0,0,0.25)] lg:hidden"
            role="dialog"
            aria-modal="true"
            aria-label="Main navigation"
          >
            <Sidebar variant="drawer" onClose={closeMobileNav} onNavigate={closeMobileNav} />
          </div>
        </>
      ) : null}
    </div>
  )
}
