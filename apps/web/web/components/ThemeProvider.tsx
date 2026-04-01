'use client'

import { createContext, useContext, useState, useEffect, useLayoutEffect, ReactNode } from 'react'

import { syncHtmlTheme } from '@/lib/sync-html-theme'
import {
  isStoredThemeId,
  MODE_STORAGE_KEY,
  THEME_STORAGE_KEY,
  type StoredThemeId,
  type ThemeMode,
} from '@/lib/theme-storage'

interface ThemeContextType {
  theme: StoredThemeId
  mode: ThemeMode
  setTheme: (theme: StoredThemeId) => void
  setMode: (mode: ThemeMode) => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<StoredThemeId>('blue')
  const [mode, setMode] = useState<ThemeMode>('dark')
  const [mounted, setMounted] = useState(false)

  useLayoutEffect(() => {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY)
    const savedMode = localStorage.getItem(MODE_STORAGE_KEY) as ThemeMode
    const t = isStoredThemeId(savedTheme) ? savedTheme : 'blue'
    const m = savedMode === 'light' || savedMode === 'dark' ? savedMode : 'dark'
    setTheme(t)
    setMode(m)
    syncHtmlTheme(m, t)
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted) return
    syncHtmlTheme(mode, theme)
    localStorage.setItem(THEME_STORAGE_KEY, theme)
    localStorage.setItem(MODE_STORAGE_KEY, mode)
  }, [theme, mode, mounted])

  return (
    <ThemeContext.Provider value={{ theme, mode, setTheme, setMode }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

/** Accent presets — ids kept for localStorage; hues match ``globals.css`` theme-* */
export const THEMES: { id: StoredThemeId; name: string; color: string }[] = [
  { id: 'blue', name: 'Ochre', color: '#d4a24a' },
  { id: 'purple', name: 'Lilac', color: '#b8a3d4' },
  { id: 'pink', name: 'Rose', color: '#d898b4' },
  { id: 'red', name: 'Terracotta', color: '#d87868' },
  { id: 'orange', name: 'Honey', color: '#e8a54b' },
  { id: 'green', name: 'Sage', color: '#8fbc8f' },
  { id: 'teal', name: 'Patina', color: '#7eb8b0' },
]
