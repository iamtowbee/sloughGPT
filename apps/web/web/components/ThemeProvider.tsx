'use client'

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'

type Theme = 'blue' | 'purple' | 'pink' | 'red' | 'orange' | 'green' | 'teal'
type Mode = 'dark' | 'light'

interface ThemeContextType {
  theme: Theme
  mode: Mode
  setTheme: (theme: Theme) => void
  setMode: (mode: Mode) => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>('blue')
  const [mode, setMode] = useState<Mode>('dark')
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    const savedTheme = localStorage.getItem('sloughgpt_theme') as Theme
    const savedMode = localStorage.getItem('sloughgpt_mode') as Mode
    if (savedTheme && ['blue', 'purple', 'pink', 'red', 'orange', 'green', 'teal'].includes(savedTheme)) {
      setTheme(savedTheme)
    }
    if (savedMode === 'light' || savedMode === 'dark') {
      setMode(savedMode)
    }
    setMounted(true)
  }, [])

  useEffect(() => {
    if (mounted) {
      document.body.className = `${mode} theme-${theme}`
      localStorage.setItem('sloughgpt_theme', theme)
      localStorage.setItem('sloughgpt_mode', mode)
    }
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
    throw new Error('useTheme must be used within ThemeProvider')
  }
  return context
}

export const THEMES: { id: Theme; name: string; color: string }[] = [
  { id: 'blue', name: 'Blue', color: '#3b82f6' },
  { id: 'purple', name: 'Purple', color: '#a855f7' },
  { id: 'pink', name: 'Pink', color: '#ec4899' },
  { id: 'red', name: 'Red', color: '#ef4444' },
  { id: 'orange', name: 'Orange', color: '#f97316' },
  { id: 'green', name: 'Green', color: '#22c55e' },
  { id: 'teal', name: 'Teal', color: '#14b8a6' },
]
