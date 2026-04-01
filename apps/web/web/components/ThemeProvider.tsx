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
      document.body.className = ['font-sans', 'antialiased', mode, `theme-${theme}`].filter(Boolean).join(' ')
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

/** Accent presets — ids kept for localStorage; hues match ``globals.css`` theme-* */
export const THEMES: { id: Theme; name: string; color: string }[] = [
  { id: 'blue', name: 'Ochre', color: '#d4a24a' },
  { id: 'purple', name: 'Lilac', color: '#b8a3d4' },
  { id: 'pink', name: 'Rose', color: '#d898b4' },
  { id: 'red', name: 'Terracotta', color: '#d87868' },
  { id: 'orange', name: 'Honey', color: '#e8a54b' },
  { id: 'green', name: 'Sage', color: '#8fbc8f' },
  { id: 'teal', name: 'Patina', color: '#7eb8b0' },
]
