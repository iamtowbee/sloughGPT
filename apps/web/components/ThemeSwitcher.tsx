'use client'

import { IconMoon, IconSun } from '@/components/icons/NavIcons'

import { useTheme, THEMES } from './ThemeProvider'

export function ThemeSwitcher() {
  const { theme, mode, setTheme, setMode } = useTheme()

  return (
    <div className="flex items-center justify-between gap-2">
      <button
        type="button"
        onClick={() => setMode(mode === 'dark' ? 'light' : 'dark')}
        className="p-1.5 rounded-none text-foreground/65 transition-colors duration-200 ease-smooth hover:bg-secondary hover:text-foreground dark:text-muted-foreground"
        title={mode === 'dark' ? 'Switch to light' : 'Switch to dark'}
        aria-label={mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        {mode === 'dark' ? <IconMoon /> : <IconSun />}
      </button>
      
      <div className="flex items-center gap-1">
        {THEMES.map((t) => (
          <button
            key={t.id}
            onClick={() => setTheme(t.id)}
            className={`h-3.5 w-3.5 rounded-none transition-all duration-200 ease-smooth ${
              theme === t.id
                ? 'ring-2 ring-primary ring-offset-2 ring-offset-card scale-110 shadow-sm'
                : 'opacity-80 hover:scale-105 hover:opacity-100'
            }`}
            style={{ backgroundColor: t.color }}
            title={t.name}
          />
        ))}
      </div>
    </div>
  )
}
