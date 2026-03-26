'use client'

import { useTheme, THEMES } from './ThemeProvider'

export function ThemeSwitcher() {
  const { theme, mode, setTheme, setMode } = useTheme()

  return (
    <div className="flex items-center justify-between">
      <button
        onClick={() => setMode(mode === 'dark' ? 'light' : 'dark')}
        className="p-1.5 rounded-md hover:bg-secondary text-muted-foreground transition-colors"
        title={mode === 'dark' ? 'Switch to light' : 'Switch to dark'}
      >
        {mode === 'dark' ? '🌙' : '☀️'}
      </button>
      
      <div className="flex items-center gap-1">
        {THEMES.map((t) => (
          <button
            key={t.id}
            onClick={() => setTheme(t.id)}
            className={`w-4 h-4 rounded-full transition-all ${
              theme === t.id 
                ? 'ring-2 ring-primary ring-offset-1 ring-offset-background scale-110' 
                : 'hover:scale-110 opacity-70 hover:opacity-100'
            }`}
            style={{ backgroundColor: t.color }}
            title={t.name}
          />
        ))}
      </div>
    </div>
  )
}
