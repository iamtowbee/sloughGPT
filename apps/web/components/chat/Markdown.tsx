'use client'

import { useMemo } from 'react'
import { cn } from '@/lib/cn'

interface MarkdownProps {
  content: string
  className?: string
}

function parseInlineMarkdown(text: string): React.ReactNode[] {
  const parts: React.ReactNode[] = []
  let remaining = text
  let key = 0

  while (remaining.length > 0) {
    let match: RegExpMatchArray | null

    match = remaining.match(/\*\*(.+?)\*\*/)
    if (match) {
      const idx = remaining.indexOf(match[0])
      if (idx > 0) {
        parts.push(remaining.slice(0, idx))
        remaining = remaining.slice(idx)
      }
      parts.push(<strong key={key++}>{match[1]}</strong>)
      remaining = remaining.slice(match[0].length)
      continue
    }

    match = remaining.match(/\*(.+?)\*/)
    if (match) {
      const idx = remaining.indexOf(match[0])
      if (idx > 0) {
        parts.push(remaining.slice(0, idx))
        remaining = remaining.slice(idx)
      }
      parts.push(<em key={key++}>{match[1]}</em>)
      remaining = remaining.slice(match[0].length)
      continue
    }

    match = remaining.match(/`(.+?)`/)
    if (match) {
      const idx = remaining.indexOf(match[0])
      if (idx > 0) {
        parts.push(remaining.slice(0, idx))
        remaining = remaining.slice(idx)
      }
      parts.push(
        <code key={key++} className="rounded bg-muted/50 px-1 py-0.5 font-mono text-[0.875em]">
          {match[1]}
        </code>
      )
      remaining = remaining.slice(match[0].length)
      continue
    }

    match = remaining.match(/\[(.+?)\]\((.+?)\)/)
    if (match) {
      const idx = remaining.indexOf(match[0])
      if (idx > 0) {
        parts.push(remaining.slice(0, idx))
        remaining = remaining.slice(idx)
      }
      parts.push(
        <a 
          key={key++} 
          href={match[2]} 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-primary underline underline-offset-2 hover:text-primary/80"
        >
          {match[1]}
        </a>
      )
      remaining = remaining.slice(match[0].length)
      continue
    }

    const newlineIdx = remaining.indexOf('\n')
    if (newlineIdx === -1) {
      parts.push(remaining)
      break
    } else if (newlineIdx === 0) {
      parts.push(<br key={key++} />)
      remaining = remaining.slice(1)
    } else {
      parts.push(remaining.slice(0, newlineIdx + 1))
      remaining = remaining.slice(newlineIdx + 1)
    }
  }

  return parts
}

export function Markdown({ content, className }: MarkdownProps) {
  const rendered = useMemo(() => parseInlineMarkdown(content), [content])

  return (
    <div className={cn("whitespace-pre-wrap break-words", className)}>
      {rendered}
    </div>
  )
}
