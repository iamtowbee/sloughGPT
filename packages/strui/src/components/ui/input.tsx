import * as React from 'react'

import { cn } from '../../lib/cn'

/**
 * Shared field chrome for `<input>` and `<Textarea>` — solid surface, readable placeholder,
 * primary-tinted hover/focus (avoids flat “disabled gray” look).
 */
export const inputFieldClassName = cn(
  'flex w-full rounded-none border-2 border-border bg-card px-3 py-2 text-sm text-foreground shadow-sm',
  'transition-[border-color,box-shadow,background-color,color] duration-200 ease-smooth',
  'placeholder:text-foreground/42 dark:placeholder:text-foreground/36',
  'selection:bg-primary/20 selection:text-foreground',
  'hover:border-primary/35 hover:shadow-md',
  'focus-visible:border-primary/55 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/25 focus-visible:ring-offset-2 focus-visible:ring-offset-background',
  'disabled:cursor-not-allowed disabled:opacity-[0.72]'
)

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>

const Input = React.forwardRef<HTMLInputElement, InputProps>(({ className, type, ...props }, ref) => {
  return (
    <input
      type={type}
      className={cn(inputFieldClassName, 'h-10', className)}
      ref={ref}
      {...props}
    />
  )
})
Input.displayName = 'Input'

export { Input }
