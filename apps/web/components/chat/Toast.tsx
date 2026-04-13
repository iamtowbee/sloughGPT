'use client'

import { useEffect, useState } from 'react'
import { cn } from '@/lib/cn'

export type ToastType = 'success' | 'error' | 'info'

export interface Toast {
  id: string
  message: string
  type: ToastType
}

interface ToastItemProps {
  toast: Toast
  onDismiss: (id: string) => void
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
  )
}

function XIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}

function InfoIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
}

function ToastItem({ toast, onDismiss }: ToastItemProps) {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    setIsVisible(true)
    const timer = setTimeout(() => {
      setIsVisible(false)
      setTimeout(() => onDismiss(toast.id), 200)
    }, 2800)
    return () => clearTimeout(timer)
  }, [toast.id, onDismiss])

  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-lg px-3 py-2 text-xs shadow-lg transition-all duration-200",
        toast.type === 'success' && "bg-emerald-500/95 text-white",
        toast.type === 'error' && "bg-red-500/95 text-white",
        toast.type === 'info' && "bg-blue-500/95 text-white",
        isVisible ? "opacity-100 translate-x-0" : "opacity-0 translate-x-4"
      )}
    >
      {toast.type === 'success' && <CheckIcon className="h-3.5 w-3.5 shrink-0" />}
      {toast.type === 'error' && <XIcon className="h-3.5 w-3.5 shrink-0" />}
      {toast.type === 'info' && <InfoIcon className="h-3.5 w-3.5 shrink-0" />}
      <span>{toast.message}</span>
      <button 
        onClick={() => onDismiss(toast.id)} 
        className="ml-1 opacity-60 hover:opacity-100"
      >
        <XIcon className="h-3 w-3" />
      </button>
    </div>
  )
}

interface ToastContainerProps {
  toasts: Toast[]
  onDismiss: (id: string) => void
}

export function ToastContainer({ toasts, onDismiss }: ToastContainerProps) {
  if (toasts.length === 0) return null
  
  return (
    <div className="fixed bottom-24 right-3 z-50 flex flex-col gap-2 sm:bottom-28 sm:right-4">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
      ))}
    </div>
  )
}
