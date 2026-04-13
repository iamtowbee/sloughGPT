'use client'

import { useState, useCallback, useEffect } from 'react'
import { cn } from '@/lib/cn'

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList
}

interface SpeechRecognitionResultList {
  length: number
  item(index: number): SpeechRecognitionResult
  [index: number]: SpeechRecognitionResult
}

interface SpeechRecognitionResult {
  length: number
  item(index: number): SpeechRecognitionAlternative
  [index: number]: SpeechRecognitionAlternative
  isFinal: boolean
}

interface SpeechRecognitionAlternative {
  transcript: string
  confidence: number
}

interface SpeechRecognition extends EventTarget {
  new (): SpeechRecognition
  continuous: boolean
  interimResults: boolean
  lang: string
  start(): void
  stop(): void
  abort(): void
  onstart: (() => void) | null
  onend: (() => void) | null
  onerror: ((event: Event) => void) | null
  onresult: ((event: SpeechRecognitionEvent) => void) | null
}

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognition
    webkitSpeechRecognition?: SpeechRecognition
  }
}

interface VoiceInputProps {
  onTranscript: (text: string) => void
  disabled?: boolean
}

function MicrophoneIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
    </svg>
  )
}

function WaveformIcon({ className, isActive }: { className?: string; isActive: boolean }) {
  return (
    <svg className={cn(className, isActive && "animate-pulse")} fill="currentColor" viewBox="0 0 24 24" aria-hidden>
      <rect x="4" y="10" width="2" height="4" rx="1" className={isActive ? "animate-pulse" : ""} />
      <rect x="8" y="7" width="2" height="10" rx="1" className={isActive ? "animate-pulse" : ""} style={{ animationDelay: '100ms' }} />
      <rect x="12" y="4" width="2" height="16" rx="1" className={isActive ? "animate-pulse" : ""} style={{ animationDelay: '200ms' }} />
      <rect x="16" y="7" width="2" height="10" rx="1" className={isActive ? "animate-pulse" : ""} style={{ animationDelay: '300ms' }} />
      <rect x="20" y="10" width="2" height="4" rx="1" className={isActive ? "animate-pulse" : ""} style={{ animationDelay: '400ms' }} />
    </svg>
  )
}

export function VoiceInput({ onTranscript, disabled }: VoiceInputProps) {
  const [isListening, setIsListening] = useState(false)
  const [isSupported, setIsSupported] = useState(false)

  useEffect(() => {
    setIsSupported(typeof window !== 'undefined' && ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window))
  }, [])

  const startListening = useCallback(() => {
    if (!isSupported || disabled) return

    const SRConstructor = window.SpeechRecognition || window.webkitSpeechRecognition
    
    if (!SRConstructor) return

    const recognition = new SRConstructor()
    recognition.continuous = false
    recognition.interimResults = true
    recognition.lang = 'en-US'

    recognition.onstart = () => setIsListening(true)
    recognition.onend = () => setIsListening(false)
    recognition.onerror = () => setIsListening(false)

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = Array.from(event.results)
        .map((result: SpeechRecognitionResult) => result[0].transcript)
        .join('')
      if (transcript) {
        onTranscript(transcript)
      }
    }

    recognition.start()
  }, [isSupported, disabled, onTranscript])

  const toggleListening = useCallback(() => {
    if (isListening) {
      setIsListening(false)
    } else {
      startListening()
    }
  }, [isListening, startListening])

  if (!isSupported) return null

  return (
    <button
      type="button"
      onClick={toggleListening}
      disabled={disabled}
      className={cn(
        "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg transition-all duration-200",
        isListening 
          ? "bg-red-500/20 text-red-500 animate-pulse" 
          : "text-muted-foreground hover:bg-secondary hover:text-foreground",
        disabled && "opacity-50 cursor-not-allowed"
      )}
      aria-label={isListening ? "Stop listening" : "Start voice input"}
      title={isListening ? "Stop listening" : "Voice input"}
    >
      {isListening ? (
        <WaveformIcon className="h-5 w-5" isActive={true} />
      ) : (
        <MicrophoneIcon className="h-5 w-5" />
      )}
    </button>
  )
}
