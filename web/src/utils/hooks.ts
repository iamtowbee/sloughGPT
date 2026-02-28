import { useState, useEffect, useCallback, useRef } from 'react'

interface PollingOptions<T> {
  fetchFn: () => Promise<T>
  interval?: number
  enabled?: boolean
  onSuccess?: (data: T) => void
  onError?: (error: Error) => void
}

interface PollingState<T> {
  data: T | null
  isLoading: boolean
  error: Error | null
  lastUpdated: Date | null
}

export const usePolling = <T>(options: PollingOptions<T>): PollingState<T> & { refetch: () => Promise<void> } => {
  const {
    fetchFn,
    interval = 5000,
    enabled = true,
    onSuccess,
    onError
  } = options

  const [state, setState] = useState<PollingState<T>>({
    data: null,
    isLoading: false,
    error: null,
    lastUpdated: null
  })

  const intervalRef = useRef<NodeJS.Timeout>()
  const isMountedRef = useRef(true)

  const fetchData = useCallback(async () => {
    if (!enabled) return

    try {
      const data = await fetchFn()
      
      if (isMountedRef.current) {
        setState(prev => ({
          ...prev,
          data,
          isLoading: false,
          error: null,
          lastUpdated: new Date()
        }))
        onSuccess?.(data)
      }
    } catch (error) {
      if (isMountedRef.current) {
        const err = error instanceof Error ? error : new Error('Unknown error')
        setState(prev => ({
          ...prev,
          isLoading: false,
          error: err
        }))
        onError?.(err)
      }
    }
  }, [fetchFn, enabled, onSuccess, onError])

  useEffect(() => {
    isMountedRef.current = true
    
    // Initial fetch
    setState(prev => ({ ...prev, isLoading: true }))
    fetchData()

    // Set up polling
    if (enabled && interval > 0) {
      intervalRef.current = setInterval(fetchData, interval)
    }

    return () => {
      isMountedRef.current = false
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [fetchData, interval, enabled])

  return {
    ...state,
    refetch: fetchData
  }
}

// Hook for debounced value
export const useDebounce = <T>(value: T, delay: number): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value)

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value)
    }, delay)

    return () => {
      clearTimeout(handler)
    }
  }, [value, delay])

  return debouncedValue
}

// Hook for local storage
export const useLocalStorage = <T>(key: string, initialValue: T): [T, (value: T) => void] => {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key)
      return item ? JSON.parse(item) : initialValue
    } catch (error) {
      console.error('Error reading from localStorage:', error)
      return initialValue
    }
  })

  const setValue = (value: T) => {
    try {
      setStoredValue(value)
      window.localStorage.setItem(key, JSON.stringify(value))
    } catch (error) {
      console.error('Error writing to localStorage:', error)
    }
  }

  return [storedValue, setValue]
}

// Hook for window size
export const useWindowSize = () => {
  const [size, setSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 0,
    height: typeof window !== 'undefined' ? window.innerHeight : 0
  })

  useEffect(() => {
    const handleResize = () => {
      setSize({
        width: window.innerWidth,
        height: window.innerHeight
      })
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return size
}

// Hook for click outside
export const useClickOutside = (
  ref: React.RefObject<HTMLElement>,
  handler: () => void
) => {
  useEffect(() => {
    const listener = (event: MouseEvent | TouchEvent) => {
      if (!ref.current || ref.current.contains(event.target as Node)) {
        return
      }
      handler()
    }

    document.addEventListener('mousedown', listener)
    document.addEventListener('touchstart', listener)

    return () => {
      document.removeEventListener('mousedown', listener)
      document.removeEventListener('touchstart', listener)
    }
  }, [ref, handler])
}

// Hook for keyboard shortcuts
export const useKeyboardShortcut = (
  key: string,
  callback: () => void,
  options: { ctrl?: boolean; shift?: boolean; alt?: boolean } = {}
) => {
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (
        event.key.toLowerCase() === key.toLowerCase() &&
        (options.ctrl ? event.ctrlKey || event.metaKey : true) &&
        (options.shift ? event.shiftKey : true) &&
        (options.alt ? event.altKey : true)
      ) {
        event.preventDefault()
        callback()
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [key, callback, options.ctrl, options.shift, options.alt])
}


interface WebSocketOptions {
  url?: string
  onMessage?: (data: any) => void
  onConnect?: () => void
  onDisconnect?: () => void
  reconnectInterval?: number
}

export const useWebSocket = (options: WebSocketOptions = {}) => {
  const {
    url = `ws://localhost:8000/ws`,
    onMessage,
    onConnect,
    onDisconnect,
    reconnectInterval = 5000
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url)
      
      ws.onopen = () => {
        setIsConnected(true)
        onConnect?.()
      }
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          onMessage?.(data)
        } catch {
          console.error('Failed to parse WS message')
        }
      }
      
      ws.onclose = () => {
        setIsConnected(false)
        onDisconnect?.()
        reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval)
      }
      
      ws.onerror = () => {
        ws.close()
      }
      
      wsRef.current = ws
    } catch {
      reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval)
    }
  }, [url, onConnect, onDisconnect, onMessage, reconnectInterval])

  useEffect(() => {
    connect()
    return () => {
      reconnectTimeoutRef.current && clearTimeout(reconnectTimeoutRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const send = useCallback((data: any) => {
    wsRef.current?.send(JSON.stringify(data))
  }, [])

  return { isConnected, send }
}
