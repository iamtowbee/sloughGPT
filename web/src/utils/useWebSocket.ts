import { useEffect, useRef, useState, useCallback } from 'react'

interface UseWebSocketOptions {
  url: string
  onMessage?: (data: any) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
  reconnectAttempts?: number
  reconnectInterval?: number
}

interface WebSocketState {
  isConnected: boolean
  lastMessage: any | null
  error: Event | null
}

export const useWebSocket = (options: UseWebSocketOptions) => {
  const {
    url,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    reconnectAttempts = 5,
    reconnectInterval = 3000
  } = options

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectCountRef = useRef(0)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()

  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    lastMessage: null,
    error: null
  })

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    const ws = new WebSocket(url)

    ws.onopen = () => {
      setState(prev => ({ ...prev, isConnected: true, error: null }))
      reconnectCountRef.current = 0
      onConnect?.()
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setState(prev => ({ ...prev, lastMessage: data }))
        onMessage?.(data)
      } catch {
        setState(prev => ({ ...prev, lastMessage: event.data }))
        onMessage?.(event.data)
      }
    }

    ws.onerror = (error) => {
      setState(prev => ({ ...prev, error }))
      onError?.(error)
    }

    ws.onclose = () => {
      setState(prev => ({ ...prev, isConnected: false }))
      onDisconnect?.()

      // Attempt reconnection
      if (reconnectCountRef.current < reconnectAttempts) {
        reconnectCountRef.current += 1
        reconnectTimeoutRef.current = setTimeout(() => {
          connect()
        }, reconnectInterval)
      }
    }

    wsRef.current = ws
  }, [url, onMessage, onConnect, onDisconnect, onError, reconnectAttempts, reconnectInterval])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    reconnectCountRef.current = reconnectAttempts // Prevent further reconnections

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [reconnectAttempts])

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(typeof data === 'string' ? data : JSON.stringify(data))
    }
  }, [])

  const sendJson = useCallback((data: any) => {
    send(JSON.stringify(data))
  }, [send])

  useEffect(() => {
    connect()

    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    ...state,
    connect,
    disconnect,
    send,
    sendJson
  }
}

// Hook for subscribing to specific channels
export const useWebSocketChannel = (
  wsUrl: string,
  channel: string,
  onMessage: (data: any) => void
) => {
  const { isConnected, sendJson } = useWebSocket({
    url: wsUrl,
    onMessage: (data) => {
      if (data.channel === channel || data.type === channel) {
        onMessage(data)
      }
    }
  })

  const subscribe = useCallback(() => {
    sendJson({
      type: 'subscribe',
      channel
    })
  }, [sendJson, channel])

  const unsubscribe = useCallback(() => {
    sendJson({
      type: 'unsubscribe',
      channel
    })
  }, [sendJson, channel])

  return {
    isConnected,
    subscribe,
    unsubscribe
  }
}

export default useWebSocket
