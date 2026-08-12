/**
 * useWebSocket.js
 * ===============
 * Native WebSocket hook with automatic reconnection.
 *
 * Features:
 *  - Exponential back-off reconnect (max 8s)
 *  - JSON message parsing
 *  - Connection state tracking
 *  - No polling — event-driven only
 *  - Memoized callbacks to prevent unnecessary re-renders
 */

import { useEffect, useRef, useState, useCallback } from 'react'

const WS_URL = 'ws://localhost:8000/ws/live'
const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 8000

export function useWebSocket() {
  const [data, setData] = useState(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)
  const reconnectDelay = useRef(RECONNECT_BASE_MS)
  const mountedRef = useRef(true)

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (wsRef.current && wsRef.current.readyState < 2) {
      // Already open or connecting
      return
    }

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      if (!mountedRef.current) return
      setConnected(true)
      reconnectDelay.current = RECONNECT_BASE_MS
    }

    ws.onmessage = (event) => {
      if (!mountedRef.current) return
      try {
        const parsed = JSON.parse(event.data)
        setData(parsed)
      } catch {
        // Ignore malformed messages
      }
    }

    ws.onclose = () => {
      if (!mountedRef.current) return
      setConnected(false)
      // Schedule reconnect with back-off
      reconnectTimer.current = setTimeout(() => {
        reconnectDelay.current = Math.min(
          reconnectDelay.current * 1.5,
          RECONNECT_MAX_MS
        )
        connect()
      }, reconnectDelay.current)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      clearTimeout(reconnectTimer.current)
      if (wsRef.current) {
        wsRef.current.onclose = null  // Prevent reconnect on intentional unmount
        wsRef.current.close()
      }
    }
  }, [connect])

  return { data, connected }
}
