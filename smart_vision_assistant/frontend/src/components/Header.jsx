/**
 * Header.jsx
 * ==========
 * Top bar with title and live system metrics.
 * Updates from WebSocket data (no separate polling).
 */

import React, { memo, useMemo } from 'react'

// ── Helper: format session time ──────────────────────────────────────────────
function formatTime(seconds) {
  if (!seconds || seconds < 0) return '0s'
  const s = Math.floor(seconds)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const rem = s % 60
  if (m < 60) return `${m}m ${rem.toString().padStart(2, '0')}s`
  const h = Math.floor(m / 60)
  return `${h}h ${(m % 60).toString().padStart(2, '0')}m`
}

// ── Helper: colour fps ───────────────────────────────────────────────────────
function fpsColour(fps) {
  if (fps >= 20) return 'text-success'
  if (fps >= 10) return 'text-warning'
  return 'text-danger'
}

// ── Helper: colour cpu ───────────────────────────────────────────────────────
function cpuColour(cpu) {
  if (cpu < 60) return 'text-success'
  if (cpu < 80) return 'text-warning'
  return 'text-danger'
}

// ── StatChip — small inline metric ──────────────────────────────────────────
const StatChip = memo(({ label, value, valueClass = 'text-text-primary' }) => (
  <div className="flex flex-col items-center min-w-[64px]">
    <span className={`text-sm font-semibold font-mono ${valueClass}`}>{value}</span>
    <span className="text-[10px] text-text-muted uppercase tracking-wider mt-0.5">{label}</span>
  </div>
))

// ── Divider ──────────────────────────────────────────────────────────────────
const Divider = () => (
  <div className="w-px h-8 bg-border mx-1" />
)

// ── Header ───────────────────────────────────────────────────────────────────
const Header = memo(({ status, connected, sessionTime }) => {
  const fps           = status?.fps ?? 0
  const cpu           = status?.cpu ?? 0
  const ramMb         = status?.ram_mb ?? 0
  const activeObjects = status?.active_objects ?? 0
  const sessionObjects= status?.session_objects ?? 0

  const fpsStr = useMemo(() => `${fps.toFixed(1)}`, [fps])
  const cpuStr = useMemo(() => `${Math.round(cpu)}%`, [cpu])
  const ramStr = useMemo(() => {
    if (ramMb >= 1024) return `${(ramMb / 1024).toFixed(1)}GB`
    return `${Math.round(ramMb)}MB`
  }, [ramMb])
  const timeStr = useMemo(() => formatTime(sessionTime), [sessionTime])

  return (
    <header className="sticky top-0 z-50 bg-card/95 backdrop-blur-sm border-b border-border">
      <div className="flex items-center justify-between px-6 py-3">

        {/* ── Left: Title ──────────────────────────────────────────────────── */}
        <div className="flex items-center gap-3">
          {/* Live indicator */}
          <div className="relative flex items-center justify-center">
            <div className={`status-dot ${connected ? 'status-dot-green live-ring' : 'status-dot-red'}`} />
          </div>

          <div>
            <h1 className="text-base font-bold text-text-primary leading-tight tracking-tight">
              Smart Vision Assistant
            </h1>
            <p className="text-[11px] text-text-muted leading-none mt-0.5 tracking-wide">
              Visual Scene Intelligence Engine
            </p>
          </div>

          {/* Connection badge */}
          <span className={`badge ${connected ? 'badge-success' : 'badge-warning'} ml-2`}>
            {connected ? '● LIVE' : '○ Reconnecting…'}
          </span>
        </div>

        {/* ── Right: Live Stats ─────────────────────────────────────────────── */}
        <div className="flex items-center gap-1">
          <StatChip label="FPS" value={fpsStr} valueClass={fpsColour(fps)} />
          <Divider />
          <StatChip label="CPU" value={cpuStr} valueClass={cpuColour(cpu)} />
          <Divider />
          <StatChip label="RAM" value={ramStr} valueClass="text-accent" />
          <Divider />
          <StatChip
            label="Active"
            value={activeObjects}
            valueClass="text-success"
          />
          <Divider />
          <StatChip
            label="Session"
            value={sessionObjects}
            valueClass="text-accent"
          />
          <Divider />
          <StatChip label="Session" value={timeStr} valueClass="text-text-secondary" />
        </div>
      </div>
    </header>
  )
})

export default Header
