/**
 * ObjectWatchlist.jsx
 * ====================
 * Real-time object monitoring panel — MONITORING section.
 *
 * State machine:
 *   idle  → (select + start) → visible
 *   visible → (missing > 3s) → lost   [⚠ alert]
 *   lost    → (entity returns) → returned  [✓ banner, 5s]
 *   returned → (5s elapsed)  → visible
 *   any → (stop) → idle
 *
 * Data source:
 *   Primary  — WebSocket `watchlist` field (pushed every 1s by server)
 *   Fallback — GET /watchlist/status (polled if WS field unavailable)
 *
 * Objects available for selection: GET /watchlist/available
 *   → EntityRegistry.get_active(conf≥80%, age≥2s) — stabilised entities only.
 */

import React, {
  memo,
  useState,
  useEffect,
  useCallback,
  useRef,
} from 'react'

const API = 'http://localhost:8000'

// ── helpers ───────────────────────────────────────────────────────────────────

const confColor = (pct) => {
  if (pct >= 85) return 'text-emerald-400'
  if (pct >= 70) return 'text-yellow-400'
  return 'text-orange-400'
}

// ── sub-components ────────────────────────────────────────────────────────────

/** Pulsing dot indicator */
const StatusDot = memo(({ status }) => {
  const cfg = {
    visible:   { cls: 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.7)]', anim: 'animate-pulse' },
    lost:      { cls: 'bg-amber-400  shadow-[0_0_8px_rgba(251,191,36,0.8)]',  anim: 'animate-ping'  },
    returned:  { cls: 'bg-cyan-400   shadow-[0_0_8px_rgba(34,211,238,0.7)]',  anim: 'animate-pulse' },
    idle:      { cls: 'bg-slate-500',  anim: '' },
  }[status] ?? { cls: 'bg-slate-500', anim: '' }

  return (
    <span className="relative inline-flex h-2.5 w-2.5">
      {cfg.anim === 'animate-ping' && (
        <span className={`absolute inline-flex h-full w-full rounded-full ${cfg.cls} opacity-60 animate-ping`} />
      )}
      <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${cfg.cls}`} />
    </span>
  )
})

/** IDLE state — object selector */
const SelectorPanel = memo(({ available, onStart }) => {
  const [selectedUuid, setSelectedUuid] = useState('')

  const selectedEntity = available.find(e => e.entity_uuid === selectedUuid)

  const handleStart = () => {
    if (!selectedEntity) return
    onStart(selectedEntity)
  }

  return (
    <div className="space-y-3">
      {/* Dropdown */}
      <div>
        <label className="block text-[10px] font-semibold uppercase tracking-widest text-slate-400 mb-1.5">
          Select Object to Monitor
        </label>
        <select
          value={selectedUuid}
          onChange={e => setSelectedUuid(e.target.value)}
          className="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2
                     text-sm text-slate-100 focus:outline-none focus:border-violet-500
                     focus:ring-1 focus:ring-violet-500/40 transition-colors appearance-none cursor-pointer"
        >
          <option value="" disabled>
            {available.length === 0 ? '— No active objects detected —' : '— Choose an object —'}
          </option>
          {available.map(e => (
            <option key={e.entity_uuid} value={e.entity_uuid}>
              {e.display_label}  ·  {e.position}  ·  {e.confidence}% conf  ·  {e.duration_str}
            </option>
          ))}
        </select>
      </div>

      {/* Preview row */}
      {selectedEntity && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-violet-500/10 border border-violet-500/25">
          <span className="text-violet-300 text-xs font-medium">Selected:</span>
          <span className="text-white text-xs font-semibold">{selectedEntity.display_label}</span>
          <span className="ml-auto text-slate-400 text-[10px]">{selectedEntity.position}</span>
        </div>
      )}

      {/* Start button */}
      <button
        onClick={handleStart}
        disabled={!selectedEntity}
        className="w-full py-2 px-4 rounded-lg text-sm font-semibold transition-all duration-200
                   bg-violet-600 hover:bg-violet-500 text-white
                   disabled:opacity-30 disabled:cursor-not-allowed
                   focus:outline-none focus:ring-2 focus:ring-violet-500/50
                   shadow-[0_0_12px_rgba(139,92,246,0.25)] hover:shadow-[0_0_18px_rgba(139,92,246,0.4)]"
      >
        Start Monitoring
      </button>
    </div>
  )
})

/** VISIBLE state */
const VisiblePanel = memo(({ item, onStop }) => (
  <div className="space-y-3">
    {/* Status row */}
    <div className="flex items-center justify-between px-3 py-2.5 rounded-lg
                    bg-emerald-500/10 border border-emerald-500/25">
      <div className="flex items-center gap-2">
        <StatusDot status="visible" />
        <span className="text-emerald-300 text-sm font-semibold">VISIBLE</span>
      </div>
      <span className="text-[10px] text-slate-400 font-mono">{item.last_seen}</span>
    </div>

    {/* Info grid */}
    <div className="grid grid-cols-2 gap-2">
      <InfoCell label="Tracking" value={item.display_label} valueClass="text-white font-semibold" />
      <InfoCell label="Position" value={item.last_position} valueClass="text-emerald-300" />
      <InfoCell label="Observed" value={item.observed_for_str} valueClass="text-slate-200" />
      <InfoCell label="Confidence" value={`${item.last_confidence}%`}
                valueClass={confColor(item.last_confidence)} />
    </div>

    <StopButton onStop={onStop} />
  </div>
))

/** LOST state — pulsing amber alert */
const LostPanel = memo(({ item, onStop }) => (
  <div className="space-y-3">
    {/* Alert banner */}
    <div className="rounded-lg border border-amber-500/50 bg-amber-500/10 px-3 py-3
                    shadow-[0_0_20px_rgba(245,158,11,0.15)] animate-[alertGlow_2s_ease-in-out_infinite]">
      <div className="flex items-center gap-2 mb-1.5">
        <StatusDot status="lost" />
        <span className="text-amber-400 text-sm font-bold tracking-wide">⚠ WATCH ALERT</span>
      </div>
      <p className="text-white text-sm font-semibold">
        &ldquo;{item.display_label}&rdquo; is out of vision
      </p>
    </div>

    {/* Detail grid */}
    <div className="grid grid-cols-2 gap-2">
      <InfoCell label="Missing For"
                value={item.seconds_missing_str}
                valueClass="text-amber-400 font-bold" />
      <InfoCell label="Last Seen"
                value={item.last_seen}
                valueClass="text-slate-300 font-mono" />
      <InfoCell label="Last Position"
                value={item.last_position}
                valueClass="text-slate-300" />
      <InfoCell label="Last Confidence"
                value={`${item.last_confidence}%`}
                valueClass={confColor(item.last_confidence)} />
    </div>

    <StopButton onStop={onStop} variant="amber" />
  </div>
))

/** RETURNED state — cyan flash banner */
const ReturnedPanel = memo(({ item, onStop }) => (
  <div className="space-y-3">
    {/* Returned banner */}
    <div className="rounded-lg border border-cyan-500/50 bg-cyan-500/10 px-3 py-3
                    shadow-[0_0_18px_rgba(34,211,238,0.15)]">
      <div className="flex items-center gap-2 mb-1.5">
        <StatusDot status="returned" />
        <span className="text-cyan-400 text-sm font-bold">✓ VISIBLE AGAIN</span>
      </div>
      <p className="text-white text-sm font-semibold">
        &ldquo;{item.display_label}&rdquo; returned to camera view
      </p>
    </div>

    {/* Info grid */}
    <div className="grid grid-cols-2 gap-2">
      <InfoCell label="Tracking" value={item.display_label} valueClass="text-white font-semibold" />
      <InfoCell label="Position" value={item.last_position} valueClass="text-cyan-300" />
      <InfoCell label="Observed" value={item.observed_for_str} valueClass="text-slate-200" />
      <InfoCell label="Confidence" value={`${item.last_confidence}%`}
                valueClass={confColor(item.last_confidence)} />
    </div>

    <StopButton onStop={onStop} />
  </div>
))

/** Shared info cell */
const InfoCell = ({ label, value, valueClass = 'text-slate-200' }) => (
  <div className="bg-slate-800/60 rounded-lg px-2.5 py-2 border border-slate-700/50">
    <p className="text-[9px] uppercase tracking-widest text-slate-500 mb-0.5">{label}</p>
    <p className={`text-sm font-medium truncate ${valueClass}`}>{value || '—'}</p>
  </div>
)

/** Stop button */
const StopButton = ({ onStop, variant = 'default' }) => {
  const base = 'w-full py-2 px-4 rounded-lg text-sm font-medium transition-all duration-200 focus:outline-none focus:ring-2'
  const styles = variant === 'amber'
    ? `${base} bg-slate-700 hover:bg-slate-600 text-slate-300 border border-slate-600 focus:ring-slate-500/50`
    : `${base} bg-slate-700 hover:bg-slate-600 text-slate-300 border border-slate-600 focus:ring-slate-500/50`
  return (
    <button onClick={onStop} className={styles}>
      Stop Monitoring
    </button>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────

/**
 * ObjectWatchlist
 *
 * Props:
 *   wsWatchlist — watchlist payload forwarded from the parent WebSocket hook
 *                 (optional: component falls back to polling if not provided)
 */
const ObjectWatchlist = memo(({ wsWatchlist }) => {
  const [collapsed, setCollapsed]   = useState(false)
  const [available, setAvailable]   = useState([])
  const [watchStatus, setWatchStatus] = useState({ watching: false, item: null })
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState('')
  const pollRef = useRef(null)

  // ── Fetch available objects for dropdown ──────────────────────────────────
  const fetchAvailable = useCallback(async () => {
    try {
      const res = await fetch(`${API}/watchlist/available`)
      if (!res.ok) return
      const data = await res.json()
      setAvailable(data.available ?? [])
    } catch { /* server may not be up yet */ }
  }, [])

  // ── Poll fallback (used when WebSocket watchlist field is absent) ──────────
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API}/watchlist/status`)
      if (!res.ok) return
      const data = await res.json()
      setWatchStatus(data)
    } catch { /* ignore */ }
  }, [])

  // ── Prefer WebSocket push over polling ────────────────────────────────────
  useEffect(() => {
    if (wsWatchlist !== undefined && wsWatchlist !== null) {
      setWatchStatus(wsWatchlist)
    }
  }, [wsWatchlist])

  // ── Poll /watchlist/status every 1s as fallback ───────────────────────────
  useEffect(() => {
    if (wsWatchlist !== undefined) return   // WS covers us, no need to poll
    pollRef.current = setInterval(fetchStatus, 1000)
    return () => clearInterval(pollRef.current)
  }, [fetchStatus, wsWatchlist])

  // ── Refresh available list every 2s when idle (dropdown populated live) ───
  useEffect(() => {
    fetchAvailable()
    if (!watchStatus.watching) {
      const t = setInterval(fetchAvailable, 2000)
      return () => clearInterval(t)
    }
  }, [fetchAvailable, watchStatus.watching])

  // ── Start tracking ────────────────────────────────────────────────────────
  const handleStart = useCallback(async (entity) => {
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API}/watchlist/watch`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          entity_uuid:   entity.entity_uuid,
          display_label: entity.display_label,
          category:      entity.category ?? '',
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error ?? 'Failed to start monitoring')
      await fetchStatus()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [fetchStatus])

  // ── Stop tracking ─────────────────────────────────────────────────────────
  const handleStop = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      await fetch(`${API}/watchlist/stop`, { method: 'POST' })
      setWatchStatus({ watching: false, item: null })
      fetchAvailable()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [fetchAvailable])

  // ── Derive current status for rendering ───────────────────────────────────
  const item        = watchStatus.item
  const watchState  = item?.watch_status ?? 'idle'   // 'idle'|'visible'|'lost'|'returned'
  const isWatching  = watchStatus.watching

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <section className="mx-4 mb-4">
      {/* MONITORING section header */}
      <button
        onClick={() => setCollapsed(v => !v)}
        className="w-full flex items-center justify-between py-2 px-1 group"
        aria-expanded={!collapsed}
      >
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500
                           group-hover:text-slate-400 transition-colors">
            Monitoring
          </span>
          {/* Live indicator when actively watching */}
          {isWatching && (
            <span className="flex items-center gap-1 px-1.5 py-0.5 rounded-full
                             bg-violet-500/15 border border-violet-500/30 text-[9px]
                             font-semibold text-violet-400 uppercase tracking-wide">
              <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse inline-block" />
              Active
            </span>
          )}
        </div>
        <svg
          className={`w-3.5 h-3.5 text-slate-500 transition-transform duration-200
                      ${collapsed ? '-rotate-90' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Divider */}
      <div className="h-px bg-slate-700/50 mb-3" />

      {/* Collapsible body */}
      {!collapsed && (
        <div className="rounded-xl border border-slate-700/60 bg-slate-800/40
                        backdrop-blur-sm overflow-hidden">

          {/* Panel header */}
          <div className="px-4 py-3 border-b border-slate-700/50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              {/* Eye icon */}
              <svg className="w-4 h-4 text-violet-400" fill="none" viewBox="0 0 24 24"
                   stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round"
                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round"
                      d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7
                         -1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
              <span className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                Object Watchlist
              </span>
            </div>
            {/* Status badge */}
            {isWatching && item && (
              <WatchBadge status={watchState} />
            )}
          </div>

          {/* Panel body */}
          <div className="px-4 py-4">
            {loading && (
              <div className="text-center py-4">
                <div className="inline-block w-5 h-5 border-2 border-violet-500/30 border-t-violet-500
                                rounded-full animate-spin" />
              </div>
            )}

            {!loading && !isWatching && (
              <SelectorPanel available={available} onStart={handleStart} />
            )}

            {!loading && isWatching && item && watchState === 'visible' && (
              <VisiblePanel item={item} onStop={handleStop} />
            )}

            {!loading && isWatching && item && watchState === 'lost' && (
              <LostPanel item={item} onStop={handleStop} />
            )}

            {!loading && isWatching && item && watchState === 'returned' && (
              <ReturnedPanel item={item} onStop={handleStop} />
            )}

            {/* Error display */}
            {error && (
              <p className="mt-2 text-xs text-red-400 text-center">{error}</p>
            )}
          </div>
        </div>
      )}
    </section>
  )
})

/** Compact status badge shown in panel header */
const WatchBadge = ({ status }) => {
  const cfg = {
    visible:  { label: 'VISIBLE',      cls: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' },
    lost:     { label: 'OUT OF VIEW',  cls: 'bg-amber-500/15  text-amber-400  border-amber-500/30 animate-pulse' },
    returned: { label: 'RETURNED',     cls: 'bg-cyan-500/15   text-cyan-400   border-cyan-500/30'  },
  }[status] ?? { label: status.toUpperCase(), cls: 'bg-slate-700 text-slate-400 border-slate-600' }

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full
                      text-[9px] font-bold uppercase tracking-widest border ${cfg.cls}`}>
      {cfg.label}
    </span>
  )
}

export default ObjectWatchlist
