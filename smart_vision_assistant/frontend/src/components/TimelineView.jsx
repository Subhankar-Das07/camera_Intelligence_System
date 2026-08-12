/**
 * TimelineView.jsx
 * ================
 * Full-featured Session Timeline panel.
 * Polls GET /timeline every 5 seconds and renders a chronological
 * event feed with type badges, category tags, filtering, and stats.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react'

const API_BASE = 'http://localhost:8000'

const EVENT_CONFIG = {
  new:                  { color: 'bg-green-500/20 text-green-400 border-green-500/40',  icon: '✨', label: 'Entered' },
  returned:             { color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40', icon: '♻️', label: 'Returned' },
  removed:              { color: 'bg-red-500/20 text-red-400 border-red-500/40',        icon: '❌', label: 'Removed' },
  stationary:           { color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/40', icon: '⏸', label: 'Stationary' },
  moved:                { color: 'bg-blue-500/20 text-blue-400 border-blue-500/40',     icon: '↗', label: 'Moved' },
  abandoned:            { color: 'bg-orange-500/20 text-orange-400 border-orange-500/40', icon: '⚠', label: 'Abandoned' },
  verified:             { color: 'bg-purple-500/20 text-purple-400 border-purple-500/40', icon: '✓', label: 'Verified' },
  description_updated:  { color: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/40',    icon: '📝', label: 'Described' },
  near:                 { color: 'bg-indigo-500/20 text-indigo-400 border-indigo-500/40', icon: '⟵⟶', label: 'Near' },
  separated:            { color: 'bg-gray-500/20 text-gray-400 border-gray-500/40',    icon: '↔', label: 'Separated' },
}

const getEventConfig = (type) =>
  EVENT_CONFIG[type] || { color: 'bg-border/30 text-text-muted border-border/40', icon: '·', label: type }

const CATEGORY_COLORS = {
  Humans:      'text-pink-400',
  Electronics: 'text-blue-400',
  Kitchen:     'text-orange-400',
  Furniture:   'text-amber-400',
  Vehicles:    'text-teal-400',
  Animals:     'text-lime-400',
  Sports:      'text-violet-400',
}

const TimelineView = ({ connected }) => {
  const [timeline,    setTimeline]    = useState([])
  const [loading,     setLoading]     = useState(true)
  const [error,       setError]       = useState(null)
  const [filterType,  setFilterType]  = useState('All')
  const [filterLabel, setFilterLabel] = useState('')
  const [paused,      setPaused]      = useState(false)
  const [showAllSessions, setShowAllSessions] = useState(false)
  const [sessionId,   setSessionId]   = useState(null)
  const [count,       setCount]       = useState(0)

  const fetchTimeline = useCallback(async () => {
    if (paused) return
    try {
      const limit = showAllSessions ? 500 : 300
      const query = showAllSessions ? `?limit=${limit}&all_sessions=true` : `?limit=${limit}`
      const res = await fetch(`${API_BASE}/timeline${query}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setTimeline(data.timeline || [])
      setSessionId(data.session_id)
      setCount(data.count || 0)
      setError(null)
    } catch (err) {
      setError(`Cannot reach backend: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }, [paused, showAllSessions])

  useEffect(() => {
    fetchTimeline()
    const interval = setInterval(fetchTimeline, 5000)
    return () => clearInterval(interval)
  }, [fetchTimeline])

  // Derived stats
  const stats = useMemo(() => {
    const counts = {}
    const objects = {}
    timeline.forEach(ev => {
      counts[ev.type] = (counts[ev.type] || 0) + 1
      if (ev.label) objects[ev.label] = (objects[ev.label] || 0) + 1
    })
    const topObjects = Object.entries(objects)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
    return { counts, topObjects }
  }, [timeline])

  // All unique event types seen
  const eventTypes = useMemo(() => {
    const types = new Set(timeline.map(e => e.type))
    return ['All', ...Array.from(types)]
  }, [timeline])

  // Filtered view
  const filtered = useMemo(() => {
    return timeline.filter(ev => {
      if (filterType !== 'All' && ev.type !== filterType) return false
      if (filterLabel.trim() && !(ev.label || '').toLowerCase().includes(filterLabel.toLowerCase())) return false
      return true
    })
  }, [timeline, filterType, filterLabel])

  const formatTime = (iso) => {
    if (!iso) return '—'
    try {
      const parts = iso.split('T')
      return parts[1] ? parts[1].split('.')[0] : iso
    } catch { return iso }
  }

  const formatDate = (iso) => {
    if (!iso) return ''
    try { return iso.split('T')[0] } catch { return '' }
  }

  // Group by date for section headers
  const groupedByDate = useMemo(() => {
    const groups = {}
    filtered.forEach(ev => {
      const d = formatDate(ev.time)
      if (!groups[d]) groups[d] = []
      groups[d].push(ev)
    })
    return groups
  }, [filtered])

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-bg">
      
      {/* ── Toolbar ─────────────────────────────────────────────────── */}
      <div className="p-5 border-b border-border bg-card/60">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h2 className="text-xl font-semibold text-text-primary">Session Timeline</h2>
              <p className="text-xs text-text-muted mt-0.5">
                Chronological event history · Session #{sessionId ?? '—'} · {count} events total
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  setShowAllSessions(p => !p)
                  setTimeline([]) // Clear while fetching
                }}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
                  showAllSessions
                    ? 'bg-accent/20 text-accent border-accent/40 hover:bg-accent/30'
                    : 'bg-bg border-border text-text-muted hover:text-accent hover:border-accent'
                }`}
              >
                {showAllSessions ? '🌍 All History' : '⏱️ Current Session'}
              </button>
              <button
                onClick={() => setPaused(p => !p)}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
                  paused
                    ? 'bg-warning/20 text-warning border-warning/40 hover:bg-warning/30'
                    : 'bg-bg border-border text-text-muted hover:text-accent hover:border-accent'
                }`}
              >
                {paused ? '▶ Resume' : '⏸ Pause'}
              </button>
              <button
                onClick={fetchTimeline}
                className="text-xs px-3 py-1.5 rounded-lg border border-border text-text-muted hover:text-accent hover:border-accent transition-colors"
              >
                ↻ Refresh
              </button>
            </div>
          </div>

          {/* Stats row */}
          {timeline.length > 0 && (
            <div className="flex gap-3 flex-wrap mb-4">
              {Object.entries(stats.counts).map(([type, n]) => {
                const cfg = getEventConfig(type)
                return (
                  <div key={type} className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border ${cfg.color}`}>
                    <span>{cfg.icon}</span>
                    <span className="font-medium">{n}</span>
                    <span className="opacity-70">{cfg.label}</span>
                  </div>
                )
              })}
            </div>
          )}

          {/* Filters */}
          <div className="flex gap-3 flex-wrap items-center">
            {/* Type filter chips */}
            <div className="flex gap-1.5 flex-wrap">
              {eventTypes.map(t => {
                const cfg = getEventConfig(t)
                return (
                  <button
                    key={t}
                    onClick={() => setFilterType(t)}
                    className={`text-[11px] px-2.5 py-1 rounded-full border transition-colors ${
                      filterType === t
                        ? (t === 'All' ? 'bg-accent text-bg border-accent' : cfg.color.replace('/20', '/40'))
                        : 'bg-bg border-border text-text-muted hover:border-accent/50 hover:text-text-secondary'
                    }`}
                  >
                    {t === 'All' ? 'All Events' : (cfg.icon + ' ' + cfg.label)}
                  </button>
                )
              })}
            </div>

            {/* Label search */}
            <input
              type="text"
              placeholder="Filter by label…"
              value={filterLabel}
              onChange={e => setFilterLabel(e.target.value)}
              className="ml-auto bg-bg border border-border rounded-lg px-3 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent transition-colors w-36"
            />
          </div>
        </div>
      </div>

      {/* ── Event feed ──────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-5">
        <div className="max-w-5xl mx-auto">
          
          {loading && (
            <div className="space-y-3">
              {[1,2,3,4,5].map(i => (
                <div key={i} className="card p-3 animate-pulse flex gap-4">
                  <div className="w-16 h-4 bg-border/40 rounded" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-border/30 rounded w-1/3" />
                    <div className="h-3 bg-border/20 rounded w-2/3" />
                  </div>
                </div>
              ))}
            </div>
          )}

          {error && !loading && (
            <div className="p-4 bg-warning/10 border border-warning/40 text-warning rounded-lg text-sm">
              ⚠️ {error}
            </div>
          )}

          {!loading && !error && filtered.length === 0 && (
            <div className="py-20 text-center">
              <div className="text-5xl mb-4">📋</div>
              <p className="text-text-muted text-sm">
                {timeline.length === 0
                  ? 'No events recorded yet. Events appear as the camera detects objects.'
                  : 'No events match your filters.'}
              </p>
              {filterType !== 'All' || filterLabel ? (
                <button
                  onClick={() => { setFilterType('All'); setFilterLabel('') }}
                  className="mt-3 text-xs text-accent hover:underline"
                >
                  Clear filters
                </button>
              ) : null}
            </div>
          )}

          {!loading && !error && filtered.length > 0 && (
            <div>
              {/* Show filtered count if different from total */}
              {filtered.length !== timeline.length && (
                <p className="text-xs text-text-muted mb-4">
                  Showing {filtered.length} of {timeline.length} events
                </p>
              )}

              {/* Timeline feed */}
              <div className="relative">
                {/* Vertical timeline line */}
                <div className="absolute left-[72px] top-0 bottom-0 w-px bg-border/40 pointer-events-none" />

                <div className="space-y-2">
                  {filtered.map((ev, i) => {
                    const cfg = getEventConfig(ev.type)
                    const catColor = CATEGORY_COLORS[ev.category] || 'text-text-muted'

                    return (
                      <div key={i} className="flex gap-4 items-start group">
                        
                        {/* Timestamp */}
                        <div className="text-[10px] text-text-muted w-16 pt-2.5 text-right flex-shrink-0 leading-none">
                          {formatTime(ev.time)}
                        </div>

                        {/* Dot */}
                        <div className={`relative z-10 w-3 h-3 mt-2.5 rounded-full border-2 flex-shrink-0 ${cfg.color}`} />

                        {/* Content card */}
                        <div className="flex-1 card p-3 group-hover:border-border/60 transition-colors mb-1">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className={`text-[11px] px-2 py-0.5 rounded-full border font-medium ${cfg.color}`}>
                              {cfg.icon} {cfg.label}
                            </span>
                            <span className="text-sm font-semibold text-text-primary">
                              #{ev.track_id} {ev.label || '—'}
                            </span>
                            {ev.category && (
                              <span className={`text-[10px] ${catColor}`}>
                                {ev.category}
                              </span>
                            )}
                            {ev.brand && (
                              <span className="text-[10px] text-blue-400 bg-blue-500/10 px-1.5 py-0.5 rounded">
                                {ev.brand}
                              </span>
                            )}
                          </div>
                          {ev.extra && (
                            <p className="text-xs text-text-secondary mt-1.5 leading-relaxed">
                              {ev.extra}
                            </p>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Top objects panel */}
              {stats.topObjects.length > 0 && (
                <div className="mt-8 card p-4">
                  <h3 className="text-sm font-semibold text-text-primary mb-3">Most Active Objects This Session</h3>
                  <div className="space-y-2">
                    {stats.topObjects.map(([label, count], i) => (
                      <div key={label} className="flex items-center gap-3">
                        <span className="text-xs text-text-muted w-4">{i + 1}.</span>
                        <span className="text-sm text-text-primary flex-1">{label}</span>
                        <div className="flex items-center gap-2">
                          <div
                            className="h-1.5 bg-accent/60 rounded-full"
                            style={{ width: `${Math.max(20, (count / stats.topObjects[0][1]) * 100)}px` }}
                          />
                          <span className="text-xs text-accent font-medium">{count}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default TimelineView
