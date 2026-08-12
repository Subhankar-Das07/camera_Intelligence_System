/**
 * MemorySearch.jsx
 * ================
 * Live working Memory Search Engine panel.
 * Calls GET /search with query and filter params.
 * Falls back gracefully when no data exists yet.
 */

import React, { useState, useEffect, useCallback } from 'react'

const API_BASE = 'http://localhost:8000'

const CATEGORY_CHIPS = ['All', 'Humans', 'Electronics', 'Kitchen', 'Furniture', 'Vehicles', 'Animals', 'Sports']
const EVENT_CHIPS    = ['All', 'new', 'removed', 'stationary', 'moved', 'abandoned', 'verified']

const EVENT_COLORS = {
  new:       'bg-green-500/20 text-green-400 border-green-500/30',
  removed:   'bg-red-500/20 text-red-400 border-red-500/30',
  stationary:'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  moved:     'bg-blue-500/20 text-blue-400 border-blue-500/30',
  abandoned: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  verified:  'bg-purple-500/20 text-purple-400 border-purple-500/30',
}

const MemorySearch = () => {
  const [query,     setQuery]     = useState('')
  const [category,  setCategory]  = useState('All')
  const [eventType, setEventType] = useState('All')
  const [results,   setResults]   = useState([])
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [dbStats,   setDbStats]   = useState(null)
  const [statsOpen, setStatsOpen] = useState(false)

  // Fetch DB stats once for the verification panel
  const fetchDbStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/db-stats`)
      if (res.ok) setDbStats(await res.json())
    } catch (_) {}
  }, [])

  const handleSearch = useCallback(async (e) => {
    if (e) e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const params = new URLSearchParams()
      if (query.trim())          params.set('query',    query.trim())
      if (category !== 'All')    params.set('category', category)
      if (eventType !== 'All')   params.set('event',    eventType)

      const res = await fetch(`${API_BASE}/search?${params.toString()}`)
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const data = await res.json()
      setResults(data.results || [])
      if (data.session_id) setSessionId(data.session_id)
    } catch (err) {
      setError(`Search failed: ${err.message}. Is the backend running?`)
    } finally {
      setLoading(false)
    }
  }, [query, category, eventType])

  // Auto-search on mount and when filters change
  useEffect(() => {
    handleSearch()
    fetchDbStats()
  }, [category, eventType]) // eslint-disable-line react-hooks/exhaustive-deps

  const formatTime = (iso) => {
    if (!iso) return '—'
    try { return iso.split('T')[1].split('.')[0] } catch { return iso }
  }

  return (
    <div className="flex-1 flex flex-col p-6 overflow-y-auto bg-bg">
      <div className="max-w-6xl mx-auto w-full space-y-5">

        {/* ── Header ────────────────────────────────────────────────── */}
        <div className="card p-5">
          <div className="flex items-start justify-between mb-3">
            <div>
              <h2 className="text-xl font-semibold text-text-primary">Memory Search Engine</h2>
              <p className="text-text-muted text-xs mt-1">
                Search session history. Try: "show laptops", "stationary objects", "Dell", "person".
              </p>
            </div>
            <button
              onClick={() => { setStatsOpen(s => !s); fetchDbStats() }}
              className="text-xs bg-bg border border-border rounded-lg px-3 py-1.5 text-text-muted hover:text-accent hover:border-accent transition-colors"
            >
              📊 DB Stats
            </button>
          </div>

          {/* Search bar */}
          <form onSubmit={handleSearch} className="flex gap-3">
            <input
              type="text"
              placeholder="Natural language query or leave blank to show all…"
              className="flex-1 bg-bg border border-border rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-accent transition-colors"
              value={query}
              onChange={e => setQuery(e.target.value)}
            />
            <button
              type="submit"
              disabled={loading}
              className="bg-accent text-bg font-semibold px-6 py-2.5 rounded-lg text-sm hover:opacity-90 disabled:opacity-50 transition"
            >
              {loading ? '⏳ Searching…' : '🔍 Search'}
            </button>
          </form>

          {/* Category chips */}
          <div className="flex flex-wrap gap-2 mt-3">
            <span className="text-[10px] text-text-muted self-center uppercase tracking-wider">Category:</span>
            {CATEGORY_CHIPS.map(c => (
              <button
                key={c}
                onClick={() => setCategory(c)}
                className={`text-xs px-3 py-1 rounded-full border transition-colors ${
                  category === c
                    ? 'bg-accent text-bg border-accent'
                    : 'bg-bg border-border text-text-muted hover:border-accent hover:text-accent'
                }`}
              >
                {c}
              </button>
            ))}
          </div>

          {/* Event filter chips */}
          <div className="flex flex-wrap gap-2 mt-2">
            <span className="text-[10px] text-text-muted self-center uppercase tracking-wider">Event:</span>
            {EVENT_CHIPS.map(ev => (
              <button
                key={ev}
                onClick={() => setEventType(ev)}
                className={`text-xs px-3 py-1 rounded-full border transition-colors ${
                  eventType === ev
                    ? 'bg-accent text-bg border-accent'
                    : 'bg-bg border-border text-text-muted hover:border-accent hover:text-accent'
                }`}
              >
                {ev}
              </button>
            ))}
          </div>
        </div>

        {/* ── DB Stats panel (toggleable) ──────────────────────────── */}
        {statsOpen && dbStats && (
          <div className="card p-4 border border-border/60 text-sm">
            <h3 className="font-semibold text-text-primary mb-3 flex items-center gap-2">
              📊 Database Verification
              <span className="text-xs text-text-muted font-normal">Session #{dbStats.session_id}</span>
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {['sessions','reports','report_objects','object_events','tracked_objects','ocr_events'].map(t => (
                <div key={t} className="bg-bg rounded-lg p-3 border border-border/50">
                  <div className="text-[10px] text-text-muted uppercase tracking-wider mb-1">{t.replace('_',' ')}</div>
                  <div className="text-lg font-bold text-accent">{dbStats[t] ?? '—'}</div>
                  <div className="text-[10px] text-text-muted">this session</div>
                </div>
              ))}
            </div>
            {dbStats.all_sessions_totals && (
              <p className="text-xs text-text-muted mt-3">
                All-time totals — tracked_objects: <span className="text-accent">{dbStats.all_sessions_totals.tracked_objects}</span>,
                object_events: <span className="text-accent">{dbStats.all_sessions_totals.object_events}</span>,
                reports: <span className="text-accent">{dbStats.all_sessions_totals.reports}</span>
              </p>
            )}
          </div>
        )}

        {/* ── Error ────────────────────────────────────────────────── */}
        {error && (
          <div className="p-4 bg-warning/10 border border-warning/40 text-warning rounded-lg text-sm">
            ⚠️ {error}
          </div>
        )}

        {/* ── Result count ─────────────────────────────────────────── */}
        {!loading && !error && (
          <div className="flex items-center justify-between">
            <p className="text-xs text-text-muted">
              {results.length === 0
                ? 'No objects match — try a blank search to see all history.'
                : `${results.length} object${results.length !== 1 ? 's' : ''} found in session memory`}
            </p>
            {results.length > 0 && (
              <button
                onClick={() => { setQuery(''); setCategory('All'); setEventType('All') }}
                className="text-xs text-text-muted hover:text-accent transition-colors"
              >
                Clear filters
              </button>
            )}
          </div>
        )}

        {/* ── Loading skeleton ──────────────────────────────────────── */}
        {loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1,2,3].map(i => (
              <div key={i} className="card bg-card border border-border/50 p-4 animate-pulse">
                <div className="h-4 bg-border/40 rounded w-3/4 mb-3" />
                <div className="h-20 bg-border/30 rounded mb-3" />
                <div className="h-3 bg-border/30 rounded w-1/2" />
              </div>
            ))}
          </div>
        )}

        {/* ── Results grid ──────────────────────────────────────────── */}
        {!loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {results.map((r) => (
              <div key={`${r.track_id}-${r.first_seen}`} className="card bg-card border border-border p-4 flex flex-col gap-3 hover:border-accent/40 transition-colors">
                
                {/* Title row */}
                <div className="flex justify-between items-start">
                  <div className="min-w-0 flex-1">
                    <h3 className="font-semibold text-text-primary text-sm truncate" title={r.inferred_label}>
                      #{r.track_id} {r.inferred_label || r.display_label || '—'}
                    </h3>
                    <p className="text-xs text-text-muted mt-0.5">
                      {r.category}
                      {r.brand ? <span className="text-blue-400"> · {r.brand}</span> : ''}
                      {r.product_type ? <span className="text-purple-400"> · {r.product_type}</span> : ''}
                    </p>
                  </div>
                  <span className="ml-2 text-xs font-bold text-accent bg-accent/10 border border-accent/30 px-2 py-0.5 rounded-full whitespace-nowrap">
                    {((r.confidence || 0) * 100).toFixed(0)}%
                  </span>
                </div>

                {/* Snapshots */}
                {sessionId && (
                  <div className="flex gap-1.5 h-20">
                    {['first', 'best', 'last'].map(type => (
                      <img
                        key={type}
                        src={`${API_BASE}/snapshots/${sessionId}/${r.track_id}_${type}.jpg`}
                        alt={type}
                        title={`${type.charAt(0).toUpperCase() + type.slice(1)} seen`}
                        className="w-1/3 h-full object-cover rounded border border-border/40 bg-bg/50"
                        onError={e => { e.target.style.display = 'none' }}
                      />
                    ))}
                  </div>
                )}

                {/* Metadata grid */}
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                  {r.ocr_text && <>
                    <span className="text-text-muted">OCR Text</span>
                    <span className="text-blue-400 truncate text-right">{r.ocr_text}</span>
                  </>}
                  <span className="text-text-muted">First seen</span>
                  <span className="text-text-secondary text-right">{formatTime(r.first_seen)}</span>
                  <span className="text-text-muted">Last seen</span>
                  <span className="text-text-secondary text-right">{formatTime(r.last_seen)}</span>
                  <span className="text-text-muted">Duration</span>
                  <span className="text-text-secondary text-right">
                    {r.duration != null ? `${r.duration.toFixed(1)}s` : '—'}
                  </span>
                </div>

                {/* Events */}
                {r.events && r.events.length > 0 && (
                  <div className="pt-2 border-t border-border/40">
                    <p className="text-[10px] uppercase text-text-muted mb-1.5 tracking-wider">Events</p>
                    <div className="flex flex-wrap gap-1">
                      {r.events.map((ev, idx) => (
                        <span
                          key={idx}
                          title={ev.at}
                          className={`text-[10px] px-2 py-0.5 rounded-full border ${EVENT_COLORS[ev.type] || 'bg-border/30 text-text-muted border-border/50'}`}
                        >
                          {ev.type}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {!loading && results.length === 0 && !error && (
              <div className="col-span-full py-16 text-center">
                <div className="text-4xl mb-3">🔍</div>
                <p className="text-text-muted text-sm">No historical objects match your search.</p>
                <p className="text-text-muted text-xs mt-1">Objects appear here once the camera detects them with ≥80% confidence for ≥2 seconds.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default MemorySearch
