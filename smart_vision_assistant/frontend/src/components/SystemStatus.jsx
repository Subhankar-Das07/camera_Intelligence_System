/**
 * SystemStatus.jsx
 * ================
 * System health panel — bottom of the dashboard.
 *
 * Displays:
 *  - Component status rows: YOLO / Tracker / Database / Gemini
 *  - Live metrics: FPS / CPU / RAM
 *  - FPS health bar
 */

import React, { memo, useState, useEffect } from 'react'

const API_BASE = 'http://localhost:8000'


// ── Status row ────────────────────────────────────────────────────────────────
const StatusRow = memo(({ label, value, state }) => {
  const dotClass = {
    running: 'status-dot-green',
    connected: 'status-dot-green',
    ready: 'status-dot-green',
    unavailable: 'status-dot-amber',
    error: 'status-dot-red',
  }[state?.toLowerCase()] ?? 'status-dot-amber'

  const valueClass = {
    running: 'text-success',
    connected: 'text-success',
    ready: 'text-success',
    unavailable: 'text-warning',
    error: 'text-danger',
  }[state?.toLowerCase()] ?? 'text-text-secondary'

  return (
    <div className="flex items-center justify-between py-2 border-b border-border/40 last:border-0">
      <span className="text-xs text-text-muted">{label}</span>
      <div className="flex items-center gap-1.5">
        <div className={`status-dot ${dotClass}`} />
        <span className={`text-xs font-medium capitalize ${valueClass}`}>
          {value ?? state ?? '—'}
        </span>
      </div>
    </div>
  )
})

// ── Metric block ──────────────────────────────────────────────────────────────
const MetricBlock = memo(({ label, value, unit, colour = 'text-text-primary', barPct }) => (
  <div className="card flex-1">
    <div className="flex items-end justify-between mb-2">
      <span className="text-[10px] uppercase tracking-widest text-text-muted">{label}</span>
      <span className={`text-2xl font-bold font-mono ${colour}`}>
        {value}<span className="text-sm font-normal text-text-muted ml-0.5">{unit}</span>
      </span>
    </div>
    {barPct !== undefined && (
      <div className="w-full h-1.5 bg-border rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full fps-bar-fill ${colour.replace('text-', 'bg-')}`}
          style={{ width: `${Math.min(100, barPct)}%` }}
        />
      </div>
    )}
  </div>
))

// ── SystemStatus ──────────────────────────────────────────────────────────────
const SystemStatus = memo(({ status, report }) => {
  const fps = report?.fps ?? status?.fps ?? 0
  const cpu = report?.cpu ?? status?.cpu ?? 0
  const ramMb = report?.ram_mb ?? status?.ram_mb ?? 0

  const [dbStats, setDbStats] = useState(null)

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/db-stats`)
        if (res.ok) setDbStats(await res.json())
      } catch (_) {}
    }
    fetchStats()
    const interval = setInterval(fetchStats, 10000)
    return () => clearInterval(interval)
  }, [])

  // FPS colour
  const fpsColour = fps >= 20 ? 'text-success' : fps >= 10 ? 'text-warning' : 'text-danger'
  const cpuColour = cpu < 60 ? 'text-success' : cpu < 80 ? 'text-warning' : 'text-danger'

  const yolo = status?.yolo ?? 'running'
  const tracker = status?.tracker ?? 'running'
  const database = status?.database ?? 'connected'
  const gemini = status?.gemini ?? 'ready'

  return (
    <section className="px-4 py-4 pb-8 border-t border-border">
      <div className="max-w-screen-xl mx-auto">
        <div className="section-title mb-4">System Status</div>

        <div className="flex flex-col lg:flex-row gap-4">

          {/* Component status */}
          <div className="card flex-shrink-0 w-full lg:w-64">
            <p className="text-[10px] uppercase tracking-widest text-text-muted mb-2">Components</p>
            <StatusRow label="YOLO Detector" state={yolo} />
            <StatusRow label="BoT-SORT Tracker" state={tracker} />
            <StatusRow label="Database" state={database} />
            <StatusRow label="Gemini" state={gemini} />
          </div>

          {/* Live metrics */}
          <div className="flex flex-col sm:flex-row gap-3 flex-1">
            <MetricBlock
              label="FPS"
              value={fps.toFixed(1)}
              unit=""
              colour={fpsColour}
              barPct={(fps / 30) * 100}
            />
            <MetricBlock
              label="CPU"
              value={Math.round(cpu)}
              unit="%"
              colour={cpuColour}
              barPct={cpu}
            />
            <MetricBlock
              label="RAM"
              value={ramMb >= 1024 ? (ramMb / 1024).toFixed(1) : Math.round(ramMb)}
              unit={ramMb >= 1024 ? 'GB' : 'MB'}
              colour="text-accent"
              barPct={(ramMb / 2048) * 100}
            />
          </div>

          {/* Session info */}
          <div className="card flex-shrink-0 w-full lg:w-48">
            <p className="text-[10px] uppercase tracking-widest text-text-muted mb-2">Session</p>
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-text-muted">Total Seen</span>
                <span className="font-mono text-text-secondary">
                  {report?.total_registered ?? '—'}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-text-muted">Active Now</span>
                <span className="font-mono text-success">
                  {report?.active_objects ?? '—'}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-text-muted">Stability</span>
                <span className="font-mono text-text-secondary">
                  {report?.scene_stability !== undefined
                    ? `${report.scene_stability.toFixed(1)}%`
                    : '—'}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-text-muted">Uptime</span>
                <span className="font-mono text-text-secondary">
                  {report?.session_time_str ?? '—'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* DB Record counts */}
        {dbStats && (
          <div className="mt-4">
            <div className="section-title mb-3 text-[10px] uppercase tracking-widest text-text-muted">Database Records (This Session)</div>
            <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
              {['tracked_objects','object_events','reports','report_objects','sessions','ocr_events'].map(t => (
                <div key={t} className="card text-center py-3">
                  <div className="text-xl font-bold font-mono text-accent">{dbStats[t] ?? '—'}</div>
                  <div className="text-[9px] uppercase tracking-wide text-text-muted mt-1">{t.replace(/_/g,' ')}</div>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-text-muted mt-2">
              DB: <span className="text-text-secondary">{dbStats.db_path ?? '—'}</span>
            </p>
          </div>
        )}
      </div>
    </section>
  )
})

export default SystemStatus
