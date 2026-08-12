/**
 * SceneReport.jsx
 * ===============
 * Session Intelligence Report — the complete session memory panel.
 *
 * Renamed from "Live Scene Report". This panel now shows:
 *  - Scene summary (active objects only)
 *  - ACTIVE OBJECTS section (detailed cards)
 *  - INACTIVE OBJECTS section (historical cards with 'last seen Xm ago')
 *  - Session relationships (with observation counts)
 *  - Session events (cumulative)
 *  - Category breakdowns (active vs. total session)
 *
 * Data NEVER shrinks during a session. The report grows monotonically.
 */

import React, { memo, useMemo } from 'react'

const CATEGORY_ICONS = {
  Humans: '🧍', Vehicles: '🚗', Animals: '🐾', Electronics: '💻',
  Furniture: '🪑', Kitchen: '🍽️', Sports: '⚽', Tools: '🔧',
  Containers: '🎒', Household: '🏠', Other: '📦',
}

// ── Confidence bar ────────────────────────────────────────────────────────────
const ConfBar = memo(({ value, dimmed }) => {
  const pct = Math.round(value * 100)
  const colour = dimmed
    ? 'bg-text-muted'
    : pct >= 90 ? 'bg-success' : pct >= 80 ? 'bg-accent' : 'bg-warning'
  return (
    <div className="flex items-center gap-2 mt-1">
      <div className="flex-1 h-1.5 bg-border rounded-full overflow-hidden">
        <div className={`h-full rounded-full fps-bar-fill ${colour}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-text-secondary w-8 text-right">{pct}%</span>
    </div>
  )
})

// ── Status badge ──────────────────────────────────────────────────────────────
const StatusBadge = memo(({ obj }) => {
  if (obj.state === 'inactive') {
    return (
      <span className="badge badge-warning">
        ⏱ {obj.last_seen_ago_str ?? 'inactive'}
      </span>
    )
  }
  const map = { new: 'badge-accent', tracked: 'badge-success', stationary: 'badge-warning' }
  return <span className={`badge ${map[obj.status] ?? 'badge-muted'}`}>{obj.status}</span>
})

// ── Metadata row ──────────────────────────────────────────────────────────────
const MetaRow = memo(({ label, value, valueClass = 'text-text-secondary' }) => (
  <div className="flex items-center justify-between text-[11px]">
    <span className="text-text-muted">{label}</span>
    <span className={`font-mono ${valueClass}`}>{value}</span>
  </div>
))

// ── Detail Card (used for both active and inactive) ───────────────────────────
const ObjectDetail = memo(({ obj, index, dimmed }) => {
  const icon = CATEGORY_ICONS[obj.category] ?? '📦'
  const borderClass = dimmed
    ? 'border-border/40 bg-card/20'
    : 'border-border bg-bg/50'

  return (
    <div className={`border rounded-lg p-3 animate-fade-in ${borderClass} ${dimmed ? 'opacity-80' : ''}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className={`text-lg ${dimmed ? 'opacity-50' : ''}`}>{icon}</span>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-text-primary truncate max-w-[200px]" title={obj.inferred_display_label || obj.label}>
                OBJECT #{index + 1} — {obj.inferred_display_label || obj.label}
              </span>
              {obj.gemini_verified && (
                <span className="badge badge-accent text-[10px]">✦ Gemini</span>
              )}
            </div>
            <span className="text-[11px] text-text-muted">
              {obj.category}
              {obj.product_type ? ` • ${obj.product_type}` : (obj.brand ? ` • ${obj.brand}` : '')}
            </span>
          </div>
        </div>
        <StatusBadge obj={obj} />
      </div>

      <ConfBar value={obj.confidence} dimmed={dimmed} />

      <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2.5">
        <MetaRow label="Track ID"   value={`#${obj.track_id}`} />
        <MetaRow label="Confidence" value={obj.confidence_pct} />
        {obj.brand && <MetaRow label="Brand" value={obj.brand} valueClass="text-accent" />}
        {obj.best_text && <MetaRow label="OCR Text" value={obj.best_text} valueClass="text-blue-400/80 max-w-[100px] truncate block text-right" />}
        <MetaRow label="First Seen" value={obj.duration_str} />
        <MetaRow label="Total Visible" value={obj.duration_str} />
        {obj.state === 'inactive' ? (
          <MetaRow
            label="Last Seen"
            value={obj.last_seen_ago_str ?? '—'}
            valueClass="text-warning"
          />
        ) : (
          <MetaRow
            label="Stationary"
            value={obj.is_stationary ? 'Yes' : 'No'}
            valueClass={obj.is_stationary ? 'text-warning' : 'text-text-secondary'}
          />
        )}
      </div>

      {/* Relationships with counts */}
      {obj.relationship_counts && obj.relationship_counts.length > 0 && (
        <div className="mt-2 pt-2 border-t border-border/50">
          <p className="text-[10px] uppercase tracking-wider text-text-muted mb-1">Relationships</p>
          <div className="flex flex-wrap gap-1">
            {obj.relationship_counts.slice(0, 4).map((r, i) => (
              <span key={i} className="badge badge-muted text-[10px]">
                {r.description} ×{r.count}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Events */}
      {obj.events && obj.events.length > 0 && (
        <div className="mt-1.5">
          <p className="text-[10px] uppercase tracking-wider text-text-muted mb-1">Events</p>
          <div className="flex flex-wrap gap-1">
            {obj.events.slice(-4).map((e, i) => (
              <span key={i} className="badge badge-warning text-[10px]">{e}</span>
            ))}
          </div>
        </div>
      )}

      {obj.gemini_description && (
        <div className="mt-2 pt-2 border-t border-border/50">
          <p className="text-[10px] uppercase tracking-wider text-text-muted mb-1">✦ Gemini Analysis</p>
          <p className="text-[11px] text-text-secondary leading-relaxed">{obj.gemini_description}</p>
        </div>
      )}
    </div>
  )
})

// ── Category pills ────────────────────────────────────────────────────────────
const CategoryPills = memo(({ byCategory, label }) => {
  if (!byCategory || Object.keys(byCategory).length === 0) return null
  return (
    <div className="mb-3">
      <p className="text-[10px] text-text-muted uppercase tracking-wider mb-1.5">{label}</p>
      <div className="flex flex-wrap gap-2">
        {Object.entries(byCategory).map(([cat, count]) => (
          <div key={cat} className="flex items-center gap-1.5 bg-bg/60 border border-border rounded-full px-2.5 py-1">
            <span className="text-sm">{CATEGORY_ICONS[cat] ?? '📦'}</span>
            <span className="text-xs text-text-primary font-medium">{count}</span>
            <span className="text-[10px] text-text-muted">{cat}</span>
          </div>
        ))}
      </div>
    </div>
  )
})

// ── Session Relationships (with counts) ───────────────────────────────────────
const RelationshipTimeline = memo(({ relationships }) => {
  if (!relationships || relationships.length === 0) return null
  return (
    <div className="card mb-4">
      <p className="section-title mb-3">Session Relationships</p>
      <div className="space-y-1.5">
        {relationships.slice(0, 8).map((r, i) => (
          <div key={i} className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-1.5">
              <span className={`status-dot ${r.state === 'active' ? 'status-dot-green' : 'status-dot-amber'}`} />
              <span className="text-text-secondary">{r.label}</span>
              <span className="text-text-muted">·</span>
              <span className="text-text-muted">{r.relationship}</span>
            </div>
            <span className="badge badge-muted text-[10px]">×{r.count}</span>
          </div>
        ))}
      </div>
    </div>
  )
})

// ── Section divider ───────────────────────────────────────────────────────────
const SectionDivider = memo(({ label, count, active }) => (
  <div className="flex items-center gap-3 my-4">
    <div className="flex-1 h-px bg-border" />
    <div className="flex items-center gap-2">
      <div className={`status-dot ${active ? 'status-dot-green' : 'status-dot-amber'}`} />
      <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
        {label}
      </span>
      <span className={`badge ${active ? 'badge-success' : 'badge-muted'}`}>{count}</span>
    </div>
    <div className="flex-1 h-px bg-border" />
  </div>
))

// ── SceneReport ───────────────────────────────────────────────────────────────
const SceneReport = memo(({ report }) => {
  const summary          = report?.summary ?? 'Waiting for session data…'
  const activeObjects    = report?.objects ?? []
  const inactiveObjects  = report?.inactive_objects ?? []
  const stability        = report?.scene_stability ?? 0
  const sessionTime      = report?.session_time_str ?? '0s'
  const activeCat        = report?.active_by_category ?? {}
  const sessionCat       = report?.session_by_category ?? {}
  const relationships    = report?.relationships ?? []

  // Client-side guard for active
  const filteredActive = useMemo(
    () => activeObjects.filter(o => o.confidence >= 0.80 && o.duration >= 2.0),
    [activeObjects]
  )

  // Client-side guard for inactive
  const filteredInactive = useMemo(
    () => inactiveObjects.filter(o => o.confidence >= 0.80),
    [inactiveObjects]
  )

  const totalSession = filteredActive.length + filteredInactive.length

  return (
    <section className="px-4 py-4 border-t border-border">
      <div className="max-w-screen-xl mx-auto">

        {/* Section header */}
        <div className="card-header mb-4">
          <div className="flex items-center gap-2">
            <span className="section-title">Session Intelligence Report</span>
            {totalSession > 0 && (
              <span className="badge badge-accent">
                {totalSession} session object{totalSession !== 1 ? 's' : ''}
              </span>
            )}
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5 text-xs text-text-muted">
              <span>Uptime</span>
              <span className="font-mono text-text-secondary">{sessionTime}</span>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-text-muted">
              <span>Stability</span>
              <span className="font-mono text-text-secondary">{stability.toFixed(1)}%</span>
            </div>
            <div className="w-16 h-1.5 bg-border rounded-full overflow-hidden">
              <div className="h-full bg-accent rounded-full fps-bar-fill" style={{ width: `${stability}%` }} />
            </div>
          </div>
        </div>

        {/* Scene summary */}
        <div className="card mb-4">
          <p className="text-sm font-medium text-text-secondary">
            <span className="text-accent font-semibold mr-2">Scene Summary</span>
            {summary}
          </p>
          {filteredInactive.length > 0 && (
            <p className="text-xs text-text-muted mt-1.5">
              📚 Session memory contains {filteredInactive.length} previously observed object{filteredInactive.length !== 1 ? 's' : ''}.
            </p>
          )}
        </div>

        {/* Category breakdowns */}
        <CategoryPills byCategory={activeCat} label="Active Now" />
        {Object.keys(sessionCat).length > Object.keys(activeCat).length && (
          <CategoryPills byCategory={sessionCat} label="All Session" />
        )}

        {/* Session relationships */}
        <RelationshipTimeline relationships={relationships} />

        {/* ── ACTIVE OBJECTS ──────────────────────────────────────────────── */}
        {filteredActive.length > 0 && (
          <>
            <SectionDivider
              label="Active Objects"
              count={filteredActive.length}
              active={true}
            />
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
              {filteredActive.map((obj, i) => (
                <ObjectDetail key={obj.entity_id ?? obj.track_id} obj={obj} index={i} dimmed={false} />
              ))}
            </div>
          </>
        )}

        {/* ── INACTIVE OBJECTS ─────────────────────────────────────────────── */}
        {filteredInactive.length > 0 && (
          <>
            <SectionDivider
              label="Previously Observed"
              count={filteredInactive.length}
              active={false}
            />
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
              {filteredInactive.map((obj, i) => (
                <ObjectDetail
                  key={obj.entity_id ?? obj.track_id}
                  obj={obj}
                  index={filteredActive.length + i}
                  dimmed={true}
                />
              ))}
            </div>
          </>
        )}

        {/* Empty state */}
        {totalSession === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="text-5xl mb-4 opacity-30">🧠</div>
            <p className="text-text-secondary text-sm font-medium">Session memory is empty</p>
            <p className="text-text-muted text-xs mt-1">
              Objects with ≥80% confidence will appear here and remain for the entire session.
            </p>
          </div>
        )}
      </div>
    </section>
  )
})

export default SceneReport
