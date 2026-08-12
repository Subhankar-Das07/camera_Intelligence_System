/**
 * ObjectCards.jsx
 * ===============
 * Session Memory Object Panel — two sections:
 *
 *   ACTIVE              — currently visible (conf ≥ 80%, age ≥ 2s)
 *   PREVIOUSLY OBSERVED — left camera view but remain in session memory
 *
 * Objects in "Previously Observed" never disappear for the session duration.
 * They show "Last seen Xm Ys ago" and their preserved metadata.
 */

import React, { memo, useMemo } from 'react'

// ── Category config ───────────────────────────────────────────────────────────
const CATEGORY_ICONS = {
  Humans: '🧍', Vehicles: '🚗', Animals: '🐾', Electronics: '💻',
  Furniture: '🪑', Kitchen: '🍽️', Sports: '⚽', Tools: '🔧',
  Containers: '🎒', Household: '🏠', Other: '📦',
}

const CATEGORY_COLOURS = {
  Humans:      'from-blue-900/40 to-bg border-blue-700/40',
  Electronics: 'from-purple-900/40 to-bg border-purple-700/40',
  Animals:     'from-emerald-900/40 to-bg border-emerald-700/40',
  Vehicles:    'from-orange-900/40 to-bg border-orange-700/40',
  Kitchen:     'from-rose-900/40 to-bg border-rose-700/40',
  Furniture:   'from-amber-900/40 to-bg border-amber-700/40',
  Sports:      'from-cyan-900/40 to-bg border-cyan-700/40',
  Tools:       'from-slate-800/60 to-bg border-slate-600/40',
  Containers:  'from-teal-900/40 to-bg border-teal-700/40',
  Household:   'from-indigo-900/40 to-bg border-indigo-700/40',
  Other:       'from-slate-800/40 to-bg border-border',
}

// ── Confidence bar ────────────────────────────────────────────────────────────
const MiniConfBar = memo(({ confidence, dimmed }) => {
  const pct = Math.round(confidence * 100)
  const colour = dimmed
    ? 'bg-text-muted'
    : pct >= 95 ? 'bg-success' : pct >= 88 ? 'bg-accent' : 'bg-warning'
  return (
    <div className="w-full h-1 bg-border rounded-full overflow-hidden mt-1.5">
      <div
        className={`h-full rounded-full fps-bar-fill ${colour}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
})

// ── Active Object Card ────────────────────────────────────────────────────────
const ActiveCard = memo(({ obj }) => {
  const icon         = CATEGORY_ICONS[obj.category] ?? '📦'
  const gradientClass = CATEGORY_COLOURS[obj.category] ?? CATEGORY_COLOURS.Other
  const pct          = Math.round(obj.confidence * 100)

  return (
    <div
      id={`card-${obj.entity_id}`}
      className={`object-card relative border rounded-xl p-4 bg-gradient-to-b ${gradientClass}
        hover:shadow-card-hover transition-shadow duration-200 cursor-default`}
    >
      {/* Active pulse */}
      <div className="absolute top-2.5 right-2.5 flex items-center gap-1">
        <div className="status-dot status-dot-green live-ring" />
      </div>

      <div className="flex items-start gap-2.5">
        <span className="text-2xl select-none">{icon}</span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-bold text-text-primary truncate leading-tight pr-6" title={obj.inferred_display_label || obj.label}>
            {obj.inferred_display_label || obj.label}
          </p>
          <p className="text-[10px] text-text-muted">
            {obj.category}
            {obj.product_type ? ` • ${obj.product_type}` : (obj.brand ? ` • ${obj.brand}` : '')}
          </p>
        </div>
        <span className="text-lg font-bold font-mono text-text-primary leading-none flex-shrink-0">
          {pct}%
        </span>
      </div>

      <MiniConfBar confidence={obj.confidence} dimmed={false} />

      <div className="flex items-center justify-between mt-2.5">
        <span className="text-[11px] text-success font-medium capitalize">{obj.status}</span>
        <span className="text-[11px] font-mono text-text-muted">
          {obj.duration_str}
        </span>
      </div>

      {obj.gemini_verified && (
        <p className="text-[10px] text-accent mt-1">✦ Gemini Verified</p>
      )}

      {obj.best_text && (
        <p className="text-[10px] text-blue-400 mt-1 truncate" title={obj.best_text}>
          🔤 {obj.best_text}
        </p>
      )}

      {obj.relationships && obj.relationships.length > 0 && (
        <div className="mt-2 pt-2 border-t border-border/40">
          <p className="text-[10px] text-text-muted truncate">
            🔗 {obj.relationships[obj.relationships.length - 1]}
          </p>
        </div>
      )}
    </div>
  )
})

// ── Inactive / Previously Observed Card ──────────────────────────────────────
const InactiveCard = memo(({ obj }) => {
  const icon = CATEGORY_ICONS[obj.category] ?? '📦'
  const pct  = Math.round(obj.confidence * 100)

  return (
    <div
      id={`card-${obj.entity_id}`}
      className="object-card relative border border-border/50 rounded-xl p-4
        bg-card/30 opacity-75 hover:opacity-100 transition-opacity duration-200 cursor-default"
    >
      <div className="flex items-start gap-2.5">
        <span className="text-xl select-none opacity-60">{icon}</span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-text-secondary truncate leading-tight" title={obj.inferred_display_label || obj.label}>
            {obj.inferred_display_label || obj.label}
          </p>
          <p className="text-[10px] text-text-muted">
            {obj.category}
            {obj.product_type ? ` • ${obj.product_type}` : (obj.brand ? ` • ${obj.brand}` : '')}
          </p>
        </div>
        <span className="text-sm font-mono text-text-muted leading-none flex-shrink-0">
          {pct}%
        </span>
      </div>

      <MiniConfBar confidence={obj.confidence} dimmed={true} />

      {/* Last seen ago — most important field for inactive objects */}
      <div className="mt-2.5 flex items-center justify-between">
        <span className="text-[11px] text-warning font-medium">
          ⏱ {obj.last_seen_ago_str ?? obj.status}
        </span>
        <span className="text-[10px] font-mono text-text-muted">
          Visible: {obj.duration_str}
        </span>
      </div>

      {/* Preserved relationships */}
      {obj.relationship_counts && obj.relationship_counts.length > 0 && (
        <div className="mt-2 pt-2 border-t border-border/30">
          <p className="text-[10px] text-text-muted truncate">
            🔗 {obj.relationship_counts[0].description} ×{obj.relationship_counts[0].count}
          </p>
        </div>
      )}

      {obj.gemini_verified && (
        <p className="text-[10px] text-accent/60 mt-1">✦ Gemini Verified</p>
      )}

      {obj.best_text && (
        <p className="text-[10px] text-blue-400/60 mt-1 truncate" title={obj.best_text}>
          🔤 {obj.best_text}
        </p>
      )}
    </div>
  )
})

// ── Section Header ────────────────────────────────────────────────────────────
const SectionHeader = memo(({ label, count, badgeClass, subtitle }) => (
  <div className="flex items-center justify-between mb-3">
    <div className="flex items-center gap-2">
      <span className="section-title">{label}</span>
      {count > 0 && (
        <span className={`badge ${badgeClass}`}>{count}</span>
      )}
    </div>
    {subtitle && (
      <span className="text-[11px] text-text-muted">{subtitle}</span>
    )}
  </div>
))

// ── Empty Active State ────────────────────────────────────────────────────────
const EmptyActive = () => (
  <div className="col-span-full flex flex-col items-center justify-center py-8 text-center">
    <div className="text-4xl mb-3 opacity-30">👁️</div>
    <p className="text-text-secondary text-sm font-medium">No objects in view</p>
    <p className="text-text-muted text-xs mt-1">
      Waiting for objects with ≥80% confidence and ≥2s visibility…
    </p>
  </div>
)

// ── ObjectCards ───────────────────────────────────────────────────────────────
const ObjectCards = memo(({ objects, inactiveObjects }) => {
  // Client-side guard: double-filter active objects
  const active = useMemo(
    () => (objects ?? []).filter(o => o.confidence >= 0.80 && o.duration >= 2.0),
    [objects]
  )

  // Inactive: already filtered server-side but guard confidence
  const inactive = useMemo(
    () => (inactiveObjects ?? []).filter(o => o.confidence >= 0.80),
    [inactiveObjects]
  )

  const totalSession = active.length + inactive.length

  return (
    <section className="px-4 py-4 border-t border-border">
      <div className="max-w-screen-xl mx-auto space-y-6">

        {/* Session counter */}
        {totalSession > 0 && (
          <div className="flex items-center gap-3 text-xs text-text-muted">
            <div className="flex-1 h-px bg-border" />
            <span className="font-medium">
              Session Memory: {totalSession} unique object{totalSession !== 1 ? 's' : ''} observed
            </span>
            <div className="flex-1 h-px bg-border" />
          </div>
        )}

        {/* ── ACTIVE SECTION ─────────────────────────────────────────────── */}
        <div>
          <SectionHeader
            label="Active Objects"
            count={active.length}
            badgeClass="badge-success"
            subtitle="Currently in camera view · conf ≥ 80% · age ≥ 2s"
          />
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
            {active.length === 0 ? (
              <EmptyActive />
            ) : (
              active.map(obj => (
                <ActiveCard key={obj.entity_id ?? obj.track_id} obj={obj} />
              ))
            )}
          </div>
        </div>

        {/* ── PREVIOUSLY OBSERVED SECTION ────────────────────────────────── */}
        {inactive.length > 0 && (
          <div>
            <SectionHeader
              label="Previously Observed"
              count={inactive.length}
              badgeClass="badge-muted"
              subtitle="Left camera view · retained for session"
            />
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
              {inactive.map(obj => (
                <InactiveCard key={obj.entity_id ?? obj.track_id} obj={obj} />
              ))}
            </div>
          </div>
        )}

      </div>
    </section>
  )
})

export default ObjectCards
