/**
 * App.jsx
 * =======
 * Root Dashboard — Session Memory Edition.
 *
 * Data flow: single WebSocket → distribute to all panels.
 * Both ACTIVE and INACTIVE (session history) objects flow through.
 */

import React, { useState, useMemo } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import Header from './components/Header'
import CameraView from './components/CameraView'
import SceneReport from './components/SceneReport'
import ObjectCards from './components/ObjectCards'
import SystemStatus from './components/SystemStatus'
import MemorySearch from './components/MemorySearch'
import TimelineView from './components/TimelineView'
import Diagnostics from './components/Diagnostics'
import ObjectWatchlist from './components/ObjectWatchlist'

export default function App() {
  const { data, connected } = useWebSocket()
  const [activeTab, setActiveTab] = useState('Dashboard')

  const status          = useMemo(() => data?.status ?? {}, [data?.status])
  const report          = useMemo(() => data?.report ?? {}, [data?.report])
  const activeObjects   = useMemo(() => data?.objects ?? [], [data?.objects])
  const inactiveObjects = useMemo(() => data?.inactive_objects ?? [], [data?.inactive_objects])
  // Watchlist state — forwarded directly from WebSocket push (no extra fetch)
  const wsWatchlist     = data?.watchlist ?? undefined

  const headerStatus = useMemo(() => ({
    ...status,
    active_objects:  report.active_objects  ?? 0,
    session_objects: report.session_objects ?? (activeObjects.length + inactiveObjects.length),
    fps:             report.fps    ?? status.fps    ?? 0,
    cpu:             report.cpu    ?? status.cpu    ?? 0,
    ram_mb:          report.ram_mb ?? status.ram_mb ?? 0,
  }), [status, report, activeObjects.length, inactiveObjects.length])

  return (
    <div className="min-h-screen bg-bg flex flex-col">

      <Header
        status={headerStatus}
        connected={connected}
        sessionTime={report.session_time ?? 0}
      />
      
      {/* Navigation Tabs */}
      <div className="bg-card border-b border-border px-6 flex gap-6">
        {['Dashboard', 'Objects', 'Memory Search', 'Timeline', 'Diagnostics', 'System'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab 
                ? 'border-accent text-accent' 
                : 'border-transparent text-text-muted hover:text-text-secondary'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      <main className="flex-1 flex flex-col overflow-hidden">
        {activeTab === 'Dashboard' && (
          <div className="overflow-y-auto">
            <CameraView connected={connected} />
            {/* MONITORING — feature panels below the camera feed */}
            <ObjectWatchlist wsWatchlist={wsWatchlist} />
            <SceneReport
              report={{ ...report, objects: activeObjects, inactive_objects: inactiveObjects }}
            />
          </div>
        )}

        {activeTab === 'Objects' && (
          <div className="overflow-y-auto p-4">
            <ObjectCards
              objects={activeObjects}
              inactiveObjects={inactiveObjects}
            />
          </div>
        )}
        
        {activeTab === 'Memory Search' && (
          <MemorySearch />
        )}
        
        {activeTab === 'Timeline' && (
          <TimelineView connected={connected} />
        )}
        
        {activeTab === 'Diagnostics' && (
          <div className="overflow-y-auto p-4">
            <Diagnostics />
          </div>
        )}

        {activeTab === 'System' && (
          <div className="overflow-y-auto p-4">
            <SystemStatus status={status} report={report} />
          </div>
        )}
      </main>

      <footer className="py-3 px-6 border-t border-border bg-card/50 text-center">
        <p className="text-[11px] text-text-muted">
          Smart Vision Assistant · Session Memory ·
          {' '}<span className={`font-medium ${connected ? 'text-success' : 'text-warning'}`}>
            {connected ? 'WebSocket Connected' : 'Reconnecting…'}
          </span>
          {' '}·
          {' '}<span className="text-accent font-medium">
            {(report.active_objects ?? 0)} active
          </span>
          {' '}·
          {' '}<span className="text-text-secondary">
            {(report.inactive_objects?.length ?? inactiveObjects.length)} in history
          </span>
        </p>
      </footer>
    </div>
  )
}
