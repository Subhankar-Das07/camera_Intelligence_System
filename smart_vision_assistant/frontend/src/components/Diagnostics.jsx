import React, { useState, useEffect } from 'react'

const Diagnostics = () => {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchDiagnostics = async () => {
      try {
        const res = await fetch('/api/diagnostics')
        if (!res.ok) throw new Error('Failed to fetch diagnostics')
        const json = await res.json()
        setData(json)
      } catch (err) {
        setError(err.message)
      }
    }

    fetchDiagnostics()
    const interval = setInterval(fetchDiagnostics, 1000)
    return () => clearInterval(interval)
  }, [])

  if (error) {
    return (
      <div className="p-4 m-4 bg-red-900/50 border border-red-500 rounded text-red-200">
        Error loading diagnostics: {error}
      </div>
    )
  }

  if (!data) {
    return (
      <div className="p-4 m-4 text-text-muted">
        Loading diagnostics...
      </div>
    )
  }

  const { ocr_metrics, system, status } = data
  
  // Calculate success rate
  const successRate = ocr_metrics.completed > 0 
    ? Math.round((ocr_metrics.text_found / ocr_metrics.completed) * 100) 
    : 0

  return (
    <div className="max-w-screen-xl mx-auto p-4 space-y-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-text-primary">System Diagnostics</h2>
        <span className={`badge ${status === 'ok' ? 'badge-success' : 'badge-warning'}`}>
          Engine: {status}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* System Metrics */}
        <div className="card">
          <h3 className="section-title mb-3">System Performance</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-text-muted">FPS</span>
              <span className="font-mono">{system?.fps?.toFixed(1) || '0.0'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">CPU Usage</span>
              <span className="font-mono">{system?.cpu?.toFixed(1) || '0.0'}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Memory</span>
              <span className="font-mono">{system?.ram_mb?.toFixed(0) || '0'} MB</span>
            </div>
          </div>
        </div>

        {/* OCR Metrics */}
        <div className="card">
          <h3 className="section-title mb-3 text-blue-400">OCR Telemetry</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-text-muted">Queue Size</span>
              <span className="font-mono text-accent">{ocr_metrics.queue_size}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Crops Submitted</span>
              <span className="font-mono">{ocr_metrics.submitted}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Tasks Completed</span>
              <span className="font-mono text-success">{ocr_metrics.completed}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Text Found</span>
              <span className="font-mono">{ocr_metrics.text_found}</span>
            </div>
            <div className="flex justify-between pt-2 border-t border-border">
              <span className="text-text-muted font-medium">Success Rate</span>
              <span className={`font-mono font-bold ${successRate > 50 ? 'text-success' : 'text-warning'}`}>
                {successRate}%
              </span>
            </div>
          </div>
        </div>

        {/* Pipeline Stats */}
        <div className="card">
          <h3 className="section-title mb-3 text-purple-400">Pipeline Stats</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-text-muted">Entities Enriched</span>
              <span className="font-mono text-success">{ocr_metrics.entities_enriched}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Dropped Frames</span>
              <span className="font-mono text-warning">{ocr_metrics.dropped}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Total Events</span>
              <span className="font-mono">{data.total_events}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Diagnostics
