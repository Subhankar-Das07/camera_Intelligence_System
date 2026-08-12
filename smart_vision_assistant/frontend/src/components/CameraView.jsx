/**
 * CameraView.jsx
 * ==============
 * Live MJPEG camera feed panel.
 *
 * Uses a plain <img> tag pointing to the MJPEG stream endpoint.
 * The browser handles multipart streaming natively — no WebSocket
 * overhead for video data.
 *
 * The img src switches between the live stream and a placeholder
 * based on connection state.
 */

import React, { memo, useState, useCallback } from 'react'

const STREAM_URL = 'http://localhost:8000/video/stream'

// ── Camera status overlay ────────────────────────────────────────────────────
const CameraOverlay = memo(({ streamError }) => (
  <div className="absolute inset-0 flex flex-col items-center justify-center bg-bg/80 backdrop-blur-sm">
    <div className="text-center">
      <div className="text-4xl mb-3">📷</div>
      <p className="text-text-secondary text-sm font-medium">
        {streamError ? 'Camera stream unavailable' : 'Connecting to camera…'}
      </p>
      <p className="text-text-muted text-xs mt-1">
        {streamError
          ? 'Ensure the FastAPI server is running on port 8000'
          : 'Starting SmartVisionHeadless…'}
      </p>
    </div>
  </div>
))

// ── Corner badge ─────────────────────────────────────────────────────────────
const CornerBadge = memo(({ connected }) => (
  <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-bg/70 backdrop-blur-sm px-2.5 py-1 rounded-full border border-border">
    <div className={`status-dot ${connected ? 'status-dot-green live-ring' : 'status-dot-red'}`} />
    <span className="text-[11px] font-semibold text-text-primary uppercase tracking-wider">
      {connected ? 'Live' : 'Offline'}
    </span>
  </div>
))

// ── Filter badge (always visible) ────────────────────────────────────────────
const FilterBadge = () => (
  <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-bg/70 backdrop-blur-sm px-2.5 py-1 rounded-full border border-border">
    <span className="text-[11px] font-medium text-accent">≥80% conf · ≥2s age</span>
  </div>
)

// ── CameraView ───────────────────────────────────────────────────────────────
const CameraView = memo(({ connected }) => {
  const [streamError, setStreamError] = useState(false)
  const [streamLoaded, setStreamLoaded] = useState(false)

  const handleLoad = useCallback(() => {
    setStreamLoaded(true)
    setStreamError(false)
  }, [])

  const handleError = useCallback(() => {
    setStreamError(true)
    setStreamLoaded(false)
  }, [])

  return (
    <section className="relative w-full bg-[#020617] overflow-hidden" style={{ height: '65vh', minHeight: '360px' }}>

      {/* MJPEG stream */}
      <img
        id="camera-feed"
        src={STREAM_URL}
        alt="Live camera stream"
        className="camera-feed"
        onLoad={handleLoad}
        onError={handleError}
        style={{ opacity: streamLoaded ? 1 : 0, transition: 'opacity 0.4s' }}
      />

      {/* Overlay shown until stream loads */}
      {(!streamLoaded || streamError) && (
        <CameraOverlay streamError={streamError} />
      )}

      {/* Corner badges */}
      <CornerBadge connected={connected && streamLoaded} />
      <FilterBadge />

      {/* Subtle gradient vignette at bottom */}
      <div className="absolute bottom-0 inset-x-0 h-16 bg-gradient-to-t from-bg/60 to-transparent pointer-events-none" />
    </section>
  )
})

export default CameraView
