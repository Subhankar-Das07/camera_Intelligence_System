/**
 * sw.js — Edge Vision Service Worker
 *
 * Caches the app shell (HTML + static assets) for fast loading on mobile.
 * Network-first strategy: always tries the network; falls back to cache
 * if offline, so the app shell still loads.
 */

const CACHE_NAME = 'edge-vision-v1';

const PRECACHE = [
  '/',
  '/static/manifest.json',
];

// ── Install: pre-cache the shell ────────────────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(PRECACHE))
  );
  self.skipWaiting();
});

// ── Activate: clean up old caches ────────────────────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// ── Fetch: network-first, cache fallback ─────────────────────────────────────
self.addEventListener('fetch', event => {
  // Skip WebSocket requests (not fetch API)
  if (event.request.url.startsWith('ws://') || event.request.url.startsWith('wss://')) return;

  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Clone and cache successful GET responses
        if (event.request.method === 'GET' && response.status === 200) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
