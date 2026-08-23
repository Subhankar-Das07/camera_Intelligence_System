# Edge architecture

```
CCTV / NVR (RTSP)  →  Mini-PC Docker (FastAPI + pipelines + Redis)
Uploaded files     ↗
                         → Site Admin APIs (`/api/site-admin`)
                         → Alert inbox + optional WhatsApp Cloud API
```

- **Redis** holds site profile, cameras, rules, alerts (`site:`, `cam:`, `rule:`, `alert:`). Face/vehicle prefixes `fr:` / `vr:` stay with those modules.
- **storage/** holds uploads, previews, alert clips (Docker volume `app_storage`).
- **Go live** starts a background scanner that runs registered pipelines on each enabled camera/rule burst.
- Compose for 24/7: `docker compose -f docker-compose.edge.yml up -d`
