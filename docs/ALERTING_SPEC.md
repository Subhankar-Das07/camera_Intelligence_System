# Alerting spec

- **Trigger:** Site Admin rule match while **Go live** is on, or during **Live Monitor** (same alert inbox).
- **Store:** Redis list `alert:inbox` (newest first, capped).
- **UI:** Site Admin → Alerts (ack, mute rule 1 hour, share WhatsApp).
- **Live Monitor:** MJPEG at `/api/site-admin/monitor/stream/{session_id}` — annotated frames with all rule ROIs; background scan pauses for that camera while monitored.
- **Channels:** `web` always via inbox; `whatsapp` if the rule includes it and Cloud API env is set.
- **WhatsApp:** Meta Cloud API (`WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`). No unofficial WhatsApp clients.
- **Cooldown:** per-rule (default 60s) to avoid floods.
- **Mute:** Redis TTL key `alert:mute:{rule_id}`.
