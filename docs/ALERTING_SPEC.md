# Alerting spec

- **Trigger:** Site Admin rule match while **Go live** is on, or during **Live Monitor** (same alert inbox).
- **Store:** Redis list `alert:inbox` (newest first, capped).
- **UI:** Site Admin → Alerts (ack, mute rule 1 hour, share WhatsApp, **Email** per row or **Email selected** for checked alerts). Manual email uses `POST /api/site-admin/alerts/{id}/share-email` and `POST /api/site-admin/alerts/share-email` (admin).
- **Live Monitor:** MJPEG at `/api/site-admin/monitor/stream/{session_id}` — annotated frames with all rule ROIs; background scan pauses for that camera while monitored.
- **Channels:** `web` always via inbox; `whatsapp` if the rule includes it and Cloud API env is set; `email` if the rule includes it and SMTP env is set.
- **WhatsApp:** Meta Cloud API (`WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`). No unofficial WhatsApp clients. Recipients: Settings → WhatsApp numbers.
- **Email:** SMTP (`SMTP_SERVER`, `SMTP_PORT`, `USERNAME_EMAIL`, `PASSWORD_EMAIL`, `EMAIL_FROM`) — same names as Zerodha Kite backend. Primary contact comes from **Setup wizard → Site profile** (`contact_name` / `contact_email`); that address is kept first in Settings → Alert emails (extras allowed). Saving or changing contact email sends a welcome confirmation when SMTP is configured. Test via `POST /api/site-admin/email/test`. New rules default Email channel on when a contact email is set. Alert emails are **HTML** with an **inline CID snapshot** (not a `/storage/...` path). Manual share and rule-channel email both use this template.
- **Cooldown:** per-rule (default 60s) to avoid floods.
- **Mute:** Redis TTL key `alert:mute:{rule_id}`.
- **Quiet hours:** when set on the site profile, WhatsApp and email are suppressed (web inbox still receives alerts). Manual Email share from Alerts is not blocked by quiet hours.
