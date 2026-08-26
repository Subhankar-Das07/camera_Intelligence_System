# Docker development cheat sheet

Use these Windows `.bat` files from the **project root**. Keep Docker Desktop running.

## Cheat sheet

| Bat file | When to use | What it does | Typical time |
|----------|-------------|--------------|--------------|
| `docker-quick.bat` | Everyday **code** updates (`core/`, `pipelines/`, `static/`, `main.py`) | Ensures stack is up, then restarts `app` | ~5–15 sec |
| `docker-up.bat` | Containers stopped; just want the site running | `docker compose up -d` (no rebuild) | ~10–30 sec |
| `docker-rebuild.bat` | Changed `requirements.txt` or `Dockerfile` | `compose build` (no `--pull`) + `up -d` | Medium (layer cache) |
| `docker-refresh.bat` | Rare full refresh: stuck containers / need latest base image | `down` → `build --pull` → `up --force-recreate` | Long |
| `docker-down.bat` | Stop everything for the day | `docker compose down` (keeps volumes) | Fast |
| `docker-logs.bat` | Debug crashes / pipeline errors | `docker compose logs -f app` | Instant |

## Decision table

| You changed… | Run |
|--------------|-----|
| Python / JS / CSS / Site Admin / pipelines / `main.py` | **`docker-quick.bat`** |
| Nothing — just start the stack | **`docker-up.bat`** |
| `requirements.txt` or `Dockerfile` (packages / models) | **`docker-rebuild.bat`** — not `docker-refresh` unless rebuild fails |
| Still broken / want latest base image | **`docker-refresh.bat`** |
| Stop Docker for this project | **`docker-down.bat`** |
| See live app logs | **`docker-logs.bat`** |

## Why `docker-quick` is enough for code

[`docker-compose.yml`](../docker-compose.yml) mounts local source into the container:

- `./pipelines` → `/app/pipelines`
- `./core` → `/app/core`
- `./static` → `/app/static`
- `./main.py` → `/app/main.py`

Restarting `app` reloads that mounted code without reinstalling Python packages or re-downloading YOLO weights.

## When to use full refresh / Hub

- **`docker-rebuild.bat`** — deps or Dockerfile changed; still reuses local image layers when possible.
- **`docker-refresh.bat`** — nuclear option (`--pull` + force recreate). Prefer this only when rebuild did not fix the issue or you need a newer base image.
- **Hub pull (no local build)** — for teammates who only run the published image; see [`GUIDE.md`](../GUIDE.md) and [`TEAM_ONBOARDING.md`](../TEAM_ONBOARDING.md).

Open the app at http://localhost:8000 after start/restart.

**DVR on home LAN (e.g. `192.168.x.x`):** If **Cameras → Test connection** works on your PC but fails in Docker, the container cannot reach the DVR. Fix host/network access first; Live Monitor uses the same HTTP snapshot path as the camera test.

## Go-live parallel runtime (env)

When **Go live / scanning** is on, Site Admin runs up to **8 camera worker processes** in parallel. Frame capture is per camera; YOLO/pipeline runs share a capacity pool.

| Variable | Default | Meaning |
|----------|---------|---------|
| `SITE_ADMIN_MAX_CAMERAS` | `8` | Max parallel camera worker processes |
| `SITE_ADMIN_INFERENCE_SLOTS` | `2` | Max concurrent pipeline/YOLO runs across all workers |
| `SITE_ADMIN_WORKER_STAGGER_MS` | `250` | Delay between spawning each worker (protects DVR) |
| `SITE_ADMIN_TICK_MIN_SEC` | `1.0` | Fastest poll interval (high-priority rules) |
| `SITE_ADMIN_TICK_MAX_SEC` | `3.0` | Slowest poll interval (backoff / low priority) |

Example for a busy 8-channel DVR on a modest PC — add to `docker-compose.yml` under `app.environment`:

```yaml
SITE_ADMIN_MAX_CAMERAS: "8"
SITE_ADMIN_INFERENCE_SLOTS: "2"
SITE_ADMIN_TICK_MIN_SEC: "1.5"
```

After changing env vars, run `docker-quick.bat` (or `docker compose up -d --force-recreate` if compose env changed).

Check logs: `docker-logs.bat` — look for `spawned scan worker for camera` and `camera worker started`.

## Email alerts (SMTP)

Copy [`.env.example`](../.env.example) to `.env` and set the same SMTP vars used by Zerodha Kite (`PASSWORD_EMAIL` = Gmail app password). Compose passes them into the `app` container.

| Variable | Meaning |
|----------|---------|
| `SMTP_SERVER` | Default `smtp.gmail.com` |
| `SMTP_PORT` | Default `465` (SSL) |
| `USERNAME_EMAIL` | SMTP login |
| `PASSWORD_EMAIL` | App password |
| `EMAIL_FROM` | From address |

Then: Settings → Alert emails → Save → **Send email test**. Enable **Email** on a rule’s Channels. After changing `.env`, recreate the app container so env is picked up (`docker compose up -d --force-recreate app`).

## Fall detection rule types

| Rule | Use on |
|------|--------|
| **Fall detection** | Live video / RTSP / NVR / Live Monitor (velocity-based pipeline) |
| **Fall detection — standing & lying** | HTTP snapshot DVR channels / go-live polling (upright → lying transition) |

Manual test (standing & lying on DVR): create the rule with an ROI over the floor area, enable go-live, stand in view, then lie down and stay ~5 s. Expect an alert after two consecutive lying snapshots (~2–6 s). Hard-refresh Site Admin after `docker-quick.bat`.

## Live Monitor parallel rules

- **Max 3 enabled rules per camera** (enforced when saving rules).
- Monitor evaluates all rules **in parallel** (up to 3 threads) on each frame.
- **Event tracker + beep**: uses a **3s debounce per rule** (`SITE_ADMIN_MONITOR_DEBOUNCE_SEC`); Alerts tab still uses each rule's **60s cooldown**.
- **Parallel workers**: `SITE_ADMIN_MONITOR_RULE_WORKERS=3` (default 3).
- Click **Test beep** on Monitor before expecting automatic event sounds (unlocks browser audio).

## Gate analytics reports

- **Footfall** in Reports = `persons_in + persons_out` from **Gate / entrance analytics** rules.
- Reports support **Today**, **Last 7 days**, **This month**, and custom **hours** windows (window totals, not lifetime).
- Go-live **Apply rules on preview** increments gate counters into Redis (same as workers/monitor); alerts stay off on preview to avoid spam.
- Count accuracy depends on DVR snapshot poll interval (`SITE_ADMIN_TICK_*`); very fast crossings may be missed on slow polls.
- Go-live workers evaluate up to 3 rules per camera in parallel (`SITE_ADMIN_SCAN_RULE_WORKERS=3`).

## FastSAM (Suggest regions) and DVR snapshots

- **Model weights offline:** Run **`download-weights.bat`** (or `docker-rebuild.bat`, which downloads automatically if `models\` is empty). Weights must exist on the **Windows host** before Docker build — the image does **not** download from GitHub during build.
  ```powershell
  download-weights.bat
  docker-rebuild.bat
  ```
- **Suggest regions** uses **FastSAM-s.pt**. Optional runtime override: `SITE_ADMIN_FASTSAM_WEIGHTS=/app/FastSAM-s.pt`.
- **Person Re-ID (journeys):** optional `models/osnet_x0_25.onnx`. If missing, OpenCV appearance fallback is used. Env: `JOURNEY_REID_ENABLED=1`, `JOURNEY_HANDOFF_SEC=120`, `JOURNEY_REID_THRESHOLD=0.62`, `SITE_ADMIN_REID_WEIGHTS=models/osnet_x0_25.onnx`.
- **Emergency build without weights:** `SKIP_WEIGHT_DOWNLOAD=1` on the build arg/env lets the image build, but Suggest regions / some pipelines fail until weights exist.
- **DVR snapshot timeouts** (`ConnectTimeout` to `192.168.x.x`) usually mean the NVR is overloaded (8 workers + preview + Rules refresh). Limit load:
  - `SNAPSHOT_MAX_CONCURRENT_PER_HOST=2` (default) — max parallel `/picture` requests per DVR IP
  - `SNAPSHOT_FETCH_TIMEOUT=15` — connect/read timeout seconds
  - Slow polling: raise `SITE_ADMIN_TICK_MIN_SEC` to `1.5`–`2.0` in `docker-compose.yml`
