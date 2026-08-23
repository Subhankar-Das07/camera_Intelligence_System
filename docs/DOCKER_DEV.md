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
| `requirements.txt` or `Dockerfile` (packages / models) | **`docker-rebuild.bat`** |
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
