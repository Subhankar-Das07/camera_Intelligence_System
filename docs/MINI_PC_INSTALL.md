# Mini-PC install

1. Fanless PC, 8–16 GB RAM, same LAN as the NVR.
2. Install Docker Engine or Docker Desktop.
3. Copy the project or pull `drpinfotech/camera-intelligence:develop`.
4. Start:

```bat
docker compose -f docker-compose.edge.yml up -d
```

5. Open `http://<mini-pc-ip>:8000/features/site-admin/`
6. Complete Wizard: site → cameras (RTSP test) → rules → Go live.

Ports: **8000** (app), **6379** (Redis). Backup Redis volume `redis_data` and `app_storage`.
