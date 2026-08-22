# Camera Intelligence System — Team Onboarding

Handout for the three feature developers. Follow this document to install Docker, pull the Hub image, clone Git, create your feature branch from `main`, merge safely, and resolve conflicts.

**Main repository (source of truth):** https://github.com/endevs/camera_Intelligence  
**Base branch:** `main` — https://github.com/endevs/camera_Intelligence/tree/main  
**Docker Hub image:** `drpinfotech/camera-intelligence:develop` (also tagged `0.1.0`)  
https://hub.docker.com/r/drpinfotech/camera-intelligence

---

## 1. Welcome and roles

Product: **Camera Intelligence System** — unified web UI + FastAPI vision pipelines (zone safety, vehicle plates, face recognition). Data for vehicles, faces, and sessions is stored in **Redis**.

| Developer focus | Feature branch | Own primarily (avoid editing others’ folders) |
|-----------------|----------------|-----------------------------------------------|
| Zone Safety | `feature/zone-safety` | `static/features/zone-safety/`, zone pipelines (`intrusion_detection`, `danger_zone`, `fall_detection`) |
| Vehicle | `feature/vehicle` | `static/features/vehicle/`, `pipelines/vehicle_recognition/` |
| Face | `feature/face` | `static/features/face/`, `face_recognition/`, face pipeline |

**Site Admin** (product ops UI) lives in `static/features/site-admin/` plus `/api/site-admin/`. Do **not** edit Zone Safety / Vehicle / Face UI to add product features — keep those folders isolated.

**Shared areas** (coordinate before changing): `static/shell/`, `static/shared/`, `main.py`, `core/`, `docker-compose*.yml`, `requirements.txt`.

---

## 2. Prerequisites / requirements

| Requirement | Notes |
|-------------|--------|
| OS | Windows 10/11 (or Linux / macOS) |
| Git | Installed; GitHub account **invited** to private repo `endevs/camera_Intelligence` |
| Docker Desktop | Required for Hub deploy path; on Windows use **WSL2** backend |
| RAM | 8 GB+ recommended (AI models) |
| Ports free | **8000** (app), **6379** (Redis in Compose) |
| Optional | Python **3.13** for local (non-Docker) runs; NVIDIA GPU optional for faster YOLO |

You must be able to open the private GitHub repo and pull the Docker image (ask the maintainer for GitHub invite + Hub access if pull is denied).

---

## 3. Technology stack

| Layer | Technology |
|-------|------------|
| API server | FastAPI, Uvicorn, python-multipart, WebSockets |
| Computer vision | OpenCV, Ultralytics YOLO, NumPy, Shapely |
| Zone Safety | Pipelines: intrusion, danger zone, fall detection |
| Face | InsightFace (SCRFD + ArcFace), ONNX Runtime, FAISS, Supervision (ByteTrack), Pillow |
| Vehicle | YOLO + RapidOCR (ONNX) |
| Database | **Redis only** (vehicles, face identities/embeddings/crops, session caches). Uploads & alert clips on Docker volume `app_storage` |
| Web UI | Static shell + feature pages under `static/features/` |
| Mobile (optional) | Flutter app in `edge_vision_app/` |
| Deploy | Docker Compose: `app` + `redis`; image `drpinfotech/camera-intelligence` |

Python dependencies are listed in `requirements.txt`.

---

## 4. Install Docker

### Windows

1. Download and install **Docker Desktop**: https://www.docker.com/products/docker-desktop/
2. During setup, enable **WSL 2**.
3. Start Docker Desktop and wait until it shows **Running**.
4. Open PowerShell or Command Prompt and verify:

```bat
docker version
docker compose version
```

5. If the Hub image is private, log in:

```bat
docker login
```

Use an account that can pull `drpinfotech/camera-intelligence`.

### Notes

- First start of the app may download YOLO / InsightFace models and take several minutes.
- Keep Docker Desktop running while you use the stack.

---

## 5. Pull the Docker image and run (no local image build)

### Recommended path for all developers

```bat
docker pull drpinfotech/camera-intelligence:develop

git clone https://github.com/endevs/camera_Intelligence.git
cd camera_Intelligence

docker compose -f docker-compose.yml -f docker-compose.hub.yml up -d --no-build
```

- Open the UI: **http://localhost:8000**
- You should see **Overview** and top nav: Zone Safety | Vehicle Recognition | Face Recognition

### Stop the stack

```bat
docker compose -f docker-compose.yml -f docker-compose.hub.yml down
```

### Useful links

- Hub: https://hub.docker.com/r/drpinfotech/camera-intelligence  
- Tags: `develop` (rolling) and `0.1.0` (release)

### Local Docker scripts (after you have a running stack)

Everyday **code** changes (`core/`, `pipelines/`, `static/`, `main.py`) → run **`docker-quick.bat`** (fast restart; no re-download).

Full decision table and all bats: [`docs/DOCKER_DEV.md`](docs/DOCKER_DEV.md).

### Optional: rebuild from source (maintainers / advanced)

```bat
docker-rebuild.bat
```

Rare full / slow refresh (`build --pull`):

```bat
docker-refresh.bat
```

Publish new Hub tags (maintainers only):

```bat
docker-publish.bat
```

---

## 6. Git: clone and create your feature branch from `main`

### Clone (once)

```bash
git clone https://github.com/endevs/camera_Intelligence.git
cd camera_Intelligence
git checkout main
git pull origin main
```

After clone, the remote is named **`origin`** and points at `endevs/camera_Intelligence`.

### Create your feature branch (once per developer)

Pick **only your** branch name:

```bash
git checkout main
git pull origin main

# Zone Safety developer:
git checkout -b feature/zone-safety

# Vehicle developer:
# git checkout -b feature/vehicle

# Face developer:
# git checkout -b feature/face

git push -u origin HEAD
```

### Daily sync before coding

Always update from `main` so you do not drift:

```bash
git checkout main
git pull origin main
git checkout feature/<your-name>
git merge main
```

If the merge reports conflicts, see **section 8**.

---

## 7. Develop, commit, and merge back to `main`

1. Stay on **your** feature branch only.
2. Edit files under your owned folders (see section 1).
3. Commit often with clear messages:

```bash
git status
git add <files>
git commit -m "Describe why you changed this"
git push
```

4. Open a **Pull Request** on GitHub:  
   **base:** `main` ← **compare:** `feature/zone-safety` (or vehicle / face)  
   https://github.com/endevs/camera_Intelligence/compare

5. After review, merge the PR into `main` (squash or merge commit is fine).

6. After your (or a teammate’s) PR is merged, everyone should:

```bash
git checkout main
git pull origin main
git checkout feature/<your-name>
git merge main
git push
```

**Do not force-push `main`.** Do not push directly to `main` unless you are the maintainer and the team agrees.

---

## 8. How to resolve merge conflicts

Conflicts usually appear when you run `git merge main` or when GitHub PR cannot merge automatically.

### Steps

1. Git lists conflicted files. Open each file and search for:

```text
<<<<<<<
=======
>>>>>>>
```

2. Decide the correct final code:
   - Prefer a **union** when both sides add different features (e.g. keep both nav links, both API routes).
   - Do not delete another developer’s feature to “win” the conflict without talking to them.
   - Extra care in shared files: `main.py`, `static/shell/shell.css`, `static/shell/shell.js`, overview nav.

3. Remove all conflict markers so the file is valid.

4. Finish the merge:

```bash
git add <resolved-files>
git commit -m "Merge main into feature/<name>; resolve conflicts"
git push
```

5. If you are stuck and want to cancel the merge:

```bash
git merge --abort
```

Then ask the owner of the conflicting folder for help.

---

## 9. Quick verification checklist

- [ ] Docker Desktop is running (`docker version` works)
- [ ] `docker pull drpinfotech/camera-intelligence:develop` succeeded
- [ ] Compose stack is up; http://localhost:8000 shows Overview + top nav
- [ ] Cloned https://github.com/endevs/camera_Intelligence
- [ ] On the correct feature branch (`feature/zone-safety` / `vehicle` / `face`)
- [ ] Can open Zone Safety, Vehicle, and Face pages from the top nav
- [ ] Know how to sync with `main` and open a PR

---

## 10. Where to get help

- Architecture / local Python notes: [`GUIDE.md`](GUIDE.md)
- Redis / Compose env: [`.env.example`](.env.example)
- Maintainer: invite access issues (GitHub private repo or Docker Hub pull)

Welcome aboard — start from `main`, own your feature folder, sync often, merge via Pull Request.
