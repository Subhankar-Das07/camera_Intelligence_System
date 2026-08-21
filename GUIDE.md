# 📖 Camera Intelligence System — Setup & Developer Guide

**New teammates:** start here → [`TEAM_ONBOARDING.md`](TEAM_ONBOARDING.md) (Docker install, Hub pull, Git feature branches, merge & conflicts).

A robust, real-time computer vision platform designed to run AI pipelines on both **RTSP IP Cameras (NVRs)** and **Android smartphones**. The system uses a FastAPI Python backend to perform YOLO object detection and tracking, serving a responsive Web Dashboard for management and a Flutter Mobile App for edge camera streaming.

---

## 🏗️ System Architecture

```text
┌─────────────────┐       ┌────────────────────────────────┐
│  RTSP NVR /     ├──────►│      FastAPI Server (main.py)  │
│  IP Cameras     │       │                                │
└─────────────────┘       │   ┌────────────────────────┐   │      ┌─────────────────┐
                          │   │      Pipeline Engine   │   ├─────►│ Web Dashboard   │
┌─────────────────┐       │   │  (ultralytics YOLOv8)  │   │      │ (Admin UI)      │
│  Android App    ├──────►│   └────────────────────────┘   │      └─────────────────┘
│  (Camera/Mic)   │       │                                │
└─────────────────┘       └────────────────────────────────┘
```

---

## 📁 Project Structure (For Developers)

The codebase is highly modularized so different teams (AI, Web, Mobile) can work independently without stepping on each other's toes.

### 1. AI & Backend Developers (`core/`, `pipelines/`, `face_recognition/`)
- **`main.py`**: The entry point. Runs the FastAPI server, manages RTSP connections, and exposes APIs for the web dashboard (including Face Recognition endpoints).
- **`core/`**: Contains the engine logic.
  - `base_pipeline.py`: The abstract class all AI models must inherit from.
  - `registry.py`: Auto-discovers and registers pipelines.
  - `mobile_ws.py`: Handles WebSocket connections from the Flutter app.
  - `video_source.py`: Background thread manager for lag-free RTSP streaming.
- **`pipelines/`**:
  - `face_recognition_pipeline.py`: Integrates the dedicated face recognition module into the pipeline architecture. Yields frames with Known/Unknown bounding boxes.
  - *To add a new AI capability (e.g., Fire Detection):* Inherit from `BaseVideoPipeline`, implement `initialize()` and `run_on_video()`.
- **`face_recognition/` (Ayush Module)**: 
  - A highly accurate module using InsightFace (SCRFD + ArcFace) and FAISS for vector search.
  - Features a unified database (`data/persons/`) where all unique faces are auto-saved on their first visit.
  - Uses `supervision.ByteTrack` for stable identity tracking and temporal consensus for high-confidence matching.

### 2. UI / UX Web Developers (`static/`)
Unified product shell with per-feature folders (branch-friendly):
- **`static/shell/`** — shared top nav and design tokens
- **`static/features/zone-safety/`** — intrusion / danger zone / fall UI
- **`static/features/vehicle/`** — vehicle recognition UI
- **`static/features/face/`** — face recognition UI
- **`static/shared/`** — shared upload/RTSP helpers

### 3. Android / Flutter Developers (`edge_vision_app/`)
This folder contains the mobile application that turns an Android phone into an edge-streaming camera.
- It connects to the server via WebSockets (`ws://<SERVER_IP>:8000/ws/capture`).
- It streams JPEG frames and receives JSON detection data back to display on the screen.
- **To develop:** Open `edge_vision_app/` in Android Studio or VS Code and run `flutter pub get`.

---

## ⚙️ Getting Started (Server Setup)

### Prerequisites
- Windows 10/11 or Linux with **Python 3.10+**
- Docker Desktop (recommended for team deploy)
- (Optional) NVIDIA GPU for faster YOLO inference.

### Team main repository (endevs)
**Source of truth:** https://github.com/endevs/camera_Intelligence  
**Base branch for all feature work:** `main`  
https://github.com/endevs/camera_Intelligence/tree/main

Clone and start a feature branch:

```bash
git clone https://github.com/endevs/camera_Intelligence.git
cd camera_Intelligence
git checkout main
git pull
git checkout -b feature/<name>
```

| Area | Branch |
|------|--------|
| Zone Safety | `feature/zone-safety` |
| Vehicle | `feature/vehicle` |
| Face | `feature/face` |

Primary folders: `static/features/zone-safety/`, `static/features/vehicle/`, `static/features/face/`

### Docker Hub image (no local build)
Image: **`drpinfotech/camera-intelligence:develop`** (also tagged `0.1.0`)  
https://hub.docker.com/r/drpinfotech/camera-intelligence

```bat
docker pull drpinfotech/camera-intelligence:develop
docker compose -f docker-compose.yml -f docker-compose.hub.yml up -d --no-build
```
Open http://localhost:8000

Local rebuild from source: run `docker-refresh.bat`  
Publish new Hub tags (maintainers): run `docker-publish.bat`

### 1. Install Dependencies (local Python)
```bash
pip install -r requirements.txt
```

### 2. Run the Server (local Python)
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- The server will start and automatically download the YOLO weights on the first run.
- Open your browser to `http://localhost:8000` to view the **Admin Web Dashboard**.

### 3. Connecting the Mobile App
1. Find your PC's IP address (e.g., `192.168.1.50`).
2. Build and install the Flutter APK from `edge_vision_app/` onto your Android device.
3. Open the app, enter the PC's IP address, and hit connect. The app will immediately begin streaming to the `core/mobile_ws.py` engine.
