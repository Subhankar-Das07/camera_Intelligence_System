# 📖 Camera Intelligence System — Setup & Developer Guide

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
- **`main.py`**: The entry point. Runs the FastAPI server, manages RTSP connections, and exposes APIs for the web dashboard.
- **`core/`**: Contains the engine logic.
  - `base_pipeline.py`: The abstract class all AI models must inherit from.
  - `registry.py`: Auto-discovers and registers pipelines.
  - `mobile_ws.py`: Handles WebSocket connections from the Flutter app.
  - `video_source.py`: Background thread manager for lag-free RTSP streaming.
- **`pipelines/`**:
  - `face_recognition_pipeline.py`: Dedicated face recognition module. Yields frames with Known/Unknown bounding boxes.
  - `room_guardian_pipeline.py`: Advanced object protection tracking using ByteTrack identity verification.
  - `intrusion_pipeline.py`, `danger_zone_pipeline.py`, `fall_detection_pipeline.py`, `vehicle_recognition/`.
  - *To add a new AI capability:* Inherit from `BaseVideoPipeline`, implement `initialize()` and `run_on_video()`.
- **`face_recognition/` (Ayush Module)**: 
  - A highly accurate module using InsightFace (SCRFD + ArcFace) and FAISS for vector search.
  - Uses `supervision.ByteTrack` for stable identity tracking.
  - All identities and faces are persistently stored in the centralized **Redis** database.

### 2. UI / UX Web Developers (`static/`)
Unified product shell with per-feature folders (branch-friendly):
- **`static/shell/`** — shared top nav and design tokens
- **`static/features/zone-safety/`** — intrusion / danger zone / room guardian UI
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
- Redis Database
- (Optional) Docker Desktop (for Linux deployment)

### Team main repository (endevs)
**Source of truth:** https://github.com/endevs/camera_Intelligence  
**Base branch for all feature work:** `main`  

Clone and start a feature branch:
```bash
git clone https://github.com/endevs/camera_Intelligence.git
cd camera_Intelligence
git checkout main
git pull
git checkout -b feature/<name>
```

---

## 🚀 How to Run the Project

There are **two** distinct ways to run this project depending on what hardware you need to access.

### Method 1: Native Windows Execution (Required for USB Webcams)
**Use this method if you want to use your local PC's built-in USB webcam.** Docker on Windows runs inside a Linux virtual machine and physically cannot access Windows USB webcams.
1. Run the local startup script:
   ```powershell
   .\start-windows-app.bat
   ```
2. The script will automatically launch a native Windows Redis server in the background and start the FastAPI application on port `8000`.
3. Open your browser to `http://127.0.0.1:8000`.

### Method 2: Docker Execution (For Linux / RTSP streaming)
**Use this method if you are deploying to a Linux server or only testing with RTSP IP cameras and uploaded video files.**
1. Rebuild and launch the Docker containers:
   ```powershell
   .\docker-refresh.bat
   ```
2. The Docker engine will pull the `redis:7-alpine` database and build the `camera-intelligence:develop` python application.
3. Open your browser to `http://localhost:8000`.

---

## 💾 Database Architecture (Redis)
This project has been fully migrated to use **Redis** as the centralized canonical database for all features. Local SQLite files (`vehicle_intelligence.db`) and local `.json` / `.npz` storage files have been deprecated to support seamless, persistent, multi-container deployments. 

Ensure Redis is running (either natively or via Docker) on port `6379` before launching the FastAPI application.
