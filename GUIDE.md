# 📖 Camera Intelligence System — Setup & Developer Guide

A robust, real-time edge computer vision platform designed to run state-of-the-art AI pipelines on **RTSP IP Cameras (NVRs)**, local **USB Webcams**, and **Android smartphones**. 

This system leverages a highly concurrent FastAPI Python backend to perform YOLO-based object detection, semantic segmentation, ALPR (License Plate Recognition), and biometric vector-search tracking. It serves a responsive Web Dashboard for management and interfaces with a Flutter Mobile App for remote edge-camera streaming.

---

## 🏗️ 1. High-Level System Architecture

The architecture is designed to be highly modular, separating the video ingestion, AI inference loop, state management, and user interface into distinct layers.

```mermaid
graph TD
    %% Video Sources
    subgraph Video Ingestion Layer
        RTSP[RTSP NVR / IP Cameras]
        USB[Local USB Webcams]
        App[Flutter Android App via WebSocket]
        File[Uploaded Video Files]
    end

    %% Backend Server
    subgraph FastAPI Backend Server
        VS[Video Source Manager]
        Reg[Pipeline Registry]
        
        %% Pipelines
        subgraph AI Pipeline Engine
            YOLO[Ultralytics YOLOv8]
            Byte[Supervision ByteTrack]
            Face[InsightFace SCRFD + ArcFace]
            FastSAM[FastSAM Segmentation]
            OCR[EasyOCR / ALPR]
        end
        
        API[RESTful API Endpoints]
        WS[WebSocket Manager]
    end

    %% State Management
    subgraph Database Layer
        Redis[(Redis Database)]
        FAISS[(FAISS In-Memory Vector Index)]
    end

    %% User Interface
    subgraph Presentation Layer
        Dashboard[Web Dashboard HTML/JS]
    end

    %% Connections
    RTSP --> VS
    USB --> VS
    File --> VS
    App -->|ws://| WS
    WS --> VS

    VS --> Reg
    Reg --> AI Pipeline Engine
    AI Pipeline Engine --> API
    
    Face <--> FAISS
    AI Pipeline Engine <--> Redis
    API <--> Redis

    API -->|HTTP/MJPEG| Dashboard
```

---

## 🛠️ 2. Technology Stack

### Backend & Core Engine
- **Core Framework**: [FastAPI](https://fastapi.tiangolo.com/) with Uvicorn (ASGI) for highly concurrent, asynchronous HTTP and WebSocket serving.
- **Computer Vision**: [OpenCV](https://opencv.org/) (`cv2`) for video buffer manipulation, threading, and MJPEG stream encoding.
- **State & Caching**: **Redis** (`redis-py`) is the sole persistence layer for alerts, detections, and identities.

### AI & Machine Learning Models
- **General Object Detection**: [Ultralytics](https://ultralytics.com/) YOLOv8 (nano/small variants for edge efficiency).
- **Human Pose & Fall Detection**: YOLOv8-Pose for 17-keypoint human skeleton extraction.
- **Semantic Segmentation**: FastSAM (Fast Segment Anything) for zero-shot object bounding and pixel masking.
- **Face Recognition**: [InsightFace](https://github.com/deepinsight/insightface) (SCRFD for bounding box detection + ArcFace for 512-dim embedding extraction).
- **Tracking**: [Supervision](https://supervision.roboflow.com/) (`ByteTrack`) combined with internal Kalman Filters to prevent ID switching during occlusion.
- **Vector Search**: **FAISS** (`faiss-cpu`) for \( L_2 \) distance cosine similarity matching of face vectors in micro-seconds.
- **License Plate OCR**: Built-in edge Optical Character Recognition routines (ALPR).

### Frontend & Mobile
- **Web Dashboard**: Vanilla JavaScript (ES6 Modules), HTML5, and pure CSS. Zero-build-step design for maximum stability and hot-reloading speed.
- **Mobile Edge App**: **Flutter** (Dart). Broadcasts raw phone camera frames via WebSockets and renders returned AI JSON telemetry natively over the feed.

---

## 📁 3. Codebase Structure & Developer Workflow

The repository is organized by feature domains so independent teams (AI, Security, Web, Mobile) can work concurrently.

### 🧠 Core Engine (`core/`)
The foundational backend mechanics that keep the system running.
- `base_pipeline.py`: The abstract base class (`ABC`). **All** AI models must inherit from this and implement `initialize()` and `run_on_video()`.
- `registry.py`: Singleton registry that auto-discovers and registers enabled AI pipelines.
- `video_source.py`: Background thread manager designed to proactively clear OpenCV buffers, preventing lag in RTSP streams.
- `redis_client.py`: The global connection pool and helper methods for Redis integration.
- `mobile_ws.py`: Manages high-throughput WebSocket ingestion from the Flutter app.

### 🏭 Vision Pipelines (`pipelines/`)
The specific computer vision use-cases. Each pipeline is a distinct module.
- **`intrusion_pipeline.py`**: Detects humans crossing forbidden boundary lines (virtual tripwires).
- **`danger_zone_pipeline.py`**: Detects unauthorized human entry into dynamically drawn polygon areas.
- **`fall_detection_pipeline.py`**: Uses YOLO-Pose to map human skeletons, calculating fall angles based on bounding box aspect ratios and spine orientation vectors.
- **`room_guardian_pipeline.py`**: Tracks static objects (e.g., backpacks, laptops) using FastSAM and ByteTrack. Emits alerts if an object is moved, removed, or occluded.
- **`face_recognition_pipeline.py`**: Interacts with the `face_recognition/` package to translate raw frames into Known/Unknown identities, driving the Attendance and Visitor tracking systems.
- **`vehicle_recognition/`**: A specialized sub-package dedicated to ALPR (reading license plates), tracking unique vehicle visits, and identifying vehicle color and type.

### 👤 Face Recognition Engine (`face_recognition/`)
A dedicated, highly-accurate biometric sub-module.
- `embedder.py`: Handles raw facial extraction and embedding generation.
- `tracker.py`: Wraps ByteTrack to maintain temporal continuity of a face across frames.
- `recognizer.py`: Manages the sliding-window temporal consensus algorithm (requiring N/M votes to prevent false positives).
- `identity_manager.py`: Interfaces with Redis to persist JSON metadata, base64 image crops, and float32 arrays.

### 🌐 Web Presentation (`static/`)
Unified product shell with per-feature isolated folders.
- `static/shell/`: Shared top navigation, sidebar, and CSS design tokens.
- `static/features/zone-safety/`: UI control panels for Intrusion, Danger Zone, Fall Detection, and Room Guardian.
- `static/features/face/`: UI for Identity Management, Attendance logs, and Registration.
- `static/features/vehicle/`: UI for Vehicle monitoring and ALPR logs.

### 📱 Mobile Presentation (`edge_vision_app/`)
The Flutter application codebase.
- **To develop:** Open this folder in Android Studio or VS Code, run `flutter pub get`, and deploy to a physical Android device.

---

## ⚙️ 4. Local Setup & Execution Guide

### Prerequisites
- Windows 10/11 or Linux.
- **Python 3.10+**
- **Docker Desktop** (For Redis and Linux containerization).

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

### 🚀 Running the Application

There are **two distinct ways** to run the system depending on your hardware requirements.

#### Method A: Native Windows Execution (Required for USB Webcams)
**Use this method if you need to test with your laptop's built-in webcam or a USB camera.** Docker for Windows runs inside a Linux VM and physically cannot access Windows USB webcams.
1. Run the native startup script:
   ```powershell
   .\start-windows-app.bat
   ```
   *Note: This script automatically spins up a native Redis instance in the background and boots the FastAPI server.*
2. Open your browser to `http://127.0.0.1:8000`.

#### Method B: Docker Compose (For Linux Deployments & RTSP)
**Use this method if deploying to a Linux server, or if you are only testing via RTSP streams / uploaded MP4 files.**
1. Execute the Docker refresh script:
   ```powershell
   .\docker-refresh.bat
   ```
   *Note: This rebuilds the `camera-intelligence:develop` image and starts both the App and Redis containers via Docker Compose.*
2. Open your browser to `http://localhost:8000`.

---

## 📡 5. Mobile App Connectivity

To use an Android phone as a wireless edge camera:
1. Ensure your PC and the Android phone are on the **same Wi-Fi network**.
2. Find your PC's local IPv4 address (e.g., `192.168.1.50`) by running `ipconfig` in your terminal.
3. Build and install the Flutter APK from the `edge_vision_app/` folder onto your phone.
4. Open the app, enter your PC's IP address (`192.168.1.50`), and tap **Connect**. 
5. The video feed will instantly appear on the Web Dashboard under the "Mobile Stream" input source.

---

## 🗄️ 6. Redis Database Architecture

This project uses **Redis** as the sole, centralized persistence layer for the *entire* system. Local SQLite databases and JSON flat-files have been entirely deprecated. 

Data is logically separated using key prefixes to prevent collisions between modules:

### 🚨 Generic System Alerts (`alerts:`)
Generated by the Zone Safety pipelines (Intrusion, Fall Detection, Danger Zone, Room Guardian).
- `alerts:{session_id}` → Redis List (`RPUSH` / `LRANGE`) storing chronologically ordered JSON objects containing alert metadata, timestamps, and paths to saved `.webm` video clips.

### 🚗 Vehicle Recognition (`vr:`)
Generated by the ALPR and Vehicle Tracking module.
- `vr:det:{session_id}` → Redis List containing raw chronological vehicle detection events (License plate string, color, vehicle type).
- `vr:plates:{session_id}` → Redis Hash map storing aggregated statistics. Keys are license plate strings, values are total visit counts and last-seen timestamps.

### 👤 Face Recognition & Identity (`fr:`)
Generated by the InsightFace biometric engine.
- `fr:identities` → Hash map of `{person_id: json_metadata}` (Name, first seen, last seen, confidence scores).
- `fr:emb:{person_id}` → Raw `float32` numpy bytes containing the 512-dim ArcFace mathematical embedding.
- `fr:embmeta:{person_id}` → Metadata for numpy reconstruction (shape, dtype).
- `fr:face:{person_id}:{n}` → Raw JPEG bytes of cropped face images, retrieved by the UI for preview avatars.

### 🎓 Attendance Tracking (`attendance:`)
Generated by the specialized Classroom/Attendance operational mode.
- `attendance:sessions` → Hash map tracking currently active classroom sessions, mode states, and start/stop timestamps.
- `attendance:present:{session_id}` → Redis Set (`SADD`) containing the unique `person_id`s of students who have been biometrically verified in front of the camera during a given session window.
