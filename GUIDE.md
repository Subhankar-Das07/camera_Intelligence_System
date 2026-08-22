# 📖 Camera Intelligence System — Comprehensive Setup & Architecture Guide

A robust, real-time edge computer vision platform designed to run state-of-the-art AI pipelines on **RTSP IP Cameras (NVRs)**, local **USB Webcams**, and **Android smartphones**. 

This system leverages a highly concurrent FastAPI Python backend to perform YOLO-based object detection, semantic segmentation, and biometric vector-search tracking. It serves a responsive Web Dashboard for management and interfaces with a Flutter Mobile App for remote edge-camera streaming.

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

### Backend & AI Engine
- **Core Framework**: [FastAPI](https://fastapi.tiangolo.com/) with Uvicorn (ASGI) for highly concurrent, asynchronous HTTP and WebSocket serving.
- **Computer Vision**: [OpenCV](https://opencv.org/) (`cv2`) for frame manipulation and MJPEG stream encoding.
- **Object Detection & Pose**: [Ultralytics](https://ultralytics.com/) (YOLOv8, YOLO-Pose) and FastSAM.
- **Face Recognition**: [InsightFace](https://github.com/deepinsight/insightface) (SCRFD for face detection + ArcFace for 512-dim embedding extraction).
- **Object Tracking**: [Supervision](https://supervision.roboflow.com/) (`ByteTrack`) integrated with Kalman Filters.

### State Management & Databases
- **Primary Database**: **Redis** (`redis-py`). Used as the single source of truth for identity persistence, attendance tracking, and system state.
- **Vector Search**: **FAISS** (`faiss-cpu`). Utilized by the Face Recognition module for lightning-fast \( L_2 \) distance / cosine similarity matching against embedded vectors. Built dynamically in-memory from Redis on startup.

### Frontend & Mobile
- **Web Dashboard**: Vanilla JavaScript (ES6 Modules), HTML5, and pure CSS. Designed to be lightweight and fast without requiring Node.js build pipelines.
- **Mobile Edge App**: **Flutter** (Dart). Connects to the backend via WebSockets to stream mobile camera frames and receive JSON-formatted bounding boxes.

---

## 📁 3. Codebase Structure & Developer Workflow

The repository is organized by feature domains to allow independent teams (AI, Web, Mobile) to work concurrently without merge conflicts.

### 🧠 Core Engine (`core/`)
The foundational backend mechanics that keep the system running.
- `base_pipeline.py`: The abstract base class (`ABC`). **All** AI models must inherit from this and implement `initialize()` and `run_on_video()`.
- `registry.py`: Singleton registry that auto-discovers and instantiates enabled pipelines.
- `video_source.py`: Background thread manager designed to clear OpenCV buffers, ensuring lag-free RTSP streaming.
- `redis_client.py`: The global connection pool and helper methods for Redis integration.

### 🏭 AI Pipelines (`pipelines/`)
Specific computer vision use-cases that process frames and yield MJPEG bytes alongside alert metadata.
- `face_recognition_pipeline.py`: Translates raw frames into Known/Unknown bounding boxes, driving the Attendance and Visitor tracking systems.
- `room_guardian_pipeline.py`: Tracks static objects (e.g., bags, laptops) using FastSAM and ByteTrack. Emits alerts if an object is moved or removed.
- `intrusion_pipeline.py`: Detects humans crossing forbidden boundary lines.
- `danger_zone_pipeline.py`: Detects unauthorized entry into dynamically drawn polygon areas.
- `fall_detection_pipeline.py`: Uses YOLOv8-Pose to map human skeletons and calculate fall angles based on bounding box aspect ratios and spine vectors.
- `vehicle_recognition/`: Specialized sub-module for detecting vehicles, reading license plates (ALPR), and identifying vehicle color/type.

### 👤 Face Recognition Domain (`face_recognition/`)
A dedicated, highly-accurate sub-module built on InsightFace.
- `embedder.py`: Handles raw facial extraction and embedding generation.
- `tracker.py`: Wraps ByteTrack to maintain temporal continuity of a face across frames.
- `recognizer.py`: Manages the sliding-window temporal consensus algorithm (requiring N/M votes to prevent false positives) and executes FAISS searches.
- `identity_manager.py`: Directly interfaces with Redis to store/retrieve JSON metadata, base64 image crops, and float32 numpy arrays.

### 🌐 Web Presentation (`static/`)
Unified product shell with per-feature isolated folders.
- `static/shell/`: Shared top navigation, sidebar, and CSS design tokens.
- `static/features/zone-safety/`: UI for Intrusion, Danger Zone, Fall Detection, and Room Guardian.
- `static/features/face/`: UI for Identity Management, Attendance logs, and Face Registration.
- `static/features/vehicle/`: UI for Vehicle and License Plate tracking.

### 📱 Mobile Presentation (`edge_vision_app/`)
The Flutter application codebase.
- **To develop:** Open this folder in Android Studio or VS Code, run `flutter pub get`, and deploy to a physical Android device.

---

## ⚙️ 4. Local Setup & Execution Guide

### Prerequisites
- Windows 10/11 or Linux.
- **Python 3.10+**
- **Docker Desktop** (For Redis and Linux containerization).
- (Optional) NVIDIA GPU with CUDA for maximum YOLO inference speed.

### Installation
1. Clone the repository and checkout the main branch:
   ```bash
   git clone https://github.com/endevs/camera_Intelligence.git
   cd camera_Intelligence
   git checkout main
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
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

## 🗄️ 6. Redis Database Internals

Redis is the sole source of truth. All data is prefixed by domains to avoid collisions.

### Face Recognition Domain (`fr:`)
- `fr:identities` → Hash map of `{person_id: json_metadata}`.
- `fr:emb:{person_id}` → Raw `float32` numpy bytes containing the 512-dim ArcFace embedding.
- `fr:embmeta:{person_id}` → Shape and dtype parameters for numpy reconstruction.
- `fr:face:{person_id}:{n}` → Raw JPEG bytes of cropped face images for UI display.
- `fr:face_count:{person_id}` → Integer tracking the number of crops stored.

### Attendance Domain (`attendance:`)
- `attendance:sessions` → Hash map tracking active classroom sessions, mode, and timestamps.
- `attendance:present:{session_id}` → Redis Set (`SADD`) containing the `person_id`s of students who have been verified in front of the camera.
