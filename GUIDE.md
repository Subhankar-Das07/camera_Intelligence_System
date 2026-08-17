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

### 1. AI & Backend Developers (`core/` and `pipelines/`)
- **`main.py`**: The entry point. Runs the FastAPI server, manages RTSP connections, and exposes APIs for the web dashboard.
- **`core/`**: Contains the engine logic.
  - `base_pipeline.py`: The abstract class all AI models must inherit from.
  - `registry.py`: Auto-discovers and registers pipelines.
  - `mobile_ws.py`: Handles WebSocket connections from the Flutter app.
  - `video_source.py`: Background thread manager for lag-free RTSP streaming.
- **`pipelines/`**: **Add new AI models here!**
  - To add a new AI capability (e.g., Face Detection, Fire Detection):
    1. Create a new file (e.g., `fire_pipeline.py`).
    2. Inherit from `BaseVideoPipeline`.
    3. Implement `initialize()` to load your PyTorch/YOLO model.
    4. Implement `run_on_video()` to yield frames and JSON alerts.
    5. The server will automatically discover it and add it to the Web Dashboard dropdown!

### 2. UI / UX Web Developers (`static/`)
This folder contains the Admin Web Dashboard that manages cameras and views live RTSP streams.
- **`index.html`**: The main layout and DOM structure.
- **`style.css`**: All styling. Uses a modern, dark-mode, flexbox-driven design.
- **`app.js`**: Handles API calls to `main.py`, manages the video player, and calculates the exact pixel mapping for the Region of Interest (ROI) drawing canvas.

### 3. Android / Flutter Developers (`edge_vision_app/`)
This folder contains the mobile application that turns an Android phone into an edge-streaming camera.
- It connects to the server via WebSockets (`ws://<SERVER_IP>:8000/ws/capture`).
- It streams JPEG frames and receives JSON detection data back to display on the screen.
- **To develop:** Open `edge_vision_app/` in Android Studio or VS Code and run `flutter pub get`.

---

## ⚙️ Getting Started (Server Setup)

### Prerequisites
- Windows 10/11 or Linux with **Python 3.10+**
- (Optional) NVIDIA GPU for faster YOLO inference.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- The server will start and automatically download the YOLO weights on the first run.
- Open your browser to `http://localhost:8000` to view the **Admin Web Dashboard**.

### 3. Connecting the Mobile App
1. Find your PC's IP address (e.g., `192.168.1.50`).
2. Build and install the Flutter APK from `edge_vision_app/` onto your Android device.
3. Open the app, enter the PC's IP address, and hit connect. The app will immediately begin streaming to the `core/mobile_ws.py` engine.
