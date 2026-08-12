# 📖 Edge Vision — Setup & Usage Guide

A real-time computer vision system that turns your **Android phone** into a smart camera. The phone streams live video over WiFi to a Python server running on your PC, which performs **YOLO object detection + tracking** and displays results in real time.

---

## 🏗️ Architecture

```
┌──────────────┐   JPEG frames (WebSocket)   ┌──────────────────────┐
│  Android App │ ──────────────────────────► │  Python Server (PC)  │
│  (Camera)    │                              │  YOLO11 + ByteTrack  │
└──────────────┘                              │  cv2.imshow (Dev)    │
                                              └──────────┬───────────┘
┌──────────────┐   JSON detection reports                │
│  Monitor Tab │ ◄───────────────────────────────────────┘
│  (App/Web)   │
└──────────────┘
```

---

## ⚙️ Prerequisites

| Requirement | Details |
|---|---|
| **PC (Server)** | Windows 10/11 with Python 3.10+ |
| **Android Phone** | Android 9+ (API 28+) |
| **Network** | Both devices on the **same WiFi network** |
| **GPU (optional)** | NVIDIA GPU speeds up YOLO inference, but CPU works fine |

---

## 🖥️ Part 1 — Server Setup (Your PC)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Subhankar-Das07/camera_Intelligence_System.git
cd camera_Intelligence_System
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements_server.txt
```

### Step 3: Find Your PC's Local IP

```bash
# Windows
ipconfig
# Look for "IPv4 Address" under your WiFi adapter (e.g., 192.168.1.37)
```

### Step 4: Start the Server

```bash
python server.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
[Worker] Inference worker started.
[Server] Startup complete. Inference worker running.
```

A window titled **"Developer View — Smart Vision"** will appear on your PC — this is where you'll see the live annotated camera feed with bounding boxes.

> **Firewall Note:** If your phone can't connect, make sure Windows Firewall allows inbound connections on **port 8000**. You can temporarily disable the firewall for your private network or add a rule.

---

## 📱 Part 2 — Getting the APK on Your Phone

### Option A: Build from Source (Recommended)

**Prerequisites:** [Flutter SDK](https://docs.flutter.dev/get-started/install) installed on your PC.

```bash
cd edge_vision_app

# Install dependencies
flutter pub get

# Build the release APK (smaller size, ~16 MB)
flutter build apk --release --split-per-abi --split-debug-info=build/symbols --obfuscate

# Output APKs will be at:
#   build/app/outputs/flutter-apk/app-arm64-v8a-release.apk    ← Most phones
#   build/app/outputs/flutter-apk/app-armeabi-v7a-release.apk  ← Older phones
```

### Option B: Transfer Pre-built APK via USB / File Sharing

If someone has already built the APK, transfer `app-arm64-v8a-release.apk` to your phone via USB, email, Google Drive, etc.

### Installing the APK

1. **Transfer** the `.apk` file to your Android phone
2. **Open** the file on your phone (use a file manager if needed)
3. **Allow** "Install from unknown sources" when prompted
4. **Tap Install** and wait for it to complete

> **Which APK do I use?**
> - `app-arm64-v8a-release.apk` — **Use this.** Works on all phones made after ~2017.
> - `app-armeabi-v7a-release.apk` — For very old 32-bit phones.
> - `app-x86_64-release.apk` — For Android emulators only.

---

## 🚀 Part 3 — Running a Detection Test

### Step 1: Start the Server on Your PC

```bash
cd camera_Intelligence_System
python server.py
```

### Step 2: Open the App on Your Phone

1. Launch **Edge Vision** on your phone
2. Enter your **PC's local IP address** (e.g., `192.168.1.37`) in the server IP field
3. Choose a mode:

### 📷 Capture Mode
- Tap **Capture** to start streaming your phone's camera to the server
- The status bar will show **"Connected to Server"** when the WebSocket handshake completes
- You'll see the **frames sent** counter incrementing
- On your **PC screen**, the "Developer View" window will show the live camera feed with **coloured bounding boxes** drawn around detected objects

### 📊 Monitor Mode
- Tap **Monitor** to see **live detection results** as JSON data
- Shows: detected objects, their labels, confidence scores, tracking IDs, duration in scene, and scene stability

### What to Point the Camera At

Try pointing your camera at:
- **People** — detected as "person"
- **Phones, laptops, keyboards** — detected as "electronics"
- **Cups, bottles** — detected as "kitchenware"
- **Chairs, tables** — detected as "furniture"

The system uses **YOLO11n** and tracks objects across frames with **ByteTrack**, so you'll see consistent tracking IDs even as objects move.

---

## 📁 Project Structure

```
camera_Intelligence_System/
├── server.py                 # FastAPI WebSocket server (main entry point)
├── engine.py                 # CV inference bridge (YOLO + tracking + reports)
├── yolo11n.pt                # YOLO11 nano model weights
├── requirements_server.txt   # Python dependencies
├── templates/
│   └── index.html            # PWA web interface (served at /)
├── static/                   # PWA assets (manifest, icons, service worker)
├── smart_vision_assistant/   # Core CV modules (detector, scene memory, etc.)
├── edge_vision_app/          # Flutter mobile app source code
│   ├── lib/
│   │   ├── main.dart         # App entry point + home screen
│   │   ├── capture_screen.dart   # Camera streaming to server
│   │   └── monitor_screen.dart   # Live detection results viewer
│   └── pubspec.yaml          # Flutter dependencies
├── GUIDE.md                  # ← This file
└── .gitignore
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| App shows "Connecting..." forever | Check PC IP is correct, server is running, both on same WiFi |
| Server starts but no frames arrive | Check Windows Firewall allows port 8000 |
| "Developer View" window doesn't appear | Run `python server.py` directly in a terminal (not as a background process) |
| App crashes on launch | Ensure you installed the correct APK for your phone's architecture (arm64 for modern phones) |
| Low FPS / laggy detection | Close other apps on PC; a GPU significantly improves performance |

---

## ⚠️ Notes

- This is a **pre-release version (v1)**. Not all features are final.
- The server must be running **before** you tap Capture/Monitor in the app.
- The `cv2.imshow` developer window only appears on the PC running the server — it is never streamed to clients.
- Gemini AI verification is **disabled** in this PoC build.
