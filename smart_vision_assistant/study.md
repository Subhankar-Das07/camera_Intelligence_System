# Smart Vision Assistant: Comprehensive Study Guide

Welcome to the ultimate masterclass for the **Smart Vision Assistant**. This document is designed to teach you everything about this project, from its high-level architecture down to the intricate details of its multi-threaded data flow, intelligent engines, and persistent memory.

Whether you are studying computer vision pipelines, real-time system optimizations, or AI integration, this guide covers every component you need to know.

---

## 1. Introduction: What is the Smart Vision Assistant?

The **Smart Vision Assistant** is a real-time, high-performance Visual Scene Intelligence Engine. It captures video from a webcam, detects and tracks objects, reasons about their spatial relationships, infers their behavioral states, and persists this knowledge into a database. 

It is designed to run efficiently on standard CPUs (without requiring heavy GPUs) by utilizing adaptive frame skipping, multi-threading, and non-blocking asynchronous architectures. While it is highly optimized for offline execution using local models (YOLO), it selectively uses cloud AI (Google Gemini) in the background to refine ambiguous objects without impacting the real-time frame rate.

---

## 2. The Big Picture: System Architecture & Data Flow

To maintain a smooth 15–30 FPS while simultaneously handling heavy tasks like database writes, spatial mathematics, and API calls, the system is strictly separated into multiple concurrent threads.

### 2.1 Architectural Blueprint

```mermaid
graph TD
    A[Webcam Feed] -->|Raw Frames| B(CameraThread: Lock-Protected Grab)
    B -->|Double-Buffer Swap| C(Main Vision Loop / Headless Loop)
    C -->|Frame Skip Check| D(ObjectDetector: YOLO11 + BoT-SORT)
    D -->|Bounding Boxes & IDs| E(SceneMemory / EntityRegistry)
    
    %% Intelligent Engines
    E -->|Active Records| F(EventEngine: Stationary/Moved)
    E -->|Active Records| G(RelationshipEngine: Near/Inside)
    
    %% Background Tasks
    E -->|New Object/Age Check| H{Gemini Call Needed?}
    H -->|Yes (Async)| I(GeminiVerifier Queue)
    H -->|No| J(DatabaseManager Queue)
    I -->|Async API Response| J
    J -->|Background Writer Thread| K[(SQLite Database: smart_vision.db)]
    
    %% Outputs
    C -->|Periodic Tick| L(ReportEngine: Console/JSON)
    C -->|MJPEG Encoding| M(FastAPI / Web Interface)
```

### 2.2 Technology Stack
*   **Computer Vision:** OpenCV (`cv2`) for frame capture and rendering, Ultralytics YOLO11 for object detection, BoT-SORT for object tracking.
*   **Concurrency:** Python `threading` (Locks, Queues, Events) for non-blocking IO.
*   **Database:** SQLite3 running in Write-Ahead Logging (WAL) mode for fast concurrent access.
*   **AI Integration:** Google GenAI SDK (Gemini 2.0 Flash Lite) for advanced vision-language processing.
*   **Web/API (Optional Headless Mode):** FastAPI for REST/WebSockets, React/TailwindCSS for the frontend dashboard.
*   **Analytics:** `psutil` for hardware telemetry (CPU, RAM monitoring).

---

## 3. Deep Dive: Core Subsystems and Engines

Let's dissect the project file-by-file and engine-by-engine.

### 3.1 `main.py` & `smart_vision_headless.py` (The Orchestrators)
These files contain the central loop. Depending on whether you run the local OpenCV window (`main.py`) or the web server (`smart_vision_headless.py`), the loop operates similarly:
1.  **Frame Capture:** Pulls the latest frame from the `CameraThread`.
2.  **Adaptive Frame Skipping:** Monitors current FPS. If FPS drops below a threshold (e.g., due to CPU load), it skips YOLO inference on some frames and reuses previous bounding boxes.
3.  **Engine Dispatch:** Passes detections to the `SceneMemory`, then triggers the `EventEngine`, `RelationshipEngine`, and `GeminiVerifier`.
4.  **Rendering/Streaming:** Draws bounding boxes for local display or encodes the frame to JPEG for web streaming.

**Key Concept: The CameraThread Double-Buffer**
To prevent the camera capture from blocking the inference loop, `CameraThread` continuously grabs frames in a background thread. It uses a double-buffer approach protected by a `threading.Lock()` where it writes to a hidden buffer, and the main thread only swaps pointers during a read. This avoids costly memory copying (memcpy) under a lock.

### 3.2 `detector.py` (Object Detection & Tracking)
Wraps the YOLO11 model and the BoT-SORT tracker. 
*   **YOLO (You Only Look Once):** Finds objects in the frame (e.g., a cup, a person) and draws a bounding box.
*   **BoT-SORT:** Assigns a persistent numeric ID (`track_id`) to the object so the system knows that "Cup #1" in frame 1 is the same as "Cup #1" in frame 2.

### 3.3 `scene_memory.py` & `entity_registry.py` (Persistent Identity)
Standard trackers are "amnesiac"—if an object leaves the frame and returns, it gets a brand new ID. 
The Scene Memory / Entity Registry solves this:
*   **Stable Identity:** Wraps tracker IDs into permanent UUIDs. 
*   **Relinking:** If an object disappears and reappears nearby within a short time (e.g., 30 seconds) with the same class, the registry maps the new track ID back to the old UUID.
*   **Confidence Gating:** Noise is filtered out. An object must have >= 80% confidence and be visible for >= 2 seconds to be officially written to memory.
*   **ACTIVE vs INACTIVE:** Objects currently in view are ACTIVE. Objects that leave the view become INACTIVE but are retained in memory with their last known state, reducing CPU load to zero while maintaining historical context.

### 3.4 `event_engine.py` (Behavioral Intelligence)
Analyzes the state of objects over time to determine behavior.
*   It monitors bounding box movement.
*   If an object stays in the same place for X seconds, it fires a `Stationary` event.
*   If it is moved, it fires a `Moved` event.
*   If left unattended for a long time, it can fire an `Abandoned` event.

### 3.5 `spatial_engine.py` & `relationship_engine.py` (Geometric Reasoning)
Calculates relationships between objects using pure mathematics (bounding box overlaps and distances) rather than expensive neural networks.
*   Detects if an object is `inside` another.
*   Detects if two objects are `near` each other (proximity).
*   Detects if an object is `on_top_of` another.

### 3.6 `gemini_verifier.py` (Asynchronous AI Refinement)
YOLO is fast but generic (it sees a "dog"). We want specific details ("Golden Retriever"). 
*   **Budgeting:** AI calls cost money and time. The verifier uses a strict budget and only queries eligible, stationary objects.
*   **Non-Blocking:** Frame crops are sent to a background `Queue`. A daemon thread calls the Gemini API and processes the response asynchronously.
*   **Callbacks:** Once Gemini returns a detailed 3-bullet description or a refined label, it triggers a callback to update the `SceneMemory` and the database seamlessly.

### 3.7 `report_engine.py` & `timeline_engine.py` (Analytics)
*   **Report Engine:** Periodically (e.g., every 30 seconds) freezes the current state of the scene and prints a structured JSON/Console report detailing active objects, FPS stability, and memory usage.
*   **Timeline Engine:** Reconstructs the chronological history of an object from birth (`new`) to death (`removed`), including all verify and movement events.

### 3.8 `ocr_engine.py` & `product_knowledge.py` (Text & Brand Recognition)
*   Used for reading text off objects (using PaddleOCR/EasyOCR on cropped bounding boxes).
*   The product knowledge base maps extracted raw text (e.g., "bisleri") to semantic product types (e.g., "Water Bottle").

---

## 4. Database Architecture & Persistence

A critical evolution of this system is its persistence layer (`database_manager.py`). 

### 4.1 Producer-Consumer Threading Model
Writing to a disk database directly inside the vision loop causes micro-stutters and ruins the frame rate.
*   **Producer:** The main vision loop pushes SQL operations (like logging a new object or an event) into a thread-safe `queue.Queue`.
*   **Consumer:** A single dedicated background thread (`DBWriter`) constantly reads from this queue and executes the SQL commands.
*   **WAL Mode:** The SQLite database is set to Write-Ahead Logging, making concurrent reads and writes extremely fast.

### 4.2 Relational 5-Table Schema
The database (`smart_vision.db`) uses a strictly normalized structure:
1.  **`sessions`**: Tracks start/end times and total statistics for every time you run the app.
2.  **`reports`**: Periodic telemetry logs (FPS, CPU usage, RAM).
3.  **`report_objects`**: The exact state of every object at the exact moment a report was generated.
4.  **`object_events`**: A ledger of state transitions (`new`, `removed`, `verified`, `moved`).
5.  **`tracked_objects`**: The ultimate long-term memory. Aggregates lifetime stats, total duration, highest confidence, and Gemini descriptions for every unique entity ever seen.

---

## 5. Web Dashboard Integration (FastAPI & React)

When running the system headless (`smart_vision_headless.py` + `web_server.py`), the UI shifts to the browser:

### 5.1 Real-Time Video Streaming (MJPEG)
Sending pure video frames over WebSockets is inefficient. Instead, the backend encodes annotated frames into JPEG bytes using OpenCV (`cv2.imencode`), and streams them via an HTTP multipart response (`Content-Type: multipart/x-mixed-replace; boundary=frame`). The React `<img src="/video/stream" />` tag decodes this natively at high speed.

### 5.2 Telemetry via WebSockets
Every 1 second, a JSON payload containing active objects, inactive objects, relationships, and system telemetry is pushed over a native WebSocket (`/ws/live`) to the React frontend. This populates the UI cards, timelines, and status monitors without tying up the video feed.

---

## 6. Summary for Students

To truly master this architecture, pay attention to these three core philosophies:
1.  **Never Block the Main Thread:** Camera capture, Database writes, and LLM API calls are *always* pushed to background threads. The main loop must cycle as fast as possible to maintain FPS.
2.  **Memory over Transience:** Computer vision usually forgets an object the moment it leaves the frame. By introducing UUIDs, Relinking, and an INACTIVE state, the system builds a persistent "understanding" of the room.
3.  **Tiered Intelligence:** Math is cheaper than AI. Use YOLO for rapid bounds, use pure geometry (Relationship Engine) for spatial reasoning, and only spend expensive API calls (Gemini) when an object is stationary, ambiguous, and within budget.

---

## 7. Step-by-Step Execution Flow: From Startup to the Infinite Loop

This section details exactly how the project boots up, which files activate in sequence, and how the program loops endlessly to provide real-time AI.

### Phase 1: Bootstrapping (The Startup Sequence)
1. **The User Trigger:** The user double-clicks `start_web.bat`.
2. **Spawning Servers:** The batch script opens two command windows:
   - Window 1 runs `npm run dev` in the `frontend/` folder, spinning up the Vite/React server on port 5173.
   - Window 2 runs `uvicorn web_server:app --port 8000`, spinning up the FastAPI Python backend.
3. **API Initialization (`web_server.py`):** 
   - FastAPI executes its `@asynccontextmanager` called `lifespan`. 
   - This explicitly instantiates the `SmartVisionHeadless()` class, which is the heart of the backend.
4. **Engine Instantiation (`smart_vision_headless.py`):** 
   - Inside the constructor of `SmartVisionHeadless`, all the "brains" are born. It creates the `DatabaseManager`, `ObjectDetector`, `SceneMemory`, `EventEngine`, `RelationshipEngine`, and `GeminiVerifier`.
   - The `DatabaseManager` spawns its own background thread instantly to prepare for SQL writes.
5. **Thread Ignition:** 
   - `SmartVisionHeadless` starts a `CameraThread`, which connects to the webcam and begins silently buffering frames in the background.
   - It then starts the `_vision_thread`, which is the infinite loop where the real AI work happens.

### Phase 2: The Infinite Vision Loop (The Heartbeat)
Inside the `_vision_thread` of `smart_vision_headless.py`, the following sequence repeats 15-30 times every second:

1. **Grab Frame:** The loop asks the `CameraThread` for the latest frame. This happens instantly via a double-buffer pointer swap (no lag).
2. **FPS & Skip Check:** It calculates the current FPS. If the CPU is choking and FPS drops below the threshold defined in `config.py`, it will skip detection on this frame and reuse the last known bounding boxes.
3. **YOLO Detection (`detector.py`):** If not skipped, the frame is passed into the Ultralytics YOLO model. YOLO returns raw bounding boxes, and BoT-SORT assigns numeric tracking IDs to them.
4. **Memory Update (`scene_memory.py`):** The raw detections are pushed into memory. The memory registers new objects, handles UUID relinking, marks missing objects as INACTIVE, and immediately queues SQL commands to the `DatabaseManager` to log these events.
5. **Event Evaluation (`event_engine.py`):** Active objects are scanned to see if they have remained stationary or have been moved. New events are fired and logged to the DB.
6. **Relationship Evaluation (`relationship_engine.py`):** The system does pure geometric math to check if any active objects are near or inside each other.
7. **Gemini Eligibility (`gemini_verifier.py`):** The system checks if any stationary objects need a refined label or description. If so, a cropped image of the object is pushed into a queue for a background thread to send to Google.
8. **Periodic Reporting (`report_engine.py`):** Checks if it's time (e.g. every 30s) to compile a full JSON summary of the scene.
9. **Encoding & State Cache:** The loop draws boxes on the frame and encodes it to a JPEG byte array. It then locks a shared dictionary `_cached_state` and updates it with the latest JSON data and JPEG image.
10. **Loop repeats.**

### Phase 3: Web Delivery (The Consumer)
While Phase 2 loops infinitely, the web server acts independently:
- **Streaming Video:** When the React UI connects to `GET /video/stream`, FastAPI runs a loop that constantly grabs the `_latest_jpeg` from the shared cache and sends it over HTTP as a multipart MJPEG stream.
- **Streaming Data:** When React connects to `WS /ws/live`, FastAPI runs a loop that grabs the `_cached_state` JSON and pushes it through the WebSocket every 1 second.
- **Frontend Reaction:** React receives the JSON, updates its hooks, and re-renders the DOM to show the latest cards, timelines, and hardware telemetry to the user.

---

## 8. Detailed File-by-File Dictionary (Every .py File Explained)

Here is a comprehensive dictionary of every Python file in this project, explaining exactly what it is responsible for.

*   **`check_db.py`**: A developer utility script. Run this manually in the terminal to inspect the contents of the `smart_vision.db` SQLite database (views sessions, event counts, etc.).
*   **`config.py`**: The central brain for settings. It holds all global constants, thresholds (like confidence scores, FPS limits), file paths, and UI colors. Editing this file changes how the entire system behaves.
*   **`database_manager.py`**: The persistent storage engine. It uses a Producer-Consumer pattern where a background thread (`DBWriter`) exclusively handles disk writes asynchronously, preventing SQLite from blocking the fast vision loop.
*   **`detector.py`**: The AI vision wrapper. It initializes the YOLO11 model and the BoT-SORT tracker. Its `detect()` function takes a raw frame and returns structured bounding boxes and tracking IDs.
*   **`entity_inference.py`**: Contains helper logic used to make semantic, high-level inferences about entities based on their class or accumulated data.
*   **`entity_registry.py`**: The core of object identity. It converts amnesiac YOLO track IDs into stable UUIDs, handles relinking (when an object briefly disappears), and manages the ACTIVE/INACTIVE state transition.
*   **`event_engine.py`**: The behavioral rules engine. It tracks the physical movement of bounding boxes over time to fire logical events like `Stationary`, `Moved`, or `Abandoned`.
*   **`gemini_test.py` & `gemini_test_models.py`**: Testing utilities used during development to verify that the Google Gemini API key is valid and to list available LLM models.
*   **`gemini_verifier.py`**: The asynchronous AI refining engine. It runs a background daemon thread that waits for image crops in a queue, sends them to the Gemini API, and processes the returned 3-bullet descriptions and refined labels without stalling the webcam.
*   **`knowledge_graph.py`**: An experimental structure for mapping objects and their semantic relationships into a formal graph structure for advanced reasoning.
*   **`main.py`**: The local Desktop entry point. If a user doesn't want the web interface, running this file boots the vision loop and uses `cv2.imshow()` to draw a local window directly on the desktop.
*   **`ocr_engine.py`**: The text extraction engine. Wraps libraries like PaddleOCR or EasyOCR to read text off image crops.
*   **`ocr_processor.py`**: Orchestrates sending specific bounding box crops (like bottles or boxes) to the `ocr_engine.py` to hunt for text.
*   **`ocr_text_memory.py`**: A caching mechanism that remembers text that has already been read from specific objects, preventing the system from re-running expensive OCR on the same object frame after frame.
*   **`product_knowledge.py`**: A localized mapping dictionary. It takes raw text found by OCR (e.g. "bisleri") and maps it to semantic categories (e.g. "Water Bottle").
*   **`query_interpreter.py`**: Translates natural language queries (e.g. "When did the cup arrive?") into structured SQL queries to search the database.
*   **`relationship_engine.py`**: The spatial reasoning engine. It takes active bounding boxes and uses pure math to determine proximity states (`inside`, `near`, `on_top_of`).
*   **`report_engine.py`**: Periodically aggregates the current state of all active objects, system telemetry (FPS, CPU), and creates a unified report snapshot to print to the console or save to the DB.
*   **`scene_memory.py`**: A high-level orchestration wrapper that sits on top of the `entity_registry.py`. It is the primary API the vision loop calls to update tracking states.
*   **`search_engine.py`**: The interface layer for interacting with the database to find historical objects or events.
*   **`smart_vision_headless.py`**: The Web Server's version of `main.py`. It runs the exact same vision loop, but instead of rendering a desktop window, it writes the frame to a JPEG buffer and saves the state to a shared dictionary for FastAPI to read.
*   **`snapshot_engine.py`**: A utility class used to save explicit images or frames to the disk for logging or debugging purposes.
*   **`spatial_engine.py`**: The mathematical core. It contains raw geometric functions (Intersection-over-Union (IoU), distance calculations, centroid math) that the `relationship_engine.py` relies on.
*   **`timeline_engine.py`**: An analytics tool that queries the `object_events` database table to reconstruct a chronological UI timeline for any given object.
*   **`web_server.py`**: The FastAPI application. It routes REST requests, manages WebSocket (`/ws/live`) connections for data streaming, and handles the MJPEG HTTP multipart stream (`/video/stream`). It also manages the lifespan startup/shutdown of the headless loop.

Happy coding and exploring the Smart Vision Assistant!
