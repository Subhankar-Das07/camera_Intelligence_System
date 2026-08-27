# Face Recognition Performance Analysis & Improvement Plan

Based on the analysis of the `face_recognition` module (specifically `embedder.py` and `pipeline.py`), here are the root causes of the severe latency and low FPS when multiple people are in the camera's view, along with the planned solutions.

## 🔴 Root Causes of High Latency

### 1. Redundant Heavy Neural Network Execution
The single biggest issue is how `InsightFace` is being called in `embedder.py`:
```python
faces = self._app.get(frame)
```
The `app.get()` method is a high-level wrapper that automatically runs **both** Face Detection (SCRFD) **AND** Feature Extraction (ArcFace) on the image. 
Because this is called on **every single frame** of the RTSP stream, if there are 5 people in the room, the system is forcing the heavy ArcFace model to run 5 separate times per frame. If your camera is running at 30 FPS, the system is attempting 150 heavy Neural Network inferences per second!

### 2. Underutilized Face Tracker
While the system correctly uses ByteTrack (`FaceTracker`) to track faces across frames, it is only using the tracker to smooth the final database lookups (FAISS). 
In `pipeline.py`, there is a variable `do_recognition = (frame_idx % RECOGNITION_EVERY_N_FRAMES == 0)`. However, this only skips the FAISS database lookup. The system still extracts the 512-dimensional embeddings for all 5 faces on every frame, which completely defeats the purpose of tracking for performance.

### 3. CPU Execution Bottleneck
The `FaceAnalysis` model is explicitly initialized with `"CPUExecutionProvider"`. Running 150 inferences per second on a CPU is mathematically impossible for most modern processors without introducing massive input lag and dropping the FPS to near 0.

---

## 🟢 Implementation Plan for Better Performance

To solve this efficiently without requiring GPU hardware, we must completely decouple Face Detection from Feature Extraction.

### Option 1: True Tracker-Based Decoupling (Chosen Approach)
Instead of calling `self._app.get(frame)`, we manually extract the detection and recognition models from `InsightFace`.
1. **Detection Phase:** On every frame, we ONLY run the SCRFD detection model. This is very lightweight and can easily handle 5+ people at 30 FPS on a CPU.
2. **Tracking Phase:** ByteTrack assigns a consistent `track_id` to each person.
3. **Recognition Phase:** We only run the heavy ArcFace embedding model **once every N frames** (e.g., every 5-10 frames) per `track_id`. If a person's identity is already known and confident, we don't need to re-embed their face on every single frame.

### Option 2: Frame Skipping (Alternative)
We only process 1 out of every 5 frames from the RTSP stream, entirely ignoring the frames in between. This immediately cuts the workload by 80%, but makes the bounding boxes look laggy/choppy on the frontend stream.

*Note: Option 1 is the recommended approach and will be implemented when requested.*
