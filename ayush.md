# ayush.md — Change Log (Face Recognition Module)

This file tracks every change made by Ayush's branch for the Face Recognition feature.

---

## Session: 2026-08-18 to 2026-08-19 — Face Recognition Implementation

### Core Philosophy
1. **Single Database Storage:** All unique faces are automatically stored upon their first visit in a unified database (Redis). There is no physical separation of "known" vs "unknown" folders.
2. **"Unknown" vs "Known" Live Logic:** 
   - A person is labeled **Unknown** on the live feed ONLY IF they were *never* stored in the DB before this current visit (i.e. confidence score `== 0.0`). The system auto-saves them behind the scenes.
   - A person is labeled **Known** on the live feed IF they match an existing DB entry (i.e. confidence score `> 0.0`), proving they have visited before.
3. **Identity Manager UI:** A clean, right-side storage panel simply lists everyone the system has ever seen under the "Known" tab. Users can rename auto-saved `Person_001` identities to real names.

### Technical Decisions
| Decision | Choice | Reason |
|---|---|---|
| Detection | InsightFace buffalo_l SCRFD | Best accuracy/speed, ONNX |
| Recognition | ArcFace (buffalo_l) | Phone-unlock-level accuracy |
| Tracking | supervision.ByteTrack | Stable track IDs, pip install |
| Vector DB | faiss-cpu | In-memory fast local search |
| Hardware | CPU-only (onnxruntime) | Hardware-agnostic; GPU = swap package |
| Recognition freq | Every 3rd frame | 66% CPU saving, configurable |

---

## Architecture Overview

```text
Webcam / RTSP / Upload
        ↓
    OpenCV (cap.read)
        ↓
  SCRFD Detection (InsightFace)
        ↓
  Quality Filter (size + Laplacian sharpness)
        ↓
  ByteTrack (every frame, stable IDs)
        ↓
  ArcFace Embedding (every 3rd frame)
        ↓
  FAISS Top-K Search (In-Memory)
        ↓
  Temporal Consensus (N/M votes in sliding window)
        ↓
  ┌─────────────┴─────────────┐
(conf > 0.0)              (conf == 0.0)
Match found               No Match (Auto-saving to Redis)
  │                           │
"Known" (Green Box)       "Unknown" (Orange Box)
```

## Redis Data Structure

All persistent data for Face Recognition is now stored natively in Redis, ensuring rapid access and avoiding local file-system clutter.

```text
fr:identities              → JSON object {person_id: metadata}
fr:emb:{person_id}         → numpy float32 matrix bytes
fr:embmeta:{person_id}     → shape and dtype metadata
fr:face:{person_id}:{n}    → JPEG face crop bytes
fr:face_count:{person_id}  → integer count of crops
```

## Configuration Reference (face_recognition/config.py)

| Constant | Default | Purpose |
|---|---|---|
| `SIMILARITY_THRESHOLD` | 0.30 | cosine distance cutoff |
| `MARGIN_THRESHOLD` | 0.04 | best vs 2nd-best gap |
| `TEMPORAL_WINDOW_SIZE` | 8 | sliding window frames |
| `CONSENSUS_MIN_VOTES` | 4 | votes needed to commit |
| `MIN_FACE_SIZE_PX` | 40 | min bbox size in pixels |
| `SHARPNESS_THRESHOLD` | 50.0 | Laplacian variance gate |
| `CANDIDATE_MIN_EMBEDDINGS` | 3 | embeddings before new ID |
| `RECOGNITION_EVERY_N_FRAMES` | 3 | 1-in-N frame recognition |

---

## Session: 2026-08-20 — Attendance Tracking Implementation

### What was added
1. **Dual Identity Databases**: Separated identity storage keys in Redis for the Attendance mode versus Visitor mode.
2. **Attendance Tracker Mode**: Added an Operation Mode selector to the UI. When enabled:
   - Auto-registration of unknown faces is disabled.
   - Known users (students) are marked as `Present` (green bounding box).
   - Unknown people are marked as `Unknown` (red bounding box) but not saved.
   - When the session completes, any registered student that was not seen is marked as `Absent` (red).
3. **Session APIs**: Added `/api/attendance/start`, `/api/attendance/stop`, and `/api/attendance/status` endpoints to `main.py` to track active attendance sessions.

---

## Session: 2026-08-22 — Windows Native Execution & Redis Migration

### Native Windows Camera Support
Because Docker on Windows runs inside a Linux virtual machine, it physically cannot access native Windows USB webcams. To fix this, we:
1. Created `start-windows-app.bat` to launch the server natively directly on Windows.
2. Maintained the `docker-compose.yml` configuration for deployments to Linux servers.
3. This allows seamless local development using `http://127.0.0.1:8000` while utilizing the local physical webcam.

### Redis Migration
We executed a complete data migration from local storage (`identities.json`, `embeddings.npz`, `faces.faiss`, and the `persons/` directory) into a centralized **Redis** database.
- `identity_manager.py` was completely rewritten to read and write directly to Redis.
- FAISS IndexFlatIP is now rebuilt in-memory dynamically from Redis data on application startup.
