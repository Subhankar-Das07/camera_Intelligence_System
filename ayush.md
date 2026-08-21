# ayush.md — Change Log (Face Recognition Module)

This file tracks every change made by Ayush's branch for the Face Recognition feature.
Updated in parallel with implementation. Nothing from the existing project was deleted.

---

## Session: 2026-08-18 to 2026-08-19 — Face Recognition Implementation

### Core Philosophy
1. **Single Database Storage:** All unique faces are automatically stored upon their first visit in a unified database (`persons/`). There is no physical separation of "known" vs "unknown" folders.
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
| Vector DB | faiss-cpu | Fast local search, no cloud |
| Hardware | CPU-only (onnxruntime) | Hardware-agnostic; GPU = swap package |
| Recognition freq | Every 3rd frame | 66% CPU saving, configurable |

---

## Files Created (All New — Nothing Deleted)

### Core Engine
| File | Purpose |
|---|---|
| `face_recognition/__init__.py` | Package init, module docstring |
| `face_recognition/config.py` | All tunable constants in one place |
| `face_recognition/identity_manager.py` | FAISS + NPZ + JSON + image crop persistence |
| `face_recognition/embedder.py` | InsightFace buffalo_l: SCRFD detect + ArcFace embed |
| `face_recognition/tracker.py` | ByteTrack wrapper, candidate embedding buffers |
| `face_recognition/recognizer.py` | FAISS search, temporal consensus, identity lifecycle |
| `face_recognition/data/.gitkeep` | Keep data dir in git, actual data gitignored |

### Pipeline Integration
| File | Purpose |
|---|---|
| `pipelines/face_recognition_pipeline.py` | Full pipeline implementing BaseVideoPipeline. Translates confidence scores into Known/Unknown bounding box labels. |

### Frontend
| File | Purpose |
|---|---|
| `static/face_recognition/index.html` | Dedicated FR dashboard page |
| `static/face_recognition/fr_style.css` | Dark-mode styles, purple accent |
| `static/face_recognition/fr_app.js` | Full dashboard JavaScript |

---

## Files Modified (Additive Only — Existing Code Untouched)

### `core/registry.py`
- Added import: `from pipelines.face_recognition_pipeline import FaceRecognitionPipeline`
- Added registration: `registry.register("face_recognition", FaceRecognitionPipeline)`

### `main.py`
- Appended Face Recognition API endpoints block after existing `get_alerts` endpoint:
  - `GET  /api/faces/status`
  - `GET  /api/faces/identities`
  - `POST /api/faces/register`
  - `POST /api/faces/snapshot/{stream_id}`
  - `PATCH /api/faces/identity/{person_id}`
- Added static mounts: `/face_data` → `face_recognition/data/`

### `static/index.html`
- Added `👤 Face Recognition →` button in Section 2 (Vision Pipeline), below the dropdown
- Opens `face_recognition/` in a new browser tab

### `static/style.css`
- Appended `.fr-nav-btn` styles (purple gradient button with pulse animation)
- Appended `.alert-inline-video` style (was missing from original)

### `requirements.txt`
- Appended: `insightface`, `onnxruntime`, `faiss-cpu`, `supervision`, `Pillow`

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
  FAISS Top-K Search
        ↓
  Temporal Consensus (N/M votes in sliding window)
        ↓
  ┌─────────────┴─────────────┐
(conf > 0.0)              (conf == 0.0)
Match found               No Match (Auto-saving)
  │                           │
"Known" (Green Box)       "Unknown" (Orange Box)
```

## Local Storage Structure

```text
face_recognition/data/
├── identities.json      ← person metadata (names, timestamps)
├── embeddings.npz       ← per-person embedding arrays
├── faces.faiss          ← FAISS flat L2 index
└── persons/             ← unified storage
    ├── P000001/         ← individual person directories
    │   └── face_001.jpg
    └── P000002/
        └── face_001.jpg
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
1. **Dual Identity Databases**: Separated identity storage (`attendance_data`) for the Attendance mode. `IdentityManager` now accepts a `base_dir` initialization parameter.
2. **Attendance Tracker Mode**: Added an Operation Mode selector to the UI. When enabled:
   - Auto-registration of unknown faces is disabled via a new `auto_register=False` flag in `FaceRecognizer.classify()`.
   - Known users (students) are marked as `Present` (green bounding box).
   - Unknown people are marked as `Unknown` (red bounding box) but not saved.
   - When the session completes, any registered student that was not seen is marked as `Absent` (red).
3. **Session APIs**: Added `/api/attendance/start`, `/api/attendance/stop`, and `/api/attendance/status` endpoints to `main.py` to track active attendance sessions.
4. **UI Updates**: 
   - Operation Mode dropdown added to the sidebar.
   - Dynamic Registration Panel based on active mode.
   - Bounding box and identity label colors adapted for attendance context (Green for Present, Red for Absent/Unknown, Grey for Not Seen).
