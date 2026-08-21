"""
config.py — Single source of truth for all face recognition constants.

Storage model (v3):
    ONE folder: face_recognition/data/persons/
    Every unique person gets a subfolder: persons/P000001/, persons/P000002/, ...
    No separate "known" / "unknown" split in storage.
    Auto-label: Person_001, Person_002, ...  (rename via UI to a real name anytime)

    On every server start, all embeddings in persons/ are loaded into FAISS.
    When a webcam face matches FAISS score >= SIMILARITY_THRESHOLD → show as KNOWN (green).
    When no match found → save as new Person_NNN → show as KNOWN immediately.

Similarity metric:
    faiss.IndexFlatIP on L2-normalised ArcFace embeddings → cosine similarity.
    Range: -1.0 (opposite) to 1.0 (identical).
    Same person, good conditions:  ~0.5 – 0.85
    Same person, diff angle/light: ~0.35 – 0.60
    Different person:              ~-0.1 – 0.25
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR             = os.path.join("face_recognition", "data")
ATTENDANCE_DIR       = os.path.join("face_recognition", "attendance_data")
PERSONS_DIR          = "persons"   # Relative to base_dir
IDENTITIES_FILE      = "identities.json"
EMBEDDINGS_FILE      = "embeddings.npz"
FAISS_INDEX_FILE     = "faces.faiss"
MODELS_CACHE_DIR     = os.path.join("face_recognition", "models")

# ── InsightFace / Detection ───────────────────────────────────────────────────
INSIGHTFACE_MODEL_PACK  = "buffalo_l"
DETECTION_THRESHOLD     = 0.45   # SCRFD confidence gate
MIN_FACE_SIZE_PX        = 40     # Min bbox side length

# ── Quality Filtering ─────────────────────────────────────────────────────────
SHARPNESS_THRESHOLD     = 50.0   # Laplacian variance; relaxed for webcam

# ── ArcFace / Cosine Similarity (IndexFlatIP) ─────────────────────────────────
SIMILARITY_THRESHOLD    = 0.30   # cosine_sim >= this → face recognised
MARGIN_THRESHOLD        = 0.04   # best score must beat 2nd-best by at least this

# ── Temporal Consensus ────────────────────────────────────────────────────────
TEMPORAL_WINDOW_SIZE    = 8      # Sliding window of votes per ByteTrack track
CONSENSUS_MIN_VOTES     = 4      # Votes needed to commit an identity

# ── Candidate Buffer (auto-enroll) ────────────────────────────────────────────
CANDIDATE_MIN_EMBEDDINGS = 3     # Frames of good quality needed before saving

# ── Pipeline Behaviour ────────────────────────────────────────────────────────
RECOGNITION_EVERY_N_FRAMES = 3   # Run recognition every N frames (CPU saving)

# ── Annotation Colours (BGR for OpenCV) ───────────────────────────────────────
COLOR_KNOWN       = (0,  200,  80)    # Green  — face matched / saved in DB
COLOR_UNKNOWN     = (0,  140, 255)    # Orange — face being scanned (edge-case only)
COLOR_LOW_QUALITY = (100, 100, 100)   # Gray   — too blurry / too small
COLOR_CANDIDATE   = (200, 160,  60)   # Blue   — actively collecting frames

# ── Identity Naming ───────────────────────────────────────────────────────────
PERSON_ID_PREFIX   = "P"
AUTO_LABEL_PREFIX  = "Person"    # Auto-labels: Person_001, Person_002, ...
