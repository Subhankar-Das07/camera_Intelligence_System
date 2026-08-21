"""
config.py — Single source of truth for all face recognition constants.

Persistence: Redis only (identities, embeddings, face crop bytes).
FAISS IndexFlatIP is rebuilt in-memory from Redis on startup/mutation.
InsightFace model packs still download under face_recognition/models/.
"""

import os

# ── Paths (models only; identity DB is Redis) ─────────────────────────────────
MODELS_CACHE_DIR = os.path.join("face_recognition", "models")

# ── InsightFace / Detection ───────────────────────────────────────────────────
INSIGHTFACE_MODEL_PACK  = "buffalo_l"
DETECTION_THRESHOLD     = 0.45
MIN_FACE_SIZE_PX        = 40

# ── Quality Filtering ─────────────────────────────────────────────────────────
SHARPNESS_THRESHOLD     = 50.0

# ── ArcFace / Cosine Similarity (IndexFlatIP) ─────────────────────────────────
SIMILARITY_THRESHOLD    = 0.30
MARGIN_THRESHOLD        = 0.04

# ── Temporal Consensus ────────────────────────────────────────────────────────
TEMPORAL_WINDOW_SIZE    = 8
CONSENSUS_MIN_VOTES     = 4

# ── Candidate Buffer (auto-enroll) ────────────────────────────────────────────
CANDIDATE_MIN_EMBEDDINGS = 3

# ── Pipeline Behaviour ────────────────────────────────────────────────────────
RECOGNITION_EVERY_N_FRAMES = 3

# ── Annotation Colours (BGR for OpenCV) ───────────────────────────────────────
COLOR_KNOWN       = (0,  200,  80)
COLOR_UNKNOWN     = (0,  140, 255)
COLOR_LOW_QUALITY = (100, 100, 100)
COLOR_CANDIDATE   = (200, 160,  60)

# ── Identity Naming ───────────────────────────────────────────────────────────
PERSON_ID_PREFIX   = "P"
AUTO_LABEL_PREFIX  = "Person"
