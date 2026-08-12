"""
config.py
=========
Configuration parameters for the Smart Vision Assistant.
Terminal-output visual perception engine — no voice/TTS.
"""

# ──────────────────────────────────────────────────────────────────────────────
# CAMERA
# ──────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX: int = 0               # 0 = default webcam
FRAME_WIDTH: int = 640              # Capture width
FRAME_HEIGHT: int = 480             # Capture height
CAMERA_FPS: int = 30                # Target camera capture rate

# ──────────────────────────────────────────────────────────────────────────────
# DETECTION (YOLO11)
# ──────────────────────────────────────────────────────────────────────────────
YOLO_MODEL_FAST: str = "yolo11n.pt"      # Default fast model
YOLO_MODEL_ACCURATE: str = "yolo11m.pt"  # Verification model
MODELS_DIR: str = "models"               # Local directory for cached weights

NORMAL_INFERENCE_RESOLUTION: int = 640   # Base YOLO input resolution
SMALL_OBJECT_INFERENCE_RESOLUTION: int = 960 # ROI verification resolution

CONFIDENCE_THRESHOLD: float = 0.25       # Minimum detection confidence (lowered from 0.45)
NMS_IOU_THRESHOLD: float = 0.45          # Non-max suppression IoU
MAX_DETECTIONS: int = 100                # Cap per-frame detection count
TRACKER_TYPE: str = "bytetrack.yaml"     # ByteTrack (motion-only, ~35% faster than BoT-SORT)

# Small Object Verification
ENABLE_SMALL_OBJECT_VERIFICATION: bool = False
SMALL_OBJECT_AREA_THRESHOLD: float = 0.02 # Trigger verification if bbox area < 2% of frame
SMALL_OBJECT_VERIFICATION_COOLDOWN: float = 1.0 # Seconds between verifications per entity
MAX_VERIFICATIONS_PER_SECOND: int = 3     # Global cap on verifications per second

# Dynamic Model Switching Hysteresis
MODEL_SWITCH_COOLDOWN: float = 60.0       # Minimum seconds between model swaps
SWITCH_TO_NANO_CPU_THRESHOLD: float = 85.0
SWITCH_TO_NANO_FPS_THRESHOLD: float = 5.0
SWITCH_TO_MEDIUM_CPU_THRESHOLD: float = 70.0
SWITCH_TO_MEDIUM_FPS_THRESHOLD: float = 8.0

# Frame skipping: run YOLO every Nth frame to protect FPS
# When FPS < FRAME_SKIP_FPS_THRESHOLD, inference is run every FRAME_SKIP_N frames
FRAME_SKIP_N: int = 2               # Run inference every 2nd frame when FPS is low
FRAME_SKIP_FPS_THRESHOLD: float = 15.0  # Skip frames if FPS drops below this

# ──────────────────────────────────────────────────────────────────────────────
# SCENE MEMORY
# ──────────────────────────────────────────────────────────────────────────────
# Performance & Inference
SKIP_FRAMES = 1
FRAME_TIMEOUT = 0.5           # Max time to wait for a frame (seconds)
OBJECT_NEW_GRACE = 2.0        # Seconds before an object is considered 'stable'
OBJECT_STALE_TIMEOUT = 2.0    # Seconds before a lost object is permanently removed

# Spatial & Analytics
DEBUG_SPATIAL_JITTER = False  # If True, prints raw vs EMA coordinates to console

# ──────────────────────────────────────────────────────────────────────────────
# REPORTING ENGINE
# ──────────────────────────────────────────────────────────────────────────────
REPORT_INTERVAL: float = 15.0       # Seconds between console scene reports
REPORT_TO_FILE: bool = True         # Also log reports to reports/ directory
STABILITY_WINDOW: float = 10.0      # Seconds of stability history for % metric

# ──────────────────────────────────────────────────────────────────────────────
# GEMINI VERIFICATION  (async background — never blocks detection loop)
# ──────────────────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-2.0-flash-lite"  # Cheapest multimodal flash model
ENABLE_GEMINI: bool = False                  # Global toggle for Gemini
ENABLE_AUTO_VERIFY: bool = False             # Toggle for auto-verification on low conf
GEMINI_VERIFY_THRESHOLD: float = 0.70        # Auto-verify detections below this conf
GEMINI_MAX_PENDING: int = 6                  # Max items queued for verification
GEMINI_VERIFY_COOLDOWN: float = 3.0          # Minimum seconds between API calls
GEMINI_SESSION_BUDGET: int = 60              # Hard cap on API calls per session
GEMINI_MIN_OBJECT_AGE: float = 1.5          # Only verify objects seen for ≥N seconds
DEEP_ANALYSIS_ENABLED: bool = True           # Request rich Gemini descriptions for all objects
DEBUG_BYPASS_FPS_GATING: bool = False       # Respect FPS limits for Gemini queueing

# ──────────────────────────────────────────────────────────────────────────────
# OCR & PRODUCT INTELLIGENCE
# ──────────────────────────────────────────────────────────────────────────────
OCR_PRIMARY_ENGINE: str = "paddleocr"        # 'paddleocr' or 'easyocr'
ENABLE_OCR: bool = False                     # Global toggle for OCR initialization and processing
OCR_MIN_CONFIDENCE: float = 0.75             # Ignore text regions below this confidence
OCR_MIN_BBOX_SIZE: int = 40                  # Min width/height of crop to run OCR
OCR_ENTITY_COOLDOWN: float = 5.0             # Seconds between OCR attempts per entity
OCR_MAX_QUEUE_SIZE: int = 5                  # Drop crops if OCR thread falls behind
OCR_MAX_WORKERS: int = 1                     # Number of async OCR threads


# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY / HUD  (cv2.imshow window — kept for visual debugging)
# ──────────────────────────────────────────────────────────────────────────────
SHOW_OVERLAYS_DEFAULT: bool = True   # Start with bounding boxes on
FONT_SCALE: float = 0.50             # Base font scale for overlays

# Colour palette (BGR format)
COLOUR_BOX_DEFAULT  = (0, 220, 80)   # Default bounding box (green)
COLOUR_BOX_VERIFIED = (0, 180, 255)  # Gemini-verified object (orange)
COLOUR_BOX_NEW      = (0, 255, 255)  # Newly appeared object (yellow)
COLOUR_TEXT         = (255, 255, 255)
COLOUR_HUD_BG       = (0, 0, 0)
COLOUR_FPS          = (0, 255, 200)
COLOUR_WARNING      = (0, 100, 255)

# ──────────────────────────────────────────────────────────────────────────────
# FPS MONITORING  (adaptive frame-skip thresholds)
# ──────────────────────────────────────────────────────────────────────────────
FPS_WARN_THRESHOLD: float = 12.0    # Warn in HUD if FPS falls below this
FPS_CRITICAL_THRESHOLD: float = 8.0 # Disable Gemini auto-verify below this
