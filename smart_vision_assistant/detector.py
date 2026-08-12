"""
detector.py
===========
Real-time YOLO11m object detector with BoT-SORT tracking.
Single responsibility: inference + box rendering.

Key changes from v1:
    - Fast model (YOLO11n) for primary loop
    - Accurate model (YOLO11m) for small object ROI verification
    - Colour-coded bounding boxes: green=default, orange=verified, yellow=new
    - Frame pre-resize before passing to YOLO (saves internal copy)
  - Clean model loading with proper fallback
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection result with optional track assignment."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2  (original frame coords)
    track_id: Optional[int] = None


class ObjectDetector:
    """Wrapper around Ultralytics YOLO11m with BoT-SORT tracking."""

    def __init__(self) -> None:
        self._model_fast = None
        self._model_accurate = None
        self._class_names: List[str] = []
        self._last_verification = {}
        self._load_model()

    # ──────────────────────────────────────────────────────────────────────────
    # Model Loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        models_dir = Path(config.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        def load_yolo(model_name: str) -> YOLO:
            candidates = [
                models_dir / model_name,
                Path(model_name),
            ]
            model_path = next((p for p in candidates if p.exists()), None)
            try:
                if model_path:
                    log.info("Loading YOLO model from: %s", model_path)
                    model = YOLO(str(model_path))
                else:
                    log.info("Model %s not found locally — downloading...", model_name)
                    model = YOLO(model_name)
                    self._cache_model(model_name, models_dir / model_name)
                return model
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load YOLO model '{model_name}': {exc}\n"
                    "Ensure internet access on first run, or place the .pt file in models/"
                ) from exc

        self._model_fast = load_yolo(config.YOLO_MODEL_FAST)
        self._model_accurate = load_yolo(config.YOLO_MODEL_ACCURATE) if config.ENABLE_SMALL_OBJECT_VERIFICATION else None

        if hasattr(self._model_fast, "names"):
            raw = self._model_fast.names
            self._class_names = list(raw.values()) if isinstance(raw, dict) else list(raw)

        log.info(
            "YOLO models loaded. Classes: %d | Base Size: %dpx | Tracker: %s",
            len(self._class_names), config.NORMAL_INFERENCE_RESOLUTION, config.TRACKER_TYPE,
        )

    def _cache_model(self, model_name: str, target: Path) -> None:
        """Copy downloaded model weights to models/ for offline future use."""
        import shutil
        candidates = [
            Path(model_name),
            Path.home() / ".ultralytics" / "assets" / model_name,
        ]
        for src in candidates:
            if src.exists() and src != target:
                shutil.copy2(str(src), str(target))
                log.info("Cached model %s to %s for offline use.", model_name, target)
                return

    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, use_accurate: bool = False) -> List[Detection]:
        """
        Run YOLO11 + BoT-SORT on one frame.
        Returns a list of Detection objects with track IDs assigned.
        If use_accurate is True, uses YOLO_MODEL_ACCURATE at NORMAL resolution.
        Otherwise uses YOLO_MODEL_FAST and performs ROI verification for small objects.
        """
        model = self._model_accurate if use_accurate else self._model_fast
        
        if model is None or frame is None or frame.size == 0:
            return []

        try:
            results = model.track(
                source=frame,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.NMS_IOU_THRESHOLD,
                max_det=config.MAX_DETECTIONS,
                imgsz=config.NORMAL_INFERENCE_RESOLUTION,
                verbose=False,
                stream=False,
                persist=True,
                tracker=config.TRACKER_TYPE,
            )
        except Exception as exc:
            log.warning("YOLO inference error: %s", exc)
            return []

        import time
        now = time.monotonic()
        frame_area = frame.shape[0] * frame.shape[1]
        
        detections: List[Detection] = []
        verifications_this_second = 0
        
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = (
                    self._class_names[cls_id]
                    if cls_id < len(self._class_names)
                    else f"class_{cls_id}"
                )
                
                # Original frame coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else None
                
                # --- Small Object Verification Logic ---
                if config.ENABLE_SMALL_OBJECT_VERIFICATION and not use_accurate and track_id is not None:
                    bbox_area = (x2 - x1) * (y2 - y1)
                    if bbox_area / frame_area < config.SMALL_OBJECT_AREA_THRESHOLD:
                        last_v = self._last_verification.get(track_id, 0)
                        if now - last_v >= config.SMALL_OBJECT_VERIFICATION_COOLDOWN:
                            if verifications_this_second < config.MAX_VERIFICATIONS_PER_SECOND:
                                # Run Verification on Crop
                                crop = frame[max(0, y1):min(frame.shape[0], y2), 
                                             max(0, x1):min(frame.shape[1], x2)]
                                if crop.size > 0:
                                    verifications_this_second += 1
                                    self._last_verification[track_id] = now
                                    try:
                                        v_res = self._model_accurate.predict(
                                            source=crop,
                                            conf=config.CONFIDENCE_THRESHOLD,
                                            imgsz=config.SMALL_OBJECT_INFERENCE_RESOLUTION,
                                            verbose=False,
                                        )
                                        if v_res and v_res[0].boxes and len(v_res[0].boxes) > 0:
                                            # Take the best verification
                                            best_vbox = sorted(v_res[0].boxes, key=lambda b: float(b.conf[0]), reverse=True)[0]
                                            v_cls = int(best_vbox.cls[0])
                                            conf = float(best_vbox.conf[0])
                                            label = self._class_names[v_cls] if v_cls < len(self._class_names) else f"class_{v_cls}"
                                            log.info("Verified small object %s (ID %s) with conf %.2f", label, track_id, conf)
                                    except Exception as exc:
                                        log.warning("YOLO verification error: %s", exc)

                detections.append(Detection(label, conf, (x1, y1, x2, y2), track_id))

        return detections

    # ──────────────────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[Detection],
        scene_memory=None,           # SceneMemory instance (optional)
        verification_cache: Dict = None,
    ) -> None:
        """
        Draw colour-coded bounding boxes onto the frame in-place.
          - Yellow  : newly appeared object
          - Orange  : Gemini-verified label
          - Green   : standard YOLO detection
        """
        verification_cache = verification_cache or {}

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            tid = det.track_id

            # Determine label to display
            display_label = det.label
            if verification_cache and tid in verification_cache:
                display_label = verification_cache[tid].get("label", det.label)

            id_str = f"#{tid}" if tid is not None else "#?"
            label_txt = f"{display_label} {id_str} {det.confidence:.0%}"

            # Colour selection
            is_verified = tid is not None and verification_cache and tid in verification_cache and bool(verification_cache[tid].get("label"))
            is_new = False
            if scene_memory is not None and tid is not None:
                rec = scene_memory.get_record(tid)
                is_new = rec.is_new if rec else False

            if is_new:
                colour = config.COLOUR_BOX_NEW
            elif is_verified:
                colour = config.COLOUR_BOX_VERIFIED
            else:
                colour = config.COLOUR_BOX_DEFAULT

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Label background + text
            (tw, th), baseline = cv2.getTextSize(
                label_txt, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1
            )
            label_y = max(y1 - 5, th + 5)
            cv2.rectangle(
                frame,
                (x1, label_y - th - 4),
                (x1 + tw + 6, label_y + baseline),
                colour, cv2.FILLED,
            )
            cv2.putText(
                frame, label_txt, (x1 + 3, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                config.COLOUR_TEXT, 1, cv2.LINE_AA,
            )
