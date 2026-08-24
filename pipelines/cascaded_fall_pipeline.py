"""
Cascaded Fall Detection Pipeline
================================
Uses a two-stage approach for highly accurate fall detection without false positives (e.g. sitting):
1. Primary Detector: YOLO-NAS (INT8/P2 optimized) for robust distant human detection.
2. Secondary Analyzer: Custom `best.pt` model to classify falls.

If `best.pt` detects a "fall" box that heavily overlaps with the primary "person" tracker, 
an alert is triggered.
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, Optional

from ultralytics import YOLO
import supervision as sv
from trackers import ByteTrackTracker

from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

log = logging.getLogger(__name__)

# Constants
FALL_CLASS_NAME = "fall"      # Based on the classes in best.pt
PRIMARY_CONF_THRESH = 0.35    # YOLO-NAS person detection
SECONDARY_CONF_THRESH = 0.60  # best.pt fall classification
FALL_IOU_OVERLAP = 0.40       # Minimum overlap to match a 'fall' box to a tracked 'person' box
ALERT_COOLDOWN = 10.0         # Seconds before re-alerting on the same tracked person


class CascadedFallPipeline(BaseVideoPipeline):
    """
    Two-stage pipeline: YOLO-NAS (person) -> best.pt (fall classification).
    """

    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        if config is None:
            config = {}

        # 1. Primary Model (YOLO-NAS INT8 optimized for distant/small objects)
        primary_weight = config.get("primary_model_weight", "yolov8s.pt") # Fallback to standard if NAS not found
        
        # 2. Secondary Model (Custom Fall Analyzer)
        secondary_weight = config.get("secondary_model_weight", "best.pt")

        log.info("[CascadedFall] Loading Primary Detector: %s", primary_weight)
        try:
            self.primary_model = YOLO(primary_weight)
        except Exception as e:
            log.warning("[CascadedFall] Failed to load %s. Falling back to yolov8n.pt. Error: %s", primary_weight, e)
            self.primary_model = YOLO("yolov8n.pt")

        log.info("[CascadedFall] Loading Secondary Analyzer: %s", secondary_weight)
        try:
            self.secondary_model = YOLO(secondary_weight)
        except Exception as e:
            log.error("[CascadedFall] Failed to load %s. Fall detection will be disabled. Error: %s", secondary_weight, e)
            self.secondary_model = None

        self.tracker = ByteTrackTracker()
        self.last_alerts = {}  # track_id -> timestamp of last alert

    def process_frame(
        self, 
        frame: np.ndarray, 
        frame_idx: int, 
        roi_polygon: np.ndarray, 
        config: Dict[str, Any]
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        # Required by BaseVideoPipeline but processing happens entirely in run_on_video
        return frame, {}

    def _iou(self, boxA, boxB) -> float:
        """Calculate Intersection over Union for two [x1, y1, x2, y2] boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def run_on_video(self, input_path: str, output_dir: str, **kwargs):
        config = kwargs.get("config", {})
        # Reload models if config asks for it
        if "primary_model_weight" in config or "secondary_model_weight" in config:
            self.initialize(config)

        # Connect to stream
        cap = get_video_source(input_path)
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            frame_idx += 1

            metadata = {"detections": [], "alerts": []}
            now = time.time()

            # --- Stage 1: Primary Detection (Find People) ---
            primary_results = self.primary_model(frame, conf=PRIMARY_CONF_THRESH, verbose=False, classes=[0])[0] # class 0 is usually 'person'
            primary_detections = sv.Detections.from_ultralytics(primary_results)
            
            # Track people
            tracked_detections = self.tracker.update(primary_detections)

            # --- Stage 2: Secondary Analysis (Are they falling?) ---
            # We only run best.pt if there are actually people in the frame to save CPU
            fall_boxes = []
            if len(tracked_detections) > 0 and self.secondary_model is not None:
                secondary_results = self.secondary_model(frame, conf=SECONDARY_CONF_THRESH, verbose=False)[0]
                
                # Filter for "fall" class only
                if secondary_results.boxes is not None:
                    names = secondary_results.names
                    for box in secondary_results.boxes:
                        cls_id = int(box.cls[0])
                        label = names.get(cls_id, "")
                        if label.lower() == FALL_CLASS_NAME:
                            xyxy = box.xyxy[0].cpu().numpy()
                            fall_boxes.append(xyxy)

            # --- Stage 3: Correlate and Alert ---
            for i, xyxy in enumerate(tracked_detections.xyxy):
                track_id = tracked_detections.tracker_id[i] if tracked_detections.tracker_id is not None else f"unk_{i}"
                
                # Check if this person overlaps heavily with a "fall" box from best.pt
                is_falling = False
                for f_box in fall_boxes:
                    if self._iou(xyxy, f_box) > FALL_IOU_OVERLAP:
                        is_falling = True
                        break

                color = (0, 255, 0) # Green = Normal
                status = "Normal"

                if is_falling:
                    color = (0, 0, 255) # Red = Fall
                    status = "FALLING"

                    # Trigger Alert Logic
                    last_time = self.last_alerts.get(track_id, 0)
                    if now - last_time > ALERT_COOLDOWN:
                        self.last_alerts[track_id] = now
                        metadata["alerts"].append({
                            "type": "fall_detected",
                            "severity": "high",
                            "message": f"Fall detected (ID: {track_id})",
                        })

                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id} {status}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                metadata["detections"].append({
                    "id": track_id,
                    "label": "person",
                    "status": status,
                    "bbox": [x1, y1, x2-x1, y2-y1]
                })

            # Overlay diagnostic HUD
            cv2.putText(frame, f"Cascaded Fall | Tracking: {len(tracked_detections)} | Falls: {len(metadata['alerts'])}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            yield frame, metadata
            
        cap.release()
