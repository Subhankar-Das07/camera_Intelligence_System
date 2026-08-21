import cv2
import numpy as np
import logging
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from ultralytics import YOLO

from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source
from .database import VehicleDatabase
from .plate_reader import PlateReader
from .visit_manager import VisitManager

logger = logging.getLogger(__name__)

# COCO class IDs that represent vehicles
_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Map COCO class IDs to human-readable vehicle type labels
_VEHICLE_TYPE_MAP: dict = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# Visual style constants
_BOX_COLOR       = (0, 220, 60)      # Bright green — confirmed plate
_LABEL_BG_COLOR  = (0, 160, 40)      # Darker green label background
_SCAN_COLOR      = (30, 170, 255)    # Amber — still scanning
_FONT            = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE      = 0.55
_FONT_THICKNESS  = 2
_BOX_THICKNESS   = 2

# OCR throttle: only run OCR every N frames per vehicle
_OCR_EVERY_N_FRAMES = 5


class VehicleRecognitionPipeline(BaseVideoPipeline):
    """
    End-to-end Vehicle Recognition & Visit Counting pipeline.

    Integrates:
    - YOLOv8 vehicle detection + ByteTrack multi-object tracking
    - Optional dedicated license-plate detector (YOLO-based)
    - RapidOCR plate reading with multi-frame consensus
    - SQLite visit database with session-aware cooldown

    Implements :class:`core.base_pipeline.BaseVideoPipeline` so it can be
    instantiated and dispatched via the existing ``PipelineRegistry``.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(
        self,
        vehicle_weights: str = "yolov8n.pt",
        plate_weights: Optional[str] = None,
        blur_threshold: float = 40.0,
        min_consensus_frames: int = 3,
        session_cooldown_seconds: int = 120,
        db_path: str = "storage/vehicle_intelligence.db",
        **kwargs,
    ) -> None:
        """
        Load all models and initialize sub-components.

        Args:
            vehicle_weights (str): Path / hub name for the YOLOv8 vehicle detector.
            plate_weights (Optional[str]): Path to a dedicated license-plate YOLO model.
                If None, the pipeline falls back to a bottom-ROI heuristic.
            blur_threshold (float): Laplacian variance threshold passed to PlateReader.
            min_consensus_frames (int): Frames required for majority-vote consensus.
            session_cooldown_seconds (int): Seconds before a returning plate is
                                            counted as a new visit.
            db_path (str): File path for the SQLite database.
        """
        logger.info("Initializing VehicleRecognitionPipeline…")

        # ── Sub-component setup ────────────────────────────────────────
        os.makedirs("storage/vehicle_images", exist_ok=True)
        self.db = VehicleDatabase(db_path=db_path)
        self.plate_reader = PlateReader(blur_threshold=blur_threshold)
        self.visit_manager = VisitManager(
            db_manager=self.db,
            min_consensus_frames=min_consensus_frames,
            session_cooldown_seconds=session_cooldown_seconds,
        )

        # ── Vehicle detector ──────────────────────────────────────────
        try:
            self.detector = YOLO(vehicle_weights)
            logger.info(f"Vehicle detector loaded: {vehicle_weights}")
        except Exception as e:
            logger.error(f"Failed to load vehicle detector '{vehicle_weights}': {e}")
            raise

        # ── Optional dedicated plate detector ─────────────────────────
        self.plate_detector: Optional[YOLO] = None
        if plate_weights:
            try:
                self.plate_detector = YOLO(plate_weights)
                logger.info(f"Plate detector loaded: {plate_weights}")
            except Exception as e:
                logger.warning(
                    f"Could not load plate detector '{plate_weights}': {e}. "
                    "Falling back to ROI heuristic."
                )

        logger.info("VehicleRecognitionPipeline initialized successfully.")

    # ------------------------------------------------------------------
    # Core frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        roi_polygon: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect, track, read plates, and annotate a single video frame.

        Args:
            frame (np.ndarray): Raw BGR frame from the video source.
            frame_idx (int): Zero-based frame index used for OCR throttling.
            roi_polygon (np.ndarray | None): Optional ROI mask polygon (unused currently).
            config (dict | None): Runtime config overrides (unused currently).

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - annotated_frame: Frame with vehicle boxes and plate labels drawn.
                - frame_metadata: Dict with list of detections in this frame.
        """
        frame_metadata: Dict[str, Any] = {"detections": []}

        if frame is None or frame.size == 0:
            return frame, frame_metadata

        # Work on a copy so any exception on this frame never corrupts the source
        annotated = frame.copy()

        try:
            # ── 1. YOLO tracking ─────────────────────────────────────
            results = self.detector.track(
                annotated,
                persist=True,
                classes=_VEHICLE_CLASSES,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            if not results or results[0].boxes.id is None:
                return annotated, frame_metadata

            boxes     = results[0].boxes.xyxy.cpu().numpy().astype(int)   # (N, 4)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)     # (N,)
            cls_ids   = results[0].boxes.cls.cpu().numpy().astype(int)    # (N,)

            # ── 2. Stale-track cleanup ────────────────────────────────
            active_track_ids: List[int] = track_ids.tolist()
            self.visit_manager.cleanup_stale_tracks(active_track_ids)

            # ── 3. Per-vehicle processing ─────────────────────────────
            h_frame, w_frame = annotated.shape[:2]

            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                try:
                    x1, y1, x2, y2 = box
                    # Clamp to frame boundaries
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w_frame, int(x2)), min(h_frame, int(y2))

                    car_crop = annotated[y1:y2, x1:x2]
                    if car_crop.size == 0:
                        continue

                    tid          = int(track_id)
                    vehicle_type = _VEHICLE_TYPE_MAP.get(int(cls_id), "Vehicle")

                    # ── OCR throttling ─────────────────────────────
                    # Skip OCR if: track already finalized  OR  not on the Nth frame
                    already_finalized = tid in self.visit_manager.finalized_tracks
                    run_ocr = (not already_finalized) and (frame_idx % _OCR_EVERY_N_FRAMES == 0)

                    if run_ocr:
                        plate_crop = self._extract_plate_crop(car_crop)
                        if plate_crop is not None and plate_crop.size > 0:
                            plate_text, conf = self.plate_reader.read_plate(plate_crop)
                            if plate_text:
                                self.visit_manager.add_frame_reading(
                                    track_id=tid,
                                    plate_text=plate_text,
                                    confidence=conf,
                                    car_crop=car_crop.copy(),
                                    plate_crop=plate_crop.copy(),
                                    vehicle_type=vehicle_type,
                                )

                    # ── Consensus check ────────────────────────────
                    status = self.visit_manager.process_track(tid)
                    
                    if status and not already_finalized:
                        try:
                            if car_crop is not None and car_crop.size > 0:
                                plate_text = status["plate"]
                                filename = f"storage/vehicle_images/{plate_text}_{int(time.time())}.jpg"
                                cv2.imwrite(filename, car_crop)
                                self.db.update_vehicle_image(plate_text, filename)
                                # Save into the state dict so it persists for this track
                                status["image_path"] = filename
                        except Exception as e:
                            logger.error(f"Error saving image/DB for plate {status.get('plate')}: {e}")
                            print(f"Error saving image/DB: {e}")

                    # ── Draw annotations directly on annotated frame ──
                    box_color = _BOX_COLOR if status else _SCAN_COLOR
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, _BOX_THICKNESS)

                    if status:
                        label    = f"{status.get('vehicle_type', 'Vehicle')}: {status['plate']}  |  Visits: {status['total_visits']}"
                        bg_color = _LABEL_BG_COLOR
                    else:
                        label    = f"ID: {tid}  |  Scanning..."
                        bg_color = _SCAN_COLOR

                    # Label banner background
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label, _FONT, _FONT_SCALE, _FONT_THICKNESS
                    )
                    banner_y1 = max(0, y1 - text_h - baseline - 6)
                    banner_y2 = y1
                    banner_x2 = min(w_frame, x1 + text_w + 8)
                    cv2.rectangle(annotated, (x1, banner_y1), (banner_x2, banner_y2), bg_color, -1)
                    cv2.putText(
                        annotated, label,
                        (x1 + 4, y1 - baseline - 2),
                        _FONT, _FONT_SCALE, (255, 255, 255), _FONT_THICKNESS, cv2.LINE_AA,
                    )

                    # ── Metadata ───────────────────────────────────
                    frame_metadata["detections"].append({
                        "track_id":     tid,
                        "box":          [x1, y1, x2, y2],
                        "plate":        status["plate"]                    if status else None,
                        "total_visits": status["total_visits"]             if status else None,
                        "image_path":   status.get("image_path")           if status else None,
                        "vehicle_type": status.get("vehicle_type", "Car") if status else None,
                    })

                except Exception as e:
                    logger.warning(f"Error processing track {track_id}: {e}", exc_info=False)
                    continue

        except Exception as e:
            # Critical error: log it but return the (partially annotated) frame
            # so the MJPEG stream NEVER crashes or times out.
            logger.error(f"Critical error in process_frame (idx={frame_idx}): {e}", exc_info=True)

        return annotated, frame_metadata

    # ------------------------------------------------------------------
    # Video loop — accepts roi_normalized from main.py _vr_mjpeg_generator
    # ------------------------------------------------------------------

    def run_on_video(
        self,
        input_path: str,
        output_dir: str,
        roi_normalized=None,   # accepted from main.py but unused; keeps signature flexible
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[Tuple[np.ndarray, Optional[Dict[str, Any]]], None, None]:
        """
        Run the pipeline on a video file or live stream, yielding annotated frames.

        Args:
            input_path: Path or URL of the video source.
            output_dir: Directory for output artefacts (unused).
            roi_normalized: Normalized ROI polygon from the UI (accepted, currently unused).
            config (dict): Runtime configuration overrides.

        Yields:
            Tuple[np.ndarray, Optional[Dict[str, Any]]]: Annotated frame and metadata.
        """
        if config is None:
            config = {}

        cap = get_video_source(input_path)

        if not cap.isOpened():
            logger.error(f"Cannot open video source: {input_path}")
            return

        frame_idx       = 0
        skip_rate       = 3   # Process 1 frame, skip the next 2
        last_valid_frame = None
        metadata         = {"detections": []}

        try:
            while cap.isOpened():
                # 1. ALWAYS grab to drain the RTSP buffer and prevent timestamp lag
                grabbed = cap.grab()
                if not grabbed:
                    break

                frame_idx += 1

                # 2. Only decode + run heavy AI on every Nth frame
                if frame_idx % skip_rate == 0:
                    ret, frame = cap.retrieve()
                    if not ret:
                        break

                    annotated_frame, metadata = self.process_frame(
                        frame, frame_idx=frame_idx, config=config
                    )
                    last_valid_frame = annotated_frame   # update memory buffer

                # 3. Yield on EVERY iteration — re-broadcast last frame on skipped ticks
                #    so the MJPEG stream never stalls or drops.
                if last_valid_frame is not None:
                    yield last_valid_frame, metadata

        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_plate_crop(self, car_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the license plate region from a vehicle crop.

        Strategy:
        - If a dedicated plate detector is available, run YOLO inference.
        - Otherwise, return the bottom 45% of the crop as the candidate region
          (plates are mounted low on vehicles).

        Args:
            car_crop (np.ndarray): Cropped vehicle image (BGR).

        Returns:
            Optional[np.ndarray]: Cropped plate region, or None on failure.
        """
        if self.plate_detector is not None:
            try:
                plate_results = self.plate_detector(car_crop, verbose=False)
                if plate_results and plate_results[0].boxes is not None and len(plate_results[0].boxes):
                    plate_boxes = plate_results[0].boxes.xyxy.cpu().numpy().astype(int)
                    confs       = plate_results[0].boxes.conf.cpu().numpy()
                    best_idx    = int(np.argmax(confs))
                    px1, py1, px2, py2 = plate_boxes[best_idx]
                    h_c, w_c = car_crop.shape[:2]
                    px1, py1 = max(0, px1), max(0, py1)
                    px2, py2 = min(w_c, px2), min(h_c, py2)
                    plate_crop = car_crop[py1:py2, px1:px2]
                    if plate_crop.size > 0:
                        return plate_crop
            except Exception as e:
                logger.debug(f"Plate detector inference failed, falling back to ROI: {e}")

        # Fallback: bottom 45% of the vehicle crop
        h, w = car_crop.shape[:2]
        roi_y_start = int(h * 0.55)
        return car_crop[roi_y_start:h, 0:w]
