"""
Room Guardian Pipeline  v5.0 - YOLO-World + OpenVINO Async + ByteTrack
============================================================================
- Open-Vocabulary YOLO-World (detect any object by text).
- OpenVINO FP16 IR export and AsyncInferQueue multithreading.
- PrePostProcessor (PPP) for zero-copy iGPU resizing.
- ByteTrack for robust object tracking on the async results.
"""

import cv2
import numpy as np
import uuid
import os
import logging
import queue
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import deque

import supervision as sv
from trackers import ByteTrackTracker

from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

log = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
CONF_THRESHOLD   = 0.20       # Detection confidence threshold
ANCHOR_IOU       = 0.40       # Min IoU to anchor user's scan box to a ByteTrack box

# ── Colours ───────────────────────────────────────────────────────────────────
COLOR_PRESENT  = (0, 255, 100)
COLOR_MISSING  = (0, 165, 255)
COLOR_ALERT    = (0, 0, 255)
COLOR_LABEL_BG = (20, 20, 20)

@dataclass
class WatchedObject:
    id:             str
    label:          str
    last_bbox:      List[int]            # [x, y, w, h] absolute px

    track_id:       Optional[int] = None
    track_ok:       bool = False
    missing_frames: int = 0
    alert_fired:    bool = False
    center_history: deque = field(default_factory=lambda: deque(maxlen=30), repr=False)

def _iou(boxA: List[int], boxB: List[int]) -> float:
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = ax1 + boxA[2], ay1 + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = bx1 + boxB[2], by1 + boxB[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0: return 0.0
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
    return inter / union if union > 0 else 0.0

def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple) -> None:
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    cv2.rectangle(frame, (x, y - th - pad * 2), (x + tw + pad * 2, y), COLOR_LABEL_BG, -1)
    cv2.putText(frame, text, (x + pad, y - pad), font, scale, color, thickness, cv2.LINE_AA)

def _write_clip(frames: list, output_path: str, fps: float, size: tuple) -> bool:
    if not frames: return False
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)
    if not writer.isOpened(): return False
    for f in frames: writer.write(f)
    writer.release()
    return True

def _detect_exit(center_history: deque, width: int, height: int, margin: float = 0.05) -> str:
    if len(center_history) < 3: return "unknown"
    pts = list(center_history)[-3:]
    dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
    last_x, last_y = pts[-1]
    if ((last_x > width*(1-margin) and dx > 0) or (last_x < width*margin and dx < 0) or
        (last_y > height*(1-margin) and dy > 0) or (last_y < height*margin and dy < 0)):
        return "left_frame"
    return "occluded_or_removed"

# ── OpenVINO Logic ───────────────────────────────────────────────────────────

def _ensure_openvino_model(pt_path: str, vocabulary: List[str] = None) -> Optional[str]:
    base = os.path.splitext(pt_path)[0]
    if vocabulary:
        import hashlib
        vocab_str = "_".join(sorted(vocabulary))
        vocab_hash = hashlib.md5(vocab_str.encode()).hexdigest()[:8]
        ov_dir = f"{base}_ov_{vocab_hash}"
    else:
        ov_dir = f"{base}_openvino_model"
        
    xml_path = os.path.join(ov_dir, f"{os.path.basename(base)}.xml")
    if os.path.isfile(xml_path):
        return xml_path

    try:
        from ultralytics import YOLOWorld
        model = YOLOWorld(pt_path)
        if vocabulary:
            model.set_classes(vocabulary)
        log.info("[Guardian] Exporting YOLO-World to OpenVINO FP16...")
        export_path = model.export(format="openvino", half=True, imgsz=640)
        if isinstance(export_path, str) and export_path.endswith(".xml"):
            return export_path
        if os.path.isfile(xml_path):
            return xml_path
        return None
    except Exception as exc:
        log.warning("[Guardian] OpenVINO export failed (%s).", exc)
        return None

def _build_ov_engine(xml_path: str, ov):
    core = ov.Core()
    available = core.available_devices
    device = "CPU"
    for candidate in ("GPU", "CPU"):
        if candidate in available:
            device = candidate
            break
    try:
        core.set_property(device, {"PERFORMANCE_HINT": "THROUGHPUT"})
    except Exception:
        pass

    ppp_enabled = False
    try:
        from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
        from openvino import Layout, Type
        model = core.read_model(xml_path)
        ppp = PrePostProcessor(model)
        inp = ppp.input()
        inp.tensor().set_spatial_dynamic_shape().set_element_type(Type.u8)\
            .set_color_format(ColorFormat.BGR).set_layout(Layout("NHWC"))
        inp.model().set_layout(Layout("NCHW"))
        inp.preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)\
            .convert_color(ColorFormat.RGB).convert_element_type(Type.f32).scale(255.0)
        model = ppp.build()
        compiled = core.compile_model(model, device)
        ppp_enabled = True
    except Exception:
        compiled = core.compile_model(xml_path, device)
    
    return compiled, compiled.inputs[0].any_name, [o.any_name for o in compiled.outputs], device, ppp_enabled

def _preprocess_frame(frame: np.ndarray, imgsz: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    dw, dh = (imgsz - new_w) // 2, (imgsz - new_h) // 2
    canvas[dh:dh+new_h, dw:dw+new_w, :] = resized
    tensor = canvas.transpose((2, 0, 1))[np.newaxis, ...]
    tensor = tensor.astype(np.float32) / 255.0
    return tensor

def _postprocess_ov(output_tensor, img_w, img_h, imgsz=640, conf_thres=0.20):
    boxes = output_tensor[0]
    boxes = np.transpose(boxes)
    scores = np.max(boxes[:, 4:], axis=1)
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
        
    class_ids = np.argmax(boxes[:, 4:], axis=1)
    
    r = min(imgsz / img_h, imgsz / img_w)
    dw, dh = (imgsz - int(img_w * r)) / 2, (imgsz - int(img_h * r)) / 2
    
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2 - dw) / r
    y1 = (cy - h / 2 - dh) / r
    x2 = (cx + w / 2 - dw) / r
    y2 = (cy + h / 2 - dh) / r
    
    xyxy = np.column_stack([x1, y1, x2, y2])
    
    # NMS
    indices = cv2.dnn.NMSBoxes(xyxy.tolist(), scores.tolist(), conf_thres, 0.45)
    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([])
    indices = indices.flatten()
    
    return xyxy[indices], scores[indices], class_ids[indices]

class RoomGuardianPipeline(BaseVideoPipeline):
    def initialize(self, model_weight: str = "yolov8s-worldv2.pt", **kwargs) -> None:
        self.pt_path = model_weight
        self.tracker = ByteTrackTracker()

    def process_frame(self, frame, frame_idx, roi_polygon, config):
        return frame, {}

    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        cap = get_video_source(input_path)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        absence_threshold = int(fps * 5)
        ring_buffer = deque(maxlen=int(fps * 4.5))
        pending_alerts = []
        frame_idx = 0

        raw_objects = config.get("watched_objects", [])
        vocabulary = config.get("config", {}).get("vocabulary", None)
        
        watched: List[WatchedObject] = []
        for obj in raw_objects:
            bn = obj.get("bbox_normalized", [0, 0, 0.1, 0.1])
            px = [int(bn[0]*width), int(bn[1]*height), int(bn[2]*width), int(bn[3]*height)]
            watched.append(WatchedObject(id=obj.get("id", str(uuid.uuid4())), label=obj.get("label", "Object"), last_bbox=px))

        self.tracker.reset()

        import openvino as ov
        xml_path = _ensure_openvino_model(self.pt_path, vocabulary)
        if not xml_path:
            log.error("[Guardian] OpenVINO compilation failed. Aborting.")
            return

        compiled, in_name, out_names, device, ppp_enabled = _build_ov_engine(xml_path, ov)
        
        num_jobs = 4
        result_q = queue.Queue(maxsize=num_jobs * 2)
        async_queue = ov.AsyncInferQueue(compiled, jobs=num_jobs)
        
        def cb(infer_request, userdata):
            out_tensor = infer_request.get_output_tensor(0).data
            xyxy, scores, class_ids = _postprocess_ov(out_tensor, userdata["img_w"], userdata["img_h"])
            result_q.put({"frame": userdata["frame"], "frame_idx": userdata["frame_idx"], "xyxy": xyxy, "scores": scores, "class_ids": class_ids})
        async_queue.set_callback(cb)
        
        in_flight = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            raw_frame = frame.copy()
            if async_queue.is_ready() or in_flight < num_jobs:
                inp = frame[np.newaxis] if ppp_enabled else _preprocess_frame(frame, 640)
                async_queue.start_async({in_name: inp}, {"frame_idx": frame_idx, "frame": frame.copy(), "img_w": width, "img_h": height})
                in_flight += 1

            while not result_q.empty() or (in_flight >= num_jobs):
                try:
                    res = result_q.get(timeout=0.1)
                    in_flight -= 1
                    
                    render_frame = res["frame"]
                    
                    if len(res["xyxy"]) > 0:
                        detections = sv.Detections(xyxy=res["xyxy"], confidence=res["scores"], class_id=res["class_ids"])
                    else:
                        detections = sv.Detections.empty()
                        
                    detections = self.tracker.update(detections)
                    
                    tracked_boxes = {}
                    if len(detections) > 0:
                        for xyxy, mask, conf, cid, tid, data in detections:
                            if tid is not None:
                                tracked_boxes[int(tid)] = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])]
                    
                    assigned_ids = set()
                    for wo in watched:
                        wo.track_ok = False
                        if wo.track_id is None:
                            best_iou, best_tid = 0, None
                            for tid, tbox in tracked_boxes.items():
                                if tid in assigned_ids: continue
                                iou = _iou(wo.last_bbox, tbox)
                                if iou > best_iou: best_iou, best_tid = iou, tid
                            if best_tid is not None and best_iou > ANCHOR_IOU:
                                wo.track_id = best_tid
                                wo.last_bbox = tracked_boxes[best_tid]
                                wo.track_ok = True
                                assigned_ids.add(best_tid)
                        else:
                            if wo.track_id in tracked_boxes:
                                wo.last_bbox = tracked_boxes[wo.track_id]
                                wo.track_ok = True
                                assigned_ids.add(wo.track_id)
                            else:
                                wo.track_ok = False
                                
                        if wo.track_ok:
                            wo.center_history.append((wo.last_bbox[0]+wo.last_bbox[2]/2, wo.last_bbox[1]+wo.last_bbox[3]/2))
                            if wo.alert_fired and wo.missing_frames > 0: wo.alert_fired = False
                            wo.missing_frames = 0
                        else:
                            wo.missing_frames += 1
                            if wo.missing_frames >= absence_threshold and not wo.alert_fired:
                                wo.alert_fired = True
                                pending_alerts.append({
                                    "wo": wo, "target_frame": frame_idx + int(fps*2),
                                    "disappearance_frame": max(0, frame_idx - absence_threshold),
                                    "exit_type": _detect_exit(wo.center_history, width, height)
                                })
                                
                        x, y, w, h = wo.last_bbox
                        color = COLOR_ALERT if wo.alert_fired else (COLOR_MISSING if wo.missing_frames > 0 else COLOR_PRESENT)
                        status = f"{wo.label} [MISSING!]" if wo.alert_fired else (f"{wo.label} [Lost]" if wo.missing_frames > 0 else f"{wo.label} [{wo.track_id}]")
                        cv2.rectangle(render_frame, (x, y), (x+w, y+h), color, 2)
                        _draw_label(render_frame, status, x, y, color)
                    
                    ring_buffer.append(raw_frame)
                    
                    alert_event = None
                    for pa in pending_alerts[:]:
                        if frame_idx >= pa["target_frame"]:
                            wo = pa["wo"]
                            alert_id = str(uuid.uuid4())
                            clip_path = os.path.join(output_dir, f"guardian_{alert_id}.mp4")
                            _write_clip(list(ring_buffer), clip_path, fps, (width, height))
                            ts_sec = pa["disappearance_frame"] / fps
                            alert_event = {
                                "id": alert_id, "object_id": wo.id, "object_label": wo.label,
                                "timestamp_sec": ts_sec, "formatted_time": f"{int(ts_sec//60):02d}:{int(ts_sec%60):02d}",
                                "clip_url": f"/storage/alerts/guardian_{alert_id}.mp4", "severity": "MISSING",
                                "immediate_buzzer_trigger": True, "exit_type": pa["exit_type"]
                            }
                            pending_alerts.remove(pa)
                            break
                            
                    cv2.putText(render_frame, f"YOLO-World + OpenVINO | Device: {device}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,100), 2)
                    yield render_frame, alert_event
                    
                except queue.Empty:
                    break
                    
            frame_idx += 1
            
        async_queue.wait_all()
        cap.release()