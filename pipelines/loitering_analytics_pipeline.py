"""
Loitering Analytics Pipeline
============================================================================
- Decoupled Multi-Process Architecture
- POSIX Shared Memory Ring Buffer
- Hardware-Accelerated OpenVINO Inference (iGPU + CPU)
- Identifies Loitering Behaviors Using Trajectory Analytics
"""

import cv2
import numpy as np
import uuid
import os
import logging
import queue
import time
from typing import Optional, List, Dict, Any
import multiprocessing as mp
from multiprocessing import shared_memory
import traceback

import supervision as sv
from trackers import ByteTrackTracker
from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source
from core.redis_client import get_redis

log = logging.getLogger(__name__)

# Shared memory configuration
SHM_PREFIX = "loiter_stream_"
RING_SIZE = 5

class LoiteringAnalyticsEngine:
    def __init__(self):
        self.trajectories = {}
        self.last_alert_times = {}
        self.MAX_TRAJECTORY_LENGTH = 300
        self.COOLDOWN = 10
        
        # Parameters from RS-WACV24_Loitering
        self.theta = 13
        self.feature_points_threshold = 4
        self.window_size = 4
        self.S_threshold_Ellipse = 72.26
        self.S_threshold_convex = 59.70
        self.S_threshold_Rectangle = 196.75
        self.T0_Rectangle = 2
        self.S_threshold_closed = 0.0445
        self.std_short = 22.16
        self.radius_short = 60
        self.frame_threshold_short = 7
        self.std_long = 22.16
        self.radius_long = 60
        self.frame_threshold_long = 94
        self.T0_sector = 51
        self.M1 = 60
        self.M2 = 112
        self.S_threshold_sector = 41.81

    def process_tracks(self, tracked_boxes: dict):
        events = []
        current_time = time.time()
        
        # We need to import inside the worker process
        from pipelines.loitering_methods import (
            find_loitering_start_and_end,
            ellipse_loitering_detection,
            convex_hull_loitering_detection,
            RectangleLoiteringDetection,
            closed_areas_loitering_detection,
            sector_loitering_detection,
            no_motion_loitering_detection
        )
        
        for track_id, bbox in tracked_boxes.items():
            x1, y1, w, h = bbox
            # Use bottom center point (footpoint) for trajectory
            center = (int(x1 + w / 2), int(y1 + h))
            
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
            self.trajectories[track_id].append(center)
            
            if len(self.trajectories[track_id]) > self.MAX_TRAJECTORY_LENGTH:
                self.trajectories[track_id].pop(0)
                
            coords = np.array(self.trajectories[track_id])
            
            # Check cooldown
            if track_id in self.last_alert_times:
                if current_time - self.last_alert_times[track_id] < self.COOLDOWN:
                    continue
                    
            if len(coords) < 10:
                continue
                
            loitering_detected = False
            
            try:
                fp_start, fp_end = find_loitering_start_and_end(coords, self.theta, self.window_size, self.feature_points_threshold)
                
                if fp_start is not None:
                    if ellipse_loitering_detection(coords, fp_start, fp_end, self.S_threshold_Ellipse, track_id):
                        loitering_detected = True
                    elif convex_hull_loitering_detection(coords, fp_start, fp_end, self.S_threshold_convex, track_id):
                        loitering_detected = True
                    elif closed_areas_loitering_detection(coords, self.S_threshold_closed, track_id):
                        loitering_detected = True
                    elif sector_loitering_detection(coords, self.T0_sector, self.M1, self.M2, self.S_threshold_sector, track_id):
                        loitering_detected = True
                    else:
                        rect = RectangleLoiteringDetection(coords, self.theta, self.T0_Rectangle, self.S_threshold_Rectangle)
                        if rect.detect_loitering():
                            loitering_detected = True
                else:
                    if no_motion_loitering_detection(coords, mode="short_term", frame_threshold=self.frame_threshold_short, radius=self.radius_short, std_threshold=self.std_short, ID=track_id):
                        loitering_detected = True
                    elif no_motion_loitering_detection(coords, mode="long_term", frame_threshold=self.frame_threshold_long, radius=self.radius_long, std_threshold=self.std_long, ID=track_id):
                        loitering_detected = True
                        
            except Exception:
                # ignore mathematical exceptions for invalid geometries
                pass
                
            if loitering_detected:
                self.last_alert_times[track_id] = current_time
                events.append({"event": "LOITERING", "track_id": track_id})
                
        return events


def _ingestion_worker(input_path, shm_name_base, frame_shape, frame_dtype, ring_size, stop_event, control_q):
    """Worker 1: Hardware-accelerated decode & Latest-Frame Buffer"""
    is_live = True
    if isinstance(input_path, str):
        if not input_path.startswith(('rtsp://', 'http://', 'https://')) and not input_path.isdigit():
            is_live = False
            
    try:
        cap = get_video_source(input_path)
        shm_blocks = []
        shm_arrays = []
        
        for i in range(ring_size):
            shm = shared_memory.SharedMemory(name=f"{shm_name_base}_{i}")
            arr = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)
            shm_blocks.append(shm)
            shm_arrays.append(arr)
            
        write_idx = 0
        while not stop_event.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            np.copyto(shm_arrays[write_idx], frame)
            
            if is_live:
                if control_q.full():
                    try:
                        control_q.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    control_q.put_nowait(write_idx)
                except queue.Full:
                    pass
            else:
                control_q.put(write_idx)
                
            write_idx = (write_idx + 1) % ring_size
            
        cap.release()
    except Exception as e:
        log.error(f"[Ingestion] Error: {e}")
        traceback.print_exc()
    finally:
        try:
            if not stop_event.is_set():
                control_q.put(-1, timeout=1)
        except Exception:
            pass
        for shm in shm_blocks:
            shm.close()

def _inference_worker(shm_name_base, frame_shape, frame_dtype, ring_size, pt_path, stop_event, control_q, output_q, camera_id, input_path):
    """Worker 2: Inference & Tracking with OpenVINO AsyncInferQueue"""
    import openvino as ov
    
    is_live = True
    if isinstance(input_path, str):
        if not input_path.startswith(('rtsp://', 'http://', 'https://')) and not input_path.isdigit():
            is_live = False
            
    try:
        shm_blocks = []
        shm_arrays = []
        for i in range(ring_size):
            shm = shared_memory.SharedMemory(name=f"{shm_name_base}_{i}")
            arr = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)
            shm_blocks.append(shm)
            shm_arrays.append(arr)
            
        xml_path = _ensure_openvino_model(pt_path)
        if not xml_path:
            log.error("[Inference] OpenVINO compilation failed. Aborting.")
            return

        core = ov.Core()
        device_str = "AUTO"
        
        available = core.available_devices
        if "GPU" in available:
            device_str = "MULTI:GPU,CPU"
            
        model = core.read_model(xml_path)
        from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
        from openvino import Layout, Type
        
        ppp = PrePostProcessor(model)
        inp = ppp.input()
        inp.tensor().set_spatial_dynamic_shape().set_element_type(Type.u8)\
            .set_color_format(ColorFormat.BGR).set_layout(Layout("NHWC"))
        inp.model().set_layout(Layout("NCHW"))
        inp.preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)\
            .convert_color(ColorFormat.RGB).convert_element_type(Type.f32).scale(255.0)
            
        model = ppp.build()
        compiled = core.compile_model(model, device_name=device_str, config={"PERFORMANCE_HINT": "THROUGHPUT"})
        in_name = compiled.inputs[0].any_name
        
        num_jobs = 4
        result_q = queue.Queue(maxsize=num_jobs * 2)
        async_queue = ov.AsyncInferQueue(compiled, jobs=num_jobs)
        
        def cb(infer_request, userdata):
            out_tensor = infer_request.get_output_tensor(0).data
            xyxy, scores, class_ids = _postprocess_ov(out_tensor, userdata["img_w"], userdata["img_h"])
            result_q.put({"frame": userdata["frame"], "xyxy": xyxy, "scores": scores, "class_ids": class_ids})
            
        async_queue.set_callback(cb)
        
        tracker = ByteTrackTracker()
        loitering_engine = LoiteringAnalyticsEngine()
        redis_client = get_redis()
        
        in_flight = 0
        img_w, img_h = frame_shape[1], frame_shape[0]
        
        ingest_done = False
        while not stop_event.is_set():
            if not ingest_done:
                try:
                    read_idx = control_q.get(timeout=0.1)
                    if read_idx == -1:
                        ingest_done = True
                    elif async_queue.is_ready() or in_flight < num_jobs:
                        frame = shm_arrays[read_idx].copy()
                        inp = frame[np.newaxis]
                        async_queue.start_async({in_name: inp}, {"frame": frame, "img_w": img_w, "img_h": img_h})
                        in_flight += 1
                except queue.Empty:
                    pass
                
            while not result_q.empty() or (in_flight >= num_jobs) or (ingest_done and in_flight > 0):
                try:
                    res = result_q.get(timeout=0.1)
                    in_flight -= 1
                    
                    render_frame = res["frame"]
                    
                    mask = res["class_ids"] == 0
                    p_xyxy = res["xyxy"][mask]
                    p_scores = res["scores"][mask]
                    p_cids = res["class_ids"][mask]
                    
                    if len(p_xyxy) > 0:
                        detections = sv.Detections(xyxy=p_xyxy, confidence=p_scores, class_id=p_cids)
                        detections = tracker.update(detections)
                    else:
                        detections = sv.Detections.empty()
                        detections = tracker.update(detections)
                        
                    tracked_boxes = {}
                    if len(detections) > 0:
                        for xyxy, m, conf, cid, tid, data in detections:
                            if tid is not None:
                                tracked_boxes[int(tid)] = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])]
                                
                    events = loitering_engine.process_tracks(tracked_boxes)
                    
                    pipeline_alert = None
                    for ev in events:
                        payload = {
                            "camera_id": camera_id,
                            "timestamp": time.time(),
                            "type": ev["event"],
                            "track_id": ev["track_id"]
                        }
                        try:
                            redis_client.xadd("stream:loitering", payload)
                        except Exception:
                            pass
                            
                        # Forward alert to the UI
                        pipeline_alert = {
                            "id": str(uuid.uuid4()), "object_id": str(ev["track_id"]),
                            "object_label": f"Person {ev['track_id']} Loitering",
                            "timestamp_sec": time.time(),
                            "formatted_time": time.strftime("%H:%M:%S"),
                            "clip_url": "", "severity": "LOITERING",
                            "immediate_buzzer_trigger": True
                        }
                    
                    # Render bounding boxes and trajectories
                    for tid, bbox in tracked_boxes.items():
                        x, y, w, h = bbox
                        color = (0, 255, 100)
                        
                        # Draw trajectory
                        if tid in loitering_engine.trajectories:
                            pts = np.array(loitering_engine.trajectories[tid], np.int32)
                            if len(pts) > 2:
                                cv2.polylines(render_frame, [pts], False, (255, 0, 0), 2)
                                
                        cv2.rectangle(render_frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(render_frame, f"ID: {tid}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                    cv2.putText(render_frame, f"Device: {device_str}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                        
                    ret, jpeg = cv2.imencode('.jpg', render_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ret:
                        if is_live:
                            try:
                                output_q.put_nowait((jpeg.tobytes(), pipeline_alert))
                            except queue.Full:
                                try:
                                    output_q.get_nowait()
                                    output_q.put_nowait((jpeg.tobytes(), pipeline_alert))
                                except Exception:
                                    pass
                        else:
                            output_q.put((jpeg.tobytes(), pipeline_alert))
                        
                except queue.Empty:
                    break
            
            if ingest_done and in_flight == 0 and result_q.empty():
                break
                
        async_queue.wait_all()
        try:
            output_q.put((None, None), timeout=1)
        except Exception:
            pass
    except Exception as e:
        log.error(f"[Inference] Error: {e}")
        traceback.print_exc()
    finally:
        for shm in shm_blocks:
            shm.close()

def _ensure_openvino_model(pt_path: str) -> Optional[str]:
    base = os.path.splitext(pt_path)[0]
    ov_dir = f"{base}_openvino_model"
    xml_path = os.path.join(ov_dir, f"{os.path.basename(base)}.xml")
    if os.path.isfile(xml_path):
        return xml_path

    try:
        from ultralytics import YOLO
        model = YOLO(pt_path)
        log.info(f"[Loitering] Exporting {pt_path} to OpenVINO FP16...")
        export_path = model.export(format="openvino", half=True, imgsz=640)
        if isinstance(export_path, str) and export_path.endswith(".xml"):
            return export_path
        if os.path.isfile(xml_path):
            return xml_path
        return None
    except Exception as exc:
        log.warning("[Loitering] OpenVINO export failed (%s).", exc)
        return None
        
def _postprocess_ov(output_tensor, img_w, img_h, imgsz=640, conf_thres=0.25):
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
    
    indices = cv2.dnn.NMSBoxes(xyxy.tolist(), scores.tolist(), conf_thres, 0.45)
    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([])
    indices = indices.flatten()
    
    return xyxy[indices], scores[indices], class_ids[indices]

class LoiteringAnalyticsPipeline(BaseVideoPipeline):
    def initialize(self, model_weight: str = "yolov8n.pt", **kwargs) -> None:
        self.pt_path = model_weight

    def process_frame(self, frame, frame_idx, roi_polygon, config):
        return frame, {}

    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        original_src = input_path.src if hasattr(input_path, 'src') else input_path
        
        cap = get_video_source(original_src)
        if not cap.isOpened():
            log.error("[Loitering] Failed to open video source.")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if cap is not input_path:
            cap.release()
            
        frame_shape = (height, width, 3)
        frame_dtype = np.uint8
        bytes_per_frame = int(np.prod(frame_shape)) * np.dtype(frame_dtype).itemsize
        
        shm_name_base = f"{SHM_PREFIX}{uuid.uuid4().hex[:8]}"
        shm_blocks = []
        try:
            for i in range(RING_SIZE):
                shm = shared_memory.SharedMemory(create=True, size=bytes_per_frame, name=f"{shm_name_base}_{i}")
                shm_blocks.append(shm)
                
            stop_event = mp.Event()
            control_q = mp.Queue(maxsize=RING_SIZE)
            output_q = mp.Queue(maxsize=10)
            
            camera_id = config.get("camera_id", "cam_01")
            
            p_ingest = mp.Process(target=_ingestion_worker, args=(original_src, shm_name_base, frame_shape, frame_dtype, RING_SIZE, stop_event, control_q))
            p_infer = mp.Process(target=_inference_worker, args=(shm_name_base, frame_shape, frame_dtype, RING_SIZE, self.pt_path, stop_event, control_q, output_q, camera_id, original_src))
            
            p_ingest.start()
            p_infer.start()
            
            try:
                while True:
                    try:
                        res = output_q.get(timeout=0.1)
                        if res == (None, None):
                            break
                        jpeg_bytes, event = res
                        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                        yield frame, event
                    except queue.Empty:
                        if not p_infer.is_alive():
                            break
                        continue
            finally:
                stop_event.set()
                p_ingest.join(timeout=2)
                p_infer.join(timeout=2)
                if p_ingest.is_alive(): p_ingest.terminate()
                if p_infer.is_alive(): p_infer.terminate()
                
        finally:
            for shm in shm_blocks:
                shm.close()
                try:
                    shm.unlink()
                except:
                    pass
