"""
Footfall Analytics Pipeline
============================================================================
- Decoupled Multi-Process Architecture
- POSIX Shared Memory Ring Buffer
- Hardware-Accelerated OpenVINO Inference (iGPU + CPU)
- Bottom-Center Footpoint Zone Analytics
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
SHM_PREFIX = "camera_stream_"
RING_SIZE = 5 # Small ring buffer of 5 frames

class GateAnalyticsEngine:
    def __init__(self, outside_roi_poly, inside_roi_poly):
        self.outside_poly = np.array(outside_roi_poly, dtype=np.int32)
        self.inside_poly = np.array(inside_roi_poly, dtype=np.int32)
        self.track_history = {}
        self.total_in = 0
        self.total_out = 0

    def get_footpoint(self, bbox):
        x1, y1, w, h = bbox
        return int(x1 + w / 2), int(y1 + h)

    def process_tracks(self, tracked_boxes: dict):
        events = []
        for track_id, bbox in tracked_boxes.items():
            footpoint = self.get_footpoint(bbox)
            in_outside = cv2.pointPolygonTest(self.outside_poly, footpoint, False) >= 0
            in_inside = cv2.pointPolygonTest(self.inside_poly, footpoint, False) >= 0
            
            curr_state = None
            if in_outside: curr_state = "OUTSIDE"
            elif in_inside: curr_state = "INSIDE"
            
            if curr_state:
                prev_state = self.track_history.get(track_id)
                if prev_state == "OUTSIDE" and curr_state == "INSIDE":
                    self.total_in += 1
                    events.append({"event": "ENTRY", "track_id": track_id})
                elif prev_state == "INSIDE" and curr_state == "OUTSIDE":
                    self.total_out += 1
                    events.append({"event": "EXIT", "track_id": track_id})
                self.track_history[track_id] = curr_state
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
        
        # Attach to shared memory blocks created by the parent
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
            
            # Write in-place without memory allocation
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
                control_q.put(write_idx) # Block to match UI speed
                
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

def _inference_worker(shm_name_base, frame_shape, frame_dtype, ring_size, pt_path, stop_event, control_q, output_q, outside_poly, inside_poly, camera_id, input_path):
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
            
        # Export or Load OpenVINO model
        xml_path = _ensure_openvino_model(pt_path)
        if not xml_path:
            log.error("[Inference] OpenVINO compilation failed. Aborting.")
            return

        core = ov.Core()
        device_str = "AUTO"
        
        # Optimize for iGPU if available (per user instruction)
        available = core.available_devices
        if "GPU" in available:
            # Use GPU for compute, CPU for fallback
            device_str = "MULTI:GPU,CPU"
            
        # Compile Model with PrePostProcessor
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
        gate_engine = GateAnalyticsEngine(outside_poly, inside_poly)
        redis_client = get_redis()
        
        in_flight = 0
        img_w, img_h = frame_shape[1], frame_shape[0]
        
        # Create polys for rendering
        pts_out = np.array(outside_poly, np.int32).reshape((-1, 1, 2))
        pts_in = np.array(inside_poly, np.int32).reshape((-1, 1, 2))
        
        ingest_done = False
        while not stop_event.is_set():
            # Get latest frame
            if not ingest_done:
                try:
                    read_idx = control_q.get(timeout=0.1)
                    if read_idx == -1:
                        ingest_done = True
                    elif async_queue.is_ready() or in_flight < num_jobs:
                        # Make a local copy of the frame from shared memory so ingestion can overwrite the block
                        frame = shm_arrays[read_idx].copy()
                        inp = frame[np.newaxis]
                        async_queue.start_async({in_name: inp}, {"frame": frame, "img_w": img_w, "img_h": img_h})
                        in_flight += 1
                except queue.Empty:
                    pass
                
            # Process results
            while not result_q.empty() or (in_flight >= num_jobs) or (ingest_done and in_flight > 0):
                try:
                    res = result_q.get(timeout=0.1)
                    in_flight -= 1
                    
                    render_frame = res["frame"]
                    
                    # Filter for person class (class_id 0 in COCO)
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
                                
                    events = gate_engine.process_tracks(tracked_boxes)
                    
                    pipeline_alert = None
                    for ev in events:
                        # Redis XADD
                        payload = {
                            "camera_id": camera_id,
                            "timestamp": time.time(),
                            "type": ev["event"],
                            "track_id": ev["track_id"],
                            "current_occupancy": gate_engine.total_in - gate_engine.total_out
                        }
                        try:
                            redis_client.xadd("stream:footfall", payload)
                        except Exception as e:
                            log.error(f"[Inference] Redis XADD failed: {e}")
                            
                        # UI alerts removed per user request
                    
                    # Render
                    cv2.polylines(render_frame, [pts_out], True, (255, 0, 0), 2)
                    cv2.polylines(render_frame, [pts_in], True, (0, 0, 255), 2)
                    
                    for tid, bbox in tracked_boxes.items():
                        x, y, w, h = bbox
                        footpoint = gate_engine.get_footpoint(bbox)
                        cv2.rectangle(render_frame, (x, y), (x+w, y+h), (0, 255, 100), 2)
                        cv2.circle(render_frame, footpoint, 5, (0, 255, 255), -1)
                        cv2.putText(render_frame, f"ID: {tid}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
                        
                    cv2.putText(render_frame, f"IN: {gate_engine.total_in} | OUT: {gate_engine.total_out}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(render_frame, f"Device: {device_str}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                        
                    # Push to output
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
        log.info(f"[Footfall] Exporting {pt_path} to OpenVINO FP16...")
        export_path = model.export(format="openvino", half=True, imgsz=640)
        if isinstance(export_path, str) and export_path.endswith(".xml"):
            return export_path
        if os.path.isfile(xml_path):
            return xml_path
        return None
    except Exception as exc:
        log.warning("[Footfall] OpenVINO export failed (%s).", exc)
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

class FootfallAnalyticsPipeline(BaseVideoPipeline):
    def initialize(self, model_weight: str = "yolov8n.pt", **kwargs) -> None:
        self.pt_path = model_weight

    def process_frame(self, frame, frame_idx, roi_polygon, config):
        return frame, {}

    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        # Allow dynamic polys from config, or default to splitting screen in half
        outside_poly = config.get("outside_poly")
        inside_poly = config.get("inside_poly")
        
        # Extract underlying source if a ThreadedCamera was passed
        original_src = input_path.src if hasattr(input_path, 'src') else input_path
        
        cap = get_video_source(original_src)
        if not cap.isOpened():
            log.error("[Footfall] Failed to open video source.")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Only release if it's a new instance we just created for checking dimensions
        if cap is not input_path:
            cap.release()
        
        if not outside_poly or not inside_poly:
            outside_poly = [[0, 0], [0, height], [width//2, height], [width//2, 0]]
            inside_poly = [[width//2, 0], [width//2, height], [width, height], [width, 0]]
        else:
            # Unnormalize if needed. Avoid isinstance(float) because JSON may parse 0.0 as int 0.
            max_val = max([max(x, y) for x, y in outside_poly])
            if max_val <= 1.0:
                outside_poly = [[int(x*width), int(y*height)] for x,y in outside_poly]
                inside_poly = [[int(x*width), int(y*height)] for x,y in inside_poly]
            
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
            p_infer = mp.Process(target=_inference_worker, args=(shm_name_base, frame_shape, frame_dtype, RING_SIZE, self.pt_path, stop_event, control_q, output_q, outside_poly, inside_poly, camera_id, original_src))
            
            p_ingest.start()
            p_infer.start()
            
            try:
                while True:
                    try:
                        res = output_q.get(timeout=0.1)
                        if res == (None, None):
                            break
                        jpeg_bytes, event = res
                        # We must yield a numpy array for the main generator, so we decode it here
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
