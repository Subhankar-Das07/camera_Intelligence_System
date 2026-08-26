"""
pipelines/new_intrusion_pipeline.py
====================================
NewIntrusionPipeline -- OpenVINO Hardware-Accelerated Intrusion Detection

Drop-in replacement for IntrusionDetectionPipeline (registered as "new_intrusion").
Identical business logic, identical generator API, identical alert event schema.

Key architectural differences from the baseline:
  1. OpenVINO AsyncInferQueue replaces blocking YOLO() call
  2. iGPU (Intel UHD / Iris Xe) with automatic CPU fallback
  3. PrePostProcessor (PPP) handles BGR->RGB, HWC->NCHW, /255 normalisation in C++
  4. Frame-drop guard prevents latency accumulation when queue is saturated
  5. Graceful fallback to Ultralytics CPU path if openvino is not installed
"""

# --- Standard library ---------------------------------------------------------
import cv2
import time
import numpy as np
import os
import uuid
import logging
import queue
from typing import Optional, Tuple, List, Dict, Any

# --- Project imports (no modification to existing files) ----------------------
from shapely.geometry import Point, Polygon
from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

log = logging.getLogger(__name__)

# --- COCO-Pose keypoint indices used by the intrusion rule --------------------
_KP_L_ANKLE = 15
_KP_R_ANKLE = 16

# --- Default model weight matching the baseline -------------------------------
_DEFAULT_WEIGHT = "yolov8n-pose.pt"

# =========================================================================
# [NEW_INTRUSION_ENHANCEMENT: OpenVINO Import Guard]
# Architectural Shift: Lazy import so the rest of the project never breaks
#                      if openvino is absent.
# Reason: Zero-dependency footprint for teams not yet on OpenVINO stack.
# =========================================================================
def _try_import_openvino():
    try:
        import openvino as ov
        return ov, True
    except ImportError:
        return None, False


# =========================================================================
# [NEW_INTRUSION_ENHANCEMENT: OpenVINO Model Export]
# Architectural Shift: One-time export of .pt -> OpenVINO FP16 IR (.xml/.bin)
#                      executed at pipeline initialisation, not at inference.
# Reason: FP16 cuts memory bandwidth 2x vs FP32 on iGPU shared LPDDR5.
#         The .pt file is NEVER overwritten.
# =========================================================================
def _ensure_openvino_model(pt_path: str) -> Optional[str]:
    base     = os.path.splitext(pt_path)[0]
    ov_dir   = f"{base}_openvino_model"
    xml_path = os.path.join(ov_dir, f"{os.path.basename(base)}.xml")

    if os.path.isfile(xml_path):
        log.info("[NewIntrusion] OpenVINO IR found: %s", xml_path)
        return xml_path

    log.info("[NewIntrusion] Exporting %s -> OpenVINO FP16 IR ...", pt_path)
    try:
        from ultralytics import YOLO as _YOLO
        model       = _YOLO(pt_path)
        export_path = model.export(format="openvino", half=True, imgsz=640)
        if isinstance(export_path, str) and export_path.endswith(".xml"):
            log.info("[NewIntrusion] Export complete: %s", export_path)
            return export_path
        if os.path.isfile(xml_path):
            return xml_path
        log.warning("[NewIntrusion] Export unexpected path: %s", export_path)
        return None
    except Exception as exc:
        log.warning("[NewIntrusion] OpenVINO export failed (%s). Will use CPU fallback.", exc)
        return None


# =========================================================================
# [NEW_INTRUSION_ENHANCEMENT: Async Inference Engine Setup]
# Architectural Shift: ov.Core + compiled model + AsyncInferQueue (N=2).
#                      Device priority: GPU -> CPU (automatic fallback).
# Reason: Intel iGPU frees CPU threads for video decode and annotation.
# =========================================================================
def _build_ov_engine(xml_path: str, ov):
    core      = ov.Core()
    available = core.available_devices
    log.info("[NewIntrusion] OpenVINO devices: %s", available)

    device = "CPU"
    for candidate in ("GPU", "CPU"):
        if candidate in available:
            device = candidate
            break

    try:
        core.set_property(device, {"PERFORMANCE_HINT": "THROUGHPUT"})
        log.info("[NewIntrusion] Set PERFORMANCE_HINT to THROUGHPUT for %s", device)
    except Exception as e:
        log.warning("[NewIntrusion] Could not set PERFORMANCE_HINT: %s", e)

    ppp_enabled = False
    try:
        from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
        from openvino import Layout, Type

        model = core.read_model(xml_path)
        ppp   = PrePostProcessor(model)
        inp   = ppp.input()

        # =========================================================================
        # [NEW_INTRUSION_ENHANCEMENT: PrePostProcessor Zero-Copy Input]
        # Architectural Shift: PPP accepts raw OpenCV BGR frames (H,W,3 uint8).
        #                      C++ backend handles Resize, HWC->NCHW, BGR->RGB, /255.
        # Reason: Eliminates per-frame NumPy allocations in the Python hot path.
        # =========================================================================
        inp.tensor() \
            .set_spatial_dynamic_shape() \
            .set_element_type(Type.u8) \
            .set_color_format(ColorFormat.BGR) \
            .set_layout(Layout("NHWC"))
        inp.model().set_layout(Layout("NCHW"))
        inp.preprocess() \
            .resize(ResizeAlgorithm.RESIZE_LINEAR) \
            .convert_color(ColorFormat.RGB) \
            .convert_element_type(Type.f32) \
            .scale(255.0)

        model    = ppp.build()
        compiled = core.compile_model(model, device)
        ppp_enabled = True
        log.info("[NewIntrusion] PPP-enabled model compiled on %s", device)

    except Exception as ppp_exc:
        log.warning("[NewIntrusion] PPP unavailable (%s); manual preprocess fallback.", ppp_exc)
        compiled    = core.compile_model(xml_path, device)
        ppp_enabled = False

    input_name   = compiled.inputs[0].any_name
    output_names = [o.any_name for o in compiled.outputs]
    return compiled, input_name, output_names, device, ppp_enabled


# =========================================================================
# [NEW_INTRUSION_ENHANCEMENT: Keypoint Tensor Decoder]
# Architectural Shift: Decode raw (1,57,N) tensor directly in NumPy.
# Reason: Avoids Ultralytics result wrapper overhead; no Python NMS loop.
# =========================================================================
def _decode_pose_output(
    raw_outputs: dict,
    output_names: List[str],
    conf_thresh: float,
    img_w: int,
    img_h: int,
    infer_w: int = 640,
    infer_h: int = 640,
) -> List[Dict]:
    tensor = raw_outputs.get(output_names[0])
    if tensor is None:
        return []

    arr = np.squeeze(tensor)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T

    if arr.shape[1] < 56:
        return []

    obj_conf = arr[:, 4]
    arr      = arr[obj_conf > conf_thresh]
    if arr.shape[0] == 0:
        return []

    sx = img_w / infer_w
    sy = img_h / infer_h
    cx, cy = arr[:, 0] * sx, arr[:, 1] * sy
    bw, bh = arr[:, 2] * sx, arr[:, 3] * sy
    x1 = (cx - bw / 2).clip(0, img_w)
    y1 = (cy - bh / 2).clip(0, img_h)
    x2 = (cx + bw / 2).clip(0, img_w)
    y2 = (cy + bh / 2).clip(0, img_h)

    results = []
    for i in range(arr.shape[0]):
        kf = arr[i, 6:57] if arr.shape[1] >= 57 else np.zeros(51)
        kf = np.resize(kf, 51).reshape(17, 3)
        kpts_xy   = kf[:, :2] * np.array([[sx, sy]])
        kpts_conf = kf[:, 2]
        results.append({
            "box":       [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
            "conf":      float(arr[i, 4]),
            "kpts_xy":   kpts_xy,
            "kpts_conf": kpts_conf,
        })
    return results


def _preprocess_frame(frame: np.ndarray, target: int = 640) -> np.ndarray:
    resized = cv2.resize(frame, (target, target))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0


# =============================================================================
# NewIntrusionPipeline
# =============================================================================
class NewIntrusionPipeline(BaseVideoPipeline):
    """
    Hardware-accelerated intrusion detection pipeline.
    Registered as "new_intrusion" in core/registry.py.
    Identical API and alert schema to IntrusionDetectionPipeline.
    Uses OpenVINO AsyncInferQueue on Intel iGPU (GPU -> CPU fallback).
    Falls back to Ultralytics CPU inference if openvino is not installed.
    """

    # =========================================================================
    # [NEW_INTRUSION_ENHANCEMENT: Async Inference Engine Setup]
    # Architectural Shift: initialize() compiles the OpenVINO engine once
    #                      so run_on_video() starts with a warm, ready backend.
    # Reason: Eliminates cold-start latency on every stream request.
    # =========================================================================
    def initialize(self, model_weight: str = _DEFAULT_WEIGHT, **kwargs) -> None:
        self._weight_path   = model_weight
        self._use_openvino  = False
        self._ov_compiled   = None
        self._ov_input_name = None
        self._ov_out_names  = []
        self._ov_device     = "CPU"
        self._ppp_enabled   = False
        self._ul_model      = None

        ov, ov_ok = _try_import_openvino()

        if ov_ok:
            xml_path = _ensure_openvino_model(model_weight)
            if xml_path:
                try:
                    compiled, inp_name, out_names, device, ppp_ok = _build_ov_engine(xml_path, ov)
                    self._ov_compiled   = compiled
                    self._ov_input_name = inp_name
                    self._ov_out_names  = out_names
                    self._ov_device     = device
                    self._ppp_enabled   = ppp_ok
                    self._use_openvino  = True
                    print(f"[NewIntrusion] OpenVINO active | device={device} | PPP={ppp_ok}")
                except Exception as exc:
                    print(f"[NewIntrusion] Engine build failed ({exc}). Falling back.")
                    self._use_openvino = False
            else:
                self._use_openvino = False

        if not self._use_openvino:
            print("[NewIntrusion] Using Ultralytics CPU fallback.")
            from ultralytics import YOLO
            self._ul_model = YOLO(model_weight)

    def process_frame(self, frame, frame_idx, roi_polygon, config):
        return frame, {}

    # =========================================================================
    # [NEW_INTRUSION_ENHANCEMENT: AsyncInferQueue Callback]
    # Architectural Shift: Inference results posted to queue.Queue the moment
    #                      the iGPU finishes. Video-read loop never stalls.
    # Reason: Overlaps GPU execution with Python decode + annotation work.
    # =========================================================================
    def _make_async_callback(self, result_q: queue.Queue):
        out_names = self._ov_out_names

        def _cb(infer_request, userdata):
            frame_meta = userdata
            try:
                raw = {n: infer_request.get_output_tensor(i).data
                       for i, n in enumerate(out_names)}
                detections = _decode_pose_output(
                    raw, out_names, conf_thresh=0.45,
                    img_w=frame_meta["img_w"], img_h=frame_meta["img_h"],
                )
                result_q.put({"frame_idx": frame_meta["frame_idx"],
                               "frame": frame_meta["frame"],
                               "detections": detections, "ok": True})
            except Exception as e:
                result_q.put({"frame_idx": frame_meta.get("frame_idx", -1),
                               "frame": frame_meta.get("frame"),
                               "detections": [], "ok": False})
        return _cb

    @staticmethod
    def _annotate_frame(frame, detections, roi_pixels, intrusion_active):
        cv2.polylines(frame, [roi_pixels], isClosed=True, color=(0, 255, 255), thickness=2)
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if intrusion_active:
            cv2.putText(frame, "ZONE BREACH DETECTED", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    @staticmethod
    def _check_intrusion(detections, roi_poly, frame):
        intrusion_in_frame = False
        for det in detections:
            kpts_xy, kpts_conf = det["kpts_xy"], det["kpts_conf"]
            for ankle_idx in [_KP_L_ANKLE, _KP_R_ANKLE]:
                if kpts_conf[ankle_idx] > 0.5:
                    x, y = kpts_xy[ankle_idx]
                    pt   = Point(x, y)
                    if roi_poly.contains(pt):
                        intrusion_in_frame = True
                        cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)
                        cv2.putText(frame, "INTRUSION", (int(x) - 20, int(y) - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)
        return intrusion_in_frame, frame

    def _run_ultralytics_path(self, cap, width, height, fps,
                              roi_pixels, roi_poly, output_dir):
        frame_idx = 0
        start_time = time.time()
        intrusion_active = False
        intrusion_frames_without_detection = 0
        patience = int(fps * 2)
        cooldown_frames_max = int(fps * 10)
        cooldown_frames = 0
        alert_writer = None
        alert_id = None
        alert_start_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if cooldown_frames > 0:
                cooldown_frames -= 1

            results = self._ul_model(frame, classes=[0], conf=0.45, imgsz=640, verbose=False)[0]
            frame   = results.plot()
            cv2.polylines(frame, [roi_pixels], isClosed=True, color=(0, 255, 255), thickness=2)

            intrusion_in_frame = False
            if results.keypoints is not None:
                kpts_xy   = results.keypoints.xy.cpu().numpy()
                kpts_conf = results.keypoints.conf.cpu().numpy()
                for i, person_kpts in enumerate(kpts_xy):
                    conf = kpts_conf[i]
                    for ankle_idx in [_KP_L_ANKLE, _KP_R_ANKLE]:
                        if conf[ankle_idx] > 0.5:
                            x, y = person_kpts[ankle_idx]
                            pt   = Point(x, y)
                            if roi_poly.contains(pt):
                                intrusion_in_frame = True
                                cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)
                                cv2.putText(frame, "INTRUSION", (int(x) - 20, int(y) - 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            else:
                                cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)

            alert_event = None
            if intrusion_in_frame:
                if not intrusion_active and cooldown_frames == 0:
                    intrusion_active  = True
                    cooldown_frames   = cooldown_frames_max
                    alert_id          = str(uuid.uuid4())
                    alert_start_frame = frame_idx
                    alert_path        = os.path.join(output_dir, f"alert_{alert_id}.mp4")
                    elapsed_time      = time.time() - start_time
                    actual_fps        = max(5.0, frame_idx / elapsed_time) if elapsed_time > 0 and frame_idx > 0 else fps
                    alert_writer      = cv2.VideoWriter(alert_path, cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (width, height))
                if intrusion_active:
                    intrusion_frames_without_detection = 0
            else:
                if intrusion_active:
                    intrusion_frames_without_detection += 1
                    if intrusion_frames_without_detection >= patience:
                        intrusion_active = False
                        if alert_writer:
                            alert_writer.release()
                            alert_writer = None
                            ts = alert_start_frame / fps
                            alert_event = {
                                "id": alert_id, "timestamp_sec": ts,
                                "formatted_time": f"{int(ts//60):02d}:{int(ts%60):02d}",
                                "clip_url": f"/storage/alerts/alert_{alert_id}.mp4",
                            }

            if intrusion_active and alert_writer:
                cv2.putText(frame, "ZONE BREACH DETECTED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                alert_writer.write(frame)

            yield frame, alert_event
            frame_idx += 1

        if alert_writer:
            alert_writer.release()
            ts = alert_start_frame / fps
            yield None, {"id": alert_id, "timestamp_sec": ts,
                         "formatted_time": f"{int(ts//60):02d}:{int(ts%60):02d}",
                         "clip_url": f"/storage/alerts/alert_{alert_id}.mp4"}
        cap.release()

    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        """
        Generator: yields (annotated_frame, alert_event_or_None).
        Identical call signature and alert schema to IntrusionDetectionPipeline.
        Works transparently with main.py _mjpeg_generator_analysis() unchanged.
        """
        cap    = get_video_source(input_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

        roi_pixels = np.array(
            [[int(x * width), int(y * height)] for x, y in roi_normalized], np.int32
        )
        roi_poly = Polygon(roi_pixels)

        if not self._use_openvino:
            log.info("[NewIntrusion] Running Ultralytics fallback path.")
            yield from self._run_ultralytics_path(
                cap, width, height, fps, roi_pixels, roi_poly, output_dir
            )
            return

        import openvino as ov

        num_jobs = 4
        result_q: queue.Queue = queue.Queue(maxsize=num_jobs * 2)
        async_queue = ov.AsyncInferQueue(self._ov_compiled, jobs=num_jobs)
        async_queue.set_callback(self._make_async_callback(result_q))

        frame_idx = 0
        start_time = time.time()
        intrusion_active = False
        intrusion_frames_without_detection = 0
        patience = int(fps * 2)
        cooldown_frames_max = int(fps * 10)
        cooldown_frames = 0
        alert_writer = None
        alert_id = None
        alert_start_frame = 0
        in_flight = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if cooldown_frames > 0:
                cooldown_frames -= 1

            # =========================================================================
            # [NEW_INTRUSION_ENHANCEMENT: Frame-Drop Saturation Guard]
            # Architectural Shift: Submit new frame only when async queue has a free
            #                      slot. If saturated, skip submission for this frame
            #                      but still drain ready results so the UI stays live.
            # Reason: Prevents latency accumulation on high-FPS / slow iGPU combos.
            # =========================================================================
            if async_queue.is_ready() or in_flight < num_jobs:
                if self._ppp_enabled:
                    input_tensor = frame[np.newaxis]
                else:
                    input_tensor = _preprocess_frame(frame, 640)

                userdata = {"frame_idx": frame_idx, "frame": frame.copy(),
                            "img_w": width, "img_h": height}
                try:
                    async_queue.start_async({self._ov_input_name: input_tensor}, userdata)
                    in_flight += 1
                except Exception as e:
                    log.debug("[NewIntrusion] start_async error: %s", e)
                    yield frame, None
                    frame_idx += 1
                    continue

            while not result_q.empty():
                result      = result_q.get_nowait()
                in_flight   = max(0, in_flight - 1)
                ann_frame   = result["frame"].copy() if result["frame"] is not None else frame.copy()
                detections  = result["detections"]

                self._annotate_frame(ann_frame, detections, roi_pixels, intrusion_active)
                intrusion_in_frame, ann_frame = self._check_intrusion(detections, roi_poly, ann_frame)

                alert_event = None
                if intrusion_in_frame:
                    if not intrusion_active and cooldown_frames == 0:
                        intrusion_active  = True
                        cooldown_frames   = cooldown_frames_max
                        alert_id          = str(uuid.uuid4())
                        alert_start_frame = result["frame_idx"]
                        alert_path        = os.path.join(output_dir, f"alert_{alert_id}.mp4")
                        elapsed_time      = time.time() - start_time
                        actual_fps        = max(5.0, frame_idx / elapsed_time) if elapsed_time > 0 and frame_idx > 0 else fps
                        alert_writer      = cv2.VideoWriter(alert_path, cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (width, height))
                    if intrusion_active:
                        intrusion_frames_without_detection = 0
                else:
                    if intrusion_active:
                        intrusion_frames_without_detection += 1
                        if intrusion_frames_without_detection >= patience:
                            intrusion_active = False
                            if alert_writer:
                                alert_writer.release()
                                alert_writer = None
                                ts = alert_start_frame / fps
                                alert_event = {
                                    "id": alert_id, "timestamp_sec": ts,
                                    "formatted_time": f"{int(ts//60):02d}:{int(ts%60):02d}",
                                    "clip_url": f"/storage/alerts/alert_{alert_id}.mp4",
                                }

                if intrusion_active and alert_writer:
                    cv2.putText(ann_frame, "ZONE BREACH DETECTED", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    alert_writer.write(ann_frame)

                yield ann_frame, alert_event

            frame_idx += 1

        async_queue.wait_all()
        while not result_q.empty():
            result     = result_q.get_nowait()
            if result["frame"] is not None:
                ann_frame  = result["frame"].copy()
                detections = result["detections"]
                self._annotate_frame(ann_frame, detections, roi_pixels, intrusion_active)
                _, ann_frame = self._check_intrusion(detections, roi_poly, ann_frame)
                yield ann_frame, None

        if alert_writer:
            alert_writer.release()
            ts = alert_start_frame / fps
            yield None, {"id": alert_id, "timestamp_sec": ts,
                         "formatted_time": f"{int(ts//60):02d}:{int(ts%60):02d}",
                         "clip_url": f"/storage/alerts/alert_{alert_id}.mp4"}
        cap.release()
        log.info("[NewIntrusion] Stream ended after %d frames.", frame_idx)
