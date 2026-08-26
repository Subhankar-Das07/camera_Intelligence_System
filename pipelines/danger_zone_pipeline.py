import cv2
import time
import numpy as np
from ultralytics import YOLO
from shapely.geometry import box, Polygon, Point
import uuid
import os
from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

class DangerZonePipeline(BaseVideoPipeline):
    def __init__(self):
        self.model = None

    def initialize(self, **kwargs):
        self.model = YOLO('yolov8n-pose.pt')

    def process_frame(self, frame, frame_idx, roi_polygon, config):
        pass

    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        cap = get_video_source(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Denormalize ROI polygon
        roi_pixels = np.array([[int(x * width), int(y * height)] for x, y in roi_normalized], np.int32)
        roi_poly = Polygon(roi_pixels)

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
        alert_severity = None

        machine_active = config.get("machine_active", False)
        buffer_distance_pixels = config.get("buffer_distance_pixels", 20)
        
        # Calculate the buffered polygon once for rendering
        buffered_poly = roi_poly.buffer(buffer_distance_pixels)
        buffered_pixels = np.array(buffered_poly.exterior.coords, dtype=np.int32)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if cooldown_frames > 0:
                cooldown_frames -= 1

            results = self.model(frame, classes=[0], conf=0.45, imgsz=640, verbose=False)[0]
            
            # Natively draw skeleton over all people
            frame = results.plot()
            
            current_severity = None

            if results.keypoints is not None:
                keypoints_xy = results.keypoints.xy.cpu().numpy()
                keypoints_conf = results.keypoints.conf.cpu().numpy()

                for i, person_kpts in enumerate(keypoints_xy):
                    conf = keypoints_conf[i]
                    
                    person_breached = False
                    
                    # Check all 17 keypoints (which inherently includes wrists 9, 10 and ankles)
                    for kpt_idx, (x, y) in enumerate(person_kpts):
                        if conf[kpt_idx] > 0.5:
                            pt = Point(x, y)
                            
                            if machine_active:
                                # Distance logic handles both intersection and proximity buffer
                                dist = roi_poly.distance(pt)
                                if dist <= buffer_distance_pixels:
                                    person_breached = True
                                    # Highlight offending body part with a bright Red circle
                                    cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)
                                else:
                                    # Safe keypoints
                                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

                    if person_breached:
                        current_severity = "SEVERE"

            # Draw ROI depending on state
            if not machine_active:
                roi_color = (128, 128, 128) # Gray
            elif current_severity == "SEVERE":
                # Flashing red effect based on frame index
                roi_color = (0, 0, 255) if frame_idx % int(fps/2) < int(fps/4) else (0, 0, 100)
            else:
                roi_color = (0, 255, 255) # Yellow

            # Draw base ROI polygon
            cv2.polylines(frame, [roi_pixels], isClosed=True, color=roi_color, thickness=3)
            
            # Draw outer buffer line
            if machine_active:
                cv2.polylines(frame, [buffered_pixels], isClosed=True, color=(0, 165, 255), thickness=1, lineType=cv2.LINE_AA)

            if current_severity == "SEVERE":
                cv2.putText(frame, "SEVERE DANGER: BUFFER BREACHED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            alert_event = None

            if current_severity:
                if not intrusion_active and cooldown_frames == 0:
                    intrusion_active = True
                    alert_severity = current_severity
                    cooldown_frames = cooldown_frames_max
                    alert_id = str(uuid.uuid4())
                    alert_start_frame = frame_idx
                    alert_path = os.path.join(output_dir, f"alert_{alert_id}.webm")
                    # Dynamically calculate actual processing FPS to prevent fast-forwarding
                    elapsed_time = time.time() - start_time
                    actual_fps = max(5.0, frame_idx / elapsed_time) if elapsed_time > 0 and frame_idx > 0 else fps
                    fourcc = cv2.VideoWriter_fourcc(*"vp80") if "'" not in "vp80" else cv2.VideoWriter_fourcc(*"vp80")
                    alert_writer = cv2.VideoWriter(alert_path, cv2.VideoWriter_fourcc(*"vp80"), actual_fps, (width, height))
                elif intrusion_active:
                    if current_severity == "SEVERE":
                        alert_severity = "SEVERE"
                
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
                            
                            timestamp_sec = alert_start_frame / fps
                            alert_event = {
                                "id": alert_id,
                                "timestamp_sec": timestamp_sec,
                                "formatted_time": f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}",
                                "clip_url": f"/storage/alerts/alert_{alert_id}.webm",
                                "immediate_buzzer_trigger": (alert_severity == "SEVERE"),
                                "severity": alert_severity
                            }

            if intrusion_active and alert_writer:
                if (frame_idx - alert_start_frame) < int(fps * 4):
                    alert_writer.write(frame)
                else:
                    alert_writer.release()
                    alert_writer = None
                    timestamp_sec = alert_start_frame / fps
                    alert_event = {
                        "id": alert_id,
                        "timestamp_sec": timestamp_sec,
                        "formatted_time": f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}",
                        "clip_url": f"/storage/alerts/alert_{alert_id}.webm",
                        "immediate_buzzer_trigger": (alert_severity == "SEVERE"),
                        "severity": alert_severity
                    }

            yield frame, alert_event
            frame_idx += 1

        if alert_writer:
            alert_writer.release()
            timestamp_sec = alert_start_frame / fps
            yield None, {
                "id": alert_id,
                "timestamp_sec": timestamp_sec,
                "formatted_time": f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}",
                "clip_url": f"/storage/alerts/alert_{alert_id}.webm",
                "immediate_buzzer_trigger": (alert_severity == "SEVERE"),
                "severity": alert_severity
            }

        cap.release()

