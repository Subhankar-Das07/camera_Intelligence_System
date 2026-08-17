import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import box, Polygon
import uuid
import os
from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

class DangerZonePipeline(BaseVideoPipeline):
    def __init__(self):
        self.model = None

    def initialize(self, **kwargs):
        self.model = YOLO('yolov8n.pt')

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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if cooldown_frames > 0:
                cooldown_frames -= 1

            results = self.model(frame, classes=[0], verbose=False)[0]
            current_severity = None

            for b in results.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0])
                
                person_box = box(x1, y1, x2, y2)

                if machine_active and roi_poly.intersects(person_box):
                    intersection = roi_poly.intersection(person_box)
                    ratio = intersection.area / person_box.area
                    
                    if ratio >= 0.10: # Severe
                        current_severity = "SEVERE"
                        color = (0, 0, 255) # Red for danger
                        cv2.putText(frame, f"SEVERE: {conf:.2f}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else: # Near
                        if current_severity != "SEVERE":
                            current_severity = "WARNING"
                        color = (0, 165, 255) # Orange
                        cv2.putText(frame, f"NEAR: {conf:.2f}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    color = (0, 255, 0) # Green if safe or machine off

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw ROI depending on state
            if not machine_active:
                roi_color = (128, 128, 128) # Gray
            elif current_severity == "SEVERE":
                # Flashing red effect based on frame index
                roi_color = (0, 0, 255) if frame_idx % int(fps/2) < int(fps/4) else (0, 0, 100)
            elif current_severity == "WARNING":
                roi_color = (0, 165, 255) # Orange
            else:
                roi_color = (0, 255, 255) # Yellow

            cv2.polylines(frame, [roi_pixels], isClosed=True, color=roi_color, thickness=3)

            if current_severity == "SEVERE":
                cv2.putText(frame, "SEVERE DANGER: SHUTDOWN TRIGGERED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            elif current_severity == "WARNING":
                cv2.putText(frame, "WARNING: PERSON NEAR MACHINE", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

            alert_event = None

            if current_severity:
                if not intrusion_active and cooldown_frames == 0:
                    intrusion_active = True
                    alert_severity = current_severity
                    cooldown_frames = cooldown_frames_max
                    alert_id = str(uuid.uuid4())
                    alert_start_frame = frame_idx
                    alert_path = os.path.join(output_dir, f"alert_{alert_id}.webm")
                    fourcc = cv2.VideoWriter_fourcc(*'vp80')
                    alert_writer = cv2.VideoWriter(alert_path, fourcc, fps, (width, height))
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
