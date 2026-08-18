import cv2
import numpy as np
from ultralytics import YOLO
import uuid
import os
from collections import deque
from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

class FallDetectionPipeline(BaseVideoPipeline):
    def initialize(self, model_weight: str = "yolov8n-pose.pt"):
        self.model = YOLO(model_weight)
        # Store tracking history: track_id -> deque of (frame_idx, hip_y, w, h)
        self.history = {}
        self.history_frames = 15 # Check history over roughly 0.5 seconds at 30fps

    def process_frame(self, frame, frame_idx, roi_polygon, config):
        pass # Implemented entirely in run_on_video for streaming compatibility

    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        cap = get_video_source(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        frame_idx = 0
        alert_active = False
        patience = int(fps * 2)
        frames_without_detection = 0
        
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

            # Use persist=True for ByteTrack tracking to maintain person IDs across frames
            results = self.model.track(frame, classes=[0], persist=True, verbose=False)[0]
            
            # Base rendering
            frame = results.plot()

            fall_detected_in_frame = False

            if results.boxes is not None and results.boxes.id is not None and results.keypoints is not None:
                track_ids = results.boxes.id.int().cpu().tolist()
                boxes_xywh = results.boxes.xywh.cpu().numpy()
                keypoints_xy = results.keypoints.xy.cpu().numpy()
                keypoints_conf = results.keypoints.conf.cpu().numpy()

                for i, track_id in enumerate(track_ids):
                    # Get box dims
                    bx, by, bw, bh = boxes_xywh[i]
                    
                    # Get hips (11=left, 12=right)
                    conf = keypoints_conf[i]
                    kpts = keypoints_xy[i]
                    
                    hip_y = None
                    if conf[11] > 0.5 and conf[12] > 0.5:
                        hip_y = (kpts[11][1] + kpts[12][1]) / 2.0
                    elif conf[11] > 0.5:
                        hip_y = kpts[11][1]
                    elif conf[12] > 0.5:
                        hip_y = kpts[12][1]
                        
                    if hip_y is None:
                        continue # Can't see hips reliably, skip tracking state

                    ankle_y = None
                    if conf[15] > 0.5 and conf[16] > 0.5:
                        ankle_y = (kpts[15][1] + kpts[16][1]) / 2.0
                    elif conf[15] > 0.5:
                        ankle_y = kpts[15][1]
                    elif conf[16] > 0.5:
                        ankle_y = kpts[16][1]
                    
                    if ankle_y is None:
                        ankle_y = by + bh/2.0 # Fallback to bottom of bounding box

                    if track_id not in self.history:
                        self.history[track_id] = deque(maxlen=self.history_frames)
                        
                    self.history[track_id].append((frame_idx, hip_y, bw, bh))

                    # Check for fall logic
                    history_list = list(self.history[track_id])
                    if len(history_list) >= 5: # Need some history
                        old_frame, old_hip_y, old_w, old_h = history_list[0]
                        curr_frame, curr_hip_y, curr_w, curr_h = history_list[-1]
                        
                        # Calculate downward drop over the history window
                        dy = curr_hip_y - old_hip_y
                        
                        # 1. Sudden drop: hip drops by at least 15% of their standing height within the time window
                        significant_drop = dy > (old_h * 0.15)
                        
                        # 2. Hips on ground: Vertical distance between hips and ankles is very small
                        hip_to_ankle_dist = ankle_y - curr_hip_y
                        hips_on_ground = hip_to_ankle_dist < (old_h * 0.25)
                        
                        if significant_drop and hips_on_ground:
                            fall_detected_in_frame = True
                            
                            # Highlight person box in Red
                            x1, y1 = int(bx - bw/2), int(by - bh/2)
                            x2, y2 = int(bx + bw/2), int(by + bh/2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            cv2.putText(frame, "FALL DETECTED!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            alert_event = None

            if fall_detected_in_frame:
                if not alert_active and cooldown_frames == 0:
                    alert_active = True
                    cooldown_frames = cooldown_frames_max
                    alert_id = str(uuid.uuid4())
                    alert_start_frame = frame_idx
                    alert_path = os.path.join(output_dir, f"alert_{alert_id}.webm")
                    fourcc = cv2.VideoWriter_fourcc(*'vp80')
                    alert_writer = cv2.VideoWriter(alert_path, fourcc, fps, (width, height))
                
                if alert_active:
                    frames_without_detection = 0
            else:
                if alert_active:
                    frames_without_detection += 1
                    if frames_without_detection >= patience:
                        alert_active = False
                        if alert_writer:
                            alert_writer.release()
                            alert_writer = None
                            
                            timestamp_sec = alert_start_frame / fps
                            alert_event = {
                                "id": alert_id,
                                "timestamp_sec": timestamp_sec,
                                "formatted_time": f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}",
                                "clip_url": f"/storage/alerts/alert_{alert_id}.webm",
                                "severity": "SEVERE",
                                "immediate_buzzer_trigger": True
                            }

            if alert_active and alert_writer:
                cv2.putText(frame, "CRITICAL: FALL DETECTED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                alert_writer.write(frame)

            yield frame, alert_event
            frame_idx += 1

        if alert_writer:
            alert_writer.release()
            timestamp_sec = alert_start_frame / fps
            yield None, {
                "id": alert_id,
                "timestamp_sec": timestamp_sec,
                "formatted_time": f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}",
                "clip_url": f"/storage/alerts/alert_{alert_id}.webm"
            }

        cap.release()
