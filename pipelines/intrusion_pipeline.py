import cv2
import time
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import os
from core.base_pipeline import BaseVideoPipeline
from core.video_source import get_video_source

class IntrusionDetectionPipeline(BaseVideoPipeline):
    def initialize(self, model_weight: str = "yolov8n-pose.pt"):
        self.model = YOLO(model_weight)

    def process_frame(self, frame: np.ndarray, frame_idx: int, roi_polygon: np.ndarray, config: dict) -> tuple:
        pass # Implementation is done mostly in run_on_video according to user prompt

    def run_on_video(self, input_path, output_dir, roi_normalized, config):
        import uuid
        
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
        patience = int(fps * 2) # Wait 2 seconds before closing clip
        
        cooldown_frames_max = int(fps * 10) # 10 second cooldown
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

            results = self.model(frame, classes=[0], conf=0.45, imgsz=640, verbose=False)[0]
            
            # Draw skeleton over the person natively
            frame = results.plot()
            
            # Draw boundary
            cv2.polylines(frame, [roi_pixels], isClosed=True, color=(0, 255, 255), thickness=2)

            intrusion_in_frame = False

            if results.keypoints is not None:
                keypoints_xy = results.keypoints.xy.cpu().numpy()
                keypoints_conf = results.keypoints.conf.cpu().numpy()

                for i, person_kpts in enumerate(keypoints_xy):
                    conf = keypoints_conf[i]
                    
                    # Extract ankles (Index 15 = Left Ankle, Index 16 = Right Ankle)
                    for ankle_idx in [15, 16]:
                        if conf[ankle_idx] > 0.5:
                            x, y = person_kpts[ankle_idx]
                            pt = Point(x, y)
                            
                            if roi_poly.contains(pt):
                                intrusion_in_frame = True
                                # Highlight breaching ankle in Red
                                cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)
                                cv2.putText(frame, "INTRUSION", (int(x) - 20, int(y) - 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            else:
                                # Highlight safe ankle in Green
                                cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)

            alert_event = None

            if intrusion_in_frame:
                if not intrusion_active and cooldown_frames == 0:
                    intrusion_active = True
                    cooldown_frames = cooldown_frames_max
                    alert_id = str(uuid.uuid4())
                    alert_start_frame = frame_idx
                    alert_path = os.path.join(output_dir, f"alert_{alert_id}.mp4")
                    # Dynamically calculate actual processing FPS to prevent fast-forwarding
                    elapsed_time = time.time() - start_time
                    actual_fps = max(5.0, frame_idx / elapsed_time) if elapsed_time > 0 and frame_idx > 0 else fps
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if "'" not in "mp4v" else cv2.VideoWriter_fourcc(*"mp4v")
                    alert_writer = cv2.VideoWriter(alert_path, cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (width, height))
                
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
                            
                            # Fire alert event
                            timestamp_sec = alert_start_frame / fps
                            alert_event = {
                                "id": alert_id,
                                "timestamp_sec": timestamp_sec,
                                "formatted_time": f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}",
                                "clip_url": f"/storage/alerts/alert_{alert_id}.mp4"
                            }

            if intrusion_active and alert_writer:
                cv2.putText(frame, "ZONE BREACH DETECTED", (30, 50),
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
                "clip_url": f"/storage/alerts/alert_{alert_id}.mp4"
            }

        cap.release()

