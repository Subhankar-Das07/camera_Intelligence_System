import cv2
import threading
import time
import logging
import os

log = logging.getLogger(__name__)

class ThreadedCamera:
    """
    A unified video source that implements a FrameBuffer.
    For RTSP and live streams, it continuously reads frames in a background thread
    and always yields the most recent frame, dropping intermediate frames to prevent lag.
    """
    def __init__(self, src):
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.src = src
        
        self.cap = cv2.VideoCapture(self.src)
            
        print(f"DEBUG: isOpened = {self.cap.isOpened()}")
        self.grabbed, self.frame = self.cap.read()
        print(f"DEBUG: read() returned grabbed={self.grabbed}, frame={'None' if self.frame is None else 'Valid'}")
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None
        self._stop_event = threading.Event()

    def start(self):
        if self.started:
            log.warning("ThreadedCamera is already started.")
            return self
        
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while not self._stop_event.is_set():
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            
            if not grabbed:
                # If the stream ends or fails, we exit the thread
                break
                
            # Yield control slightly to avoid choking the CPU if camera reads too fast
            time.sleep(0.005)

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame

    def grab(self):
        return self.grabbed

    def retrieve(self):
        return self.read()

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, propId):
        return self.cap.get(propId)

    def release(self):
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        self.started = False
        self.cap.release()

def get_video_source(input_path):
    """
    Returns a video source object. If input_path is already a camera object, returns it directly.
    For URLs, returns a background-threaded camera to ensure frames don't buffer and cause lag.
    """
    if hasattr(input_path, 'read') and hasattr(input_path, 'isOpened'):
        return input_path
        
    if isinstance(input_path, str) and input_path.isdigit():
        input_path = int(input_path)

    if str(input_path).startswith(('rtsp://', 'http://', 'https://')):
        cam = ThreadedCamera(input_path)
        return cam.start()
    else:
        return cv2.VideoCapture(input_path)
