"""
snapshot_engine.py
==================
Captures and persists image crops for tracked entities.
Maintains first seen, highest confidence, and last seen images.
"""

import os
import cv2
import queue
import threading
import logging
from pathlib import Path
from typing import Optional, Dict

import numpy as np

log = logging.getLogger("snapshot_engine")

class SnapshotEngine:
    def __init__(self, session_id: int):
        self.session_id = session_id
        self.base_dir = Path("data") / "entity_snapshots" / str(session_id)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue = queue.Queue(maxsize=1000)
        
        # Track highest confidence per entity to only write 'best' when improved
        self._best_confidences: Dict[str, float] = {}
        # In-memory buffer of the latest crop per entity to write on removal
        self._last_crops: Dict[str, np.ndarray] = {}
        
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="SnapshotWorker")
        self._thread.start()
        log.info("SnapshotEngine started for session %d", self.session_id)
        
    def stop(self) -> None:
        self._running = False
        if self._thread:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=2.0)
            
        # Flush any remaining last crops to disk
        for entity_id, crop in self._last_crops.items():
            self._save_crop(entity_id, "last", crop)
        self._last_crops.clear()
        
        log.info("SnapshotEngine stopped.")

    def process_detection(self, entity_id: str, crop: np.ndarray, confidence: float, is_new: bool) -> None:
        """Called for every detection to update snapshot state."""
        if not self._running:
            return
            
        try:
            self._queue.put_nowait(("update", entity_id, crop.copy(), confidence, is_new))
        except queue.Full:
            pass # Skip if overloaded
            
    def process_removal(self, entity_id: str) -> None:
        """Called when an entity is removed from the scene."""
        if not self._running:
            return
            
        try:
            self._queue.put_nowait(("remove", entity_id, None, 0.0, False))
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:
                    continue
                    
                action, entity_id, crop, confidence, is_new = item
                
                if action == "update":
                    # 1. Update in-memory 'last' crop
                    self._last_crops[entity_id] = crop
                    
                    # 2. Save 'first' crop if new
                    if is_new:
                        self._save_crop(entity_id, "first", crop)
                        self._best_confidences[entity_id] = confidence
                        self._save_crop(entity_id, "best", crop)
                    else:
                        # 3. Update 'best' crop if confidence improved
                        best_conf = self._best_confidences.get(entity_id, 0.0)
                        if confidence > best_conf:
                            self._best_confidences[entity_id] = confidence
                            self._save_crop(entity_id, "best", crop)
                            
                elif action == "remove":
                    # Write the last buffered crop to disk and cleanup memory
                    last_crop = self._last_crops.pop(entity_id, None)
                    if last_crop is not None:
                        self._save_crop(entity_id, "last", last_crop)
                    self._best_confidences.pop(entity_id, None)
                    
            except queue.Empty:
                pass
            except Exception as e:
                log.error("Snapshot worker error: %s", e)

    def _save_crop(self, entity_id: str, suffix: str, crop: np.ndarray) -> None:
        if crop is None or crop.size == 0:
            return
        filename = f"{entity_id}_{suffix}.jpg"
        filepath = self.base_dir / filename
        # Compress JPEG to save space
        cv2.imwrite(str(filepath), crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
