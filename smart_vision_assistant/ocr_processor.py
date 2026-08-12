"""
ocr_processor.py
================
Async OCR worker thread. Processes image crops asynchronously, calls the OCR engine,
updates the text memory accumulator, and feeds the results to the EntityRegistry.
"""

import threading
import queue
import time
import logging
from typing import Dict, Any, Optional

import numpy as np

import config
from ocr_engine import OCREngine
from ocr_text_memory import OCRTextMemory
from product_knowledge import lookup_brand
from entity_inference import infer_entity_identity

log = logging.getLogger("ocr_processor")

class OCRProcessor:
    def __init__(self, entity_registry, db_manager, session_id: int):
        self._registry = entity_registry
        self._db = db_manager
        self._session_id = session_id
        self._queue = queue.Queue(maxsize=config.OCR_MAX_QUEUE_SIZE)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._engine: Optional[OCREngine] = None
        self._memory = OCRTextMemory()
        
        # Telemetry
        self.total_submitted = 0
        self.total_completed = 0
        self.total_text_found = 0
        self.total_entities_enriched = 0
        self.avg_ocr_ms = 0.0
        
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="OCRWorker")
        self._thread.start()
        log.info("OCR worker thread started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            # push dummy item to unblock queue.get()
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=2.0)
        log.info("OCR worker thread stopped")

    def submit_crop(self, entity_id: str, crop: np.ndarray, yolo_label: str) -> bool:
        """
        Submit a crop to the OCR queue.
        Returns False if the queue is full (which protects the detection loop).
        """
        if not self._running:
            return False
            
        try:
            self.total_submitted += 1
            self._queue.put_nowait((entity_id, crop, yolo_label))
            return True
        except queue.Full:
            return False

    def _worker_loop(self) -> None:
        # Initialise heavy OCR engine inside the worker thread 
        # so it doesn't block startup of the main app.
        self._engine = OCREngine()
        
        if not self._engine.is_available():
            log.warning("OCR Engine unavailable. Worker thread halting.")
            return

        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:
                    continue  # Stop signal
                    
                entity_id, crop, yolo_label = item
                
                # 1. Read text from crop
                start_time = time.monotonic()
                raw_texts = self._engine.read_text(crop)
                self.total_completed += 1
                
                if not raw_texts:
                    continue
                    
                self.total_text_found += 1
                    
                # 2. Add to historical text memory
                best_text_changed = self._memory.add_result(entity_id, raw_texts)
                
                if best_text_changed:
                    self.total_entities_enriched += 1
                    best_text = self._memory.get_best_text(entity_id)
                    all_texts = self._memory.get_all_texts(entity_id)
                    
                    # 3. Consult Product Knowledge Base
                    kb_match = lookup_brand(best_text)
                    brand = kb_match["brand"] if kb_match else ""
                    product_type = kb_match["product_type"] if kb_match else ""
                    
                    # 4. Evidence Fusion Layer
                    inferred_label = infer_entity_identity(yolo_label, best_text, kb_match)
                    
                    # 5. Push results back to Entity Registry (thread-safe dict update)
                    self._registry.update_ocr_result(
                        entity_id=entity_id,
                        detected_texts=all_texts,
                        best_text=best_text,
                        brand=brand,
                        product_type=product_type,
                        inferred_label=inferred_label
                    )
                    
                    # 6. Database log
                    entity = self._registry._session.get(entity_id)
                    if entity:
                        self._db.on_ocr_result(
                            session_id=self._session_id,
                            entity_id=entity_id,
                            track_id=entity.track_id,
                            raw_texts=all_texts,
                            best_text=best_text,
                            brand=brand,
                            inferred_label=inferred_label
                        )
                    
                    log.info("[OCR] Updated %s: brand='%s' best_text='%s' inferred='%s'", 
                             yolo_label, brand, best_text, inferred_label)
                             
                # Update latency
                duration_ms = (time.monotonic() - start_time) * 1000
                if self.avg_ocr_ms == 0.0:
                    self.avg_ocr_ms = duration_ms
                else:
                    self.avg_ocr_ms = self.avg_ocr_ms * 0.9 + duration_ms * 0.1
                             
            except queue.Empty:
                pass
            except Exception as exc:
                log.error("OCR worker error: %s", exc, exc_info=True)

    def get_telemetry(self) -> dict:
        return {
            "crops_submitted": self.total_submitted,
            "tasks_completed": self.total_completed,
            "text_found": self.total_text_found,
            "entities_enriched": self.total_entities_enriched,
            "queue_size": self._queue.qsize(),
            "avg_latency_ms": round(self.avg_ocr_ms, 1)
        }
