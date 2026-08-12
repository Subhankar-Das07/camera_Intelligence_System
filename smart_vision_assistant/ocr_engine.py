"""
ocr_engine.py
=============
Wrapper for the local OCR engine (PaddleOCR primary, EasyOCR fallback).
Initialises the heavy models once and provides a clean API for extracting text
from image crops (numpy arrays). Runs entirely locally on CPU.
"""

import logging
from typing import List
import numpy as np
import cv2

import config

log = logging.getLogger("ocr_engine")

class OCREngine:
    def __init__(self):
        self._engine = None
        self._engine_type = None
        self._initialise_engine()

    def _initialise_engine(self) -> None:
        """Attempt to load PaddleOCR, fallback to EasyOCR if failed."""
        if config.OCR_PRIMARY_ENGINE.lower() == "paddleocr":
            if self._try_init_paddle():
                return

        if self._try_init_easyocr():
            return

        log.warning("Both PaddleOCR and EasyOCR failed to initialise. OCR disabled.")

    def _try_init_paddle(self) -> bool:
        try:
            # We explicitly disable debug logging for PaddleOCR to avoid spam
            import logging as py_logging
            py_logging.getLogger("ppocr").setLevel(py_logging.ERROR)
            
            from paddleocr import PaddleOCR
            
            log.info("Loading PaddleOCR models (this may take a moment)...")
            self._engine = PaddleOCR(
                use_angle_cls=True, 
                lang='en', 
                use_gpu=False
            )
            self._engine_type = "paddleocr"
            log.info("PaddleOCR loaded successfully.")
            return True
        except ImportError:
            log.warning("PaddleOCR not installed. Run: pip install paddleocr paddlepaddle")
            return False
        except Exception as exc:
            log.warning("PaddleOCR initialisation failed: %s", exc)
            return False

    def _try_init_easyocr(self) -> bool:
        try:
            import easyocr
            log.info("Loading EasyOCR models (this may take a moment)...")
            # gpu=False ensures it runs on CPU
            self._engine = easyocr.Reader(['en'], gpu=False, verbose=False)
            self._engine_type = "easyocr"
            log.info("EasyOCR loaded successfully.")
            return True
        except ImportError:
            log.warning("EasyOCR not installed. Run: pip install easyocr")
            return False
        except Exception as exc:
            log.warning("EasyOCR initialisation failed: %s", exc)
            return False

    def is_available(self) -> bool:
        return self._engine is not None

    def read_text(self, image_crop: np.ndarray) -> List[str]:
        """
        Extract text from an image crop. 
        Returns a list of raw strings found in the image.
        """
        if not self.is_available() or image_crop is None or image_crop.size == 0:
            return []

        h, w = image_crop.shape[:2]
        if w < config.OCR_MIN_BBOX_SIZE or h < config.OCR_MIN_BBOX_SIZE:
            return []

        # -- OpenCV Preprocessing for better OCR accuracy --
        try:
            # 1. Upscale by 2x using cubic interpolation (helps with tiny text)
            upscaled = cv2.resize(image_crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            # 2. Convert to Grayscale
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            # 4. Convert back to BGR since PaddleOCR expects 3 channels
            processed_crop = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            log.warning("OCR preprocessing failed, using original crop: %s", e)
            processed_crop = image_crop

        extracted_texts = []

        try:
            if self._engine_type == "paddleocr":
                # PaddleOCR result format:
                # [[[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('text', confidence)], ...]
                # Sometimes it returns [None] or [[[...]]]
                results = self._engine.ocr(processed_crop, cls=True)
                if not results or not results[0]:
                    return []
                
                for line in results[0]:
                    if not line or len(line) < 2:
                        continue
                    text_info = line[1]
                    text_str = text_info[0]
                    confidence = float(text_info[1])
                    
                    if confidence >= config.OCR_MIN_CONFIDENCE:
                        extracted_texts.append(text_str)

            elif self._engine_type == "easyocr":
                # EasyOCR result format:
                # [([[x,y], [x,y], [x,y], [x,y]], 'text', confidence), ...]
                results = self._engine.readtext(processed_crop)
                for bbox, text_str, confidence in results:
                    if confidence >= config.OCR_MIN_CONFIDENCE:
                        extracted_texts.append(text_str)
                        
        except Exception as exc:
            log.debug("OCR read_text error: %s", exc)

        return extracted_texts
