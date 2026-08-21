import cv2
import numpy as np
import re
import logging
from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger(__name__)


class PlateReader:
    """
    License plate OCR parser and image preprocessor.
    Validates crop quality, preprocesses the image for better OCR results,
    and extracts alphanumeric plate characters using RapidOCR with post-processing.
    """

    def __init__(self, blur_threshold: float = 50.0, min_width: int = 60, min_height: int = 20):
        """
        Initialize the PlateReader with OCR engine and quality thresholds.

        Args:
            blur_threshold (float): Minimum Laplacian variance for an image to be considered sharp.
            min_width (int): Minimum width of the plate crop in pixels.
            min_height (int): Minimum height of the plate crop in pixels.
        """
        self.blur_threshold = blur_threshold
        self.min_width = min_width
        self.min_height = min_height

        # Initialize RapidOCR — lightweight ONNX-based drop-in replacement for PaddleOCR
        try:
            self.ocr = RapidOCR()
            logger.info("RapidOCR initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RapidOCR: {e}")
            raise

    def is_valid_crop(self, plate_crop: np.ndarray) -> bool:
        """
        Assess the quality of the plate crop based on dimensions and blurriness.

        Args:
            plate_crop (np.ndarray): The cropped license plate image (BGR format).

        Returns:
            bool: True if the crop is sharp and large enough, False otherwise.
        """
        if plate_crop is None or plate_crop.size == 0:
            return False

        h, w = plate_crop.shape[:2]
        if w < self.min_width or h < self.min_height:
            logger.debug(f"Crop rejected due to size: {w}x{h} (min: {self.min_width}x{self.min_height})")
            return False

        # Calculate Laplacian variance to assess sharpness
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        if variance < self.blur_threshold:
            logger.debug(f"Crop rejected due to blurriness: var={variance:.2f} (threshold: {self.blur_threshold})")
            return False

        return True

    def preprocess_for_ocr(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess the plate image to enhance character contrast and reduce noise.

        Args:
            plate_crop (np.ndarray): The cropped license plate image (BGR format).

        Returns:
            np.ndarray: Preprocessed BGR image ready for OCR.
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to denoise while keeping edges sharp
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # RapidOCR works best on 3-channel images — convert back to BGR
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _disambiguate_characters(self, text: str) -> str:
        """
        Apply positional character disambiguation for standard vehicle plates.
        Corrects common OCR mistakes like confusing '0' and 'O', '8' and 'B', etc.

        Args:
            text (str): Raw OCR text.

        Returns:
            str: Positively disambiguated text.
        """
        letter_to_digit = {'O': '0', 'D': '0', 'B': '8', 'I': '1', 'L': '1', 'S': '5', 'Z': '2'}
        digit_to_letter = {'0': 'O', '8': 'B', '1': 'I', '5': 'S', '2': 'Z'}

        if len(text) < 6:
            return text

        chars = list(text)

        # State code: first 2 are always letters
        for i in range(min(2, len(chars))):
            if chars[i] in digit_to_letter:
                chars[i] = digit_to_letter[chars[i]]

        # Number: last 4 are always digits
        for i in range(max(0, len(chars) - 4), len(chars)):
            if chars[i] in letter_to_digit:
                chars[i] = letter_to_digit[chars[i]]

        # Handle the middle part (RTO code + Series) for standard 10-char plates (e.g., OD02DB9032)
        if len(chars) == 10:
            # Indices 2, 3 should be digits (RTO)
            for i in range(2, 4):
                if chars[i] in letter_to_digit:
                    chars[i] = letter_to_digit[chars[i]]
            # Indices 4, 5 should be letters (Series)
            for i in range(4, 6):
                if chars[i] in digit_to_letter:
                    chars[i] = digit_to_letter[chars[i]]

        return "".join(chars)

    def read_plate(self, plate_crop: np.ndarray) -> tuple[str, float]:
        """
        Extract and validate the license plate text from the crop.

        Args:
            plate_crop (np.ndarray): The cropped license plate image.

        Returns:
            tuple[str, float]: Cleaned license plate text and confidence score.
                               Returns ("", 0.0) if validation or OCR fails.
        """
        if not self.is_valid_crop(plate_crop):
            return "", 0.0

        try:
            preprocessed_img = self.preprocess_for_ocr(plate_crop)

            # Run RapidOCR inference
            # Returns: result (list of [bbox, text, score] or None), elapse (float)
            result, elapse = self.ocr(preprocessed_img)

            if not result:
                return "", 0.0

            # Find the result with the highest confidence
            best_text = ""
            best_conf = 0.0

            for item in result:
                # RapidOCR result item: [bbox_points, text, confidence_score]
                if not item or len(item) < 3:
                    continue
                text = item[1]
                conf = float(item[2]) if item[2] is not None else 0.0
                if text and conf > best_conf:
                    best_text = text
                    best_conf = conf

            if not best_text:
                return "", 0.0

            # Strip spaces and special characters, convert to uppercase
            cleaned_text = re.sub(r'[^A-Za-z0-9]', '', best_text).upper()

            # Apply positional disambiguation
            disambiguated_text = self._disambiguate_characters(cleaned_text)

            # Apply Regex Validation
            # Format: 2 Letters + 1-2 Digits + 1-3 Letters + 4 Digits
            pattern = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$')
            if not pattern.match(disambiguated_text):
                logger.debug(f"Plate '{disambiguated_text}' failed regex validation.")
                return "", 0.0

            return disambiguated_text, float(best_conf)

        except Exception as e:
            logger.error(f"Error during OCR processing: {e}")
            return "", 0.0
