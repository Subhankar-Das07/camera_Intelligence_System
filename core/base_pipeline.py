from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Generator, Optional
import numpy as np

class BaseVideoPipeline(ABC):
    """Abstract Base Class that every vision use-case must implement."""
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Load weights, initialize model detectors, configure thresholds."""
        pass

    @abstractmethod
    def process_frame(
        self, 
        frame: np.ndarray, 
        frame_idx: int, 
        roi_polygon: np.ndarray, 
        config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame.
        Returns:
            - annotated_frame: frame with visual overlays/drawings.
            - frame_metadata: alert status, detections, counts, etc.
        """
        pass

    @abstractmethod
    def run_on_video(
        self, 
        input_path: str, 
        output_dir: str, 
        roi_polygon: List[Tuple[float, float]], 
        config: Dict[str, Any]
    ) -> Generator[Tuple[np.ndarray, Optional[Dict[str, Any]]], None, None]:
        """Runs the loop over the video, yielding (annotated_frame, optional_alert_event)."""
        pass
