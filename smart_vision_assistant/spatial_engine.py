"""
spatial_engine.py
=================
Pure-math spatial intelligence engine.
Evaluates relationships (Near, Overlapping) using bounding box heuristics.
Zero CPU impact compared to ML-based scene graph generation.
"""

import math
from typing import Tuple

BoundingBox = Tuple[int, int, int, int] # x1, y1, x2, y2

def get_centroid(box: BoundingBox) -> Tuple[float, float]:
    """Returns the (x, y) center of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def euclidean_distance(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculates pixel distance between two bounding box centroids."""
    c1 = get_centroid(box1)
    c2 = get_centroid(box2)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

def intersection_area(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculates the raw intersection area of two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    return float((x2_inter - x1_inter) * (y2_inter - y1_inter))

def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculates Intersection over Union (IoU) for two bounding boxes."""
    inter_area = intersection_area(box1, box2)
    if inter_area == 0.0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def is_near(box1: BoundingBox, box2: BoundingBox, threshold: float = 150.0) -> bool:
    """Determines if two objects are near each other based on a fixed pixel threshold."""
    return euclidean_distance(box1, box2) <= threshold

def is_near_adaptive(box1: BoundingBox, box2: BoundingBox) -> bool:
    """Determines if two objects are near each other using size-adaptive thresholds."""
    w1 = max(1, box1[2] - box1[0])
    w2 = max(1, box2[2] - box2[0])
    threshold = max(w1, w2) * 1.5
    return euclidean_distance(box1, box2) <= threshold

def is_overlapping(box1: BoundingBox, box2: BoundingBox, iou_threshold: float = 0.05) -> bool:
    """Determines if two objects overlap."""
    return calculate_iou(box1, box2) >= iou_threshold

def is_inside(box1: BoundingBox, box2: BoundingBox, threshold: float = 0.85) -> bool:
    """True if box1 is heavily contained within box2."""
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    if area1 == 0: return False
    return (intersection_area(box1, box2) / area1) > threshold

def is_on_top_of(box1: BoundingBox, box2: BoundingBox) -> bool:
    """True if box1 overlaps box2 AND box1's center is vertically higher than box2's."""
    if not is_overlapping(box1, box2, iou_threshold=0.05):
        return False
    c1 = get_centroid(box1)
    c2 = get_centroid(box2)
    # y-axis increases downwards in image coordinates, so c1_y < c2_y means higher
    return c1[1] < c2[1]
