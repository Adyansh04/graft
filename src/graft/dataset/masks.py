"""Instance masks to polygons.

Pure OpenCV, no Isaac Sim. An instance split by an occluder produces several
disconnected regions; Ultralytics takes one polygon per line, so each region
becomes its own line rather than being merged or silently dropped.
"""

import numpy as np

# Fraction of the frame below which a region is treated as noise.
MIN_REGION_AREA_FRACTION = 1e-5

# Simplification tolerance as a fraction of contour perimeter.
POLYGON_EPSILON_FRACTION = 0.005

MAX_POLYGON_POINTS = 100


def mask_to_polygons(
    mask: np.ndarray,
    instance_id: int,
    *,
    min_area_fraction: float = MIN_REGION_AREA_FRACTION,
    max_points: int = MAX_POLYGON_POINTS,
) -> list[tuple[float, ...]]:
    """Outlines for one instance, in pixel coordinates.

    Returns every region above the area threshold, not just the largest —
    dropping the rest loses genuinely visible parts of an occluded object.
    """
    import cv2

    binary = (mask == instance_id).astype(np.uint8)
    if not binary.any():
        return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(mask.shape[0] * mask.shape[1])
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < frame_area * min_area_fraction:
            continue
        simplified = _simplify(contour, max_points)
        if len(simplified) < 3:
            continue
        polygons.append(tuple(float(v) for point in simplified for v in point))
    return polygons


def _simplify(contour, max_points: int):
    import cv2

    epsilon = POLYGON_EPSILON_FRACTION * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    while len(simplified) > max_points:
        epsilon *= 1.5
        simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return simplified


def mask_to_box(mask: np.ndarray, instance_id: int) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask == instance_id)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
