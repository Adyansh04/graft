"""Per-frame quality checks.

Pure numpy and OpenCV. Two families:

* Frame checks, which apply to any rendered frame.
* Cosmos checks, which compare a restyled frame against the sim frame it
  came from. Labels never pass through Cosmos, so the question is whether
  the geometry they describe survived — the object must be preserved while
  the background actually changes.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    value: float
    threshold: float

    def describe(self) -> str:
        verdict = "ok" if self.passed else "FAIL"
        return f"{self.name}={self.value:.4f} (threshold {self.threshold}) {verdict}"


def blur_variance(image: np.ndarray) -> float:
    """Variance of the Laplacian. Low means the frame is out of focus or
    otherwise smeared."""
    import cv2

    grey = _grey(image)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def check_blur(image: np.ndarray, minimum: float) -> CheckResult:
    value = blur_variance(image)
    return CheckResult("blur_lap_var", value >= minimum, value, minimum)


def check_bbox_area(boxes: list[dict], minimum_px: float) -> CheckResult:
    """Smallest annotated box in the frame.

    A box of a few pixels is not learnable and is usually an object almost
    entirely occluded or off-frame.
    """
    if not boxes:
        return CheckResult("min_bbox_area", True, float("inf"), minimum_px)
    areas = [
        max(0.0, b["x_max"] - b["x_min"]) * max(0.0, b["y_max"] - b["y_min"]) for b in boxes
    ]
    smallest = float(min(areas))
    return CheckResult("min_bbox_area", smallest >= minimum_px, smallest, minimum_px)


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Structural similarity, optionally restricted to a mask.

    Implemented directly rather than pulling in scikit-image for one
    function. Uses global statistics over the compared region, which is
    enough to tell "the object is still there" from "the object changed".
    """
    x = _grey(a).astype(np.float64)
    y = _grey(b).astype(np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")

    if mask is not None:
        selected = mask.astype(bool)
        if not selected.any():
            return 1.0
        x = x[selected]
        y = y[selected]

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx**2 + my**2 + c1) * (vx + vy + c2)
    return float(numerator / denominator) if denominator else 1.0


def check_object_preserved(
    sim_frame: np.ndarray, cosmos_frame: np.ndarray, mask: np.ndarray, minimum: float
) -> CheckResult:
    """Inside the object mask, the restyled frame must still resemble the
    original — otherwise the carried-through label no longer describes it."""
    value = ssim(sim_frame, cosmos_frame, mask)
    return CheckResult("in_mask_ssim", value >= minimum, value, minimum)


def check_background_changed(
    sim_frame: np.ndarray, cosmos_frame: np.ndarray, mask: np.ndarray, minimum: float
) -> CheckResult:
    """Outside the mask the frame should differ, or Cosmos contributed
    nothing and the clip is a duplicate of the sim frame."""
    outside = ~mask.astype(bool)
    if not outside.any():
        return CheckResult("out_mask_change", True, 1.0, minimum)
    a = _grey(sim_frame).astype(np.float64)[outside]
    b = _grey(cosmos_frame).astype(np.float64)[outside]
    value = float(np.abs(a - b).mean() / 255.0)
    return CheckResult("out_mask_change", value >= minimum, value, minimum)


def _grey(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    import cv2

    if image.shape[2] == 4:
        image = image[..., :3]
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
