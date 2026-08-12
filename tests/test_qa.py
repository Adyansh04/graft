import numpy as np
import pytest

from graft.qa import checks

RNG = np.random.default_rng(0)


def sharp(size=64):
    """High-frequency content, so the Laplacian variance is large."""
    return (RNG.random((size, size, 3)) * 255).astype(np.uint8)


def flat(size=64, value=128):
    return np.full((size, size, 3), value, dtype=np.uint8)


# --- blur ---


def test_sharp_frame_passes():
    assert checks.check_blur(sharp(), 60.0).passed


def test_flat_frame_is_rejected():
    result = checks.check_blur(flat(), 60.0)
    assert not result.passed
    assert result.value < 60.0


def test_blurred_frame_scores_lower_than_the_original():
    import cv2

    original = sharp()
    blurred = cv2.GaussianBlur(original, (9, 9), 5)
    assert checks.blur_variance(blurred) < checks.blur_variance(original)


# --- bbox area ---


def test_tiny_box_is_rejected():
    boxes = [{"x_min": 0, "y_min": 0, "x_max": 3, "y_max": 3}]
    assert not checks.check_bbox_area(boxes, 64).passed


def test_reasonable_box_passes():
    boxes = [{"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100}]
    assert checks.check_bbox_area(boxes, 64).passed


def test_smallest_box_decides():
    boxes = [
        {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100},
        {"x_min": 0, "y_min": 0, "x_max": 2, "y_max": 2},
    ]
    assert not checks.check_bbox_area(boxes, 64).passed


def test_frame_with_no_boxes_is_not_rejected():
    """Negatives are legitimate training data."""
    assert checks.check_bbox_area([], 64).passed


# --- ssim ---


def test_identical_images_score_one():
    image = sharp()
    assert checks.ssim(image, image) == pytest.approx(1.0, abs=1e-9)


def test_different_images_score_lower():
    assert checks.ssim(sharp(), sharp()) < 0.5


def test_mask_restricts_comparison():
    """The point of the masked comparison: the object region can be
    identical while the background is completely different."""
    a = flat(64, 100)
    b = flat(64, 100)
    b[32:, :] = 250

    mask = np.zeros((64, 64), dtype=bool)
    mask[:32, :] = True

    assert checks.ssim(a, b, mask) == pytest.approx(1.0, abs=1e-6)
    assert checks.ssim(a, b) < 1.0


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="shape mismatch"):
        checks.ssim(sharp(32), sharp(64))


# --- cosmos geometry checks ---


def test_preserved_object_passes():
    sim = sharp()
    restyled = sim.copy()
    restyled[32:, :] = 0  # only the background changed
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32, :] = True

    assert checks.check_object_preserved(sim, restyled, mask, 0.7).passed


def test_altered_object_is_rejected():
    """If Cosmos changed the object, the carried-through label no longer
    describes what is in the frame."""
    sim = sharp()
    restyled = sim.copy()
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32, :] = True
    restyled[:32, :] = (RNG.random((32, 64, 3)) * 255).astype(np.uint8)

    assert not checks.check_object_preserved(sim, restyled, mask, 0.7).passed


def test_unchanged_background_is_rejected():
    """A restyle that changed nothing means Cosmos contributed nothing."""
    sim = sharp()
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32, :] = True
    assert not checks.check_background_changed(sim, sim.copy(), mask, 0.05).passed


def test_changed_background_passes():
    sim = flat(64, 100)
    restyled = sim.copy()
    restyled[32:, :] = 200
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32, :] = True
    assert checks.check_background_changed(sim, restyled, mask, 0.05).passed


def test_full_frame_mask_does_not_divide_by_zero():
    sim = sharp()
    mask = np.ones((64, 64), dtype=bool)
    assert checks.check_background_changed(sim, sim, mask, 0.05).passed
