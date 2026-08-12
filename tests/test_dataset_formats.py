import numpy as np
import pytest
import yaml

from graft.dataset.formats import get_format
from graft.dataset.formats.base import Box, Frame
from graft.dataset.masks import mask_to_box, mask_to_polygons

# Import for registration side effects.
import graft.dataset.formats.yolo_detect  # noqa: F401
import graft.dataset.formats.yolo_seg  # noqa: F401

W, H = 1280, 704


def frame(**overrides) -> Frame:
    base = dict(image_path="f.png", width=W, height=H)
    base.update(overrides)
    return Frame(**base)


# --- detect ---


def test_box_converts_to_normalised_centre_form(tmp_path):
    dest = tmp_path / "f.txt"
    get_format("yolo_detect").write_labels(
        frame(boxes=(Box(0, 0.0, 0.0, W / 2, H / 2),)), dest
    )
    class_id, cx, cy, w, h = dest.read_text().split()
    assert int(class_id) == 0
    assert (float(cx), float(cy)) == pytest.approx((0.25, 0.25))
    assert (float(w), float(h)) == pytest.approx((0.5, 0.5))


def test_box_overhanging_the_frame_is_clamped(tmp_path):
    """A tight box on an object crossing the edge can exceed the frame."""
    dest = tmp_path / "f.txt"
    get_format("yolo_detect").write_labels(
        frame(boxes=(Box(0, -50.0, -20.0, W + 100.0, H + 40.0),)), dest
    )
    values = [float(v) for v in dest.read_text().split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values[2] == pytest.approx(1.0)
    assert values[3] == pytest.approx(1.0)


def test_empty_frame_still_writes_a_file(tmp_path):
    """Negatives are training signal; a missing file would drop them."""
    dest = tmp_path / "f.txt"
    get_format("yolo_detect").write_labels(frame(), dest)
    assert dest.is_file()
    assert dest.read_text() == ""


def test_degenerate_box_is_skipped(tmp_path):
    dest = tmp_path / "f.txt"
    get_format("yolo_detect").write_labels(frame(boxes=(Box(0, 10.0, 10.0, 10.0, 10.0),)), dest)
    assert dest.read_text() == ""


def test_multiple_boxes_one_per_line(tmp_path):
    dest = tmp_path / "f.txt"
    get_format("yolo_detect").write_labels(
        frame(boxes=(Box(0, 0, 0, 100, 100), Box(1, 200, 200, 300, 300))), dest
    )
    lines = dest.read_text().strip().split("\n")
    assert len(lines) == 2
    assert lines[1].startswith("1 ")


# --- manifest ---


def test_dataset_yaml_uses_an_absolute_path(tmp_path):
    """Ultralytics resolves a relative `path:` against the process CWD, so a
    relative one only works when launched from the right directory."""
    path = get_format("yolo_detect").write_manifest(
        tmp_path, ["mug", "bottle"], {"train": 8, "val": 2}
    )
    document = yaml.safe_load(path.read_text())
    assert document["path"] == str(tmp_path.resolve())
    assert document["path"].startswith("/")


def test_names_are_indexed_in_class_id_order(tmp_path):
    path = get_format("yolo_detect").write_manifest(tmp_path, ["mug", "bottle"], {"train": 1, "val": 1})
    assert yaml.safe_load(path.read_text())["names"] == {0: "mug", 1: "bottle"}


def test_test_split_only_declared_when_present(tmp_path):
    without = yaml.safe_load(
        get_format("yolo_detect").write_manifest(tmp_path, ["mug"], {"train": 1, "val": 1}).read_text()
    )
    assert "test" not in without

    with_test = yaml.safe_load(
        get_format("yolo_detect")
        .write_manifest(tmp_path, ["mug"], {"train": 1, "val": 1, "test": 1})
        .read_text()
    )
    assert with_test["test"] == "images/test"


# --- segmentation ---


def test_polygon_is_normalised(tmp_path):
    dest = tmp_path / "f.txt"
    get_format("yolo_seg").write_labels(
        frame(polygons=((0, (0.0, 0.0, float(W), 0.0, float(W), float(H))),)), dest
    )
    parts = dest.read_text().split()
    assert int(parts[0]) == 0
    assert [float(v) for v in parts[1:]] == pytest.approx([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])


def test_polygon_with_too_few_points_is_dropped(tmp_path):
    """Ultralytics requires at least three coordinate pairs."""
    dest = tmp_path / "f.txt"
    get_format("yolo_seg").write_labels(frame(polygons=((0, (0.0, 0.0, 10.0, 10.0)),)), dest)
    assert dest.read_text() == ""


def test_each_polygon_gets_its_own_line(tmp_path):
    dest = tmp_path / "f.txt"
    get_format("yolo_seg").write_labels(
        frame(
            polygons=(
                (0, (0.0, 0.0, 10.0, 0.0, 10.0, 10.0)),
                (0, (20.0, 20.0, 30.0, 20.0, 30.0, 30.0)),
            )
        ),
        dest,
    )
    assert len(dest.read_text().strip().split("\n")) == 2


# --- masks ---


def test_mask_becomes_a_polygon():
    mask = np.zeros((100, 100), dtype=np.uint16)
    mask[20:60, 30:70] = 7
    polygons = mask_to_polygons(mask, 7)
    assert len(polygons) == 1
    xs = polygons[0][0::2]
    ys = polygons[0][1::2]
    assert min(xs) == pytest.approx(30, abs=1)
    assert max(ys) == pytest.approx(59, abs=1)


def test_occluded_instance_keeps_every_visible_region():
    """An occluder splitting an object in two must not silently lose the
    smaller half."""
    mask = np.zeros((100, 100), dtype=np.uint16)
    mask[10:40, 10:40] = 3
    mask[60:90, 60:90] = 3
    assert len(mask_to_polygons(mask, 3)) == 2


def test_noise_regions_are_dropped():
    mask = np.zeros((100, 100), dtype=np.uint16)
    mask[50:80, 50:80] = 3
    mask[0, 0] = 3
    assert len(mask_to_polygons(mask, 3)) == 1


def test_absent_instance_yields_nothing():
    assert mask_to_polygons(np.zeros((50, 50), dtype=np.uint16), 9) == []
    assert mask_to_box(np.zeros((50, 50), dtype=np.uint16), 9) is None


def test_polygon_point_count_is_bounded():
    """Simplification has to converge even on a very wiggly boundary."""
    rng = np.random.default_rng(0)
    mask = np.zeros((200, 200), dtype=np.uint16)
    mask[50:150, 50:150] = 1
    noise = rng.random((200, 200)) > 0.5
    mask[noise & (mask == 1)] = 1
    for polygon in mask_to_polygons(mask, 1, max_points=20):
        assert len(polygon) // 2 <= 20


def test_box_from_mask_is_tight():
    mask = np.zeros((100, 100), dtype=np.uint16)
    mask[20:60, 30:70] = 5
    assert mask_to_box(mask, 5) == (30.0, 20.0, 70.0, 60.0)


def test_unknown_format_is_an_error():
    with pytest.raises(KeyError, match="unknown format"):
        get_format("pascal_voc")
