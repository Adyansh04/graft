import numpy as np
import yaml
from PIL import Image

from graft.eval.real_photos import ingest, validate_label_file

CLASSES = ["mug"]


def build(tmp_path, labels: dict[str, str], n_images: int | None = None):
    """A hand-labelled photo directory in Ultralytics layout."""
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    names = list(labels)
    if n_images is not None:
        names += [f"unlabelled{i}" for i in range(n_images - len(names))]
    for name in names:
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / f"{name}.jpg")
    for name, content in labels.items():
        (label_dir / f"{name}.txt").write_text(content)
    return tmp_path


# --- label validation ---


def test_valid_label_passes(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 0.5 0.5 0.2 0.3\n")
    assert validate_label_file(path, 1) == []


def test_unnormalised_coordinates_are_caught(tmp_path):
    """Pixel coordinates instead of normalised is the classic hand-labelling
    mistake, and it silently produces meaningless scores."""
    path = tmp_path / "a.txt"
    path.write_text("0 640 360 100 200\n")
    assert any("normalised" in e for e in validate_label_file(path, 1))


def test_class_id_out_of_range_is_caught(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("5 0.5 0.5 0.2 0.3\n")
    assert any("outside" in e for e in validate_label_file(path, 1))


def test_wrong_field_count_is_caught(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 0.5 0.5 0.2\n")
    assert any("5 fields" in e for e in validate_label_file(path, 1))


def test_zero_area_box_is_caught(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 0.5 0.5 0.0 0.3\n")
    assert any("zero-area" in e for e in validate_label_file(path, 1))


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 0.5 0.5 0.2 0.3\n\n")
    assert validate_label_file(path, 1) == []


def test_empty_label_file_is_valid(tmp_path):
    """A photo with the object genuinely absent is a valid negative."""
    path = tmp_path / "a.txt"
    path.write_text("")
    assert validate_label_file(path, 1) == []


# --- ingest ---


def test_ingest_accepts_a_well_formed_set(tmp_path):
    build(tmp_path, {f"p{i}": "0 0.5 0.5 0.2 0.3\n" for i in range(5)})
    report = ingest(tmp_path, CLASSES, tmp_path / "real.yaml")
    assert report.ok
    assert report.images == 5
    assert report.labelled == 5


def test_missing_label_is_reported(tmp_path):
    build(tmp_path, {"p0": "0 0.5 0.5 0.2 0.3\n"}, n_images=3)
    report = ingest(tmp_path, CLASSES, tmp_path / "real.yaml")
    assert report.images == 3
    assert report.labelled == 1
    assert any("no label" in p for p in report.problems)


def test_missing_directories_are_reported(tmp_path):
    report = ingest(tmp_path, CLASSES, tmp_path / "real.yaml")
    assert not report.ok
    assert any("missing" in p for p in report.problems)


def test_manifest_class_order_matches_training(tmp_path):
    """Ultralytics class ids are positional — a different order here would
    score predictions against the wrong class."""
    build(tmp_path, {"p0": "1 0.5 0.5 0.2 0.3\n"})
    dest = tmp_path / "real.yaml"
    ingest(tmp_path, ["mug", "bottle"], dest)
    assert yaml.safe_load(dest.read_text())["names"] == {0: "mug", 1: "bottle"}


def test_manifest_path_is_absolute(tmp_path):
    build(tmp_path, {"p0": "0 0.5 0.5 0.2 0.3\n"})
    dest = tmp_path / "real.yaml"
    ingest(tmp_path, CLASSES, dest)
    assert yaml.safe_load(dest.read_text())["path"] == str(tmp_path.resolve())
