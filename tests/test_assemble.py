import json

import numpy as np
import pytest
import yaml
from PIL import Image

import graft.dataset.formats.yolo_detect  # noqa: F401
from graft.dataset.assemble import assemble, selected_frames, split_clips
from graft.run.manifest import write_clip_done
from graft.run.paths import RunPaths

FRAMES = 121


def build_clip(paths: RunPaths, index: int, *, frames: int = FRAMES, with_boxes: bool = True):
    """A clip on disk shaped the way capture leaves it."""
    clip = paths.clip(index)
    rgb = clip / "cosmos" / "rgb"
    labels = clip / "labels"
    rgb.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    for i in range(frames):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(rgb / f"rgb_{i:06d}.png")
        boxes = (
            [{"class_id": 0, "x_min": 10, "y_min": 20, "x_max": 110, "y_max": 220}]
            if with_boxes
            else []
        )
        (labels / f"bboxes_{i:06d}.json").write_text(json.dumps({"frame": i, "boxes": boxes}))

    write_clip_done(
        paths.clip_done(index),
        index=index,
        seed=index,
        modality_counts={m: frames for m in ("rgb", "depth", "segmentation", "shaded_seg", "edges")},
        label_count=frames,
        outputs={},
    )


@pytest.fixture
def paths(tmp_path):
    p = RunPaths.for_run(tmp_path, "run")
    p.create()
    return p


# --- split ---


def test_clips_never_span_splits():
    """Frames from one clip are near-duplicates; sharing a clip across
    train and val inflates the val score."""
    splits = split_clips(list(range(10)), 0.8)
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert len(splits["train"]) + len(splits["val"]) == 10


def test_split_is_deterministic():
    assert split_clips(list(range(10)), 0.8) == split_clips(list(range(10)), 0.8)


def test_val_is_never_empty_when_clips_allow_it():
    assert split_clips([0, 1], 0.9)["val"]


def test_single_clip_goes_to_train():
    assert split_clips([0], 0.8) == {"train": [0], "val": []}


# --- stride ---


def test_stride_subsamples():
    assert selected_frames(121, 10, set()) == list(range(0, 121, 10))


def test_quarantined_frames_are_excluded():
    assert 20 not in selected_frames(121, 10, {20})


def test_stride_of_one_keeps_everything():
    assert len(selected_frames(50, 1, set())) == 50


# --- assembly ---


def test_assembles_images_and_labels(config, paths, tmp_path):
    config.dataset.stride = 10
    for index in range(4):
        build_clip(paths, index)

    result = assemble(config, paths)

    train_images = sorted((paths.dataset / "images" / "train").glob("*.png"))
    train_labels = sorted((paths.dataset / "labels" / "train").glob("*.txt"))
    assert len(train_images) == len(train_labels)
    assert result.counts["train"] > 0
    assert result.counts["val"] > 0


def test_images_and_labels_are_siblings_not_nested(config, paths):
    """Ultralytics derives label paths by replacing /images/ with /labels/."""
    config.dataset.stride = 60
    build_clip(paths, 0)
    build_clip(paths, 1)
    assemble(config, paths)
    assert (paths.dataset / "images" / "train").is_dir()
    assert (paths.dataset / "labels" / "train").is_dir()
    assert not (paths.dataset / "images" / "train" / "labels").exists()


def test_every_image_has_a_matching_label_file(config, paths):
    config.dataset.stride = 40
    build_clip(paths, 0)
    build_clip(paths, 1)
    assemble(config, paths)
    for split in ("train", "val"):
        images = {p.stem for p in (paths.dataset / "images" / split).glob("*.png")}
        labels = {p.stem for p in (paths.dataset / "labels" / split).glob("*.txt")}
        assert images == labels


def test_frames_without_objects_still_get_empty_labels(config, paths):
    config.dataset.stride = 60
    build_clip(paths, 0, with_boxes=False)
    build_clip(paths, 1, with_boxes=False)
    assemble(config, paths)
    labels = list((paths.dataset / "labels" / "train").glob("*.txt"))
    assert labels
    assert all(p.read_text() == "" for p in labels)


def test_incomplete_clips_are_ignored(config, paths):
    config.dataset.stride = 60
    build_clip(paths, 0)
    build_clip(paths, 1)
    # A clip directory with no valid marker: interrupted mid-render.
    (paths.clip(2) / "cosmos" / "rgb").mkdir(parents=True)
    assemble(config, paths)
    names = {p.name for p in (paths.dataset / "images").rglob("*.png")}
    assert not any("clip0002" in n for n in names)


def test_no_complete_clips_is_a_clear_error(config, paths):
    with pytest.raises(RuntimeError, match="run capture first"):
        assemble(config, paths)


def test_quarantined_frames_are_left_out(config, paths):
    config.dataset.stride = 10
    build_clip(paths, 0)
    build_clip(paths, 1)
    paths.qa.mkdir(parents=True, exist_ok=True)
    (paths.qa / "report.json").write_text(
        json.dumps({"quarantined": [{"clip": 0, "frame": 0, "source": "sim"}]})
    )

    result = assemble(config, paths)

    assert result.skipped_quarantined == 1
    assert not (paths.dataset / "images" / "train" / "sim_clip0000_000000.png").exists()


def test_manifest_points_at_the_dataset_root(config, paths):
    config.dataset.stride = 60
    build_clip(paths, 0)
    build_clip(paths, 1)
    result = assemble(config, paths)
    document = yaml.safe_load(result.manifest_path.read_text())
    assert document["path"] == str(paths.dataset.resolve())
    assert document["names"] == {0: "mug"}


def test_reassembly_does_not_accumulate(config, paths):
    config.dataset.stride = 60
    build_clip(paths, 0)
    build_clip(paths, 1)
    first = assemble(config, paths).counts
    second = assemble(config, paths).counts
    assert first == second


def test_images_are_linked_rather_than_duplicated(config, paths):
    """Copying every frame would double hundreds of MB per run."""
    config.dataset.stride = 60
    build_clip(paths, 0)
    build_clip(paths, 1)
    assemble(config, paths)
    images = list((paths.dataset / "images").rglob("*.png"))
    assert images
    assert any(p.is_symlink() for p in images)


def test_linked_images_resolve(config, paths):
    config.dataset.stride = 60
    build_clip(paths, 0)
    build_clip(paths, 1)
    assemble(config, paths)
    for image in (paths.dataset / "images").rglob("*.png"):
        assert image.resolve().is_file()
