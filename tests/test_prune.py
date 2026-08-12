import json

import numpy as np
import pytest
from PIL import Image

from graft.dataset.prune import CONTROL_MODALITIES, prune_control_frames
from graft.run.manifest import write_clip_done
from graft.run.paths import RunPaths

FRAMES = 4
ALL_MODALITIES = ("rgb",) + CONTROL_MODALITIES


@pytest.fixture
def paths(tmp_path):
    p = RunPaths.for_run(tmp_path, "run")
    p.create()
    return p


def build_clip(paths, index, *, frames=FRAMES, videos=True, complete=True):
    base = paths.clip(index) / "cosmos" / f"clip_{index:04d}"
    for modality in ALL_MODALITIES:
        directory = base / modality
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(frames):
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
                directory / f"{modality}_{i:04d}.png"
            )
        if videos:
            (base / f"{modality}.mp4").write_bytes(b"x" * 32)
    if complete:
        write_clip_done(
            paths.clip_done(index),
            index=index,
            seed=index,
            modality_counts={m: frames for m in ALL_MODALITIES},
            label_count=frames,
            outputs={},
        )


def count(paths, index, modality):
    return len(list(paths.clip(index).rglob(f"{modality}/*.png")))


def test_control_frames_are_removed_and_rgb_kept(paths):
    build_clip(paths, 0)
    result = prune_control_frames(paths, FRAMES)

    assert result.clips == [0]
    assert count(paths, 0, "rgb") == FRAMES
    for modality in CONTROL_MODALITIES:
        assert count(paths, 0, modality) == 0


def test_videos_survive(paths):
    """The videos are what make the frames redundant."""
    build_clip(paths, 0)
    prune_control_frames(paths, FRAMES)
    assert len(list(paths.clip(0).rglob("*.mp4"))) == len(ALL_MODALITIES)


def test_clip_without_videos_is_skipped(paths):
    """Frames would be the only copy — deleting them destroys the data."""
    build_clip(paths, 0, videos=False)
    result = prune_control_frames(paths, FRAMES)

    assert result.clips == []
    assert any("only copy" in reason for _, reason in result.skipped)
    assert count(paths, 0, "depth") == FRAMES


def test_incomplete_clip_is_skipped(paths):
    build_clip(paths, 0, complete=False)
    result = prune_control_frames(paths, FRAMES)
    assert result.clips == []
    assert count(paths, 0, "depth") == FRAMES


def test_dry_run_deletes_nothing(paths):
    build_clip(paths, 0)
    result = prune_control_frames(paths, FRAMES, dry_run=True)

    assert result.clips == [0]
    assert result.bytes_freed > 0
    assert count(paths, 0, "depth") == FRAMES


def test_marker_records_the_prune(paths):
    build_clip(paths, 0)
    prune_control_frames(paths, FRAMES)
    assert json.loads(paths.clip_done(0).read_text())["controls_pruned"] is True


def test_pruning_twice_is_harmless(paths):
    build_clip(paths, 0)
    prune_control_frames(paths, FRAMES)
    second = prune_control_frames(paths, FRAMES)
    assert second.bytes_freed == 0
    assert count(paths, 0, "rgb") == FRAMES


def test_pruned_clip_still_counts_as_complete(paths):
    """Pruning must not make resume re-render the clip."""
    from graft.run.manifest import clip_is_complete

    build_clip(paths, 0)
    prune_control_frames(paths, FRAMES)
    assert clip_is_complete(paths.clip_done(0), FRAMES)
