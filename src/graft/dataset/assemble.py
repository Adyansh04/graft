"""Turn captured clips into an image dataset.

Three things happen here that the rest of the pipeline depends on:

* Subsampling. Consecutive frames in a clip are near-duplicates; taking all
  of them inflates the dataset without adding information.
* Splitting by clip. Splitting by frame would put near-duplicate neighbours
  in both train and val and inflate the val score.
* Quarantine exclusion. QA runs first and marks frames it rejected.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from graft.config.schema import Config
from graft.dataset.formats import get_format
from graft.dataset.formats.base import Box, Frame
from graft.run.manifest import clip_is_complete
from graft.run.paths import RunPaths



@dataclass
class AssembleResult:
    counts: dict[str, int]
    manifest_path: Path
    skipped_quarantined: int = 0

    def render(self) -> str:
        parts = ", ".join(f"{split}={count}" for split, count in sorted(self.counts.items()))
        line = f"dataset: {parts}"
        if self.skipped_quarantined:
            line += f" (excluded {self.skipped_quarantined} quarantined frame(s))"
        return f"{line}\nmanifest: {self.manifest_path}"


def split_clips(clip_indices: list[int], train_fraction: float) -> dict[str, list[int]]:
    """Assign whole clips to splits.

    Deterministic in clip order rather than shuffled, so re-running assemble
    on the same clips produces the same split.
    """
    if not clip_indices:
        return {"train": [], "val": []}
    ordered = sorted(clip_indices)
    n_train = max(1, round(len(ordered) * train_fraction))
    n_train = min(n_train, max(1, len(ordered) - 1)) if len(ordered) > 1 else len(ordered)
    return {"train": ordered[:n_train], "val": ordered[n_train:]}


def selected_frames(n_frames: int, stride: int, quarantined: set[int]) -> list[int]:
    return [i for i in range(0, n_frames, stride) if i not in quarantined]


def assemble(config: Config, paths: RunPaths, *, sources: list[str] | None = None) -> AssembleResult:
    sources = sources or list(config.dataset.sources)
    frames_per_clip = config.capture.frames_per_clip

    complete = [
        index
        for index in paths.existing_clips()
        if clip_is_complete(paths.clip_done(index), frames_per_clip)
    ]
    if not complete:
        raise RuntimeError(f"no complete clips in {paths.clips}; run capture first")

    quarantine = _load_quarantine(paths)
    splits = split_clips(complete, config.dataset.split.train)

    dataset_root = paths.dataset
    _reset(dataset_root)

    label_format = get_format(config.dataset.formats[0])
    counts: dict[str, int] = {}
    skipped = 0

    for split, clip_indices in splits.items():
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for clip_index in clip_indices:
            for source in sources:
                written += _emit_clip(
                    config,
                    paths,
                    clip_index,
                    source,
                    quarantine.get((clip_index, source), set()),
                    image_dir,
                    label_dir,
                    label_format,
                )
                skipped += len(quarantine.get((clip_index, source), set()))
        counts[split] = written

    manifest = label_format.write_manifest(dataset_root, config.class_names(), counts)
    return AssembleResult(counts=counts, manifest_path=manifest, skipped_quarantined=skipped)


def _emit_clip(
    config: Config,
    paths: RunPaths,
    clip_index: int,
    source: str,
    quarantined: set[int],
    image_dir: Path,
    label_dir: Path,
    label_format,
) -> int:
    clip_dir = paths.clip(clip_index)
    if source == "sim":
        rgb_dir = paths.clip_modality(clip_index, "rgb")
    else:
        rgb_dir = paths.cosmos_frames / f"clip_{clip_index:04d}"
    if rgb_dir is None or not rgb_dir.is_dir():
        return 0

    images = sorted(rgb_dir.glob("*.png"))
    wanted = selected_frames(len(images), config.dataset.stride, quarantined)
    width, height = config.sim.resolution

    written = 0
    for frame_index in wanted:
        image = images[frame_index]
        stem = f"{source}_clip{clip_index:04d}_{frame_index:06d}"
        _link_or_copy(image, image_dir / f"{stem}.png")

        boxes = _load_boxes(clip_dir, frame_index)
        label_format.write_labels(
            Frame(
                image_path=image,
                width=width,
                height=height,
                boxes=boxes,
                polygons=_load_polygons(clip_dir, frame_index, boxes),
                clip_index=clip_index,
                frame_index=frame_index,
            ),
            label_dir / f"{stem}.txt",
        )
        written += 1
    return written


def _load_boxes(clip_dir: Path, frame_index: int) -> tuple[Box, ...]:
    path = clip_dir / "labels" / f"bboxes_{frame_index:06d}.json"
    if not path.is_file():
        return ()
    payload = json.loads(path.read_text())
    return tuple(
        Box(
            class_id=record["class_id"],
            x_min=record["x_min"],
            y_min=record["y_min"],
            x_max=record["x_max"],
            y_max=record["y_max"],
        )
        # A record with no class_id carried a label that is not in this
        # run's class list; emitting it would invent a class.
        for record in payload.get("boxes", [])
        if record.get("class_id") is not None
    )


def _load_polygons(clip_dir: Path, frame_index: int, boxes: tuple[Box, ...]):
    mask_path = clip_dir / "labels" / f"instance_{frame_index:06d}.png"
    if not mask_path.is_file() or not boxes:
        return ()

    import numpy as np
    from PIL import Image

    from graft.dataset.masks import mask_to_polygons

    mask = np.asarray(Image.open(mask_path))
    polygons = []
    for instance_id in np.unique(mask):
        if instance_id == 0:
            continue
        for polygon in mask_to_polygons(mask, int(instance_id)):
            polygons.append((boxes[0].class_id, polygon))
    return tuple(polygons)


def _load_quarantine(paths: RunPaths) -> dict[tuple[int, str], set[int]]:
    report = paths.qa / "report.json"
    if not report.is_file():
        return {}
    payload = json.loads(report.read_text())
    out: dict[tuple[int, str], set[int]] = {}
    for entry in payload.get("quarantined", []):
        key = (int(entry["clip"]), entry.get("source", "sim"))
        out.setdefault(key, set()).add(int(entry["frame"]))
    return out


def _link_or_copy(source: Path, dest: Path) -> None:
    """Relative symlink so the dataset does not duplicate hundreds of MB and
    still resolves if the run directory is moved."""
    import os
    import shutil

    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(os.path.relpath(source.resolve(), dest.parent.resolve()))
    except OSError:
        shutil.copy2(source, dest)


def _reset(root: Path) -> None:
    import shutil

    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
