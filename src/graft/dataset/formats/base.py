"""Output format interface.

Adding a format means adding one file here and registering it. The sim
writes format-neutral records; everything below turns them into a specific
layout.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Box:
    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)


@dataclass(frozen=True)
class Frame:
    """One captured frame and its annotations, in pixel coordinates."""

    image_path: Path
    width: int
    height: int
    boxes: tuple[Box, ...] = ()
    # One polygon per instance, each a flat (x, y, x, y, ...) pixel sequence.
    polygons: tuple[tuple[int, tuple[float, ...]], ...] = ()
    clip_index: int = 0
    frame_index: int = 0


class LabelFormat(Protocol):
    name: str

    def write_labels(self, frame: Frame, dest: Path) -> None:
        """Write one frame's labels. Must write an empty file when the frame
        has no annotations — negatives are training signal."""
        ...

    def write_manifest(self, root: Path, class_names: list[str], splits: dict[str, int]) -> Path:
        """Write whatever index file the format needs (dataset.yaml, etc.)."""
        ...


FORMATS: dict[str, type] = {}


def register(cls):
    FORMATS[cls.name] = cls
    return cls


def get_format(name: str) -> LabelFormat:
    if name not in FORMATS:
        raise KeyError(f"unknown format {name!r}; available: {sorted(FORMATS)}")
    return FORMATS[name]()


def normalise_box(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    """Pixel box to YOLO centre form, clamped to the frame.

    Clamping matters: a tight box on an object crossing the frame edge can
    extend slightly outside it.
    """
    x_min = max(0.0, min(box.x_min, width))
    x_max = max(0.0, min(box.x_max, width))
    y_min = max(0.0, min(box.y_min, height))
    y_max = max(0.0, min(box.y_max, height))
    cx = (x_min + x_max) / 2.0 / width
    cy = (y_min + y_max) / 2.0 / height
    w = (x_max - x_min) / width
    h = (y_max - y_min) / height
    return (
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
        min(max(w, 0.0), 1.0),
        min(max(h, 0.0), 1.0),
    )


def normalise_polygon(points: tuple[float, ...], width: int, height: int) -> list[float]:
    out = []
    for index, value in enumerate(points):
        extent = width if index % 2 == 0 else height
        out.append(min(max(value / extent, 0.0), 1.0))
    return out
