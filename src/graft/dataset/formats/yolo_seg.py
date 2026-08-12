"""Ultralytics YOLO segmentation format."""

from pathlib import Path

from graft.dataset.formats.base import Frame, normalise_polygon, register
from graft.dataset.formats.yolo_detect import YoloDetect

# Ultralytics requires at least three coordinate pairs per polygon.
MIN_POLYGON_POINTS = 3


@register
class YoloSeg:
    name = "yolo_seg"

    def write_labels(self, frame: Frame, dest: Path) -> None:
        lines = []
        for class_id, points in frame.polygons:
            if len(points) < MIN_POLYGON_POINTS * 2:
                continue
            values = normalise_polygon(points, frame.width, frame.height)
            lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in values))
        dest.write_text("\n".join(lines) + ("\n" if lines else ""))

    def write_manifest(self, root: Path, class_names: list[str], splits: dict[str, int]) -> Path:
        return YoloDetect().write_manifest(root, class_names, splits)
