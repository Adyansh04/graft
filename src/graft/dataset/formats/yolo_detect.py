"""Ultralytics YOLO detection format."""

from pathlib import Path

import yaml

from graft.dataset.formats.base import Frame, normalise_box, register


@register
class YoloDetect:
    name = "yolo_detect"

    def write_labels(self, frame: Frame, dest: Path) -> None:
        lines = []
        for box in frame.boxes:
            cx, cy, w, h = normalise_box(box, frame.width, frame.height)
            if w <= 0.0 or h <= 0.0:
                continue
            lines.append(f"{box.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        # Written even when empty: frames with no object are negatives, and
        # dropping the file would drop that signal.
        dest.write_text("\n".join(lines) + ("\n" if lines else ""))

    def write_manifest(self, root: Path, class_names: list[str], splits: dict[str, int]) -> Path:
        path = root / "dataset.yaml"
        # Ultralytics resolves a relative `path:` against the process CWD
        # rather than the yaml's own directory, so it must be absolute.
        document = {
            "path": str(root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(class_names)},
        }
        if splits.get("test"):
            document["test"] = "images/test"
        path.write_text(yaml.safe_dump(document, sort_keys=False))
        return path
