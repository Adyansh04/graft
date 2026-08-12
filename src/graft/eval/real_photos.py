"""Ingest hand-labelled photographs of the physical object.

This is the evaluation target that counts. Held-out sim clips share the
renderer, asset and randomization distribution with training, so a model can
score well on them by learning Isaac's rendering characteristics. Sim-val is
kept as a diagnostic — the gap between the two is itself the sim-to-real
signal — but the real number is the baseline.

Eval only. No real label ever enters training, which is what preserves the
zero-annotation property.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass
class IngestReport:
    images: int = 0
    labelled: int = 0
    problems: list[str] = field(default_factory=list)
    dataset_yaml: Path | None = None

    @property
    def ok(self) -> bool:
        return not self.problems and self.labelled > 0

    def render(self) -> str:
        lines = [f"real photos: {self.images} image(s), {self.labelled} with labels"]
        lines.extend(f"  {p}" for p in self.problems)
        if self.dataset_yaml:
            lines.append(f"  manifest: {self.dataset_yaml}")
        return "\n".join(lines)


def ingest(photos_dir: str | Path, class_names: list[str], dest: Path) -> IngestReport:
    """Validate a directory of photographs and labels, and emit a manifest.

    Expects `images/` and `labels/` siblings, matching Ultralytics' layout,
    with one YOLO-format .txt per image.
    """
    photos_dir = Path(photos_dir)
    report = IngestReport()

    image_dir = photos_dir / "images"
    label_dir = photos_dir / "labels"
    if not image_dir.is_dir():
        report.problems.append(f"missing {image_dir}")
        return report
    if not label_dir.is_dir():
        report.problems.append(f"missing {label_dir}")
        return report

    images = sorted(p for p in image_dir.iterdir() if p.suffix in IMAGE_SUFFIXES)
    report.images = len(images)
    if not images:
        report.problems.append(f"no images in {image_dir}")
        return report

    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            report.problems.append(f"no label for {image.name}")
            continue
        errors = validate_label_file(label, len(class_names))
        if errors:
            report.problems.extend(f"{label.name}: {e}" for e in errors)
            continue
        report.labelled += 1

    if report.labelled:
        report.dataset_yaml = _write_manifest(photos_dir, class_names, dest)
    return report


def validate_label_file(path: Path, n_classes: int) -> list[str]:
    """YOLO detect format: `class cx cy w h`, all normalised."""
    errors = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 fields, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"line {line_number}: non-numeric field")
            continue
        if not 0 <= class_id < n_classes:
            errors.append(f"line {line_number}: class id {class_id} outside 0..{n_classes - 1}")
        if any(v < 0.0 or v > 1.0 for v in values):
            errors.append(f"line {line_number}: coordinates must be normalised to 0..1")
        if values[2] <= 0 or values[3] <= 0:
            errors.append(f"line {line_number}: zero-area box")
    return errors


def _write_manifest(photos_dir: Path, class_names: list[str], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # `names` ordering must match the training manifest exactly — Ultralytics
    # class ids are positional, so a different order scores the wrong class.
    dest.write_text(
        yaml.safe_dump(
            {
                "path": str(photos_dir.resolve()),
                "train": "images",
                "val": "images",
                "names": {index: name for index, name in enumerate(class_names)},
            },
            sort_keys=False,
        )
    )
    return dest
