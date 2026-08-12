"""Run the checks over a run's frames and record what failed.

Writes `qa/report.json` with a quarantine list. Assembly reads it and leaves
those frames out, which is why QA runs before assembly rather than after.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from graft.config.schema import Config
from graft.qa import checks
from graft.run.manifest import clip_is_complete
from graft.run.paths import RunPaths


@dataclass
class QaReport:
    checked: int = 0
    quarantined: list[dict] = field(default_factory=list)
    by_reason: dict[str, int] = field(default_factory=dict)

    def reject(self, clip: int, frame: int, source: str, reason: str, value: float) -> None:
        self.quarantined.append(
            {"clip": clip, "frame": frame, "source": source, "reason": reason, "value": value}
        )
        self.by_reason[reason] = self.by_reason.get(reason, 0) + 1

    @property
    def rejection_rate(self) -> float:
        return len(self.quarantined) / self.checked if self.checked else 0.0

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "quarantined": self.quarantined,
            "by_reason": self.by_reason,
            "rejection_rate": round(self.rejection_rate, 4),
        }

    def render(self) -> str:
        lines = [
            f"checked {self.checked} frame(s), quarantined {len(self.quarantined)} "
            f"({self.rejection_rate:.1%})"
        ]
        for reason, count in sorted(self.by_reason.items()):
            lines.append(f"  {reason}: {count}")
        # NVIDIA's own Cosmos SDG work reports around 3% rejection. An order
        # of magnitude more usually means the generation config is wrong
        # rather than the filter being strict.
        if self.rejection_rate > 0.3:
            lines.append(
                "  rejection rate is high — check the render or Cosmos settings "
                "before assuming the thresholds are wrong"
            )
        return "\n".join(lines)


def run_qa(config: Config, paths: RunPaths, *, sources: list[str] | None = None) -> QaReport:
    sources = sources or list(config.dataset.sources)
    report = QaReport()

    for clip_index in paths.existing_clips():
        if not clip_is_complete(paths.clip_done(clip_index), config.capture.frames_per_clip):
            continue
        for source in sources:
            _check_clip(config, paths, clip_index, source, report)

    paths.qa.mkdir(parents=True, exist_ok=True)
    (paths.qa / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
    return report


def _check_clip(config: Config, paths: RunPaths, clip_index: int, source: str, report: QaReport) -> None:
    from PIL import Image

    clip_dir = paths.clip(clip_index)
    if source == "sim":
        frame_dir = clip_dir / "cosmos" / "rgb"
    else:
        frame_dir = paths.cosmos_frames / f"clip_{clip_index:04d}"
    if not frame_dir.is_dir():
        return

    sim_dir = clip_dir / "cosmos" / "rgb"
    frames = sorted(frame_dir.glob("*.png"))
    # Only frames that reach the dataset are worth checking.
    for frame_index in range(0, len(frames), config.dataset.stride):
        image = np.asarray(Image.open(frames[frame_index]).convert("RGB"))
        report.checked += 1

        blur = checks.check_blur(image, config.qa.blur_lap_var_min)
        if not blur.passed:
            report.reject(clip_index, frame_index, source, blur.name, blur.value)
            continue

        boxes = _load_boxes(clip_dir, frame_index)
        area = checks.check_bbox_area(boxes, config.qa.min_bbox_area_px)
        if not area.passed:
            report.reject(clip_index, frame_index, source, area.name, area.value)
            continue

        if source == "cosmos":
            _check_cosmos_frame(
                config, clip_dir, sim_dir, image, clip_index, frame_index, source, report
            )


def _check_cosmos_frame(
    config, clip_dir: Path, sim_dir: Path, image, clip_index, frame_index, source, report
) -> None:
    from PIL import Image

    sim_frames = sorted(sim_dir.glob("*.png"))
    mask_path = clip_dir / "labels" / f"instance_{frame_index:06d}.png"
    if frame_index >= len(sim_frames) or not mask_path.is_file():
        return

    sim_image = np.asarray(Image.open(sim_frames[frame_index]).convert("RGB"))
    mask = np.asarray(Image.open(mask_path)) > 0
    if mask.ndim == 3:
        mask = mask[..., 0]

    preserved = checks.check_object_preserved(
        sim_image, image, mask, config.qa.cosmos.in_mask_ssim_min
    )
    if not preserved.passed:
        report.reject(clip_index, frame_index, source, preserved.name, preserved.value)
        return

    changed = checks.check_background_changed(
        sim_image, image, mask, config.qa.cosmos.out_mask_change_min
    )
    if not changed.passed:
        report.reject(clip_index, frame_index, source, changed.name, changed.value)


def _load_boxes(clip_dir: Path, frame_index: int) -> list[dict]:
    path = clip_dir / "labels" / f"bboxes_{frame_index:06d}.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text()).get("boxes", [])
