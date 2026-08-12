"""Reclaim disk from rendered clips.

Only the RGB frames feed the dataset. The four control modalities exist to
drive the restyling stage, which consumes them as video, so once the videos
are written and verified their PNGs are redundant — about three quarters of
a clip's size.

Refuses to prune a clip whose videos are missing or short, since that would
destroy the only remaining copy.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from graft.run.manifest import clip_is_complete
from graft.run.paths import RunPaths

# rgb is kept: it is what the dataset is built from.
CONTROL_MODALITIES = ("depth", "segmentation", "shaded_seg", "edges")


@dataclass
class PruneResult:
    clips: list[int] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)
    bytes_freed: int = 0

    @property
    def mb_freed(self) -> float:
        return self.bytes_freed / 1024**2

    def render(self) -> str:
        from graft import console

        lines = [
            f"pruned {console.value(str(len(self.clips)))} clip(s), freed "
            f"{console.value(f'{self.mb_freed:.0f} MB')}"
        ]
        for clip, reason in self.skipped:
            lines.append(console.warn(f"  clip {clip} skipped: {reason}"))
        return "\n".join(lines)


def prune_control_frames(paths: RunPaths, frames: int, *, dry_run: bool = False) -> PruneResult:
    result = PruneResult()

    for index in paths.existing_clips():
        marker = paths.clip_done(index)
        if not clip_is_complete(marker, frames):
            result.skipped.append((index, "clip is not complete"))
            continue

        clip_dir = paths.clip(index)
        problem = _videos_missing(clip_dir)
        if problem:
            result.skipped.append((index, problem))
            continue

        freed = 0
        for modality in CONTROL_MODALITIES:
            for directory in (clip_dir / "cosmos").rglob(modality):
                if not directory.is_dir():
                    continue
                for png in directory.glob("*.png"):
                    freed += png.stat().st_size
                    if not dry_run:
                        png.unlink()

        result.clips.append(index)
        result.bytes_freed += freed
        if not dry_run:
            _mark_pruned(marker)

    return result


def _videos_missing(clip_dir: Path) -> str | None:
    """Every control modality must still have its video before its frames go."""
    for modality in CONTROL_MODALITIES:
        videos = list((clip_dir / "cosmos").rglob(f"{modality}.mp4"))
        if not videos:
            return f"no {modality}.mp4 — frames are the only copy"
    return None


def _mark_pruned(marker: Path) -> None:
    data = json.loads(marker.read_text())
    data["controls_pruned"] = True
    marker.write_text(json.dumps(data, indent=2))
