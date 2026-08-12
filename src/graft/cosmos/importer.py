"""Validate and decode what comes back from the Cosmos machine.

Nothing returned is trusted. The mapping from output directory to clip is
positional by batch line, resolution can be silently changed by a resize,
and a short control track truncates a whole batch — all three would surface
much later as labels that no longer describe the frames.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from graft.config.schema import Config
from graft.cosmos.encode import decode_to_pngs, probe_frame_count, probe_resolution
from graft.run.paths import RunPaths


@dataclass
class ImportResult:
    imported: list[int] = field(default_factory=list)
    rejected: list[tuple[int, str]] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"imported {len(self.imported)} clip(s): {self.imported}"]
        for clip, reason in self.rejected:
            lines.append(f"  clip {clip} rejected: {reason}")
        return "\n".join(lines)


def import_outputs(config: Config, paths: RunPaths, output_dir: Path | None = None) -> ImportResult:
    manifest_path = paths.cosmos_bundle / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"{manifest_path} missing — the bundle records which batch line belongs to "
            "which clip, and output directories are numbered by line, not by clip"
        )
    entries = json.loads(manifest_path.read_text())["clips"]
    output_dir = output_dir or paths.cosmos_output

    result = ImportResult()
    expected_resolution = tuple(config.sim.resolution)
    expected_frames = config.capture.frames_per_clip

    for entry in entries:
        clip = int(entry["clip"])
        video = output_dir / entry["expected_output"]
        if not video.is_file():
            result.rejected.append((clip, f"missing {video}"))
            continue

        problem = _validate(video, expected_resolution, expected_frames)
        if problem:
            result.rejected.append((clip, problem))
            continue

        decode_to_pngs(
            video,
            paths.cosmos_frames / f"clip_{clip:04d}",
            expect_frames=expected_frames,
        )
        result.imported.append(clip)

    return result


def _validate(video: Path, expected_resolution: tuple[int, int], expected_frames: int) -> str | None:
    resolution = probe_resolution(video)
    if resolution != expected_resolution:
        return (
            f"resolution {resolution} != {expected_resolution}; a resize or centre-crop "
            "would invalidate every label coordinate"
        )
    frames = probe_frame_count(video)
    if frames != expected_frames:
        return f"{frames} frames != {expected_frames}; frames and labels would not line up"
    return None
