"""PNG sequences to mp4, and back again.

Video is only an interchange format — Cosmos Transfer is a video model. The
dataset is images, so every encode is a round trip and quality matters more
than size.

Two rules the round trip depends on:

* Encode once, from the writer's PNGs. Never re-encode Cosmos output; decode
  it straight to PNG.
* Assert frame counts. Cosmos trims every control track to the shortest one,
  so a single short mp4 silently truncates a whole batch, and the damage
  shows up later as labels that no longer line up with frames.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Cosmos Transfer1 generates exactly this many frames per clip.
COSMOS_FRAMES = 121


class FfmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class EncodeSpec:
    crf: int = 12
    preset: str = "slow"
    pix_fmt: str = "yuv420p"
    fps: int = 30


def encode_pngs(
    png_dir: Path,
    dest: Path,
    spec: EncodeSpec,
    *,
    pattern: str = "*.png",
    expect_frames: int | None = COSMOS_FRAMES,
) -> Path:
    frames = sorted(png_dir.glob(pattern))
    if not frames:
        raise FfmpegError(f"no frames matching {pattern} in {png_dir}")
    if expect_frames is not None and len(frames) != expect_frames:
        raise FfmpegError(
            f"{png_dir} has {len(frames)} frames, expected {expect_frames}. Cosmos trims "
            "all control tracks to the shortest, so a short track would silently "
            "truncate the batch."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    listing = dest.parent / f".{dest.stem}.frames.txt"
    # A concat list avoids depending on the frames being numbered contiguously.
    listing.write_text("\n".join(f"file '{f.resolve()}'" for f in frames))

    command = [
        "ffmpeg", "-y",
        "-r", str(spec.fps),
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c:v", "libx264",
        "-crf", str(spec.crf),
        "-preset", spec.preset,
        "-pix_fmt", spec.pix_fmt,
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-movflags", "+faststart",
        str(dest),
    ]
    _run(command)
    listing.unlink(missing_ok=True)

    written = probe_frame_count(dest)
    if expect_frames is not None and written != expect_frames:
        raise FfmpegError(f"{dest} encoded {written} frames, expected {expect_frames}")
    return dest


def decode_to_pngs(video: Path, dest_dir: Path, *, expect_frames: int | None = COSMOS_FRAMES) -> list[Path]:
    """Decode without re-encoding or resampling.

    `-fps_mode passthrough` keeps demuxer timestamps; the default duplicates
    and drops frames to hit a target rate, which would desynchronise frames
    from their labels. Never pass `-r` here.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for stale in dest_dir.glob("*.png"):
        stale.unlink()

    _run(
        [
            "ffmpeg", "-y",
            "-i", str(video),
            "-fps_mode", "passthrough",
            "-start_number", "0",
            str(dest_dir / "frame_%06d.png"),
        ]
    )
    frames = sorted(dest_dir.glob("*.png"))
    if expect_frames is not None and len(frames) != expect_frames:
        raise FfmpegError(
            f"{video} decoded to {len(frames)} frames, expected {expect_frames}; "
            "frames and labels would no longer line up"
        )
    return frames


def probe_frame_count(video: Path) -> int:
    result = _run(
        [
            "ffprobe", "-v", "error",
            "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(video),
        ]
    )
    text = result.strip().splitlines()[0] if result.strip() else ""
    try:
        return int(text)
    except ValueError as exc:
        raise FfmpegError(f"could not read frame count from {video}: {text!r}") from exc


def probe_resolution(video: Path) -> tuple[int, int]:
    result = _run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(video),
        ]
    )
    stream = json.loads(result)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _run(command: list[str]) -> str:
    try:
        proc = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise FfmpegError(f"could not run {command[0]}: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise FfmpegError(f"{command[0]} failed:\n" + "\n".join(tail))
    return proc.stdout
