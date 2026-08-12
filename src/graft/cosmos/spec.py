"""controlnet_specs and the batch manifest for cosmos-transfer1.

The spec format has an unusual rule: a top-level key that is a known hint
name becomes a control branch, while any other non-dict key becomes an
argparse override. That is how `prompt`, `sigma_max` and friends legally sit
alongside `vis`/`edge`/`depth`/`seg` in the same file.

JSON values win over command-line flags that were not explicitly typed, so
everything per-clip belongs here rather than in the launch command.
"""

import json
from dataclasses import dataclass
from pathlib import Path

HINT_KEYS = ("vis", "edge", "depth", "seg")

# Documented ceiling on the summed control weights.
MAX_TOTAL_WEIGHT = 2.0


@dataclass(frozen=True)
class ClipSpec:
    clip_index: int
    input_video: str
    prompt: str
    depth_video: str | None = None
    seg_video: str | None = None


def build_controlnet_spec(clip: ClipSpec, config) -> dict:
    """One clip's spec.

    `vis` is fed the raw rgb video and blurs it internally. `seg` is fed
    shaded segmentation rather than flat class colours, which is what
    NVIDIA's own object-scale work uses. `edge` is left to auto-Canny.
    """
    weights = config.cosmos.weights
    spec: dict = {
        "prompt": clip.prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "input_video_path": clip.input_video,
        "sigma_max": config.cosmos.sigma_max,
        "guidance": config.cosmos.guidance,
        "fps": config.cosmos.fps,
        "seed": config.cosmos.seed,
    }
    if weights.vis > 0:
        spec["vis"] = {"control_weight": weights.vis, "input_control": clip.input_video}
    if weights.edge > 0:
        spec["edge"] = {"control_weight": weights.edge}
    if weights.depth > 0:
        control = {"control_weight": weights.depth}
        if clip.depth_video:
            control["input_control"] = clip.depth_video
        spec["depth"] = control
    if weights.seg > 0:
        control = {"control_weight": weights.seg}
        if clip.seg_video:
            control["input_control"] = clip.seg_video
        spec["seg"] = control

    total = sum(spec[k]["control_weight"] for k in HINT_KEYS if k in spec)
    if total > MAX_TOTAL_WEIGHT:
        raise ValueError(f"control weights sum to {total:.2f}, above the {MAX_TOTAL_WEIGHT} ceiling")
    return spec


def build_batch_jsonl(clips: list[ClipSpec], config) -> str:
    """Batch input for a single process handling every clip.

    One model load rather than N: the weights are tens of GB, so per-clip
    invocation would be dominated by loading them.
    """
    lines = []
    for clip in clips:
        overrides: dict = {}
        if clip.depth_video:
            overrides["depth"] = {"input_control": clip.depth_video}
        if clip.seg_video:
            overrides["seg"] = {"input_control": clip.seg_video}
        entry = {"visual_input": clip.input_video, "prompt": clip.prompt}
        if overrides:
            entry["control_overrides"] = overrides
        lines.append(json.dumps(entry))
    return "\n".join(lines) + "\n"


def build_line_map(clips: list[ClipSpec]) -> dict:
    """Map JSONL line index to clip index.

    Batch output lands in `video_N/`, numbered by line rather than by clip.
    If a bundle carries a subset of clips, N is not the clip index, so the
    mapping has to be recorded or the import would pair frames with the
    wrong labels.
    """
    return {
        "clips": [
            {
                "line": line,
                "clip": clip.clip_index,
                "expected_output": f"video_{line}/output.mp4",
                "input_video": clip.input_video,
            }
            for line, clip in enumerate(clips)
        ]
    }


NEGATIVE_PROMPT = (
    "The video captures a game playing, with bad crappy graphics and cartoonish "
    "frames. It represents a recording of old outdated games. The images are very "
    "pixelated and of poor CG quality. There are many subtitles in the footage. "
    "Overall, the video is unrealistic and appears cg. Plane background."
)
