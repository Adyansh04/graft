"""Export a self-contained Cosmos job.

Cosmos Transfer1 needs 40GB+ of VRAM and roughly 300GB of checkpoints, so it
runs on a different machine. This produces everything that machine needs
except the checkpoints, which it downloads itself.

The bundle carries its own manifest so the import step can verify what comes
back rather than trusting filenames.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from graft.config.schema import Config
from graft.cosmos.encode import COSMOS_FRAMES, EncodeSpec, encode_pngs, probe_frame_count
from graft.cosmos.prompts import build_prompt, load_sections
from graft.cosmos.spec import ClipSpec, build_batch_jsonl, build_controlnet_spec, build_line_map
from graft.run.manifest import clip_is_complete
from graft.run.paths import RunPaths

# The commit Isaac Lab validated its control weights against.
COSMOS_COMMIT = "e4055e39ee9c53165e85275bdab84ed20909714a"

MODALITY_FOR_CONTROL = {"rgb": "rgb", "depth": "depth", "seg": "shaded_seg"}


@dataclass
class BundleResult:
    root: Path
    clips: list[int]

    def render(self) -> str:
        return f"bundle: {self.root}\nclips: {len(self.clips)} ({self.clips})"


def export_bundle(config: Config, paths: RunPaths, encode_spec: EncodeSpec | None = None) -> BundleResult:
    frames = config.capture.frames_per_clip
    complete = [
        index
        for index in paths.existing_clips()
        if clip_is_complete(paths.clip_done(index), frames)
    ]
    if not complete:
        raise RuntimeError("no complete clips to export; run capture first")

    spec = encode_spec or EncodeSpec(
        crf=config.encode.crf,
        preset=config.encode.preset,
        pix_fmt=config.encode.pix_fmt,
        fps=config.encode.fps,
    )

    root = paths.cosmos_bundle
    shutil.rmtree(root, ignore_errors=True)
    (root / "videos").mkdir(parents=True)
    (root / "specs").mkdir(parents=True)

    sections = load_sections(config.cosmos.prompt.sections_file)
    clip_specs: list[ClipSpec] = []

    for clip_index in complete:
        videos = _encode_clip(paths, clip_index, root, spec, frames)
        prompt = build_prompt(
            sections, config.cosmos.prompt.invariant, config.cosmos.seed + clip_index
        )
        clip_specs.append(
            ClipSpec(
                clip_index=clip_index,
                input_video=videos["rgb"],
                prompt=prompt,
                depth_video=videos.get("depth"),
                seg_video=videos.get("seg"),
            )
        )

    for clip in clip_specs:
        (root / "specs" / f"clip_{clip.clip_index:04d}.json").write_text(
            json.dumps(build_controlnet_spec(clip, config), indent=2)
        )

    (root / "batch.jsonl").write_text(build_batch_jsonl(clip_specs, config))
    (root / "manifest.json").write_text(json.dumps(build_line_map(clip_specs), indent=2))
    (root / "run.sh").write_text(_run_script(config))
    (root / "run.sh").chmod(0o755)
    (root / "README.md").write_text(_readme(config, clip_specs))

    return BundleResult(root=root, clips=[c.clip_index for c in clip_specs])


def _encode_clip(
    paths: RunPaths, clip_index: int, root: Path, spec: EncodeSpec, frames: int
) -> dict[str, str]:
    """Encode from the writer's PNGs rather than reusing its mp4s.

    CosmosWriter's own encoder settings are undocumented, and control
    signals are exactly where compression artifacts do the most harm.
    """
    clip_dir = paths.clip(clip_index)
    videos: dict[str, str] = {}
    for control, modality in MODALITY_FOR_CONTROL.items():
        png_dir = clip_dir / "cosmos" / modality
        if not png_dir.is_dir():
            continue
        rel = f"videos/clip_{clip_index:04d}_{modality}.mp4"
        encode_pngs(png_dir, root / rel, spec, expect_frames=frames)
        videos[control] = rel
    if "rgb" not in videos:
        raise RuntimeError(f"clip {clip_index} has no rgb frames to encode")
    return videos


def _run_script(config: Config) -> str:
    return f"""#!/usr/bin/env bash
# Run this inside a cosmos-transfer1 checkout on a GPU with >=40GB VRAM.
# All three offload flags are required to reach that figure; without
# --offload_diffusion_transformer the requirement is far higher.
set -euo pipefail

BUNDLE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
CHECKPOINT_DIR="${{CHECKPOINT_DIR:-./checkpoints}}"
NUM_GPU="${{NUM_GPU:-1}}"

export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONPATH="$(pwd)"

torchrun --nproc_per_node="${{NUM_GPU}}" --nnodes=1 --node_rank=0 \\
    cosmos_transfer1/diffusion/inference/transfer.py \\
    --checkpoint_dir "${{CHECKPOINT_DIR}}" \\
    --controlnet_specs "${{BUNDLE}}/specs/clip_0000.json" \\
    --batch_input_path "${{BUNDLE}}/batch.jsonl" \\
    --video_save_folder "${{BUNDLE}}/output" \\
    --num_gpus "${{NUM_GPU}}" \\
    --offload_text_encoder_model \\
    --offload_guardrail_models \\
    --offload_diffusion_transformer

echo "Outputs in ${{BUNDLE}}/output — copy that directory back and run 'graft cosmos import'."
"""


def _readme(config: Config, clips: list[ClipSpec]) -> str:
    weights = config.cosmos.weights
    return f"""# Cosmos Transfer job bundle

{len(clips)} clip(s), {config.capture.frames_per_clip} frames each at
{tuple(config.sim.resolution)}.

## Requirements

- A GPU with **at least 40GB VRAM**. A 24GB card runs out of memory even with
  every offload flag, and multiple smaller GPUs do not help: context
  parallelism replicates the weights rather than sharding them.
- Roughly 300GB of free disk for checkpoints.
- Linux, Python 3.12, CUDA 12.8. Upstream's Docker image is the supported
  path; the conda environment file pins a different CUDA version than the
  wheels expect.

## Setup

```bash
git clone https://github.com/nvidia-cosmos/cosmos-transfer1
cd cosmos-transfer1
git checkout {COSMOS_COMMIT}
```

Accept the terms for `meta-llama/Llama-Guard-3-8B` on Hugging Face, then:

```bash
huggingface-cli login
PYTHONPATH=$(pwd) python scripts/download_checkpoints.py --output_dir checkpoints/
```

## Run

Copy this bundle into the checkout and run `./run.sh`.

Control weights are vis={weights.vis}, edge={weights.edge},
depth={weights.depth}, seg={weights.seg}, sigma_max={config.cosmos.sigma_max}.
`seg` is fed shaded segmentation rather than flat class colours. If the
restyle is not changing lighting enough, lower `vis` — it is the control
that preserves the source's lighting by construction.

## Expected output

One directory per batch line: `output/video_N/output.mp4`. `manifest.json`
maps each line to its clip; N is the line index, not the clip index.

Copy `output/` back into the run directory and run `graft cosmos import`,
which verifies resolution and frame counts before pairing anything.
"""
