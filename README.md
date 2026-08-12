# GRAFT

**G**enerative **R**endering of **A**nnotated **F**rames for **T**raining.

Point GRAFT at a 3D model of an object and get a trained object detector,
without hand-labelling a single photograph.

The object is dropped into a physics scene in NVIDIA Isaac Sim, filmed from
orbiting cameras under randomised lighting, and annotated automatically —
the simulator knows exactly where the object is, so the labels are exact and
free. Those clips can then be restyled by NVIDIA Cosmos Transfer to add
realistic backgrounds, materials and lighting that a renderer cannot produce
on its own, while the geometry is held fixed so the labels stay valid.

```
USD model -> validate -> render clips -> [restyle] -> QA -> dataset -> train -> evaluate
```

- **Zero hand-annotated training images.** Labels come from the simulator.
- **Resumable.** Every stage can be re-run on its own; an interrupted render
  picks up where it stopped.
- **Config-driven.** One YAML file describes a run. No flag soup.
- **Pluggable.** Output formats, asset sources and the trainer are all
  swappable.

## Requirements

| | |
|---|---|
| OS | Linux |
| Python | 3.12 |
| GPU | NVIDIA, for rendering and training |
| Disk | ~25GB for Isaac Sim, plus ~75MB per rendered clip |
| Tools | `ffmpeg`, `ffprobe`, [`uv`](https://docs.astral.sh/uv/) |

The Cosmos restyling stage needs a **40GB+ VRAM GPU** and ~300GB of model
checkpoints, so it runs on a separate machine. GRAFT exports a
self-contained job for it — see [Restyling with Cosmos](#restyling-with-cosmos).
Everything else runs on a normal workstation GPU.

## Install

```bash
git clone https://github.com/Adyansh04/graft && cd graft
make setup
```

That creates `.venv/` and installs the pipeline. Then install Isaac Sim into
its own environment (~25GB, one time, takes a while):

```bash
make setup-isaac
```

Isaac Sim prompts to accept the [NVIDIA Omniverse licence](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)
on first launch. The setup script sets `OMNI_KIT_ACCEPT_EULA=YES` on your
behalf; export `OMNI_KIT_ACCEPT_EULA=N` first if you would rather it did not.

Check everything is in order:

```bash
make doctor
```

```
graft doctor
  [ok] python: 3.12.3
  [ok] ffmpeg: /usr/bin/ffmpeg
  [ok] ffprobe: /usr/bin/ffprobe
  [ok] disk: 105GB free
  [ok] gpu: NVIDIA GeForce RTX 4080 Laptop GPU, 12282 MiB
  [ok] isaac venv: isaacsim 6.0.1.0
  [ok] isaac usd-core: absent (correct)

All checks passed.
```

## Quick start

Fetch the sample asset (a coffee mug) and run the whole pipeline:

```bash
uv run python scripts/fetch_asset.py mug
make pipeline-sim
```

That validates the model, renders clips, checks them, builds a dataset,
trains a detector and scores it. Results land in `runs/<name>/`.

To watch progress at any point:

```bash
make status
```

## Using your own object

You need a USD file. Isaac Sim's asset browser, NVIDIA SimReady assets and
converted glTF/OBJ models all work.

**1. Point the config at it.** Edit `configs/default.yaml`:

```yaml
run:
  name: my-object

asset:
  usd_path: assets/MyObject/model.usd
  target_prim_path: /Root/Geometry     # the prim holding the geometry
  expected_size_m: [0.05, 0.30]        # plausible size range, in metres

classes:
  - name: my_object
    color: [0, 255, 0, 255]
```

If you do not know the prim path, run the validator — it lists candidates
when the path is wrong.

**2. Check the asset is usable:**

```bash
make validate CONFIG=configs/default.yaml   # config only
uv run graft asset validate                 # the USD itself
```

```
asset: assets/Coffee_Mug_A01/sm_rc_dishware_mug_coffee_a01.usd
target: /RootNode/Geometry
metersPerUnit: 1.0   upAxis: Z
size: 0.121 x 0.081 x 0.097 m

PASS — 0 error(s), 1 warning(s)
```

Errors block the run: a missing prim path, no geometry, or a size far
outside `expected_size_m` (usually a units mistake). Warnings do not:
missing colliders, physics and semantic labels are all applied at render
time. One warning worth reading is **baked-in lighting** — if shadows and
highlights are painted into the texture, domain randomisation cannot vary
them, which limits how well the detector generalises. Scanned models often
have this; library assets usually do not.

**3. Run it:**

```bash
make pipeline-sim
```

## Commands

Each stage runs on its own and can be re-run independently.

| Command | What it does |
|---|---|
| `graft doctor` | Check the environment |
| `graft validate` | Validate the config file |
| `graft run init` | Create the run directory and snapshot the config |
| `graft status` | Show stage progress, clip counts, disk usage |
| `graft asset validate` | Check the USD model is usable |
| `graft sim probe` | Report what the installed Isaac Sim supports |
| `graft capture` | Render clips |
| `graft cosmos export` | Build a restyling job for a big-GPU machine |
| `graft cosmos import` | Bring restyled clips back in |
| `graft qa` | Check rendered frames, quarantine bad ones |
| `graft assemble` | Build the image dataset |
| `graft train` | Train the detector |
| `graft eval` | Score the model |

Every command takes `--config PATH` (default `configs/default.yaml`).
Stages that have already finished are skipped; `--force` redoes them along
with anything downstream.

The `make` targets wrap these: `make capture`, `make qa`, `make assemble`,
`make train`, `make eval`, and `make pipeline-sim` for all of them. Pass a
different config with `make capture CONFIG=configs/other.yaml`.

### Rendering

```bash
graft capture              # render every configured clip
graft capture --clips 2    # just two, for a quick check
graft capture --force      # re-render from scratch
graft capture --gui        # show the Isaac Sim window
```

Rendering is headless by default, which is faster.

**Interrupting is safe.** Each clip is verified before it is marked
complete, so a run killed mid-clip re-renders only that clip. Clip contents
are deterministic from `run.seed`, so a resumed run produces exactly what an
uninterrupted one would have.

### Evaluating on real photographs

Sim-only scores flatter the model: held-out simulated clips share a renderer
and lighting model with the training data, so a detector can score well on
them while failing on a real camera. Photographs are the honest measure.

Take 30–50 photos of the object in varied lighting and backgrounds, label
them, and arrange them like this:

```
real_photos/
  images/   photo_001.jpg, photo_002.jpg, ...
  labels/   photo_001.txt, photo_002.txt, ...
```

Labels are YOLO format — one line per object, `class cx cy w h`, all
normalised 0–1. Point the config at the directory:

```yaml
eval:
  real_photos_dir: real_photos
```

`graft eval` then reports both numbers side by side. The ingest step
validates the labels first and will tell you if, for example, coordinates
were written in pixels instead of normalised.

These photographs are used **only** for scoring. No hand-drawn label ever
enters training.

### Restyling with Cosmos

Optional, and the reason the project exists: it adds background, material
and lighting variety that the renderer cannot produce, which is what makes
the detector work outside the simulator.

It needs a GPU with 40GB+ VRAM, so it runs elsewhere:

```bash
graft cosmos export
```

This writes `runs/<name>/cosmos/bundle/` containing the control videos, a
generated prompt per clip, the inference settings and a `run.sh`. Copy it to
the GPU machine, follow its README, run `./run.sh`, then copy the `output/`
directory back and:

```bash
graft cosmos import
```

Import verifies resolution and frame counts before accepting anything —
silent resizing or a truncated clip would invalidate the labels. Then add
the restyled frames to the dataset:

```yaml
dataset:
  sources: [sim, cosmos]
```

Expect the sim-only score to *drop* slightly while the real-photo score
rises. That is the intended trade: the model stops overfitting to the
renderer.

## Configuration

One YAML file describes a run. It is validated on load — unknown keys are
rejected, so a typo fails immediately rather than silently using a default.
Each run stores its own copy, so editing the file mid-run cannot change a
run that is already going.

### `run`

| Key | Default | Meaning |
|---|---|---|
| `name` | — | Run name; also the directory under `out_root` |
| `seed` | `0` | Master seed. The same seed reproduces the same clips |
| `out_root` | `runs` | Where run directories are created |

### `asset`

| Key | Default | Meaning |
|---|---|---|
| `usd_path` | — | Path to the USD file |
| `target_prim_path` | — | Prim holding the object's geometry |
| `expected_size_m` | `[0.03, 0.40]` | Plausible size range in metres; catches unit mistakes |

### `classes`

An ordered list. **A class's position is its ID**, and that ordering is
shared by the labels, the dataset and the evaluation set — reordering it
invalidates existing runs.

```yaml
classes:
  - name: mug
    color: [0, 255, 0, 255]    # RGBA, used in the segmentation output
```

### `sim`

| Key | Default | Meaning |
|---|---|---|
| `resolution` | `[1280, 704]` | Render size. **Fixed** — required by Cosmos |
| `rt_subframes` | `6` | Rendering passes per frame; raise for cleaner images |
| `dlss_exec_mode` | `2` | 0 performance, 1 balanced, 2 quality, 3 auto |
| `settle.lin_vel_thresh` | `0.1` | Speed (m/s) below which the object counts as settled |
| `settle.ang_vel_thresh` | `0.1` | Rotation threshold |
| `settle.timeout_steps` | `600` | Give up settling after this many physics steps |

### `capture`

| Key | Default | Meaning |
|---|---|---|
| `n_clips` | `8` | Number of clips |
| `frames_per_clip` | `121` | **Fixed** — required by Cosmos |
| `negative_clip_fraction` | `0.0` | Fraction of clips rendered without the object |
| `camera.orbit_degrees` | `120.0` | Arc the camera sweeps per clip |
| `camera.elevation_deg` | `[15, 45]` | Camera height range, degrees above horizontal |
| `camera.radius_m` | `[0.4, 0.9]` | Distance from the object |
| `camera.focal_length` | `24.0` | Lens focal length |
| `camera.clipping_range` | `[0.01, 1000.0]` | Near/far planes. The near plane **must** be closer than the smallest orbit radius or nothing renders |
| `randomizers.lighting.dome_intensity` | `[300, 1500]` | Ambient light range |
| `randomizers.lighting.n_area_lights` | `[1, 3]` | Number of lamps |
| `randomizers.lighting.area_intensity` | `[1e4, 8e4]` | Lamp brightness range |
| `randomizers.lighting.off_probability` | `0.3` | Chance a lamp is switched off, for uneven lighting |
| `randomizers.distractors.count` | `[0, 4]` | Clutter objects per clip |
| `randomizers.distractors.pool_dir` | `null` | Directory of USD files to use as clutter |

Distractors are never labelled — they occlude and clutter without appearing
in the annotations, which is what teaches the detector to cope with mess.

### `cosmos`

| Key | Default | Meaning |
|---|---|---|
| `fps` | `30` | Frame rate of the control videos |
| `sigma_max` | `50.0` | How far the restyle may depart from the input; higher is freer |
| `guidance` | `5.0` | How closely the prompt is followed |
| `seed` | `1` | Restyle seed |
| `weights.vis` | `0.3` | Holds the original appearance and lighting. **Lower this for bigger lighting changes** |
| `weights.edge` | `0.3` | Holds outlines |
| `weights.depth` | `0.6` | Holds 3D structure |
| `weights.seg` | `0.7` | Holds object boundaries |
| `prompt.sections_file` | `configs/prompts.yaml` | Scene description phrases |
| `prompt.invariant` | — | Sentence stating what must not change, and what it looks like |

Weights must sum to 2.0 or less. `prompt.invariant` matters: describing the
object and saying it must stay unchanged is what keeps it stable while the
scene around it changes.

Edit `configs/prompts.yaml` to widen scene variety — one phrase is picked
per section per clip, so options multiply.

### `encode`

| Key | Default | Meaning |
|---|---|---|
| `crf` | `12` | Video quality, 0 best to 51 worst. Low, so compression does not leak into training data |
| `preset` | `slow` | Encoder speed/quality trade-off |
| `pix_fmt` | `yuv420p` | Pixel format |
| `fps` | `30` | Frame rate |

### `dataset`

| Key | Default | Meaning |
|---|---|---|
| `sources` | `[sim]` | `sim`, `cosmos`, or both |
| `stride` | `10` | Keep every Nth frame. Neighbouring frames are near-identical |
| `split.train` / `split.val` | `0.8` / `0.2` | Split proportions |
| `split.by` | `clip` | **Fixed.** Whole clips go to one split — splitting by frame would put near-duplicates in both and inflate the score |
| `formats` | `[yolo_detect]` | `yolo_detect`, `yolo_seg`, `coco`, `kitti` |

### `qa`

| Key | Default | Meaning |
|---|---|---|
| `blur_lap_var_min` | `60.0` | Reject frames blurrier than this |
| `min_bbox_area_px` | `64` | Reject boxes smaller than this |
| `cosmos.in_mask_ssim_min` | `0.7` | Reject restyled frames where the object changed too much |
| `cosmos.out_mask_change_min` | `0.05` | Reject restyled frames where the background did not change |
| `action` | `quarantine` | `quarantine` excludes them; `fail` stops the run |

A rejection rate above roughly 30% usually means a setting is wrong rather
than the thresholds being strict; `graft qa` says so when it sees one.

### `train`

| Key | Default | Meaning |
|---|---|---|
| `model` | `yolo11n.pt` | Starting weights. `yolo11s/m/l/x` are larger and slower |
| `epochs` | `10` | Training passes |
| `imgsz` | `640` | Training image size |
| `batch` | `8` | Batch size. Lower it if you run out of GPU memory |

### `eval`

| Key | Default | Meaning |
|---|---|---|
| `sim_val` | `true` | Score on held-out simulated clips |
| `real_photos_dir` | `null` | Directory of labelled photographs |

## Output

```
runs/<name>/
  config.snapshot.yaml   the config this run used
  manifest.json          stage progress
  clips/clip_0000/       rendered frames, videos and labels
  cosmos/                restyling job and its results
  qa/report.json         what was rejected, and why
  dataset/               images/ and labels/, plus dataset.yaml
  weights/               trained model
  eval/metrics.json      scores
```

Trained weights are at `runs/<name>/weights/train/weights/best.pt`, usable
directly with Ultralytics:

```python
from ultralytics import YOLO
YOLO("runs/my-object/weights/train/weights/best.pt").predict("photo.jpg")
```

## Disk use

About 75MB per clip. Most of that is the four control modalities, which are
also written as video — so once a clip is complete their frames are
redundant:

```bash
graft clean --controls --dry-run   # see what would go
graft clean --controls             # delete it
```

That reclaims roughly half, keeping the RGB frames the dataset is built from
and all the videos. Clips whose videos are missing are skipped rather than
stripped of their only copy, and a pruned clip still counts as complete, so
this never triggers a re-render.

## Troubleshooting

**`graft doctor` reports the Isaac venv missing.** Run `make setup-isaac`.
It is a large download; read its output rather than trusting the exit code.

**Everything renders empty.** Almost always the camera near plane. It must
be closer than `camera.radius_m`; the config validator now catches this.

**`ModuleNotFoundError: omni` / `isaacsim` / `pxr`.** A command ran in the
wrong environment. Use the `graft` CLI, which dispatches to the right one.

**Out of memory while training.** Lower `train.batch`, or `train.imgsz`.

**Cosmos import rejects clips.** It checks resolution and frame count. A
mismatch means the restyle changed the geometry, which would make the
labels wrong — the rejection is doing its job.

## Development

```bash
make test     # the full suite; no GPU or Isaac Sim needed
```

Most of the codebase is plain Python and testable without a simulator. Only
the rendering modules require Isaac Sim.

## Licence

The sample assets are third-party and carry their own licences —
`Coffee_Mug_A01` and YCB `025_mug` are both CC BY 4.0. See the `LICENSE`
file in each downloaded asset directory.
