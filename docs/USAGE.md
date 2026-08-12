# Usage

Task-oriented recipes. For an overview and the full configuration
reference, see the [README](../README.md).

## Contents

- [A first run](#a-first-run)
- [Working with your own object](#working-with-your-own-object)
- [Tuning the renders](#tuning-the-renders)
- [Getting more variety](#getting-more-variety)
- [Re-running and resuming](#re-running-and-resuming)
- [Several runs side by side](#several-runs-side-by-side)
- [Evaluating honestly](#evaluating-honestly)
- [Restyling with Cosmos](#restyling-with-cosmos)
- [Using the trained model](#using-the-trained-model)
- [Managing disk](#managing-disk)
- [Reading the output](#reading-the-output)
- [When things go wrong](#when-things-go-wrong)

## A first run

```bash
make setup           # core environment
make setup-isaac     # Isaac Sim, ~25GB, one time
make doctor          # verify
uv run python scripts/fetch_asset.py mug
make pipeline-sim
```

`pipeline-sim` chains: `init` → `capture` → `qa` → `assemble` → `train` →
`eval`. Roughly 5 minutes per clip on a laptop RTX 4080, so about 40 minutes
at the default 8 clips.

To try it faster, render two clips first:

```bash
uv run graft run init
uv run graft capture --clips 2
uv run graft qa && uv run graft assemble && uv run graft train
```

## Working with your own object

### 1. Get a USD file

Isaac Sim's asset browser, NVIDIA SimReady assets, or a converted
glTF/OBJ/FBX. Put it under `assets/` (which is not tracked by git).

### 2. Find the geometry prim

Point `asset.target_prim_path` at the prim containing the mesh. If you guess
wrong, the validator tells you what is available:

```
[error  ] missing-prim: target_prim_path '/World/Mug' does not exist.
          Candidates with geometry: ['/RootNode/Geometry/...']
```

### 3. Set a plausible size

`expected_size_m` is a sanity range for the object's largest dimension. It
catches the most common asset problem — a model authored in centimetres on a
stage declaring metres, which renders 100× too large:

```
[error  ] implausible-scale: largest dimension is 10.000 m, outside the
          expected 0.05-0.2 m. metersPerUnit is 1.0; the asset may be
          authored in different units than the stage declares.
```

### 4. Name the class

```yaml
classes:
  - name: stapler
    color: [0, 255, 0, 255]
```

The colour is used in the segmentation output. With one class the value
barely matters; with several, pick well-separated colours.

### 5. Validate, then run

```bash
uv run graft asset validate
uv run graft capture --clips 1     # eyeball one clip first
```

Look at `runs/<name>/clips/clip_0000/cosmos/*/rgb/` before committing to a
full render.

## Tuning the renders

**The object is too small or too large in frame.** Adjust
`capture.camera.radius_m`. Closer values fill more of the frame. Keep
`clipping_range[0]` below the smallest radius or nothing renders at all.

**Frames look noisy or speckled.** Raise `sim.rt_subframes` — 6 is a good
default, 16–32 gives cleaner images at proportionally more time.

**The lighting is too uniform.** Raise
`randomizers.lighting.off_probability` so more lamps are switched off, and
widen `area_intensity`. Uneven, sometimes-dark scenes teach more than evenly
lit ones.

**The object always looks the same way up.** It is dropped with a random
orientation and left to settle, so varied resting poses come for free. If
your object has one very stable base it will mostly land on it — that is
physically honest, and a camera orbit alone would be worse.

**I want the object partially hidden.** Point
`randomizers.distractors.pool_dir` at a directory of USD files. They are
placed around the object and never labelled, so they occlude and clutter
without appearing in annotations.

## Getting more variety

Diversity between clips is what makes a detector general. In rough order of
value:

1. **More clips.** `capture.n_clips`. Each is a fresh scene.
2. **Distractors**, as above.
3. **Wider lighting ranges** in `randomizers.lighting`.
4. **Wider camera ranges** — `elevation_deg`, `radius_m`, `orbit_degrees`.
5. **Cosmos restyling**, which adds background and material variety nothing
   else can.

Raising `dataset.stride` does *not* add variety — neighbouring frames in a
clip are near-identical, so a lower stride mostly adds duplicates. Prefer
more clips over denser sampling of the ones you have.

## Re-running and resuming

Every stage tracks its own state. Re-running a finished stage is a no-op:

```
capture already done; use --force to re-render
```

`--force` redoes a stage **and everything downstream of it**, since their
inputs changed:

```bash
uv run graft assemble --force    # also invalidates train and eval
```

**Interrupted renders resume automatically.** Clips are verified before
being marked complete, so `graft capture` after a crash re-renders only the
unfinished clip. Because clips are seeded from `run.seed`, the resumed run
produces exactly what an uninterrupted one would have.

To render more clips, raise `capture.n_clips` and re-run `graft capture` —
existing clips are kept and only the new ones render.

## Several runs side by side

Copy the config and change `run.name`:

```bash
cp configs/default.yaml configs/high-clip.yaml
# edit run.name and capture.n_clips
uv run graft capture --config configs/high-clip.yaml
```

Each run gets its own directory under `runs/`, so comparisons are clean.
Every `make` target accepts `CONFIG=`:

```bash
make capture CONFIG=configs/high-clip.yaml
```

## Evaluating honestly

By default, `graft eval` scores against held-out simulated clips. That
number tells you training converged — not that the detector works. It shares
a renderer, asset and lighting model with the training data.

**Photographs are the real test.** 30–50 of the physical object, varied
lighting and backgrounds:

```
real_photos/
  images/photo_001.jpg
  labels/photo_001.txt      ->  0 0.51 0.48 0.20 0.24
```

One line per object: `class cx cy w h`, normalised 0–1. Then:

```yaml
eval:
  real_photos_dir: real_photos
```

```bash
uv run graft eval --force
```

```
sim-val          mAP50=0.9950  mAP50-95=0.9110  (26 images)
real-photos      mAP50=0.6120  mAP50-95=0.4030  (42 images)
```

The gap between those two *is* the sim-to-real gap, and closing it is the
point of the project. The ingest step validates labels first and reports
problems — most commonly pixel coordinates where normalised ones were
expected.

These photographs are only ever used for scoring. No hand-drawn label enters
training.

## Restyling with Cosmos

Cosmos Transfer replaces the background, materials and lighting while
holding the object's geometry, which is what the labels depend on. It needs
a 40GB+ VRAM GPU, so it runs elsewhere.

```bash
uv run graft cosmos export
```

`runs/<name>/cosmos/bundle/` then contains:

```
videos/        control videos per clip
specs/         inference settings per clip
batch.jsonl    all clips in one batch
manifest.json  which output belongs to which clip
run.sh         the command to run
README.md      setup instructions
```

Copy it to the GPU machine, follow its README to fetch the model
checkpoints, run `./run.sh`, then copy `output/` back and:

```bash
uv run graft cosmos import
```

Import verifies each returned clip's resolution and frame count before
accepting it. A mismatch means the geometry moved and the labels no longer
describe the image, so it is rejected rather than silently used.

Then include the restyled frames:

```yaml
dataset:
  sources: [sim, cosmos]
```

```bash
uv run graft assemble --force && uv run graft train --force && uv run graft eval --force
```

### Tuning the restyle

The control weights trade fidelity against variety:

| Want | Change |
|---|---|
| More lighting change | Lower `weights.vis` — it preserves the source lighting |
| More background change | Raise `sigma_max`, lower `weights.vis` |
| Object drifting or deforming | Raise `weights.depth` and `weights.edge` |
| More prompt influence | Raise `guidance` |

Keep the total at or below 2.0. Widen scene variety by adding phrases to
`configs/prompts.yaml` — one is picked per section per clip, so options
multiply.

`prompt.invariant` is the sentence that keeps the object stable. Describe
what it looks like *and* say it must not change:

```yaml
invariant: >-
  The white stoneware coffee mug at the centre of the frame is unchanged:
  same shape, same position, same size, same smooth glazed white surface.
  Do not alter the mug in any way.
```

## Using the trained model

```python
from ultralytics import YOLO

model = YOLO("runs/mug-dev/weights/train/weights/best.pt")
results = model.predict("photo.jpg", conf=0.25)
results[0].show()
```

Standard Ultralytics weights — export to ONNX, TensorRT or CoreML as usual:

```bash
uv run yolo export model=runs/mug-dev/weights/train/weights/best.pt format=onnx
```

## Managing disk

About 75MB per clip. Check with `graft status`.

Once clips are complete, the four control modalities exist as both frames
and video, and only the RGB frames feed the dataset:

```bash
uv run graft clean --controls --dry-run
uv run graft clean --controls
```

Clips whose videos are missing are skipped rather than stripped of their
only copy, and pruned clips still count as complete — this never causes a
re-render.

## Reading the output

```
runs/mug-dev/
  config.snapshot.yaml     exactly what this run used
  manifest.json            stage states
  clips/clip_0000/
    cosmos/…/rgb/          rendered frames
    cosmos/…/*.mp4         per-modality video
    labels/bboxes_*.json   per-frame boxes
    labels/id_map.json     annotator id -> class
    clip.done              verification record
  qa/report.json           what was rejected, and why
  dataset/
    images/{train,val}/
    labels/{train,val}/
    dataset.yaml
  weights/train/weights/best.pt
  eval/metrics.json
```

`clip.done` is worth knowing about — it records the clip's seed, per-modality
frame counts and the label alignment offset. A clip without a valid one is
treated as incomplete and re-rendered.

## When things go wrong

**Every frame renders empty.** The camera near plane is further away than
the object. `capture.camera.clipping_range[0]` must be smaller than
`capture.camera.radius_m[0]`; the config validator enforces this.

**QA rejects everything.** Almost always a miscalibrated threshold rather
than bad frames. Check what it rejected in `qa/report.json`, then compare
against the actual values — clean renders of a smooth object on a plain
background have low `blur_lap_var` by nature, around 3–20.

**`ModuleNotFoundError: omni` / `isaacsim` / `pxr`.** Something ran in the
wrong environment. Use the `graft` CLI, which dispatches to the right one;
`make doctor` verifies both.

**Isaac Sim install fails.** It is a large multi-index download. Read the
script output rather than its exit code, and re-run — it is idempotent.

**Training runs out of memory.** Lower `train.batch`, then `train.imgsz`.

**`graft capture` says done but you want more clips.** Raise
`capture.n_clips` and re-run; only the new clips render.

**Cosmos import rejects clips.** It checks resolution and frame count
against what was exported. Rejection means the returned clip would not line
up with its labels.
