"""Clip capture: the Isaac Sim half of the pipeline.

Per clip: sample a scene, drop the object and let it settle, then orbit the
camera for exactly 121 frames with two writers attached to one render
product — CosmosWriter for control modalities, ours for labels.

Writers are created and attached per clip rather than using CosmosWriter's
`next_clip()`. `next_clip()` advances an internal counter with no way to
seed it, so a resumed run would restart numbering at zero and overwrite
earlier clips. Attach cost is seconds against minutes of rendering.

Runs in the Isaac Sim venv, launched by `graft capture` via a subprocess.
"""

import argparse
import sys
from pathlib import Path

from graft.config.loader import load_config
from graft.run.manifest import clip_is_complete, write_clip_done
from graft.run.paths import RunPaths
from graft.sim import bootstrap
from graft.sim.camera import orbit_trajectory
from graft.sim.randomize import SceneParams, clip_seeds, sample_scene

TARGET_PRIM = "/World/Target"
COSMOS_MODALITIES = ("rgb", "depth", "segmentation", "shaded_seg", "edges")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to config.snapshot.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--clips", type=int, default=None, help="override clip count")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    paths = RunPaths(Path(args.run_dir))
    paths.create()

    n_clips = args.clips or config.capture.n_clips
    frames = config.capture.frames_per_clip
    seeds = clip_seeds(config.run.seed, n_clips)

    pending = [i for i in range(n_clips) if not clip_is_complete(paths.clip_done(i), frames)]
    if not pending:
        print(f"all {n_clips} clips already complete")
        return 0
    print(f"{n_clips - len(pending)}/{n_clips} clips complete; rendering {pending}")

    app = bootstrap.launch(headless=not args.gui)
    try:
        _prepare_stage(app, config)
        for index in pending:
            _render_clip(app, config, paths, index, seeds[index], frames)
    finally:
        app.close()
    return 0


def _prepare_stage(app, config) -> None:
    """One-time setup: ground, physics scene, the target asset, semantics."""
    from isaacsim.core.utils.stage import add_reference_to_stage

    from graft.sim import scene, semantics

    bootstrap.apply_render_settings(config.sim.dlss_exec_mode)
    bootstrap.prepare_replicator()

    scene.build_static_scene()
    add_reference_to_stage(
        usd_path=str(Path(config.asset.usd_path).resolve()), prim_path=TARGET_PRIM
    )
    bootstrap.advance(app, 10)

    class_name = config.classes[0].name
    semantics.label_target(TARGET_PRIM, class_name)
    stripped = semantics.strip_foreign_labels(TARGET_PRIM)
    print(f"labelled {TARGET_PRIM} as {class_name!r}; stripped labels from {stripped} other prim(s)")

    import omni.replicator.core as rep

    rep.functional.physics.apply_rigid_body(_prim(TARGET_PRIM), with_collider=True)


def _render_clip(app, config, paths: RunPaths, index: int, seed: int, frames: int) -> None:
    import omni.replicator.core as rep

    from graft.sim import scene, settle
    from graft.sim.writers.label_writer import GraftLabelWriter

    clip_dir = paths.clip(index)
    if clip_dir.exists():
        _remove_tree(clip_dir)
    clip_dir.mkdir(parents=True)

    params = sample_scene(config.capture, seed)
    dynamic = _apply_scene(params, config)

    result = settle.drop_and_settle(
        TARGET_PRIM,
        lin_vel_thresh=config.sim.settle.lin_vel_thresh,
        ang_vel_thresh=config.sim.settle.ang_vel_thresh,
        timeout_steps=config.sim.settle.timeout_steps,
    )
    if not result.settled:
        print(
            f"clip {index}: object still moving at {result.final_speed:.3f} m/s after "
            f"{result.steps} steps; capturing anyway"
        )

    camera = rep.functional.create.camera(
        position=(1.0, 0.0, 0.5), look_at=(0.0, 0.0, 0.0), name=f"Camera_{index}", parent="/World"
    )
    render_product = rep.create.render_product(camera, tuple(config.sim.resolution))

    cosmos_writer, cosmos_backend = _attach_cosmos_writer(clip_dir, config)
    label_writer = _attach_label_writer(clip_dir, config, render_product)
    cosmos_writer.attach(render_product)

    target_prim = _prim(TARGET_PRIM)
    poses = orbit_trajectory(params.orbit, frames)
    for frame_index, pose in enumerate(poses):
        rep.functional.modify.pose(
            camera,
            position_value=pose.position,
            look_at_value=target_prim,
            look_at_up_axis=(0.0, 0.0, 1.0),
        )
        rep.orchestrator.step(
            rt_subframes=config.sim.rt_subframes, delta_time=0.0, pause_timeline=False
        )
        if index == 0 and frame_index == 0:
            _assert_labels_took(clip_dir, config.classes[0].name)

    rep.orchestrator.wait_until_complete()
    # Standalone runs can skip mp4 generation unless the app is pumped
    # before detach.
    bootstrap.advance(app, 10)
    cosmos_writer.detach()
    label_writer.detach()
    render_product.destroy()
    scene.clear_dynamic_prims(dynamic)

    _finalise_clip(paths, index, seed, frames, params)


def _apply_scene(params: SceneParams, config) -> list[str]:
    import omni.replicator.core as rep

    from graft.sim import scene

    lights = scene.apply_lighting(params)
    distractor_dir = config.capture.randomizers.distractors.pool_dir
    usds = sorted(str(p) for p in Path(distractor_dir).glob("*.usd")) if distractor_dir else []
    placed = scene.place_distractors(params, usds)

    rep.functional.modify.pose(
        _prim(TARGET_PRIM),
        position_value=params.target_drop.position,
        rotation_value=params.target_drop.rotation_deg,
    )
    return lights + placed


def _prim(path: str):
    import omni.usd

    return omni.usd.get_context().get_stage().GetPrimAtPath(path)


def _attach_cosmos_writer(clip_dir: Path, config):
    import omni.replicator.core as rep

    backend = rep.backends.get("DiskBackend")
    # Absolute: DiskBackend resolves a relative path against Replicator's own
    # default output root (~/omni.replicator_out), not the process CWD.
    backend.initialize(output_dir=str((clip_dir / "cosmos").resolve()))
    writer = rep.WriterRegistry.get("CosmosWriter")
    writer.initialize(
        backend=backend,
        # Fixed per-class colours rather than per-mesh instance ids, so
        # class identity stays stable across clips.
        use_instance_id=False,
        segmentation_mapping=config.segmentation_mapping(),
        canny_threshold_low=10,
        canny_threshold_high=100,
    )
    return writer, backend


def _attach_label_writer(clip_dir: Path, config, render_product):
    import omni.replicator.core as rep

    from graft.sim.writers.label_writer import GraftLabelWriter, register

    if "GraftLabelWriter" not in rep.WriterRegistry.get_writers():
        register()
    writer = rep.writers.get("GraftLabelWriter")
    writer.initialize(output_dir=str(clip_dir / "labels"), class_names=config.class_names())
    writer.attach(render_product)
    return writer


def _assert_labels_took(clip_dir: Path, class_name: str) -> None:
    """Fail loudly if the semantic label did not reach the annotator.

    A whole run producing frames with no usable labels is the expensive
    failure this prevents.
    """
    import json

    id_map = clip_dir / "labels" / "id_map.json"
    if not id_map.is_file():
        raise RuntimeError(
            "label writer produced no id_map.json on the first frame — the label "
            "writer did not run"
        )
    mapping = json.loads(id_map.read_text())
    labels = set(mapping.get("bounding_box_2d_tight", {}).values())
    if class_name not in labels:
        raise RuntimeError(
            f"semantic label {class_name!r} did not reach the annotator "
            f"(idToLabels held {labels or 'nothing'}). Every frame would be unlabelled."
        )


def _finalise_clip(paths: RunPaths, index: int, seed: int, frames: int, params: SceneParams) -> None:
    """Verify the clip in-process, then record it. Only verified clips get a
    marker, so resume re-renders anything partial."""
    clip_dir = paths.clip(index)
    cosmos_dir = clip_dir / "cosmos"

    counts = {
        modality: len(list(cosmos_dir.rglob(f"{modality}/*.png")))
        for modality in COSMOS_MODALITIES
    }
    label_count = len(list((clip_dir / "labels").glob("bboxes_*.json")))
    # CosmosWriter's layout is undocumented, so record what it wrote rather
    # than assuming a structure downstream.
    outputs = {
        "png": sorted(str(p.relative_to(clip_dir)) for p in cosmos_dir.rglob("*.png")),
        "mp4": sorted(str(p.relative_to(clip_dir)) for p in cosmos_dir.rglob("*.mp4")),
    }

    problems = [f"{m}={c}" for m, c in counts.items() if c != frames]
    if label_count != frames:
        problems.append(f"labels={label_count}")
    if problems:
        print(f"clip {index}: INCOMPLETE (expected {frames} each) — {', '.join(problems)}")
        return

    write_clip_done(
        paths.clip_done(index),
        index=index,
        seed=seed,
        modality_counts=counts,
        label_count=label_count,
        outputs=outputs,
        negative=params.negative,
    )
    print(f"clip {index}: {frames} frames x {len(counts)} modalities, {len(outputs['mp4'])} mp4(s)")


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
