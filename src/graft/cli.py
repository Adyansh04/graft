"""GRAFT command line.

Heavy imports stay inside the subcommand handlers — `graft doctor` should not
pay for ultralytics or usd-core just to check whether ffmpeg exists.
"""

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return proc.stdout.strip() or None if proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def cmd_doctor(args: argparse.Namespace) -> int:
    from graft.env import check_environment

    checks = check_environment()
    print("graft doctor")
    for check in checks:
        print(check.render())
    failed = [c for c in checks if not c.ok]
    # The Isaac venv is expected to be missing until M2, so its absence is
    # reported but does not fail the check.
    blocking = [c for c in failed if not c.name.startswith("isaac")]
    if blocking:
        print(f"\n{len(blocking)} blocking issue(s).")
        return 1
    if failed:
        print(f"\n{len(failed)} non-blocking issue(s) (Isaac Sim env not set up yet).")
    else:
        print("\nAll checks passed.")
    return 0


def cmd_run_init(args: argparse.Namespace) -> int:
    from graft.config.loader import config_hash, load_config, section_hashes, snapshot_config
    from graft.run.manifest import Manifest
    from graft.run.paths import RunPaths

    config = load_config(args.config)
    paths = RunPaths.for_run(config.run.out_root, config.run.name)
    if paths.manifest.exists() and not args.force:
        print(f"Run already initialised: {paths.root}", file=sys.stderr)
        print("Use --force to re-snapshot the config, or pick a different run.name.", file=sys.stderr)
        return 1

    paths.create()
    snapshot_config(config, paths.config_snapshot)
    manifest = Manifest(
        run_name=config.run.name,
        config_hash=config_hash(config),
        section_hashes=section_hashes(config),
        graft_git_sha=_git_sha(),
        created_at=_now(),
    )
    manifest.save(paths.manifest)
    print(f"Initialised run '{config.run.name}' at {paths.root}")
    print(f"  config snapshot: {paths.config_snapshot}")
    print(f"  classes: {', '.join(f'{i}={c.name}' for i, c in enumerate(config.classes))}")
    print(f"  clips planned: {config.capture.n_clips} x {config.capture.frames_per_clip} frames")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from graft.config.loader import load_config
    from graft.run.manifest import Manifest, Stage
    from graft.run.paths import RunPaths

    config = load_config(args.config)
    paths = RunPaths.for_run(config.run.out_root, config.run.name)
    if not paths.manifest.exists():
        print(f"No run at {paths.root}. Run 'graft run init' first.", file=sys.stderr)
        return 1

    manifest = Manifest.load(paths.manifest)
    print(f"run: {manifest.run_name}   ({paths.root})")
    print(f"created: {manifest.created_at}   graft: {manifest.graft_git_sha or 'unknown'}")
    print("\nstages:")
    for stage in Stage:
        state = manifest.stages[stage]
        detail = f"  — {state.detail}" if state.detail else ""
        print(f"  {stage.value:<15} {state.status.value}{detail}")

    expected = config.capture.frames_per_clip
    complete = manifest.completed_clips(paths, expected)
    present = paths.existing_clips()
    if present:
        partial = sorted(set(present) - complete)
        print(f"\nclips: {len(complete)}/{config.capture.n_clips} complete")
        if partial:
            print(f"  partial (will be re-rendered on resume): {partial}")

    size = _dir_size_mb(paths.root)
    if size is not None:
        print(f"\ndisk: {size:.0f} MB")
    return 0


def _dir_size_mb(path: Path) -> float | None:
    if not path.is_dir():
        return None
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024**2


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate the config alone. Cheap gate before a long GPU job."""
    from graft.config.loader import load_config

    config = load_config(args.config)
    print(f"config OK: {args.config}")
    print(f"  run: {config.run.name}   seed: {config.run.seed}")
    print(f"  asset: {config.asset.usd_path} @ {config.asset.target_prim_path}")
    print(f"  classes: {config.class_names()}")
    print(f"  render: {tuple(config.sim.resolution)} x {config.capture.frames_per_clip} frames/clip")
    return 0


def cmd_asset_validate(args: argparse.Namespace) -> int:
    from graft.assets.local import LocalUSDSource
    from graft.assets.validate import validate_asset
    from graft.config.loader import load_config

    config = load_config(args.config)
    source = LocalUSDSource(
        config.asset.usd_path,
        config.asset.target_prim_path,
        config.classes[0].name,
    )
    asset = source.resolve()
    report = validate_asset(
        asset.usd_path, asset.target_prim_path, tuple(config.asset.expected_size_m)
    )
    print(report.render())
    return 0 if report.ok else 1


def cmd_sim_probe(args: argparse.Namespace) -> int:
    from graft.config.loader import load_config
    from graft.env import run_in_isaac
    from graft.run.paths import RunPaths

    config = load_config(args.config)
    paths = RunPaths.for_run(config.run.out_root, config.run.name)
    out = paths.root / "probe"
    out.mkdir(parents=True, exist_ok=True)

    probe_args = [
        "--usd", config.asset.usd_path,
        "--prim", config.asset.target_prim_path,
        "--class-name", config.classes[0].name,
        "--out", str(out),
    ]
    if args.gui:
        probe_args.append("--gui")
    return run_in_isaac("graft.sim.probe", probe_args)


def cmd_capture(args: argparse.Namespace) -> int:
    from graft.config.loader import load_config
    from graft.env import run_in_isaac
    from graft.run.manifest import Manifest, Stage, Status
    from graft.run.paths import RunPaths

    config = load_config(args.config)
    paths = RunPaths.for_run(config.run.out_root, config.run.name)
    if not paths.manifest.exists():
        print("No run initialised. Run 'graft run init' first.", file=sys.stderr)
        return 1

    manifest = Manifest.load(paths.manifest)
    if manifest.is_done(Stage.CAPTURE) and not args.force:
        print("capture already done; use --force to re-render")
        return 0
    if args.force:
        manifest.force(Stage.CAPTURE)

    manifest.mark(Stage.CAPTURE, Status.RUNNING, now=_now())
    manifest.save(paths.manifest)

    capture_args = ["--config", str(paths.config_snapshot), "--run-dir", str(paths.root)]
    if args.gui:
        capture_args.append("--gui")
    if args.clips:
        capture_args += ["--clips", str(args.clips)]

    code = run_in_isaac("graft.sim.capture", capture_args)

    manifest = Manifest.load(paths.manifest)
    complete = manifest.completed_clips(paths, config.capture.frames_per_clip)
    expected = args.clips or config.capture.n_clips
    if code == 0 and len(complete) >= expected:
        manifest.mark(Stage.CAPTURE, Status.DONE, f"{len(complete)} clips", now=_now())
    else:
        manifest.mark(
            Stage.CAPTURE, Status.FAILED, f"{len(complete)}/{expected} clips", now=_now()
        )
    manifest.save(paths.manifest)
    return code


def _run_stage(args, stage, work):
    """Load the run, mark the stage, run `work`, record the outcome."""
    from graft.config.loader import load_config
    from graft.run.manifest import Manifest, Status
    from graft.run.paths import RunPaths

    config = load_config(args.config)
    paths = RunPaths.for_run(config.run.out_root, config.run.name)
    if not paths.manifest.exists():
        print("No run initialised. Run 'graft run init' first.", file=sys.stderr)
        return 1

    manifest = Manifest.load(paths.manifest)
    if manifest.is_done(stage) and not getattr(args, "force", False):
        print(f"{stage.value} already done; use --force to redo it")
        return 0
    if getattr(args, "force", False):
        manifest.force(stage)

    manifest.mark(stage, Status.RUNNING, now=_now())
    manifest.save(paths.manifest)
    try:
        detail = work(config, paths)
    except Exception as exc:  # noqa: BLE001 - record the failure, then surface it
        manifest = Manifest.load(paths.manifest)
        manifest.mark(stage, Status.FAILED, str(exc), now=_now())
        manifest.save(paths.manifest)
        raise

    manifest = Manifest.load(paths.manifest)
    manifest.mark(stage, Status.DONE, detail, now=_now())
    manifest.save(paths.manifest)
    return 0


def cmd_qa(args: argparse.Namespace) -> int:
    from graft.qa.gate import run_qa
    from graft.run.manifest import Stage

    def work(config, paths):
        report = run_qa(config, paths)
        print(report.render())
        return f"{len(report.quarantined)}/{report.checked} quarantined"

    return _run_stage(args, Stage.QA, work)


def cmd_assemble(args: argparse.Namespace) -> int:
    from graft.run.manifest import Stage

    def work(config, paths):
        import graft.dataset.formats.yolo_detect  # noqa: F401
        import graft.dataset.formats.yolo_seg  # noqa: F401
        from graft.dataset.assemble import assemble

        result = assemble(config, paths)
        print(result.render())
        return ", ".join(f"{k}={v}" for k, v in sorted(result.counts.items()))

    return _run_stage(args, Stage.ASSEMBLE, work)


def cmd_train(args: argparse.Namespace) -> int:
    from graft.run.manifest import Stage

    def work(config, paths):
        import graft.train.ultralytics_backend  # noqa: F401
        from graft.train.base import get_trainer

        dataset_yaml = paths.dataset / "dataset.yaml"
        if not dataset_yaml.is_file():
            raise RuntimeError(f"{dataset_yaml} missing; run 'graft assemble' first")

        result = get_trainer("ultralytics").train(
            dataset_yaml,
            paths.weights,
            model=config.train.model,
            epochs=config.train.epochs,
            imgsz=config.train.imgsz,
            batch=config.train.batch,
        )
        print(f"weights: {result.weights}")
        return f"{result.epochs} epochs, {config.train.model}"

    return _run_stage(args, Stage.TRAIN, work)


def cmd_eval(args: argparse.Namespace) -> int:
    from graft.run.manifest import Stage

    def work(config, paths):
        from graft.eval.evaluate import evaluate, render

        weights = Path(args.weights) if args.weights else paths.weights / "train" / "weights" / "best.pt"
        if not weights.is_file():
            raise RuntimeError(f"weights not found at {weights}; run 'graft train' first")
        payload = evaluate(config, paths, weights)
        print(render(payload))
        return ", ".join(payload.get("targets", {}))

    return _run_stage(args, Stage.EVAL, work)


def cmd_cosmos_export(args: argparse.Namespace) -> int:
    from graft.run.manifest import Stage

    def work(config, paths):
        from graft.cosmos.bundle import export_bundle

        result = export_bundle(config, paths)
        print(result.render())
        return f"{len(result.clips)} clips"

    return _run_stage(args, Stage.COSMOS_EXPORT, work)


def cmd_cosmos_import(args: argparse.Namespace) -> int:
    from graft.run.manifest import Stage

    def work(config, paths):
        from graft.cosmos.importer import import_outputs

        result = import_outputs(
            config, paths, Path(args.output) if args.output else None
        )
        print(result.render())
        if result.rejected and not result.imported:
            raise RuntimeError("no clips imported; see the rejections above")
        return f"{len(result.imported)} imported, {len(result.rejected)} rejected"

    return _run_stage(args, Stage.COSMOS_IMPORT, work)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graft", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--config", "-c", default="configs/default.yaml", help="path to the YAML config")
        return p

    doctor = sub.add_parser("doctor", help="check the environment is sane")
    doctor.set_defaults(func=cmd_doctor)

    validate = with_config(sub.add_parser("validate", help="validate the config and exit"))
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="run directory management")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    run_init = with_config(run_sub.add_parser("init", help="create a run directory and snapshot the config"))
    run_init.add_argument("--force", action="store_true", help="re-snapshot an existing run")
    run_init.set_defaults(func=cmd_run_init)

    status = with_config(sub.add_parser("status", help="show stage and clip progress"))
    status.set_defaults(func=cmd_status)

    sim = sub.add_parser("sim", help="Isaac Sim stages")
    sim_sub = sim.add_subparsers(dest="sim_command", required=True)
    sim_probe = with_config(
        sim_sub.add_parser("probe", help="report what the installed Isaac Sim actually does")
    )
    sim_probe.add_argument("--gui", action="store_true", help="show the UI instead of headless")
    sim_probe.set_defaults(func=cmd_sim_probe)

    capture = with_config(sub.add_parser("capture", help="render clips in Isaac Sim"))
    capture.add_argument("--gui", action="store_true", help="show the UI instead of headless")
    capture.add_argument("--clips", type=int, help="override the configured clip count")
    capture.add_argument("--force", action="store_true", help="re-render completed clips")
    capture.set_defaults(func=cmd_capture)

    for name, help_text, handler in (
        ("qa", "check rendered frames and record quarantines", cmd_qa),
        ("assemble", "build the image dataset from captured clips", cmd_assemble),
        ("train", "train the detector", cmd_train),
    ):
        stage_parser = with_config(sub.add_parser(name, help=help_text))
        stage_parser.add_argument("--force", action="store_true", help="redo a completed stage")
        stage_parser.set_defaults(func=handler)

    evaluate = with_config(sub.add_parser("eval", help="score the model on sim-val and real photos"))
    evaluate.add_argument("--force", action="store_true", help="redo a completed stage")
    evaluate.add_argument("--weights", help="override the weights path")
    evaluate.set_defaults(func=cmd_eval)

    cosmos = sub.add_parser("cosmos", help="Cosmos Transfer augmentation")
    cosmos_sub = cosmos.add_subparsers(dest="cosmos_command", required=True)

    cosmos_export = with_config(
        cosmos_sub.add_parser("export", help="build a self-contained job bundle for a GPU machine")
    )
    cosmos_export.add_argument("--force", action="store_true")
    cosmos_export.set_defaults(func=cmd_cosmos_export)

    cosmos_import = with_config(
        cosmos_sub.add_parser("import", help="validate and decode returned Cosmos output")
    )
    cosmos_import.add_argument("--output", help="directory holding video_N/output.mp4")
    cosmos_import.add_argument("--force", action="store_true")
    cosmos_import.set_defaults(func=cmd_cosmos_import)

    asset = sub.add_parser("asset", help="asset intake")
    asset_sub = asset.add_subparsers(dest="asset_command", required=True)
    asset_validate = with_config(
        asset_sub.add_parser("validate", help="check a USD asset is usable before rendering")
    )
    asset_validate.set_defaults(func=cmd_asset_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
