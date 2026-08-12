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
