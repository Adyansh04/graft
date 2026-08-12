"""Report what the installed Isaac Sim actually does.

Isaac Sim 6.0 changed the Replicator and semantics APIs, and the annotator
data layout is undocumented. Every tutorial in circulation predates it. This
prints the real shapes so later code is written against measured behaviour
rather than assumptions:

* annotator structured-array field names
* the `idToLabels` value format (6.0 returns dicts; older returns strings)
* where CosmosWriter actually writes, and what modalities it emits

Run via `graft sim probe`, which dispatches it into the Isaac venv.
"""

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--prim", required=True)
    parser.add_argument("--class-name", default="mug")
    parser.add_argument("--out", required=True, help="directory for probe output")
    parser.add_argument("--gui", action="store_true", help="run with the UI instead of headless")
    args = parser.parse_args(argv)

    from graft.sim import bootstrap

    # SimulationApp.close() does not return, so the report is written before
    # it and after every step that could fail.
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = out / "probe.json"

    findings: dict = {"usd": args.usd, "prim": args.prim}

    def save() -> None:
        report.write_text(json.dumps(findings, indent=2, default=str))

    save()
    app = bootstrap.launch(headless=not args.gui)
    try:
        _probe(app, args, findings, save)
    except Exception as exc:  # noqa: BLE001 - a probe records failures
        findings["fatal"] = repr(exc)
        import traceback

        findings["traceback"] = traceback.format_exc()
    save()
    print(f"\nprobe report: {report}")
    print(json.dumps(findings, indent=2, default=str))
    app.close()
    return 0


def _probe(app, args, findings: dict, save) -> None:
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Usd

    from graft.sim import bootstrap

    bootstrap.apply_render_settings()
    bootstrap.prepare_replicator()

    findings["isaacsim_version"] = _version()
    findings["replicator_version"] = getattr(rep, "__version__", "unknown")
    save()

    add_reference_to_stage(usd_path=str(Path(args.usd).resolve()), prim_path="/World/Target")
    stage = omni.usd.get_context().get_stage()
    bootstrap.advance(app, 10)

    findings["semantics_api"] = _apply_semantics(stage, "/World/Target", args.class_name)
    findings["physics_api"] = _apply_physics(stage, "/World/Target")

    prim = stage.GetPrimAtPath("/World/Target")
    findings["mesh_count"] = sum(
        1 for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()) if p.GetTypeName() == "Mesh"
    )
    save()

    camera = rep.functional.create.camera(position=(0.6, 0.6, 0.4), look_at=(0.0, 0.0, 0.0))
    render_product = rep.create.render_product(camera, (1280, 704))
    findings["annotators"] = _probe_annotators(app, render_product)
    save()
    findings["cosmos_writer"] = _probe_cosmos_writer(app, render_product, Path(args.out).resolve())
    save()


def _version() -> str:
    try:
        import isaacsim

        return getattr(isaacsim, "__version__", "unknown")
    except ImportError:
        return "unknown"


def _apply_semantics(stage, prim_path: str, class_name: str) -> dict:
    """Try the 6.0 API first, then the deprecated one, and report which won."""
    prim = stage.GetPrimAtPath(prim_path)
    attempts = {}

    try:
        from isaacsim.core.experimental.utils.semantics import add_labels

        add_labels(prim, labels=[class_name], taxonomy="class")
        attempts["isaacsim.core.experimental.utils.semantics.add_labels"] = "ok"
    except Exception as exc:  # noqa: BLE001 - probing, report anything
        attempts["isaacsim.core.experimental.utils.semantics.add_labels"] = repr(exc)

    try:
        import omni.replicator.core as rep

        rep.functional.modify.semantics(prim, {"class": class_name}, mode="add")
        attempts["rep.functional.modify.semantics"] = "ok"
    except Exception as exc:  # noqa: BLE001
        attempts["rep.functional.modify.semantics"] = repr(exc)

    attempts["applied_schemas"] = [str(s) for s in prim.GetAppliedSchemas()]
    return attempts


def _apply_physics(stage, prim_path: str) -> dict:
    attempts = {}
    try:
        import omni.replicator.core as rep

        # Takes prim objects, not path strings.
        rep.functional.physics.apply_rigid_body(
            stage.GetPrimAtPath(prim_path), with_collider=True
        )
        attempts["rep.functional.physics.apply_rigid_body"] = "ok"
    except Exception as exc:  # noqa: BLE001
        attempts["rep.functional.physics.apply_rigid_body"] = repr(exc)
    return attempts


def _probe_annotators(app, render_product) -> dict:
    """The important part: real field names and the idToLabels format."""
    import omni.replicator.core as rep

    out: dict = {}
    for name, init_params in (
        ("bounding_box_2d_tight", {"semanticTypes": ["class"]}),
        ("instance_segmentation", {"semanticTypes": ["class"]}),
        ("rgb", None),
    ):
        try:
            annotator = rep.annotators.get(name, init_params=init_params) if init_params else rep.annotators.get(name)
            annotator.attach(render_product)
        except Exception as exc:  # noqa: BLE001
            out[name] = {"attach_error": repr(exc)}
            continue

        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
        try:
            data = annotator.get_data()
        except Exception as exc:  # noqa: BLE001
            out[name] = {"get_data_error": repr(exc)}
            annotator.detach()
            continue

        out[name] = _describe(data)
        annotator.detach()
    return out


def _describe(data) -> dict:
    described: dict = {"type": type(data).__name__}
    if isinstance(data, dict):
        described["keys"] = sorted(str(k) for k in data)
        payload = data.get("data")
        info = data.get("info") or {}
        described["info_keys"] = sorted(str(k) for k in info)
        id_to_labels = info.get("idToLabels")
        if id_to_labels is not None:
            described["idToLabels"] = {str(k): repr(v) for k, v in list(id_to_labels.items())[:8]}
            described["idToLabels_value_type"] = type(
                next(iter(id_to_labels.values()), None)
            ).__name__
    else:
        payload = data

    dtype = getattr(payload, "dtype", None)
    if dtype is not None:
        described["dtype_names"] = list(dtype.names) if dtype.names else str(dtype)
    described["shape"] = getattr(payload, "shape", None)
    return described


def _probe_cosmos_writer(app, render_product, out_dir: Path) -> dict:
    """CosmosWriter's on-disk layout is undocumented, so record what appears."""
    import omni.replicator.core as rep

    from graft.sim import bootstrap

    # Absolute: DiskBackend resolves a relative path against Replicator's own
    # default output root, not the process CWD.
    target = (out_dir / "cosmos_probe").resolve()
    result: dict = {"output_dir": str(target)}
    try:
        writer = rep.WriterRegistry.get("CosmosWriter")
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=str(target))
        writer.initialize(
            backend=backend,
            use_instance_id=False,
            segmentation_mapping={"mug": [0, 255, 0, 255]},
            canny_threshold_low=10,
            canny_threshold_high=100,
        )
        writer.attach(render_product)
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
        return result

    for _ in range(3):
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
    rep.orchestrator.wait_until_complete()
    # Documented issue: standalone runs can skip mp4 generation unless the
    # app is pumped before detach.
    bootstrap.advance(app, 10)
    writer.detach()

    result["files"] = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file())
    result["init_signature_accepted"] = True
    return result


if __name__ == "__main__":
    sys.exit(main())
