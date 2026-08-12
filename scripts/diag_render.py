"""Is the render pipeline broken, or is the asset invisible?

Renders a plain cube (known-good geometry) and the mug in the same launch.
If the cube shows and the mug does not, the asset is the problem; if neither
shows, the capture loop is.
"""

import json
import sys
from pathlib import Path

USD = "assets/Coffee_Mug_A01/sm_rc_dishware_mug_coffee_a01.usd"


def main() -> int:
    from graft.sim import bootstrap

    out = Path("runs/mug-dev/probe/render_diag.json").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    findings: dict = {}

    def save():
        out.write_text(json.dumps(findings, indent=2, default=str))

    app = bootstrap.launch(headless=True)
    try:
        _run(app, findings, save)
    except Exception as exc:  # noqa: BLE001
        import traceback

        findings["fatal"] = repr(exc)
        findings["traceback"] = traceback.format_exc()
    save()
    print(json.dumps(findings, indent=2, default=str))
    app.close()
    return 0


def _describe(rgb, depth, seg):
    import numpy as np

    out = {}
    a = np.asarray(rgb)
    out["rgb_unique"] = int(len(np.unique(a[..., :3])))
    out["rgb_mean"] = round(float(a[..., :3].mean()), 2)
    d = np.asarray(depth)
    out["depth_finite_frac"] = round(float(np.isfinite(d).mean()), 3)
    out["depth_nonzero_frac"] = round(float((np.nan_to_num(d) > 0).mean()), 4)
    s = np.asarray(seg)
    out["seg_unique"] = [int(v) for v in np.unique(s)][:6]
    return out


def _run(app, findings, save):
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.experimental.utils.semantics import add_labels
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Usd, UsdGeom

    from graft.sim import bootstrap

    bootstrap.apply_render_settings()
    bootstrap.prepare_replicator()
    stage = omni.usd.get_context().get_stage()

    # A light, so RGB is not black for reasons unrelated to geometry.
    rep.functional.create.dome_light(intensity=1000.0, name="Dome", parent="/World")

    # Known-good geometry at a known place.
    cube = rep.functional.create.cube(
        position=(0.0, 0.0, 0.0), scale=(0.1, 0.1, 0.1), name="RefCube", parent="/World"
    )
    add_labels(cube if not isinstance(cube, list) else cube[0], labels=["cube"], taxonomy="class")

    add_reference_to_stage(usd_path=str(Path(USD).resolve()), prim_path="/World/Mug")
    bootstrap.advance(app, 20)

    mug = stage.GetPrimAtPath("/World/Mug")
    findings["mug_is_valid"] = bool(mug and mug.IsValid())
    findings["mug_is_loaded"] = bool(mug and mug.IsLoaded())
    findings["mug_active"] = bool(mug and mug.IsActive())
    imageable = UsdGeom.Imageable(mug)
    findings["mug_visibility"] = (
        str(imageable.ComputeVisibility()) if imageable else "not imageable"
    )
    findings["stage_load_rules"] = str(stage.GetLoadRules())
    findings["unloaded_prims"] = [
        str(p.GetPath()) for p in stage.Traverse(Usd.PrimIsActive) if not p.IsLoaded()
    ][:10]
    findings["payload_prims"] = [
        str(p.GetPath()) for p in stage.Traverse() if p.HasAuthoredPayloads()
    ][:10]
    save()

    # Move the mug clear of the cube so both are in frame separately.
    rep.functional.modify.pose(mug, position_value=(0.35, 0.0, 0.0))
    for path in [str(p.GetPath()) for p in Usd.PrimRange(mug, Usd.TraverseInstanceProxies()) if p.IsInstance()]:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsInstance():
            prim.SetInstanceable(False)
    bootstrap.advance(app, 10)
    for p in Usd.PrimRange(mug):
        if p.GetTypeName() == "Mesh":
            add_labels(p, labels=["mug"], taxonomy="class")
    bootstrap.advance(app, 10)

    camera = rep.functional.create.camera(
        position=(0.9, 0.9, 0.7), look_at=(0.17, 0.0, 0.05), name="DiagCam", parent="/World"
    )
    rp = rep.create.render_product(camera, (1280, 704))

    rgb = rep.annotators.get("rgb")
    depth = rep.annotators.get("distance_to_camera")
    seg = rep.annotators.get("semantic_segmentation", init_params={"semanticTypes": ["class"]})
    bbox = rep.annotators.get("bounding_box_2d_tight", init_params={"semanticTypes": ["class"]})
    for a in (rgb, depth, seg, bbox):
        a.attach(rp)

    for label, subframes in (("first", 8), ("second", 16)):
        rep.orchestrator.step(rt_subframes=subframes, delta_time=0.0, pause_timeline=False)
        b = bbox.get_data()
        findings[f"render_{label}"] = {
            **_describe(rgb.get_data(), depth.get_data(), seg.get_data().get("data")),
            "n_boxes": int(len(b.get("data", []))),
            "bbox_idToLabels": {
                str(k): str(v) for k, v in (b.get("info", {}).get("idToLabels") or {}).items()
            },
            "seg_idToLabels": {
                str(k): str(v)
                for k, v in (seg.get_data().get("info", {}).get("idToLabels") or {}).items()
            },
        }
        save()


if __name__ == "__main__":
    sys.exit(main())
