"""Why does a labelled prim produce no annotations?

Tests, in one Isaac launch:
  1. is the object in frame at all
  2. semantics on the reference root (what capture does today)
  3. semantics on the mesh reached through an instance proxy
  4. semantics after clearing the instanceable flag

Run: uv run python scripts/diag_semantics.py  (dispatches into .venv-isaac)
"""

import json
import sys
from pathlib import Path

USD = "assets/Coffee_Mug_A01/sm_rc_dishware_mug_coffee_a01.usd"
ROOT = "/World/Target"


def main() -> int:
    from graft.sim import bootstrap

    out = Path("runs/mug-dev/probe/semantics_diag.json").resolve()
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


def _run(app, findings, save):
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.experimental.utils.semantics import add_labels, get_labels
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Usd, UsdGeom

    from graft.sim import bootstrap

    bootstrap.apply_render_settings()
    bootstrap.prepare_replicator()

    add_reference_to_stage(usd_path=str(Path(USD).resolve()), prim_path=ROOT)
    stage = omni.usd.get_context().get_stage()
    bootstrap.advance(app, 10)

    root = stage.GetPrimAtPath(ROOT)

    # Where is it, and how big?
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound = cache.ComputeWorldBound(root).ComputeAlignedRange()
    findings["world_bounds"] = {
        "min": [float(v) for v in bound.GetMin()],
        "max": [float(v) for v in bound.GetMax()],
        "size": [float(v) for v in bound.GetSize()],
        "centre": [float(v) for v in bound.GetMidpoint()],
    }

    meshes = [
        str(p.GetPath())
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if p.GetTypeName() == "Mesh"
    ]
    instance_roots = [
        str(p.GetPath())
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if p.IsInstance()
    ]
    findings["meshes"] = meshes
    findings["instance_roots"] = instance_roots
    save()

    centre = bound.GetMidpoint()
    target = (float(centre[0]), float(centre[1]), float(centre[2]))
    radius = max(float(v) for v in bound.GetSize()) * 4 or 1.0

    camera = rep.functional.create.camera(
        position=(target[0] + radius, target[1] + radius, target[2] + radius),
        look_at=target,
        name="DiagCam",
        parent="/World",
    )
    render_product = rep.create.render_product(camera, (1280, 704))

    bbox = rep.annotators.get("bounding_box_2d_tight", init_params={"semanticTypes": ["class"]})
    seg = rep.annotators.get("instance_segmentation", init_params={"semanticTypes": ["class"]})
    bbox.attach(render_product)
    seg.attach(render_product)

    def sample(label: str):
        rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=False)
        b = bbox.get_data()
        s = seg.get_data()
        mask = np.asarray(s.get("data"))
        result = {
            "n_boxes": int(len(b.get("data", []))),
            "idToLabels": {str(k): str(v) for k, v in (b.get("info", {}).get("idToLabels") or {}).items()},
            "seg_unique_ids": [int(v) for v in np.unique(mask)][:8],
            "seg_idToLabels": {
                str(k): str(v) for k, v in (s.get("info", {}).get("idToLabels") or {}).items()
            },
        }
        findings[label] = result
        save()
        return result

    sample("1_no_semantics")

    # 2: label the reference root, which is what capture does today
    add_labels(root, labels=["mug"], taxonomy="class")
    findings["labels_on_root"] = get_labels(root)
    sample("2_semantics_on_root")

    # 3: label the mesh reached through the instance proxy
    proxy_error = None
    if meshes:
        try:
            add_labels(stage.GetPrimAtPath(meshes[0]), labels=["mug"], taxonomy="class")
        except Exception as exc:  # noqa: BLE001
            proxy_error = repr(exc)
    findings["mesh_label_error"] = proxy_error
    sample("3_semantics_on_mesh_proxy")

    # 4: clear the instanceable flag so meshes become real, editable prims
    cleared = []
    for path in instance_roots:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsInstance():
            prim.SetInstanceable(False)
            cleared.append(path)
    findings["instanceable_cleared"] = cleared
    bootstrap.advance(app, 5)

    real_meshes = [
        str(p.GetPath()) for p in Usd.PrimRange(root) if p.GetTypeName() == "Mesh"
    ]
    findings["meshes_after_clearing"] = real_meshes
    for path in real_meshes:
        add_labels(stage.GetPrimAtPath(path), labels=["mug"], taxonomy="class")
    bootstrap.advance(app, 5)
    sample("4_semantics_after_clearing_instanceable")


if __name__ == "__main__":
    sys.exit(main())
