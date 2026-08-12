"""Does render_product need a camera path rather than the prim object?

The probe passed the value returned by rep.functional.create.camera straight
into rep.create.render_product and rendered nothing. The working diagnostic
passed a path string. Same scene, two render products, one variable.
"""

import json
import sys
from pathlib import Path


def main() -> int:
    from graft.sim import bootstrap

    out = Path("runs/mug-dev/probe/renderproduct_diag.json").resolve()
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
    from isaacsim.core.experimental.utils.semantics import add_labels
    from pxr import Usd, UsdGeom

    from graft.sim import bootstrap

    bootstrap.apply_render_settings()
    bootstrap.prepare_replicator()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World")

    rep.functional.create.dome_light(intensity=1000.0, name="Dome", parent="/World")
    cube = rep.functional.create.cube(
        position=(0.0, 0.0, 0.0), scale=(0.1, 0.1, 0.1), name="RefCube", parent="/World"
    )
    cube_prim = cube[0] if isinstance(cube, list) else cube
    add_labels(cube_prim, labels=["cube"], taxonomy="class")

    # Default clipping_range is (1.0, 1000000.0) — a camera 0.75 m from a
    # 0.1 m object sits inside the near plane and sees nothing.
    camera = rep.functional.create.camera(
        position=(0.5, 0.5, 0.4),
        look_at=(0.0, 0.0, 0.0),
        clipping_range=(0.01, 1000.0),
        name="Cam",
        parent="/World",
    )
    far_camera = rep.functional.create.camera(
        position=(0.5, 0.5, 0.4), look_at=(0.0, 0.0, 0.0), name="DefaultClipCam", parent="/World"
    )
    far_prim = far_camera[0] if isinstance(far_camera, list) else far_camera
    findings["default_clip_camera"] = str(far_prim.GetPath())
    findings["camera_return_type"] = type(camera).__name__
    findings["camera_is_list"] = isinstance(camera, list)
    camera_prim = camera[0] if isinstance(camera, list) else camera
    camera_path = str(camera_prim.GetPath())
    findings["camera_path"] = camera_path
    save()

    bootstrap.advance(app, 10)

    # Where is the cube actually, and where is the camera looking?
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    cube_range = cache.ComputeWorldBound(cube_prim).ComputeAlignedRange()
    findings["cube_bounds"] = {
        "min": [round(float(v), 4) for v in cube_range.GetMin()],
        "max": [round(float(v), 4) for v in cube_range.GetMax()],
        "empty": bool(cube_range.IsEmpty()),
    }
    findings["cube_type"] = str(cube_prim.GetTypeName())
    findings["cube_visibility"] = str(UsdGeom.Imageable(cube_prim).ComputeVisibility())
    findings["cube_children"] = [
        f"{p.GetPath()} ({p.GetTypeName()})" for p in Usd.PrimRange(cube_prim)
    ][:6]
    cam_xf = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    findings["camera_translation"] = [round(float(v), 4) for v in cam_xf.ExtractTranslation()]
    findings["all_world_children"] = [
        f"{p.GetPath()} ({p.GetTypeName()})"
        for p in stage.GetPrimAtPath("/World").GetChildren()
    ]
    save()

    # How many steps before the buffers actually contain the scene?
    rp = rep.create.render_product(camera_path, (1280, 704))
    rgb = rep.annotators.get("rgb")
    bbox = rep.annotators.get("bounding_box_2d_tight", init_params={"semanticTypes": ["class"]})
    rgb.attach(rp)
    bbox.attach(rp)

    steps = []
    for index in range(6):
        rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=False)
        a = np.asarray(rgb.get_data())
        steps.append(
            {
                "step": index,
                "rgb_unique": int(len(np.unique(a[..., :3]))),
                "rgb_mean": round(float(a[..., :3].mean()), 2),
                "n_boxes": int(len(bbox.get_data().get("data", []))),
            }
        )
        findings["steps"] = steps
        save()

    rgb.detach()
    bbox.detach()
    rp.destroy()

    # Same camera position, default clipping range — the control.
    rp2 = rep.create.render_product(str(far_prim.GetPath()), (1280, 704))
    rgb2 = rep.annotators.get("rgb")
    bbox2 = rep.annotators.get("bounding_box_2d_tight", init_params={"semanticTypes": ["class"]})
    rgb2.attach(rp2)
    bbox2.attach(rp2)
    rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=False)
    a = np.asarray(rgb2.get_data())
    findings["default_clipping_range"] = {
        "rgb_unique": int(len(np.unique(a[..., :3]))),
        "n_boxes": int(len(bbox2.get_data().get("data", []))),
    }
    save()


if __name__ == "__main__":
    sys.exit(main())
