"""Fixtures are built at runtime with usd-core into tmp dirs — no binary
assets in the repo, and each test states the exact defect it exercises.
"""

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdSemantics, UsdShade

from graft.assets.local import LocalUSDSource
from graft.assets.validate import BAKED_LIGHTING_LUMINANCE_RATIO, validate_asset

TARGET = "/World/Mug"


def make_stage(
    tmp_path,
    *,
    meters_per_unit: float = 1.0,
    size: float = 0.1,
    with_mesh: bool = True,
    with_collider: bool = False,
    with_rigid_body: bool = False,
    labels: list[str] | None = None,
    origin_offset: float = 0.0,
    name: str = "asset.usda",
):
    path = tmp_path / name
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    xform = UsdGeom.Xform.Define(stage, TARGET)

    if with_mesh:
        mesh = UsdGeom.Mesh.Define(stage, f"{TARGET}/Geom")
        half = size / 2.0 / meters_per_unit
        shift = origin_offset / meters_per_unit
        corners = [
            (-half + shift, -half, -half), (half + shift, -half, -half),
            (half + shift, half, -half), (-half + shift, half, -half),
            (-half + shift, -half, half), (half + shift, -half, half),
            (half + shift, half, half), (-half + shift, half, half),
        ]
        mesh.CreatePointsAttr([tuple(map(float, c)) for c in corners])
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr(
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 0, 3, 7, 4, 1, 2, 6, 5]
        )
        mesh.CreateExtentAttr(
            [(-half + shift, -half, -half), (half + shift, half, half)]
        )
        if with_collider:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        if with_rigid_body:
            UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())

    if labels is not None:
        api = UsdSemantics.LabelsAPI.Apply(xform.GetPrim(), "class")
        api.CreateLabelsAttr(labels)

    stage.GetRootLayer().Save()
    return path


def write_texture(path, *, ratio: float, size: int = 64):
    """Greyscale-ish albedo with a controlled bright/dark luminance ratio."""
    import cv2

    image = np.zeros((size, size, 3), dtype=np.uint8)
    dark = 40
    image[:, :] = dark
    image[: size // 2, :] = min(255, int(dark * ratio))
    cv2.imwrite(str(path), image)
    return path


def bind_material(usd_path, texture_path, *, extra_inputs: tuple[str, ...] = ()):
    """Attach a material with a diffuse texture, plus optional shading maps."""
    stage = Usd.Stage.Open(str(usd_path))
    material = UsdShade.Material.Define(stage, "/World/Looks/Mat")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    tex = UsdShade.Shader.Define(stage, "/World/Looks/Mat/DiffuseTex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("diffuse_texture", Sdf_asset()).Set(str(texture_path))
    for name in extra_inputs:
        node = UsdShade.Shader.Define(stage, f"/World/Looks/Mat/{name}")
        node.CreateIdAttr("UsdUVTexture")
        node.CreateInput(name, Sdf_asset()).Set(str(texture_path))
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{TARGET}/Geom"))
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(f"{TARGET}/Geom")).Bind(material)
    stage.GetRootLayer().Save()
    return usd_path


def Sdf_asset():
    from pxr import Sdf

    return Sdf.ValueTypeNames.Asset


# --- errors: things capture cannot repair ---


def make_instanced_stage(tmp_path, size: float = 0.1, name: str = "instanced.usda"):
    """Geometry behind an instanceable reference, as SimReady assets ship it.

    A plain PrimRange sees nothing below the instance root, so this is the
    shape that made the validator report a real mug as having no geometry.
    """
    path = tmp_path / name
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    proto = UsdGeom.Xform.Define(stage, "/Prototypes/MugProto")
    mesh = UsdGeom.Mesh.Define(stage, "/Prototypes/MugProto/Mesh")
    half = size / 2.0
    mesh.CreatePointsAttr(
        [
            (-half, -half, -half), (half, -half, -half), (half, half, -half),
            (-half, half, -half), (-half, -half, half), (half, -half, half),
            (half, half, half), (-half, half, half),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 0, 3, 7, 4, 1, 2, 6, 5]
    )
    mesh.CreateExtentAttr([(-half, -half, -half), (half, half, half)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, TARGET)
    instance = UsdGeom.Xform.Define(stage, f"{TARGET}/Instanced")
    instance.GetPrim().GetReferences().AddInternalReference(proto.GetPath())
    instance.GetPrim().SetInstanceable(True)

    stage.GetRootLayer().Save()
    return path


def test_geometry_behind_instancing_is_found(tmp_path):
    """Regression: SimReady assets instance their geometry, and the real mug
    was reported as having no mesh at all."""
    path = make_instanced_stage(tmp_path)
    report = validate_asset(path, TARGET, expected_size_m=(0.05, 0.20))
    assert report.ok, [f.message for f in report.errors]
    assert report.size_m[0] == pytest.approx(0.1, abs=1e-6)


def test_collider_inside_an_instance_is_detected(tmp_path):
    """The physics check has to see through instancing too, or every
    instanced asset warns about colliders it actually has."""
    path = make_instanced_stage(tmp_path)
    report = validate_asset(path, TARGET, expected_size_m=(0.05, 0.20))
    assert "no-collider" not in {f.code for f in report.warnings}


def test_clean_asset_passes(tmp_path):
    path = make_stage(tmp_path, with_collider=True, with_rigid_body=True, labels=["mug"])
    report = validate_asset(path, TARGET)
    assert report.ok
    assert report.size_m is not None
    assert report.size_m[0] == pytest.approx(0.1, abs=1e-6)


def test_missing_file_is_an_error(tmp_path):
    report = validate_asset(tmp_path / "nope.usda", TARGET)
    assert not report.ok
    assert report.errors[0].code == "missing-file"


def test_wrong_prim_path_errors_and_suggests_candidates(tmp_path):
    path = make_stage(tmp_path)
    report = validate_asset(path, "/World/Bottle")
    assert not report.ok
    finding = report.errors[0]
    assert finding.code == "missing-prim"
    # A wrong path is only actionable if it says what the right one might be.
    assert f"{TARGET}/Geom" in finding.message


def test_prim_without_geometry_is_an_error(tmp_path):
    path = make_stage(tmp_path, with_mesh=False)
    report = validate_asset(path, TARGET)
    assert not report.ok
    assert any(f.code == "no-geometry" for f in report.errors)


def test_metres_per_unit_is_honoured_not_assumed(tmp_path):
    """A 0.1 m mug is 10 units on a centimetre stage. Size must be judged in
    metres, so this passes despite the raw numbers looking large."""
    path = make_stage(tmp_path, meters_per_unit=0.01, size=0.1)
    report = validate_asset(path, TARGET, expected_size_m=(0.05, 0.20))
    assert report.ok, [f.message for f in report.errors]
    assert report.size_m[0] == pytest.approx(0.1, abs=1e-6)


def test_object_scaled_100x_too_large_is_caught(tmp_path):
    """The units mistake that actually bites: geometry authored for
    centimetres left on a metre stage, so the mug renders 10 m across."""
    path = make_stage(tmp_path, meters_per_unit=1.0, size=10.0)
    report = validate_asset(path, TARGET, expected_size_m=(0.05, 0.20))
    assert not report.ok
    assert any(f.code == "implausible-scale" for f in report.errors)


def test_too_small_asset_is_caught(tmp_path):
    path = make_stage(tmp_path, size=0.001)
    report = validate_asset(path, TARGET, expected_size_m=(0.05, 0.20))
    assert not report.ok
    assert any(f.code == "implausible-scale" for f in report.errors)


# --- warnings: things capture applies at runtime ---


def test_missing_physics_warns_but_does_not_fail(tmp_path):
    """Capture applies rigid body and colliders itself, so their absence in
    the source is informational."""
    path = make_stage(tmp_path, labels=["mug"])
    report = validate_asset(path, TARGET)
    assert report.ok
    codes = {f.code for f in report.warnings}
    assert "no-collider" in codes
    assert "no-rigid-body" in codes


def test_missing_semantics_warns_but_does_not_fail(tmp_path):
    path = make_stage(tmp_path, with_collider=True, with_rigid_body=True)
    report = validate_asset(path, TARGET)
    assert report.ok
    assert any(f.code == "no-semantics" for f in report.warnings)


def test_existing_semantics_are_reported(tmp_path):
    """Library assets ship their own labels; capture strips everything except
    the configured target, so this is worth surfacing."""
    path = make_stage(tmp_path, labels=["crockery", "vessel"])
    report = validate_asset(path, TARGET)
    warning = next(f for f in report.warnings if f.code == "existing-semantics")
    assert "crockery" in warning.message


def test_offset_pivot_warns(tmp_path):
    path = make_stage(tmp_path, size=0.1, origin_offset=1.0)
    report = validate_asset(path, TARGET, expected_size_m=(0.05, 0.20))
    assert any(f.code == "offset-origin" for f in report.warnings)


# --- the baked-lighting check, which exists for scanned assets ---


def test_albedo_only_material_warns(tmp_path):
    """The YCB signature: a colour texture and nothing else. Shading has
    nowhere to live except painted into the albedo."""
    texture = write_texture(tmp_path / "albedo.png", ratio=1.1)
    path = make_stage(tmp_path)
    bind_material(path, texture)
    report = validate_asset(path, TARGET)
    assert any(f.code == "albedo-only-material" for f in report.warnings)


def test_baked_lighting_detected_from_luminance_spread(tmp_path):
    texture = write_texture(tmp_path / "albedo.png", ratio=BAKED_LIGHTING_LUMINANCE_RATIO + 2)
    path = make_stage(tmp_path)
    bind_material(path, texture)
    report = validate_asset(path, TARGET)
    assert any(f.code == "baked-lighting" for f in report.warnings)


def test_unused_uv_padding_does_not_trigger_baked_lighting(tmp_path):
    """Unwrapped margins are padded with black. Counting them as surface
    would flag almost every real texture."""
    import cv2

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:32, :] = 130  # the textured half, flat colour
    texture = tmp_path / "albedo.png"
    cv2.imwrite(str(texture), image)

    path = make_stage(tmp_path)
    bind_material(path, texture, extra_inputs=("normal",))
    report = validate_asset(path, TARGET)
    assert "baked-lighting" not in {f.code for f in report.warnings}


def test_flat_albedo_does_not_trigger_baked_lighting(tmp_path):
    """A clean library texture must not false-positive."""
    texture = write_texture(tmp_path / "albedo.png", ratio=1.1)
    path = make_stage(tmp_path)
    bind_material(path, texture, extra_inputs=("normal", "roughness"))
    report = validate_asset(path, TARGET)
    codes = {f.code for f in report.warnings}
    assert "baked-lighting" not in codes
    assert "albedo-only-material" not in codes


# --- source ---


def test_local_source_resolves(tmp_path):
    path = make_stage(tmp_path)
    asset = LocalUSDSource(path, TARGET, "mug").resolve()
    assert asset.usd_path == path
    assert asset.class_name == "mug"


def test_local_source_missing_file_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_asset"):
        LocalUSDSource(tmp_path / "nope.usd", TARGET, "mug").resolve()
