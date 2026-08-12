"""USD intake validation, using usd-core rather than the Isaac Sim runtime.

Errors are things that make the asset unusable and cannot be repaired at
render time: a missing or wrong prim path, no geometry, a scale that would
render the object as a speck or fill the frame.

Warnings are things capture applies itself — colliders, rigid bodies and
semantic labels are attached to the target at runtime, so their absence in
the source file is informational. The loud failure for semantics lives in
the capture stage, which asserts the label actually took on the first
captured frame.

The baked-lighting check exists for scanned assets. Library assets are clean
and will not trigger it; scans generally will.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

Level = Literal["error", "warning"]

# Below this ratio between bright and dark regions of an albedo texture, the
# variation is plausible surface colour. Above it, shading is likely painted
# in. A uniformly glazed surface measures near 1.2; YCB scans measure ~4.5.
BAKED_LIGHTING_LUMINANCE_RATIO = 3.0

# Unused UV space is padded with black. Counting it as surface would flag
# every texture with an unwrapped margin.
UV_PADDING_LUMINANCE = 8.0

ALBEDO_HINTS = ("diffuse", "albedo", "basecolor", "base_color")
SHADING_MAP_HINTS = ("normal", "roughness", "metallic", "orm", "specular", "bump")


@dataclass
class Finding:
    level: Level
    code: str
    message: str

    def render(self) -> str:
        return f"  [{self.level:<7}] {self.code}: {self.message}"


@dataclass
class ValidationReport:
    usd_path: Path
    target_prim_path: str
    findings: list[Finding] = field(default_factory=list)
    size_m: tuple[float, float, float] | None = None
    meters_per_unit: float | None = None
    up_axis: str | None = None

    def add(self, level: Level, code: str, message: str) -> None:
        self.findings.append(Finding(level, code, message))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"asset: {self.usd_path}", f"target: {self.target_prim_path}"]
        if self.meters_per_unit is not None:
            lines.append(f"metersPerUnit: {self.meters_per_unit}   upAxis: {self.up_axis}")
        if self.size_m is not None:
            x, y, z = self.size_m
            lines.append(f"size: {x:.3f} x {y:.3f} x {z:.3f} m")
        if self.findings:
            lines.append("")
            lines.extend(f.render() for f in self.findings)
        verdict = "PASS" if self.ok else "FAIL"
        lines.append(f"\n{verdict} — {len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        return "\n".join(lines)


def validate_asset(
    usd_path: str | Path,
    target_prim_path: str,
    expected_size_m: tuple[float, float] = (0.03, 0.40),
) -> ValidationReport:
    usd_path = Path(usd_path)
    report = ValidationReport(usd_path=usd_path, target_prim_path=target_prim_path)

    if not usd_path.is_file():
        report.add("error", "missing-file", f"no such file: {usd_path}")
        return report

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        report.add("error", "unopenable", "USD stage could not be opened")
        return report

    report.meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    report.up_axis = str(UsdGeom.GetStageUpAxis(stage))

    prim = stage.GetPrimAtPath(target_prim_path)
    if not prim or not prim.IsValid():
        available = _candidate_prims(stage)
        report.add(
            "error",
            "missing-prim",
            f"target_prim_path {target_prim_path!r} does not exist. "
            f"Candidates with geometry: {available or 'none found'}",
        )
        return report

    _check_geometry_and_scale(stage, prim, report, expected_size_m)
    _check_physics(prim, report)
    _check_semantics(prim, report)
    _check_baked_lighting(stage, prim, usd_path, report)
    return report


def _iter_self_and_descendants(prim: Usd.Prim):
    # Instance proxies must be traversed explicitly. SimReady assets put
    # their geometry inside instanced prims, where a plain PrimRange sees
    # nothing below the instance root.
    yield from Usd.PrimRange(prim, Usd.TraverseInstanceProxies())


def _candidate_prims(stage: Usd.Stage, limit: int = 8) -> list[str]:
    """Prim paths that hold geometry, to make a wrong path actionable."""
    out = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            out.append(str(prim.GetPath()))
            if len(out) >= limit:
                break
    return out


def _check_geometry_and_scale(
    stage: Usd.Stage,
    prim: Usd.Prim,
    report: ValidationReport,
    expected_size_m: tuple[float, float],
) -> None:
    meshes = [p for p in _iter_self_and_descendants(prim) if p.IsA(UsdGeom.Mesh)]
    if not meshes:
        report.add(
            "error",
            "no-geometry",
            f"{prim.GetPath()} and its descendants contain no UsdGeom.Mesh",
        )
        return

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if bound.IsEmpty():
        report.add("error", "empty-bounds", f"{prim.GetPath()} has an empty bounding box")
        return

    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    size = tuple(float(v) * mpu for v in bound.GetSize())
    report.size_m = size

    largest = max(size)
    lo, hi = expected_size_m
    if largest < lo or largest > hi:
        report.add(
            "error",
            "implausible-scale",
            f"largest dimension is {largest:.3f} m, outside the expected "
            f"{lo}-{hi} m. metersPerUnit is {mpu}; the asset may be authored in "
            "different units than the stage declares.",
        )

    # A pivot far outside the geometry makes physics drops and camera look-at
    # behave unintuitively.
    centre = bound.GetMidpoint()
    origin_offset = max(abs(float(v)) for v in centre) * mpu
    if origin_offset > largest * 2:
        report.add(
            "warning",
            "offset-origin",
            f"geometry centre sits {origin_offset:.3f} m from the prim origin, "
            f"more than twice its size — the pivot is probably not on the object",
        )


def _has_api_in_subtree(prim: Usd.Prim, api) -> bool:
    return any(p.HasAPI(api) for p in _iter_self_and_descendants(prim))


def _check_physics(prim: Usd.Prim, report: ValidationReport) -> None:
    if not _has_api_in_subtree(prim, UsdPhysics.CollisionAPI):
        report.add(
            "warning",
            "no-collider",
            "no UsdPhysics.CollisionAPI in the subtree; capture applies one at runtime",
        )
    if not _has_api_in_subtree(prim, UsdPhysics.RigidBodyAPI):
        report.add(
            "warning",
            "no-rigid-body",
            "no UsdPhysics.RigidBodyAPI in the subtree; capture applies one at runtime",
        )


def semantic_labels(prim: Usd.Prim) -> dict[str, list[str]]:
    """Labels keyed by taxonomy.

    The applied-schema name carries the taxonomy (`SemanticsLabelsAPI:class`),
    so this reads them without knowing the taxonomy in advance.
    """
    from pxr import UsdSemantics

    found: dict[str, list[str]] = {}
    for p in _iter_self_and_descendants(prim):
        for schema in p.GetAppliedSchemas():
            if not schema.startswith("SemanticsLabelsAPI:"):
                continue
            taxonomy = schema.split(":", 1)[-1]
            attr = UsdSemantics.LabelsAPI(p, taxonomy).GetLabelsAttr()
            values = list(attr.Get() or []) if attr else []
            if values:
                found.setdefault(taxonomy, []).extend(str(v) for v in values)
    return found


def _check_semantics(prim: Usd.Prim, report: ValidationReport) -> None:
    labels = semantic_labels(prim)
    if not labels:
        report.add(
            "warning",
            "no-semantics",
            "no UsdSemantics labels in the subtree; capture applies the configured "
            "class at runtime and asserts it took on the first captured frame",
        )
    else:
        report.add(
            "warning",
            "existing-semantics",
            f"asset ships its own labels {labels}; capture strips all semantics except "
            "the configured target so stray labels cannot leak into annotations",
        )


def _texture_inputs(stage: Usd.Stage, prim: Usd.Prim) -> list[tuple[str, Path]]:
    """(shader input name, resolved texture path) for every bound material."""
    out: list[tuple[str, Path]] = []
    for p in _iter_self_and_descendants(prim):
        material, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
        if not material:
            continue
        for shader_prim in Usd.PrimRange(material.GetPrim()):
            shader = UsdShade.Shader(shader_prim)
            if not shader:
                continue
            for shader_input in shader.GetInputs():
                value = shader_input.Get()
                if value is None or not hasattr(value, "resolvedPath"):
                    continue
                resolved = value.resolvedPath or value.path
                if resolved:
                    out.append((shader_input.GetBaseName().lower(), Path(resolved)))
    return out


def _check_baked_lighting(
    stage: Usd.Stage, prim: Usd.Prim, usd_path: Path, report: ValidationReport
) -> None:
    textures = _texture_inputs(stage, prim)
    if not textures:
        return

    names = [name for name, _ in textures]
    albedo = [(n, p) for n, p in textures if any(h in n for h in ALBEDO_HINTS)]
    has_shading_maps = any(any(h in n for h in SHADING_MAP_HINTS) for n in names)

    if albedo and not has_shading_maps:
        report.add(
            "warning",
            "albedo-only-material",
            "material has a colour texture but no normal/roughness/metallic maps — "
            "typical of a scan, where shading is painted into the albedo",
        )

    for name, path in albedo:
        ratio = _luminance_ratio(path if path.is_absolute() else usd_path.parent / path)
        if ratio is None:
            continue
        if ratio > BAKED_LIGHTING_LUMINANCE_RATIO:
            report.add(
                "warning",
                "baked-lighting",
                f"{name} texture {path.name} has a bright/dark luminance ratio of "
                f"{ratio:.1f} (threshold {BAKED_LIGHTING_LUMINANCE_RATIO}); lighting "
                "is likely baked into the albedo, which fights domain randomization",
            )


def _luminance_ratio(path: Path) -> float | None:
    """95th over 5th percentile luminance across the textured area.

    Pixels at or near black are excluded: unused UV space is padded with
    black, and counting it would flag any texture with an unwrapped margin.
    None if the image cannot be read or is almost entirely padding.
    """
    if not path.is_file():
        return None
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    textured = luminance[luminance > UV_PADDING_LUMINANCE]
    if textured.size < luminance.size * 0.01:
        return None
    dark = max(float(np.percentile(textured, 5)), 1.0)
    return float(np.percentile(textured, 95)) / dark
