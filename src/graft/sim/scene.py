"""Scene construction: ground, lighting, distractors.

Applies a `SceneParams` sampled by `sim.randomize`. Nothing here decides
anything — sampling is pure and lives there, so it stays testable without
Isaac Sim.
"""

from graft.sim.randomize import SceneParams

GROUND_PATH = "/World/Ground"
DOME_PATH = "/World/DomeLight"
DISTRACTOR_ROOT = "/World/Distractors"


def build_static_scene(ground_size_m: float = 4.0) -> None:
    """Ground plane and a physics scene. Created once per run."""
    import omni.replicator.core as rep

    rep.functional.physics.create_physics_scene("/PhysicsScene", timeStepsPerSecond=60)
    ground = rep.functional.create.plane(
        name="Ground", parent="/World", scale=(ground_size_m, ground_size_m, 1.0)
    )
    rep.functional.physics.apply_collider(ground)


def apply_lighting(params: SceneParams) -> None:
    import omni.replicator.core as rep

    dome = rep.functional.create.light(
        light_type="Dome",
        intensity=params.lighting.dome_intensity,
        rotation=(0.0, 0.0, params.lighting.dome_rotation_deg),
        name="DomeLight",
        parent="/World",
    )
    for index, light in enumerate(params.lighting.area_lights):
        if not light.enabled:
            continue
        rep.functional.create.light(
            light_type="Rect",
            intensity=light.intensity,
            color=light.color,
            position=light.position,
            name=f"AreaLight_{index}",
            parent="/World",
        )
    return dome


def place_distractors(params: SceneParams, distractor_usds: list[str]) -> list[str]:
    """Distractors occlude and clutter but are never labelled.

    They get no semantics, so they cannot appear in annotations even though
    they appear in frame.
    """
    import omni.replicator.core as rep

    if not distractor_usds:
        return []

    placed = []
    for index, placement in enumerate(params.distractors):
        usd = distractor_usds[index % len(distractor_usds)]
        path = f"{DISTRACTOR_ROOT}/Distractor_{index}"
        prim = rep.functional.create.from_usd(usd, path=path)
        rep.functional.modify.pose(
            prim, position_value=placement.position, rotation_value=placement.rotation_deg
        )
        rep.functional.physics.apply_rigid_body(prim, with_collider=True)
        placed.append(path)
    return placed


def clear_dynamic_prims(paths: list[str]) -> None:
    """Remove per-clip prims so the next clip starts from a clean stage."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for path in paths:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
