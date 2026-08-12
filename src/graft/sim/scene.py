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


def apply_lighting(params: SceneParams) -> list[str]:
    """Returns the paths of the lights created, for per-clip teardown."""
    import omni.replicator.core as rep

    created = []
    dome = rep.functional.create.dome_light(
        intensity=params.lighting.dome_intensity,
        rotation=(0.0, 0.0, params.lighting.dome_rotation_deg),
        name="DomeLight",
        parent="/World",
    )
    created.append(_path_of(dome))

    for index, light in enumerate(params.lighting.area_lights):
        if not light.enabled:
            continue
        prim = rep.functional.create.rect_light(
            intensity=light.intensity,
            color=light.color,
            position=light.position,
            name=f"AreaLight_{index}",
            parent="/World",
        )
        created.append(_path_of(prim))
    return [p for p in created if p]


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
        prim = rep.functional.create.reference(
            usd_path=usd,
            name=f"Distractor_{index}",
            parent=DISTRACTOR_ROOT,
            position=placement.position,
            rotation=placement.rotation_deg,
        )
        rep.functional.physics.apply_rigid_body(prim, with_collider=True)
        path = _path_of(prim)
        if path:
            placed.append(path)
    return placed


def _path_of(created) -> str | None:
    """rep.functional.create.* returns a prim or a list of them."""
    if created is None:
        return None
    if isinstance(created, (list, tuple)):
        created = created[0] if created else None
    if created is None:
        return None
    getter = getattr(created, "GetPath", None)
    return str(getter()) if getter else str(created)


def clear_dynamic_prims(paths: list[str]) -> None:
    """Remove per-clip prims so the next clip starts from a clean stage."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for path in paths:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
