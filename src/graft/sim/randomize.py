"""Per-clip scene randomization.

Pure numpy — no Isaac Sim. A seed produces a declarative `SceneParams`, and
`sim.capture` is the only thing that applies it. Keeping sampling separate
from application means every randomizer is testable without a GPU, and
adding one is a sampler plus an apply branch.

Randomization happens between clips, never within one: Cosmos Transfer is a
video model and needs temporal coherence inside a clip.
"""

from dataclasses import dataclass

import numpy as np

from graft.config.schema import CaptureCfg
from graft.sim.camera import OrbitSpec


@dataclass(frozen=True)
class AreaLight:
    position: tuple[float, float, float]
    intensity: float
    color: tuple[float, float, float]
    enabled: bool


@dataclass(frozen=True)
class Lighting:
    dome_intensity: float
    dome_rotation_deg: float
    area_lights: tuple[AreaLight, ...]


@dataclass(frozen=True)
class Placement:
    position: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]


@dataclass(frozen=True)
class SceneParams:
    seed: int
    orbit: OrbitSpec
    lighting: Lighting
    target_drop: Placement
    distractors: tuple[Placement, ...]
    negative: bool


def clip_seeds(master_seed: int, n_clips: int) -> list[int]:
    """Independent per-clip seeds.

    Resume depends on these: a re-rendered clip must reproduce exactly what
    the interrupted run would have produced.
    """
    children = np.random.SeedSequence(master_seed).spawn(n_clips)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def sample_scene(capture: CaptureCfg, seed: int, *, negative: bool = False) -> SceneParams:
    rng = np.random.default_rng(seed)
    camera = capture.camera

    elevation = tuple(
        float(v) for v in rng.uniform(camera.elevation_deg[0], camera.elevation_deg[1], size=2)
    )
    radius = tuple(float(v) for v in rng.uniform(camera.radius_m[0], camera.radius_m[1], size=2))
    orbit = OrbitSpec(
        start_azimuth_deg=float(rng.uniform(0.0, 360.0)),
        # Direction alternates so clips do not all sweep the same way.
        sweep_deg=float(camera.orbit_degrees * rng.choice([-1.0, 1.0])),
        elevation_deg=elevation,
        radius_m=radius,
    )

    lighting = _sample_lighting(capture, rng)
    target_drop = _sample_drop(rng)
    distractors = _sample_distractors(capture, rng)

    return SceneParams(
        seed=seed,
        orbit=orbit,
        lighting=lighting,
        target_drop=target_drop,
        # A negative clip renders the scene without the target, producing
        # frames with no annotation.
        distractors=distractors,
        negative=negative,
    )


def _sample_lighting(capture: CaptureCfg, rng: np.random.Generator) -> Lighting:
    cfg = capture.randomizers.lighting
    n_lights = int(rng.integers(cfg.n_area_lights[0], cfg.n_area_lights[1] + 1))
    lights = []
    for _ in range(n_lights):
        azimuth = rng.uniform(0.0, 2 * np.pi)
        distance = rng.uniform(0.6, 2.0)
        lights.append(
            AreaLight(
                position=(
                    float(distance * np.cos(azimuth)),
                    float(distance * np.sin(azimuth)),
                    float(rng.uniform(0.5, 2.0)),
                ),
                intensity=float(rng.uniform(*cfg.area_intensity)),
                color=tuple(float(c) for c in rng.uniform(0.7, 1.0, size=3)),
                enabled=bool(rng.random() >= cfg.off_probability),
            )
        )
    return Lighting(
        dome_intensity=float(rng.uniform(*cfg.dome_intensity)),
        dome_rotation_deg=float(rng.uniform(0.0, 360.0)),
        area_lights=tuple(lights),
    )


def _sample_drop(rng: np.random.Generator) -> Placement:
    """Start pose for the physics drop.

    Dropped from a height with a random orientation so the object settles
    into genuinely varied stable poses. A camera orbit alone would only ever
    show it upright.
    """
    return Placement(
        position=(
            float(rng.uniform(-0.05, 0.05)),
            float(rng.uniform(-0.05, 0.05)),
            float(rng.uniform(0.15, 0.35)),
        ),
        rotation_deg=tuple(float(a) for a in rng.uniform(0.0, 360.0, size=3)),
    )


def _sample_distractors(capture: CaptureCfg, rng: np.random.Generator) -> tuple[Placement, ...]:
    cfg = capture.randomizers.distractors
    count = int(rng.integers(cfg.count[0], cfg.count[1] + 1))
    out = []
    for _ in range(count):
        azimuth = rng.uniform(0.0, 2 * np.pi)
        distance = rng.uniform(0.12, 0.45)
        out.append(
            Placement(
                position=(
                    float(distance * np.cos(azimuth)),
                    float(distance * np.sin(azimuth)),
                    float(rng.uniform(0.05, 0.25)),
                ),
                rotation_deg=tuple(float(a) for a in rng.uniform(0.0, 360.0, size=3)),
            )
        )
    return tuple(out)
