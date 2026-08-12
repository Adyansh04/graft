"""Camera trajectory generation.

Pure numpy — no Isaac Sim. `sim.capture` applies these poses via
`rep.functional.modify.pose`.

Clips must be temporally coherent because Cosmos Transfer is a video model,
so a trajectory is a smooth sweep rather than independent samples. Variation
happens between clips, not within one.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]


@dataclass(frozen=True)
class OrbitSpec:
    """One clip's camera motion. Sampled per clip by `sim.randomize`."""

    start_azimuth_deg: float
    sweep_deg: float
    elevation_deg: tuple[float, float]
    radius_m: tuple[float, float]
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)


def orbit_trajectory(spec: OrbitSpec, n_frames: int) -> list[CameraPose]:
    """Smooth arc around the target, Z-up.

    Elevation and radius interpolate across the sweep so the camera does not
    trace the same circle every clip.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")

    # endpoint=False keeps the last frame from duplicating the first on a
    # full 360-degree sweep.
    full_circle = abs(spec.sweep_deg % 360.0) < 1e-9 and spec.sweep_deg != 0.0
    fractions = np.linspace(0.0, 1.0, n_frames, endpoint=not full_circle)

    azimuths = np.radians(spec.start_azimuth_deg + fractions * spec.sweep_deg)
    elevations = np.radians(np.interp(fractions, [0.0, 1.0], spec.elevation_deg))
    radii = np.interp(fractions, [0.0, 1.0], spec.radius_m)

    tx, ty, tz = spec.target
    horizontal = radii * np.cos(elevations)
    xs = tx + horizontal * np.cos(azimuths)
    ys = ty + horizontal * np.sin(azimuths)
    zs = tz + radii * np.sin(elevations)

    return [
        CameraPose(position=(float(x), float(y), float(z)), look_at=spec.target)
        for x, y, z in zip(xs, ys, zs)
    ]


def max_step_degrees(poses: list[CameraPose], target: tuple[float, float, float]) -> float:
    """Largest angular gap between consecutive frames, about the target.

    Used to keep a sweep slow enough that the clip stays temporally coherent.
    """
    if len(poses) < 2:
        return 0.0
    centre = np.asarray(target, dtype=float)
    vectors = np.asarray([p.position for p in poses], dtype=float) - centre
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.clip(norms, 1e-12, None)
    dots = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).max())
