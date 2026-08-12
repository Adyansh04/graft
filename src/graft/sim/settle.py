"""Physics drop and settling.

The object is dropped and allowed to come to rest, which is what produces
varied stable poses. A camera orbit alone would teach the detector the
object is always upright.

Settling is judged by how far the object actually moved between steps, not
by its velocity attributes: PhysX does not reliably write velocities back
into USD, so reading them reported "still moving at 0.000 m/s" forever.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SettleResult:
    settled: bool
    steps: int
    final_speed: float

    def describe(self) -> str:
        state = "settled" if self.settled else "still moving"
        return f"{state} after {self.steps} steps ({self.final_speed:.4f} m/step)"


def drop_and_settle(
    prim_path: str,
    *,
    lin_vel_thresh: float = 0.1,
    ang_vel_thresh: float = 0.1,
    timeout_steps: int = 600,
    step_dt: float = 1.0 / 60.0,
    quiet_steps: int = 5,
) -> SettleResult:
    """Step physics until the object stops moving.

    `lin_vel_thresh` is metres per second; it is compared against measured
    displacement per step. `quiet_steps` consecutive slow steps are required
    so a momentary pause at the top of a bounce does not count as settled.
    """
    import numpy as np
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"cannot settle missing prim {prim_path}")

    xformable = UsdGeom.Xformable(prim)

    def pose() -> np.ndarray:
        matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.asarray([float(v) for v in matrix.ExtractTranslation()], dtype=float)

    previous = pose()
    quiet = 0
    speed = float("inf")

    for step in range(1, timeout_steps + 1):
        SimulationManager.step()
        current = pose()
        speed = float(np.linalg.norm(current - previous)) / step_dt
        previous = current

        if speed < lin_vel_thresh:
            quiet += 1
            if quiet >= quiet_steps:
                return SettleResult(settled=True, steps=step, final_speed=speed)
        else:
            quiet = 0

    return SettleResult(settled=False, steps=timeout_steps, final_speed=speed)
