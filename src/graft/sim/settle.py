"""Physics drop and settling.

The object is dropped and allowed to come to rest, which is what produces
varied stable poses. A camera orbit alone would teach the detector the
object is always upright.

Settling is detected by polling velocity rather than stepping a fixed
number of frames — a fixed count either wastes time or captures mid-bounce.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SettleResult:
    settled: bool
    steps: int
    final_speed: float


def drop_and_settle(
    prim_path: str,
    *,
    lin_vel_thresh: float = 0.1,
    ang_vel_thresh: float = 0.1,
    timeout_steps: int = 600,
) -> SettleResult:
    import numpy as np
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    body = UsdPhysics.RigidBodyAPI(prim)
    if not body:
        raise RuntimeError(f"{prim_path} has no RigidBodyAPI; cannot settle it")

    speed = float("inf")
    for step in range(1, timeout_steps + 1):
        SimulationManager.step()
        linear = np.asarray(body.GetVelocityAttr().Get() or (0.0, 0.0, 0.0), dtype=float)
        angular = np.asarray(body.GetAngularVelocityAttr().Get() or (0.0, 0.0, 0.0), dtype=float)
        speed = float(np.linalg.norm(linear))
        spin = float(np.linalg.norm(angular))
        if speed < lin_vel_thresh and spin < ang_vel_thresh:
            return SettleResult(settled=True, steps=step, final_speed=speed)

    return SettleResult(settled=False, steps=timeout_steps, final_speed=speed)
