"""Isaac Sim startup.

`SimulationApp` must be constructed before any `omni`/`isaacsim` import —
it bootstraps the Kit runtime those modules need. Import them at call time,
never at module level.

Runs headless by default: rendering goes through the RTX pipeline either
way, and skipping the UI is faster and works over SSH.
"""

from typing import Any

# CosmosWriter is implemented with OmniGraph script nodes, which are opt-in.
SCRIPTNODE_OPT_IN = "/app/omni.graph.scriptnode/opt_in"

# No leading slash — this is the spelling NVIDIA's own examples use.
DLSS_EXEC_MODE = "rtx/post/dlss/execMode"


def launch(*, headless: bool = True, extra_config: dict[str, Any] | None = None):
    """Start Isaac Sim and apply the settings SDG needs.

    Returns the SimulationApp; the caller must call `.close()`.
    """
    from isaacsim import SimulationApp

    config = {"headless": headless}
    if extra_config:
        config.update(extra_config)
    app = SimulationApp(config)

    import carb

    settings = carb.settings.get_settings()
    settings.set_bool(SCRIPTNODE_OPT_IN, True)
    return app


def apply_render_settings(dlss_exec_mode: int = 2) -> None:
    """Call after `launch`, once the Kit runtime exists."""
    import carb

    carb.settings.get_settings().set(DLSS_EXEC_MODE, dlss_exec_mode)


def prepare_replicator() -> None:
    """Put Replicator in explicit-step mode.

    Capture drives frames with `rep.orchestrator.step()`; capture-on-play
    would emit frames on its own schedule and break clip boundaries.
    """
    import omni.replicator.core as rep

    rep.orchestrator.set_capture_on_play(False)


def advance(app, frames: int = 1) -> None:
    for _ in range(frames):
        app.update()
