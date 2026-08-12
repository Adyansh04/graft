"""Enforces the sim/pure split.

Only a small set of files may touch the Isaac Sim runtime. Everything else
must import and run in the core venv, which is what makes most of the
codebase testable on a machine with no GPU and no Isaac install. This test is
the mechanism; without it the rule erodes the first time something is
convenient.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "graft"

ISAAC_ROOTS = {"omni", "isaacsim", "pxr", "carb", "usdrt"}

# The only modules permitted to import the sim runtime.
ISAAC_ONLY = {
    "sim/bootstrap.py",
    "sim/capture.py",
    "sim/scene.py",
    "sim/settle.py",
    "sim/semantics.py",
    "sim/writers/label_writer.py",
}

# usd-core provides `pxr` in the core venv for offline USD inspection. It must
# never be installed into the Isaac venv (Boost.Python collision), so the
# module that uses it is not allowed to run there.
USD_CORE_ONLY = {"assets/validate.py"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_only_designated_modules_import_the_sim_runtime():
    offenders = {}
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel in ISAAC_ONLY:
            continue
        allowed = ISAAC_ROOTS - {"pxr"} if rel in USD_CORE_ONLY else ISAAC_ROOTS
        leaked = _imported_roots(path) & allowed
        if leaked:
            offenders[rel] = sorted(leaked)

    assert not offenders, (
        f"Isaac Sim imports outside the permitted modules: {offenders}. "
        "Either move the logic into an existing sim/ module, or keep it pure "
        "and have sim/capture.py apply it."
    )


def test_pure_sim_modules_stay_importable_without_isaac():
    """Trajectory and randomization logic must run in the core venv — that is
    what makes them testable without a GPU."""
    import graft.sim.camera  # noqa: F401
    import graft.sim.randomize  # noqa: F401
