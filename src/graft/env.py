"""Environment checks and dispatch into the Isaac Sim venv.

The two venvs are both Python 3.12 and are separated for dependency
isolation, not version:

* `usd-core` (core) ships its own compiled `pxr` that collides with Isaac's
  bundled one at the Boost.Python level.
* `ultralytics`' torch collides with `isaacsim`'s.

`graft` is therefore never installed into the Isaac venv — sim stages run as
a subprocess with PYTHONPATH pointing at our source tree.
"""

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Isaac Sim plus its extension cache is roughly 25GB. Refuse to start an
# install that will run the disk dry partway through.
MIN_FREE_GB_FOR_ISAAC = 35


def repo_root() -> Path:
    """Walk up to the directory holding pyproject.toml."""
    for candidate in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def isaac_python(root: Path | None = None) -> Path:
    return (root or repo_root()) / ".venv-isaac" / "bin" / "python"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def render(self) -> str:
        return f"  [{'ok' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def _run(args: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)


def check_environment(root: Path | None = None) -> list[Check]:
    """Preflight the things this architecture makes it possible to get wrong."""
    root = root or repo_root()
    checks: list[Check] = []

    checks.append(Check("python", True, f"{sys.version.split()[0]} ({sys.executable})"))

    ffmpeg = shutil.which("ffmpeg")
    checks.append(Check("ffmpeg", bool(ffmpeg), ffmpeg or "not found — needed for the video round-trip"))

    ffprobe = shutil.which("ffprobe")
    checks.append(
        Check("ffprobe", bool(ffprobe), ffprobe or "not found — needed for frame-count assertions")
    )

    free = free_gb(root)
    checks.append(
        Check(
            "disk",
            free >= MIN_FREE_GB_FOR_ISAAC,
            f"{free:.0f}GB free (need >={MIN_FREE_GB_FOR_ISAAC}GB before an Isaac Sim install)",
        )
    )

    if shutil.which("nvidia-smi"):
        code, out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        checks.append(Check("gpu", code == 0, out.splitlines()[0] if code == 0 else out))
    else:
        checks.append(Check("gpu", False, "nvidia-smi not found"))

    checks.extend(check_isaac_venv(root))
    return checks


def check_isaac_venv(root: Path | None = None) -> list[Check]:
    """Verify the Isaac venv exists and has not been contaminated."""
    root = root or repo_root()
    python = isaac_python(root)
    if not python.is_file():
        return [
            Check(
                "isaac venv",
                False,
                f"{python} missing — run scripts/setup_isaac_sim_env.sh (not needed until M2)",
            )
        ]

    checks = [Check("isaac venv", True, str(python))]

    # usd-core in the Isaac venv is the Boost.Python collision. Assert rather
    # than trust the setup script.
    code, _ = _run([str(python), "-c", "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('pxr') else 1)"])
    if code == 0:
        code2, origin = _run([str(python), "-c", "import pxr; print(pxr.__file__)"])
        contaminated = "site-packages" in origin and "isaacsim" not in origin.lower()
        checks.append(
            Check(
                "isaac pxr",
                not contaminated,
                origin if not contaminated else f"{origin} — looks like usd-core, not Isaac's bundle",
            )
        )
    else:
        checks.append(Check("isaac pxr", False, "pxr not importable in the Isaac venv"))

    return checks


def run_in_isaac(module: str, args: list[str], root: Path | None = None) -> int:
    """Dispatch a sim stage into the Isaac venv.

    graft is not installed there, so the source tree goes on PYTHONPATH.
    """
    import os

    root = root or repo_root()
    python = isaac_python(root)
    if not python.is_file():
        raise RuntimeError(
            f"Isaac Sim venv not found at {python}. Run scripts/setup_isaac_sim_env.sh first."
        )
    env = dict(os.environ)
    src = str(root / "src")
    env["PYTHONPATH"] = f"{src}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src
    return subprocess.call([str(python), "-m", module, *args], env=env)
