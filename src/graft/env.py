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


EULA_ENV = {"OMNI_KIT_ACCEPT_EULA": "YES"}


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
        from graft import console

        tag = console.status(self.ok)
        detail = self.detail if self.ok else console.warn(self.detail)
        return f"  [{tag}] {console.bold(self.name)}: {detail}"


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def _run(args: list[str], timeout: int = 60) -> tuple[int, str]:
    import os

    env = {**os.environ, **EULA_ENV}
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
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

    probe = (
        "import isaacsim, importlib.metadata as m; "
        "print(m.version('isaacsim'))"
    )
    code, out = _run([str(python), "-c", probe], timeout=300)
    if code != 0:
        return [
            Check(
                "isaac venv",
                False,
                "venv exists but isaacsim does not import — the install did not "
                "complete. Re-run scripts/setup_isaac_sim_env.sh and read its "
                "output rather than its exit code.",
            )
        ]
    checks = [Check("isaac venv", True, f"isaacsim {out.splitlines()[-1]}")]

    # Isaac ships its own pxr into site-packages, so the path says nothing.
    # Ask whether the usd-core distribution is installed instead.
    probe = (
        "import importlib.metadata as m\n"
        "try:\n"
        "    print('usd-core', m.version('usd-core'))\n"
        "except m.PackageNotFoundError:\n"
        "    print('absent')\n"
    )
    code, out = _run([str(python), "-c", probe])
    contaminated = code == 0 and out.strip().startswith("usd-core")
    checks.append(
        Check(
            "isaac usd-core",
            not contaminated,
            f"{out.strip()} — collides with Isaac's bundled pxr at the Boost.Python "
            "level; uninstall it from .venv-isaac"
            if contaminated
            else "absent (correct)",
        )
    )
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
    env = {**os.environ, **EULA_ENV}
    src = str(root / "src")
    env["PYTHONPATH"] = f"{src}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src
    return subprocess.call([str(python), "-m", module, *args], env=env)
