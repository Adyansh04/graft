"""Loading, snapshotting, and change detection for configs.

Stages read the snapshot inside a run directory, never the live config file.
That way editing the config mid-run cannot half-change a run's behaviour.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from graft.config.schema import Config

# Sections whose hashes are tracked. A change to one invalidates only the
# stages that depend on it — see graft.run.manifest.SECTION_STAGES.
SECTIONS = (
    "run",
    "asset",
    "classes",
    "sim",
    "capture",
    "cosmos",
    "encode",
    "dataset",
    "qa",
    "train",
    "eval",
)


def load_config(path: str | Path) -> Config:
    """Parse and validate a YAML config. Raises on unknown keys."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a YAML mapping, got {type(raw).__name__}")
    return Config.model_validate(raw)


def _canonical(value: Any) -> str:
    """Stable JSON for hashing — key order must not affect the hash."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def section_hashes(config: Config) -> dict[str, str]:
    """Per-section content hashes, used to decide what a config edit invalidates."""
    dumped = config.model_dump(mode="json")
    return {
        name: hashlib.sha256(_canonical(dumped[name]).encode()).hexdigest()[:16]
        for name in SECTIONS
        if name in dumped
    }


def config_hash(config: Config) -> str:
    return hashlib.sha256(_canonical(config.model_dump(mode="json")).encode()).hexdigest()


def snapshot_config(config: Config, dest: Path) -> Path:
    """Write the validated config to a run directory. This is what stages read."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    return dest


def changed_sections(old: dict[str, str], new: dict[str, str]) -> set[str]:
    """Sections that differ, including ones added or removed."""
    return {name for name in set(old) | set(new) if old.get(name) != new.get(name)}
