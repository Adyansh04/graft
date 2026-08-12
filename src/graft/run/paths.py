"""Run directory layout — the single source of truth for where things go.

Nothing here stores absolute paths on disk. A run directory is
self-describing and can be produced on one machine, archived, and moved to
another. The one place an absolute path is genuinely required (Ultralytics
resolves a relative `path:` in dataset.yaml against the process CWD, not the
yaml's own directory) is generated at assemble time from wherever the run
directory actually is.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @classmethod
    def for_run(cls, out_root: str | Path, name: str) -> "RunPaths":
        return cls(Path(out_root) / name)

    # --- top level ---
    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def config_snapshot(self) -> Path:
        return self.root / "config.snapshot.yaml"

    # --- stages ---
    @property
    def clips(self) -> Path:
        return self.root / "clips"

    @property
    def cosmos(self) -> Path:
        return self.root / "cosmos"

    @property
    def cosmos_bundle(self) -> Path:
        return self.cosmos / "bundle"

    @property
    def cosmos_output(self) -> Path:
        return self.cosmos / "output"

    @property
    def cosmos_frames(self) -> Path:
        return self.cosmos / "frames"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def qa(self) -> Path:
        return self.root / "qa"

    @property
    def weights(self) -> Path:
        return self.root / "weights"

    @property
    def eval(self) -> Path:
        return self.root / "eval"

    # --- per clip ---
    def clip(self, index: int) -> Path:
        return self.clips / f"clip_{index:04d}"

    def clip_done(self, index: int) -> Path:
        return self.clip(index) / "clip.done"

    def clip_labels(self, index: int) -> Path:
        return self.clip(index) / "labels"

    def clip_modality(self, index: int, modality: str) -> Path | None:
        """Locate a modality's frames inside a clip.

        CosmosWriter nests its output under a directory of its own naming, so
        the layout below `cosmos/` is discovered rather than assumed.
        """
        root = self.clip(index) / "cosmos"
        if not root.is_dir():
            return None
        for candidate in sorted(root.rglob(modality)):
            if candidate.is_dir():
                return candidate
        return None

    def existing_clips(self) -> list[int]:
        """Clip indices with a directory present, in order. Says nothing about
        whether they are complete — ask `manifest.completed_clips` for that."""
        if not self.clips.is_dir():
            return []
        out = []
        for child in sorted(self.clips.iterdir()):
            if child.is_dir() and child.name.startswith("clip_"):
                try:
                    out.append(int(child.name.removeprefix("clip_")))
                except ValueError:
                    continue
        return out

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.clips.mkdir(exist_ok=True)
