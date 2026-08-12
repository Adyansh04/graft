"""Stage status tracking and the resume contract.

Two levels of granularity, deliberately:

* Stage level, for everything. A `done` stage re-invoked is a no-op.
* Per-clip level, for capture only, because re-rendering a clip is minutes
  while re-running assemble is seconds. Cheap stages just wipe and redo.

A clip counts as complete only if its `clip.done` marker parses, is marked
verified, and its recorded counts match. Anything else means the directory
is partial and gets deleted and re-rendered from its deterministic seed.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from graft.run.paths import RunPaths

MANIFEST_VERSION = 1


class Stage(StrEnum):
    ASSET = "asset"
    CAPTURE = "capture"
    ENCODE = "encode"
    COSMOS_EXPORT = "cosmos-export"
    COSMOS_IMPORT = "cosmos-import"
    QA = "qa"
    ASSEMBLE = "assemble"
    TRAIN = "train"
    EVAL = "eval"


class Status(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# Direct dependencies. Used to decide what --force cascades to. Stages still
# check their concrete inputs exist and fail loudly; this is not a scheduler.
STAGE_DEPS: dict[Stage, tuple[Stage, ...]] = {
    Stage.ASSET: (),
    Stage.CAPTURE: (Stage.ASSET,),
    Stage.ENCODE: (Stage.CAPTURE,),
    Stage.COSMOS_EXPORT: (Stage.ENCODE,),
    Stage.COSMOS_IMPORT: (Stage.COSMOS_EXPORT,),
    Stage.QA: (Stage.CAPTURE,),
    Stage.ASSEMBLE: (Stage.QA,),
    Stage.TRAIN: (Stage.ASSEMBLE,),
    Stage.EVAL: (Stage.TRAIN,),
}

# Which stages a config section invalidates when it changes. This is what
# makes "re-run Cosmos with different weights without re-rendering" work.
SECTION_STAGES: dict[str, tuple[Stage, ...]] = {
    "run": tuple(Stage),
    "asset": (Stage.ASSET, Stage.CAPTURE),
    "classes": (Stage.CAPTURE, Stage.QA, Stage.ASSEMBLE, Stage.TRAIN, Stage.EVAL),
    "sim": (Stage.CAPTURE,),
    "capture": (Stage.CAPTURE,),
    "cosmos": (Stage.COSMOS_EXPORT, Stage.COSMOS_IMPORT),
    "encode": (Stage.ENCODE, Stage.COSMOS_EXPORT),
    "dataset": (Stage.ASSEMBLE,),
    "qa": (Stage.QA,),
    "train": (Stage.TRAIN,),
    "eval": (Stage.EVAL,),
}


def dependents_of(stage: Stage) -> set[Stage]:
    """Every stage downstream of `stage`, transitively."""
    out: set[Stage] = set()
    frontier = [stage]
    while frontier:
        current = frontier.pop()
        for candidate, deps in STAGE_DEPS.items():
            if current in deps and candidate not in out:
                out.add(candidate)
                frontier.append(candidate)
    return out


@dataclass
class StageState:
    status: Status = Status.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StageState":
        return cls(
            status=Status(raw.get("status", Status.PENDING)),
            started_at=raw.get("started_at"),
            finished_at=raw.get("finished_at"),
            detail=raw.get("detail"),
        )


@dataclass
class Manifest:
    run_name: str
    config_hash: str = ""
    section_hashes: dict[str, str] = field(default_factory=dict)
    graft_git_sha: str | None = None
    created_at: str | None = None
    stages: dict[Stage, StageState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for stage in Stage:
            self.stages.setdefault(stage, StageState())

    # --- persistence ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "run_name": self.run_name,
            "created_at": self.created_at,
            "graft_git_sha": self.graft_git_sha,
            "config_hash": self.config_hash,
            "section_hashes": self.section_hashes,
            "stages": {str(k): v.to_dict() for k, v in self.stages.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Manifest":
        manifest = cls(
            run_name=raw["run_name"],
            config_hash=raw.get("config_hash", ""),
            section_hashes=raw.get("section_hashes", {}),
            graft_git_sha=raw.get("graft_git_sha"),
            created_at=raw.get("created_at"),
        )
        for name, state in (raw.get("stages") or {}).items():
            try:
                manifest.stages[Stage(name)] = StageState.from_dict(state)
            except ValueError:
                continue  # unknown stage from a newer version; ignore
        return manifest

    def save(self, path: Path) -> None:
        """Atomic write — a crash mid-write must not corrupt the manifest."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(self.to_dict(), handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        return cls.from_dict(json.loads(path.read_text()))

    # --- stage transitions ---

    def status_of(self, stage: Stage) -> Status:
        return self.stages[stage].status

    def is_done(self, stage: Stage) -> bool:
        return self.stages[stage].status is Status.DONE

    def mark(self, stage: Stage, status: Status, detail: str | None = None, *, now: str | None = None) -> None:
        state = self.stages[stage]
        state.status = status
        state.detail = detail
        if status is Status.RUNNING:
            state.started_at = now
            state.finished_at = None
        elif status in (Status.DONE, Status.FAILED):
            state.finished_at = now

    def reset(self, stage: Stage) -> None:
        self.stages[stage] = StageState()

    def force(self, stage: Stage) -> set[Stage]:
        """Clear a stage and everything downstream. Returns what was cleared."""
        cleared = {stage} | dependents_of(stage)
        for item in cleared:
            self.reset(item)
        return cleared

    def apply_section_changes(self, changed: set[str]) -> set[Stage]:
        """Invalidate only the stages affected by the changed config sections."""
        affected: set[Stage] = set()
        for section in changed:
            for stage in SECTION_STAGES.get(section, ()):
                affected |= {stage} | dependents_of(stage)
        for stage in affected:
            self.reset(stage)
        return affected

    # --- per-clip resume ---

    def completed_clips(self, paths: RunPaths, expected_frames: int) -> set[int]:
        """Clips whose done-marker verifies. Everything else is re-rendered."""
        done = set()
        for index in paths.existing_clips():
            if clip_is_complete(paths.clip_done(index), expected_frames):
                done.add(index)
        return done


def clip_is_complete(marker: Path, expected_frames: int) -> bool:
    """A clip is complete only if its marker parses, says verified, and its
    recorded counts match. A bare marker file is not enough — a crash between
    writing frames and finishing the clip would otherwise look successful.
    """
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if not data.get("verified"):
        return False
    if data.get("label_count") != expected_frames:
        return False
    counts = data.get("modality_counts") or {}
    if not counts:
        return False
    return all(count == expected_frames for count in counts.values())


def write_clip_done(
    marker: Path,
    *,
    index: int,
    seed: int,
    modality_counts: dict[str, int],
    label_count: int,
    outputs: dict[str, list[str]],
    negative: bool = False,
    controls_pruned: bool = False,
) -> None:
    """Record a verified clip.

    `outputs` holds the paths CosmosWriter actually wrote, relative to the
    clip directory — its on-disk layout is undocumented, so downstream stages
    read recorded paths instead of assuming a structure.
    """
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "index": index,
                "seed": seed,
                "verified": True,
                "negative": negative,
                "controls_pruned": controls_pruned,
                "modality_counts": modality_counts,
                "label_count": label_count,
                "outputs": outputs,
            },
            indent=2,
        )
    )
