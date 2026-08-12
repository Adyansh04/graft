"""Trainer interface, so the detector is swappable."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TrainResult:
    weights: Path
    epochs: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class EvalResult:
    """Scores for one evaluation target.

    `target` distinguishes sim-val from real photographs — they mean very
    different things and must not be conflated.
    """

    target: str
    metrics: dict[str, float]
    images: int


class Trainer(Protocol):
    name: str

    def train(self, dataset_yaml: Path, out_dir: Path, **kwargs) -> TrainResult: ...

    def evaluate(self, weights: Path, dataset_yaml: Path, split: str, **kwargs) -> EvalResult: ...


TRAINERS: dict[str, type] = {}


def register(cls):
    TRAINERS[cls.name] = cls
    return cls


def get_trainer(name: str) -> Trainer:
    if name not in TRAINERS:
        raise KeyError(f"unknown trainer {name!r}; available: {sorted(TRAINERS)}")
    return TRAINERS[name]()
