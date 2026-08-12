"""Prompt generation for the restyle stage.

A prompt is assembled by picking one sentence per section and joining them,
plus one invariant clause that is never randomised. That clause states what
must not change and what it currently looks like — Cosmos guidance is
explicit that naming the thing to preserve, and describing it, is what keeps
it stable.

Sections are configuration, not code: adding scene variety means editing
YAML.
"""

from pathlib import Path

import numpy as np
import yaml

# Cosmos guidance puts the useful range around 120 words.
TARGET_WORDS = 120


def load_sections(path: str | Path) -> dict[str, list[str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"prompt sections not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    sections = {
        str(name): [str(v) for v in values]
        for name, values in raw.items()
        if isinstance(values, list) and values
    }
    if not sections:
        raise ValueError(f"{path} defines no non-empty prompt sections")
    return sections


def build_prompt(sections: dict[str, list[str]], invariant: str, seed: int) -> str:
    """One sentence per section, then the invariant clause last.

    The invariant goes last so it is not buried between randomised clauses.
    """
    rng = np.random.default_rng(seed)
    parts = [str(rng.choice(options)) for _, options in sorted(sections.items())]
    parts.append(invariant.strip())
    return " ".join(part.strip() for part in parts if part.strip())


def build_prompts(sections: dict[str, list[str]], invariant: str, seeds: list[int]) -> list[str]:
    return [build_prompt(sections, invariant, seed) for seed in seeds]


def word_count(prompt: str) -> int:
    return len(prompt.split())


def combinations(sections: dict[str, list[str]]) -> int:
    total = 1
    for options in sections.values():
        total *= len(options)
    return total
