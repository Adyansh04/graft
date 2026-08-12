"""Score a trained model against both evaluation targets.

Sim-val and real photographs are reported side by side and never merged.
The real number is the baseline; sim-val is a diagnostic, and a large gap
between them is the sim-to-real signal.
"""

import json
from pathlib import Path

from graft.config.schema import Config
from graft.eval.real_photos import ingest
from graft.run.paths import RunPaths
from graft.train.base import EvalResult, get_trainer

# Import for registration side effects.
import graft.train.ultralytics_backend  # noqa: F401


def evaluate(config: Config, paths: RunPaths, weights: Path) -> dict:
    trainer = get_trainer("ultralytics")
    results: list[EvalResult] = []

    if config.eval.sim_val:
        dataset_yaml = paths.dataset / "dataset.yaml"
        if dataset_yaml.is_file():
            results.append(
                trainer.evaluate(weights, dataset_yaml, split="val", target="sim-val")
            )

    if config.eval.real_photos_dir:
        report = ingest(
            config.eval.real_photos_dir, config.class_names(), paths.eval / "real_photos.yaml"
        )
        print(report.render())
        if report.ok:
            results.append(
                trainer.evaluate(weights, report.dataset_yaml, split="val", target="real-photos")
            )
        else:
            print("real-photo evaluation skipped — fix the problems above")

    payload = {
        "weights": str(weights),
        "targets": {r.target: {"metrics": r.metrics, "images": r.images} for r in results},
    }
    paths.eval.mkdir(parents=True, exist_ok=True)
    (paths.eval / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def render(payload: dict) -> str:
    lines = []
    for target, data in payload.get("targets", {}).items():
        metrics = data["metrics"]
        lines.append(
            f"{target:<14} mAP50={metrics.get('map50', float('nan')):.4f}  "
            f"mAP50-95={metrics.get('map50_95', float('nan')):.4f}  "
            f"({data['images']} images)"
        )
    if "real-photos" not in payload.get("targets", {}):
        lines.append(
            "no real-photo score — sim-val alone measures whether training "
            "converged, not whether the detector works"
        )
    return "\n".join(lines)
