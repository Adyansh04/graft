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
                trainer.evaluate(
                    weights,
                    dataset_yaml,
                    split="val",
                    target="sim-val",
                    out_dir=paths.eval,
                    images=_count_images(paths.dataset / "images" / "val"),
                )
            )

    if config.eval.real_photos_dir:
        report = ingest(
            config.eval.real_photos_dir, config.class_names(), paths.eval / "real_photos.yaml"
        )
        print(report.render())
        if report.ok:
            results.append(
                trainer.evaluate(
                    weights,
                    report.dataset_yaml,
                    split="val",
                    target="real-photos",
                    out_dir=paths.eval,
                    images=report.labelled,
                )
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


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def render(payload: dict) -> str:
    from graft import console

    nan = float("nan")
    lines = []
    for target, data in payload.get("targets", {}).items():
        metrics = data["metrics"]
        map50 = console.value(format(metrics.get("map50", nan), ".4f"))
        map50_95 = console.value(format(metrics.get("map50_95", nan), ".4f"))
        images = console.dim(f"({data['images']} images)")
        # The real-photo number is the baseline; sim-val is a diagnostic.
        name = console.heading(target) if target == "real-photos" else console.dim(target)
        lines.append(f"{name:<24} mAP50={map50}  mAP50-95={map50_95}  {images}")
    if "real-photos" not in payload.get("targets", {}):
        lines.append(
            console.warn(
                "no real-photo score — sim-val alone measures whether training "
                "converged, not whether the detector works"
            )
        )
    return "\n".join(lines)
