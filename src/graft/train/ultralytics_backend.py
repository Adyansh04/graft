"""Ultralytics YOLO training and evaluation."""

from pathlib import Path

from graft.train.base import EvalResult, TrainResult, register


@register
class UltralyticsTrainer:
    name = "ultralytics"

    def train(
        self,
        dataset_yaml: Path,
        out_dir: Path,
        *,
        model: str = "yolo11n.pt",
        epochs: int = 10,
        imgsz: int = 640,
        batch: int = 8,
        **kwargs,
    ) -> TrainResult:
        from ultralytics import YOLO

        detector = YOLO(model)
        results = detector.train(
            data=str(Path(dataset_yaml).resolve()),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            # Absolute: a relative project is resolved against Ultralytics'
            # own configured runs directory, not the run directory.
            project=str(Path(out_dir).resolve()),
            name="train",
            exist_ok=True,
            **kwargs,
        )
        weights = Path(results.save_dir) / "weights" / "best.pt"
        return TrainResult(
            weights=weights, epochs=epochs, metrics=_metrics(getattr(results, "results_dict", {}))
        )

    def evaluate(
        self,
        weights: Path,
        dataset_yaml: Path,
        split: str = "val",
        *,
        target: str = "sim-val",
        out_dir: Path | None = None,
        images: int | None = None,
        **kwargs,
    ) -> EvalResult:
        from ultralytics import YOLO

        detector = YOLO(str(weights))
        if out_dir is not None:
            kwargs.setdefault("project", str(Path(out_dir).resolve()))
            kwargs.setdefault("name", target)
            kwargs.setdefault("exist_ok", True)
        results = detector.val(data=str(Path(dataset_yaml).resolve()), split=split, **kwargs)
        return EvalResult(
            target=target,
            metrics=_extract_metrics(results),
            images=images if images is not None else _seen(results),
        )


def _seen(results) -> int:
    """How many images were scored, or 0 if the version does not say.

    Deliberately narrow: a wrong-but-plausible count is worse than none, so
    only fields that genuinely mean "images seen" are consulted.
    """
    for source in (
        getattr(results, "seen", None),
        getattr(getattr(results, "validator", None), "seen", None),
    ):
        try:
            value = int(source)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def _extract_metrics(results) -> dict[str, float]:
    out: dict[str, float] = {}
    box = getattr(results, "box", None)
    if box is not None:
        out["map50"] = float(getattr(box, "map50", float("nan")))
        out["map50_95"] = float(getattr(box, "map", float("nan")))
        out["precision"] = float(getattr(box, "mp", float("nan")))
        out["recall"] = float(getattr(box, "mr", float("nan")))
    seg = getattr(results, "seg", None)
    if seg is not None:
        out["seg_map50"] = float(getattr(seg, "map50", float("nan")))
        out["seg_map50_95"] = float(getattr(seg, "map", float("nan")))
    return out


def _metrics(raw: dict) -> dict[str, float]:
    return {str(k): float(v) for k, v in (raw or {}).items() if isinstance(v, (int, float))}
