"""Replicator writer emitting format-neutral labels.

Deliberately not a YOLO writer. Output formats are pluggable and their
conversion is testable pure-Python work, so this writes raw per-frame
geometry and lets `dataset.formats` turn it into YOLO, COCO or KITTI.

Runs alongside CosmosWriter on the same render product. All attached
writers fire on every `rep.orchestrator.step()`, so this emits exactly one
label file per captured frame.
"""

import json
from pathlib import Path

import omni.replicator.core as rep

from graft.sim.labels import box_records, normalise_id_labels, single_render_product


class GraftLabelWriter(rep.Writer):
    """Per frame: `bboxes_NNNNNN.json` plus `instance_NNNNNN.png`.

    `id_map.json` records the annotator's id-to-class mapping once per clip.

    Subclasses Replicator's `Writer`: the registry rejects anything else, and
    the base supplies `attach`/`detach`/`initialize`.
    """

    version = "1.0.0"
    data_structure = "renderProduct"

    def __init__(self, output_dir: str | None = None, class_names: list[str] | None = None):
        # The registry constructs with no arguments and the real values
        # arrive via initialize(), which calls __init__ again.
        self._out = Path(output_dir) if output_dir else None
        if self._out:
            self._out.mkdir(parents=True, exist_ok=True)
        self._class_names = list(class_names or [])
        self._frame = 0
        self._id_map_written = False

        # No semanticTypes filter: it selects on the pre-6.0 `Semantics`
        # schema, while labels are now authored as UsdSemantics taxonomies,
        # so filtering on it matches nothing and every frame comes back
        # unlabelled.
        self.annotators = [
            rep.annotators.get("bounding_box_2d_tight"),
            rep.annotators.get("instance_segmentation"),
        ]

    @property
    def frames_written(self) -> int:
        return self._frame

    def write(self, data: dict) -> None:
        if self._out is None:
            raise RuntimeError(
                "GraftLabelWriter has no output_dir — call initialize(output_dir=..., "
                "class_names=...) before attaching it"
            )
        payload = single_render_product(data)
        boxes = payload.get("bounding_box_2d_tight") or {}
        instances = payload.get("instance_segmentation") or {}

        if not self._id_map_written:
            self._dump_payload_shape(data, payload)
            self._write_id_map(boxes, instances)
            self._id_map_written = True

        rows = boxes.get("data")
        records = box_records(
            rows,
            getattr(getattr(rows, "dtype", None), "names", None),
            normalise_id_labels((boxes.get("info") or {}).get("idToLabels") or {}),
            self._class_names,
        )
        (self._out / f"bboxes_{self._frame:06d}.json").write_text(
            json.dumps({"frame": self._frame, "boxes": records})
        )
        self._write_instance_mask(instances)
        self._frame += 1

    def _dump_payload_shape(self, raw: dict, payload: dict) -> None:
        """Record what Replicator actually handed over.

        The payload nesting is undocumented for 6.0 and reading it wrongly
        produces empty labels that look identical to a labelling failure.
        """

        def shape(value, depth: int = 0):
            if depth > 3:
                return type(value).__name__
            if isinstance(value, dict):
                return {str(k): shape(v, depth + 1) for k, v in value.items()}
            if hasattr(value, "shape"):
                return f"{type(value).__name__}{tuple(value.shape)}"
            if isinstance(value, (list, tuple)):
                return f"{type(value).__name__}[{len(value)}]"
            return type(value).__name__

        def raw_id_labels(name: str):
            entry = payload.get(name) or {}
            return {
                "idToLabels": {str(k): repr(v) for k, v in (entry.get("idToLabels") or {}).items()},
                "idToSemantics": {
                    str(k): repr(v) for k, v in (entry.get("idToSemantics") or {}).items()
                },
                "primPaths": [str(p) for p in (entry.get("primPaths") or [])][:6],
            }

        (self._out / "payload_shape.json").write_text(
            json.dumps(
                {
                    "raw": shape(raw),
                    "unwrapped_keys": sorted(str(k) for k in payload),
                    "attached_annotators": [
                        getattr(a, "name", type(a).__name__) for a in (self.annotators or [])
                    ],
                    "bounding_box_2d_tight": raw_id_labels("bounding_box_2d_tight"),
                    "instance_segmentation": raw_id_labels("instance_segmentation"),
                },
                indent=2,
            )
        )

    def _write_id_map(self, boxes: dict, instances: dict) -> None:
        mapping = {
            "bounding_box_2d_tight": normalise_id_labels(
                (boxes.get("info") or {}).get("idToLabels") or {}
            ),
            "instance_segmentation": normalise_id_labels(
                (instances.get("info") or {}).get("idToLabels") or {}
            ),
            "class_names": self._class_names,
        }
        (self._out / "id_map.json").write_text(json.dumps(mapping, indent=2))

    def _write_instance_mask(self, instances: dict) -> None:
        array = instances.get("data")
        if array is None:
            return
        import numpy as np
        from PIL import Image

        mask = np.asarray(array)
        if mask.ndim == 3:
            mask = mask[..., 0]
        Image.fromarray(mask.astype(np.uint16)).save(
            self._out / f"instance_{self._frame:06d}.png"
        )


def register() -> None:
    rep.writers.register_writer(GraftLabelWriter)
