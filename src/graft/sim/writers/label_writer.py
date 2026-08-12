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

from graft.sim.labels import box_records, normalise_id_labels, single_render_product


class GraftLabelWriter:
    """Per frame: `bboxes_NNNNNN.json` plus `instance_NNNNNN.png`.

    `id_map.json` records the annotator's id-to-class mapping once per clip.
    """

    def __init__(self, output_dir: str, class_names: list[str]):
        import omni.replicator.core as rep

        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._class_names = list(class_names)
        self._frame = 0
        self._id_map_written = False

        self.annotators = [
            rep.annotators.get(
                "bounding_box_2d_tight", init_params={"semanticTypes": ["class"]}
            ),
            rep.annotators.get(
                "instance_segmentation", init_params={"semanticTypes": ["class"]}
            ),
        ]
        self.data_structure = "renderProduct"

    @property
    def frames_written(self) -> int:
        return self._frame

    def write(self, data: dict) -> None:
        payload = single_render_product(data)
        boxes = payload.get("bounding_box_2d_tight") or {}
        instances = payload.get("instance_segmentation") or {}

        if not self._id_map_written:
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
    import omni.replicator.core as rep

    rep.writers.register_writer(GraftLabelWriter)
