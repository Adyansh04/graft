"""Pure helpers for annotator output.

Kept out of the writer so they can be tested without Isaac Sim — these
handle a format change between Isaac Sim versions, which is exactly the
kind of thing that needs a test.
"""


def normalise_id_labels(raw: dict) -> dict[str, str]:
    """Map annotator semantic ids to class names.

    Isaac Sim 6.0 returns `{"class": "mug"}` per id; earlier versions
    returned `"class:mug"`. NVIDIA's own 6.0 examples still guard for both,
    so both are accepted. Entries in neither shape are dropped rather than
    guessed at.
    """
    out: dict[str, str] = {}
    for key, value in (raw or {}).items():
        if isinstance(value, dict):
            label = value.get("class")
        elif isinstance(value, str):
            label = value.split(":", 1)[-1] if ":" in value else value
        else:
            label = None
        if label:
            out[str(key)] = str(label)
    return out


def single_render_product(data: dict) -> dict:
    """Unwrap Replicator's render-product keyed structure.

    Capture attaches one render product. If the payload arrives nested under
    a product name, take the only entry rather than guessing the name.
    """
    if "bounding_box_2d_tight" in data or "instance_segmentation" in data:
        return data
    nested = [v for v in data.values() if isinstance(v, dict)]
    return nested[0] if len(nested) == 1 else data


def box_records(rows, dtype_names, id_to_class: dict[str, str], class_names: list[str]) -> list[dict]:
    """Turn the annotator's structured array into plain dicts.

    Field names for `bounding_box_2d_tight` are not documented for 6.0, so
    the semantic-id column is located by name at runtime.
    """
    if rows is None or len(rows) == 0:
        return []
    semantic_field = next((n for n in (dtype_names or ()) if "semantic" in n.lower()), None)

    records = []
    for row in rows:
        semantic_id = int(row[semantic_field]) if semantic_field else None
        class_name = id_to_class.get(str(semantic_id)) if semantic_id is not None else None
        records.append(
            {
                "semantic_id": semantic_id,
                "class_name": class_name,
                "class_id": class_names.index(class_name) if class_name in class_names else None,
                "x_min": float(row["x_min"]),
                "y_min": float(row["y_min"]),
                "x_max": float(row["x_max"]),
                "y_max": float(row["y_max"]),
            }
        )
    return records
