"""Pure helpers for annotator output.

Kept out of the writer so they can be tested without Isaac Sim — these
handle a format change between Isaac Sim versions, which is exactly the
kind of thing that needs a test.
"""


TAXONOMY = "class"


def normalise_id_labels(raw: dict) -> dict[str, str]:
    """Map annotator semantic ids to class names.

    Isaac Sim 6.0 returns one entry per id holding every taxonomy the prim
    carries, e.g. `{"class": "mug", "wikidata_class": ...}`; earlier
    versions returned `"class:mug"`. Both are accepted.

    The 6.0 values are mapping-*like* rather than real `dict`s, so this
    duck-types on `.get` — an `isinstance(value, dict)` check silently
    discards every entry and produces unlabelled frames.
    """
    out: dict[str, str] = {}
    for key, value in (raw or {}).items():
        label = None
        getter = getattr(value, "get", None)
        if callable(getter):
            label = getter(TAXONOMY)
        elif isinstance(value, str):
            label = value.split(":", 1)[-1] if ":" in value else value
        if label:
            out[str(key)] = str(label)
    return out


ANNOTATOR_KEYS = ("bounding_box_2d_tight", "instance_segmentation", "rgb")

# Measured payload for data_structure="renderProduct" on 6.0.1:
#   {swhFrameNumber, reference_time, distribution_outputs, trigger_outputs,
#    named_outputs, renderProducts: {<name>: {camera, resolution, <annotator>...}}}
RENDER_PRODUCTS_KEY = "renderProducts"


def single_render_product(data: dict) -> dict:
    """Reach the annotator data inside Replicator's payload.

    Capture attaches one render product, so the single entry under
    `renderProducts` is the one wanted. The wrapper carries several sibling
    dicts alongside it, so this navigates by key rather than by looking for
    a lone nested dict.
    """
    if any(key in data for key in ANNOTATOR_KEYS):
        return data

    products = data.get(RENDER_PRODUCTS_KEY)
    if isinstance(products, dict) and products:
        entries = [v for v in products.values() if isinstance(v, dict)]
        if entries:
            return entries[0]

    # Older/flatter shapes: a single wrapping dict.
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
