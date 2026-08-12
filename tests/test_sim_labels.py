import numpy as np
import pytest

from graft.sim.labels import box_records, normalise_id_labels, single_render_product

CLASSES = ["mug", "bottle"]

BOX_DTYPE = np.dtype(
    [
        ("semanticId", "<u4"),
        ("x_min", "<i4"),
        ("y_min", "<i4"),
        ("x_max", "<i4"),
        ("y_max", "<i4"),
    ]
)


def rows(*entries):
    if not entries:
        return np.empty(0, dtype=BOX_DTYPE)
    return np.array(list(entries), dtype=BOX_DTYPE)


# --- idToLabels, which changed format between Isaac Sim versions ---


def test_six_zero_dict_format():
    assert normalise_id_labels({0: {"class": "mug"}}) == {"0": "mug"}


def test_legacy_string_format():
    """Older Isaac Sim returned "class:mug". Copying a 4.x tutorial verbatim
    against 6.0 silently produces wrong class indices."""
    assert normalise_id_labels({0: "class:mug"}) == {"0": "mug"}


def test_bare_string_without_taxonomy():
    assert normalise_id_labels({3: "mug"}) == {"3": "mug"}


def test_unrecognised_entries_are_dropped_not_guessed():
    assert normalise_id_labels({0: None, 1: 42, 2: {}, 3: {"other": "x"}}) == {}


def test_empty_and_none_are_safe():
    assert normalise_id_labels({}) == {}
    assert normalise_id_labels(None) == {}


def test_mixed_formats_in_one_payload():
    assert normalise_id_labels({0: {"class": "mug"}, 1: "class:bottle"}) == {
        "0": "mug",
        "1": "bottle",
    }


# --- render-product unwrapping ---


def test_flat_payload_passes_through():
    data = {"bounding_box_2d_tight": {"data": None}}
    assert single_render_product(data) is data


def test_nested_single_product_is_unwrapped():
    inner = {"bounding_box_2d_tight": {"data": None}}
    assert single_render_product({"RenderProduct_Replicator": inner}) is inner


def test_ambiguous_multi_product_is_left_alone():
    data = {"a": {"x": 1}, "b": {"y": 2}}
    assert single_render_product(data) is data


# --- box records ---


def test_boxes_map_to_class_ids():
    records = box_records(
        rows((1, 10, 20, 110, 220)), BOX_DTYPE.names, {"1": "mug"}, CLASSES
    )
    assert len(records) == 1
    assert records[0]["class_name"] == "mug"
    assert records[0]["class_id"] == 0
    assert (records[0]["x_min"], records[0]["y_max"]) == (10.0, 220.0)


def test_second_class_gets_index_one():
    records = box_records(rows((7, 0, 0, 5, 5)), BOX_DTYPE.names, {"7": "bottle"}, CLASSES)
    assert records[0]["class_id"] == 1


def test_unknown_class_yields_null_id_rather_than_a_wrong_one():
    """A stray label from a library asset must not silently become class 0."""
    records = box_records(rows((9, 0, 0, 5, 5)), BOX_DTYPE.names, {"9": "forklift"}, CLASSES)
    assert records[0]["class_name"] == "forklift"
    assert records[0]["class_id"] is None


def test_empty_frame_produces_no_records():
    assert box_records(rows(), BOX_DTYPE.names, {"1": "mug"}, CLASSES) == []
    assert box_records(None, BOX_DTYPE.names, {}, CLASSES) == []


def test_semantic_field_is_located_by_name_not_position():
    """6.0 does not document these field names, so a renamed or reordered
    column must not break the mapping."""
    dtype = np.dtype(
        [
            ("x_min", "<i4"),
            ("y_min", "<i4"),
            ("x_max", "<i4"),
            ("y_max", "<i4"),
            ("semanticIdentifier", "<u4"),
        ]
    )
    data = np.array([(1, 2, 3, 4, 5)], dtype=dtype)
    records = box_records(data, dtype.names, {"5": "mug"}, CLASSES)
    assert records[0]["class_id"] == 0


def test_missing_semantic_field_degrades_without_crashing():
    dtype = np.dtype([("x_min", "<i4"), ("y_min", "<i4"), ("x_max", "<i4"), ("y_max", "<i4")])
    data = np.array([(1, 2, 3, 4)], dtype=dtype)
    records = box_records(data, dtype.names, {"0": "mug"}, CLASSES)
    assert records[0]["semantic_id"] is None
    assert records[0]["class_id"] is None
