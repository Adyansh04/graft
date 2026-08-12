import pytest
from pydantic import ValidationError

from graft.config.schema import Config


def test_default_config_is_valid(config):
    assert config.run.name
    assert config.classes


def test_unknown_key_is_rejected(raw_config):
    """extra='forbid' — a typo must fail at load, not silently default."""
    raw_config["capture"]["n_clipz"] = 4
    with pytest.raises(ValidationError, match="n_clipz"):
        Config.model_validate(raw_config)


def test_off_bucket_resolution_is_rejected(raw_config):
    """1280x720 is the tempting mistake: Cosmos would center-crop it and
    silently invalidate every label coordinate."""
    raw_config["sim"]["resolution"] = [1280, 720]
    with pytest.raises(ValidationError, match="Cosmos landscape bucket"):
        Config.model_validate(raw_config)


def test_frames_per_clip_is_pinned(raw_config):
    raw_config["capture"]["frames_per_clip"] = 60
    with pytest.raises(ValidationError):
        Config.model_validate(raw_config)


def test_frame_level_split_is_impossible(raw_config):
    """Splitting by frame leaks near-duplicate consecutive frames into val."""
    raw_config["dataset"]["split"]["by"] = "frame"
    with pytest.raises(ValidationError):
        Config.model_validate(raw_config)


def test_split_must_sum_to_one(raw_config):
    raw_config["dataset"]["split"] = {"train": 0.7, "val": 0.2, "by": "clip"}
    with pytest.raises(ValidationError, match="sum to 1.0"):
        Config.model_validate(raw_config)


def test_cosmos_weight_ceiling(raw_config):
    raw_config["cosmos"]["weights"] = {"vis": 0.9, "edge": 0.9, "depth": 0.9, "seg": 0.9}
    with pytest.raises(ValidationError, match="ceiling"):
        Config.model_validate(raw_config)


def test_cosmos_weights_at_ceiling_are_allowed(raw_config):
    raw_config["cosmos"]["weights"] = {"vis": 0.5, "edge": 0.5, "depth": 0.5, "seg": 0.5}
    assert Config.model_validate(raw_config).cosmos.weights.vis == 0.5


def test_duplicate_class_colors_rejected(raw_config):
    """Two classes sharing a colour would make segmentation decode ambiguous."""
    raw_config["classes"] = [
        {"name": "mug", "color": [0, 255, 0, 255]},
        {"name": "bottle", "color": [0, 255, 0, 255]},
    ]
    with pytest.raises(ValidationError, match="duplicate class colours"):
        Config.model_validate(raw_config)


def test_duplicate_class_names_rejected(raw_config):
    raw_config["classes"] = [
        {"name": "mug", "color": [0, 255, 0, 255]},
        {"name": "mug", "color": [255, 0, 0, 255]},
    ]
    with pytest.raises(ValidationError, match="duplicate class names"):
        Config.model_validate(raw_config)


def test_class_id_is_index_and_shared_by_all_consumers(raw_config):
    """The sim writer, dataset names, and QA all key off this one ordering."""
    raw_config["classes"] = [
        {"name": "mug", "color": [0, 255, 0, 255]},
        {"name": "bottle", "color": [255, 0, 0, 255]},
    ]
    config = Config.model_validate(raw_config)
    assert config.class_id("mug") == 0
    assert config.class_id("bottle") == 1
    assert config.class_names() == ["mug", "bottle"]
    assert config.segmentation_mapping() == {
        "mug": [0, 255, 0, 255],
        "bottle": [255, 0, 0, 255],
    }


def test_unknown_class_raises(config):
    with pytest.raises(KeyError):
        config.class_id("forklift")


def test_colour_channel_range_enforced(raw_config):
    raw_config["classes"] = [{"name": "mug", "color": [0, 300, 0, 255]}]
    with pytest.raises(ValidationError, match="0-255"):
        Config.model_validate(raw_config)


def test_expected_size_must_be_ordered(raw_config):
    raw_config["asset"]["expected_size_m"] = [0.4, 0.1]
    with pytest.raises(ValidationError, match="min < max"):
        Config.model_validate(raw_config)


def test_sigma_max_bounds(raw_config):
    raw_config["cosmos"]["sigma_max"] = 120.0
    with pytest.raises(ValidationError):
        Config.model_validate(raw_config)
