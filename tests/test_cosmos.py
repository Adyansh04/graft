import json

import numpy as np
import pytest
from PIL import Image

from graft.cosmos.encode import (
    EncodeSpec,
    FfmpegError,
    decode_to_pngs,
    encode_pngs,
    probe_frame_count,
    probe_resolution,
)
from graft.cosmos.prompts import build_prompt, build_prompts, combinations, load_sections, word_count
from graft.cosmos.spec import ClipSpec, build_batch_jsonl, build_controlnet_spec, build_line_map

SMALL = EncodeSpec(crf=30, preset="ultrafast", fps=30)
W, H = 64, 32


def write_frames(directory, count, size=(W, H)):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(count):
        frame = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
        Image.fromarray(frame).save(directory / f"f_{i:06d}.png")
    return directory


# --- encode / decode round trip ---


def test_round_trip_preserves_frame_count(tmp_path):
    src = write_frames(tmp_path / "png", 12)
    video = encode_pngs(src, tmp_path / "clip.mp4", SMALL, expect_frames=12)
    assert probe_frame_count(video) == 12

    frames = decode_to_pngs(video, tmp_path / "out", expect_frames=12)
    assert len(frames) == 12


def test_round_trip_preserves_resolution(tmp_path):
    src = write_frames(tmp_path / "png", 4)
    video = encode_pngs(src, tmp_path / "clip.mp4", SMALL, expect_frames=4)
    assert probe_resolution(video) == (W, H)


def test_wrong_frame_count_is_refused_before_encoding(tmp_path):
    """One short control track silently truncates a whole Cosmos batch."""
    src = write_frames(tmp_path / "png", 10)
    with pytest.raises(FfmpegError, match="expected 121"):
        encode_pngs(src, tmp_path / "clip.mp4", SMALL, expect_frames=121)


def test_empty_directory_is_an_error(tmp_path):
    (tmp_path / "png").mkdir()
    with pytest.raises(FfmpegError, match="no frames"):
        encode_pngs(tmp_path / "png", tmp_path / "clip.mp4", SMALL, expect_frames=None)


def test_decode_rejects_a_short_video(tmp_path):
    src = write_frames(tmp_path / "png", 5)
    video = encode_pngs(src, tmp_path / "clip.mp4", SMALL, expect_frames=5)
    with pytest.raises(FfmpegError, match="would no longer line up"):
        decode_to_pngs(video, tmp_path / "out", expect_frames=121)


def test_decode_clears_stale_frames(tmp_path):
    src = write_frames(tmp_path / "png", 4)
    video = encode_pngs(src, tmp_path / "clip.mp4", SMALL, expect_frames=4)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "frame_999999.png").write_bytes(b"stale")
    frames = decode_to_pngs(video, dest, expect_frames=4)
    assert len(frames) == 4
    assert not (dest / "frame_999999.png").exists()


# --- prompts ---


def sections():
    return {
        "surface": ["On oak.", "On slate."],
        "lighting": ["Warm light.", "Cool light.", "Dim light."],
    }


def test_prompt_includes_the_invariant_clause():
    """Naming what must not change, and describing it, is what keeps the
    object stable through the restyle."""
    invariant = "The white mug is unchanged."
    assert invariant in build_prompt(sections(), invariant, 0)


def test_prompt_is_deterministic_per_seed():
    assert build_prompt(sections(), "X.", 5) == build_prompt(sections(), "X.", 5)


def test_prompts_vary_between_clips():
    assert len(set(build_prompts(sections(), "X.", list(range(20))))) > 1


def test_invariant_comes_last():
    prompt = build_prompt(sections(), "The mug is unchanged.", 1)
    assert prompt.endswith("The mug is unchanged.")


def test_combination_count():
    assert combinations(sections()) == 6


def test_shipped_sections_give_plenty_of_variety():
    loaded = load_sections("configs/prompts.yaml")
    assert combinations(loaded) > 100


def test_shipped_prompts_are_a_reasonable_length():
    """Cosmos guidance puts the useful range around 120 words."""
    loaded = load_sections("configs/prompts.yaml")
    invariant = (
        "The white stoneware coffee mug at the centre of the frame is unchanged: "
        "same shape, same position, same size, same smooth glazed white surface."
    )
    for prompt in build_prompts(loaded, invariant, list(range(10))):
        assert 25 <= word_count(prompt) <= 200


def test_missing_sections_file_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_sections("configs/does-not-exist.yaml")


# --- controlnet specs ---


def clip(index=0):
    return ClipSpec(
        clip_index=index,
        input_video="videos/clip_0000_rgb.mp4",
        prompt="a mug",
        depth_video="videos/clip_0000_depth.mp4",
        seg_video="videos/clip_0000_shaded_seg.mp4",
    )


def test_spec_carries_hints_and_overrides(config):
    spec = build_controlnet_spec(clip(), config)
    assert set(spec) >= {"vis", "edge", "depth", "seg", "prompt", "input_video_path"}
    assert spec["depth"]["control_weight"] == config.cosmos.weights.depth
    # Per-clip values live in the JSON because it wins over untyped flags.
    assert spec["sigma_max"] == config.cosmos.sigma_max
    assert spec["guidance"] == config.cosmos.guidance


def test_seg_is_fed_shaded_segmentation(config):
    spec = build_controlnet_spec(clip(), config)
    assert "shaded_seg" in spec["seg"]["input_control"]


def test_vis_gets_the_raw_rgb(config):
    """Cosmos blurs it internally; handing it a pre-blurred track would blur
    twice."""
    spec = build_controlnet_spec(clip(), config)
    assert spec["vis"]["input_control"] == spec["input_video_path"]


def test_edge_is_left_to_auto_canny(config):
    assert "input_control" not in build_controlnet_spec(clip(), config)["edge"]


def test_zero_weight_drops_the_control(config):
    config.cosmos.weights.vis = 0.0
    assert "vis" not in build_controlnet_spec(clip(), config)


def test_weights_above_the_ceiling_are_refused(config):
    config.cosmos.weights.vis = 0.9
    config.cosmos.weights.edge = 0.9
    config.cosmos.weights.depth = 0.9
    config.cosmos.weights.seg = 0.9
    with pytest.raises(ValueError, match="ceiling"):
        build_controlnet_spec(clip(), config)


# --- batch mapping ---


def test_batch_jsonl_has_one_line_per_clip(config):
    clips = [clip(0), clip(3)]
    lines = build_batch_jsonl(clips, config).strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["visual_input"] == clips[0].input_video


def test_line_map_records_the_clip_for_each_line():
    """Output lands in video_N by line index, so a bundle carrying a subset
    of clips would otherwise pair frames with the wrong labels."""
    mapping = build_line_map([clip(5), clip(9)])
    assert mapping["clips"][0]["line"] == 0
    assert mapping["clips"][0]["clip"] == 5
    assert mapping["clips"][1]["clip"] == 9
    assert mapping["clips"][1]["expected_output"] == "video_1/output.mp4"
