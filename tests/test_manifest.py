import json

from graft.config.loader import changed_sections, section_hashes
from graft.run.manifest import (
    Manifest,
    Stage,
    Status,
    clip_is_complete,
    dependents_of,
    write_clip_done,
)
from graft.run.paths import RunPaths

FRAMES = 121
MODALITIES = {"rgb": FRAMES, "depth": FRAMES, "segmentation": FRAMES, "shaded_seg": FRAMES, "edges": FRAMES}


def _paths(tmp_path) -> RunPaths:
    paths = RunPaths.for_run(tmp_path, "t")
    paths.create()
    return paths


def _done(paths: RunPaths, index: int, **overrides) -> None:
    kwargs = dict(
        index=index,
        seed=index,
        modality_counts=dict(MODALITIES),
        label_count=FRAMES,
        outputs={"mp4": [f"rgb.mp4"]},
    )
    kwargs.update(overrides)
    paths.clip(index).mkdir(parents=True, exist_ok=True)
    write_clip_done(paths.clip_done(index), **kwargs)


def test_roundtrip_preserves_state(tmp_path):
    manifest = Manifest(run_name="t", config_hash="abc")
    manifest.mark(Stage.CAPTURE, Status.DONE, detail="8 clips", now="2026-01-01T00:00:00+00:00")
    path = tmp_path / "manifest.json"
    manifest.save(path)

    loaded = Manifest.load(path)
    assert loaded.is_done(Stage.CAPTURE)
    assert loaded.stages[Stage.CAPTURE].detail == "8 clips"
    assert loaded.status_of(Stage.TRAIN) is Status.PENDING


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "manifest.json"
    Manifest(run_name="t").save(path)
    Manifest(run_name="t").save(path)
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]


def test_dependents_are_transitive():
    downstream = dependents_of(Stage.CAPTURE)
    # capture -> encode -> cosmos-export -> cosmos-import, and capture -> qa
    # -> assemble -> train -> eval
    assert {Stage.ENCODE, Stage.COSMOS_EXPORT, Stage.COSMOS_IMPORT} <= downstream
    assert {Stage.QA, Stage.ASSEMBLE, Stage.TRAIN, Stage.EVAL} <= downstream
    assert Stage.ASSET not in downstream
    assert Stage.CAPTURE not in downstream


def test_force_clears_stage_and_everything_downstream(tmp_path):
    manifest = Manifest(run_name="t")
    for stage in Stage:
        manifest.mark(stage, Status.DONE)

    cleared = manifest.force(Stage.QA)

    assert Stage.QA in cleared
    assert manifest.status_of(Stage.QA) is Status.PENDING
    assert manifest.status_of(Stage.TRAIN) is Status.PENDING
    # Upstream work is expensive and unaffected — that is the whole point.
    assert manifest.is_done(Stage.CAPTURE)
    assert manifest.is_done(Stage.ENCODE)


def test_changing_cosmos_config_does_not_invalidate_capture(config):
    """The headline resume property: re-run Cosmos with new control weights
    without re-rendering anything."""
    manifest = Manifest(run_name="t")
    for stage in Stage:
        manifest.mark(stage, Status.DONE)

    before = section_hashes(config)
    config.cosmos.weights.depth = 0.9
    changed = changed_sections(before, section_hashes(config))
    assert changed == {"cosmos"}

    affected = manifest.apply_section_changes(changed)

    assert manifest.is_done(Stage.CAPTURE)
    assert manifest.is_done(Stage.ENCODE)
    assert Stage.COSMOS_IMPORT in affected
    assert manifest.status_of(Stage.COSMOS_EXPORT) is Status.PENDING
    # Cosmos output feeds the dataset, so downstream must redo.
    assert manifest.status_of(Stage.QA) is Status.PENDING


def test_changing_capture_config_invalidates_the_render(config):
    manifest = Manifest(run_name="t")
    for stage in Stage:
        manifest.mark(stage, Status.DONE)

    before = section_hashes(config)
    config.capture.n_clips = 32
    manifest.apply_section_changes(changed_sections(before, section_hashes(config)))

    assert manifest.status_of(Stage.CAPTURE) is Status.PENDING
    assert manifest.status_of(Stage.TRAIN) is Status.PENDING
    assert manifest.is_done(Stage.ASSET)


def test_section_hash_is_order_independent(config):
    """Reordering keys in the YAML must not look like a change."""
    first = section_hashes(config)
    second = section_hashes(config.model_copy(deep=True))
    assert first == second


def test_unchanged_config_invalidates_nothing(config):
    assert changed_sections(section_hashes(config), section_hashes(config)) == set()


def test_complete_clip_is_recognised(tmp_path):
    paths = _paths(tmp_path)
    _done(paths, 0)
    assert clip_is_complete(paths.clip_done(0), FRAMES)


def test_missing_marker_is_incomplete(tmp_path):
    paths = _paths(tmp_path)
    paths.clip(0).mkdir(parents=True)
    assert not clip_is_complete(paths.clip_done(0), FRAMES)


def test_short_modality_is_incomplete(tmp_path):
    """One short track silently truncates a whole Cosmos batch, so a clip
    with 120 depth frames must not count as done."""
    paths = _paths(tmp_path)
    _done(paths, 0, modality_counts={**MODALITIES, "depth": 120})
    assert not clip_is_complete(paths.clip_done(0), FRAMES)


def test_label_count_mismatch_is_incomplete(tmp_path):
    paths = _paths(tmp_path)
    _done(paths, 0, label_count=119)
    assert not clip_is_complete(paths.clip_done(0), FRAMES)


def test_corrupt_marker_is_incomplete(tmp_path):
    paths = _paths(tmp_path)
    paths.clip(0).mkdir(parents=True)
    paths.clip_done(0).write_text("{ truncated")
    assert not clip_is_complete(paths.clip_done(0), FRAMES)


def test_unverified_marker_is_incomplete(tmp_path):
    paths = _paths(tmp_path)
    paths.clip(0).mkdir(parents=True)
    paths.clip_done(0).write_text(
        json.dumps({"verified": False, "label_count": FRAMES, "modality_counts": MODALITIES})
    )
    assert not clip_is_complete(paths.clip_done(0), FRAMES)


def test_resume_identifies_exactly_the_partial_clips(tmp_path):
    paths = _paths(tmp_path)
    _done(paths, 0)
    _done(paths, 1, label_count=3)  # interrupted mid-clip
    paths.clip(2).mkdir(parents=True)  # crashed before any marker

    complete = Manifest(run_name="t").completed_clips(paths, FRAMES)

    assert complete == {0}
    assert sorted(set(paths.existing_clips()) - complete) == [1, 2]


def test_clip_done_records_seed_and_writer_outputs(tmp_path):
    """CosmosWriter's on-disk layout is undocumented, so paths are recorded
    rather than assumed."""
    paths = _paths(tmp_path)
    _done(paths, 7, seed=12345, outputs={"mp4": ["rgb.mp4", "depth.mp4"]})
    data = json.loads(paths.clip_done(7).read_text())
    assert data["seed"] == 12345
    assert data["outputs"]["mp4"] == ["rgb.mp4", "depth.mp4"]
