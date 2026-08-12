import json

import pytest
import yaml

from graft.cli import main
from graft.run.manifest import Manifest, Stage
from graft.run.paths import RunPaths


@pytest.fixture
def config_file(tmp_path, raw_config):
    raw_config["run"]["out_root"] = str(tmp_path / "runs")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw_config))
    return path


def test_validate_accepts_the_default_config(config_file, capsys):
    assert main(["validate", "--config", str(config_file)]) == 0
    assert "config OK" in capsys.readouterr().out


def test_validate_rejects_a_bad_config(tmp_path, raw_config, capsys):
    raw_config["sim"]["resolution"] = [640, 480]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw_config))

    assert main(["validate", "--config", str(path)]) == 1
    assert "Cosmos landscape bucket" in capsys.readouterr().err


def test_missing_config_is_a_clean_error(tmp_path, capsys):
    assert main(["validate", "--config", str(tmp_path / "nope.yaml")]) == 1
    assert "config not found" in capsys.readouterr().err


def test_run_init_creates_snapshot_and_manifest(config_file, raw_config):
    assert main(["run", "init", "--config", str(config_file)]) == 0

    paths = RunPaths.for_run(raw_config["run"]["out_root"], raw_config["run"]["name"])
    assert paths.config_snapshot.is_file()
    assert paths.clips.is_dir()

    manifest = Manifest.load(paths.manifest)
    assert manifest.run_name == raw_config["run"]["name"]
    assert manifest.section_hashes
    assert all(manifest.status_of(s).value == "pending" for s in Stage)


def test_snapshot_is_independent_of_the_live_config(config_file, raw_config):
    """Stages read the snapshot, so editing the config mid-run must not
    half-change a run's behaviour."""
    main(["run", "init", "--config", str(config_file)])
    paths = RunPaths.for_run(raw_config["run"]["out_root"], raw_config["run"]["name"])
    snapshot_before = yaml.safe_load(paths.config_snapshot.read_text())

    edited = dict(raw_config)
    edited["capture"] = {**raw_config["capture"], "n_clips": 999}
    config_file.write_text(yaml.safe_dump(edited))

    assert yaml.safe_load(paths.config_snapshot.read_text()) == snapshot_before


def test_run_init_refuses_to_clobber_without_force(config_file, capsys):
    assert main(["run", "init", "--config", str(config_file)]) == 0
    assert main(["run", "init", "--config", str(config_file)]) == 1
    assert "already initialised" in capsys.readouterr().err
    assert main(["run", "init", "--config", str(config_file), "--force"]) == 0


def test_status_before_init_is_a_clean_error(config_file, capsys):
    assert main(["status", "--config", str(config_file)]) == 1
    assert "No run at" in capsys.readouterr().err


def test_status_reports_stages_and_partial_clips(config_file, raw_config, capsys):
    main(["run", "init", "--config", str(config_file)])
    paths = RunPaths.for_run(raw_config["run"]["out_root"], raw_config["run"]["name"])

    from graft.run.manifest import write_clip_done

    frames = raw_config["capture"]["frames_per_clip"]
    paths.clip(0).mkdir(parents=True)
    write_clip_done(
        paths.clip_done(0),
        index=0,
        seed=0,
        modality_counts={"rgb": frames},
        label_count=frames,
        outputs={},
    )
    paths.clip(1).mkdir(parents=True)  # partial

    assert main(["status", "--config", str(config_file)]) == 0
    out = capsys.readouterr().out
    assert "1/8 complete" in out
    assert "[1]" in out  # partial clip listed for re-render


def test_doctor_runs_and_reports(capsys):
    main(["doctor"])
    out = capsys.readouterr().out
    assert "ffmpeg" in out
    assert "disk" in out
