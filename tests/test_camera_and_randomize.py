import numpy as np
import pytest

from graft.sim.camera import CameraPose, OrbitSpec, max_step_degrees, orbit_trajectory
from graft.sim.randomize import clip_seeds, sample_scene

FRAMES = 121


def spec(**overrides) -> OrbitSpec:
    base = dict(
        start_azimuth_deg=0.0,
        sweep_deg=120.0,
        elevation_deg=(30.0, 30.0),
        radius_m=(0.6, 0.6),
    )
    base.update(overrides)
    return OrbitSpec(**base)


# --- camera ---


def test_trajectory_has_one_pose_per_frame():
    poses = orbit_trajectory(spec(), FRAMES)
    assert len(poses) == FRAMES
    assert all(isinstance(p, CameraPose) for p in poses)


def test_every_pose_looks_at_the_target():
    target = (0.1, -0.2, 0.05)
    poses = orbit_trajectory(spec(target=target), 10)
    assert all(p.look_at == target for p in poses)


def test_radius_is_held_when_start_and_end_match():
    poses = orbit_trajectory(spec(radius_m=(0.6, 0.6)), 32)
    distances = [float(np.linalg.norm(p.position)) for p in poses]
    assert distances == pytest.approx([0.6] * 32, abs=1e-9)


def test_radius_interpolates_across_the_sweep():
    poses = orbit_trajectory(spec(radius_m=(0.4, 0.9)), 16)
    first = float(np.linalg.norm(poses[0].position))
    last = float(np.linalg.norm(poses[-1].position))
    assert first == pytest.approx(0.4, abs=1e-9)
    assert last == pytest.approx(0.9, abs=1e-9)


def test_elevation_sets_height_sign():
    above = orbit_trajectory(spec(elevation_deg=(45.0, 45.0)), 8)
    assert all(p.position[2] > 0 for p in above)
    below = orbit_trajectory(spec(elevation_deg=(-20.0, -20.0)), 8)
    assert all(p.position[2] < 0 for p in below)


def test_motion_is_smooth_enough_for_a_coherent_clip():
    """Cosmos is a video model, so within-clip motion must be continuous.
    A 120-degree sweep over 121 frames is about a degree per frame."""
    poses = orbit_trajectory(spec(sweep_deg=120.0), FRAMES)
    assert max_step_degrees(poses, (0.0, 0.0, 0.0)) < 2.0


def test_full_circle_does_not_duplicate_the_first_frame():
    poses = orbit_trajectory(spec(sweep_deg=360.0), 8)
    assert poses[0].position != pytest.approx(poses[-1].position)


def test_orbit_advances_by_a_constant_step():
    """Uniform angular motion is what keeps a clip temporally coherent."""
    poses = orbit_trajectory(spec(sweep_deg=360.0), 8)
    origin = np.zeros(3)
    steps = [
        max_step_degrees([a, b], tuple(origin)) for a, b in zip(poses[:-1], poses[1:])
    ]
    assert steps == pytest.approx([steps[0]] * len(steps), abs=1e-9)


def test_negative_sweep_orbits_the_other_way():
    forward = orbit_trajectory(spec(sweep_deg=90.0), 4)
    backward = orbit_trajectory(spec(sweep_deg=-90.0), 4)
    assert forward[-1].position[1] > 0
    assert backward[-1].position[1] < 0


def test_single_frame_is_allowed():
    assert len(orbit_trajectory(spec(), 1)) == 1


def test_zero_frames_is_rejected():
    with pytest.raises(ValueError, match="n_frames"):
        orbit_trajectory(spec(), 0)


# --- seeds ---


def test_clip_seeds_are_distinct_and_deterministic():
    first = clip_seeds(0, 8)
    assert len(set(first)) == 8
    assert first == clip_seeds(0, 8)


def test_clip_seeds_differ_by_master_seed():
    assert clip_seeds(0, 4) != clip_seeds(1, 4)


def test_clip_seed_is_stable_when_the_clip_count_grows():
    """Resume re-renders one clip; its seed must not depend on how many
    other clips exist, or a resumed run diverges from the original."""
    assert clip_seeds(0, 16)[:4] == clip_seeds(0, 4)


# --- scene sampling ---


def test_same_seed_reproduces_the_scene(config):
    """The resume contract: a re-rendered clip must match what the
    interrupted run would have produced."""
    assert sample_scene(config.capture, 1234) == sample_scene(config.capture, 1234)


def test_different_seeds_vary_the_scene(config):
    assert sample_scene(config.capture, 1) != sample_scene(config.capture, 2)


def test_sampled_values_respect_config_ranges(config):
    camera = config.capture.camera
    lighting = config.capture.randomizers.lighting
    distractors = config.capture.randomizers.distractors

    for seed in clip_seeds(0, 24):
        scene = sample_scene(config.capture, seed)

        assert all(camera.elevation_deg[0] <= e <= camera.elevation_deg[1] for e in scene.orbit.elevation_deg)
        assert all(camera.radius_m[0] <= r <= camera.radius_m[1] for r in scene.orbit.radius_m)
        assert abs(scene.orbit.sweep_deg) == camera.orbit_degrees
        assert 0.0 <= scene.orbit.start_azimuth_deg <= 360.0

        assert lighting.dome_intensity[0] <= scene.lighting.dome_intensity <= lighting.dome_intensity[1]
        assert (
            lighting.n_area_lights[0]
            <= len(scene.lighting.area_lights)
            <= lighting.n_area_lights[1]
        )
        for light in scene.lighting.area_lights:
            assert lighting.area_intensity[0] <= light.intensity <= lighting.area_intensity[1]

        assert distractors.count[0] <= len(scene.distractors) <= distractors.count[1]


def test_object_is_dropped_from_above_with_varied_orientation(config):
    """Camera orbits alone would only ever show the object upright, so the
    drop supplies the pose variety."""
    scenes = [sample_scene(config.capture, s) for s in clip_seeds(0, 12)]
    assert all(s.target_drop.position[2] > 0.1 for s in scenes)
    rotations = {s.target_drop.rotation_deg for s in scenes}
    assert len(rotations) == len(scenes)


def test_some_lights_get_switched_off(config):
    """Uniformly lit scenes are the failure mode; off_probability exists to
    produce genuinely uneven lighting."""
    states = [
        light.enabled
        for seed in clip_seeds(0, 40)
        for light in sample_scene(config.capture, seed).lighting.area_lights
    ]
    assert any(states) and not all(states)


def test_sweep_direction_varies_between_clips(config):
    directions = {
        np.sign(sample_scene(config.capture, s).orbit.sweep_deg) for s in clip_seeds(0, 20)
    }
    assert directions == {-1.0, 1.0}


def test_negative_clip_is_flagged(config):
    assert sample_scene(config.capture, 7, negative=True).negative
    assert not sample_scene(config.capture, 7).negative
