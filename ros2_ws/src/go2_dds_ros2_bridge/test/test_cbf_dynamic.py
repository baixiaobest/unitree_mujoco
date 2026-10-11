"""Deployment projection parity and held-scan geometry checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from go2_dds_ros2_bridge.cbf_dynamic import history_arrays, prepare_points, project_history


def test_four_frame_projection_matches_training_fixture():
    fixture = Path(__file__).resolve().parents[6] / (
        "scripts/reinforcement_learning/rsl_rl/lidar_velocity_predictor/fixtures/projection_fixture_v3.json"
    )
    data = json.loads(fixture.read_text())
    hits = np.zeros((4, 256, 2), dtype=np.float64)
    states = np.zeros((4, 256), dtype=np.uint8)
    original_hits = np.asarray(data["world_hit_xy_newest_first"])
    original_states = np.asarray(data["ray_state_newest_first"])
    hits[:, :original_hits.shape[1]] = original_hits
    states[:, :original_states.shape[1]] = original_states
    tensor, winner = project_history(hits, states, data["evaluation_xy"], data["evaluation_yaw"])
    np.testing.assert_array_equal(tensor, data["expected_input_2x4x128"])
    np.testing.assert_array_equal(winner, data["expected_newest_winner_ray"])


def test_held_pose_and_sampling_phase_keep_winner_aligned():
    hits = np.zeros((4, 256, 2))
    states = np.zeros((4, 256), dtype=np.uint8)
    hits[:, 20] = [3.0, 0.0]
    hits[:, 21] = [2.0, 0.0]
    states[:, 20] = 2
    states[0, 21] = 2  # New phase adds a closer return.
    tensor, winner = project_history(hits, states, [0.0, 0.0], 0.0)
    assert winner[64] == 21
    assert tensor[0, 0, 64] == pytest.approx(0.1)
    shifted, shifted_winner = project_history(hits, states, [0.0, 0.5], 0.3)
    assert np.flatnonzero(shifted_winner == 21).size == 1
    assert shifted[1, 0].sum() >= 1
    assert np.all(states[0, 22:] == 0)


def test_evaluation_yaw_rotation_and_held_position_propagation():
    hits = np.zeros((256, 2))
    hits[21] = [2.0, 0.0]
    winner = np.full(128, -1, dtype=np.int32)
    winner[64] = 21
    body_velocity = np.zeros((128, 2))
    body_velocity[64] = [1.0, 0.0]
    points, velocity, mask = prepare_points(hits, winner, body_velocity, np.pi / 2, 0.3)
    np.testing.assert_allclose(points[64], [2.0, 0.3], atol=1e-12)
    np.testing.assert_allclose(velocity[64], [0.0, 1.0], atol=1e-12)
    assert mask.sum() == 1
    assert not velocity[63].any()


def test_history_order_and_invalid_prediction():
    scans = [SimpleNamespace(capture_points_xy_m=np.full(512, index), ray_states=[2] * 256)
             for index in range(4)]
    hits, states = history_arrays(scans)
    assert np.array_equal(hits[:, 0, 0], [0, 1, 2, 3])
    assert np.all(states == 2)
    bad = np.zeros((128, 2))
    bad[64] = np.nan
    with pytest.raises(ValueError, match="invalid_dynamic_prediction"):
        prepare_points(hits[0], np.full(128, -1), bad, 0.0, 0.1)
