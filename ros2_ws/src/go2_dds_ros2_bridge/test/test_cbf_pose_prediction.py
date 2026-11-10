import math
from collections import deque

import numpy as np
import pytest

from go2_dds_ros2_bridge.cbf_pose_prediction import (
    bounded_prediction_seconds, claim_pair_once, newest_sample_at, propagate_base_pose,
)


def test_translation_and_rotation_use_base_axes():
    yaw = math.pi / 2
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    position, orientation = propagate_base_pose(
        [1, 2, 0], rotation, [1, 0, 0], [0, 0, math.pi], 0.1)
    np.testing.assert_allclose(position, [1 - math.sin(math.pi * 0.05) * 0.1,
                                          2 + math.cos(math.pi * 0.05) * 0.1, 0], atol=1e-9)
    np.testing.assert_allclose(orientation, rotation @ np.array([
        [math.cos(math.pi * 0.1), -math.sin(math.pi * 0.1), 0],
        [math.sin(math.pi * 0.1), math.cos(math.pi * 0.1), 0], [0, 0, 1]]), atol=1e-9)


def test_stale_samples_and_invalid_motion_are_rejected():
    history = deque([(1_000_000_000, np.array([1.0, 0.0, 0.0]))])
    np.testing.assert_array_equal(newest_sample_at(history, 1_040_000_000, 50_000_000), [1, 0, 0])
    with pytest.raises(ValueError, match="stale"):
        newest_sample_at(history, 1_060_000_000, 50_000_000)
    with pytest.raises(ValueError, match="invalid"):
        propagate_base_pose([0, 0, 0], np.eye(3), [float("nan"), 0, 0], [0, 0, 0], 0.1)


def test_pair_is_claimed_once():
    claimed = set()
    assert claim_pair_once(claimed, (1, 2))
    assert not claim_pair_once(claimed, (1, 2))
    assert claim_pair_once(claimed, (3, 4))


def test_prediction_horizon_is_bounded_to_200_ms():
    assert bounded_prediction_seconds(1_000_000_000, 1_200_000_000, 0.20) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="horizon"):
        bounded_prediction_seconds(1_000_000_000, 1_200_000_001, 0.20)
    with pytest.raises(ValueError, match="not_future"):
        bounded_prediction_seconds(1_000_000_000, 1_000_000_000, 0.20)
