import math

import numpy as np

from go2_dds_ros2_bridge.temporal_lidar_processing import (
    CompletedScan,
    CompletedScanHistory,
    deskew_points_to_reference_base,
    FOV_BINS,
    MAX_DISTANCE_M,
    WORLD_BINS,
    front_arc_bin_indices,
    is_adjacent_cloud_pair,
    normalized_scan_age,
    polar_bins_to_base_points,
    project_history_to_polar_bins,
    project_history_to_front_arc,
    reduce_front_capture_rays,
    upsample_polar_bins,
)


def test_history_keeps_newest_four_scans():
    history = CompletedScanHistory(depth=4)
    for stamp in range(5):
        history.push(
            CompletedScan(
                stamp_ns=stamp,
                endpoints_xyz_m=np.empty((0, 3)),
                ray_states=np.empty(0, dtype=np.uint8),
            )
        )
    assert [scan.stamp_ns for scan in history.newest_first()] == [4, 3, 2, 1]
    assert len(history) == 4


def test_adjacent_cloud_pair_accepts_expected_interval_only():
    assert is_adjacent_cloud_pair(
        1_000_000_000, 1_065_000_000, expected_period_s=0.065, tolerance_s=0.01
    )
    assert not is_adjacent_cloud_pair(
        1_000_000_000, 1_120_000_000, expected_period_s=0.065, tolerance_s=0.01
    )


def test_scan_age_matches_training_normalization_and_clamping():
    assert normalized_scan_age(1_065_000_000, 1_000_000_000, 0.13) == 0.5
    assert normalized_scan_age(1_500_000_000, 1_000_000_000, 0.13) == 1.0


def test_uncovered_bins_are_max_range_and_invalid():
    distances, validity = project_history_to_front_arc(
        (
            CompletedScan(
                stamp_ns=1,
                endpoints_xyz_m=np.empty((0, 3)),
                ray_states=np.empty(0, dtype=np.uint8),
            ),
        ),
        current_xy_m=np.array((0.0, 0.0)),
        current_yaw_rad=0.0,
    )
    assert distances.shape == (4, FOV_BINS)
    assert validity.shape == (4, FOV_BINS)
    assert np.all(distances == 1.0)
    assert np.all(validity == 0)


def test_front_capture_ray_reducer_builds_fixed_hit_free_fan_and_discards_rear():
    # Four front rays are centred at -67.5, -22.5, 22.5 and 67.5 degrees.
    points = np.array(
        (
            (2.0, 0.0, 0.3),
            (4.0, 0.0, 0.4),
            (-1.0, 0.0, 0.3),  # Direct rear: must not enter the front scan.
        )
    )
    endpoints, states = reduce_front_capture_rays(
        points, capture_rays=4, fov_degrees=180.0, range_percentile=0.0
    )
    assert endpoints.shape == (4, 3)
    assert np.array_equal(states, np.array((1, 1, 2, 1), dtype=np.uint8))
    assert np.isclose(np.linalg.norm(endpoints[2, :2]), 2.0)
    assert np.isclose(math.atan2(endpoints[2, 1], endpoints[2, 0]), math.radians(22.5))
    assert np.allclose(np.linalg.norm(endpoints[states == 1, :2], axis=1), MAX_DISTANCE_M)


def test_empty_cloud_produces_valid_free_rays_only_in_front_fan():
    endpoints, states = reduce_front_capture_rays(np.empty((0, 3)))
    history = (CompletedScan(stamp_ns=1, endpoints_xyz_m=endpoints, ray_states=states),)
    distances, validity = project_history_to_polar_bins(history, current_xy_m=np.zeros(2))
    front = front_arc_bin_indices(0.0)
    rear = np.setdiff1d(np.arange(WORLD_BINS), front)
    assert np.all(distances[0, front] == 1.0)
    assert np.all(validity[0, front] == 1)
    assert np.all(validity[0, rear] == 0)


def test_coarse_world_bins_repeat_into_the_policy_virtual_grid():
    distances = np.array(((0.15, 1.0, 0.40, 0.75),), dtype=np.float32)
    validity = np.array(((1, 1, 0, 1),), dtype=np.uint8)
    upsampled_distances, upsampled_validity = upsample_polar_bins(
        distances, validity, target_bins=8
    )
    assert np.allclose(upsampled_distances, ((0.15, 0.15, 1.0, 1.0, 0.40, 0.40, 0.75, 0.75),))
    assert np.array_equal(upsampled_validity, ((1, 1, 1, 1, 0, 0, 1, 1),))


def test_free_ray_stays_at_max_range_after_reprojection():
    scan = CompletedScan(
        stamp_ns=1,
        endpoints_xyz_m=np.array(((MAX_DISTANCE_M, 0.0, 0.0),)),
        ray_states=np.array((1,), dtype=np.uint8),
    )
    distances, validity = project_history_to_front_arc(
        (scan,), current_xy_m=np.array((1.0, 0.0)), current_yaw_rad=0.0
    )
    center = FOV_BINS // 2
    assert validity[0, center] == 1
    assert distances[0, center] == 1.0


def test_deskew_interpolates_rolling_points_into_reference_base():
    points = np.array(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    deskewed = deskew_points_to_reference_base(
        points,
        np.array((0.0, 1.0)),
        start_translation_w_m=np.array((0.0, 0.0, 0.0)),
        start_rotation_w_lidar=np.eye(3),
        end_translation_w_m=np.array((1.0, 0.0, 0.0)),
        end_rotation_w_lidar=np.eye(3),
        reference_translation_w_m=np.array((1.0, 0.0, 0.0)),
        reference_rotation_w_base=np.eye(3),
    )
    assert np.allclose(deskewed, np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))))


def test_nearest_world_return_wins_and_front_arc_is_yaw_centered():
    points = np.array(((2.0, 0.0, 0.3), (5.0, 0.0, 0.3)))
    history = (
        CompletedScan(stamp_ns=1, endpoints_xyz_m=points, ray_states=np.array((2, 2), dtype=np.uint8)),
    )
    distances, validity = project_history_to_front_arc(history, current_xy_m=np.zeros(2), current_yaw_rad=0.0)
    center = FOV_BINS // 2
    assert validity[0, center] == 1
    assert np.isclose(distances[0, center], 2.0 / MAX_DISTANCE_M)


def test_world_returns_reenter_front_arc_after_robot_turns_back():
    history = (
        CompletedScan(
            stamp_ns=1,
            endpoints_xyz_m=np.array(((4.0, 0.0, 0.3),)),
            ray_states=np.array((2,), dtype=np.uint8),
        ),
    )
    _, right_validity = project_history_to_front_arc(
        history, current_xy_m=np.zeros(2), current_yaw_rad=-math.pi / 2.0
    )
    _, forward_validity = project_history_to_front_arc(
        history, current_xy_m=np.zeros(2), current_yaw_rad=0.0
    )
    assert not np.any(right_validity[0])
    assert forward_validity[0, FOV_BINS // 2] == 1


def test_full_360_debug_bins_retain_return_outside_current_front_arc():
    history = (
        CompletedScan(
            stamp_ns=1,
            endpoints_xyz_m=np.array(((-4.0, 0.0, 0.3),)),
            ray_states=np.array((2,), dtype=np.uint8),
        ),
    )
    distances, validity = project_history_to_polar_bins(history, current_xy_m=np.zeros(2))
    rear_bin = 0
    assert distances.shape == (4, WORLD_BINS)
    assert validity[0, rear_bin] == 1
    front = front_arc_bin_indices(0.0)
    assert rear_bin not in front


def test_debug_points_reconstruct_valid_front_bin_in_current_base_frame():
    front = front_arc_bin_indices(0.0)
    distances = np.ones(FOV_BINS, dtype=np.float32)
    validity = np.zeros(FOV_BINS, dtype=np.uint8)
    center = FOV_BINS // 2
    distances[center] = 0.2
    validity[center] = 1
    points = polar_bins_to_base_points(
        distances, validity, current_yaw_rad=0.0, bin_indices=front,
    )
    assert points.shape == (1, 3)
    assert np.isclose(np.linalg.norm(points[0, :2]), 0.2 * MAX_DISTANCE_M)
