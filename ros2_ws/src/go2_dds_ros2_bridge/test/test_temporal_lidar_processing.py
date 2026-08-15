import math

import numpy as np

from go2_dds_ros2_bridge.temporal_lidar_processing import (
    CompletedScan,
    CompletedScanHistory,
    FOV_BINS,
    MAX_DISTANCE_M,
    TwoCloudAssembler,
    WORLD_BINS,
    front_arc_bin_indices,
    normalized_scan_age,
    polar_bins_to_base_points,
    project_history_to_polar_bins,
    project_history_to_front_arc,
)


def test_history_keeps_newest_four_scans():
    history = CompletedScanHistory(depth=4)
    for stamp in range(5):
        history.push(CompletedScan(stamp_ns=stamp, points_xyz_m=np.empty((0, 3))))
    assert [scan.stamp_ns for scan in history.newest_first()] == [4, 3, 2, 1]


def test_two_cloud_assembler_completes_only_adjacent_raw_clouds():
    assembler = TwoCloudAssembler(raw_cloud_period_s=0.065, intercloud_tolerance_s=0.01)
    first = np.array(((1.0, 0.0, 0.3),))
    second = np.array(((2.0, 0.0, 0.3),))
    completed, rejected = assembler.push(first, 1_000_000_000)
    assert completed is None and not rejected
    completed, rejected = assembler.push(second, 1_065_000_000)
    assert not rejected
    assert completed is not None
    assert completed.stamp_ns == 1_065_000_000
    assert completed.points_xyz_m.shape == (2, 3)


def test_two_cloud_assembler_discards_a_gapped_partial_pair():
    assembler = TwoCloudAssembler(raw_cloud_period_s=0.065, intercloud_tolerance_s=0.01)
    assembler.push(np.empty((0, 3)), 1_000_000_000)
    completed, rejected = assembler.push(np.empty((0, 3)), 1_120_000_000)
    assert completed is None and rejected
    completed, rejected = assembler.push(np.empty((0, 3)), 1_185_000_000)
    assert not rejected and completed is not None


def test_scan_age_matches_training_normalization_and_clamping():
    assert normalized_scan_age(1_125_000_000, 1_000_000_000, 0.25) == 0.5
    assert normalized_scan_age(1_500_000_000, 1_000_000_000, 0.25) == 1.0


def test_unknown_bins_are_max_range_and_invalid():
    distances, validity = project_history_to_front_arc(
        (CompletedScan(stamp_ns=1, points_xyz_m=np.empty((0, 3))),),
        current_xy_m=np.array((0.0, 0.0)),
        current_yaw_rad=0.0,
    )
    assert distances.shape == (4, FOV_BINS)
    assert validity.shape == (4, FOV_BINS)
    assert np.all(distances == 1.0)
    assert np.all(validity == 0)


def test_covered_empty_bin_is_known_free_but_uncovered_bin_is_unknown():
    coverage = np.array(((MAX_DISTANCE_M, 0.0, 0.0),))
    distances, validity = project_history_to_polar_bins(
        (CompletedScan(stamp_ns=1, points_xyz_m=np.empty((0, 3)), coverage_endpoints_xyz_m=coverage),),
        current_xy_m=np.zeros(2),
    )
    front_bin = WORLD_BINS // 2
    assert distances[0, front_bin] == 1.0
    assert validity[0, front_bin] == 1
    assert distances[0, 0] == 1.0
    assert validity[0, 0] == 0


def test_surface_return_overrides_known_free_coverage_range():
    coverage = np.array(((MAX_DISTANCE_M, 0.0, 0.0),))
    points = np.array(((2.0, 0.0, 0.3),))
    distances, validity = project_history_to_polar_bins(
        (CompletedScan(stamp_ns=1, points_xyz_m=points, coverage_endpoints_xyz_m=coverage),),
        current_xy_m=np.zeros(2),
    )
    front_bin = WORLD_BINS // 2
    assert validity[0, front_bin] == 1
    assert np.isclose(distances[0, front_bin], 2.0 / MAX_DISTANCE_M)


def test_nearest_world_return_wins_and_front_arc_is_yaw_centered():
    points = np.array(((2.0, 0.0, 0.3), (5.0, 0.0, 0.3)))
    history = (CompletedScan(stamp_ns=1, points_xyz_m=points),)
    distances, validity = project_history_to_front_arc(history, current_xy_m=np.zeros(2), current_yaw_rad=0.0)
    center = FOV_BINS // 2
    assert validity[0, center] == 1
    assert np.isclose(distances[0, center], 2.0 / MAX_DISTANCE_M)


def test_world_returns_reenter_front_arc_after_robot_turns_back():
    history = (CompletedScan(stamp_ns=1, points_xyz_m=np.array(((4.0, 0.0, 0.3),))),)
    _, right_validity = project_history_to_front_arc(
        history, current_xy_m=np.zeros(2), current_yaw_rad=-math.pi / 2.0
    )
    _, forward_validity = project_history_to_front_arc(
        history, current_xy_m=np.zeros(2), current_yaw_rad=0.0
    )
    assert not np.any(right_validity[0])
    assert forward_validity[0, FOV_BINS // 2] == 1


def test_full_360_debug_bins_retain_return_outside_current_front_arc():
    history = (CompletedScan(stamp_ns=1, points_xyz_m=np.array(((-4.0, 0.0, 0.3),))),)
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
