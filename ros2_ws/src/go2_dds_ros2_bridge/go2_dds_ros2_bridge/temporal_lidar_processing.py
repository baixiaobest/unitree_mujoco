"""Pure geometry and history helpers for the temporal raw-cloud lidar path."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from go2_dds_ros2_bridge.tf_utils import quaternion_from_rotation_matrix


HISTORY_FRAMES = 4
WORLD_BINS = 256
FOV_BINS = 128
MAX_DISTANCE_M = 20.0
CAPTURE_RAYS = 256
CAPTURE_FOV_DEG = 180.0


@dataclass(frozen=True)
class CompletedScan:
    """One fixed front-fan scan stored in corrected world coordinates.

    ``endpoints_xyz_m`` and ``ray_states`` have one entry per capture ray.
    State 2 marks a surface hit and state 1 marks an observed free-space ray;
    state 0 is reserved for unavailable/reset data.  Free endpoints exist only
    inside the captured front fan--a completed scan never fabricates rear rays.
    """

    stamp_ns: int
    endpoints_xyz_m: np.ndarray
    ray_states: np.ndarray


class CompletedScanHistory:
    """Newest-first bounded history of completed raw-cloud scans."""

    def __init__(self, depth: int = HISTORY_FRAMES) -> None:
        self._frames: deque[CompletedScan] = deque(maxlen=depth)

    def push(self, scan: CompletedScan) -> None:
        self._frames.appendleft(scan)

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        """Return the number of completed scans currently retained."""
        return len(self._frames)

    @property
    def newest_stamp_ns(self) -> int | None:
        return self._frames[0].stamp_ns if self._frames else None

    def newest_first(self) -> tuple[CompletedScan, ...]:
        return tuple(self._frames)


def is_adjacent_cloud_pair(
    first_stamp_ns: int,
    second_stamp_ns: int,
    *,
    expected_period_s: float,
    tolerance_s: float,
) -> bool:
    """Return whether two raw-cloud headers form one valid partial-scan pair."""
    if expected_period_s <= 0.0 or tolerance_s < 0.0:
        raise ValueError("expected_period_s must be positive and tolerance_s non-negative")
    interval_s = (second_stamp_ns - first_stamp_ns) / 1e9
    return interval_s > 0.0 and abs(interval_s - expected_period_s) <= tolerance_s


def reduce_front_capture_rays(
    points_xyz_m: np.ndarray,
    *,
    capture_rays: int = CAPTURE_RAYS,
    fov_degrees: float = CAPTURE_FOV_DEG,
    max_distance_m: float = MAX_DISTANCE_M,
    min_points_per_ray: int = 1,
    range_percentile: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a deskewed reference-base cloud to a fixed front lidar fan.

    A real point cloud has many returns per angular ray, unlike the simulator's
    one raycast result.  This function chooses a stable low-percentile range in
    each capture ray and puts that hit at the exact ray centre. Every other
    front capture ray becomes a valid max-range free-space measurement. No rear
    ray is generated.

    Returns:
        A fixed ``(capture_rays, 3)`` endpoint array and a fixed
        ``(capture_rays,)`` uint8 state array. State 1 is free and state 2 is a
        hit, matching the simulator's held-scan collector.
    """
    if capture_rays <= 0 or fov_degrees <= 0.0 or fov_degrees > 360.0:
        raise ValueError("capture_rays must be positive and fov_degrees must be in (0, 360]")
    if max_distance_m <= 0.0 or min_points_per_ray <= 0 or not 0.0 <= range_percentile <= 1.0:
        raise ValueError("Invalid capture-ray reduction parameters")
    half_fov = math.radians(fov_degrees) * 0.5
    ray_angles = -half_fov + (np.arange(capture_rays, dtype=np.float64) + 0.5) * (
        2.0 * half_fov / capture_rays
    )
    endpoints = np.column_stack(
        (
            max_distance_m * np.cos(ray_angles),
            max_distance_m * np.sin(ray_angles),
            np.zeros(capture_rays, dtype=np.float64),
        )
    )
    ray_states = np.ones(capture_rays, dtype=np.uint8)

    points = np.asarray(points_xyz_m, dtype=np.float64)
    if points.size == 0:
        return endpoints, ray_states
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_xyz_m must have shape (N, >=3)")

    xy = points[:, :2]
    ranges = np.linalg.norm(xy, axis=1)
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    valid = (
        np.isfinite(points[:, :3]).all(axis=1)
        & np.isfinite(ranges)
        & (ranges > 0.0)
        & (ranges <= max_distance_m)
        & (angles >= -half_fov)
        & (angles <= half_fov)
    )
    if not np.any(valid):
        return endpoints, ray_states

    selected_ranges = ranges[valid]
    selected_z = points[valid, 2]
    positions = (angles[valid] + half_fov) / (2.0 * half_fov)
    ray_indices = np.minimum((positions * capture_rays).astype(np.int64), capture_rays - 1)
    for ray_index in np.unique(ray_indices):
        ray_mask = ray_indices == ray_index
        if int(np.count_nonzero(ray_mask)) < min_points_per_ray:
            continue
        hit_range = float(np.quantile(selected_ranges[ray_mask], range_percentile))
        hit_z = float(np.median(selected_z[ray_mask]))
        ray_angle = ray_angles[ray_index]
        endpoints[ray_index] = (hit_range * math.cos(ray_angle), hit_range * math.sin(ray_angle), hit_z)
        ray_states[ray_index] = 2
    return endpoints, ray_states


def deskew_points_to_reference_base(
    points_lidar_m: np.ndarray,
    offsets_s: np.ndarray,
    *,
    start_translation_w_m: np.ndarray,
    start_rotation_w_lidar: np.ndarray,
    end_translation_w_m: np.ndarray,
    end_rotation_w_lidar: np.ndarray,
    reference_translation_w_m: np.ndarray,
    reference_rotation_w_base: np.ndarray,
) -> np.ndarray:
    """Deskew one rolling cloud into a common reference ``base_link`` frame.

    Start/end transforms map the lidar frame into world coordinates at the
    earliest/latest finite point timestamp. Translation is interpolated linearly
    and rotation with quaternion SLERP for every point timestamp.
    """
    points = np.asarray(points_lidar_m, dtype=np.float64)
    offsets = np.asarray(offsets_s, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or offsets.shape != (points.shape[0],):
        raise ValueError("points_lidar_m must be (N, 3) and offsets_s must be (N,)")
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if not np.isfinite(points).all() or not np.isfinite(offsets).all():
        raise ValueError("Deskew inputs must be finite")

    start_translation = np.asarray(start_translation_w_m, dtype=np.float64)
    end_translation = np.asarray(end_translation_w_m, dtype=np.float64)
    reference_translation = np.asarray(reference_translation_w_m, dtype=np.float64)
    start_rotation = np.asarray(start_rotation_w_lidar, dtype=np.float64)
    end_rotation = np.asarray(end_rotation_w_lidar, dtype=np.float64)
    reference_rotation = np.asarray(reference_rotation_w_base, dtype=np.float64)
    if any(value.shape != (3,) for value in (start_translation, end_translation, reference_translation)):
        raise ValueError("Translations must have shape (3,)")
    if any(value.shape != (3, 3) for value in (start_rotation, end_rotation, reference_rotation)):
        raise ValueError("Rotations must have shape (3, 3)")

    offset_min = float(np.min(offsets))
    offset_span = float(np.max(offsets) - offset_min)
    alpha = np.zeros(points.shape[0], dtype=np.float64) if offset_span <= 1e-9 else (offsets - offset_min) / offset_span
    translations = start_translation[None, :] + alpha[:, None] * (end_translation - start_translation)[None, :]
    rotations = _slerp_rotation_matrices(start_rotation, end_rotation, alpha)
    world_points = np.einsum("nij,nj->ni", rotations, points) + translations
    return np.ascontiguousarray((world_points - reference_translation[None, :]) @ reference_rotation, dtype=np.float64)


def _slerp_rotation_matrices(start_rotation: np.ndarray, end_rotation: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Vectorized quaternion SLERP, returning one rotation matrix per alpha."""
    q0 = np.asarray(quaternion_from_rotation_matrix(start_rotation), dtype=np.float64)
    q1 = np.asarray(quaternion_from_rotation_matrix(end_rotation), dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    alpha = np.asarray(alpha, dtype=np.float64)
    if dot > 0.9995:
        quaternions = q0[None, :] + alpha[:, None] * (q1 - q0)[None, :]
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        w0 = np.sin((1.0 - alpha) * theta) / sin_theta
        w1 = np.sin(alpha * theta) / sin_theta
        quaternions = w0[:, None] * q0[None, :] + w1[:, None] * q1[None, :]

    x, y, z, w = (quaternions[:, index] for index in range(4))
    rotations = np.empty((alpha.size, 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - w * z)
    rotations[:, 0, 2] = 2.0 * (x * z + w * y)
    rotations[:, 1, 0] = 2.0 * (x * y + w * z)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - w * x)
    rotations[:, 2, 0] = 2.0 * (x * z - w * y)
    rotations[:, 2, 1] = 2.0 * (y * z + w * x)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotations


def normalized_scan_age(now_ns: int, scan_stamp_ns: int, max_age_s: float) -> float:
    """Return the training-compatible normalized held-scan age."""
    if max_age_s <= 0.0:
        raise ValueError("max_age_s must be positive")
    return float(min(max((now_ns - scan_stamp_ns) / 1e9, 0.0) / max_age_s, 1.0))


def project_history_to_front_arc(
    history: tuple[CompletedScan, ...],
    *,
    current_xy_m: np.ndarray,
    current_yaw_rad: float,
    max_distance_m: float = MAX_DISTANCE_M,
    world_bins: int = WORLD_BINS,
    fov_bins: int = FOV_BINS,
    history_frames: int = HISTORY_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Project stored world rays exactly as ``TemporalLidarScan`` does.

    Returned arrays have shape ``(history_frames, fov_bins)``. Missing history
    is max-distance and invalid. Captured free-space rays are max-distance and
    valid, while captured hits use their reprojected distance. The
    requested front arc is selected only after all points have been placed in a
    360-degree world-aligned bin grid, so returns remain available when the
    robot turns away and later turns back within the history horizon.
    """
    if world_bins <= 0 or fov_bins <= 0 or fov_bins > world_bins:
        raise ValueError("Expected 0 < fov_bins <= world_bins")
    if fov_bins % 2 != 0:
        raise ValueError("fov_bins must be even for a symmetric front arc")

    world_distances, world_validity = project_history_to_polar_bins(
        history,
        current_xy_m=current_xy_m,
        max_distance_m=max_distance_m,
        world_bins=world_bins,
        history_frames=history_frames,
    )
    current_xy = np.asarray(current_xy_m, dtype=np.float64)
    if current_xy.shape != (2,):
        raise ValueError("current_xy_m must have shape (2,)")

    front_indices = front_arc_bin_indices(current_yaw_rad, world_bins=world_bins, fov_bins=fov_bins)

    return (
        world_distances[:, front_indices],
        world_validity[:, front_indices],
    )


def project_history_to_polar_bins(
    history: tuple[CompletedScan, ...],
    *,
    current_xy_m: np.ndarray,
    max_distance_m: float = MAX_DISTANCE_M,
    world_bins: int = WORLD_BINS,
    history_frames: int = HISTORY_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce fixed hit/free front-ray history into world polar bins.

    Free rays use their stored endpoint only to select an angular bin; their
    distance remains max range after robot motion. Bins without any captured
    front ray remain invalid. The 360-degree grid is only an internal
    reprojection domain--a completed scan never directly observes the rear.
    """
    if world_bins <= 0 or history_frames <= 0:
        raise ValueError("world_bins and history_frames must be positive")
    distances = np.ones((history_frames, world_bins), dtype=np.float32)
    validity = np.zeros((history_frames, world_bins), dtype=np.uint8)
    current_xy = np.asarray(current_xy_m, dtype=np.float64)
    if current_xy.shape != (2,):
        raise ValueError("current_xy_m must have shape (2,)")

    for frame_index, scan in enumerate(history[:history_frames]):
        world_distances = np.full(world_bins, max_distance_m, dtype=np.float64)
        world_validity = np.zeros(world_bins, dtype=np.uint8)

        endpoints = np.asarray(scan.endpoints_xyz_m, dtype=np.float64)
        ray_states = np.asarray(scan.ray_states, dtype=np.uint8)
        if endpoints.size:
            if endpoints.ndim != 2 or endpoints.shape[1] < 2:
                raise ValueError("Completed scan endpoints must have shape (N, >=2)")
            if ray_states.shape != (endpoints.shape[0],):
                raise ValueError("Completed scan ray states must match its endpoints")
            delta_xy = endpoints[:, :2] - current_xy[None, :]
            ranges = np.linalg.norm(delta_xy, axis=1)
            real = (ray_states > 0) & np.isfinite(ranges) & np.isfinite(delta_xy).all(axis=1)
            if np.any(real):
                delta_xy = delta_xy[real]
                ranges = np.minimum(ranges[real], max_distance_m)
                real_states = ray_states[real]
                angles = np.arctan2(delta_xy[:, 1], delta_xy[:, 0])
                bin_indices = (
                    ((angles + math.pi) / (2.0 * math.pi) * world_bins).astype(np.int64) % world_bins
                )
                ray_distances = np.where(real_states == 2, ranges, max_distance_m)
                np.minimum.at(world_distances, bin_indices, ray_distances)
                world_validity[bin_indices] = 1

        distances[frame_index] = (world_distances / max_distance_m).astype(np.float32)
        validity[frame_index] = world_validity

    return distances, validity


def upsample_polar_bins(
    distances: np.ndarray,
    validity: np.ndarray,
    *,
    target_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat each coarse 360-degree bin into an integer number of fine bins.

    The temporal policy remains trained on a 256-bin virtual world grid and a
    128-bin front arc.  Hardware may instead form a coarser physical grid (128
    bins globally / 64 bins across the front), then repeat each bin so the
    policy interface and its angular extent remain unchanged.
    """
    source_distances = np.asarray(distances)
    source_validity = np.asarray(validity)
    if source_distances.ndim != 2 or source_validity.shape != source_distances.shape:
        raise ValueError("distances and validity must be matching two-dimensional arrays")
    source_bins = source_distances.shape[1]
    if source_bins <= 0 or target_bins < source_bins or target_bins % source_bins != 0:
        raise ValueError("target_bins must be an integer multiple of the source bin count")
    factor = target_bins // source_bins
    return (
        np.repeat(source_distances, factor, axis=1),
        np.repeat(source_validity, factor, axis=1),
    )


def front_arc_bin_indices(
    current_yaw_rad: float,
    *,
    world_bins: int = WORLD_BINS,
    fov_bins: int = FOV_BINS,
) -> np.ndarray:
    """Return world-bin indices in the exact left-to-right policy arc order."""
    if world_bins <= 0 or fov_bins <= 0 or fov_bins > world_bins or fov_bins % 2 != 0:
        raise ValueError("Expected an even 0 < fov_bins <= world_bins")
    center_bin = int(((current_yaw_rad + math.pi) / (2.0 * math.pi)) * world_bins) % world_bins
    offsets = np.arange(-(fov_bins // 2), fov_bins - (fov_bins // 2), dtype=np.int64)
    return (center_bin + offsets) % world_bins


def polar_bins_to_base_points(
    distances: np.ndarray,
    validity: np.ndarray,
    *,
    current_yaw_rad: float,
    max_distance_m: float = MAX_DISTANCE_M,
    bin_indices: np.ndarray | None = None,
    world_bins: int = WORLD_BINS,
) -> np.ndarray:
    """Convert valid normalized polar bins to XY points in the current base frame."""
    distances = np.asarray(distances, dtype=np.float32)
    validity = np.asarray(validity, dtype=np.uint8)
    if distances.ndim != 1 or validity.shape != distances.shape:
        raise ValueError("distances and validity must be matching one-dimensional arrays")
    if bin_indices is None:
        bin_indices = np.arange(distances.size, dtype=np.int64)
    bin_indices = np.asarray(bin_indices, dtype=np.int64)
    if bin_indices.shape != distances.shape:
        raise ValueError("bin_indices must match distances")
    if world_bins <= 0 or max_distance_m <= 0:
        raise ValueError("world_bins and max_distance_m must be positive")

    present = validity > 0
    if not np.any(present):
        return np.empty((0, 3), dtype=np.float32)
    world_angles = ((bin_indices[present].astype(np.float64) + 0.5) / world_bins) * (2.0 * math.pi) - math.pi
    base_angles = world_angles - current_yaw_rad
    ranges = distances[present].astype(np.float64) * max_distance_m
    return np.column_stack((ranges * np.cos(base_angles), ranges * np.sin(base_angles), np.zeros(ranges.size))).astype(np.float32)
