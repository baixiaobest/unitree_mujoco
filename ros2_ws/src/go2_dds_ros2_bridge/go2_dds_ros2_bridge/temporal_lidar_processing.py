"""Pure geometry and history helpers for the temporal raw-cloud lidar path."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


HISTORY_FRAMES = 4
WORLD_BINS = 256
FOV_BINS = 128
MAX_DISTANCE_M = 20.0


@dataclass(frozen=True)
class CompletedScan:
    """One completed raw-cloud scan stored in corrected world coordinates.

    ``points_xyz_m`` contains physical surface returns.  ``free_endpoints_xyz_m``
    contains the max-range endpoints of directions covered by a valid completed
    scan.  This mirrors the simulator's two ray states: a surface hit has a
    reprojected range, while a covered no-return ray is valid free space at
    ``max_distance``.
    """

    stamp_ns: int
    points_xyz_m: np.ndarray
    free_endpoints_xyz_m: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )


class CompletedScanHistory:
    """Newest-first bounded history of completed raw-cloud scans."""

    def __init__(self, depth: int = HISTORY_FRAMES) -> None:
        self._frames: deque[CompletedScan] = deque(maxlen=depth)

    def push(self, scan: CompletedScan) -> None:
        self._frames.appendleft(scan)

    def clear(self) -> None:
        self._frames.clear()

    @property
    def newest_stamp_ns(self) -> int | None:
        return self._frames[0].stamp_ns if self._frames else None

    def newest_first(self) -> tuple[CompletedScan, ...]:
        return tuple(self._frames)


class TwoCloudAssembler:
    """Turn adjacent partial clouds into timestamped completed scans."""

    def __init__(self, raw_cloud_period_s: float, intercloud_tolerance_s: float) -> None:
        self.raw_cloud_period_s = raw_cloud_period_s
        self.intercloud_tolerance_s = intercloud_tolerance_s
        self._pending_points_xyz_m: np.ndarray | None = None
        self._pending_stamp_ns: int | None = None
        self.last_interval_s: float | None = None

    def discard(self) -> None:
        self._pending_points_xyz_m = None
        self._pending_stamp_ns = None
        self.last_interval_s = None

    def push(
        self,
        points_xyz_m: np.ndarray,
        stamp_ns: int,
        *,
        free_endpoints_xyz_m: np.ndarray | None = None,
    ) -> tuple[CompletedScan | None, bool]:
        """Return ``(completed_scan, rejected_prior_pair)`` for one raw cloud."""
        if self._pending_stamp_ns is None:
            self._pending_points_xyz_m = points_xyz_m
            self._pending_stamp_ns = stamp_ns
            return None, False

        interval_s = (stamp_ns - self._pending_stamp_ns) / 1e9
        self.last_interval_s = interval_s
        valid_interval = interval_s > 0.0 and abs(interval_s - self.raw_cloud_period_s) <= self.intercloud_tolerance_s
        if not valid_interval:
            self._pending_points_xyz_m = points_xyz_m
            self._pending_stamp_ns = stamp_ns
            return None, True

        points = np.concatenate((self._pending_points_xyz_m, points_xyz_m), axis=0)
        free_endpoints = (
            np.empty((0, 3), dtype=np.float64)
            if free_endpoints_xyz_m is None
            else np.asarray(free_endpoints_xyz_m, dtype=np.float64)
        )
        self.discard()
        return CompletedScan(
            stamp_ns=stamp_ns,
            points_xyz_m=points,
            free_endpoints_xyz_m=free_endpoints,
        ), False


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
    """Project stored world returns exactly as TemporalLidarScan's hit path does.

    Returned arrays have shape ``(history_frames, fov_bins)``. Missing history
    and bins without a physical return are max-distance and invalid. The
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
    """Reduce every history frame into 360-degree polar bins.

    A free endpoint creates a valid max-range bin; a physical return then
    replaces that range with the closest surface return.  Bins without either
    remain max-range but invalid (unknown).  This is the same hit/free binning
    contract as ``TemporalLidarScan`` in training.  The bins are world-angle
    indexed, so the debug cloud exactly matches the polar reduction used before
    the policy's front-arc selection.
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

        free_endpoints = np.asarray(scan.free_endpoints_xyz_m, dtype=np.float64)
        if free_endpoints.size:
            if free_endpoints.ndim != 2 or free_endpoints.shape[1] < 2:
                raise ValueError("Completed scan free endpoints must have shape (N, >=2)")
            free_delta = free_endpoints[:, :2] - current_xy[None, :]
            free_finite = np.isfinite(free_delta).all(axis=1)
            if np.any(free_finite):
                free_angles = np.arctan2(free_delta[free_finite, 1], free_delta[free_finite, 0])
                free_bins = (
                    ((free_angles + math.pi) / (2.0 * math.pi) * world_bins).astype(np.int64) % world_bins
                )
                # Covered no-return rays are valid but always contribute max range.
                world_validity[free_bins] = 1

        points = np.asarray(scan.points_xyz_m, dtype=np.float64)
        if points.size:
            if points.ndim != 2 or points.shape[1] < 2:
                raise ValueError("Completed scan points must have shape (N, >=2)")
            delta_xy = points[:, :2] - current_xy[None, :]
            ranges = np.linalg.norm(delta_xy, axis=1)
            finite = np.isfinite(ranges) & np.isfinite(delta_xy).all(axis=1)
            if np.any(finite):
                delta_xy = delta_xy[finite]
                ranges = np.minimum(ranges[finite], max_distance_m)
                angles = np.arctan2(delta_xy[:, 1], delta_xy[:, 0])
                bin_indices = (
                    ((angles + math.pi) / (2.0 * math.pi) * world_bins).astype(np.int64) % world_bins
                )
                np.minimum.at(world_distances, bin_indices, ranges)
                world_validity[bin_indices] = 1

        distances[frame_index] = (world_distances / max_distance_m).astype(np.float32)
        validity[frame_index] = world_validity

    return distances, validity


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
