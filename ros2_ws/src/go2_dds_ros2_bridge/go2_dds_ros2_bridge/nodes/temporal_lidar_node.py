#!/usr/bin/env python3
"""Aggregate corrected raw clouds into temporal policy observations."""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from dataclasses import dataclass
from time import time_ns

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

from go2_dds_ros2_bridge_msgs.msg import TemporalLidarObservation
from go2_dds_ros2_bridge.occupancy_map import extract_xyz_time_points
from go2_dds_ros2_bridge.temporal_lidar_processing import (
    CAPTURE_FOV_DEG,
    CAPTURE_RAYS,
    CompletedScan,
    CompletedScanHistory,
    FOV_BINS,
    HISTORY_FRAMES,
    MAX_DISTANCE_M,
    WORLD_BINS,
    deskew_points_to_reference_base,
    front_arc_bin_indices,
    is_adjacent_cloud_pair,
    normalized_scan_age,
    polar_bins_to_base_points,
    project_history_to_polar_bins,
    reduce_front_capture_rays,
    upsample_polar_bins,
)
from go2_dds_ros2_bridge.tf_utils import rotation_matrix_from_quaternion_xyzw


DEFAULT_INPUT_TOPIC = "/utlidar/time_corrected/cloud"
DEFAULT_OUTPUT_TOPIC = "/temporal_lidar/observation"
DEFAULT_MAP_FRAME = "camera_init_correct"
DEFAULT_BASE_FRAME = "base_link"
DEFAULT_RAW_CLOUD_PERIOD_S = 0.065
DEFAULT_INTERCLOUD_TOLERANCE_S = 0.025
DEFAULT_POLICY_HZ = 12.5
DEFAULT_PROCESSING_HZ = 100.0
DEFAULT_TF_WAIT_S = 0.25
DEFAULT_MAX_PENDING_CLOUDS = 16
DEFAULT_MIN_Z_M = -0.25
DEFAULT_MAX_Z_M = 1.4
DEFAULT_MAX_RANGE_M = MAX_DISTANCE_M
DEFAULT_SCAN_AGE_MAX_S = 0.250
DEFAULT_DEBUG_ENABLED = True
DEFAULT_DEBUG_TOPIC_PREFIX = "/temporal_lidar/debug"
DEFAULT_FREEZE_OBSERVATION = False
DEFAULT_POINT_TIME_FIELD = "time"
DEFAULT_POINT_TIME_SCALE_S = 1.0
DEFAULT_CAPTURE_RAYS = CAPTURE_RAYS
DEFAULT_CAPTURE_FOV_DEG = CAPTURE_FOV_DEG
DEFAULT_MIN_POINTS_PER_CAPTURE_RAY = 1
DEFAULT_CAPTURE_RETURN_PERCENTILE = 0.1
DEFAULT_WORLD_BINS = WORLD_BINS

FRAME_COLORS_RGB = (0xFF3333, 0x33CCFF, 0x66FF66, 0xCC66FF)

CLOUD_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
OBSERVATION_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


@dataclass(frozen=True)
class TemporalLidarConfig:
    input_topic: str
    output_topic: str
    map_frame: str
    base_frame: str
    raw_cloud_period_s: float
    intercloud_tolerance_s: float
    policy_hz: float
    processing_hz: float
    tf_wait_s: float
    max_pending_clouds: int
    min_z_m: float
    max_z_m: float
    max_range_m: float
    scan_age_max_s: float
    debug_enabled: bool
    debug_topic_prefix: str
    freeze_observation: bool
    point_time_field: str
    point_time_scale_s: float
    capture_rays: int
    capture_fov_deg: float
    min_points_per_capture_ray: int
    capture_return_percentile: float
    world_bins: int


def parse_args() -> TemporalLidarConfig:
    parser = argparse.ArgumentParser(description="Build four-frame temporal lidar observations from corrected raw clouds.")
    parser.add_argument("--input-topic", default=DEFAULT_INPUT_TOPIC)
    parser.add_argument("--output-topic", default=DEFAULT_OUTPUT_TOPIC)
    parser.add_argument("--map-frame", default=DEFAULT_MAP_FRAME)
    parser.add_argument("--base-frame", default=DEFAULT_BASE_FRAME)
    parser.add_argument("--raw-cloud-period-s", type=float, default=DEFAULT_RAW_CLOUD_PERIOD_S)
    parser.add_argument("--intercloud-tolerance-s", type=float, default=DEFAULT_INTERCLOUD_TOLERANCE_S)
    parser.add_argument("--policy-hz", type=float, default=DEFAULT_POLICY_HZ)
    parser.add_argument("--processing-hz", type=float, default=DEFAULT_PROCESSING_HZ)
    parser.add_argument("--tf-wait-s", type=float, default=DEFAULT_TF_WAIT_S)
    parser.add_argument("--max-pending-clouds", type=int, default=DEFAULT_MAX_PENDING_CLOUDS)
    parser.add_argument("--min-z-m", type=float, default=DEFAULT_MIN_Z_M)
    parser.add_argument("--max-z-m", type=float, default=DEFAULT_MAX_Z_M)
    parser.add_argument("--max-range-m", type=float, default=DEFAULT_MAX_RANGE_M)
    parser.add_argument("--scan-age-max-s", type=float, default=DEFAULT_SCAN_AGE_MAX_S)
    parser.add_argument("--debug-enabled", action=argparse.BooleanOptionalAction, default=DEFAULT_DEBUG_ENABLED)
    parser.add_argument("--debug-topic-prefix", default=DEFAULT_DEBUG_TOPIC_PREFIX)
    parser.add_argument("--point-time-field", default=DEFAULT_POINT_TIME_FIELD)
    parser.add_argument("--point-time-scale-s", type=float, default=DEFAULT_POINT_TIME_SCALE_S)
    parser.add_argument("--capture-rays", type=int, default=DEFAULT_CAPTURE_RAYS)
    parser.add_argument("--capture-fov-deg", type=float, default=DEFAULT_CAPTURE_FOV_DEG)
    parser.add_argument("--min-points-per-capture-ray", type=int, default=DEFAULT_MIN_POINTS_PER_CAPTURE_RAY)
    parser.add_argument("--capture-return-percentile", type=float, default=DEFAULT_CAPTURE_RETURN_PERCENTILE)
    parser.add_argument("--world-bins", type=int, default=DEFAULT_WORLD_BINS)
    parser.add_argument(
        "--freeze-observation",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FREEZE_OBSERVATION,
        help=(
            "After four completed scans are available, freeze their projected policy observation and "
            "republish it with fresh ROS timestamps. Intended only for lidar-ablation experiments."
        ),
    )
    args = parser.parse_args(rclpy.utilities.remove_ros_args(args=sys.argv)[1:])
    if args.raw_cloud_period_s <= 0 or args.intercloud_tolerance_s < 0:
        raise SystemExit("raw-cloud-period-s must be positive and intercloud-tolerance-s non-negative")
    if args.policy_hz <= 0 or args.processing_hz <= 0 or args.max_range_m <= 0:
        raise SystemExit("policy-hz, processing-hz, and max-range-m must be positive")
    if args.tf_wait_s <= 0 or args.max_pending_clouds <= 0:
        raise SystemExit("tf-wait-s and max-pending-clouds must be positive")
    if args.max_z_m <= args.min_z_m or args.scan_age_max_s <= 0:
        raise SystemExit("max-z-m must exceed min-z-m and scan-age-max-s must be positive")
    if not args.point_time_field or args.point_time_scale_s <= 0.0:
        raise SystemExit("point-time-field must be non-empty and point-time-scale-s must be positive")
    if args.capture_rays <= 0 or not 0.0 < args.capture_fov_deg <= 360.0:
        raise SystemExit("capture-rays must be positive and capture-fov-deg must be in (0, 360]")
    if args.min_points_per_capture_ray <= 0 or not 0.0 <= args.capture_return_percentile <= 1.0:
        raise SystemExit("Invalid capture-ray return-reduction parameters")
    if args.world_bins <= 0 or args.world_bins > WORLD_BINS or WORLD_BINS % args.world_bins != 0:
        raise SystemExit("world-bins must be a positive divisor of %d" % WORLD_BINS)
    if not args.debug_topic_prefix:
        raise SystemExit("debug-topic-prefix must not be empty")
    return TemporalLidarConfig(**vars(args))


@dataclass(frozen=True)
class PendingCloud:
    message: PointCloud2
    stamp: Time
    received_ns: int


@dataclass(frozen=True)
class DecodedRollingCloud:
    points_lidar_m: np.ndarray
    time_offsets_s: np.ndarray
    start_stamp_ns: int
    end_stamp_ns: int
    frame_id: str


class TemporalLidarNode(Node):
    def __init__(self, config: TemporalLidarConfig) -> None:
        super().__init__("temporal_lidar")
        for field, value in vars(config).items():
            self.declare_parameter(field, value)
        self._config = TemporalLidarConfig(**{field: self.get_parameter(field).value for field in vars(config)})
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._history = CompletedScanHistory(HISTORY_FRAMES)
        self._frozen_distances: np.ndarray | None = None
        self._frozen_validity: np.ndarray | None = None
        self._frozen_normalized_scan_age: float | None = None
        self._pending_clouds: deque[PendingCloud] = deque()
        self._tf_warning_last_ns = 0
        self._gap_warning_last_ns = 0
        self._subscription = self.create_subscription(PointCloud2, self._config.input_topic, self._cloud_callback, CLOUD_QOS)
        self._publisher = self.create_publisher(TemporalLidarObservation, self._config.output_topic, OBSERVATION_QOS)
        self._debug_frame_publishers = []
        self._debug_all_publisher = None
        self._debug_full_360_publisher = None
        if self._config.debug_enabled:
            prefix = self._config.debug_topic_prefix.rstrip("/")
            self._debug_frame_publishers = [
                self.create_publisher(PointCloud2, f"{prefix}/frame_{index}_bins", OBSERVATION_QOS)
                for index in range(HISTORY_FRAMES)
            ]
            self._debug_all_publisher = self.create_publisher(PointCloud2, f"{prefix}/all_bins", OBSERVATION_QOS)
            self._debug_full_360_publisher = self.create_publisher(
                PointCloud2, f"{prefix}/full_360_bins", OBSERVATION_QOS
            )
        self._process_timer = self.create_timer(1.0 / self._config.processing_hz, self._process_pending_cloud)
        self._observation_timer = self.create_timer(1.0 / self._config.policy_hz, self._publish_observation)
        self.get_logger().info(
            "Temporal lidar: '%s' -> '%s'; two rolling corrected raw clouds (expected %.3fs apart) are deskewed "
            "to cloud-two end time, reduced to %d fixed hit/free rays across the front %.0f degrees, then retained "
            "as %d world-frame "
            "histories at %.1f Hz policy output. Deferred TF processing at %.1f Hz. "
            "RViz bin debug: %s. Frozen-observation experiment: %s."
            % (self._config.input_topic, self._config.output_topic, self._config.raw_cloud_period_s,
               self._config.capture_rays, self._config.capture_fov_deg, HISTORY_FRAMES,
               self._config.policy_hz, self._config.processing_hz,
               self._config.debug_topic_prefix if self._config.debug_enabled else "disabled",
               "enabled" if self._config.freeze_observation else "disabled")
        )

    def _warn_throttled(self, message: str, *, gap: bool = False) -> None:
        now_ns = time_ns()
        previous = self._gap_warning_last_ns if gap else self._tf_warning_last_ns
        if now_ns - previous >= 2_000_000_000:
            if gap:
                self._gap_warning_last_ns = now_ns
            else:
                self._tf_warning_last_ns = now_ns
            self.get_logger().warning(message)

    def _lookup_transform(self, target_frame: str, source_frame: str, stamp: Time) -> tuple[np.ndarray, np.ndarray] | None:
        """Look up a transform without blocking this node's single-threaded executor."""
        if target_frame == source_frame:
            return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
        try:
            transform = self._tf_buffer.lookup_transform(target_frame, source_frame, stamp, timeout=Duration())
        except TransformException as error:
            self._warn_throttled("Missing timestamped transform %s <- %s: %s" % (target_frame, source_frame, error))
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            np.array((translation.x, translation.y, translation.z), dtype=np.float64),
            rotation_matrix_from_quaternion_xyzw(rotation.x, rotation.y, rotation.z, rotation.w),
        )

    def _cloud_callback(self, cloud_msg: PointCloud2) -> None:
        if self._frozen_distances is not None:
            return
        stamp = Time.from_msg(cloud_msg.header.stamp)
        if stamp.nanoseconds == 0:
            self._warn_throttled("Dropping corrected raw cloud without a header timestamp.")
            return
        if len(self._pending_clouds) >= self._config.max_pending_clouds:
            self._pending_clouds.popleft()
            self._warn_throttled(
                "Pending corrected-raw-cloud queue is full; dropping its oldest cloud and restarting pair assembly.",
                gap=True,
            )
        self._pending_clouds.append(
            PendingCloud(message=cloud_msg, stamp=stamp, received_ns=self.get_clock().now().nanoseconds)
        )

    def _process_pending_cloud(self) -> None:
        """Deskew and reduce the oldest valid pair once all its TF samples exist."""
        if self._frozen_distances is not None:
            self._pending_clouds.clear()
            return
        if len(self._pending_clouds) < 2:
            return
        first, second = self._pending_clouds[0], self._pending_clouds[1]
        interval_s = (second.stamp.nanoseconds - first.stamp.nanoseconds) / 1e9
        if not is_adjacent_cloud_pair(
            first.stamp.nanoseconds,
            second.stamp.nanoseconds,
            expected_period_s=self._config.raw_cloud_period_s,
            tolerance_s=self._config.intercloud_tolerance_s,
        ):
            self._pending_clouds.popleft()
            self._warn_throttled(
                "Corrected-raw-cloud interval %.3fs is not %.3f±%.3fs; keeping the newer cloud as the next pair start."
                % (interval_s, self._config.raw_cloud_period_s, self._config.intercloud_tolerance_s),
                gap=True,
            )
            return

        try:
            first_cloud = self._decode_rolling_cloud(first)
            second_cloud = self._decode_rolling_cloud(second)
        except ValueError as error:
            self._pending_clouds.popleft()
            self._pending_clouds.popleft()
            self._warn_throttled("Dropping malformed rolling-cloud pair: %s" % error, gap=True)
            return

        completed = self._complete_deskewed_pair(first_cloud, second_cloud)
        if completed is None:
            wait_s = (self.get_clock().now().nanoseconds - second.received_ns) / 1e9
            if wait_s <= self._config.tf_wait_s:
                return
            self._pending_clouds.popleft()
            self._pending_clouds.popleft()
            self._warn_throttled(
                "Dropping rolling-cloud pair after waiting %.3fs for deskew transforms." % wait_s,
                gap=True,
            )
            return
        self._pending_clouds.popleft()
        self._pending_clouds.popleft()
        self._history.push(completed)

    def _decode_rolling_cloud(self, pending: PendingCloud) -> DecodedRollingCloud:
        points, offsets, time_field = extract_xyz_time_points(
            pending.message, time_field_name=self._config.point_time_field
        )
        offsets_s = offsets * self._config.point_time_scale_s
        if np.any(offsets_s < -1e-6):
            raise ValueError("Per-point '%s' offsets must be non-negative relative to the header timestamp" % time_field)
        offsets_s = np.maximum(offsets_s, 0.0)
        return DecodedRollingCloud(
            points_lidar_m=points,
            time_offsets_s=offsets_s,
            start_stamp_ns=pending.stamp.nanoseconds + int(round(float(np.min(offsets_s)) * 1e9)),
            end_stamp_ns=pending.stamp.nanoseconds + int(round(float(np.max(offsets_s)) * 1e9)),
            frame_id=pending.message.header.frame_id or self._config.map_frame,
        )

    def _complete_deskewed_pair(
        self, first: DecodedRollingCloud, second: DecodedRollingCloud
    ) -> "CompletedScan | None":
        reference_stamp = Time(nanoseconds=second.end_stamp_ns)
        reference_base = self._lookup_transform(self._config.map_frame, self._config.base_frame, reference_stamp)
        transforms = []
        for cloud in (first, second):
            start = self._lookup_transform(
                self._config.map_frame, cloud.frame_id, Time(nanoseconds=cloud.start_stamp_ns)
            )
            end = self._lookup_transform(
                self._config.map_frame, cloud.frame_id, Time(nanoseconds=cloud.end_stamp_ns)
            )
            transforms.append((start, end))
        if reference_base is None or any(start is None or end is None for start, end in transforms):
            return None

        reference_translation, reference_rotation = reference_base
        deskewed_clouds = []
        for cloud, (start, end) in zip((first, second), transforms):
            start_translation, start_rotation = start
            end_translation, end_rotation = end
            deskewed_clouds.append(
                deskew_points_to_reference_base(
                    cloud.points_lidar_m,
                    cloud.time_offsets_s,
                    start_translation_w_m=start_translation,
                    start_rotation_w_lidar=start_rotation,
                    end_translation_w_m=end_translation,
                    end_rotation_w_lidar=end_rotation,
                    reference_translation_w_m=reference_translation,
                    reference_rotation_w_base=reference_rotation,
                )
            )
        points_reference = np.concatenate(deskewed_clouds, axis=0)
        horizontal_range = np.linalg.norm(points_reference[:, :2], axis=1)
        height_and_range = (
            (points_reference[:, 2] >= self._config.min_z_m)
            & (points_reference[:, 2] <= self._config.max_z_m)
            & (horizontal_range <= self._config.max_range_m)
        )
        capture_endpoints_reference, ray_states = reduce_front_capture_rays(
            points_reference[height_and_range],
            capture_rays=self._config.capture_rays,
            fov_degrees=self._config.capture_fov_deg,
            max_distance_m=self._config.max_range_m,
            min_points_per_ray=self._config.min_points_per_capture_ray,
            range_percentile=self._config.capture_return_percentile,
        )
        capture_endpoints_world = np.ascontiguousarray(
            capture_endpoints_reference @ reference_rotation.T + reference_translation,
            dtype=np.float64,
        )
        return CompletedScan(
            stamp_ns=second.end_stamp_ns,
            endpoints_xyz_m=capture_endpoints_world,
            ray_states=ray_states,
        )

    def _current_pose(self) -> tuple[np.ndarray, float] | None:
        result = self._lookup_transform(self._config.map_frame, self._config.base_frame, Time())
        if result is None:
            return None
        translation, rotation = result
        return translation[:2], math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))

    def _build_debug_cloud(self, *, stamp: Time, points_xyz_m: np.ndarray, frame_indices: np.ndarray) -> PointCloud2:
        """Build a colorized base-frame cloud; every point is one valid polar bin."""
        count = int(points_xyz_m.shape[0])
        cloud = PointCloud2()
        cloud.header.stamp = stamp.to_msg()
        cloud.header.frame_id = self._config.base_frame
        cloud.height = 1
        cloud.width = count
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
            PointField(name="frame_index", offset=16, datatype=PointField.UINT8, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 20
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        packed = np.empty(
            count,
            dtype=np.dtype({
                "names": ("x", "y", "z", "rgb", "frame_index"),
                "formats": ("<f4", "<f4", "<f4", "<u4", "u1"),
                "offsets": (0, 4, 8, 12, 16),
                "itemsize": 20,
            }),
        )
        if count:
            packed["x"] = points_xyz_m[:, 0]
            packed["y"] = points_xyz_m[:, 1]
            packed["z"] = points_xyz_m[:, 2]
            packed["rgb"] = np.asarray([FRAME_COLORS_RGB[int(i)] for i in frame_indices], dtype=np.uint32)
            packed["frame_index"] = frame_indices
        cloud.data = packed.tobytes()
        return cloud

    def _publish_debug_clouds(
        self,
        *,
        stamp: Time,
        world_distances: np.ndarray,
        world_validity: np.ndarray,
        front_indices: np.ndarray,
        current_yaw: float,
    ) -> None:
        if not self._debug_frame_publishers:
            return
        frame_points: list[np.ndarray] = []
        frame_ids: list[np.ndarray] = []
        full_points: list[np.ndarray] = []
        full_ids: list[np.ndarray] = []
        for frame_index, publisher in enumerate(self._debug_frame_publishers):
            front_points = polar_bins_to_base_points(
                world_distances[frame_index, front_indices], world_validity[frame_index, front_indices],
                current_yaw_rad=current_yaw, max_distance_m=self._config.max_range_m,
                bin_indices=front_indices, world_bins=WORLD_BINS,
            )
            frame_id = np.full(front_points.shape[0], frame_index, dtype=np.uint8)
            publisher.publish(self._build_debug_cloud(stamp=stamp, points_xyz_m=front_points, frame_indices=frame_id))
            frame_points.append(front_points)
            frame_ids.append(frame_id)

            all_points = polar_bins_to_base_points(
                world_distances[frame_index], world_validity[frame_index], current_yaw_rad=current_yaw,
                max_distance_m=self._config.max_range_m, world_bins=WORLD_BINS,
            )
            full_points.append(all_points)
            full_ids.append(np.full(all_points.shape[0], frame_index, dtype=np.uint8))

        if self._debug_all_publisher is not None:
            all_front = np.concatenate(frame_points, axis=0) if frame_points else np.empty((0, 3), dtype=np.float32)
            all_front_ids = np.concatenate(frame_ids) if frame_ids else np.empty(0, dtype=np.uint8)
            self._debug_all_publisher.publish(
                self._build_debug_cloud(stamp=stamp, points_xyz_m=all_front, frame_indices=all_front_ids)
            )
        if self._debug_full_360_publisher is not None:
            all_360 = np.concatenate(full_points, axis=0) if full_points else np.empty((0, 3), dtype=np.float32)
            all_360_ids = np.concatenate(full_ids) if full_ids else np.empty(0, dtype=np.uint8)
            self._debug_full_360_publisher.publish(
                self._build_debug_cloud(stamp=stamp, points_xyz_m=all_360, frame_indices=all_360_ids)
            )

    def _publish_observation(self) -> None:
        now = self.get_clock().now()
        if self._frozen_distances is not None:
            # ``scan_stamp`` must be current so navigation's completed-scan timeout
            # continues to test only the frozen lidar contents, rather than timing
            # out the experiment. The frozen normalized age remains part of the
            # injected policy observation.
            message = TemporalLidarObservation()
            message.header.stamp = now.to_msg()
            message.header.frame_id = self._config.base_frame
            message.scan_stamp = now.to_msg()
            message.normalized_scan_age = self._frozen_normalized_scan_age
            message.distances = self._frozen_distances.tolist()
            message.validity = self._frozen_validity.tolist()
            self._publisher.publish(message)
            return

        newest_stamp_ns = self._history.newest_stamp_ns
        if newest_stamp_ns is None:
            return
        pose = self._current_pose()
        if pose is None:
            return
        current_xy, current_yaw = pose
        coarse_world_distances, coarse_world_validity = project_history_to_polar_bins(
            self._history.newest_first(), current_xy_m=current_xy,
            max_distance_m=self._config.max_range_m,
            world_bins=self._config.world_bins,
        )
        world_distances, world_validity = upsample_polar_bins(
            coarse_world_distances, coarse_world_validity, target_bins=WORLD_BINS
        )
        front_indices = front_arc_bin_indices(current_yaw, world_bins=WORLD_BINS, fov_bins=FOV_BINS)
        distances = world_distances[:, front_indices]
        validity = world_validity[:, front_indices]
        normalized_age = normalized_scan_age(
            now.nanoseconds, newest_stamp_ns, self._config.scan_age_max_s
        )

        if self._config.freeze_observation and len(self._history) >= HISTORY_FRAMES:
            self._frozen_distances = distances.reshape(-1).copy()
            self._frozen_validity = validity.reshape(-1).copy()
            self._frozen_normalized_scan_age = normalized_age
            self._pending_clouds.clear()
            self.get_logger().warning(
                "Frozen temporal-lidar observation after %d completed scans; live raw clouds now have no effect. "
                "Restart with freeze_observation:=false to restore live lidar."
                % HISTORY_FRAMES
            )

        message = TemporalLidarObservation()
        message.header.stamp = now.to_msg()
        message.header.frame_id = self._config.base_frame
        message.scan_stamp = Time(nanoseconds=newest_stamp_ns).to_msg()
        message.normalized_scan_age = normalized_age
        message.distances = distances.reshape(-1).tolist()
        message.validity = validity.reshape(-1).tolist()
        self._publisher.publish(message)
        self._publish_debug_clouds(
            stamp=now,
            world_distances=world_distances,
            world_validity=world_validity,
            front_indices=front_indices,
            current_yaw=current_yaw,
        )


def main() -> None:
    config = parse_args()
    rclpy.init(args=None)
    node = TemporalLidarNode(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
