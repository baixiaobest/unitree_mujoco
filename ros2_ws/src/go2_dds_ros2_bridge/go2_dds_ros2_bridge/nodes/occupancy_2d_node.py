#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from time import time_ns

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

from go2_dds_ros2_bridge.occupancy_map import (
    NUMBA_AVAILABLE,
    OccupancyMapSnapshot,
    RollingOccupancyMap,
    extract_xyz_points,
)
from go2_dds_ros2_bridge.tf_utils import rotation_matrix_from_quaternion_xyzw


DEFAULT_INPUT_TOPIC = "/cloud_registered"
DEFAULT_OUTPUT_TOPIC = "/slow_occupancy"
DEFAULT_FAST_OUTPUT_TOPIC = "/fast_occupancy"
DEFAULT_DYNAMIC_OUTPUT_TOPIC = "/dynamic_occupancy"
DEFAULT_DEBUG_FILTERED_POINTCLOUD_TOPIC = "/occupancy_2d/filtered_pointcloud"
DEFAULT_MAP_FRAME = "camera_init_correct"
DEFAULT_REQUIRED_CORRECTION_PARENT_FRAME = "camera_init_correct"
DEFAULT_REQUIRED_CORRECTION_CHILD_FRAME = "camera_init"
DEFAULT_BASE_FRAME = "base_link"
DEFAULT_LIDAR_FRAME = "utlidar_lidar"
DEFAULT_MIN_Z_M = 0.1
DEFAULT_MAX_Z_M = 1.5
DEFAULT_RESOLUTION_M = 0.1
DEFAULT_WIDTH_M = 15.0
DEFAULT_HEIGHT_M = 15.0
DEFAULT_MAX_TF_AGE_SEC = 1.0
DEFAULT_DEBUG = False
DEFAULT_FAST_HIT_LOG_ODDS_INCREMENT = 1.0
DEFAULT_FAST_MISS_LOG_ODDS_DECREMENT = -0.5
DEFAULT_SLOW_HIT_LOG_ODDS_INCREMENT = 0.1
DEFAULT_SLOW_MISS_LOG_ODDS_DECREMENT = -0.05
DEFAULT_FAST_OCCUPANCY_THRESHOLD = 2.0
DEFAULT_SLOW_OCCUPANCY_THRESHOLD = 2.0
DEFAULT_FAST_DECAY_FACTOR = 0.999
DEFAULT_SLOW_DECAY_FACTOR = 0.999
DEFAULT_MIN_LOG_ODDS = -8.0
DEFAULT_MAX_LOG_ODDS = 8.0
OUTPUT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


@dataclass(frozen=True)
class Occupancy2DConfig:
    input_topic: str
    output_topic: str
    fast_output_topic: str
    dynamic_output_topic: str
    debug_filtered_pointcloud_topic: str
    map_frame: str
    base_frame: str
    lidar_frame: str
    min_z_m: float
    max_z_m: float
    resolution_m: float
    width_m: float
    height_m: float
    max_tf_age_sec: float
    debug: bool
    fast_hit_log_odds_increment: float
    fast_miss_log_odds_decrement: float
    slow_hit_log_odds_increment: float
    slow_miss_log_odds_decrement: float
    fast_occupancy_threshold: float
    slow_occupancy_threshold: float
    fast_decay_factor: float
    slow_decay_factor: float
    min_log_odds: float
    max_log_odds: float


def parse_args() -> Occupancy2DConfig:
    parser = argparse.ArgumentParser(
        description="Build a rolling 2D occupancy grid from FAST-LIO registered point clouds."
    )
    parser.add_argument("--input-topic", type=str, default=DEFAULT_INPUT_TOPIC)
    parser.add_argument("--output-topic", type=str, default=DEFAULT_OUTPUT_TOPIC)
    parser.add_argument("--fast-output-topic", type=str, default=DEFAULT_FAST_OUTPUT_TOPIC)
    parser.add_argument("--dynamic-output-topic", type=str, default=DEFAULT_DYNAMIC_OUTPUT_TOPIC)
    parser.add_argument("--debug-filtered-pointcloud-topic", type=str, default=DEFAULT_DEBUG_FILTERED_POINTCLOUD_TOPIC)
    parser.add_argument("--map-frame", type=str, default=DEFAULT_MAP_FRAME)
    parser.add_argument("--base-frame", type=str, default=DEFAULT_BASE_FRAME)
    parser.add_argument("--lidar-frame", type=str, default=DEFAULT_LIDAR_FRAME)
    parser.add_argument("--min-z-m", type=float, default=DEFAULT_MIN_Z_M)
    parser.add_argument("--max-z-m", type=float, default=DEFAULT_MAX_Z_M)
    parser.add_argument("--resolution-m", type=float, default=DEFAULT_RESOLUTION_M)
    parser.add_argument("--width-m", type=float, default=DEFAULT_WIDTH_M)
    parser.add_argument("--height-m", type=float, default=DEFAULT_HEIGHT_M)
    parser.add_argument("--max-tf-age-sec", type=float, default=DEFAULT_MAX_TF_AGE_SEC)
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DEBUG,
    )
    parser.add_argument("--fast-hit-log-odds-increment", type=float, default=DEFAULT_FAST_HIT_LOG_ODDS_INCREMENT)
    parser.add_argument("--fast-miss-log-odds-decrement", type=float, default=DEFAULT_FAST_MISS_LOG_ODDS_DECREMENT)
    parser.add_argument("--slow-hit-log-odds-increment", type=float, default=DEFAULT_SLOW_HIT_LOG_ODDS_INCREMENT)
    parser.add_argument("--slow-miss-log-odds-decrement", type=float, default=DEFAULT_SLOW_MISS_LOG_ODDS_DECREMENT)
    parser.add_argument("--fast-occupancy-threshold", type=float, default=DEFAULT_FAST_OCCUPANCY_THRESHOLD)
    parser.add_argument("--slow-occupancy-threshold", type=float, default=DEFAULT_SLOW_OCCUPANCY_THRESHOLD)
    parser.add_argument("--fast-decay-factor", type=float, default=DEFAULT_FAST_DECAY_FACTOR)
    parser.add_argument("--slow-decay-factor", type=float, default=DEFAULT_SLOW_DECAY_FACTOR)
    parser.add_argument("--min-log-odds", type=float, default=DEFAULT_MIN_LOG_ODDS)
    parser.add_argument("--max-log-odds", type=float, default=DEFAULT_MAX_LOG_ODDS)

    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)
    if args.min_z_m >= args.max_z_m:
        raise SystemExit("--min-z-m must be smaller than --max-z-m")
    if args.fast_hit_log_odds_increment <= 0.0 or args.slow_hit_log_odds_increment <= 0.0:
        raise SystemExit("--fast-hit-log-odds-increment and --slow-hit-log-odds-increment must be positive")
    if args.fast_miss_log_odds_decrement >= 0.0 or args.slow_miss_log_odds_decrement >= 0.0:
        raise SystemExit("--fast-miss-log-odds-decrement and --slow-miss-log-odds-decrement must be negative")
    if args.fast_decay_factor <= 0.0 or args.fast_decay_factor > 1.0:
        raise SystemExit("--fast-decay-factor must be in the interval (0, 1]")
    if args.slow_decay_factor <= 0.0 or args.slow_decay_factor > 1.0:
        raise SystemExit("--slow-decay-factor must be in the interval (0, 1]")
    if args.min_log_odds >= args.max_log_odds:
        raise SystemExit("--min-log-odds must be smaller than --max-log-odds")
    if args.fast_occupancy_threshold < args.min_log_odds or args.fast_occupancy_threshold > args.max_log_odds:
        raise SystemExit("--fast-occupancy-threshold must lie within [--min-log-odds, --max-log-odds]")
    if args.slow_occupancy_threshold < args.min_log_odds or args.slow_occupancy_threshold > args.max_log_odds:
        raise SystemExit("--slow-occupancy-threshold must lie within [--min-log-odds, --max-log-odds]")
    if args.max_tf_age_sec < 0.0:
        raise SystemExit("--max-tf-age-sec must be non-negative")

    return Occupancy2DConfig(
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        fast_output_topic=args.fast_output_topic,
        dynamic_output_topic=args.dynamic_output_topic,
        debug_filtered_pointcloud_topic=args.debug_filtered_pointcloud_topic,
        map_frame=args.map_frame,
        base_frame=args.base_frame,
        lidar_frame=args.lidar_frame,
        min_z_m=float(args.min_z_m),
        max_z_m=float(args.max_z_m),
        resolution_m=float(args.resolution_m),
        width_m=float(args.width_m),
        height_m=float(args.height_m),
        max_tf_age_sec=float(args.max_tf_age_sec),
        debug=bool(args.debug),
        fast_hit_log_odds_increment=float(args.fast_hit_log_odds_increment),
        fast_miss_log_odds_decrement=float(args.fast_miss_log_odds_decrement),
        slow_hit_log_odds_increment=float(args.slow_hit_log_odds_increment),
        slow_miss_log_odds_decrement=float(args.slow_miss_log_odds_decrement),
        fast_occupancy_threshold=float(args.fast_occupancy_threshold),
        slow_occupancy_threshold=float(args.slow_occupancy_threshold),
        fast_decay_factor=float(args.fast_decay_factor),
        slow_decay_factor=float(args.slow_decay_factor),
        min_log_odds=float(args.min_log_odds),
        max_log_odds=float(args.max_log_odds),
    )


class Occupancy2DNode(Node):
    def __init__(self, config: Occupancy2DConfig) -> None:
        super().__init__("occupancy_2d")
        self.declare_parameter("input_topic", config.input_topic)
        self.declare_parameter("output_topic", config.output_topic)
        self.declare_parameter("fast_output_topic", config.fast_output_topic)
        self.declare_parameter("dynamic_output_topic", config.dynamic_output_topic)
        self.declare_parameter("debug_filtered_pointcloud_topic", config.debug_filtered_pointcloud_topic)
        self.declare_parameter("map_frame", config.map_frame)
        self.declare_parameter("base_frame", config.base_frame)
        self.declare_parameter("lidar_frame", config.lidar_frame)
        self.declare_parameter("min_z_m", config.min_z_m)
        self.declare_parameter("max_z_m", config.max_z_m)
        self.declare_parameter("resolution_m", config.resolution_m)
        self.declare_parameter("width_m", config.width_m)
        self.declare_parameter("height_m", config.height_m)
        self.declare_parameter("max_tf_age_sec", config.max_tf_age_sec)
        self.declare_parameter("debug", config.debug)
        self.declare_parameter("fast_hit_log_odds_increment", config.fast_hit_log_odds_increment)
        self.declare_parameter("fast_miss_log_odds_decrement", config.fast_miss_log_odds_decrement)
        self.declare_parameter("slow_hit_log_odds_increment", config.slow_hit_log_odds_increment)
        self.declare_parameter("slow_miss_log_odds_decrement", config.slow_miss_log_odds_decrement)
        self.declare_parameter("fast_occupancy_threshold", config.fast_occupancy_threshold)
        self.declare_parameter("slow_occupancy_threshold", config.slow_occupancy_threshold)
        self.declare_parameter("fast_decay_factor", config.fast_decay_factor)
        self.declare_parameter("slow_decay_factor", config.slow_decay_factor)
        self.declare_parameter("min_log_odds", config.min_log_odds)
        self.declare_parameter("max_log_odds", config.max_log_odds)
        self._config = Occupancy2DConfig(
            input_topic=str(self.get_parameter("input_topic").value),
            output_topic=str(self.get_parameter("output_topic").value),
            fast_output_topic=str(self.get_parameter("fast_output_topic").value),
            dynamic_output_topic=str(self.get_parameter("dynamic_output_topic").value),
            debug_filtered_pointcloud_topic=str(self.get_parameter("debug_filtered_pointcloud_topic").value),
            map_frame=str(self.get_parameter("map_frame").value),
            base_frame=str(self.get_parameter("base_frame").value),
            lidar_frame=str(self.get_parameter("lidar_frame").value),
            min_z_m=float(self.get_parameter("min_z_m").value),
            max_z_m=float(self.get_parameter("max_z_m").value),
            resolution_m=float(self.get_parameter("resolution_m").value),
            width_m=float(self.get_parameter("width_m").value),
            height_m=float(self.get_parameter("height_m").value),
            max_tf_age_sec=float(self.get_parameter("max_tf_age_sec").value),
            debug=bool(self.get_parameter("debug").value),
            fast_hit_log_odds_increment=float(self.get_parameter("fast_hit_log_odds_increment").value),
            fast_miss_log_odds_decrement=float(self.get_parameter("fast_miss_log_odds_decrement").value),
            slow_hit_log_odds_increment=float(self.get_parameter("slow_hit_log_odds_increment").value),
            slow_miss_log_odds_decrement=float(self.get_parameter("slow_miss_log_odds_decrement").value),
            fast_occupancy_threshold=float(self.get_parameter("fast_occupancy_threshold").value),
            slow_occupancy_threshold=float(self.get_parameter("slow_occupancy_threshold").value),
            fast_decay_factor=float(self.get_parameter("fast_decay_factor").value),
            slow_decay_factor=float(self.get_parameter("slow_decay_factor").value),
            min_log_odds=float(self.get_parameter("min_log_odds").value),
            max_log_odds=float(self.get_parameter("max_log_odds").value),
        )
        self._mapper = RollingOccupancyMap(
            width_m=self._config.width_m,
            height_m=self._config.height_m,
            resolution_m=self._config.resolution_m,
            fast_hit_log_odds_increment=self._config.fast_hit_log_odds_increment,
            fast_miss_log_odds_decrement=self._config.fast_miss_log_odds_decrement,
            slow_hit_log_odds_increment=self._config.slow_hit_log_odds_increment,
            slow_miss_log_odds_decrement=self._config.slow_miss_log_odds_decrement,
            fast_occupancy_threshold=self._config.fast_occupancy_threshold,
            slow_occupancy_threshold=self._config.slow_occupancy_threshold,
            fast_decay_factor=self._config.fast_decay_factor,
            slow_decay_factor=self._config.slow_decay_factor,
            min_log_odds=self._config.min_log_odds,
            max_log_odds=self._config.max_log_odds,
        )
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._publisher = self.create_publisher(OccupancyGrid, self._config.output_topic, OUTPUT_QOS)
        self._fast_publisher = self.create_publisher(OccupancyGrid, self._config.fast_output_topic, OUTPUT_QOS)
        self._dynamic_publisher = self.create_publisher(OccupancyGrid, self._config.dynamic_output_topic, OUTPUT_QOS)
        self._debug_filtered_pointcloud_publisher = (
            self.create_publisher(PointCloud2, self._config.debug_filtered_pointcloud_topic, 10)
            if self._config.debug
            else None
        )
        self._subscription = self.create_subscription(
            PointCloud2,
            self._config.input_topic,
            self._cloud_callback,
            10,
        )
        self._tf_warning_last_ns = 0
        self._stale_tf_warning_last_ns = 0
        self._cloud_transform_info_emitted = False
        self._missing_correction_warning_emitted = False

        if not NUMBA_AVAILABLE:
            self.get_logger().warning(
                "numba is not available in the ROS runtime. The occupancy node will run with slower Python kernels."
            )

        self.get_logger().info(
            "Building dual-timescale rolling occupancy grids from '%s' to slow '%s', fast '%s', and dynamic '%s' in frame '%s' with size %.1fm x %.1fm at %.2fm resolution "
            "and z filtering in [%.2f, %.2f] m using lidar frame '%s' and base frame '%s'. Maximum accepted latest-TF age is %.2f s. "
            "Debug filtered cloud publishing is %s on '%s'. "
            "Fast log-odds uses hit=%.3f, miss=%.3f, threshold=%.3f, decay=%.5f; slow log-odds uses hit=%.3f, miss=%.3f, threshold=%.3f, decay=%.5f; clamp=[%.2f, %.2f]."
            % (
                self._config.input_topic,
                self._config.output_topic,
                self._config.fast_output_topic,
                self._config.dynamic_output_topic,
                self._config.map_frame,
                self._config.width_m,
                self._config.height_m,
                self._config.resolution_m,
                self._config.min_z_m,
                self._config.max_z_m,
                self._config.lidar_frame,
                self._config.base_frame,
                self._config.max_tf_age_sec,
                self._config.debug,
                self._config.debug_filtered_pointcloud_topic,
                self._config.fast_hit_log_odds_increment,
                self._config.fast_miss_log_odds_decrement,
                self._config.fast_occupancy_threshold,
                self._config.fast_decay_factor,
                self._config.slow_hit_log_odds_increment,
                self._config.slow_miss_log_odds_decrement,
                self._config.slow_occupancy_threshold,
                self._config.slow_decay_factor,
                self._config.min_log_odds,
                self._config.max_log_odds,
            )
        )

    def _build_xyz_cloud_message(self, *, stamp: Time, points_xyz_m: np.ndarray) -> PointCloud2:
        cloud_msg = PointCloud2()
        cloud_msg.header.stamp = stamp.to_msg()
        cloud_msg.header.frame_id = self._config.map_frame
        cloud_msg.height = 1
        cloud_msg.width = int(points_xyz_m.shape[0])
        cloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 12
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = True
        xyz_float32 = np.ascontiguousarray(points_xyz_m, dtype=np.float32)
        cloud_msg.data = xyz_float32.tobytes()
        return cloud_msg

    def _lookup_transform(
        self,
        *,
        target_frame: str,
        source_frame: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        last_error: Exception | None = None
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as error:
            last_error = error
            now_ns = time_ns()
            if now_ns - self._tf_warning_last_ns >= 2_000_000_000:
                self._tf_warning_last_ns = now_ns
                self.get_logger().warning(
                    "Failed to resolve transform %s -> %s for occupancy mapping: %s"
                    % (target_frame, source_frame, last_error)
                )
            return None

        if not self._is_transform_recent_enough(transform):
            now_ns = time_ns()
            if now_ns - self._stale_tf_warning_last_ns >= 2_000_000_000:
                self._stale_tf_warning_last_ns = now_ns
                self.get_logger().warning(
                    "Latest transform %s -> %s is stale by %.3f s for occupancy mapping."
                    % (target_frame, source_frame, self._transform_age_sec(transform))
                )
            return None

        translation = np.array(
            [
                float(transform.transform.translation.x),
                float(transform.transform.translation.y),
                float(transform.transform.translation.z),
            ],
            dtype=np.float64,
        )
        rotation = transform.transform.rotation
        rotation_matrix = rotation_matrix_from_quaternion_xyzw(
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        )
        return translation, rotation_matrix

    def _transform_age_sec(self, transform) -> float:
        transform_stamp = Time.from_msg(transform.header.stamp)
        if transform_stamp.nanoseconds == 0:
            return 0.0
        now_ns = self.get_clock().now().nanoseconds
        return max(0.0, (now_ns - transform_stamp.nanoseconds) / 1e9)

    def _is_transform_recent_enough(self, transform) -> bool:
        return self._transform_age_sec(transform) <= self._config.max_tf_age_sec

    def _lookup_translation(
        self,
        *,
        target_frame: str,
        source_frame: str,
    ) -> np.ndarray | None:
        transform = self._lookup_transform(
            target_frame=target_frame,
            source_frame=source_frame,
        )
        if transform is None:
            return None
        translation, _ = transform
        return translation

    def _can_resolve_transform(
        self,
        *,
        target_frame: str,
        source_frame: str,
    ) -> bool:
        transform = self._lookup_transform(
            target_frame=target_frame,
            source_frame=source_frame,
        )
        return transform is not None

    def _requires_correction_transform(self) -> bool:
        return self._config.map_frame == DEFAULT_REQUIRED_CORRECTION_PARENT_FRAME

    def _cloud_callback(self, cloud_msg: PointCloud2) -> None:
        stamp = Time.from_msg(cloud_msg.header.stamp)
        cloud_frame = cloud_msg.header.frame_id or self._config.map_frame
        cloud_transform: tuple[np.ndarray, np.ndarray] | None = None

        if self._requires_correction_transform():
            if not self._can_resolve_transform(
                target_frame=self._config.map_frame,
                source_frame=DEFAULT_REQUIRED_CORRECTION_CHILD_FRAME,
            ):
                if not self._missing_correction_warning_emitted:
                    self._missing_correction_warning_emitted = True
                    self.get_logger().warning(
                        "Skipping occupancy updates until correction transform %s -> %s becomes available."
                        % (self._config.map_frame, DEFAULT_REQUIRED_CORRECTION_CHILD_FRAME)
                    )
                return
            self._missing_correction_warning_emitted = False
            if cloud_frame == DEFAULT_REQUIRED_CORRECTION_CHILD_FRAME:
                cloud_transform = self._lookup_transform(
                    target_frame=self._config.map_frame,
                    source_frame=cloud_frame,
                )
                if cloud_transform is None:
                    return

        if cloud_frame != self._config.map_frame:
            if not self._cloud_transform_info_emitted:
                self._cloud_transform_info_emitted = True
                self.get_logger().info(
                    "Transforming registered clouds from '%s' into occupancy map frame '%s'."
                    % (cloud_frame, self._config.map_frame)
                )
            if cloud_transform is None:
                cloud_transform = self._lookup_transform(
                    target_frame=self._config.map_frame,
                    source_frame=cloud_frame,
                )
                if cloud_transform is None:
                    return

        lidar_translation = self._lookup_translation(
            target_frame=self._config.map_frame,
            source_frame=self._config.lidar_frame,
        )
        base_translation = self._lookup_translation(
            target_frame=self._config.map_frame,
            source_frame=self._config.base_frame,
        )
        if lidar_translation is None or base_translation is None:
            return

        try:
            points_xyz = extract_xyz_points(cloud_msg)
        except ValueError as error:
            self.get_logger().warning(f"Skipping occupancy update because point cloud decoding failed: {error}")
            return

        if points_xyz.size > 0:
            filtered_points = np.ascontiguousarray(points_xyz, dtype=np.float64)
        else:
            filtered_points = np.empty((0, 3), dtype=np.float64)

        if filtered_points.size > 0 and cloud_transform is not None:
            cloud_translation, cloud_rotation = cloud_transform
            filtered_points = np.ascontiguousarray(filtered_points @ cloud_rotation.T + cloud_translation, dtype=np.float64)

        if filtered_points.size > 0:
            height_mask = (filtered_points[:, 2] >= self._config.min_z_m) & (filtered_points[:, 2] <= self._config.max_z_m)
            filtered_points = np.ascontiguousarray(filtered_points[height_mask], dtype=np.float64)

        if self._debug_filtered_pointcloud_publisher is not None:
            self._debug_filtered_pointcloud_publisher.publish(
                self._build_xyz_cloud_message(stamp=stamp, points_xyz_m=filtered_points)
            )

        self._mapper.integrate_point_cloud(
            points_xyz_m=filtered_points,
            ray_origin_xy_m=lidar_translation[:2],
            map_center_xy_m=base_translation[:2],
        )
        self._publisher.publish(self._build_message(stamp=stamp, snapshot=self._mapper.snapshot()))
        self._fast_publisher.publish(self._build_message(stamp=stamp, snapshot=self._mapper.fast_snapshot()))
        self._dynamic_publisher.publish(self._build_message(stamp=stamp, snapshot=self._mapper.dynamic_snapshot()))

    def _build_message(self, *, stamp: Time, snapshot: OccupancyMapSnapshot) -> OccupancyGrid:
        message = OccupancyGrid()
        message.header.stamp = stamp.to_msg()
        message.header.frame_id = self._config.map_frame
        message.info.map_load_time = stamp.to_msg()
        message.info.resolution = float(snapshot.resolution_m)
        message.info.width = int(snapshot.width)
        message.info.height = int(snapshot.height)
        message.info.origin = Pose()
        message.info.origin.position.x = float(snapshot.origin_x_m)
        message.info.origin.position.y = float(snapshot.origin_y_m)
        message.info.origin.position.z = 0.0
        message.info.origin.orientation.w = 1.0
        message.data = snapshot.data.ravel(order="C").astype(np.int8, copy=False).tolist()
        return message


def main() -> None:
    config = parse_args()
    rclpy.init(args=None)
    node = Occupancy2DNode(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()