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
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from go2_dds_ros2_bridge.occupancy_map import (
    NUMBA_AVAILABLE,
    RollingOccupancyMap,
    extract_xyz_points,
)


DEFAULT_INPUT_TOPIC = "/cloud_registered"
DEFAULT_OUTPUT_TOPIC = "/static_occupancy"
DEFAULT_MAP_FRAME = "camera_init"
DEFAULT_BASE_FRAME = "base_link"
DEFAULT_LIDAR_FRAME = "utlidar_lidar"
DEFAULT_MIN_Z_M = 0.1
DEFAULT_MAX_Z_M = 1.5
DEFAULT_RESOLUTION_M = 0.1
DEFAULT_WIDTH_M = 15.0
DEFAULT_HEIGHT_M = 15.0
DEFAULT_HIT_LOG_ODDS_INCREMENT = 0.85
DEFAULT_MISS_LOG_ODDS_DECREMENT = 0.4
DEFAULT_DECAY_FACTOR = 0.999
DEFAULT_MIN_LOG_ODDS = -4.0
DEFAULT_MAX_LOG_ODDS = 4.0
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
    map_frame: str
    base_frame: str
    lidar_frame: str
    min_z_m: float
    max_z_m: float
    resolution_m: float
    width_m: float
    height_m: float
    hit_log_odds_increment: float
    miss_log_odds_decrement: float
    decay_factor: float
    min_log_odds: float
    max_log_odds: float


def parse_args() -> Occupancy2DConfig:
    parser = argparse.ArgumentParser(
        description="Build a rolling 2D occupancy grid from FAST-LIO registered point clouds."
    )
    parser.add_argument("--input-topic", type=str, default=DEFAULT_INPUT_TOPIC)
    parser.add_argument("--output-topic", type=str, default=DEFAULT_OUTPUT_TOPIC)
    parser.add_argument("--map-frame", type=str, default=DEFAULT_MAP_FRAME)
    parser.add_argument("--base-frame", type=str, default=DEFAULT_BASE_FRAME)
    parser.add_argument("--lidar-frame", type=str, default=DEFAULT_LIDAR_FRAME)
    parser.add_argument("--min-z-m", type=float, default=DEFAULT_MIN_Z_M)
    parser.add_argument("--max-z-m", type=float, default=DEFAULT_MAX_Z_M)
    parser.add_argument("--resolution-m", type=float, default=DEFAULT_RESOLUTION_M)
    parser.add_argument("--width-m", type=float, default=DEFAULT_WIDTH_M)
    parser.add_argument("--height-m", type=float, default=DEFAULT_HEIGHT_M)
    parser.add_argument("--hit-log-odds-increment", type=float, default=DEFAULT_HIT_LOG_ODDS_INCREMENT)
    parser.add_argument("--miss-log-odds-decrement", type=float, default=DEFAULT_MISS_LOG_ODDS_DECREMENT)
    parser.add_argument("--decay-factor", type=float, default=DEFAULT_DECAY_FACTOR)
    parser.add_argument("--min-log-odds", type=float, default=DEFAULT_MIN_LOG_ODDS)
    parser.add_argument("--max-log-odds", type=float, default=DEFAULT_MAX_LOG_ODDS)

    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)
    if args.min_z_m >= args.max_z_m:
        raise SystemExit("--min-z-m must be smaller than --max-z-m")
    if args.hit_log_odds_increment <= 0.0:
        raise SystemExit("--hit-log-odds-increment must be positive")
    if args.miss_log_odds_decrement <= 0.0:
        raise SystemExit("--miss-log-odds-decrement must be positive")
    if args.decay_factor <= 0.0 or args.decay_factor > 1.0:
        raise SystemExit("--decay-factor must be in the interval (0, 1]")
    if args.min_log_odds >= args.max_log_odds:
        raise SystemExit("--min-log-odds must be smaller than --max-log-odds")

    return Occupancy2DConfig(
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        map_frame=args.map_frame,
        base_frame=args.base_frame,
        lidar_frame=args.lidar_frame,
        min_z_m=float(args.min_z_m),
        max_z_m=float(args.max_z_m),
        resolution_m=float(args.resolution_m),
        width_m=float(args.width_m),
        height_m=float(args.height_m),
        hit_log_odds_increment=float(args.hit_log_odds_increment),
        miss_log_odds_decrement=float(args.miss_log_odds_decrement),
        decay_factor=float(args.decay_factor),
        min_log_odds=float(args.min_log_odds),
        max_log_odds=float(args.max_log_odds),
    )


class Occupancy2DNode(Node):
    def __init__(self, config: Occupancy2DConfig) -> None:
        super().__init__("occupancy_2d")
        self.declare_parameter("input_topic", config.input_topic)
        self.declare_parameter("output_topic", config.output_topic)
        self.declare_parameter("map_frame", config.map_frame)
        self.declare_parameter("base_frame", config.base_frame)
        self.declare_parameter("lidar_frame", config.lidar_frame)
        self.declare_parameter("min_z_m", config.min_z_m)
        self.declare_parameter("max_z_m", config.max_z_m)
        self.declare_parameter("resolution_m", config.resolution_m)
        self.declare_parameter("width_m", config.width_m)
        self.declare_parameter("height_m", config.height_m)
        self.declare_parameter("hit_log_odds_increment", config.hit_log_odds_increment)
        self.declare_parameter("miss_log_odds_decrement", config.miss_log_odds_decrement)
        self.declare_parameter("decay_factor", config.decay_factor)
        self.declare_parameter("min_log_odds", config.min_log_odds)
        self.declare_parameter("max_log_odds", config.max_log_odds)
        self._config = Occupancy2DConfig(
            input_topic=str(self.get_parameter("input_topic").value),
            output_topic=str(self.get_parameter("output_topic").value),
            map_frame=str(self.get_parameter("map_frame").value),
            base_frame=str(self.get_parameter("base_frame").value),
            lidar_frame=str(self.get_parameter("lidar_frame").value),
            min_z_m=float(self.get_parameter("min_z_m").value),
            max_z_m=float(self.get_parameter("max_z_m").value),
            resolution_m=float(self.get_parameter("resolution_m").value),
            width_m=float(self.get_parameter("width_m").value),
            height_m=float(self.get_parameter("height_m").value),
            hit_log_odds_increment=float(self.get_parameter("hit_log_odds_increment").value),
            miss_log_odds_decrement=float(self.get_parameter("miss_log_odds_decrement").value),
            decay_factor=float(self.get_parameter("decay_factor").value),
            min_log_odds=float(self.get_parameter("min_log_odds").value),
            max_log_odds=float(self.get_parameter("max_log_odds").value),
        )
        self._mapper = RollingOccupancyMap(
            width_m=self._config.width_m,
            height_m=self._config.height_m,
            resolution_m=self._config.resolution_m,
            hit_log_odds_increment=self._config.hit_log_odds_increment,
            miss_log_odds_decrement=self._config.miss_log_odds_decrement,
            decay_factor=self._config.decay_factor,
            min_log_odds=self._config.min_log_odds,
            max_log_odds=self._config.max_log_odds,
        )
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._publisher = self.create_publisher(OccupancyGrid, self._config.output_topic, OUTPUT_QOS)
        self._subscription = self.create_subscription(
            PointCloud2,
            self._config.input_topic,
            self._cloud_callback,
            10,
        )
        self._tf_warning_last_ns = 0
        self._frame_warning_emitted = False

        if not NUMBA_AVAILABLE:
            self.get_logger().warning(
                "numba is not available in the ROS runtime. The occupancy node will run with slower Python kernels."
            )

        self.get_logger().info(
            "Building a rolling 2D occupancy grid from '%s' to '%s' in frame '%s' with size %.1fm x %.1fm at %.2fm resolution "
            "and z filtering in [%.2f, %.2f] m using lidar frame '%s' and base frame '%s'. "
            "Log-odds fusion uses hit=%.3f, miss=%.3f, decay=%.5f, clamp=[%.2f, %.2f]."
            % (
                self._config.input_topic,
                self._config.output_topic,
                self._config.map_frame,
                self._config.width_m,
                self._config.height_m,
                self._config.resolution_m,
                self._config.min_z_m,
                self._config.max_z_m,
                self._config.lidar_frame,
                self._config.base_frame,
                self._config.hit_log_odds_increment,
                self._config.miss_log_odds_decrement,
                self._config.decay_factor,
                self._config.min_log_odds,
                self._config.max_log_odds,
            )
        )

    def _lookup_translation(self, *, target_frame: str, source_frame: str, stamp: Time) -> np.ndarray | None:
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=0.05),
            )
        except TransformException as error:
            now_ns = time_ns()
            if now_ns - self._tf_warning_last_ns >= 2_000_000_000:
                self._tf_warning_last_ns = now_ns
                self.get_logger().warning(
                    "Failed to resolve transform %s -> %s for occupancy mapping: %s"
                    % (target_frame, source_frame, error)
                )
            return None

        return np.array(
            [
                float(transform.transform.translation.x),
                float(transform.transform.translation.y),
                float(transform.transform.translation.z),
            ],
            dtype=np.float64,
        )

    def _cloud_callback(self, cloud_msg: PointCloud2) -> None:
        if not self._frame_warning_emitted and cloud_msg.header.frame_id != self._config.map_frame:
            self._frame_warning_emitted = True
            self.get_logger().warning(
                "Incoming registered cloud frame_id '%s' does not match configured map frame '%s'."
                % (cloud_msg.header.frame_id, self._config.map_frame)
            )

        stamp = Time.from_msg(cloud_msg.header.stamp)
        lidar_translation = self._lookup_translation(
            target_frame=self._config.map_frame,
            source_frame=self._config.lidar_frame,
            stamp=stamp,
        )
        base_translation = self._lookup_translation(
            target_frame=self._config.map_frame,
            source_frame=self._config.base_frame,
            stamp=stamp,
        )
        if lidar_translation is None or base_translation is None:
            return

        try:
            points_xyz = extract_xyz_points(cloud_msg)
        except ValueError as error:
            self.get_logger().warning(f"Skipping occupancy update because point cloud decoding failed: {error}")
            return

        if points_xyz.size > 0:
            height_mask = (points_xyz[:, 2] >= self._config.min_z_m) & (points_xyz[:, 2] <= self._config.max_z_m)
            filtered_points = np.ascontiguousarray(points_xyz[height_mask], dtype=np.float64)
        else:
            filtered_points = np.empty((0, 3), dtype=np.float64)

        self._mapper.integrate_point_cloud(
            points_xyz_m=filtered_points,
            ray_origin_xy_m=lidar_translation[:2],
            map_center_xy_m=base_translation[:2],
        )
        self._publisher.publish(self._build_message(stamp))

    def _build_message(self, stamp: Time) -> OccupancyGrid:
        snapshot = self._mapper.snapshot()
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