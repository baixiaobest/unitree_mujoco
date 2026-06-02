#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from time import time_ns

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from go2_dds_ros2_bridge.tf_utils import rotation_matrix_from_quaternion_xyzw

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


DEFAULT_INPUT_TOPIC = "/fast_occupancy"
DEFAULT_OUTPUT_TOPIC = "/occupancy_scan"
DEFAULT_DEBUG_COSTMAP_TOPIC = "/occupancy_scan/debug_costmap"
DEFAULT_MAP_FRAME = "camera_init_correct"
DEFAULT_BASE_FRAME = "base_link"
DEFAULT_NUM_RAYS = 128
DEFAULT_FOV_DEG = 180.0
DEFAULT_MAX_RANGE_M = 20.0
DEFAULT_OCCUPANCY_THRESHOLD = 50
DEFAULT_MAX_TF_AGE_SEC = 1.0

SCAN_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


@njit(cache=True)
def cast_rays(
    grid_flat: np.ndarray,
    grid_width: int,
    grid_height: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
    robot_x: float,
    robot_y: float,
    cos_angles: np.ndarray,
    sin_angles: np.ndarray,
    max_range: float,
    occ_threshold: int,
) -> np.ndarray:
    num_rays = cos_angles.shape[0]
    distances = np.empty(num_rays, dtype=np.float32)
    step = resolution * 0.5

    for i in range(num_rays):
        dx = cos_angles[i]
        dy = sin_angles[i]
        dist = 0.0
        hit = False
        while dist < max_range:
            px = robot_x + dist * dx
            py = robot_y + dist * dy
            col = int(math.floor((px - origin_x) / resolution))
            row = int(math.floor((py - origin_y) / resolution))
            if col < 0 or col >= grid_width or row < 0 or row >= grid_height:
                dist = max_range
                hit = True
                break
            if grid_flat[row * grid_width + col] > occ_threshold:
                hit = True
                break
            dist += step
        if not hit:
            dist = max_range
        distances[i] = dist

    return distances


@dataclass(frozen=True)
class OccupancyScanConfig:
    input_topic: str
    output_topic: str
    debug_costmap_topic: str
    map_frame: str
    base_frame: str
    num_rays: int
    fov_deg: float
    max_range_m: float
    occupancy_threshold: int
    max_tf_age_sec: float


def parse_args() -> OccupancyScanConfig:
    parser = argparse.ArgumentParser(
        description="Cast rays through a 2D occupancy grid and publish a LaserScan."
    )
    parser.add_argument("--input-topic", type=str, default=DEFAULT_INPUT_TOPIC)
    parser.add_argument("--output-topic", type=str, default=DEFAULT_OUTPUT_TOPIC)
    parser.add_argument("--debug-costmap-topic", type=str, default=DEFAULT_DEBUG_COSTMAP_TOPIC)
    parser.add_argument("--map-frame", type=str, default=DEFAULT_MAP_FRAME)
    parser.add_argument("--base-frame", type=str, default=DEFAULT_BASE_FRAME)
    parser.add_argument("--num-rays", type=int, default=DEFAULT_NUM_RAYS)
    parser.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG)
    parser.add_argument("--max-range-m", type=float, default=DEFAULT_MAX_RANGE_M)
    parser.add_argument("--occupancy-threshold", type=int, default=DEFAULT_OCCUPANCY_THRESHOLD)
    parser.add_argument("--max-tf-age-sec", type=float, default=DEFAULT_MAX_TF_AGE_SEC)

    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)
    if args.num_rays < 2:
        raise SystemExit("--num-rays must be at least 2")
    if args.fov_deg <= 0.0 or args.fov_deg > 360.0:
        raise SystemExit("--fov-deg must be in (0, 360]")
    if args.max_range_m <= 0.0:
        raise SystemExit("--max-range-m must be positive")
    if args.occupancy_threshold < 0 or args.occupancy_threshold > 100:
        raise SystemExit("--occupancy-threshold must be in [0, 100]")
    if args.max_tf_age_sec < 0.0:
        raise SystemExit("--max-tf-age-sec must be non-negative")

    return OccupancyScanConfig(
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        debug_costmap_topic=args.debug_costmap_topic,
        map_frame=args.map_frame,
        base_frame=args.base_frame,
        num_rays=int(args.num_rays),
        fov_deg=float(args.fov_deg),
        max_range_m=float(args.max_range_m),
        occupancy_threshold=int(args.occupancy_threshold),
        max_tf_age_sec=float(args.max_tf_age_sec),
    )


class OccupancyScanNode(Node):
    def __init__(self, config: OccupancyScanConfig) -> None:
        super().__init__("occupancy_scan")
        self.declare_parameter("input_topic", config.input_topic)
        self.declare_parameter("output_topic", config.output_topic)
        self.declare_parameter("debug_costmap_topic", config.debug_costmap_topic)
        self.declare_parameter("map_frame", config.map_frame)
        self.declare_parameter("base_frame", config.base_frame)
        self.declare_parameter("num_rays", config.num_rays)
        self.declare_parameter("fov_deg", config.fov_deg)
        self.declare_parameter("max_range_m", config.max_range_m)
        self.declare_parameter("occupancy_threshold", config.occupancy_threshold)
        self.declare_parameter("max_tf_age_sec", config.max_tf_age_sec)
        self._config = OccupancyScanConfig(
            input_topic=str(self.get_parameter("input_topic").value),
            output_topic=str(self.get_parameter("output_topic").value),
            debug_costmap_topic=str(self.get_parameter("debug_costmap_topic").value),
            map_frame=str(self.get_parameter("map_frame").value),
            base_frame=str(self.get_parameter("base_frame").value),
            num_rays=int(self.get_parameter("num_rays").value),
            fov_deg=float(self.get_parameter("fov_deg").value),
            max_range_m=float(self.get_parameter("max_range_m").value),
            occupancy_threshold=int(self.get_parameter("occupancy_threshold").value),
            max_tf_age_sec=float(self.get_parameter("max_tf_age_sec").value),
        )
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._scan_publisher = self.create_publisher(LaserScan, self._config.output_topic, SCAN_QOS)
        self._debug_publisher = self.create_publisher(
            OccupancyGrid, self._config.debug_costmap_topic, 10
        )
        self._subscription = self.create_subscription(
            OccupancyGrid,
            self._config.input_topic,
            self._grid_callback,
            10,
        )
        self._tf_warning_last_ns: int = 0
        self._stale_tf_warning_last_ns: int = 0

        if not NUMBA_AVAILABLE:
            self.get_logger().warning(
                "numba is not available. The occupancy scan node will run with slower Python kernels."
            )

        self.get_logger().info(
            "Occupancy scan node: subscribing to '%s', publishing LaserScan to '%s' and debug costmap to '%s'. "
            "Map frame: '%s', base frame: '%s'. %d rays over %.1f° FOV, max range %.1f m, "
            "occupancy threshold %d, max TF age %.2f s."
            % (
                self._config.input_topic,
                self._config.output_topic,
                self._config.debug_costmap_topic,
                self._config.map_frame,
                self._config.base_frame,
                self._config.num_rays,
                self._config.fov_deg,
                self._config.max_range_m,
                self._config.occupancy_threshold,
                self._config.max_tf_age_sec,
            )
        )

    def _transform_age_sec(self, transform) -> float:
        transform_stamp = Time.from_msg(transform.header.stamp)
        if transform_stamp.nanoseconds == 0:
            return 0.0
        now_ns = self.get_clock().now().nanoseconds
        return max(0.0, (now_ns - transform_stamp.nanoseconds) / 1e9)

    def _is_transform_recent_enough(self, transform) -> bool:
        return self._transform_age_sec(transform) <= self._config.max_tf_age_sec

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
                    "Failed to resolve transform %s -> %s for occupancy scan: %s"
                    % (target_frame, source_frame, last_error)
                )
            return None

        if not self._is_transform_recent_enough(transform):
            now_ns = time_ns()
            if now_ns - self._stale_tf_warning_last_ns >= 2_000_000_000:
                self._stale_tf_warning_last_ns = now_ns
                self.get_logger().warning(
                    "Latest transform %s -> %s is stale by %.3f s for occupancy scan."
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

    def _grid_callback(self, grid_msg: OccupancyGrid) -> None:
        stamp = Time.from_msg(grid_msg.header.stamp)

        result = self._lookup_transform(
            target_frame=self._config.map_frame,
            source_frame=self._config.base_frame,
        )
        if result is None:
            return
        translation, rotation_matrix = result
        robot_x = translation[0]
        robot_y = translation[1]
        yaw = math.atan2(float(rotation_matrix[1, 0]), float(rotation_matrix[0, 0]))

        fov_rad = math.radians(self._config.fov_deg)
        angle_offsets = np.linspace(-fov_rad / 2.0, fov_rad / 2.0, self._config.num_rays, dtype=np.float64)
        map_angles = yaw + angle_offsets
        cos_angles = np.cos(map_angles)
        sin_angles = np.sin(map_angles)

        grid_data = np.array(grid_msg.data, dtype=np.int8)
        width = int(grid_msg.info.width)
        height = int(grid_msg.info.height)
        ox = float(grid_msg.info.origin.position.x)
        oy = float(grid_msg.info.origin.position.y)
        res = float(grid_msg.info.resolution)

        distances = cast_rays(
            grid_data,
            width,
            height,
            ox,
            oy,
            res,
            robot_x,
            robot_y,
            cos_angles,
            sin_angles,
            float(self._config.max_range_m),
            int(self._config.occupancy_threshold),
        )

        end_xs = robot_x + distances * cos_angles
        end_ys = robot_y + distances * sin_angles
        hit_cols = np.floor((end_xs - ox) / res).astype(np.int32)
        hit_rows = np.floor((end_ys - oy) / res).astype(np.int32)

        self._scan_publisher.publish(self._build_scan_message(stamp, distances, fov_rad))
        self._debug_publisher.publish(self._build_debug_costmap(grid_msg, width, height, hit_cols, hit_rows))

    def _build_scan_message(self, stamp: Time, distances: np.ndarray, fov_rad: float) -> LaserScan:
        msg = LaserScan()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self._config.base_frame
        msg.angle_min = float(-fov_rad / 2.0)
        msg.angle_max = float(fov_rad / 2.0)
        msg.angle_increment = float(fov_rad / (self._config.num_rays - 1))
        msg.time_increment = 0.0
        msg.scan_time = 0.0
        msg.range_min = 0.0
        msg.range_max = float(self._config.max_range_m)
        msg.ranges = distances.tolist()
        msg.intensities = []
        return msg

    def _build_debug_costmap(
        self,
        grid_msg: OccupancyGrid,
        width: int,
        height: int,
        hit_cols: np.ndarray,
        hit_rows: np.ndarray,
    ) -> OccupancyGrid:
        out_data = np.zeros(height * width, dtype=np.int8)
        for k in range(hit_cols.shape[0]):
            c = int(hit_cols[k])
            r = int(hit_rows[k])
            if 0 <= c < width and 0 <= r < height:
                out_data[r * width + c] = 100

        msg = OccupancyGrid()
        msg.header.stamp = grid_msg.header.stamp
        msg.header.frame_id = grid_msg.header.frame_id
        msg.info = grid_msg.info
        msg.data = out_data.tolist()
        return msg


def main() -> None:
    config = parse_args()
    rclpy.init(args=None)
    node = OccupancyScanNode(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
