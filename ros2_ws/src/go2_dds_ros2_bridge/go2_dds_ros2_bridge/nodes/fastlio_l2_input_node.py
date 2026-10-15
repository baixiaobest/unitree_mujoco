#!/usr/bin/env python3
"""Bridge the L2 cloud and IMU with one sensor-clock to ROS-clock mapping."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
import threading
from time import time_ns

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2, PointField

from go2_dds_ros2_bridge.dds_runtime import add_dds_runtime_arguments, resolve_runtime_arguments
from go2_dds_ros2_bridge.nodes.time_sync_diagnostic_node import import_raw_dds_dependencies
from go2_dds_ros2_bridge.nodes.clock_offset_bridge_node import (
    FRONT_LEG_FILTER_BOXES_BASE_LINK,
    dtype_from_fields,
)
from go2_dds_ros2_bridge.tf_utils import (
    DEFAULT_LIDAR_TF_RPY_DEG,
    DEFAULT_LIDAR_TF_XYZ,
    transform_matrix_from_xyz_rpy_degrees,
)


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ros_stamp(value: int) -> Time:
    result = Time()
    result.sec, result.nanosec = divmod(value, 1_000_000_000)
    return result


@dataclass(frozen=True)
class Config:
    domain: int
    interface: str
    cloud_dds: str
    imu_dds: str
    cloud_ros: str
    imu_ros: str
    imu_frame: str
    lidar_frame: str
    preserve_sensor_stamps: bool


class PairedL2Bridge(Node):
    def __init__(self, config: Config, subscriber_class, cloud_class, imu_class):
        super().__init__("fastlio_l2_input")
        self._config = config
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=20)
        self._cloud_pub = self.create_publisher(PointCloud2, config.cloud_ros, qos)
        self._imu_pub = self.create_publisher(Imu, config.imu_ros, qos)
        self._lock = threading.Lock()
        self._callback_lock = threading.Lock()
        self._offset_samples: deque[tuple[int, int]] = deque()
        self._offset_ns: int | None = None
        self._last_cloud_ns = 0
        self._last_imu_ns = 0
        self._cloud_count = 0
        self._imu_count = 0
        self._imu_sub = subscriber_class(config.imu_dds, imu_class)
        self._imu_sub.Init(self._on_imu, 100)
        self._cloud_sub = subscriber_class(config.cloud_dds, cloud_class)
        self._cloud_sub.Init(self._on_cloud, 10)
        self.get_logger().info(f"L2 DDS {config.cloud_dds}, {config.imu_dds} -> {config.cloud_ros}, {config.imu_ros}")

    def _on_imu(self, raw):
        # DDS may invoke callbacks concurrently. Keep stamp checks and ROS publication
        # in one order so FAST-LIO never receives an older sample after a newer one.
        with self._callback_lock:
            self._process_imu(raw)

    def _process_imu(self, raw):
        source_ns = stamp_ns(raw.header.stamp)
        values = (raw.angular_velocity.x, raw.angular_velocity.y, raw.angular_velocity.z,
                  raw.linear_acceleration.x, raw.linear_acceleration.y, raw.linear_acceleration.z)
        if source_ns <= 0 or not all(math.isfinite(float(value)) for value in values):
            return
        receipt_ns = time_ns()
        with self._lock:
            if self._config.preserve_sensor_stamps:
                corrected_ns = source_ns
            else:
                self._offset_samples.append((receipt_ns, receipt_ns - source_ns))
                cutoff = receipt_ns - 5_000_000_000
                while self._offset_samples and self._offset_samples[0][0] < cutoff:
                    self._offset_samples.popleft()
                offsets = sorted(offset for _, offset in self._offset_samples)
                self._offset_ns = offsets[int(0.05 * (len(offsets) - 1))]
                corrected_ns = source_ns + self._offset_ns
            if corrected_ns <= self._last_imu_ns:
                return
            self._last_imu_ns = corrected_ns
        msg = Imu()
        msg.header.stamp = ros_stamp(corrected_ns)
        msg.header.frame_id = self._config.imu_frame
        msg.orientation.x = float(raw.orientation.x)
        msg.orientation.y = float(raw.orientation.y)
        msg.orientation.z = float(raw.orientation.z)
        msg.orientation.w = float(raw.orientation.w)
        msg.orientation_covariance = list(raw.orientation_covariance)
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = map(float, values[:3])
        msg.angular_velocity_covariance = list(raw.angular_velocity_covariance)
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = map(float, values[3:])
        msg.linear_acceleration_covariance = list(raw.linear_acceleration_covariance)
        self._imu_pub.publish(msg)
        self._imu_count += 1

    def _on_cloud(self, raw):
        with self._callback_lock:
            self._process_cloud(raw)

    def _process_cloud(self, raw):
        source_ns = stamp_ns(raw.header.stamp)
        with self._lock:
            if source_ns <= 0 or (self._offset_ns is None and not self._config.preserve_sensor_stamps):
                return
            corrected_ns = source_ns if self._config.preserve_sensor_stamps else source_ns + self._offset_ns
            if corrected_ns <= self._last_cloud_ns:
                return
            self._last_cloud_ns = corrected_ns
        msg = PointCloud2()
        msg.header.stamp = ros_stamp(corrected_ns)
        msg.header.frame_id = self._config.lidar_frame
        msg.height = int(raw.height)
        msg.width = int(raw.width)
        msg.fields = [PointField(name=str(field.name), offset=int(field.offset),
                                 datatype=int(field.datatype), count=int(field.count)) for field in raw.fields]
        msg.is_bigendian = bool(raw.is_bigendian)
        msg.point_step = int(raw.point_step)
        msg.row_step = int(raw.row_step)
        msg.data = bytes(raw.data)
        msg.is_dense = bool(raw.is_dense)
        try:
            self._mask_front_legs(msg)
        except ValueError as error:
            self.get_logger().warning(f"Front-leg cloud filter skipped: {error}")
        self._cloud_pub.publish(msg)
        self._cloud_count += 1

    @staticmethod
    def _mask_front_legs(msg: PointCloud2):
        dtype = dtype_from_fields(msg.fields, msg.point_step, msg.is_bigendian)
        if msg.row_step < msg.width * msg.point_step or len(msg.data) < msg.row_step * msg.height:
            raise ValueError("invalid PointCloud2 row layout")
        cloud = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=bytes(msg.data),
                           strides=(msg.row_step, msg.point_step)).reshape(-1).copy()
        if not {"x", "y", "z"}.issubset(cloud.dtype.names or ()):
            raise ValueError("cloud has no x/y/z fields")
        points = np.column_stack([cloud[name] for name in ("x", "y", "z")]).astype(np.float64)
        finite = np.isfinite(points).all(axis=1)
        transform = transform_matrix_from_xyz_rpy_degrees(
            *DEFAULT_LIDAR_TF_XYZ, *DEFAULT_LIDAR_TF_RPY_DEG)
        base_points = points[finite] @ transform[:3, :3].T + transform[:3, 3]
        remove = np.zeros(len(base_points), dtype=bool)
        for x_range, y_range, z_range in FRONT_LEG_FILTER_BOXES_BASE_LINK:
            remove |= ((base_points[:, 0] >= x_range[0]) & (base_points[:, 0] <= x_range[1])
                       & (base_points[:, 1] >= y_range[0]) & (base_points[:, 1] <= y_range[1])
                       & (base_points[:, 2] >= z_range[0]) & (base_points[:, 2] <= z_range[1]))
        keep = np.ones(len(cloud), dtype=bool)
        keep[np.flatnonzero(finite)[remove]] = False
        if not keep.all() or msg.row_step != msg.width * msg.point_step or msg.height != 1:
            cloud = cloud[keep]
            msg.height = 1
            msg.width = len(cloud)
            msg.row_step = msg.width * msg.point_step
            msg.data = cloud.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dds_runtime_arguments(parser)
    parser.add_argument("--cloud-dds", default="rt/utlidar/cloud")
    parser.add_argument("--imu-dds", default="rt/utlidar/imu")
    parser.add_argument("--cloud-ros", default="/fastlio_l2/input/cloud")
    parser.add_argument("--imu-ros", default="/fastlio_l2/input/imu")
    parser.add_argument("--imu-frame", default="fastlio_l2_imu")
    parser.add_argument("--lidar-frame", default="fastlio_l2_lidar")
    parser.add_argument("--preserve-sensor-stamps", action="store_true",
                        help="Publish raw shared L2 clock stamps for LiDAR-IMU calibration")
    import sys
    args = parser.parse_args(rclpy.utilities.remove_ros_args(args=sys.argv)[1:])
    runtime = resolve_runtime_arguments(args)
    init_dds, subscriber_class, cloud_class, _, imu_class = import_raw_dds_dependencies()
    init_dds(runtime.domain_id, runtime.interface)
    rclpy.init()
    node = PairedL2Bridge(Config(runtime.domain_id, runtime.interface, args.cloud_dds,
                                 args.imu_dds, args.cloud_ros, args.imu_ros,
                                 args.imu_frame, args.lidar_frame, args.preserve_sensor_stamps),
                          subscriber_class, cloud_class, imu_class)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
