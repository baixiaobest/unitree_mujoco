#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster

from go2_dds_ros2_bridge.tf_utils import (
    DEFAULT_LIDAR_TF_RPY_DEG,
    quaternion_from_rotation_matrix,
    rotation_matrix_from_rpy_degrees,
    rotation_vector_from_matrix,
)


DEFAULT_INPUT_TOPIC = "/utlidar/time_corrected/cloud"
DEFAULT_ODOM_TOPIC = "/kiss/odom"
DEFAULT_ODOM_FRAME = "kiss_odom"
DEFAULT_CHILD_FRAME = "kiss_lidar"

TIMESTAMP_FIELD_NAMES = ("t", "timestamp", "timestamps", "time", "time_stamp")
POINT_FIELD_DTYPES = {
    PointField.INT8: np.dtype(np.int8),
    PointField.UINT8: np.dtype(np.uint8),
    PointField.INT16: np.dtype(np.int16),
    PointField.UINT16: np.dtype(np.uint16),
    PointField.INT32: np.dtype(np.int32),
    PointField.UINT32: np.dtype(np.uint32),
    PointField.FLOAT32: np.dtype(np.float32),
    PointField.FLOAT64: np.dtype(np.float64),
}
POINT_FIELD_TYPE_NAMES = {
    PointField.INT8: "int8",
    PointField.UINT8: "uint8",
    PointField.INT16: "int16",
    PointField.UINT16: "uint16",
    PointField.INT32: "int32",
    PointField.UINT32: "uint32",
    PointField.FLOAT32: "float32",
    PointField.FLOAT64: "float64",
}
TIMESTAMP_DEBUG_LOG_LIMIT = 3


@dataclass(frozen=True)
class BridgeConfig:
    input_topic: str
    odom_topic: str
    odom_frame: str
    child_frame: str
    config_file: Path | None
    accumulate_frames: int
    reset_after_registrations: int
    publish_tf: bool
    deskew: bool
    position_covariance: float
    orientation_covariance: float
    twist_covariance: float


def import_kiss_icp_dependencies():
    try:
        from kiss_icp.config import load_config
        from kiss_icp.kiss_icp import KissICP
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("kiss_icp"):
            raise SystemExit(
                "The KISS-ICP bridge cannot import 'kiss_icp'.\n"
                "Install kiss-icp into the same Python interpreter that runs ROS2.\n"
                "For ROS Humble on this machine that is typically /usr/bin/python3 (Python 3.10)."
            ) from error
        raise

    return load_config, KissICP


def parse_args() -> BridgeConfig:
    parser = argparse.ArgumentParser(
        description="Run KISS-ICP on the time-corrected Unitree LiDAR cloud and publish ROS2 odometry."
    )
    parser.add_argument(
        "--input-topic",
        type=str,
        default=DEFAULT_INPUT_TOPIC,
        help="PointCloud2 topic carrying the time-corrected LiDAR cloud.",
    )
    parser.add_argument(
        "--odom-topic",
        type=str,
        default=DEFAULT_ODOM_TOPIC,
        help="Odometry topic published from KISS-ICP.",
    )
    parser.add_argument(
        "--odom-frame",
        type=str,
        default=DEFAULT_ODOM_FRAME,
        help="Frame id used for the published KISS odometry frame.",
    )
    parser.add_argument(
        "--child-frame",
        type=str,
        default=DEFAULT_CHILD_FRAME,
        help="Child frame id used for the published KISS odometry and TF.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Optional KISS-ICP YAML config file.",
    )
    parser.add_argument(
        "--accumulate-frames",
        type=int,
        default=10,
        help="Number of incoming PointCloud2 messages to accumulate before registering one KISS frame.",
    )
    parser.add_argument(
        "--reset-after-registrations",
        type=int,
        default=0,
        help="Reinitialize KISS-ICP after this many registered frames. Use 0 to disable periodic resets.",
    )
    parser.add_argument(
        "--publish-tf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish a TF from odom-frame to child-frame.",
    )
    parser.add_argument(
        "--deskew",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable KISS-ICP scan deskewing when the point cloud contains a timestamp field.",
    )
    parser.add_argument(
        "--position-covariance",
        type=float,
        default=0.1,
        help="Diagonal covariance value used for published odometry position.",
    )
    parser.add_argument(
        "--orientation-covariance",
        type=float,
        default=0.1,
        help="Diagonal covariance value used for published odometry orientation.",
    )
    parser.add_argument(
        "--twist-covariance",
        type=float,
        default=10.0,
        help="Diagonal covariance value used for published odometry twist.",
    )

    args = parser.parse_args()
    return BridgeConfig(
        input_topic=args.input_topic,
        odom_topic=args.odom_topic,
        odom_frame=args.odom_frame,
        child_frame=args.child_frame,
        config_file=args.config_file,
        accumulate_frames=max(args.accumulate_frames, 1),
        reset_after_registrations=max(args.reset_after_registrations, 0),
        publish_tf=args.publish_tf,
        deskew=args.deskew,
        position_covariance=max(args.position_covariance, 0.0),
        orientation_covariance=max(args.orientation_covariance, 0.0),
        twist_covariance=max(args.twist_covariance, 0.0),
    )


def dtype_from_fields(fields: list[PointField], point_step: int, is_bigendian: bool) -> np.dtype:
    names: list[str] = []
    formats: list[np.dtype] = []
    offsets: list[int] = []
    byte_order = ">" if is_bigendian else "<"

    for index, field in enumerate(fields):
        base_dtype = POINT_FIELD_DTYPES.get(field.datatype)
        if base_dtype is None:
            raise ValueError(f"Unsupported PointField datatype: {field.datatype}")
        field_dtype = base_dtype.newbyteorder(byte_order)
        if field.count > 1:
            field_dtype = np.dtype((field_dtype, field.count))
        names.append(field.name or f"unnamed_field_{index}")
        formats.append(field_dtype)
        offsets.append(int(field.offset))

    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(point_step),
        }
    )


def normalize_timestamps(timestamps: np.ndarray) -> np.ndarray:
    if timestamps.size == 0:
        return timestamps
    min_time = float(np.min(timestamps))
    max_time = float(np.max(timestamps))
    if not np.isfinite(min_time) or not np.isfinite(max_time) or max_time <= min_time:
        return np.array([], dtype=np.float64)
    return np.divide(timestamps - min_time, max_time - min_time).astype(np.float64, copy=False)


class KissOdometryNode(Node):
    def __init__(self, config: BridgeConfig, load_config_fn, kiss_icp_cls) -> None:
        super().__init__("go2_kiss_odom")
        self._config = config
        self._load_config_fn = load_config_fn
        self._kiss_icp_cls = kiss_icp_cls
        self._kiss_config = self._load_config_fn(self._config.config_file)
        self._kiss_config.data.deskew = self._config.deskew
        self._kiss_icp = self._kiss_icp_cls(self._kiss_config)
        self._lidar_unrotation = rotation_matrix_from_rpy_degrees(*DEFAULT_LIDAR_TF_RPY_DEG).T
        self._publisher = self.create_publisher(Odometry, self._config.odom_topic, 10)
        self._subscription = self.create_subscription(
            PointCloud2,
            self._config.input_topic,
            self._cloud_callback,
            qos_profile_sensor_data,
        )
        self._tf_broadcaster = TransformBroadcaster(self) if self._config.publish_tf else None
        self._last_pose: np.ndarray | None = None
        self._last_stamp_ns: int | None = None
        self._timestamp_warning_emitted = False
        self._timestamp_debug_count = 0
        self._input_frame_info_emitted = False
        self._pending_points: list[np.ndarray] = []
        self._pending_timestamps: list[np.ndarray] = []
        self._pending_header_stamps_ns: list[int] = []
        self._pending_timestamp_compatible = True
        self._pending_latest_msg: PointCloud2 | None = None
        self._pose_anchor = np.eye(4, dtype=np.float64)
        self._registered_frame_count = 0

        self.get_logger().info(
            "Running KISS-ICP on '%s' and publishing odometry on '%s' with TF %s -> %s (deskew=%s, accumulate_frames=%d, reset_after_registrations=%d, unrotating child frame by lidar rpy_deg=(%.2f, %.2f, %.2f))"
            % (
                self._config.input_topic,
                self._config.odom_topic,
                self._config.odom_frame,
                self._config.child_frame,
                self._config.deskew,
                self._config.accumulate_frames,
                self._config.reset_after_registrations,
                DEFAULT_LIDAR_TF_RPY_DEG[0],
                DEFAULT_LIDAR_TF_RPY_DEG[1],
                DEFAULT_LIDAR_TF_RPY_DEG[2],
            )
        )

    def _cloud_callback(self, msg: PointCloud2) -> None:
        if msg.header.frame_id and not self._input_frame_info_emitted:
            self._input_frame_info_emitted = True
            self.get_logger().info(
                "Incoming cloud frame_id is '%s'. Publishing KISS odometry in virtual child frame '%s' after removing the fixed LiDAR mount rotation."
                % (msg.header.frame_id, self._config.child_frame)
            )

        try:
            points, timestamps, timestamp_field_name = self._extract_points_and_timestamps(msg)
        except ValueError as error:
            self.get_logger().error(f"Failed to parse PointCloud2 for KISS-ICP: {error}")
            return

        if points.shape[0] == 0:
            return

        self._pending_points.append(points)
        self._pending_timestamps.append(timestamps)
        self._pending_header_stamps_ns.append(int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec))
        self._pending_latest_msg = msg
        if timestamps.size == 0:
            self._pending_timestamp_compatible = False

        if len(self._pending_points) < self._config.accumulate_frames:
            return

        points, timestamps, accumulated_frame_count = self._consume_accumulated_cloud()
        msg = self._pending_latest_msg
        if msg is None:
            return

        if self._config.deskew and timestamps.size == 0 and not self._timestamp_warning_emitted:
            self._timestamp_warning_emitted = True
            self.get_logger().warning(
                "Deskew requested but no supported per-point timestamp field was found. "
                "KISS-ICP will continue without scan deskewing."
            )

        try:
            self._kiss_icp.register_frame(points, timestamps)
        except Exception as error:
            self.get_logger().error(f"KISS-ICP failed to register the incoming cloud: {error}")
            return

        local_pose = np.asarray(self._kiss_icp.last_pose, dtype=np.float64)
        pose = self._make_unrotated_pose(self._pose_anchor @ local_pose)
        current_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        odom_msg = self._make_odometry_message(msg, pose, current_stamp_ns)
        self._publisher.publish(odom_msg)
        if self._tf_broadcaster is not None:
            self._tf_broadcaster.sendTransform(self._make_transform_message(odom_msg))

        self._last_pose = pose
        self._last_stamp_ns = current_stamp_ns
        self._registered_frame_count += 1
        self._maybe_reset_kiss(local_pose)

    def _extract_points_and_timestamps(self, msg: PointCloud2) -> tuple[np.ndarray, np.ndarray, str | None]:
        cloud_dtype = dtype_from_fields(msg.fields, msg.point_step, msg.is_bigendian)
        width = int(msg.width)
        height = int(msg.height)
        point_step = int(msg.point_step)
        row_step = int(msg.row_step) if int(msg.row_step) > 0 else width * point_step
        packed_row_step = width * point_step
        point_count = width * height

        if row_step < packed_row_step:
            raise ValueError(
                f"row_step={row_step} is smaller than width * point_step={packed_row_step}"
            )

        expected_buffer_size = row_step * height
        if len(msg.data) < expected_buffer_size:
            raise ValueError(
                f"PointCloud2 data buffer is too small: len(data)={len(msg.data)} expected>={expected_buffer_size}"
            )

        cloud = np.ndarray(
            shape=(height, width),
            dtype=cloud_dtype,
            buffer=msg.data,
            strides=(row_step, point_step),
        )
        field_names = cloud.dtype.names or ()
        for field_name in ("x", "y", "z"):
            if field_name not in field_names:
                raise ValueError(f"Point cloud is missing required '{field_name}' field")

        points = np.column_stack(
            (
                np.asarray(cloud["x"], dtype=np.float64).reshape(-1),
                np.asarray(cloud["y"], dtype=np.float64).reshape(-1),
                np.asarray(cloud["z"], dtype=np.float64).reshape(-1),
            )
        )
        mask = np.all(np.isfinite(points), axis=1)

        timestamp_field_name = next((name for name in TIMESTAMP_FIELD_NAMES if name in field_names), None)
        if timestamp_field_name is None:
            timestamps = np.array([], dtype=np.float64)
        else:
            raw_timestamps = np.asarray(cloud[timestamp_field_name], dtype=np.float64).reshape(-1)
            mask &= np.isfinite(raw_timestamps)
            timestamps = raw_timestamps[mask]

        points = points[mask]
        if timestamp_field_name is None:
            timestamps = np.array([], dtype=np.float64)
        elif timestamps.shape[0] != points.shape[0]:
            timestamps = np.array([], dtype=np.float64)

        self._log_timestamp_debug(msg, timestamp_field_name, timestamps)

        return points, timestamps, timestamp_field_name

    def _log_timestamp_debug(
        self,
        msg: PointCloud2,
        timestamp_field_name: str | None,
        timestamps: np.ndarray,
    ) -> None:
        if self._timestamp_debug_count >= TIMESTAMP_DEBUG_LOG_LIMIT:
            return

        if timestamp_field_name is None:
            field_descriptions = ", ".join(
                f"{field.name}:{POINT_FIELD_TYPE_NAMES.get(field.datatype, str(field.datatype))}[{int(field.count)}]@{int(field.offset)}"
                for field in msg.fields
            )
            self.get_logger().warning(
                "Cloud has no supported per-point timestamp field for deskew. Available fields: %s"
                % field_descriptions
            )
            self._timestamp_debug_count += 1
            return

        field_info = next((field for field in msg.fields if field.name == timestamp_field_name), None)
        field_type_name = (
            POINT_FIELD_TYPE_NAMES.get(field_info.datatype, str(field_info.datatype))
            if field_info is not None
            else "unknown"
        )
        field_count = int(field_info.count) if field_info is not None else -1
        field_offset = int(field_info.offset) if field_info is not None else -1

        if timestamps.size == 0:
            self.get_logger().warning(
                "Cloud timestamp field '%s' (%s[%d]@%d) is present, but no finite per-point timestamps survived filtering."
                % (timestamp_field_name, field_type_name, field_count, field_offset)
            )
            self._timestamp_debug_count += 1
            return

        min_timestamp = float(np.min(timestamps))
        max_timestamp = float(np.max(timestamps))
        self.get_logger().info(
            "Cloud timestamp debug: field='%s' type=%s[%d] offset=%d finite_points=%d min=%.9f max=%.9f span=%.9f deskew=%s accumulate_frames=%d"
            % (
                timestamp_field_name,
                field_type_name,
                field_count,
                field_offset,
                timestamps.size,
                min_timestamp,
                max_timestamp,
                max_timestamp - min_timestamp,
                self._config.deskew,
                self._config.accumulate_frames,
            )
        )
        self._timestamp_debug_count += 1

    def _consume_accumulated_cloud(self) -> tuple[np.ndarray, np.ndarray, int]:
        accumulated_frame_count = len(self._pending_points)
        points = np.concatenate(self._pending_points, axis=0)

        timestamps = np.array([], dtype=np.float64)
        if self._pending_timestamp_compatible and self._pending_timestamps and self._pending_header_stamps_ns:
            base_header_stamp_ns = self._pending_header_stamps_ns[0]
            shifted_timestamps = []
            for header_stamp_ns, local_timestamps in zip(self._pending_header_stamps_ns, self._pending_timestamps):
                header_offset_sec = (header_stamp_ns - base_header_stamp_ns) * 1e-9
                shifted_timestamps.append(local_timestamps + header_offset_sec)
            timestamps = normalize_timestamps(np.concatenate(shifted_timestamps, axis=0))

        self._pending_points = []
        self._pending_timestamps = []
        self._pending_header_stamps_ns = []
        self._pending_timestamp_compatible = True
        return points, timestamps, accumulated_frame_count

    def _maybe_reset_kiss(self, local_pose: np.ndarray) -> None:
        reset_after_registrations = self._config.reset_after_registrations
        if reset_after_registrations <= 0 or self._registered_frame_count < reset_after_registrations:
            return

        self._pose_anchor = self._pose_anchor @ local_pose
        self._kiss_icp = self._kiss_icp_cls(self._kiss_config)
        self._registered_frame_count = 0
        self.get_logger().info(
            "Reinitialized KISS-ICP after %d registered frames to limit map history."
            % reset_after_registrations
        )

    def _make_unrotated_pose(self, pose: np.ndarray) -> np.ndarray:
        corrected_pose = np.array(pose, dtype=np.float64, copy=True)
        corrected_pose[:3, :3] = corrected_pose[:3, :3] @ self._lidar_unrotation
        return corrected_pose

    def _make_odometry_message(self, msg: PointCloud2, pose: np.ndarray, stamp_ns: int) -> Odometry:
        odom_msg = Odometry()
        odom_msg.header = msg.header
        odom_msg.header.frame_id = self._config.odom_frame
        odom_msg.child_frame_id = self._config.child_frame

        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        qx, qy, qz, qw = quaternion_from_rotation_matrix(rotation)
        odom_msg.pose.pose.position.x = float(translation[0])
        odom_msg.pose.pose.position.y = float(translation[1])
        odom_msg.pose.pose.position.z = float(translation[2])
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw

        pose_covariance = [0.0] * 36
        for index in (0, 7, 14):
            pose_covariance[index] = self._config.position_covariance
        for index in (21, 28, 35):
            pose_covariance[index] = self._config.orientation_covariance
        odom_msg.pose.covariance = pose_covariance

        twist_covariance = [0.0] * 36
        for index in (0, 7, 14, 21, 28, 35):
            twist_covariance[index] = self._config.twist_covariance
        odom_msg.twist.covariance = twist_covariance

        if self._last_pose is not None and self._last_stamp_ns is not None and stamp_ns > self._last_stamp_ns:
            dt = (stamp_ns - self._last_stamp_ns) * 1e-9
            delta_pose = np.linalg.inv(self._last_pose) @ pose
            linear_velocity = np.divide(delta_pose[:3, 3], dt)
            angular_velocity = np.divide(rotation_vector_from_matrix(delta_pose[:3, :3]), dt)
            odom_msg.twist.twist.linear.x = float(linear_velocity[0])
            odom_msg.twist.twist.linear.y = float(linear_velocity[1])
            odom_msg.twist.twist.linear.z = float(linear_velocity[2])
            odom_msg.twist.twist.angular.x = float(angular_velocity[0])
            odom_msg.twist.twist.angular.y = float(angular_velocity[1])
            odom_msg.twist.twist.angular.z = float(angular_velocity[2])

        return odom_msg

    @staticmethod
    def _make_transform_message(odometry: Odometry) -> TransformStamped:
        transform = TransformStamped()
        transform.header = odometry.header
        transform.child_frame_id = odometry.child_frame_id
        transform.transform.translation.x = odometry.pose.pose.position.x
        transform.transform.translation.y = odometry.pose.pose.position.y
        transform.transform.translation.z = odometry.pose.pose.position.z
        transform.transform.rotation = odometry.pose.pose.orientation
        return transform


def main() -> None:
    config = parse_args()
    load_config_fn, kiss_icp_cls = import_kiss_icp_dependencies()
    rclpy.init(args=None)

    node = KissOdometryNode(config, load_config_fn, kiss_icp_cls)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()