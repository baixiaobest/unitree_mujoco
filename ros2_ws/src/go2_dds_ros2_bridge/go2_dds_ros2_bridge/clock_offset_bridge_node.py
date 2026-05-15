#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
from dataclasses import dataclass
from time import time_ns

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster

from go2_dds_ros2_bridge.tf_utils import DEFAULT_LIDAR_TF_RPY_DEG, DEFAULT_LIDAR_TF_XYZ, quaternion_from_rpy


SIMULATION_DOMAIN_ID = 1
HARDWARE_DOMAIN_ID = 0
SIMULATION_INTERFACE = "wlo1"
HARDWARE_INTERFACE = "enp108s0"

DEFAULT_DDS_TOPIC = "rt/utlidar/cloud"
DEFAULT_ROS_TOPIC = "/go2_clock_offset_sec"
DEFAULT_OUTPUT_TOPIC = "/utlidar/time_corrected/cloud"
DEFAULT_TF_PARENT_FRAME = "base_link"
DEFAULT_TF_CHILD_FRAME = "utlidar_lidar"
OUTPUT_CLOUD_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


@dataclass(frozen=True)
class BridgeConfig:
    dds_topic: str
    ros_topic: str
    output_topic: str
    dds_domain_id: int
    dds_interface: str
    publish_hz: float
    filter_alpha: float
    publish_tf: bool


def import_raw_dds_dependencies():
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_ as DdsPointCloud2
    except ModuleNotFoundError as error:
        if error.name == "cyclonedds":
            raise SystemExit(
                "The ROS2 bridge runtime cannot import 'cyclonedds'.\n"
                "Install cyclonedds into the same Python interpreter that runs ROS2.\n"
                "For ROS Humble on this machine that is typically /usr/bin/python3 (Python 3.10).\n"
                "The Unitree conda environment has cyclonedds for Python 3.12, which is not import-compatible\n"
                "with the ROS Humble Python runtime."
            ) from error
        raise SystemExit(
            "The ROS2 bridge runtime cannot import unitree_sdk2py.\n"
            "Install unitree_sdk2py into the same Python interpreter that runs ROS2, or extend PYTHONPATH\n"
            "to point at a compatible installation built for the same Python version."
        ) from error

    return ChannelFactoryInitialize, ChannelSubscriber, DdsPointCloud2


def parse_args() -> BridgeConfig:
    parser = argparse.ArgumentParser(
        description="Estimate and publish the onboard-to-local clock offset from the Unitree DDS LiDAR cloud topic."
    )
    parser.add_argument(
        "--run-mode",
        choices=("simulation", "hardware"),
        default="hardware",
        help="Select the default DDS domain/interface pair.",
    )
    parser.add_argument(
        "--dds-domain-id",
        type=int,
        default=None,
        help="Override the raw DDS domain id used to subscribe to the Unitree LiDAR cloud topic.",
    )
    parser.add_argument(
        "--dds-interface",
        type=str,
        default=None,
        help="Override the network interface used for the raw DDS subscriber.",
    )
    parser.add_argument(
        "--dds-topic",
        type=str,
        default=DEFAULT_DDS_TOPIC,
        help="Raw DDS LiDAR cloud topic to subscribe to.",
    )
    parser.add_argument(
        "--ros-topic",
        type=str,
        default=DEFAULT_ROS_TOPIC,
        help="ROS2 topic that publishes the filtered clock offset in seconds.",
    )
    parser.add_argument(
        "--output-topic",
        type=str,
        default=DEFAULT_OUTPUT_TOPIC,
        help="ROS2 PointCloud2 topic published with corrected local timestamps.",
    )
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=20.0,
        help="Maximum ROS2 publish rate for forwarding the filtered offset.",
    )
    parser.add_argument(
        "--filter-alpha",
        type=float,
        default=0.05,
        help="Low-pass filter alpha applied to each new offset sample, in (0, 1].",
    )
    parser.add_argument(
        "--publish-tf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish the base_link to utlidar_lidar transform.",
    )

    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)
    default_domain = SIMULATION_DOMAIN_ID if args.run_mode == "simulation" else HARDWARE_DOMAIN_ID
    default_interface = SIMULATION_INTERFACE if args.run_mode == "simulation" else HARDWARE_INTERFACE
    filter_alpha = min(max(args.filter_alpha, 1e-4), 1.0)

    return BridgeConfig(
        dds_topic=args.dds_topic,
        ros_topic=args.ros_topic,
        output_topic=args.output_topic,
        dds_domain_id=default_domain if args.dds_domain_id is None else args.dds_domain_id,
        dds_interface=default_interface if args.dds_interface is None else args.dds_interface,
        publish_hz=max(args.publish_hz, 1.0),
        filter_alpha=filter_alpha,
        publish_tf=args.publish_tf,
    )


class Go2ClockOffsetBridge(Node):
    def __init__(self, config: BridgeConfig, channel_subscriber_cls, point_cloud_type) -> None:
        super().__init__("go2_clock_offset_bridge")
        self._config = config
        self._publisher = self.create_publisher(Float64, self._config.ros_topic, 10)
        self._cloud_publisher = self.create_publisher(PointCloud2, self._config.output_topic, OUTPUT_CLOUD_QOS)
        self._sample_lock = threading.Lock()
        self._filtered_offset_sec: float | None = None
        self._message_version = 0
        self._last_published_version = 0
        self._sample_count = 0
        self._last_log_time_ns = 0
        self._frame_warning_emitted = False
        self._tf_lock = threading.Lock()
        self._lidar_tf_rpy_deg = DEFAULT_LIDAR_TF_RPY_DEG

        self._dds_subscriber = channel_subscriber_cls(self._config.dds_topic, point_cloud_type)
        self._dds_subscriber.Init(self._dds_state_handler, 10)
        self._publish_timer = self.create_timer(1.0 / self._config.publish_hz, self._publish_pending_offset)
        self._tf_broadcaster = TransformBroadcaster(self) if self._config.publish_tf else None
        self.declare_parameter("lidar_roll_deg", DEFAULT_LIDAR_TF_RPY_DEG[0])
        self.declare_parameter("lidar_pitch_deg", DEFAULT_LIDAR_TF_RPY_DEG[1])
        self.declare_parameter("lidar_yaw_deg", DEFAULT_LIDAR_TF_RPY_DEG[2])
        self._lidar_tf_rpy_deg = self._read_lidar_tf_rpy_parameters_deg()
        self.add_on_set_parameters_callback(self._handle_parameter_update)
        self._publish_lidar_tf()

        ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "<unset>")
        self.get_logger().info(
            "Estimating GO2 clock offset from DDS topic '%s' (domain=%d, interface=%s, using cloud header stamp) "
            "and re-publishing corrected clouds on '%s' (ROS_DOMAIN_ID=%s, alpha=%.4f, publish_tf=%s). "
            "Publishing LiDAR TF %s -> %s with fixed xyz=(%.4f, %.4f, %.4f) and configurable rpy_deg=(%.2f, %.2f, %.2f)"
            % (
                self._config.dds_topic,
                self._config.dds_domain_id,
                self._config.dds_interface,
                self._config.output_topic,
                ros_domain_id,
                self._config.filter_alpha,
                self._config.publish_tf,
                DEFAULT_TF_PARENT_FRAME,
                DEFAULT_TF_CHILD_FRAME,
                DEFAULT_LIDAR_TF_XYZ[0],
                DEFAULT_LIDAR_TF_XYZ[1],
                DEFAULT_LIDAR_TF_XYZ[2],
                self._lidar_tf_rpy_deg[0],
                self._lidar_tf_rpy_deg[1],
                self._lidar_tf_rpy_deg[2],
            )
        )

    def _read_lidar_tf_rpy_parameters_deg(self) -> tuple[float, float, float]:
        return (
            float(self.get_parameter("lidar_roll_deg").value),
            float(self.get_parameter("lidar_pitch_deg").value),
            float(self.get_parameter("lidar_yaw_deg").value),
        )

    def _handle_parameter_update(self, parameters: list[Parameter]) -> SetParametersResult:
        updated_rpy_deg = list(self._lidar_tf_rpy_deg)
        name_to_index = {"lidar_roll_deg": 0, "lidar_pitch_deg": 1, "lidar_yaw_deg": 2}

        for parameter in parameters:
            if parameter.name not in name_to_index:
                continue
            if parameter.type_ not in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                return SetParametersResult(
                    successful=False,
                    reason=f"{parameter.name} must be numeric",
                )
            value = float(parameter.value)
            if not math.isfinite(value):
                return SetParametersResult(
                    successful=False,
                    reason=f"{parameter.name} must be finite",
                )
            updated_rpy_deg[name_to_index[parameter.name]] = value

        new_rpy_deg = tuple(updated_rpy_deg)
        with self._tf_lock:
            old_rpy_deg = self._lidar_tf_rpy_deg
            self._lidar_tf_rpy_deg = new_rpy_deg

        if new_rpy_deg != old_rpy_deg:
            self.get_logger().info(
                "Updated LiDAR TF rpy_deg to (%.2f, %.2f, %.2f)"
                % (new_rpy_deg[0], new_rpy_deg[1], new_rpy_deg[2])
            )
            self._publish_lidar_tf()

        return SetParametersResult(successful=True)

    def _publish_lidar_tf(self, stamp: Time | None = None) -> None:
        if self._tf_broadcaster is None:
            return
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg() if stamp is None else stamp
        transform.header.frame_id = DEFAULT_TF_PARENT_FRAME
        transform.child_frame_id = DEFAULT_TF_CHILD_FRAME
        transform.transform.translation.x = DEFAULT_LIDAR_TF_XYZ[0]
        transform.transform.translation.y = DEFAULT_LIDAR_TF_XYZ[1]
        transform.transform.translation.z = DEFAULT_LIDAR_TF_XYZ[2]

        with self._tf_lock:
            lidar_tf_rpy_deg = self._lidar_tf_rpy_deg

        qx, qy, qz, qw = quaternion_from_rpy(*(math.radians(value) for value in lidar_tf_rpy_deg))
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(transform)

    def _dds_state_handler(self, msg) -> None:
        if not self._frame_warning_emitted and msg.header.frame_id != DEFAULT_TF_CHILD_FRAME:
            self._frame_warning_emitted = True
            self.get_logger().warning(
                "Incoming cloud frame_id '%s' does not match configured static TF child '%s'. "
                "RViz will still need a transform for the actual cloud frame."
                % (msg.header.frame_id, DEFAULT_TF_CHILD_FRAME)
            )

        remote_stamp_ns = self._stamp_to_ns(msg.header.stamp)
        if remote_stamp_ns <= 0:
            return

        local_receive_ns = time_ns()
        raw_offset_sec = (local_receive_ns - remote_stamp_ns) * 1e-9
        corrected_stamp_ns = remote_stamp_ns + int(round(raw_offset_sec * 1_000_000_000.0))
        if corrected_stamp_ns < 0:
            corrected_stamp_ns = 0

        with self._sample_lock:
            if self._filtered_offset_sec is None:
                self._filtered_offset_sec = raw_offset_sec
            else:
                alpha = self._config.filter_alpha
                self._filtered_offset_sec = (1.0 - alpha) * self._filtered_offset_sec + alpha * raw_offset_sec
            self._message_version += 1
            self._sample_count += 1
            filtered_offset_sec = self._filtered_offset_sec
            sample_count = self._sample_count

        if local_receive_ns - self._last_log_time_ns >= 2_000_000_000:
            self._last_log_time_ns = local_receive_ns
            self.get_logger().info(
                "Filtered GO2 clock offset: %.6f s after %d samples"
                % (filtered_offset_sec, sample_count)
            )

        self._publish_lidar_tf(self._ns_to_stamp(corrected_stamp_ns))
        self._cloud_publisher.publish(self._make_corrected_cloud_message(msg, corrected_stamp_ns))

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        timestamp_sec = int(stamp.sec)
        timestamp_nanosec = int(stamp.nanosec)
        if timestamp_sec < 0 or timestamp_nanosec < 0:
            return 0
        return timestamp_sec * 1_000_000_000 + timestamp_nanosec

    @staticmethod
    def _ns_to_stamp(stamp_ns: int) -> Time:
        stamp = Time()
        stamp.sec = int(stamp_ns // 1_000_000_000)
        stamp.nanosec = int(stamp_ns % 1_000_000_000)
        return stamp

    def _make_corrected_cloud_message(self, dds_msg, corrected_stamp_ns: int) -> PointCloud2:
        ros_msg = PointCloud2()
        ros_msg.header.stamp = self._ns_to_stamp(corrected_stamp_ns)
        ros_msg.header.frame_id = dds_msg.header.frame_id
        ros_msg.height = int(dds_msg.height)
        ros_msg.width = int(dds_msg.width)
        ros_msg.fields = [
            PointField(
                name=field.name,
                offset=int(field.offset),
                datatype=int(field.datatype),
                count=int(field.count),
            )
            for field in dds_msg.fields
        ]
        ros_msg.is_bigendian = bool(dds_msg.is_bigendian)
        ros_msg.point_step = int(dds_msg.point_step)
        ros_msg.row_step = int(dds_msg.row_step)
        ros_msg.data = bytes(dds_msg.data)
        ros_msg.is_dense = bool(dds_msg.is_dense)
        return ros_msg

    def _publish_pending_offset(self) -> None:
        with self._sample_lock:
            if self._filtered_offset_sec is None or self._message_version == self._last_published_version:
                return
            offset_sec = self._filtered_offset_sec
            message_version = self._message_version

        message = Float64()
        message.data = offset_sec
        self._publisher.publish(message)
        self._last_published_version = message_version


def main() -> None:
    config = parse_args()
    ChannelFactoryInitialize, channel_subscriber_cls, point_cloud_type = import_raw_dds_dependencies()
    ChannelFactoryInitialize(config.dds_domain_id, config.dds_interface)
    rclpy.init(args=None)

    node = Go2ClockOffsetBridge(config, channel_subscriber_cls, point_cloud_type)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()