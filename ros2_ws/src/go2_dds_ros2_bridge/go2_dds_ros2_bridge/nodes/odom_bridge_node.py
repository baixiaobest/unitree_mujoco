#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from go2_dds_ros2_bridge.covariance_utils import build_pose_covariance, build_twist_covariance, load_state_variances
from go2_dds_ros2_bridge.dds_runtime import add_dds_runtime_arguments, resolve_runtime_arguments

DEFAULT_DDS_ODOM_TOPIC = "rt/odom"
DEFAULT_ROS_ODOM_TOPIC = "/odom"
ODOM_BRIDGE_DEFAULT_VARIANCES = {
    "x": 1.0,
    "y": 1.0,
    "z": 1.0,
    "roll": 1.0,
    "pitch": 1.0,
    "yaw": 1.0,
    "vx": 0.1,
    "vy": 0.1,
    "vz": 0.1,
    "roll_dt": 0.1,
    "pitch_dt": 0.1,
    "yaw_dt": 0.1,
}


@dataclass(frozen=True)
class BridgeConfig:
    dds_topic: str
    ros_topic: str
    dds_domain_id: int
    dds_interface: str
    publish_hz: float
    covariance_file: Path | None


def import_raw_dds_dependencies():
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_ as DdsOdometry
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

    return ChannelFactoryInitialize, ChannelSubscriber, DdsOdometry


def parse_args() -> BridgeConfig:
    parser = argparse.ArgumentParser(
        description="Bridge Unitree raw DDS odometry to a ROS2 /odom topic."
    )
    add_dds_runtime_arguments(parser)
    parser.add_argument(
        "--dds-topic",
        type=str,
        default=DEFAULT_DDS_ODOM_TOPIC,
        help="Raw DDS odometry topic to subscribe to.",
    )
    parser.add_argument(
        "--ros-topic",
        type=str,
        default=DEFAULT_ROS_ODOM_TOPIC,
        help="ROS2 odometry topic to publish.",
    )
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=100.0,
        help="Maximum ROS2 publish rate for forwarding odometry messages.",
    )
    parser.add_argument(
        "--covariance-file",
        type=Path,
        default=None,
        help="Optional YAML file containing diagonal variances for the published odometry covariance fields.",
    )
    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)
    runtime_profile = resolve_runtime_arguments(args)

    return BridgeConfig(
        dds_topic=args.dds_topic,
        ros_topic=args.ros_topic,
        dds_domain_id=runtime_profile.domain_id,
        dds_interface=runtime_profile.interface,
        publish_hz=max(args.publish_hz, 1.0),
        covariance_file=args.covariance_file,
    )


class DdsOdometryBridge(Node):
    def __init__(self, config: BridgeConfig, channel_subscriber_cls, dds_odometry_type) -> None:
        super().__init__("go2_odometry_bridge")
        self._config = config
        self._publisher = self.create_publisher(Odometry, self._config.ros_topic, 10)
        covariance_variances, covariance_source = load_state_variances(
            self._config.covariance_file,
            ODOM_BRIDGE_DEFAULT_VARIANCES,
        )
        self._pose_covariance = build_pose_covariance(covariance_variances)
        self._twist_covariance = build_twist_covariance(covariance_variances)
        self._message_lock = threading.Lock()
        self._pending_message: Odometry | None = None
        self._message_version = 0
        self._last_published_version = 0

        self._dds_subscriber = channel_subscriber_cls(self._config.dds_topic, dds_odometry_type)
        self._dds_subscriber.Init(self._dds_odometry_handler, 10)
        self._publish_timer = self.create_timer(1.0 / self._config.publish_hz, self._publish_pending_message)

        ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "<unset>")
        self.get_logger().info(
            "Bridging raw DDS odometry '%s' (domain=%d, interface=%s) to ROS2 topic '%s' (ROS_DOMAIN_ID=%s, covariance_source=%s)"
            % (
                self._config.dds_topic,
                self._config.dds_domain_id,
                self._config.dds_interface,
                self._config.ros_topic,
                ros_domain_id,
                covariance_source,
            )
        )

    def _dds_odometry_handler(self, msg) -> None:
        ros_msg = Odometry()
        ros_msg.header.stamp.sec = msg.header.stamp.sec
        ros_msg.header.stamp.nanosec = msg.header.stamp.nanosec
        ros_msg.header.frame_id = msg.header.frame_id
        ros_msg.child_frame_id = msg.child_frame_id

        ros_msg.pose.pose.position.x = msg.pose.pose.position.x
        ros_msg.pose.pose.position.y = msg.pose.pose.position.y
        ros_msg.pose.pose.position.z = msg.pose.pose.position.z
        ros_msg.pose.pose.orientation.w = msg.pose.pose.orientation.w
        ros_msg.pose.pose.orientation.x = msg.pose.pose.orientation.x
        ros_msg.pose.pose.orientation.y = msg.pose.pose.orientation.y
        ros_msg.pose.pose.orientation.z = msg.pose.pose.orientation.z
        ros_msg.pose.covariance = list(self._pose_covariance)

        ros_msg.twist.twist.linear.x = msg.twist.twist.linear.x
        ros_msg.twist.twist.linear.y = msg.twist.twist.linear.y
        ros_msg.twist.twist.linear.z = msg.twist.twist.linear.z
        ros_msg.twist.twist.angular.x = msg.twist.twist.angular.x
        ros_msg.twist.twist.angular.y = msg.twist.twist.angular.y
        ros_msg.twist.twist.angular.z = msg.twist.twist.angular.z
        ros_msg.twist.covariance = list(self._twist_covariance)

        with self._message_lock:
            self._pending_message = ros_msg
            self._message_version += 1

    def _publish_pending_message(self) -> None:
        with self._message_lock:
            if self._pending_message is None or self._message_version == self._last_published_version:
                return
            ros_msg = self._pending_message
            message_version = self._message_version

        self._publisher.publish(ros_msg)
        self._last_published_version = message_version


def main() -> None:
    config = parse_args()
    ChannelFactoryInitialize, channel_subscriber_cls, dds_odometry_type = import_raw_dds_dependencies()
    ChannelFactoryInitialize(config.dds_domain_id, config.dds_interface)
    rclpy.init(args=None)

    node = DdsOdometryBridge(config, channel_subscriber_cls, dds_odometry_type)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()