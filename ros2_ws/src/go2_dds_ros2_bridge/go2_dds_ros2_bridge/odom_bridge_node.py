#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import threading
from dataclasses import dataclass

from geometry_msgs.msg import TransformStamped
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


SIMULATION_DOMAIN_ID = 1
HARDWARE_DOMAIN_ID = 0
SIMULATION_INTERFACE = "wlo1"
HARDWARE_INTERFACE = "enp108s0"

DEFAULT_DDS_ODOM_TOPIC = "rt/odom"
DEFAULT_ROS_ODOM_TOPIC = "/odom"


@dataclass(frozen=True)
class BridgeConfig:
    dds_topic: str
    ros_topic: str
    dds_domain_id: int
    dds_interface: str
    publish_hz: float
    publish_tf: bool


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
        help="Override the raw DDS domain id used to subscribe to the Unitree odometry topic.",
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
        "--publish-tf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish an odom-to-base_link TF using the bridged odometry pose.",
    )

    args = parser.parse_args()
    default_domain = SIMULATION_DOMAIN_ID if args.run_mode == "simulation" else HARDWARE_DOMAIN_ID
    default_interface = SIMULATION_INTERFACE if args.run_mode == "simulation" else HARDWARE_INTERFACE

    return BridgeConfig(
        dds_topic=args.dds_topic,
        ros_topic=args.ros_topic,
        dds_domain_id=default_domain if args.dds_domain_id is None else args.dds_domain_id,
        dds_interface=default_interface if args.dds_interface is None else args.dds_interface,
        publish_hz=max(args.publish_hz, 1.0),
        publish_tf=args.publish_tf,
    )


class DdsOdometryBridge(Node):
    def __init__(self, config: BridgeConfig, channel_subscriber_cls, dds_odometry_type) -> None:
        super().__init__("go2_odometry_bridge")
        self._config = config
        self._publisher = self.create_publisher(Odometry, self._config.ros_topic, 10)
        self._tf_broadcaster = TransformBroadcaster(self) if self._config.publish_tf else None
        self._tf_publish_error_logged = False
        self._message_lock = threading.Lock()
        self._pending_message: Odometry | None = None
        self._message_version = 0
        self._last_published_version = 0

        self._dds_subscriber = channel_subscriber_cls(self._config.dds_topic, dds_odometry_type)
        self._dds_subscriber.Init(self._dds_odometry_handler, 10)
        self._publish_timer = self.create_timer(1.0 / self._config.publish_hz, self._publish_pending_message)

        ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "<unset>")
        self.get_logger().info(
            "Bridging raw DDS odometry '%s' (domain=%d, interface=%s) to ROS2 topic '%s' (ROS_DOMAIN_ID=%s, publish_tf=%s)"
            % (
                self._config.dds_topic,
                self._config.dds_domain_id,
                self._config.dds_interface,
                self._config.ros_topic,
                ros_domain_id,
                self._config.publish_tf,
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
        ros_msg.pose.covariance = list(msg.pose.covariance)

        ros_msg.twist.twist.linear.x = msg.twist.twist.linear.x
        ros_msg.twist.twist.linear.y = msg.twist.twist.linear.y
        ros_msg.twist.twist.linear.z = msg.twist.twist.linear.z
        ros_msg.twist.twist.angular.x = msg.twist.twist.angular.x
        ros_msg.twist.twist.angular.y = msg.twist.twist.angular.y
        ros_msg.twist.twist.angular.z = msg.twist.twist.angular.z
        ros_msg.twist.covariance = list(msg.twist.covariance)

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
        if self._tf_broadcaster is not None:
            try:
                self._tf_broadcaster.sendTransform(self._make_transform(ros_msg))
            except Exception as error:
                if not self._tf_publish_error_logged:
                    self.get_logger().error(f"Failed to publish TF from bridged odometry: {error}")
                    self._tf_publish_error_logged = True
        self._last_published_version = message_version

    def _make_transform(self, odometry: Odometry) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp.sec = odometry.header.stamp.sec
        transform.header.stamp.nanosec = odometry.header.stamp.nanosec
        transform.header.frame_id = odometry.header.frame_id
        transform.child_frame_id = odometry.child_frame_id
        transform.transform.translation.x = odometry.pose.pose.position.x
        transform.transform.translation.y = odometry.pose.pose.position.y
        transform.transform.translation.z = odometry.pose.pose.position.z
        transform.transform.rotation.x = odometry.pose.pose.orientation.x
        transform.transform.rotation.y = odometry.pose.pose.orientation.y
        transform.transform.rotation.z = odometry.pose.pose.orientation.z
        transform.transform.rotation.w = odometry.pose.pose.orientation.w
        return transform


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