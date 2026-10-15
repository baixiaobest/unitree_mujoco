#!/usr/bin/env python3
"""Anchor the L2 display world at the initial virtual Go2 IMU pose."""

from __future__ import annotations

import math
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from go2_dds_ros2_bridge.tf_utils import (
    lidar_pose_in_imu_frame,
    quaternion_from_rotation_matrix,
    rotation_matrix_from_quaternion_xyzw,
)


WORLD = "fastlio_l2_world"
ODOM = "fastlio_l2_odom"
IMU = "fastlio_l2_imu"
GO2_IMU = "fastlio_l2_go2_imu"
LIDAR = "fastlio_l2_lidar"


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def pose_matrix(translation, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix_from_quaternion_xyzw(*quaternion)
    matrix[:3, 3] = translation
    return matrix


class FastLioL2Frames(Node):
    def __init__(self):
        super().__init__("fastlio_l2_frames")
        self._static_tf = StaticTransformBroadcaster(self)
        self._dynamic_tf = TransformBroadcaster(self)
        self._latest_extrinsic = None
        self._latest_odom = None
        self._anchored = False
        translation, rotation = lidar_pose_in_imu_frame()  # Go2 IMU <- LiDAR
        self._go2_lidar = np.eye(4)
        self._go2_lidar[:3, :3] = rotation
        self._go2_lidar[:3, 3] = translation
        self._lidar_go2 = np.linalg.inv(self._go2_lidar)
        mount = TransformStamped()
        mount.header.stamp = self.get_clock().now().to_msg()
        mount.header.frame_id = LIDAR
        mount.child_frame_id = GO2_IMU
        (mount.transform.translation.x, mount.transform.translation.y,
         mount.transform.translation.z) = self._lidar_go2[:3, 3]
        q_out = quaternion_from_rotation_matrix(self._lidar_go2[:3, :3])
        (mount.transform.rotation.x, mount.transform.rotation.y,
         mount.transform.rotation.z, mount.transform.rotation.w) = q_out
        self._mount_tf = mount
        self._static_tf.sendTransform(mount)
        self.create_subscription(TransformStamped, "/fastlio_l2/estimated_extrinsic", self._on_extrinsic, 20)
        self.create_subscription(Odometry, "/fastlio_l2/odom", self._on_odom, 20)

    def _on_extrinsic(self, msg):
        if msg.header.frame_id != IMU or msg.child_frame_id != LIDAR:
            return
        t = msg.transform.translation
        q = msg.transform.rotation
        values = (t.x, t.y, t.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(value) for value in values) or sum(value * value for value in values[3:]) < 1e-12:
            return
        imu_lidar = pose_matrix(values[:3], values[3:])
        imu_go2 = imu_lidar @ self._lidar_go2
        self._latest_extrinsic = (stamp_ns(msg.header.stamp), imu_go2)
        self._dynamic_tf.sendTransform(msg)
        self._try_anchor()

    def _on_odom(self, msg):
        if msg.header.frame_id != ODOM or msg.child_frame_id != IMU:
            return
        self._latest_odom = msg
        self._try_anchor()

    def _try_anchor(self):
        if self._anchored or self._latest_extrinsic is None or self._latest_odom is None:
            return
        msg = self._latest_odom
        extrinsic_stamp, imu_go2 = self._latest_extrinsic
        if abs(stamp_ns(msg.header.stamp) - extrinsic_stamp) > 1_000_000:
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(value) for value in values) or sum(value * value for value in values[3:]) < 1e-12:
            return
        odom_imu = pose_matrix(values[:3], values[3:])
        odom_go2 = odom_imu @ imu_go2
        initial_go2_position = odom_go2[:3, 3]
        world_odom_rotation = odom_go2[:3, :3].T
        world_odom_translation = -world_odom_rotation @ initial_go2_position
        anchor = TransformStamped()
        anchor.header.stamp = msg.header.stamp
        anchor.header.frame_id = WORLD
        anchor.child_frame_id = ODOM
        (anchor.transform.translation.x, anchor.transform.translation.y,
         anchor.transform.translation.z) = world_odom_translation
        q_out = quaternion_from_rotation_matrix(world_odom_rotation)
        (anchor.transform.rotation.x, anchor.transform.rotation.y,
         anchor.transform.rotation.z, anchor.transform.rotation.w) = q_out
        # Humble's static broadcaster replaces its latched TFMessage on each
        # call, so include both edges for subscribers that join after startup.
        self._static_tf.sendTransform([self._mount_tf, anchor])
        self._anchored = True
        self.get_logger().info(
            f"Anchored {WORLD} at initial Go2 IMU pose "
            f"{initial_go2_position.tolist()}; axes aligned")


def main():
    rclpy.init()
    node = FastLioL2Frames()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
