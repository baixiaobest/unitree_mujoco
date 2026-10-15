#!/usr/bin/env python3
"""Level the L2 display world and anchor it at the initial Go2 IMU pose."""

from __future__ import annotations

import math
from collections import deque
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu as ImuMessage
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


def rotation_from_to(source, target):
    """Shortest rotation mapping one unit direction to another."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    if dot < -1.0 + 1e-8:
        axis = np.cross(source, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(source, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew / (1.0 + dot)


def level_world_rotation(odom_imu_rotation, imu_go2_rotation, mean_acceleration):
    """Make gravity vertical and the initial Go2 IMU heading zero."""
    up_in_odom = odom_imu_rotation @ mean_acceleration
    level = rotation_from_to(up_in_odom, [0.0, 0.0, 1.0])
    go2_in_level = level @ odom_imu_rotation @ imu_go2_rotation
    heading = math.atan2(go2_in_level[1, 0], go2_in_level[0, 0])
    c, s = math.cos(heading), math.sin(heading)
    remove_heading = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return remove_heading @ level


class FastLioL2Frames(Node):
    def __init__(self):
        super().__init__("fastlio_l2_frames")
        self._static_tf = StaticTransformBroadcaster(self)
        self._dynamic_tf = TransformBroadcaster(self)
        self._latest_extrinsic = None
        self._latest_odom = None
        self._anchored = False
        self._stationary_imu = deque(maxlen=250)
        translation, rotation = lidar_pose_in_imu_frame()  # Go2 IMU <- LiDAR
        self._go2_lidar = np.eye(4)
        self._go2_lidar[:3, :3] = rotation
        self._go2_lidar[:3, 3] = translation
        self.create_subscription(TransformStamped, "/fastlio_l2/estimated_extrinsic", self._on_extrinsic, 20)
        self.create_subscription(Odometry, "/fastlio_l2/odom", self._on_odom, 20)
        self.create_subscription(ImuMessage, "/fastlio_l2/input/imu", self._on_imu, 100)

    def _on_imu(self, msg):
        if self._anchored or msg.header.frame_id != IMU:
            return
        acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y,
                        msg.linear_acceleration.z], dtype=float)
        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                         msg.angular_velocity.z], dtype=float)
        if np.isfinite(acc).all() and np.isfinite(gyro).all() and 7.0 < np.linalg.norm(acc) < 12.0:
            self._stationary_imu.append((acc, gyro))
            self._try_anchor()

    def _on_extrinsic(self, msg):
        if msg.header.frame_id != IMU or msg.child_frame_id != LIDAR:
            return
        t = msg.transform.translation
        q = msg.transform.rotation
        values = (t.x, t.y, t.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(value) for value in values) or sum(value * value for value in values[3:]) < 1e-12:
            return
        imu_lidar = pose_matrix(values[:3], values[3:])
        imu_go2 = imu_lidar @ np.linalg.inv(self._go2_lidar)
        self._latest_extrinsic = (stamp_ns(msg.header.stamp), imu_go2)
        body = TransformStamped()
        body.header.stamp = msg.header.stamp
        body.header.frame_id = IMU
        body.child_frame_id = GO2_IMU
        body.transform.translation.x, body.transform.translation.y, body.transform.translation.z = imu_go2[:3, 3]
        q_out = quaternion_from_rotation_matrix(imu_go2[:3, :3])
        body.transform.rotation.x, body.transform.rotation.y, body.transform.rotation.z, body.transform.rotation.w = q_out
        self._dynamic_tf.sendTransform([msg, body])
        self._try_anchor()

    def _on_odom(self, msg):
        if msg.header.frame_id != ODOM or msg.child_frame_id != IMU:
            return
        self._latest_odom = msg
        self._try_anchor()

    def _try_anchor(self):
        if (self._anchored or self._latest_extrinsic is None or self._latest_odom is None
                or len(self._stationary_imu) < 200):
            return
        samples = np.array(self._stationary_imu)
        if np.max(np.linalg.norm(samples[:, 1, :], axis=1)) > 0.05:
            return
        mean_acceleration = samples[:, 0, :].mean(axis=0)
        if np.max(np.std(samples[:, 0, :], axis=0)) > 0.35:
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
        initial_go2_position = (odom_imu @ imu_go2)[:3, 3]
        world_odom_rotation = level_world_rotation(
            odom_imu[:3, :3], imu_go2[:3, :3], mean_acceleration)
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
        self._static_tf.sendTransform(anchor)
        self._anchored = True
        self.get_logger().info(
            f"Anchored level {WORLD} at initial Go2 IMU position "
            f"{initial_go2_position.tolist()}; heading zeroed")


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
