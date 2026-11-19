#!/usr/bin/env python3
"""Prepare atomic CBF snapshots without blocking the control/QP thread."""
from collections import deque
from uuid import uuid4
from copy import deepcopy
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped, TransformStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster, TransformException
from go2_dds_ros2_bridge_msgs.msg import CbfScan3D, CbfWorldScan, CbfControlSnapshot
from go2_dds_ros2_bridge.cbf_world import pose_at, aligned_state, world_points
from go2_dds_ros2_bridge.tf_utils import DEFAULT_LIDAR_TF_XYZ, rotation_matrix_from_quaternion_xyzw, quaternion_from_rotation_matrix


def ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class CbfWorldNode(Node):
    def __init__(self):
        super().__init__('cbf_world_state')
        self.declare_parameter('max_state_age_s', .1)
        self.declare_parameter('max_interpolation_gap_s', .1)
        self.declare_parameter('scan_timeout_s', .5)
        self.declare_parameter('global_frame', 'camera_init_correct')
        self.declare_parameter('map_rear_obstacles', False)
        self.tf = Buffer(cache_time=Duration(seconds=10))
        self.listener = TransformListener(self.tf, self)
        self.broadcaster = StaticTransformBroadcaster(self)
        self.poses, self.velocities, self.pending, self.scans = (deque(maxlen=2048), deque(maxlen=2048), deque(maxlen=16), deque(maxlen=16))
        self.reference = None
        self.blocked = False
        self.session = uuid4().hex
        self.started = self.get_clock().now().nanoseconds
        self.last_now = self.started
        self.map_rear = None
        self.map_enabled = bool(self.get_parameter('map_rear_obstacles').value)
        if self.map_enabled:
            self.create_subscription(PointCloud2, '/cbf/map_rear_obstacles', self.map_callback, 10)
            self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.reset_pose_callback, 10)
            self.create_subscription(PoseWithCovarianceStamped, '/icp_result', self.reset_pose_callback, 10)
        self.create_subscription(Odometry, '/Odometry', self.pose_callback, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, '/estimated_velocity', self.velocity_callback, qos_profile_sensor_data)
        self.create_subscription(CbfScan3D, '/cbf/scan_xyz', self.scan_callback, qos_profile_sensor_data)
        self.world_pub = self.create_publisher(CbfWorldScan, '/cbf/world_scan', 10)
        self.snapshot_pub = self.create_publisher(CbfControlSnapshot, '/cbf/snapshot', 1)
        self.create_service(Trigger, '/cbf/reset_world', self.reset_callback)
        self.create_timer(.01, self.tick)

    def invalidate(self):
        self.poses.clear(); self.velocities.clear(); self.pending.clear(); self.scans.clear()
        self.reference = None
        self.session = uuid4().hex
        self.started = self.get_clock().now().nanoseconds
        self.map_rear = None

    def reset_pose_callback(self, msg):
        self.invalidate()

    def map_callback(self, msg):
        if msg.header.frame_id != self.get_parameter('global_frame').value or msg.point_step != 12:
            return
        if [f.name for f in msg.fields] != ['x', 'y', 'z']:
            return
        points = np.frombuffer(msg.data, dtype='<f4').reshape(-1, 3).astype(np.float64)
        stamp = ns(msg.header.stamp)
        if stamp >= self.started and np.isfinite(points).all():
            self.map_rear = stamp, points

    def merge_map_rear(self, scan, robot_xy, yaw, now):
        if not self.map_enabled or self.map_rear is None:
            return scan
        stamp, points = self.map_rear
        if now - stamp > 200_000_000 or stamp > now + 1_000_000 or not len(points):
            return scan
        merged = deepcopy(scan)
        c, s = math.cos(yaw), math.sin(yaw)
        lidar_x = robot_xy[0] + c * DEFAULT_LIDAR_TF_XYZ[0] - s * DEFAULT_LIDAR_TF_XYZ[1]
        lidar_y = robot_xy[1] + s * DEFAULT_LIDAR_TF_XYZ[0] + c * DEFAULT_LIDAR_TF_XYZ[1]
        for point in points:
            dx, dy = point[0] - lidar_x, point[1] - lidar_y
            x, y = c * dx + s * dy, -s * dx + c * dy
            if x >= 0:
                continue
            index = min(63, max(0, int((math.atan2(y, -x) / math.pi + 0.5) * 64)))
            old_x, old_y = merged.static_points_xy_m[2 * index:2 * index + 2]
            if not merged.static_hits[index] or dx * dx + dy * dy < (old_x - lidar_x) ** 2 + (old_y - lidar_y) ** 2:
                merged.static_points_xy_m[2 * index:2 * index + 2] = [float(point[0]), float(point[1])]
                merged.static_hits[index] = 1
        return merged

    def reset_callback(self, request, response):
        self.invalidate()
        self.blocked = False
        response.success = True
        response.message = 'Caches cleared; waiting for fresh state and leveling transform.'
        return response

    def append_sample(self, history, sample):
        if sample[0] < self.started or sample[0] > self.get_clock().now().nanoseconds + 1_000_000:
            return
        if history and sample[0] <= history[-1][0]:
            return  # duplicates/out-of-order input cannot move the state backwards
        history.append(sample)

    def pose_callback(self, msg):
        if msg.header.frame_id != 'camera_init' or msg.child_frame_id != 'body':
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        values = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        if np.isfinite(values).all() and np.linalg.norm(values[3:]) > 1e-9:
            self.append_sample(self.poses, (ns(msg.header.stamp), values[:3], values[3:]))

    def velocity_callback(self, msg):
        if msg.header.frame_id != 'base_link':
            return
        v = msg.twist.linear
        values = np.array([v.x, v.y, v.z])
        if np.isfinite(values).all():
            self.append_sample(self.velocities, (ns(msg.header.stamp), values))

    def scan_callback(self, msg):
        stamp = ns(msg.header.stamp)
        if msg.header.frame_id != 'base_link' or stamp < self.started or ns(msg.scan_start) > stamp:
            return
        if self.pending and stamp <= ns(self.pending[-1].header.stamp):
            return
        if self.scans and stamp <= ns(self.scans[-1].header.stamp):
            return
        self.pending.append(msg)

    def initialize_reference(self):
        # camera_init has an existing parent; publish the INVERSE as its child.
        lookup_time = Time(nanoseconds=self.poses[-1][0]) if self.map_enabled else Time()
        correction = self.tf.lookup_transform(self.get_parameter('global_frame').value, 'camera_init', lookup_time)
        offset = self.tf.lookup_transform('body', 'base_link', Time())
        t, q = correction.transform.translation, correction.transform.rotation
        rotation = rotation_matrix_from_quaternion_xyzw(q.x, q.y, q.z, q.w)
        translation = np.array([t.x, t.y, t.z])
        b = offset.transform.translation
        oq = offset.transform.rotation
        if not np.allclose(rotation_matrix_from_quaternion_xyzw(oq.x, oq.y, oq.z, oq.w), np.eye(3), atol=1e-8):
            raise ValueError('unsupported_body_base_rotation')
        self.reference = rotation, translation, np.array([b.x, b.y, b.z])
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'camera_init'
        tf.child_frame_id = 'cbf_world'
        inverse_t = -rotation.T @ translation
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, inverse_t)
        q = quaternion_from_rotation_matrix(rotation.T)
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, q)
        self.broadcaster.sendTransform(tf)

    def tick(self):
        now = self.get_clock().now().nanoseconds
        snapshot = CbfControlSnapshot()
        snapshot.header.frame_id = 'cbf_world'
        snapshot.header.stamp = Time(nanoseconds=now).to_msg()
        snapshot.session_id = self.session
        try:
            if now < self.last_now:
                self.invalidate()
                self.blocked = True
            if self.blocked:
                raise ValueError('world_reset_required')
            if self.reference is None:
                if self.map_enabled and not self.poses:
                    raise ValueError('state_unavailable')
                self.initialize_reference()
            rotation, translation, offset = self.reference
            gap_ns = int(self.get_parameter('max_interpolation_gap_s').value * 1e9)
            age_ns = int(self.get_parameter('max_state_age_s').value * 1e9)
            scan_age_ns = int(self.get_parameter('scan_timeout_s').value * 1e9)
            while self.pending:
                raw = self.pending[0]
                stamp = ns(raw.header.stamp)
                if now - ns(raw.scan_start) > scan_age_ns or (self.poses and stamp < self.poses[0][0]):
                    self.pending.popleft(); continue
                try:
                    p, r, _ = pose_at(self.poses, stamp, gap_ns)
                except ValueError:
                    break  # bounded queue retries when the next pose arrives
                self.pending.popleft()
                world = CbfWorldScan()
                world.header = raw.header
                world.header.frame_id = 'cbf_world'
                world.scan_start, world.sequence = raw.scan_start, raw.sequence
                world.session_id = self.session
                wr = rotation @ r
                world.capture_yaw = float(np.arctan2(wr[1, 0], wr[0, 0]))
                world.points_xy_m = world_points(raw.points_xyz_m, p, r, rotation, translation, offset)[:, :2].ravel().tolist()
                world.static_points_xy_m = world_points(raw.static_points_xyz_m, p, r, rotation, translation, offset)[:, :2].ravel().tolist()
                world.hits, world.static_hits = raw.hits, raw.static_hits
                world.capture_points_xy_m = world_points(
                    raw.capture_points_xyz_m, p, r, rotation, translation, offset
                )[:, :2].ravel().tolist()
                world.ray_states = raw.ray_states
                self.scans.append(world)
                capture_position = rotation @ p + translation + wr @ offset
                world = self.merge_map_rear(world, capture_position[:2], world.capture_yaw, now)
                self.world_pub.publish(world)
            # Only samples whose own attitude is covered can supply world velocity.
            velocities = [v for v in self.velocities if self.poses and self.poses[0][0] <= v[0] <= self.poses[-1][0]]
            stamp, p, yaw, velocity, gap = aligned_state(self.poses, velocities, rotation, translation, offset, now, age_ns, gap_ns)
            eligible = [s for s in self.scans if ns(s.header.stamp) <= stamp and now - ns(s.scan_start) <= scan_age_ns]
            if not eligible:
                raise ValueError('eligible_scan_unavailable')
            snapshot.header.stamp = Time(nanoseconds=stamp).to_msg()
            snapshot.robot_xy_m = p[:2].tolist()
            snapshot.robot_yaw = float(yaw)
            snapshot.velocity_xy_mps = velocity[:2].tolist()
            snapshot.interpolation_gap_s = gap
            snapshot.scan = self.merge_map_rear(eligible[-1], p[:2], yaw, now)
            snapshot.valid = True
            snapshot.reason = 'aligned_measured_state'
        except (ValueError, TransformException) as error:
            snapshot.reason = str(error)
        self.last_now = now
        snapshot.session_id = self.session
        self.snapshot_pub.publish(snapshot)


def main():
    rclpy.init()
    node = CbfWorldNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
