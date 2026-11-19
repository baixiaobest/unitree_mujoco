#!/usr/bin/env python3
"""Publish nearby mapped rear obstacles and the fitted local floor for RViz."""
import numpy as np
from scipy.spatial import cKDTree
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import PoseWithCovarianceStamped, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException
from go2_dds_ros2_bridge.cbf_map import load_pcd_xyz, floor_and_rear, voxel_downsample
from go2_dds_ros2_bridge.tf_utils import DEFAULT_LIDAR_TF_XYZ, rotation_matrix_from_quaternion_xyzw


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class CbfMapObstacles(Node):
    def __init__(self):
        super().__init__('cbf_map_obstacles')
        self.declare_parameter('map_path', '')
        self.declare_parameter('map_obstacle_range_m', 3.0)
        self.declare_parameter('map_voxel_m', 0.1)
        self.declare_parameter('min_obstacle_height_m', 0.12)
        self.declare_parameter('max_obstacle_height_m', 1.4)
        self.declare_parameter('floor_tolerance_m', 0.08)
        self.declare_parameter('support_radius_m', 0.2)
        self.declare_parameter('min_support_points', 3)
        self.declare_parameter('max_pose_age_s', 0.2)
        self.map_obstacle_range_m = float(self.get_parameter('map_obstacle_range_m').value)
        if not np.isfinite(self.map_obstacle_range_m) or self.map_obstacle_range_m <= 0:
            raise ValueError('map_obstacle_range_m must be a finite positive distance')
        self.min_obstacle_height_m = float(self.get_parameter('min_obstacle_height_m').value)
        self.max_obstacle_height_m = float(self.get_parameter('max_obstacle_height_m').value)
        self.floor_tolerance_m = float(self.get_parameter('floor_tolerance_m').value)
        if (not all(np.isfinite(v) for v in (self.min_obstacle_height_m,
                                            self.max_obstacle_height_m, self.floor_tolerance_m))
                or self.min_obstacle_height_m < 0
                or self.max_obstacle_height_m <= self.min_obstacle_height_m
                or self.floor_tolerance_m <= 0):
            raise ValueError('Invalid map obstacle height band or floor tolerance')
        self.points = voxel_downsample(
            load_pcd_xyz(self.get_parameter('map_path').value),
            float(self.get_parameter('map_voxel_m').value),
        )
        if not len(self.points):
            raise ValueError('CBF map has no finite XYZ points')
        self.tree = cKDTree(self.points[:, :2])
        self.tf = Buffer(cache_time=Duration(seconds=5))
        self.listener = TransformListener(self.tf, self)
        self.active = False
        self.icp_stamp = None
        self.icp_position = None
        self.aligned = False
        self.odom_stamp = None
        self.pub = self.create_publisher(PointCloud2, '/cbf/map_rear_obstacles', 10)
        self.filtered_pub = self.create_publisher(PointCloud2, '/cbf/map_filtered_obstacles', 10)
        self.rejected_pub = self.create_publisher(PointCloud2, '/cbf/map_rejected_obstacles', 10)
        self.floor_pub = self.create_publisher(MarkerArray, '/cbf/map_floor', 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.on_initialpose, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/icp_result', self.on_icp, 10)
        self.create_subscription(Odometry, '/Odometry', self.on_odom, 10)
        self.create_timer(0.1, self.tick)

    def clear(self):
        self.active = False
        self.odom_stamp = None
        marker = Marker()
        marker.action = Marker.DELETEALL
        self.floor_pub.publish(MarkerArray(markers=[marker]))
        self.publish_cloud(np.empty((0, 3)), self.get_clock().now().to_msg())
        self.publish_debug_clouds(np.empty((0, 3)), np.empty((0, 3)), self.get_clock().now().to_msg())

    def on_initialpose(self, msg):
        self.icp_stamp = None
        self.icp_position = None
        self.clear()

    def on_icp(self, msg):
        if msg.header.frame_id != 'map':
            return
        self.icp_stamp = self.get_clock().now().nanoseconds
        p = msg.pose.pose.position
        self.icp_position = np.array([p.x, p.y, p.z])
        self.aligned = False
        self.clear()

    def on_odom(self, msg):
        if self.icp_stamp is None or msg.header.frame_id != 'camera_init' or msg.child_frame_id != 'body':
            return
        stamp = stamp_ns(msg.header.stamp)
        if stamp > self.icp_stamp:
            self.odom_stamp = stamp
            self.active = True

    def build_cloud(self, points, stamp):
        msg = PointCloud2()
        msg.header.frame_id = 'map'
        msg.header.stamp = stamp
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = True
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.width * 12
        msg.fields = [PointField(name=n, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(('x', 'y', 'z'))]
        msg.data = np.asarray(points, dtype='<f4').reshape(-1, 3).tobytes()
        return msg

    def publish_cloud(self, points, stamp):
        self.pub.publish(self.build_cloud(points, stamp))

    def publish_debug_clouds(self, filtered, rejected, stamp):
        self.filtered_pub.publish(self.build_cloud(filtered, stamp))
        self.rejected_pub.publish(self.build_cloud(rejected, stamp))

    def publish_floor(self, support_base, plane, rotation, translation, stamp):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = stamp
        marker.ns = 'local_floor'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.025
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.1, 0.9, 0.2, 0.9
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(seconds=0).to_msg()
        def point(x, y):
            local = np.array([x, y, plane[0] * x + plane[1] * y + plane[2]])
            world = rotation @ local + translation
            return Point(x=float(world[0]), y=float(world[1]), z=float(world[2]))
        for v in np.linspace(-2, 2, 9):
            marker.points.extend((point(-2, v), point(2, v), point(v, -2), point(v, 2)))
        support = Marker()
        support.header = marker.header
        support.ns = 'floor_support'
        support.id = 1
        support.type = Marker.POINTS
        support.action = Marker.ADD
        support.pose.orientation.w = 1.0
        support.scale.x = support.scale.y = 0.035
        support.color.r, support.color.g, support.color.b, support.color.a = 0.1, 0.5, 1.0, 0.9
        support.lifetime = marker.lifetime
        for local in support_base[::max(1, len(support_base) // 500)]:
            world = rotation @ local + translation
            support.points.append(Point(x=float(world[0]), y=float(world[1]), z=float(world[2])))
        self.floor_pub.publish(MarkerArray(markers=[marker, support]))

    def tick(self):
        now = self.get_clock().now()
        if not self.active or self.odom_stamp is None:
            return
        if (now.nanoseconds - self.odom_stamp) * 1e-9 > self.get_parameter('max_pose_age_s').value:
            self.clear()
            return
        try:
            tf = self.tf.lookup_transform('map', 'base_link', Time(nanoseconds=self.odom_stamp))
        except TransformException:
            # Odometry may arrive before TF at the same timestamp. Retry next tick.
            # The age check above still clears stale output.
            return
        t, q = tf.transform.translation, tf.transform.rotation
        translation = np.array([t.x, t.y, t.z])
        if not self.aligned:
            try:
                body_tf = self.tf.lookup_transform('map', 'body', Time(nanoseconds=self.odom_stamp))
            except TransformException:
                return
            b = body_tf.transform.translation
            if np.linalg.norm(np.array([b.x, b.y, b.z]) - self.icp_position) > 1.0:
                return
            self.aligned = True
        rotation = rotation_matrix_from_quaternion_xyzw(q.x, q.y, q.z, q.w)
        ids = self.tree.query_ball_point(translation[:2], self.map_obstacle_range_m + 1.0)
        nearby = self.points[ids]
        local = (nearby - translation) @ rotation
        result = floor_and_rear(
            local, max_range=self.map_obstacle_range_m,
            min_obstacle_height=self.min_obstacle_height_m,
            max_obstacle_height=self.max_obstacle_height_m,
            floor_tolerance=self.floor_tolerance_m,
            support_radius_m=float(self.get_parameter('support_radius_m').value),
            min_support_points=int(self.get_parameter('min_support_points').value),
            lidar_origin_xy=DEFAULT_LIDAR_TF_XYZ[:2],
        )
        if result is None:
            self.publish_cloud(np.empty((0, 3)), now.to_msg())
            self.publish_debug_clouds(np.empty((0, 3)), np.empty((0, 3)), now.to_msg())
            self.floor_pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
            return
        support, rear, plane, filtered, rejected = result
        self.publish_cloud(rear @ rotation.T + translation, now.to_msg())
        self.publish_debug_clouds(filtered @ rotation.T + translation,
                                  rejected @ rotation.T + translation, now.to_msg())
        self.publish_floor(support, plane, rotation, translation, now.to_msg())


def main():
    rclpy.init()
    node = CbfMapObstacles()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
