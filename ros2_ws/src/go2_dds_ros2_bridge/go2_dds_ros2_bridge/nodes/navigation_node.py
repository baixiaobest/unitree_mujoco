#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from time import time_ns

import numpy as np
import torch
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Imu, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from go2_dds_ros2_bridge.tf_utils import rotation_matrix_from_quaternion_xyzw

DEFAULT_POLICY_PATH = ""
DEFAULT_GOAL_TOPIC = "/goal_pose"
DEFAULT_VELOCITY_TOPIC = "/estimated_velocity"
DEFAULT_IMU_TOPIC = "/imu/data_fastlio"
DEFAULT_SCAN_TOPIC = "/occupancy_scan"
DEFAULT_CMD_TOPIC = "/cmd_vel"
DEFAULT_MAP_FRAME = "camera_init_correct"
DEFAULT_BODY_FRAME = "base_link"
DEFAULT_POLICY_HZ = 15.0
DEFAULT_LIDAR_MAX_DISTANCE = 20.0
DEFAULT_NUM_LIDAR_RAYS = 128
DEFAULT_CMD_SCALE_VX = 1.0
DEFAULT_CMD_SCALE_VY = 1.0
DEFAULT_CMD_SCALE_WZ = 1.0
DEFAULT_GOAL_Z = 0.35
DEFAULT_MAX_TF_AGE_SEC = 1.0

SCAN_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


@dataclass(frozen=True)
class NavigationConfig:
    policy_path: str
    goal_topic: str
    velocity_topic: str
    imu_topic: str
    scan_topic: str
    cmd_topic: str
    map_frame: str
    body_frame: str
    policy_hz: float
    lidar_max_distance: float
    num_lidar_rays: int
    cmd_scale_vx: float
    cmd_scale_vy: float
    cmd_scale_wz: float
    goal_z: float
    max_tf_age_sec: float
    device: str


def parse_args() -> NavigationConfig:
    parser = argparse.ArgumentParser(
        description="Run obstacle-avoidance navigation policy inference and publish velocity commands."
    )
    parser.add_argument("--policy-path", type=str, default=DEFAULT_POLICY_PATH,
                        help="Path to the JIT navigation policy (.pt file).")
    parser.add_argument("--goal-topic", type=str, default=DEFAULT_GOAL_TOPIC,
                        help="PoseStamped topic for the navigation goal (from RViz 2D Nav Goal).")
    parser.add_argument("--velocity-topic", type=str, default=DEFAULT_VELOCITY_TOPIC,
                        help="TwistStamped topic for body-frame linear velocity estimates.")
    parser.add_argument("--imu-topic", type=str, default=DEFAULT_IMU_TOPIC,
                        help="Imu topic for angular velocity.")
    parser.add_argument("--scan-topic", type=str, default=DEFAULT_SCAN_TOPIC,
                        help="LaserScan topic from the occupancy scan node.")
    parser.add_argument("--cmd-topic", type=str, default=DEFAULT_CMD_TOPIC,
                        help="TwistStamped topic to publish velocity commands.")
    parser.add_argument("--map-frame", type=str, default=DEFAULT_MAP_FRAME,
                        help="Map/world TF frame (goal is expressed in this frame).")
    parser.add_argument("--body-frame", type=str, default=DEFAULT_BODY_FRAME,
                        help="Robot body TF frame.")
    parser.add_argument("--policy-hz", type=float, default=DEFAULT_POLICY_HZ,
                        help="Policy inference rate in Hz.")
    parser.add_argument("--lidar-max-distance", type=float, default=DEFAULT_LIDAR_MAX_DISTANCE,
                        help="Maximum lidar range used during training (scales scan input).")
    parser.add_argument("--num-lidar-rays", type=int, default=DEFAULT_NUM_LIDAR_RAYS,
                        help="Expected number of lidar rays in the scan message.")
    parser.add_argument("--cmd-scale-vx", type=float, default=DEFAULT_CMD_SCALE_VX,
                        help="Scale factor applied to the policy's vx output.")
    parser.add_argument("--cmd-scale-vy", type=float, default=DEFAULT_CMD_SCALE_VY,
                        help="Scale factor applied to the policy's vy output.")
    parser.add_argument("--cmd-scale-wz", type=float, default=DEFAULT_CMD_SCALE_WZ,
                        help="Scale factor applied to the policy's wz output.")
    parser.add_argument("--goal-z", type=float, default=DEFAULT_GOAL_Z,
                        help="Target z coordinate in map frame used for the pose command (nominal standing height).")
    parser.add_argument("--max-tf-age-sec", type=float, default=DEFAULT_MAX_TF_AGE_SEC,
                        help="Maximum age of a TF transform before it is considered stale.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for policy inference ('cpu' or 'cuda').")

    non_ros_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(non_ros_args)

    if args.policy_hz <= 0.0:
        raise SystemExit("--policy-hz must be positive")
    if args.lidar_max_distance <= 0.0:
        raise SystemExit("--lidar-max-distance must be positive")
    if args.num_lidar_rays < 1:
        raise SystemExit("--num-lidar-rays must be at least 1")

    return NavigationConfig(
        policy_path=args.policy_path,
        goal_topic=args.goal_topic,
        velocity_topic=args.velocity_topic,
        imu_topic=args.imu_topic,
        scan_topic=args.scan_topic,
        cmd_topic=args.cmd_topic,
        map_frame=args.map_frame,
        body_frame=args.body_frame,
        policy_hz=float(args.policy_hz),
        lidar_max_distance=float(args.lidar_max_distance),
        num_lidar_rays=int(args.num_lidar_rays),
        cmd_scale_vx=float(args.cmd_scale_vx),
        cmd_scale_vy=float(args.cmd_scale_vy),
        cmd_scale_wz=float(args.cmd_scale_wz),
        goal_z=float(args.goal_z),
        max_tf_age_sec=float(args.max_tf_age_sec),
        device=args.device,
    )


class NavigationNode(Node):
    def __init__(self, config: NavigationConfig) -> None:
        super().__init__("navigation")

        self.declare_parameter("policy_path", config.policy_path)
        self.declare_parameter("goal_topic", config.goal_topic)
        self.declare_parameter("velocity_topic", config.velocity_topic)
        self.declare_parameter("imu_topic", config.imu_topic)
        self.declare_parameter("scan_topic", config.scan_topic)
        self.declare_parameter("cmd_topic", config.cmd_topic)
        self.declare_parameter("map_frame", config.map_frame)
        self.declare_parameter("body_frame", config.body_frame)
        self.declare_parameter("policy_hz", config.policy_hz)
        self.declare_parameter("lidar_max_distance", config.lidar_max_distance)
        self.declare_parameter("num_lidar_rays", config.num_lidar_rays)
        self.declare_parameter("cmd_scale_vx", config.cmd_scale_vx)
        self.declare_parameter("cmd_scale_vy", config.cmd_scale_vy)
        self.declare_parameter("cmd_scale_wz", config.cmd_scale_wz)
        self.declare_parameter("goal_z", config.goal_z)
        self.declare_parameter("max_tf_age_sec", config.max_tf_age_sec)
        self.declare_parameter("device", config.device)

        self._config = NavigationConfig(
            policy_path=str(self.get_parameter("policy_path").value),
            goal_topic=str(self.get_parameter("goal_topic").value),
            velocity_topic=str(self.get_parameter("velocity_topic").value),
            imu_topic=str(self.get_parameter("imu_topic").value),
            scan_topic=str(self.get_parameter("scan_topic").value),
            cmd_topic=str(self.get_parameter("cmd_topic").value),
            map_frame=str(self.get_parameter("map_frame").value),
            body_frame=str(self.get_parameter("body_frame").value),
            policy_hz=float(self.get_parameter("policy_hz").value),
            lidar_max_distance=float(self.get_parameter("lidar_max_distance").value),
            num_lidar_rays=int(self.get_parameter("num_lidar_rays").value),
            cmd_scale_vx=float(self.get_parameter("cmd_scale_vx").value),
            cmd_scale_vy=float(self.get_parameter("cmd_scale_vy").value),
            cmd_scale_wz=float(self.get_parameter("cmd_scale_wz").value),
            goal_z=float(self.get_parameter("goal_z").value),
            max_tf_age_sec=float(self.get_parameter("max_tf_age_sec").value),
            device=str(self.get_parameter("device").value),
        )

        if not self._config.policy_path:
            raise SystemExit(
                "Navigation node requires a policy path. "
                "Pass --policy-path or set the 'policy_path' ROS2 parameter."
            )

        self._device = torch.device(self._config.device)
        self._policy = torch.jit.load(self._config.policy_path, map_location=self._device).eval()

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._goal_sub = self.create_subscription(
            PoseStamped, self._config.goal_topic, self._goal_callback, 10
        )
        self._velocity_sub = self.create_subscription(
            TwistStamped, self._config.velocity_topic, self._velocity_callback, 10
        )
        self._imu_sub = self.create_subscription(
            Imu, self._config.imu_topic, self._imu_callback, 10
        )
        self._scan_sub = self.create_subscription(
            LaserScan, self._config.scan_topic, self._scan_callback, SCAN_QOS
        )
        self._cmd_publisher = self.create_publisher(
            TwistStamped, self._config.cmd_topic, 10
        )

        self._goal_pos_world: np.ndarray | None = None
        self._goal_yaw_world: float | None = None
        self._lin_vel: np.ndarray | None = None
        self._ang_vel: np.ndarray | None = None
        self._scan: np.ndarray | None = None
        self._last_action = torch.zeros(3, dtype=torch.float32)

        self._tf_warning_last_ns: int = 0
        self._stale_tf_warning_last_ns: int = 0

        self._policy_timer = self.create_timer(1.0 / self._config.policy_hz, self._policy_step)

        self.get_logger().info(
            "Navigation node: policy='%s', goal='%s', vel='%s', imu='%s', scan='%s', cmd='%s'. "
            "Map frame: '%s', body frame: '%s'. %.1f Hz, lidar_max=%.1f, rays=%d, "
            "cmd_scale=(%.2f, %.2f, %.2f), goal_z=%.3f, device=%s."
            % (
                self._config.policy_path,
                self._config.goal_topic,
                self._config.velocity_topic,
                self._config.imu_topic,
                self._config.scan_topic,
                self._config.cmd_topic,
                self._config.map_frame,
                self._config.body_frame,
                self._config.policy_hz,
                self._config.lidar_max_distance,
                self._config.num_lidar_rays,
                self._config.cmd_scale_vx,
                self._config.cmd_scale_vy,
                self._config.cmd_scale_wz,
                self._config.goal_z,
                self._config.device,
            )
        )

    def _goal_callback(self, msg: PoseStamped) -> None:
        self._goal_pos_world = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        q = msg.pose.orientation
        self._goal_yaw_world = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
        )

    def _velocity_callback(self, msg: TwistStamped) -> None:
        self._lin_vel = np.array(
            [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z],
            dtype=np.float32,
        )

    def _imu_callback(self, msg: Imu) -> None:
        self._ang_vel = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            dtype=np.float32,
        )

    def _scan_callback(self, msg: LaserScan) -> None:
        ranges = np.clip(
            np.array(msg.ranges, dtype=np.float32), 0.0, self._config.lidar_max_distance
        )
        self._scan = ranges / self._config.lidar_max_distance

    def _transform_age_sec(self, transform) -> float:
        transform_stamp = Time.from_msg(transform.header.stamp)
        if transform_stamp.nanoseconds == 0:
            return 0.0
        now_ns = self.get_clock().now().nanoseconds
        return max(0.0, (now_ns - transform_stamp.nanoseconds) / 1e9)

    def _is_transform_recent_enough(self, transform) -> bool:
        return self._transform_age_sec(transform) <= self._config.max_tf_age_sec

    def _lookup_robot_pose(self) -> tuple[np.ndarray, float] | None:
        last_error: Exception | None = None
        try:
            transform = self._tf_buffer.lookup_transform(
                self._config.map_frame,
                self._config.body_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as error:
            last_error = error
            now_ns = time_ns()
            if now_ns - self._tf_warning_last_ns >= 2_000_000_000:
                self._tf_warning_last_ns = now_ns
                self.get_logger().warning(
                    "Failed to resolve transform %s -> %s for navigation: %s"
                    % (self._config.map_frame, self._config.body_frame, last_error)
                )
            return None

        if not self._is_transform_recent_enough(transform):
            now_ns = time_ns()
            if now_ns - self._stale_tf_warning_last_ns >= 2_000_000_000:
                self._stale_tf_warning_last_ns = now_ns
                self.get_logger().warning(
                    "Latest transform %s -> %s is stale by %.3f s for navigation."
                    % (
                        self._config.map_frame,
                        self._config.body_frame,
                        self._transform_age_sec(transform),
                    )
                )
            return None

        t = transform.transform.translation
        robot_pos = np.array([float(t.x), float(t.y), float(t.z)], dtype=np.float64)

        r = transform.transform.rotation
        rotation_matrix = rotation_matrix_from_quaternion_xyzw(
            float(r.x), float(r.y), float(r.z), float(r.w)
        )
        robot_yaw = math.atan2(float(rotation_matrix[1, 0]), float(rotation_matrix[0, 0]))

        return robot_pos, robot_yaw

    def _build_pose_command(self, robot_pos: np.ndarray, robot_yaw: float) -> np.ndarray:
        dx = self._goal_pos_world[0] - robot_pos[0]
        dy = self._goal_pos_world[1] - robot_pos[1]
        # Use configured goal_z so the z component stays near the training distribution
        dz = self._config.goal_z - robot_pos[2]
        # Yaw-only inverse rotation — matches quat_rotate_inverse(yaw_quat(q), target_vec) in sim
        cos_y = math.cos(robot_yaw)
        sin_y = math.sin(robot_yaw)
        pos_body_x = cos_y * dx + sin_y * dy
        pos_body_y = -sin_y * dx + cos_y * dy
        heading_err = (self._goal_yaw_world - robot_yaw + math.pi) % (2.0 * math.pi) - math.pi
        return np.array([pos_body_x, pos_body_y, dz, heading_err], dtype=np.float32)

    def _policy_step(self) -> None:
        if any(
            v is None
            for v in [self._goal_pos_world, self._goal_yaw_world, self._lin_vel, self._ang_vel, self._scan]
        ):
            return

        if len(self._scan) != self._config.num_lidar_rays:
            return

        pose_result = self._lookup_robot_pose()
        if pose_result is None:
            return
        robot_pos, robot_yaw = pose_result

        pose_cmd = self._build_pose_command(robot_pos, robot_yaw)

        obs = np.concatenate([
            pose_cmd,                    # (4,)  pose_2d_command
            self._lin_vel,               # (3,)  base_lin_vel
            self._ang_vel,               # (3,)  imu_ang_vel
            self._last_action.numpy(),   # (3,)  actions (last output)
            self._scan,                  # (128,) obstacle_scan
        ]).astype(np.float32)            # (141,) total

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self._device)
        with torch.no_grad():
            action = self._policy(obs_t).squeeze(0).cpu()  # (3,) → [vx, vy, wz]

        self._last_action = action

        cmd = TwistStamped()

        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self._config.body_frame

        cmd.twist.linear.x = float(action[0]) * self._config.cmd_scale_vx
        cmd.twist.linear.y = float(action[1]) * self._config.cmd_scale_vy
        cmd.twist.angular.z = float(action[2]) * self._config.cmd_scale_wz

        self._cmd_publisher.publish(cmd)


def main() -> None:
    config = parse_args()
    rclpy.init(args=None)
    node = NavigationNode(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
