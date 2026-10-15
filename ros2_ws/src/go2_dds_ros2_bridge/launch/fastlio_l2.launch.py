"""Evaluation-only FAST-LIO using the L2 internal IMU and raw L2 cloud."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from go2_dds_ros2_bridge.dds_runtime import DEFAULT_HARDWARE_DDS_RUNTIME


def _nodes(context):
    args = ["--run-mode", "hardware", "--dds-domain-id", LaunchConfiguration("dds_domain_id").perform(context),
            "--dds-interface", LaunchConfiguration("dds_interface").perform(context)]
    return [
        Node(package="go2_dds_ros2_bridge", executable="fastlio_l2_input", name="fastlio_l2_input",
             output="screen", arguments=args),
        Node(package="fast_lio", executable="fastlio_mapping", name="fastlio_l2_mapping",
             output="screen", parameters=[LaunchConfiguration("config")],
             remappings=[("/Odometry", "/fastlio_l2/odom"),
                         ("/path", "/fastlio_l2/path"),
                         ("/cloud_registered", "/fastlio_l2/cloud_registered"),
                         ("/cloud_registered_body", "/fastlio_l2/cloud_registered_imu"),
                         ("/cloud_effected", "/fastlio_l2/cloud_effected"),
                         ("/Laser_map", "/fastlio_l2/laser_map")]),
        Node(package="go2_dds_ros2_bridge", executable="fastlio_l2_frames", name="fastlio_l2_frames",
             output="screen"),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("dds_domain_id", default_value=str(DEFAULT_HARDWARE_DDS_RUNTIME.domain_id)),
        DeclareLaunchArgument("dds_interface", default_value=DEFAULT_HARDWARE_DDS_RUNTIME.interface),
        DeclareLaunchArgument("config", default_value=PathJoinSubstitution([
            FindPackageShare("go2_dds_ros2_bridge"), "config", "fastlio_l2.yaml"])),
        OpaqueFunction(function=_nodes),
    ])
