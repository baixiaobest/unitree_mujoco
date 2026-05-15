from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from go2_dds_ros2_bridge.tf_utils import DEFAULT_LIDAR_TF_RPY_DEG, DEFAULT_LIDAR_TF_XYZ


def _make_common_bridge_arguments(context) -> list[str]:
    run_mode = LaunchConfiguration("run_mode").perform(context)
    dds_domain_id = LaunchConfiguration("dds_domain_id").perform(context).strip()
    dds_interface = LaunchConfiguration("dds_interface").perform(context).strip()

    arguments = ["--run-mode", run_mode]
    if dds_domain_id:
        arguments.extend(["--dds-domain-id", dds_domain_id])
    if dds_interface:
        arguments.extend(["--dds-interface", dds_interface])
    return arguments


def _make_runtime_nodes(context, *args, **kwargs):
    ekf_config = LaunchConfiguration("ekf_config")
    odom_covariance_config = LaunchConfiguration("odom_covariance_config")
    kiss_covariance_config = LaunchConfiguration("kiss_covariance_config")
    common_bridge_arguments = _make_common_bridge_arguments(context)

    return [
        Node(
            package="go2_dds_ros2_bridge",
            executable="clock_offset_bridge",
            name="clock_offset_bridge",
            output="screen",
            arguments=common_bridge_arguments + ["--no-publish-tf"],
        ),
        Node(
            package="go2_dds_ros2_bridge",
            executable="odom_bridge",
            name="odom_bridge",
            output="screen",
            arguments=common_bridge_arguments + ["--covariance-file", odom_covariance_config],
        ),
        Node(
            package="go2_dds_ros2_bridge",
            executable="kiss_odom_node",
            name="kiss_odom_node",
            output="screen",
            arguments=[
                "--odom-frame", "odom",
                "--child-frame", "base_link",
                "--accumulate-frames", "10",
                "--covariance-file", kiss_covariance_config,
                "--reset-after-registrations", "100",
                "--deskew",
            ],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="go2_lidar_static_tf",
            output="screen",
            arguments=[
                "--x", str(DEFAULT_LIDAR_TF_XYZ[0]),
                "--y", str(DEFAULT_LIDAR_TF_XYZ[1]),
                "--z", str(DEFAULT_LIDAR_TF_XYZ[2]),
                "--roll", str(DEFAULT_LIDAR_TF_RPY_DEG[0] * 3.141592653589793 / 180.0),
                "--pitch", str(DEFAULT_LIDAR_TF_RPY_DEG[1] * 3.141592653589793 / 180.0),
                "--yaw", str(DEFAULT_LIDAR_TF_RPY_DEG[2] * 3.141592653589793 / 180.0),
                "--frame-id", "base_link",
                "--child-frame-id", "utlidar_lidar",
            ],
        ),
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_filter_node",
            output="screen",
            parameters=[ekf_config],
        ),
    ]


def generate_launch_description() -> LaunchDescription:

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "run_mode",
                default_value="hardware",
                description="Select the default DDS domain/interface pair for the raw bridge nodes.",
            ),
            DeclareLaunchArgument(
                "dds_domain_id",
                default_value="",
                description="Optional raw DDS domain override passed to the clock and odometry bridges.",
            ),
            DeclareLaunchArgument(
                "dds_interface",
                default_value="",
                description="Optional raw DDS interface override passed to the clock and odometry bridges.",
            ),
            DeclareLaunchArgument(
                "ekf_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_dds_ros2_bridge"), "config", "ekf_localization.yaml"]
                ),
                description="Path to the robot_localization EKF configuration file.",
            ),
            DeclareLaunchArgument(
                "odom_covariance_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_dds_ros2_bridge"), "config", "odom_bridge_covariance.yaml"]
                ),
                description="Path to the odom_bridge diagonal covariance YAML file.",
            ),
            DeclareLaunchArgument(
                "kiss_covariance_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_dds_ros2_bridge"), "config", "kiss_odom_covariance.yaml"]
                ),
                description="Path to the kiss_odom diagonal covariance YAML file.",
            ),
            OpaqueFunction(function=_make_runtime_nodes),
        ]
    )