from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from go2_dds_ros2_bridge.tf_utils import (
    DEFAULT_IMU_TF_XYZ,
    DEFAULT_LIDAR_TF_RPY_DEG,
    DEFAULT_LIDAR_TF_XYZ,
    flatten_rotation_matrix,
    lidar_pose_in_imu_frame,
)


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
    common_bridge_arguments = _make_common_bridge_arguments(context)
    fast_lio_config = LaunchConfiguration("fast_lio_config").perform(context)
    cloud_topic = LaunchConfiguration("cloud_topic").perform(context)
    imu_topic = LaunchConfiguration("imu_topic").perform(context)
    occupancy_config = LaunchConfiguration("occupancy_config").perform(context)
    occupancy_cloud_topic = LaunchConfiguration("occupancy_cloud_topic").perform(context)
    occupancy_output_topic = LaunchConfiguration("occupancy_output_topic").perform(context)
    rviz_config = LaunchConfiguration("rviz_config")

    # FAST-LIO expects the pose of the frame that the incoming points are expressed in.
    # The corrected Unitree cloud preserves the incoming utlidar_lidar frame, so pass
    # the physical LiDAR pose in the IMU/body frame.
    extrinsic_translation, extrinsic_rotation = lidar_pose_in_imu_frame()
    body_to_base_translation = (-DEFAULT_IMU_TF_XYZ[0], -DEFAULT_IMU_TF_XYZ[1], -DEFAULT_IMU_TF_XYZ[2])

    return [
        Node(
            package="go2_dds_ros2_bridge",
            executable="clock_offset_bridge",
            name="clock_offset_bridge",
            output="screen",
            arguments=common_bridge_arguments + ["--no-publish-tf", "--output-topic", cloud_topic],
        ),
        Node(
            package="go2_dds_ros2_bridge",
            executable="imu_bridge",
            name="imu_bridge",
            output="screen",
            arguments=common_bridge_arguments + ["--ros-topic", imu_topic, "--frame-id", "body"],
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
            package="tf2_ros",
            executable="static_transform_publisher",
            name="go2_body_to_base_link_static_tf",
            output="screen",
            arguments=[
                "--x", str(body_to_base_translation[0]),
                "--y", str(body_to_base_translation[1]),
                "--z", str(body_to_base_translation[2]),
                "--roll", "0.0",
                "--pitch", "0.0",
                "--yaw", "0.0",
                "--frame-id", "body",
                "--child-frame-id", "base_link",
            ],
        ),
        Node(
            package="fast_lio",
            executable="fastlio_mapping",
            name="fastlio_mapping",
            output="screen",
            parameters=[
                fast_lio_config,
                {
                    "common.lid_topic": cloud_topic,
                    "common.imu_topic": imu_topic,
                    "mapping.extrinsic_est_en": False,
                    "mapping.extrinsic_T": [float(value) for value in extrinsic_translation],
                    "mapping.extrinsic_R": flatten_rotation_matrix(extrinsic_rotation),
                },
            ],
        ),
        Node(
            package="go2_dds_ros2_bridge",
            executable="occupancy_2d",
            name="occupancy_2d",
            output="screen",
            parameters=[
                occupancy_config,
                {
                    "input_topic": occupancy_cloud_topic,
                    "output_topic": occupancy_output_topic,
                },
            ],
            condition=IfCondition(LaunchConfiguration("occupancy")),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="fast_lio_rviz",
            arguments=["-d", rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
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
                description="Optional raw DDS domain override passed to the bridge nodes.",
            ),
            DeclareLaunchArgument(
                "dds_interface",
                default_value="",
                description="Optional raw DDS interface override passed to the bridge nodes.",
            ),
            DeclareLaunchArgument(
                "fast_lio_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_dds_ros2_bridge"), "config", "fast_lio.yaml"]
                ),
                description="Path to the FAST-LIO parameter YAML file.",
            ),
            DeclareLaunchArgument(
                "cloud_topic",
                default_value="/utlidar/time_corrected/cloud",
                description="Corrected PointCloud2 topic forwarded into FAST-LIO.",
            ),
            DeclareLaunchArgument(
                "imu_topic",
                default_value="/imu/data_fastlio",
                description="Cadence-corrected IMU topic forwarded into FAST-LIO.",
            ),
            DeclareLaunchArgument(
                "occupancy",
                default_value="true",
                description="Launch the rolling 2D occupancy mapper on /cloud_registered.",
            ),
            DeclareLaunchArgument(
                "occupancy_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_dds_ros2_bridge"), "config", "occupancy_2d.yaml"]
                ),
                description="Path to the 2D occupancy mapper parameter YAML file.",
            ),
            DeclareLaunchArgument(
                "occupancy_cloud_topic",
                default_value="/cloud_registered",
                description="Registered point cloud topic consumed by the 2D occupancy mapper.",
            ),
            DeclareLaunchArgument(
                "occupancy_output_topic",
                default_value="/static_occupancy",
                description="OccupancyGrid topic published by the 2D occupancy mapper.",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Launch RViz with the upstream FAST-LIO config.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution([FindPackageShare("fast_lio"), "rviz", "fastlio.rviz"]),
                description="Path to the RViz config file.",
            ),
            OpaqueFunction(function=_make_runtime_nodes),
        ]
    )