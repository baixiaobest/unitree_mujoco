from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from go2_dds_ros2_bridge.dds_runtime import DEFAULT_HARDWARE_DDS_RUNTIME, DEFAULT_SIMULATION_DDS_RUNTIME
from go2_dds_ros2_bridge.tf_utils import (
    DEFAULT_IMU_TF_XYZ,
    DEFAULT_LIDAR_TF_RPY_DEG,
    DEFAULT_LIDAR_TF_XYZ,
    flatten_rotation_matrix,
    lidar_pose_in_imu_frame,
)


def _make_common_bridge_arguments(context) -> list[str]:
    run_mode = LaunchConfiguration("run_mode").perform(context)
    dds_domain_id_override = LaunchConfiguration("dds_domain_id").perform(context).strip()
    dds_interface_override = LaunchConfiguration("dds_interface").perform(context).strip()
    simulation_dds_domain_id = LaunchConfiguration("simulation_dds_domain_id").perform(context).strip()
    hardware_dds_domain_id = LaunchConfiguration("hardware_dds_domain_id").perform(context).strip()
    simulation_dds_interface = LaunchConfiguration("simulation_dds_interface").perform(context).strip()
    hardware_dds_interface = LaunchConfiguration("hardware_dds_interface").perform(context).strip()

    if run_mode == "simulation":
        resolved_domain_id = (
            dds_domain_id_override
            or simulation_dds_domain_id
            or str(DEFAULT_SIMULATION_DDS_RUNTIME.domain_id)
        )
        resolved_interface = (
            dds_interface_override
            or simulation_dds_interface
            or DEFAULT_SIMULATION_DDS_RUNTIME.interface
        )
    else:
        resolved_domain_id = (
            dds_domain_id_override
            or hardware_dds_domain_id
            or str(DEFAULT_HARDWARE_DDS_RUNTIME.domain_id)
        )
        resolved_interface = (
            dds_interface_override
            or hardware_dds_interface
            or DEFAULT_HARDWARE_DDS_RUNTIME.interface
        )

    arguments = ["--run-mode", run_mode, "--dds-domain-id", resolved_domain_id]
    if resolved_interface:
        arguments.extend(["--dds-interface", resolved_interface])
    return arguments


def _make_runtime_nodes(context, *args, **kwargs):
    common_bridge_arguments = _make_common_bridge_arguments(context)
    fast_lio_config = LaunchConfiguration("fast_lio_config").perform(context)
    cloud_topic = LaunchConfiguration("cloud_topic").perform(context)
    imu_topic = LaunchConfiguration("imu_topic").perform(context)
    corrected_map_frame = LaunchConfiguration("corrected_map_frame").perform(context)
    occupancy_config = LaunchConfiguration("occupancy_config").perform(context)
    occupancy_scan_config = LaunchConfiguration("occupancy_scan_config").perform(context)
    occupancy_scan_output_topic = LaunchConfiguration("occupancy_scan_output_topic").perform(context)
    temporal_lidar_config = LaunchConfiguration("temporal_lidar_config").perform(context)
    navigation_config = LaunchConfiguration("navigation_config").perform(context)
    navigation_policy_path = LaunchConfiguration("navigation_policy_path").perform(context)
    rviz_config = LaunchConfiguration("rviz_config")

    # FAST-LIO expects the pose of the frame that the incoming points are expressed in.
    # The corrected Unitree cloud preserves the incoming utlidar_lidar frame, so pass
    # the physical LiDAR pose in the IMU/body frame.
    extrinsic_translation, extrinsic_rotation = lidar_pose_in_imu_frame()
    body_to_base_translation = (-DEFAULT_IMU_TF_XYZ[0], -DEFAULT_IMU_TF_XYZ[1], -DEFAULT_IMU_TF_XYZ[2])

    return [
        Node(
            package="go2_dds_ros2_bridge",
            executable="temporal_lidar",
            name="temporal_lidar",
            output="screen",
            parameters=[
                temporal_lidar_config,
                {"map_frame": corrected_map_frame},
            ],
            condition=IfCondition(LaunchConfiguration("temporal_lidar")),
        ),
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
            executable="fast_lio_frame_correction",
            name="fast_lio_frame_correction",
            output="screen",
            arguments=common_bridge_arguments
            + [
                "--imu-topic", imu_topic,
                "--parent-frame", corrected_map_frame,
                "--child-frame", "camera_init",
                "--body-frame", "body",
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
                    "map_frame": corrected_map_frame,
                    "debug": True,
                },
            ],
            condition=IfCondition(LaunchConfiguration("occupancy")),
        ),
        Node(
            package="go2_dds_ros2_bridge",
            executable="occupancy_scan",
            name="occupancy_scan",
            output="screen",
            parameters=[
                occupancy_scan_config,
                {
                    "map_frame": corrected_map_frame,
                    "output_topic": occupancy_scan_output_topic,
                },
            ],
            condition=IfCondition(LaunchConfiguration("occupancy_scan")),
        ),
        Node(
            package="go2_dds_ros2_bridge",
            executable="navigation",
            name="navigation",
            output="screen",
            parameters=[
                navigation_config,
                *([ {"policy_path": navigation_policy_path} ] if navigation_policy_path else []),
            ],
            condition=IfCondition(LaunchConfiguration("navigation")),
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
                description="Optional global raw DDS domain override passed to the bridge nodes.",
            ),
            DeclareLaunchArgument(
                "dds_interface",
                default_value="",
                description="Optional global raw DDS interface override passed to the bridge nodes.",
            ),
            DeclareLaunchArgument(
                "simulation_dds_domain_id",
                default_value=str(DEFAULT_SIMULATION_DDS_RUNTIME.domain_id),
                description="Shared DDS domain id used when run_mode=simulation and no explicit dds_domain_id override is set.",
            ),
            DeclareLaunchArgument(
                "hardware_dds_domain_id",
                default_value=str(DEFAULT_HARDWARE_DDS_RUNTIME.domain_id),
                description="Shared DDS domain id used when run_mode=hardware and no explicit dds_domain_id override is set.",
            ),
            DeclareLaunchArgument(
                "simulation_dds_interface",
                default_value=DEFAULT_SIMULATION_DDS_RUNTIME.interface,
                description="Shared DDS network interface used when run_mode=simulation and no explicit dds_interface override is set.",
            ),
            DeclareLaunchArgument(
                "hardware_dds_interface",
                default_value=DEFAULT_HARDWARE_DDS_RUNTIME.interface,
                description="Shared DDS network interface used when run_mode=hardware and no explicit dds_interface override is set.",
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
                "corrected_map_frame",
                default_value="camera_init_correct",
                description="Parent frame that levels FAST-LIO's camera_init using gravity-based roll/pitch correction.",
            ),
            DeclareLaunchArgument(
                "occupancy",
                default_value="true",
                description="Launch the legacy rolling 2D occupancy mapper on /cloud_registered.",
            ),
            DeclareLaunchArgument(
                "occupancy_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_dds_ros2_bridge"), "config", "occupancy_2d.yaml"]
                ),
                description="Path to the 2D occupancy mapper parameter YAML file.",
            ),
            DeclareLaunchArgument(
                "occupancy_scan",
                default_value="true",
                description="Launch the legacy occupancy ray-cast LaserScan node.",
            ),
            DeclareLaunchArgument(
                "occupancy_scan_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_dds_ros2_bridge"), "config", "occupancy_scan.yaml"]
                ),
                description="Path to the occupancy scan node parameter YAML file.",
            ),
            DeclareLaunchArgument(
                "occupancy_scan_output_topic",
                default_value="/occupancy_scan",
                description="LaserScan topic published by the occupancy scan node.",
            ),
            DeclareLaunchArgument(
                "temporal_lidar",
                default_value="true",
                description="Launch the dense registered-cloud temporal lidar policy-observation node.",
            ),
            DeclareLaunchArgument(
                "temporal_lidar_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_dds_ros2_bridge"), "config", "temporal_lidar.yaml"]
                ),
                description="Path to the temporal registered-cloud lidar parameter YAML file.",
            ),
            DeclareLaunchArgument(
                "navigation",
                default_value="true",
                description="Launch the navigation policy inference node.",
            ),
            DeclareLaunchArgument(
                "navigation_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_dds_ros2_bridge"), "config", "navigation.yaml"]
                ),
                description="Path to the navigation node parameter YAML file.",
            ),
            DeclareLaunchArgument(
                "navigation_policy_path",
                default_value="",
                description="Path to the JIT navigation policy file (.pt).",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Launch RViz with the corrected FAST-LIO config.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_dds_ros2_bridge"), "config", "fast_lio_corrected.rviz"]
                ),
                description="Path to the RViz config file.",
            ),
            OpaqueFunction(function=_make_runtime_nodes),
        ]
    )
