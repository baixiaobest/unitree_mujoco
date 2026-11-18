from pathlib import Path

import yaml

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

DEFAULT_MAP_PATH = str(Path.home() / ".local" / "share" / "go2_fast_lio" / "maps" / "map.pcd")


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


def _make_runtime_nodes(context, mode: str):
    common_bridge_arguments = _make_common_bridge_arguments(context)
    fast_lio_config = LaunchConfiguration("fast_lio_config").perform(context)
    cloud_topic = LaunchConfiguration("cloud_topic").perform(context)
    localization = mode == "localization"
    map_path = LaunchConfiguration("map_path").perform(context).strip() if mode != "odometry" else ""
    checkpoint_map_path = LaunchConfiguration("checkpoint_map_path").perform(context).strip() if mode == "mapping" else ""
    checkpoint_mapping = bool(checkpoint_map_path)
    imu_time_offset = 0.0
    if localization or checkpoint_mapping:
        fast_lio_parameters = yaml.safe_load(Path(fast_lio_config).read_text())["/**"]["ros__parameters"]
        imu_time_offset = float(fast_lio_parameters.get("common", {}).get(
            "time_offset_lidar_to_imu", fast_lio_parameters.get("common.time_offset_lidar_to_imu", 0.0)
        ))
    if checkpoint_mapping:
        checkpoint = Path(checkpoint_map_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError(f"checkpoint_map_path does not exist: {checkpoint}")
        if map_path == DEFAULT_MAP_PATH:
            map_path = str(checkpoint.with_name(checkpoint.stem + "_resumed.pcd"))
        if Path(map_path).expanduser().resolve() == checkpoint:
            raise ValueError("Checkpoint input and mapping output must be different PCD files")
    if mode == "mapping":
        Path(map_path).parent.mkdir(parents=True, exist_ok=True)
    if localization and (not map_path or not Path(map_path).is_file()):
        raise ValueError("localization requires map_path pointing to an existing PCD file")
    imu_topic = LaunchConfiguration("imu_topic").perform(context)
    robot_model = LaunchConfiguration("robot_model").perform(context).strip().lower() in ("true", "1", "yes")
    imu_bridge_config = LaunchConfiguration("imu_bridge_config").perform(context)
    if LaunchConfiguration("run_mode").perform(context) == "simulation":
        imu_bridge_config = str(Path(imu_bridge_config).with_name("imu_bridge.yaml"))
    corrected_map_frame = LaunchConfiguration("corrected_map_frame").perform(context)
    if localization or checkpoint_mapping:
        corrected_map_frame = "map"
    occupancy_config = LaunchConfiguration("occupancy_config").perform(context)
    occupancy_scan_config = LaunchConfiguration("occupancy_scan_config").perform(context)
    occupancy_scan_output_topic = LaunchConfiguration("occupancy_scan_output_topic").perform(context)
    temporal_lidar_config = LaunchConfiguration("temporal_lidar_config").perform(context)
    navigation_config = LaunchConfiguration("navigation_config").perform(context)
    navigation_policy_path = LaunchConfiguration("navigation_policy_path").perform(context)
    cbf_enabled = LaunchConfiguration("cbf").perform(context).strip().lower() in ("true", "1", "yes")
    map_obstacles = localization and cbf_enabled and LaunchConfiguration("cbf_map_obstacles").perform(context).strip().lower() in ("true", "1", "yes") if localization else False
    cbf_config = LaunchConfiguration("cbf_config").perform(context)
    cbf_mode = yaml.safe_load(Path(cbf_config).read_text())["cbf_control"]["ros__parameters"].get("cbf_mode", "static")
    if cbf_mode not in ("static", "dynamic"):
        raise ValueError("cbf_mode must be static or dynamic")
    rviz_config = LaunchConfiguration("rviz_config").perform(context)
    if checkpoint_mapping and rviz_config.endswith("fast_lio_mapping.rviz"):
        rviz_config = str(Path(rviz_config).with_name("fast_lio_localization.rviz"))

    # FAST-LIO expects the pose of the frame that the incoming points are expressed in.
    # The corrected Unitree cloud preserves the incoming utlidar_lidar frame, so pass
    # the physical LiDAR pose in the IMU/body frame.
    extrinsic_translation, extrinsic_rotation = lidar_pose_in_imu_frame()
    body_to_base_translation = (-DEFAULT_IMU_TF_XYZ[0], -DEFAULT_IMU_TF_XYZ[1], -DEFAULT_IMU_TF_XYZ[2])

    return [
        *([Node(package="icp_relocalization", executable="icp_node", name="icp_node",
                output="screen", parameters=[{
                    "map_path": checkpoint_map_path if checkpoint_mapping else map_path,
                    "checkpoint_mapping": checkpoint_mapping,
                    "map_frame_id": "map", "pcl_type": "pointcloud2",
                    "fitness_score_thre": 0.2, "converged_count_thre": 5,
                    "scan_stack_size": 5,
                    "scan_stack_max_duration_sec": 0.8,
                    "scan_stack_max_rotation_deg": 3.0,
                    "scan_stack_max_accel_rms_m_s2": 0.8,
                    "imu_topic": imu_topic,
                    "imu_time_offset_lidar_to_imu": imu_time_offset,
                    "max_correspondence_distance": 1.0,
                    "lidar_in_imu_translation": [float(value) for value in extrinsic_translation],
                    "lidar_in_imu_rotation": flatten_rotation_matrix(extrinsic_rotation),
                }], remappings=[("/pointcloud2", cloud_topic)])] if localization or checkpoint_mapping else []),
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
            arguments=common_bridge_arguments + ["--ros-topic", imu_topic,
                                                  "--raw-ros-topic", f"{imu_topic}_raw",
                                                  "--frame-id", "body", "--config-file", imu_bridge_config]
                      + (["--joint-states-topic", "/joint_states"] if robot_model else []),
        ),
        *([Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="go2_robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": Path(PathJoinSubstitution([
                FindPackageShare("go2_description"), "urdf", "go2_description.urdf"
            ]).perform(context)).read_text()}],
        )] if robot_model else []),
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
                    "locate_in_prior_map": localization or checkpoint_mapping,
                    "append_to_prior_map": checkpoint_mapping,
                    "mapping.extrinsic_est_en": False,
                    "mapping.extrinsic_T": [float(value) for value in extrinsic_translation],
                    "mapping.extrinsic_R": flatten_rotation_matrix(extrinsic_rotation),
                },
                *([{"prior_map_path": checkpoint_map_path if checkpoint_mapping else map_path,
                    "map_file_path": map_path,
                    "pcd_save.pcd_save_en": not localization,
                    "publish.map_en": not localization}] if map_path else []),
            ],
        ),
        Node(
            package="go2_dds_ros2_bridge",
            executable="fast_lio_frame_correction",
            name="fast_lio_frame_correction",
            output="screen",
            condition=IfCondition("false" if localization or checkpoint_mapping else "true"),
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
                {"command_mode": "cbf" if cbf_enabled else "direct", "map_frame": corrected_map_frame},
            ],
            condition=IfCondition(LaunchConfiguration("navigation")),
        ),
        Node(package="go2_dds_ros2_bridge", executable="cbf_world_state",
             name="cbf_world_state", output="screen",
             parameters=[cbf_config, {"global_frame": corrected_map_frame, "map_rear_obstacles": map_obstacles}],
             condition=IfCondition(LaunchConfiguration("cbf"))),
        *([Node(package="go2_dds_ros2_bridge", executable="cbf_map_obstacles",
                name="cbf_map_obstacles", output="screen",
                # Small repeated NumPy matrix operations otherwise wake the full OpenBLAS pool.
                additional_env={"OPENBLAS_NUM_THREADS": "1"},
                parameters=[
                    LaunchConfiguration("cbf_map_obstacles_config"), {"map_path": map_path},
                ])]
          if map_obstacles else []),
        Node(package="go2_dds_ros2_bridge", executable="cbf_velocity_predictor",
             name="cbf_velocity_predictor", output="screen", parameters=[cbf_config, {"cbf_mode": cbf_mode}],
             condition=IfCondition(LaunchConfiguration("cbf"))),
        Node(
            package="go2_cbf_control",
            executable="cbf_control",
            name="cbf_control",
            output="screen",
            parameters=[cbf_config, {"reserve_rear_slots": map_obstacles}],
            condition=IfCondition(LaunchConfiguration("cbf")),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="fast_lio_rviz",
            arguments=["-d", rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ]


def generate_launch_description(mode: str = "odometry") -> LaunchDescription:
    if mode not in ("odometry", "mapping", "localization"):
        raise ValueError(f"Unknown FAST-LIO launch mode: {mode}")
    mode_arguments = []
    if mode != "odometry":
        mode_arguments.append(DeclareLaunchArgument(
            "map_path", default_value=DEFAULT_MAP_PATH,
            description="Absolute PCD map path. Mapping creates its parent directory; localization reads the file.",
        ))
    if mode == "mapping":
        mode_arguments.append(DeclareLaunchArgument(
            "checkpoint_map_path", default_value="",
            description="Existing PCD to resume mapping from; waits for an RViz 2D Pose Estimate.",
        ))
    if mode == "localization":
        mode_arguments.append(DeclareLaunchArgument(
            "cbf_map_obstacles", default_value="true",
            description="Use ground-filtered rear obstacles from the localization PCD in CBF.",
        ))
        mode_arguments.append(DeclareLaunchArgument(
            "cbf_map_obstacles_config",
            default_value=PathJoinSubstitution(
                [FindPackageShare("go2_dds_ros2_bridge"), "config", "cbf_map_obstacles.yaml"]
            ),
            description="YAML parameters for the mapped rear CBF obstacle node.",
        ))
    return LaunchDescription(
        [
            *mode_arguments,
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
                "imu_bridge_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_dds_ros2_bridge"), "config", "imu_bridge_fastlio.yaml"]
                ),
                description="Go2 body IMU bias, scale, and reported covariance configuration for hardware.",
            ),
            DeclareLaunchArgument(
                "corrected_map_frame",
                default_value="camera_init_correct",
                description="Parent frame that levels FAST-LIO's camera_init using gravity-based roll/pitch correction.",
            ),
            DeclareLaunchArgument(
                "occupancy",
                default_value="false",
                description="Launch the legacy rolling 2D occupancy mapper on /cloud_registered (disabled by default).",
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
                default_value="false",
                description="Launch the legacy occupancy ray-cast LaserScan node (disabled by default).",
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
                "cbf",
                default_value="true",
                description="Route navigation through the native static CBF controller (default). Set false to use legacy direct /cmd_vel navigation.",
            ),
            DeclareLaunchArgument(
                "cbf_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_cbf_control"), "config", "cbf_control.yaml"]
                ),
                description="Path to the native CBF controller parameter YAML file.",
            ),
            DeclareLaunchArgument(
                "robot_model",
                default_value="true",
                description="Publish Go2 joint states and robot model TF for RViz.",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Launch RViz with the corrected FAST-LIO config.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_dds_ros2_bridge"),
                     "rviz", "fast_lio_localization.rviz" if mode == "localization"
                     else "fast_lio_mapping.rviz" if mode == "mapping"
                     else "fast_lio_odometry.rviz"]
                ),
                description="Path to the RViz config file.",
            ),
            OpaqueFunction(function=lambda context: _make_runtime_nodes(context, mode)),
        ]
    )
