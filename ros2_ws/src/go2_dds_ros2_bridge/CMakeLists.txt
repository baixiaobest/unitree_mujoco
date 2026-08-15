cmake_minimum_required(VERSION 3.8)
project(go2_dds_ros2_bridge)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(go2_dds_ros2_bridge_msgs REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    go2_dds_ros2_bridge/nodes/clock_offset_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME clock_offset_bridge
)

install(
  PROGRAMS
    go2_dds_ros2_bridge/nodes/odom_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME odom_bridge
)

install(
  PROGRAMS
    go2_dds_ros2_bridge/nodes/imu_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME imu_bridge
)

install(
  PROGRAMS
    go2_dds_ros2_bridge/nodes/fast_lio_frame_correction_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fast_lio_frame_correction
)

install(
  PROGRAMS
    go2_dds_ros2_bridge/tick_experiment_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME tick_experiment_node
)

install(
  PROGRAMS
    go2_dds_ros2_bridge/nodes/cloud_experiment_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cloud_experiment_node
)

install(
  PROGRAMS
    go2_dds_ros2_bridge/nodes/imu_experiment_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME imu_experiment_node
)

install(
  PROGRAMS
    go2_dds_ros2_bridge/nodes/occupancy_2d_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME occupancy_2d
)

install(
  PROGRAMS
    go2_dds_ros2_bridge/nodes/occupancy_scan_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME occupancy_scan
)

install(
  PROGRAMS
    go2_dds_ros2_bridge/nodes/temporal_lidar_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME temporal_lidar
)

install(
  PROGRAMS
    go2_dds_ros2_bridge/nodes/navigation_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME navigation
)

install(
  DIRECTORY
    launch
    config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_temporal_lidar_processing test/test_temporal_lidar_processing.py)
endif()

ament_package()
